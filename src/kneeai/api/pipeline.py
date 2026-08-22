"""The case pipeline: a segmentation in, a contract-compliant case directory out.

    segmentation -> canonicalise -> post-process -> reconstruct -> landmarks
                 -> measure -> implant match -> plan -> simulate -> artifacts

Every stage is an existing, verified engine. This module owns the *order*, the
error handling and the artifact layout; it implements none of the geometry.

Used by the CLI (``kneeai demo run``, ``kneeai infer``), by the API's processing
endpoint and by the worker, so all three produce byte-identical case directories.

Ordering note: ``segmentation.nii.gz`` and ``result.json`` are written before
anything derived from them, so the manifest always describes the exact volume the
meshes and measurements were computed from. ``result.json`` is rewritten once at
the end so warnings raised by later stages reach it too.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kneeai.api.adapter import SegmentationOutcome
from kneeai.api.contract import (
    MESH_DIRNAME,
    SEGMENTATION_FILENAME,
    canonical_label_volume,
    write_result_json,
)
from kneeai.core.anatomy import Side, Structure
from kneeai.core.errors import KneeAIError, PlanningError
from kneeai.logging import get_logger
from kneeai.ml.contract import ArtifactSource

__all__ = ["PipelineOptions", "CaseArtifacts", "run_case_pipeline"]

log = get_logger("api.pipeline")

#: Printed by anything that surfaces a plan.
DISCLAIMER = (
    "NOT FOR CLINICAL USE. Reference geometry and synthetic data only; no output "
    "of this tool is a clinical recommendation or an approved surgical plan."
)


@dataclass
class PipelineOptions:
    """Knobs for one pipeline run."""

    strategy: str = "restricted_kinematic"
    optimize: bool = True
    flexion_step_deg: int = 20
    compress_meshes: bool = True

    #: Number of implant configurations to keep. More than one is required for
    #: the UI's "runner-up and why it lost" panel.
    implant_candidates: int = 6

    #: Called with a human-readable stage label as each stage starts.
    on_stage: Callable[[str], None] | None = None

    def stage(self, label: str) -> None:
        if self.on_stage is not None:
            self.on_stage(label)


@dataclass
class CaseArtifacts:
    """What a run produced, in memory and on disk."""

    case_id: str
    case_dir: Path
    source: ArtifactSource
    result_path: Path
    segmentation_path: Path
    mask: Any = None
    mesh_manifest: Any = None
    meshes: Any = None
    landmarks: Any = None
    frame: Any = None
    measurements: Any = None
    fits: list[Any] = field(default_factory=list)
    plan: Any = None
    simulation: Any = None
    kinematics: Any = None
    truth: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def side(self) -> Side:
        return self.mask.side if self.mask is not None else Side.UNKNOWN

    @property
    def planned(self) -> bool:
        return self.plan is not None

    @property
    def total_seconds(self) -> float:
        return round(sum(self.timings_s.values()), 3)

    def artifact_paths(self) -> dict[str, Path]:
        """Every file this run wrote that exists."""
        candidates = {
            "result": self.case_dir / "result.json",
            "segmentation": self.segmentation_path,
            "measurements": self.case_dir / "measurements.json",
            "landmarks": self.case_dir / "landmarks.json",
            "frame": self.case_dir / "frame.json",
            "mesh_quality": self.case_dir / "mesh_quality.json",
            "mesh_manifest": self.case_dir / MESH_DIRNAME / "manifest.json",
            "implants": self.case_dir / "implants.json",
            "plan": self.case_dir / "plan.json",
            "simulation": self.case_dir / "simulation.json",
            "kinematics": self.case_dir / "kinematics.json",
            "optimization": self.case_dir / "optimization.json",
        }
        return {name: path for name, path in candidates.items() if path.exists()}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")
    return path


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _resolve_strategy(name: str) -> Any:
    from kneeai.planning import AlignmentStrategy

    try:
        return AlignmentStrategy(name)
    except ValueError as exc:
        valid = [s.value for s in AlignmentStrategy]
        raise KneeAIError(f"unknown alignment strategy {name!r}; choose from {valid}") from exc


def run_case_pipeline(
    outcome: SegmentationOutcome,
    *,
    case_id: str,
    out_root: Path | str,
    options: PipelineOptions | None = None,
) -> CaseArtifacts:
    """Run the pipeline and write ``<out_root>/cases/<case_id>/``."""
    from kneeai.api.meshes import export_meshes
    from kneeai.imaging.io import write_volume
    from kneeai.implants import (
        FemoralDimensions,
        load_catalog,
        match_configurations,
        provisional_tibial_plane,
        resection_footprint,
    )
    from kneeai.landmarks import detect_landmarks_and_frame
    from kneeai.measure import measure_all
    from kneeai.ml.postprocess import postprocess_mask
    from kneeai.planning import PlanningInputs, optimize_plan, plan_resections, targets_for
    from kneeai.recon import reconstruct_all
    from kneeai.simulation import KinematicConfig, simulate_flexion, simulate_plan

    opts = options or PipelineOptions()
    strategy = _resolve_strategy(opts.strategy)
    case_dir = (Path(out_root) / "cases" / case_id).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    warnings = list(outcome.warnings)

    # --- 1. canonicalise the label map (contract rule 4) ------------------
    opts.stage("canonicalising label map")
    t0 = time.perf_counter()
    canonical, canonical_warnings = canonical_label_volume(outcome.mask)
    warnings.extend(canonical_warnings)
    # int16 is required of segmentation.nii.gz by the contract.
    canonical = canonical.with_array(canonical.array.astype(np.int16))
    timings["canonicalise"] = time.perf_counter() - t0

    # --- 2. post-process --------------------------------------------------
    opts.stage("post-processing mask")
    t0 = time.perf_counter()
    cleaned, pp_report = postprocess_mask(canonical)
    cleaned = cleaned.with_array(cleaned.array.astype(np.int16))
    warnings.extend(pp_report.warnings)
    timings["postprocess"] = time.perf_counter() - t0

    if not cleaned.structures_present():
        raise KneeAIError(
            "the mask contains no labelled voxels after post-processing; there is "
            "nothing to reconstruct"
        )

    # --- 3. contract artifacts, written first -----------------------------
    opts.stage("writing segmentation.nii.gz + result.json")
    t0 = time.perf_counter()
    segmentation_path = case_dir / SEGMENTATION_FILENAME
    write_volume(cleaned, segmentation_path)
    payload = outcome.result_payload(case_id=case_id, mask=cleaned)
    payload["warnings"] = _dedupe([*payload.get("warnings", []), *warnings])
    result_path = write_result_json(payload, case_dir)
    timings["write_contract"] = time.perf_counter() - t0

    artifacts = CaseArtifacts(
        case_id=case_id,
        case_dir=case_dir,
        source=outcome.source,
        result_path=result_path,
        segmentation_path=segmentation_path,
        mask=cleaned,
        truth=dict(outcome.truth),
    )

    # --- 4. reconstruct ---------------------------------------------------
    opts.stage("reconstructing surfaces")
    t0 = time.perf_counter()
    meshes, failures = reconstruct_all(cleaned)
    for structure, reason in (failures or {}).items():
        label = getattr(structure, "value", str(structure))
        warnings.append(f"surface reconstruction failed for {label}: {reason}")
    if not meshes:
        raise KneeAIError("no surfaces could be reconstructed from the mask")
    artifacts.meshes = meshes
    timings["reconstruct"] = time.perf_counter() - t0

    # --- 5. landmarks + patient frame ------------------------------------
    opts.stage("detecting landmarks and building the patient frame")
    t0 = time.perf_counter()
    landmarks, frame = detect_landmarks_and_frame(
        meshes, external=outcome.external_landmarks or None
    )
    artifacts.landmarks, artifacts.frame = landmarks, frame
    timings["landmarks"] = time.perf_counter() - t0

    # --- 6. measure -------------------------------------------------------
    opts.stage("measuring")
    t0 = time.perf_counter()
    measurements = measure_all(cleaned, meshes, landmarks, frame)
    artifacts.measurements = measurements
    timings["measure"] = time.perf_counter() - t0

    side = cleaned.side if cleaned.side is not Side.UNKNOWN else measurements.side

    # --- 7-9. implants, plan, simulation ---------------------------------
    resected: dict[Structure, Any] = {}
    components: dict[str, Any] = {}

    if side is Side.UNKNOWN:
        # Not an exception: the segmentation and measurements are still valid and
        # worth showing. Only the side-dependent stages are withheld.
        warnings.append(
            "Implant matching, planning and simulation were SKIPPED because the "
            "operative side is unknown: medial and lateral cannot be resolved, and "
            "guessing is a wrong-site-surgery class error (core/anatomy.py). "
            "Establish the side and re-run."
        )
    else:
        opts.stage("matching implant configurations")
        t0 = time.perf_counter()
        from kneeai.config import get_settings

        catalog = load_catalog(get_settings().implant_catalog)
        femoral_dims = FemoralDimensions.from_measurements(measurements, side)
        tibial_plane = provisional_tibial_plane(meshes, landmarks, frame)
        tibial_fp = resection_footprint(
            meshes.require(Structure.TIBIA), tibial_plane, frame, provisional=True
        )
        fits, _provenance = match_configurations(
            femoral_dims,
            tibial_fp,
            catalog,
            side=side,
            limit=max(2, opts.implant_candidates),
        )
        if not fits:
            raise KneeAIError("no implant configuration could be evaluated")
        artifacts.fits = fits
        best = fits[0]
        if not best.acceptable:
            warnings.append(
                "The best-fitting implant configuration still carries constraint "
                "violations; see implants.json for the per-criterion reasoning."
            )
        timings["implants"] = time.perf_counter() - t0

        opts.stage(f"planning resections ({strategy.value})")
        t0 = time.perf_counter()
        inputs = PlanningInputs(
            meshes=meshes,
            landmarks=landmarks,
            frame=frame,
            measurements=measurements,
            configuration=best.configuration,
            targets=targets_for(strategy),
            fit=best,
            case_id=case_id,
            label=f"Plan ({strategy.value})",
        )
        plan = plan_resections(inputs)
        if opts.optimize:
            try:
                optimisation = optimize_plan(inputs)
            except PlanningError as exc:
                warnings.append(
                    f"Plan optimisation failed ({exc}); the unoptimised plan was kept."
                )
            else:
                plan = optimisation.plan
                _write_json(
                    case_dir / "optimization.json",
                    {
                        "explanation": optimisation.explanation(),
                        "result": (
                            optimisation.to_dict()
                            if hasattr(optimisation, "to_dict")
                            else None
                        ),
                    },
                )
        artifacts.plan = plan
        warnings.extend(plan.warnings)
        timings["planning"] = time.perf_counter() - t0

        opts.stage("simulating cuts and flexion")
        t0 = time.perf_counter()
        simulation = simulate_plan(plan, meshes, best.configuration, frame)
        artifacts.simulation = simulation
        resected = dict(simulation.resected_meshes)
        components = dict(simulation.component_meshes)
        if not simulation.passed:
            warnings.append(
                f"Geometric simulation reported {len(simulation.failures)} failure(s); "
                "the plan must not be approved until they are resolved."
            )
        step = max(1, int(opts.flexion_step_deg))
        artifacts.kinematics = simulate_flexion(
            simulation,
            landmarks,
            frame,
            config=KinematicConfig(
                angles_deg=tuple(float(a) for a in range(0, 141, step))
            ),
        )
        timings["simulation"] = time.perf_counter() - t0

    # --- 10. meshes -------------------------------------------------------
    opts.stage("exporting GLB meshes")
    t0 = time.perf_counter()
    manifest = export_meshes(
        case_id,
        case_dir / MESH_DIRNAME,
        anatomy=meshes,
        resected=resected,
        components=components,
        compress=opts.compress_meshes,
    )
    warnings.extend(manifest.warnings)
    artifacts.mesh_manifest = manifest
    timings["meshes"] = time.perf_counter() - t0

    # --- 11. derived, application-owned artifacts ------------------------
    opts.stage("writing derived artifacts")
    t0 = time.perf_counter()
    _write_json(case_dir / "measurements.json", measurements.to_dict())
    _write_json(case_dir / "landmarks.json", landmarks.to_dict())
    _write_json(case_dir / "frame.json", frame.to_dict())
    _write_json(case_dir / "mesh_quality.json", meshes.quality_report())
    if artifacts.fits:
        _write_json(case_dir / "implants.json", _implants_payload(artifacts.fits))
    if artifacts.plan is not None:
        _write_json(case_dir / "plan.json", artifacts.plan.to_dict())
    if artifacts.simulation is not None:
        _write_json(case_dir / "simulation.json", artifacts.simulation.to_dict())
    if artifacts.kinematics is not None:
        _write_json(case_dir / "kinematics.json", artifacts.kinematics.to_dict())
    timings["derived"] = time.perf_counter() - t0

    # Rewrite result.json so late-stage warnings are in the manifest, not only
    # the console.
    payload["warnings"] = _dedupe(payload.get("warnings", []) + warnings)
    write_result_json(payload, case_dir)
    artifacts.warnings = list(payload["warnings"])
    artifacts.timings_s = {k: round(v, 3) for k, v in timings.items()}

    log.info(
        "case pipeline complete",
        fields={
            "case_id": case_id,
            "source": outcome.source.value,
            "side": side.value,
            "planned": artifacts.planned,
            "faces": manifest.total_faces,
            "seconds": artifacts.total_seconds,
            "warnings": len(artifacts.warnings),
        },
    )
    return artifacts


def _implants_payload(fits: list[Any]) -> dict[str, Any]:
    """Selection payload including the runner-up and why it lost.

    The UI is required to show the runner-up and the reason it was not chosen, so
    the comparison is computed here rather than left to the client to infer from
    two score numbers.
    """
    best = fits[0]
    payload: dict[str, Any] = {
        "selected": best.to_dict(),
        "rationale": best.rationale(),
        "candidates": [f.to_dict() for f in fits],
        "runner_up": None,
        "disclaimer": DISCLAIMER,
        "method": (
            "Deterministic geometric matching: the femoral component is assessed "
            "dimensionally and the tibial baseplate by footprint polygon against "
            "the patient's own resected cross-section. No learned ranking is used."
        ),
    }
    if len(fits) > 1:
        runner_up = fits[1]
        reasons: list[str] = []
        if runner_up.violations:
            reasons.extend(runner_up.violations)
        deficit = best.score - runner_up.score
        term_by_name = {t.name: t for t in runner_up.femoral_fit.terms}
        for term in best.femoral_fit.terms:
            other = term_by_name.get(term.name)
            if other is not None and other.score_0_100 + 1e-9 < term.score_0_100:
                reasons.append(
                    f"femoral {term.name}: scored {other.score_0_100:.1f} vs "
                    f"{term.score_0_100:.1f} for the selected size "
                    f"({other.explanation})"
                )
        term_by_name = {t.name: t for t in runner_up.tibial_fit.terms}
        for term in best.tibial_fit.terms:
            other = term_by_name.get(term.name)
            if other is not None and other.score_0_100 + 1e-9 < term.score_0_100:
                reasons.append(
                    f"tibial {term.name}: scored {other.score_0_100:.1f} vs "
                    f"{term.score_0_100:.1f} for the selected size "
                    f"({other.explanation})"
                )
        payload["runner_up"] = {
            "fit": runner_up.to_dict(),
            "rationale": runner_up.rationale(),
            "score_deficit": round(deficit, 2),
            "why_it_lost": reasons
            or [
                f"scored {runner_up.score:.1f} against {best.score:.1f}; no single "
                "criterion dominated the difference"
            ],
        }
    return payload
