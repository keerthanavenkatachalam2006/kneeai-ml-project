"""Robotic execution readiness, as a percentage with its working shown.

A single number is a dangerous way to summarise a surgical case, so this module is
built to make the number auditable rather than impressive:

**It scores the artefacts, not the patient.** The question answered is "how
completely and self-consistently does this case support a robotic execution of its
plan" — segmentation confidence, watertight surfaces, a reliable anatomical frame,
computable measurements, an acceptable implant fit, a simulation that passed. It is
*not* a judgement that a patient is a good candidate for robotic surgery. That
judgement needs the patient, and this pipeline has never met them.

**Blockers cap, they do not subtract.** A weighted average will happily report 78 %
for a case whose geometric simulation failed, because five good criteria outvote
one bad one. Anything that genuinely prevents execution therefore imposes a
ceiling on the total instead of contributing a term to it. A case with an unknown
operative side cannot exceed 25 % no matter how clean its meshes are.

**Every criterion reports what it read.** Each entry carries the inputs it scored
and a human-readable detail string, so a low score points at the artefact that
caused it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "READINESS_DISCLAIMER",
    "ReadinessCriterion",
    "RoboticReadiness",
    "assess_readiness",
]

READINESS_DISCLAIMER = (
    "Simulation readiness only. This measures whether the case artefacts are "
    "complete and internally consistent enough to animate a constrained robotic "
    "execution of the plan. It is not a clinical assessment, not a recommendation "
    "for robotic surgery, and no hardware is controlled by this system."
)

#: Band thresholds on the final percentage.
BANDS: tuple[tuple[float, str, str], ...] = (
    (85.0, "ready", "Artefacts are complete and consistent; simulation can proceed."),
    (
        60.0,
        "conditional",
        "Simulation can proceed, but named gaps below should be resolved first.",
    ),
    (
        0.0,
        "not_ready",
        "Material gaps prevent a meaningful robotic simulation of this plan.",
    ),
)


def _band(percentage: float) -> tuple[str, str]:
    for threshold, name, note in BANDS:
        if percentage >= threshold:
            return name, note
    return "not_ready", BANDS[-1][2]


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if v is not None and _is_finite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _is_finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


@dataclass
class ReadinessCriterion:
    """One weighted component of the readiness percentage."""

    name: str
    display_name: str
    weight: float
    score: float
    detail: str
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted(self) -> float:
        return self.weight * self.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "weight": round(self.weight, 4),
            "score": round(self.score, 4),
            "score_pct": round(100.0 * self.score, 1),
            "weighted_contribution_pct": round(100.0 * self.weighted, 1),
            "detail": self.detail,
            "inputs": self.inputs,
        }


@dataclass
class RoboticReadiness:
    """The readiness percentage and everything behind it."""

    percentage: float
    band: str
    band_note: str
    criteria: list[ReadinessCriterion] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    ceiling_pct: float = 100.0
    raw_percentage: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "percentage": round(self.percentage, 1),
            "band": self.band,
            "band_note": self.band_note,
            "raw_weighted_percentage": round(self.raw_percentage, 1),
            "ceiling_pct": round(self.ceiling_pct, 1),
            "capped": self.percentage < self.raw_percentage - 1e-9,
            "criteria": [c.to_dict() for c in self.criteria],
            "blockers": list(self.blockers),
            "n_blockers": len(self.blockers),
            "warnings": list(self.warnings),
            "disclaimer": READINESS_DISCLAIMER,
            "method": (
                "Weighted mean of six criteria scored from the case artefacts, with "
                "hard blockers imposing a ceiling on the total rather than "
                "contributing a term. Weights and per-criterion scores are reported "
                "in full so the number can be recomputed by hand."
            ),
        }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def _segmentation_criterion(result: Mapping[str, Any]) -> ReadinessCriterion:
    segmentation = result.get("segmentation") or {}
    confidence = (result.get("confidence") or {}).get("per_structure") or {}
    present = list(segmentation.get("structures_present") or [])
    source = str(result.get("source", "unknown"))

    bone = [confidence.get("femur"), confidence.get("tibia")]
    bone_score = _mean([v for v in bone if _is_finite(v)])
    if not _is_finite(bone_score):
        bone_score = 0.0

    # Bone drives the cuts, so it is weighted above cartilage here. Cartilage
    # accuracy matters to the OA question, not to whether a plane can be cut.
    required = {"femur", "tibia"}
    have_bone = required.issubset(set(present))
    coverage = len(present) / 5.0

    score = _clamp(0.6 * bone_score + 0.25 * coverage + (0.15 if have_bone else 0.0))
    if source in ("phantom", "mock"):
        # Synthetic geometry is perfectly self-consistent and tells you nothing
        # about readiness on a real scan.
        score = min(score, 0.5)

    return ReadinessCriterion(
        name="segmentation_quality",
        display_name="Segmentation quality",
        weight=0.20,
        score=score,
        detail=(
            f"{len(present)}/5 structures segmented; mean bone confidence "
            f"{bone_score:.2f}" + ("" if have_bone else "; FEMUR OR TIBIA MISSING")
            + (f"; source is {source}" if source in ("phantom", "mock") else "")
        ),
        inputs={
            "structures_present": present,
            "femur_confidence": confidence.get("femur"),
            "tibia_confidence": confidence.get("tibia"),
            "source": source,
        },
    )


def _mesh_criterion(mesh_quality: Mapping[str, Any] | None) -> ReadinessCriterion:
    if not mesh_quality:
        return ReadinessCriterion(
            name="surface_reconstruction",
            display_name="Surface reconstruction",
            weight=0.15,
            score=0.0,
            detail="no mesh quality report for this case",
        )

    meshes = mesh_quality.get("meshes")
    if isinstance(mesh_quality, Mapping) and not isinstance(meshes, list):
        # Some writers key the report by structure name.
        meshes = [
            v for v in mesh_quality.values() if isinstance(v, Mapping) and "n_faces" in v
        ]
    meshes = meshes or []

    total = len(meshes)
    watertight = sum(1 for m in meshes if m.get("is_watertight"))
    fraction = watertight / total if total else 0.0

    return ReadinessCriterion(
        name="surface_reconstruction",
        display_name="Surface reconstruction",
        weight=0.15,
        score=_clamp(fraction),
        detail=(
            f"{watertight}/{total} reconstructed surfaces are watertight. A "
            "non-watertight surface does not enclose a well-defined volume, so "
            "resection volumes computed against it are unreliable."
            if total
            else "no surfaces reconstructed"
        ),
        inputs={"n_meshes": total, "n_watertight": watertight},
    )


def _frame_criterion(
    landmarks: Mapping[str, Any] | None, frame: Mapping[str, Any] | None
) -> ReadinessCriterion:
    if not landmarks:
        return ReadinessCriterion(
            name="anatomical_frame",
            display_name="Anatomical frame",
            weight=0.15,
            score=0.0,
            detail="no landmarks detected, so no patient coordinate frame exists",
        )

    mean_confidence = landmarks.get("mean_confidence")
    score = float(mean_confidence) if _is_finite(mean_confidence) else 0.0
    method = str((frame or {}).get("method") or "unknown")

    # The fallback frame is built without the femoral head and ankle centres, so
    # it defines rotation but not the mechanical axis. Usable, but not complete.
    fallback = "shaft" in method.lower() or "tea" in method.lower()
    if fallback:
        score = min(score, 0.75)

    return ReadinessCriterion(
        name="anatomical_frame",
        display_name="Anatomical frame",
        weight=0.15,
        score=_clamp(score),
        detail=(
            f"{landmarks.get('n_landmarks', 0)} landmarks, mean confidence "
            f"{score:.2f}; frame method '{method}'"
            + (
                ". This is the fallback frame: the femoral head and ankle centres "
                "are outside a knee MR field of view, so the mechanical axis is "
                "not established."
                if fallback
                else ""
            )
        ),
        inputs={
            "n_landmarks": landmarks.get("n_landmarks"),
            "mean_confidence": mean_confidence,
            "frame_method": method,
            "fallback_frame": fallback,
        },
    )


def _measurement_criterion(
    measurements: Mapping[str, Any] | None,
) -> ReadinessCriterion:
    if not measurements:
        return ReadinessCriterion(
            name="measurement_completeness",
            display_name="Measurement completeness",
            weight=0.15,
            score=0.0,
            detail="no measurements computed for this case",
        )

    computed = list(measurements.get("measurements") or [])
    unavailable = measurements.get("unavailable") or {}
    n_unavailable = len(unavailable)
    n_total = len(computed) + n_unavailable
    completeness = len(computed) / n_total if n_total else 0.0

    out_of_range = [
        m.get("name")
        for m in computed
        if m.get("in_reference_range") is False
    ]
    checked = [m for m in computed if m.get("in_reference_range") is not None]
    in_range_fraction = (
        (len(checked) - len(out_of_range)) / len(checked) if checked else 1.0
    )

    score = _clamp(0.6 * completeness + 0.4 * in_range_fraction)

    return ReadinessCriterion(
        name="measurement_completeness",
        display_name="Measurement completeness",
        weight=0.15,
        score=score,
        detail=(
            f"{len(computed)} of {n_total} measurements computed; "
            f"{len(out_of_range)} of {len(checked)} range-checked values outside "
            "their reference range"
            + (
                f". Unavailable: {', '.join(sorted(unavailable)[:4])}"
                + ("…" if n_unavailable > 4 else "")
                if n_unavailable
                else ""
            )
        ),
        inputs={
            "n_computed": len(computed),
            "n_unavailable": n_unavailable,
            "out_of_range": out_of_range,
        },
    )


def _implant_criterion(implants: Mapping[str, Any] | None) -> ReadinessCriterion:
    if not implants or not implants.get("selected"):
        return ReadinessCriterion(
            name="implant_fit",
            display_name="Implant fit",
            weight=0.15,
            score=0.0,
            detail="no implant configuration was matched for this case",
        )

    selected = implants["selected"]
    raw_score = selected.get("score")
    score = float(raw_score) if _is_finite(raw_score) else 0.0
    # Scores are already 0..1 in the matcher; guard anyway.
    if score > 1.0:
        score = score / 100.0
    acceptable = bool(selected.get("acceptable"))
    violations = list(selected.get("violations") or [])

    if not acceptable:
        score = min(score, 0.5)

    configuration = selected.get("configuration") or {}
    return ReadinessCriterion(
        name="implant_fit",
        display_name="Implant fit",
        weight=0.15,
        score=_clamp(score),
        detail=(
            f"{configuration.get('label', 'selected configuration')} — fit score "
            f"{score:.2f}, {'acceptable' if acceptable else 'NOT acceptable'}"
            + (f", {len(violations)} constraint violation(s)" if violations else "")
        ),
        inputs={
            "configuration_id": configuration.get("configuration_id"),
            "score": raw_score,
            "acceptable": acceptable,
            "violations": violations,
        },
    )


def _simulation_criterion(
    simulation: Mapping[str, Any] | None,
    safety: Mapping[str, Any] | None,
) -> ReadinessCriterion:
    if not simulation:
        return ReadinessCriterion(
            name="plan_simulation_integrity",
            display_name="Plan & simulation integrity",
            weight=0.20,
            score=0.0,
            detail="the plan has not been geometrically simulated",
        )

    passed = bool(simulation.get("passed"))
    n_failures = int(simulation.get("n_failures") or 0)
    n_warnings = int(simulation.get("n_warnings") or 0)

    cuts = list(simulation.get("cuts") or [])
    succeeded = sum(1 for c in cuts if c.get("succeeded"))
    cut_fraction = succeeded / len(cuts) if cuts else 0.0

    score = 0.5 * cut_fraction + (0.5 if passed else 0.0)
    score -= 0.05 * min(n_warnings, 4)

    if safety:
        blocking = int(safety.get("n_blocking_failures") or 0)
        if blocking:
            score = min(score, 0.4)

    return ReadinessCriterion(
        name="plan_simulation_integrity",
        display_name="Plan & simulation integrity",
        weight=0.20,
        score=_clamp(score),
        detail=(
            f"{succeeded}/{len(cuts)} planned cuts simulated successfully; "
            f"{n_failures} failure(s), {n_warnings} warning(s)"
            + (
                f"; {safety.get('n_blocking_failures')} blocking safety check(s) failed"
                if safety and safety.get("n_blocking_failures")
                else ""
            )
        ),
        inputs={
            "passed": passed,
            "n_failures": n_failures,
            "n_warnings": n_warnings,
            "n_cuts": len(cuts),
            "n_cuts_succeeded": succeeded,
            "interpenetration_mm": simulation.get("interpenetration_mm"),
            "seating_gap_mm": simulation.get("seating_gap_mm"),
            "safety_blocking_failures": (safety or {}).get("n_blocking_failures"),
        },
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def assess_readiness(
    *,
    result: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
    simulation: Mapping[str, Any] | None = None,
    measurements: Mapping[str, Any] | None = None,
    implants: Mapping[str, Any] | None = None,
    landmarks: Mapping[str, Any] | None = None,
    frame: Mapping[str, Any] | None = None,
    mesh_quality: Mapping[str, Any] | None = None,
    safety_report: Mapping[str, Any] | None = None,
) -> RoboticReadiness:
    """Score how ready a case is for a simulated robotic execution of its plan."""
    criteria = [
        _segmentation_criterion(result),
        _mesh_criterion(mesh_quality),
        _frame_criterion(landmarks, frame),
        _measurement_criterion(measurements),
        _implant_criterion(implants),
        _simulation_criterion(simulation, safety_report),
    ]

    total_weight = sum(c.weight for c in criteria)
    raw = 100.0 * sum(c.weighted for c in criteria) / total_weight

    # --- blockers, which impose a ceiling rather than a penalty ---
    blockers: list[dict[str, Any]] = []
    ceiling = 100.0

    side = str((result.get("geometry") or {}).get("side", "unknown"))
    if side not in ("left", "right"):
        blockers.append(
            {
                "code": "side_unknown",
                "ceiling_pct": 25.0,
                "message": (
                    "The operative side is not established. Medial and lateral "
                    "cannot be resolved, which is a wrong-site-surgery class error, "
                    "so no plan may be executed."
                ),
            }
        )
        ceiling = min(ceiling, 25.0)

    if not plan:
        blockers.append(
            {
                "code": "no_plan",
                "ceiling_pct": 30.0,
                "message": "There is no surgical plan to execute.",
            }
        )
        ceiling = min(ceiling, 30.0)

    if simulation is not None and not simulation.get("passed"):
        blockers.append(
            {
                "code": "simulation_failed",
                "ceiling_pct": 45.0,
                "message": (
                    f"The geometric simulation reported "
                    f"{simulation.get('n_failures', 0)} failure(s). The plan must "
                    "not be executed until they are resolved."
                ),
            }
        )
        ceiling = min(ceiling, 45.0)

    if safety_report and safety_report.get("n_blocking_failures"):
        failed = [
            c.get("name")
            for c in safety_report.get("checks", []) or []
            if c.get("blocking") and not c.get("passed")
        ]
        blockers.append(
            {
                "code": "safety_checks_failed",
                "ceiling_pct": 50.0,
                "message": "Blocking safety checks failed: " + ", ".join(failed),
            }
        )
        ceiling = min(ceiling, 50.0)

    source = str(result.get("source", "unknown"))
    warnings: list[str] = []
    if source in ("phantom", "mock"):
        warnings.append(
            f"This case is {source} geometry, not a patient scan. The readiness "
            "percentage describes synthetic data and does not transfer to a real case."
        )
    if source == "ground_truth":
        warnings.append(
            "This case uses expert ground-truth segmentation, not model output. "
            "Readiness computed here is an upper bound on what the model achieves."
        )

    percentage = min(raw, ceiling)
    band, note = _band(percentage)

    return RoboticReadiness(
        percentage=percentage,
        band=band,
        band_note=note,
        criteria=criteria,
        blockers=blockers,
        ceiling_pct=ceiling,
        raw_percentage=raw,
        warnings=warnings,
    )
