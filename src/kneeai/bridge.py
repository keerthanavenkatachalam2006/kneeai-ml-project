"""Bridge between the existing backend segmentation and the new kneeai pipeline.

This module wraps the legacy ``backend.app.segmentation.inference.KneeSegmentationModel``
so its output can be consumed by the new ``kneeai.api.pipeline.run_case_pipeline``.

The key translation:
    1. Load a NIfTI volume from disk
    2. Run the existing SegResNet model (backend/) to get a label volume
    3. Wrap that into a :class:`~kneeai.api.adapter.SegmentationOutcome` compatible
       with the new pipeline
    4. Optionally run the legacy measurement, reconstruction, implant matching,
       and OA analysis for comparison

Usage::

    from kneeai.bridge import BridgeConfig, run_bridge_pipeline

    result = run_bridge_pipeline(
        case_id="oaizib_001",
        config=BridgeConfig(image_path=Path("data/processed/extracted/imagesTr/oaizib_001_0000.nii.gz")),
    )
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from kneeai.logging import get_logger

__all__ = [
    "BridgeConfig",
    "LegacySegmentationResult",
    "run_legacy_segmentation",
    "wrap_as_outcome",
    "run_bridge_pipeline",
]

log = get_logger("kneeai.bridge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The existing backend lives here
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent / "backend"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class BridgeConfig:
    """Configuration for the bridge pipeline."""

    # Input image path (NIfTI)
    image_path: Path | None = None

    # Existing model checkpoint
    model_path: Path = _PROJECT_ROOT / "models" / "segresnet_knee" / "best_model.pth"

    # Data roots
    data_root: Path = _PROJECT_ROOT / "data" / "processed" / "extracted"

    # Output directory for new pipeline artifacts
    artifact_root: Path = _PROJECT_ROOT / "artifacts"

    # Device for inference
    device: str | None = None

    # Operative side (default RIGHT for OAIZIB-CM)
    side: str = "right"

    # Pipeline options
    strategy: str = "restricted_kinematic"
    optimize: bool = True

    # Whether to also run legacy measurements for comparison
    run_legacy_comparison: bool = False


@dataclass
class LegacySegmentationResult:
    """Result from the existing backend segmentation model."""

    label_volume: np.ndarray
    confidence_map: np.ndarray | None = None
    voxel_size: tuple = (0.7, 0.3646, 0.3646)
    image_shape: tuple = (160, 384, 384)
    inference_time_s: float = 0.0
    model_path: str = ""


# ---------------------------------------------------------------------------
# Legacy segmentation (existing backend)
# ---------------------------------------------------------------------------


def _ensure_backend_importable() -> None:
    """Add the backend directory to sys.path so its modules are importable."""
    backend_str = str(_BACKEND_ROOT)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def run_legacy_segmentation(
    image: np.ndarray,
    voxel_size: tuple = (0.7, 0.3646, 0.3646),
    config: BridgeConfig | None = None,
) -> LegacySegmentationResult:
    """Run the existing backend's KneeSegmentationModel on a raw image array.

    Args:
        image: 3D numpy array (the raw MRI volume).
        voxel_size: Physical voxel spacing in mm.
        config: Bridge configuration.

    Returns:
        LegacySegmentationResult with the label volume and optional confidence map.
    """
    _ensure_backend_importable()
    cfg = config or BridgeConfig()

    from app.segmentation.inference import KneeSegmentationModel

    model = KneeSegmentationModel(
        model_path=cfg.model_path,
        device=cfg.device,
    )

    if not model.load():
        raise RuntimeError(
            f"Failed to load segmentation model from {cfg.model_path}. "
            "Ensure the checkpoint exists."
        )

    t0 = time.perf_counter()
    labels, confidence = model.predict_with_confidence(image, voxel_size)
    elapsed = time.perf_counter() - t0

    log.info(
        "legacy segmentation complete",
        fields={
            "model_path": str(cfg.model_path),
            "inference_time_s": round(elapsed, 2),
            "structures_found": int(len(np.unique(labels)) - 1),  # exclude background
            "image_shape": list(image.shape),
        },
    )

    return LegacySegmentationResult(
        label_volume=labels,
        confidence_map=confidence,
        voxel_size=voxel_size,
        image_shape=image.shape,
        inference_time_s=elapsed,
        model_path=str(cfg.model_path),
    )


# ---------------------------------------------------------------------------
# Convert legacy output → new pipeline's SegmentationOutcome
# ---------------------------------------------------------------------------


def wrap_as_outcome(
    legacy_result: LegacySegmentationResult,
    *,
    image_path: Path | None = None,
    side: str = "right",
) -> "SegmentationOutcome":
    """Convert a legacy segmentation result into the new pipeline's input format.

    This creates a :class:`~kneeai.api.adapter.SegmentationOutcome` that
    ``run_case_pipeline`` can consume directly.
    """
    from kneeai.api.adapter import SegmentationOutcome
    from kneeai.core.anatomy import DEFAULT_LABEL_MAP, Side, Structure
    from kneeai.core.provenance import Origin, Provenance
    from kneeai.imaging.volume import Volume
    from kneeai.ml.contract import ArtifactSource
    from kneeai.version import PLATFORM_VERSION

    # Map the legacy label integers to the new system's Structure enum
    # Legacy: 0=BG, 1=Femur, 2=Fem.Cart, 3=Tibia, 4=Med.Tib.Cart, 5=Lat.Tib.Cart
    # The new system's DEFAULT_LABEL_MAP uses the same convention
    label_map = {
        Structure.BACKGROUND: 0,
        Structure.FEMUR: 1,
        Structure.FEMORAL_CARTILAGE: 2,
        Structure.TIBIA: 3,
        Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
        Structure.LATERAL_TIBIAL_CARTILAGE: 5,
    }

    # Determine operative side
    resolved_side = Side.RIGHT if side.lower() == "right" else Side.LEFT

    # Create a Volume object (the new system's representation)
    spacing = np.array(legacy_result.voxel_size, dtype=np.float64)

    mask = Volume.from_labels(
        array=legacy_result.label_volume.astype(np.int16),
        label_map=label_map,
        spacing=spacing,
        side=resolved_side,
    )

    # Build per-structure confidence from the confidence map
    confidence: dict[Structure, float] = {}
    if legacy_result.confidence_map is not None:
        for structure, label_idx in label_map.items():
            if structure is Structure.BACKGROUND:
                continue
            structure_mask = legacy_result.label_volume == label_idx
            if structure_mask.any():
                confidence[structure] = float(
                    legacy_result.confidence_map[structure_mask].mean()
                )
            else:
                confidence[structure] = float("nan")

    # Build provenance
    provenance = Provenance(
        component="kneeai.bridge",
        component_version=PLATFORM_VERSION,
        origin=Origin.AI,
        model_id="segresnet_knee",
        model_version="1.0",
        parameters={
            "model": "SegResNet (backend/app/segmentation)",
            "model_path": legacy_result.model_path,
            "inference_time_s": legacy_result.inference_time_s,
            "image_path": str(image_path) if image_path else None,
            "voxel_size": list(legacy_result.voxel_size),
        },
        notes=(
            "Segmentation produced by the existing backend SegResNet model, "
            "bridged into the new pipeline via kneeai.bridge."
        ),
    )

    return SegmentationOutcome(
        mask=mask,
        source=ArtifactSource.AI_PREDICTION,
        confidence=confidence,
        confidence_method=(
            "mean softmax probability over voxels assigned to each structure, "
            "computed by the legacy KneeSegmentationModel"
        ),
        provenance=provenance,
        model_id="segresnet_knee",
        model_version="1.0",
        architecture="SegResNet",
        warnings=[],
        requested_backend=ArtifactSource.AI_PREDICTION,
    )


# ---------------------------------------------------------------------------
# Legacy comparison pipeline (measurements, reconstruction, implant matching)
# ---------------------------------------------------------------------------


def run_legacy_analysis(
    legacy_result: LegacySegmentationResult,
) -> dict[str, Any]:
    """Run the existing backend's measurement, reconstruction, and implant matching.

    Returns a dictionary with the legacy pipeline's results for comparison.
    """
    _ensure_backend_importable()

    from app.measurements import KneeMeasurementEngine
    from app.implant_database import ImplantMatcher
    from app.oa_research import OAResearchEngine

    results: dict[str, Any] = {}

    # Measurements
    engine = KneeMeasurementEngine()
    measurements = engine.measure_case(
        legacy_result.label_volume, legacy_result.voxel_size
    )
    results["measurements"] = measurements

    # Implant matching
    matcher = ImplantMatcher()
    implant_results = matcher.match_all(measurements)
    results["implants"] = {
        component: [match.model_dump() for match in matches]
        for component, matches in implant_results.items()
    }

    # OA assessment
    oa_engine = OAResearchEngine()
    oa_result = oa_engine.assess_oa_morphometrics(measurements)
    results["oa_assessment"] = oa_result

    return results


# ---------------------------------------------------------------------------
# Full bridge pipeline
# ---------------------------------------------------------------------------


def run_bridge_pipeline(
    case_id: str,
    config: BridgeConfig | None = None,
    *,
    on_stage: Any = None,
) -> dict[str, Any]:
    """Run the full unified pipeline: legacy segmentation → new planning pipeline.

    This is the main entry point for running a case through both systems:
    1. Load the NIfTI image
    2. Run the existing SegResNet model
    3. Convert to new pipeline format
    4. Run the full new pipeline (landmarks → planning → simulation)
    5. Optionally run legacy analysis for comparison

    Args:
        case_id: Case identifier (e.g. "oaizib_001").
        config: Bridge configuration.
        on_stage: Optional callback for progress reporting.

    Returns:
        Dictionary with pipeline results including artifacts path.
    """
    import nibabel as nib

    from kneeai.api.pipeline import CaseArtifacts, PipelineOptions, run_case_pipeline

    cfg = config or BridgeConfig()
    results: dict[str, Any] = {"case_id": case_id, "stages": {}, "timings_s": {}}

    # --- 1. Load the NIfTI image ---
    if on_stage:
        on_stage("loading image")

    image_path = cfg.image_path
    if image_path is None:
        # Try to find it in the standard data directory
        candidates = [
            cfg.data_root / "imagesTr" / f"{case_id}_0000.nii.gz",
            cfg.data_root / "imagesTs" / f"{case_id}_0000.nii.gz",
        ]
        for candidate in candidates:
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            raise FileNotFoundError(
                f"No image found for case {case_id}. Searched: {candidates}"
            )

    nii = nib.load(str(image_path))
    image_array = np.asarray(nii.dataobj, dtype=np.float32)
    voxel_size = tuple(float(x) for x in nii.header.get_zooms()[:3])

    log.info(
        "image loaded",
        fields={
            "case_id": case_id,
            "image_path": str(image_path),
            "shape": list(image_array.shape),
            "voxel_size": list(voxel_size),
        },
    )

    # --- 2. Run legacy segmentation ---
    if on_stage:
        on_stage("running segmentation (existing SegResNet)")

    t0 = time.perf_counter()
    legacy_result = run_legacy_segmentation(image_array, voxel_size, cfg)
    results["timings_s"]["segmentation"] = round(time.perf_counter() - t0, 3)
    results["stages"]["segmentation"] = {
        "inference_time_s": legacy_result.inference_time_s,
        "structures_found": int(len(np.unique(legacy_result.label_volume)) - 1),
        "image_shape": list(legacy_result.image_shape),
        "voxel_size": list(legacy_result.voxel_size),
    }

    # --- 3. Convert to new pipeline format ---
    if on_stage:
        on_stage("converting to new pipeline format")

    outcome = wrap_as_outcome(legacy_result, image_path=image_path, side=cfg.side)

    # --- 4. Run the new pipeline ---
    if on_stage:
        on_stage("running unified pipeline (landmarks → planning → simulation)")

    t0 = time.perf_counter()
    pipeline_opts = PipelineOptions(
        strategy=cfg.strategy,
        optimize=cfg.optimize,
        on_stage=on_stage,
    )

    artifacts: CaseArtifacts = run_case_pipeline(
        outcome,
        case_id=case_id,
        out_root=cfg.artifact_root,
        options=pipeline_opts,
    )
    results["timings_s"]["pipeline"] = round(time.perf_counter() - t0, 3)
    results["stages"]["pipeline"] = {
        "planned": artifacts.planned,
        "side": artifacts.side.value,
        "warnings_count": len(artifacts.warnings),
        "total_seconds": artifacts.total_seconds,
        "timings": artifacts.timings_s,
    }
    results["case_dir"] = str(artifacts.case_dir)
    results["warnings"] = artifacts.warnings

    # --- 5. Optionally run legacy comparison ---
    if cfg.run_legacy_comparison:
        if on_stage:
            on_stage("running legacy analysis for comparison")

        t0 = time.perf_counter()
        legacy_analysis = run_legacy_analysis(legacy_result)
        results["timings_s"]["legacy_comparison"] = round(time.perf_counter() - t0, 3)
        results["stages"]["legacy_comparison"] = legacy_analysis

    # --- Summary ---
    results["total_time_s"] = round(
        sum(results["timings_s"].values()), 3
    )
    results["artifact_paths"] = {
        name: str(path) for name, path in artifacts.artifact_paths().items()
    }

    log.info(
        "bridge pipeline complete",
        fields={
            "case_id": case_id,
            "total_time_s": results["total_time_s"],
            "planned": artifacts.planned,
            "artifacts": len(results["artifact_paths"]),
        },
    )

    return results
