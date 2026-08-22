"""Measurement engine: the single entry point (PRD §33).

    >>> measurements = measure_all(mask, meshes, landmarks, frame)
    >>> measurements.value(MeasurementName.FEMORAL_ML_WIDTH)

Failure policy: an individual measurement that cannot be computed is recorded in
:attr:`~kneeai.measure.types.MeasurementSet.unavailable` with the reason, and the
rest continue. A case with no long-leg radiograph still gets full femoral and
tibial sizing; it simply has no HKA, and the report says why.

The alternative — raising on the first missing landmark — would make the engine
all-or-nothing, and in practice most cases are missing something.
"""

from __future__ import annotations

from typing import Any

from kneeai.core.anatomy import Side, Structure
from kneeai.core.errors import KneeAIError, MeasurementError
from kneeai.core.provenance import Origin, Provenance
from kneeai.imaging.volume import Volume
from kneeai.landmarks.frame import PatientFrame
from kneeai.landmarks.types import LandmarkSet
from kneeai.logging import get_logger
from kneeai.measure.alignment import measure_alignment
from kneeai.measure.bone import measure_femur, measure_tibia
from kneeai.measure.soft_tissue import measure_cartilage, measure_meniscus
from kneeai.measure.types import MeasurementName, MeasurementSet
from kneeai.recon.mesh import MeshSet

__all__ = ["measure_all", "recompute_affected"]

log = get_logger("measure.engine")


def measure_all(
    mask: Volume | None,
    meshes: MeshSet,
    landmarks: LandmarkSet,
    frame: PatientFrame,
    *,
    include_soft_tissue: bool = True,
) -> MeasurementSet:
    """Compute every available measurement for a case.

    Args:
        mask: label volume. Required for cartilage and meniscus volumes, which are
            measured from voxels rather than meshes — a voxel count is a more
            direct volume estimate for a thin structure than an enclosed mesh
            volume, which depends on the surface having closed correctly.
        meshes: reconstructed surfaces in the IMAGE frame.
        landmarks: detected landmarks in the IMAGE frame.
        frame: the patient anatomical frame.
    """
    if landmarks.frame is not frame.source_frame:
        raise MeasurementError(
            f"landmarks are in '{landmarks.frame.value}' but the patient frame was "
            f"built from '{frame.source_frame.value}'"
        )
    if meshes.frame is not frame.source_frame:
        raise MeasurementError(
            f"meshes are in '{meshes.frame.value}' but the patient frame was built "
            f"from '{frame.source_frame.value}'"
        )
    if frame.side is Side.UNKNOWN:
        raise MeasurementError(
            "the operative side must be known: medial and lateral measurements are "
            "otherwise undefined"
        )

    result = MeasurementSet(side=frame.side)
    detail: dict[str, Any] = {}

    # --- bone ---
    for label, fn in (("femur", measure_femur), ("tibia", measure_tibia)):
        try:
            result.add_all(fn(meshes, landmarks, frame))
        except KneeAIError as exc:
            result.mark_unavailable(f"{label}_measurements", str(exc))
            log.warning(
                "bone measurements failed",
                fields={"group": label, "reason": str(exc)},
            )

    # --- alignment ---
    try:
        alignment, alignment_unavailable = measure_alignment(landmarks, frame)
        result.add_all(alignment)
        for key, reason in alignment_unavailable.items():
            result.mark_unavailable(key, reason)
    except KneeAIError as exc:
        result.mark_unavailable("alignment_measurements", str(exc))
        log.warning("alignment measurements failed", fields={"reason": str(exc)})

    # --- soft tissue ---
    if include_soft_tissue:
        if mask is None:
            reason = (
                "cartilage and meniscus quantification requires the label volume; "
                "only surfaces were supplied"
            )
            for name in (
                MeasurementName.FEMORAL_CARTILAGE_VOLUME,
                MeasurementName.MEDIAL_TIBIAL_CARTILAGE_VOLUME,
                MeasurementName.LATERAL_TIBIAL_CARTILAGE_VOLUME,
                MeasurementName.MEDIAL_MENISCUS_VOLUME,
                MeasurementName.LATERAL_MENISCUS_VOLUME,
            ):
                result.mark_unavailable(name, reason)
        else:
            try:
                cartilage, cartilage_unavailable, cartilage_detail = measure_cartilage(
                    mask, meshes
                )
                result.add_all(cartilage)
                for key, reason in cartilage_unavailable.items():
                    result.mark_unavailable(key, reason)
                detail["cartilage"] = cartilage_detail
            except KneeAIError as exc:
                result.mark_unavailable("cartilage_measurements", str(exc))
                log.warning("cartilage measurements failed", fields={"reason": str(exc)})

            try:
                meniscus, meniscus_unavailable, meniscus_detail = measure_meniscus(
                    mask, meshes, frame
                )
                result.add_all(meniscus)
                for key, reason in meniscus_unavailable.items():
                    result.mark_unavailable(key, reason)
                detail["meniscus"] = meniscus_detail
            except KneeAIError as exc:
                result.mark_unavailable("meniscus_measurements", str(exc))
                log.warning("meniscus measurements failed", fields={"reason": str(exc)})

    # --- provenance ---
    # Origin propagates from the landmarks: a measurement built on surgeon-adjusted
    # landmarks is no longer purely AI-derived, and the report must reflect that
    # (PRD §32).
    origin = Origin.DETERMINISTIC
    if landmarks.surgeon_edited():
        origin = Origin.SURGEON_EDITED_AI
    elif landmarks.provenance is not None and landmarks.provenance.origin is Origin.AI:
        origin = Origin.AI

    result.provenance = Provenance.for_component(
        "measure.engine",
        origin=origin,
        model_id=landmarks.provenance.model_id if landmarks.provenance else None,
        model_version=(
            landmarks.provenance.model_version if landmarks.provenance else None
        ),
        parameters={
            "side": frame.side.value,
            "patient_frame_method": frame.method,
            "n_landmarks": len(landmarks),
            "landmark_mean_confidence": round(landmarks.mean_confidence, 4),
            "surgeon_edited_landmarks": [
                lm.name.value for lm in landmarks.surgeon_edited()
            ],
            "structures_available": sorted(s.value for s in meshes),
            "mask_spacing_mm": (
                [round(float(s), 4) for s in mask.spacing] if mask is not None else None
            ),
            "soft_tissue_detail": detail,
        },
    )

    log.info(
        "measurements complete",
        fields={
            "computed": len(result),
            "unavailable": len(result.unavailable),
            "with_caveats": len(result.with_caveats()),
            "outside_reference_range": [
                m.name.value for m in result.out_of_reference_range()
            ],
        },
    )
    return result


def recompute_affected(
    previous: MeasurementSet,
    mask: Volume | None,
    meshes: MeshSet,
    landmarks: LandmarkSet,
    frame: PatientFrame,
) -> tuple[MeasurementSet, list[str]]:
    """Recompute everything and report which values actually changed.

    Called after a surgeon edits a landmark. Recomputation is wholesale — it is
    cheap and avoids a stale-dependency bug — but the *diff* is what the UI needs,
    so the surgeon sees which numbers their edit moved rather than a silently
    refreshed panel.
    """
    updated = measure_all(mask, meshes, landmarks, frame)

    changed: list[str] = []
    for name, measurement in updated.measurements.items():
        before = previous.get(name)
        if before is None:
            changed.append(f"{name.value}: newly available ({measurement.format()})")
            continue
        if abs(before.value - measurement.value) > 1e-6:
            changed.append(
                f"{name.value}: {before.format()} -> {measurement.format()}"
            )
    for name in previous.measurements:
        if name not in updated.measurements:
            changed.append(f"{name.value}: no longer available")

    log.info("measurements recomputed", fields={"n_changed": len(changed)})
    return updated, changed
