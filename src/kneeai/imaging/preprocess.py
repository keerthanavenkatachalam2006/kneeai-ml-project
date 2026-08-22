"""Preprocessing: canonical orientation, resampling, ROI cropping, intensity norm.

Two rules govern this module.

**Geometry is never lost.** Every operation updates the volume's origin/spacing/
direction so ``voxel_to_image`` stays correct. That is what allows a mask
predicted on a cropped, resampled volume to be mapped back onto the original
acquisition grid exactly, via :func:`resample_like`, rather than approximately.

**Preprocessing is recorded, not implied.** The parameters actually applied are
stored in the result's provenance so inference can reproduce the exact transform
chain a model was trained with. A train/inference preprocessing mismatch is
silent and degrades accuracy without ever raising.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import Modality
from kneeai.core.errors import PreprocessingError, ValidationError
from kneeai.core.provenance import Provenance
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger

__all__ = [
    "IntensityNorm",
    "PreprocessingSpec",
    "PreprocessedVolume",
    "preprocess",
    "normalize_intensity",
    "resample_like",
    "detect_roi",
    "MR_3D_SEGMENTATION_SPEC",
    "CT_BONE_SPEC",
]

log = get_logger("imaging.preprocess")


class IntensityNorm(str, Enum):
    NONE = "none"

    #: Zero mean / unit variance over the foreground. The default for MR, where
    #: absolute intensity carries no physical meaning and varies by scanner.
    ZSCORE = "zscore"

    #: Robust min-max into [0, 1] using percentile clipping.
    PERCENTILE = "percentile"

    #: Clip to a fixed HU window then scale to [0, 1]. Correct for CT, where
    #: intensities are calibrated and a z-score would discard that calibration.
    CT_WINDOW = "ct_window"


@dataclass(frozen=True)
class PreprocessingSpec:
    """A named, reproducible preprocessing configuration.

    Stored alongside every trained model. Inference reuses the model's spec
    rather than a global default.
    """

    name: str = "default"
    orientation: str = "LPS"
    target_spacing_mm: tuple[float, float, float] | None = (0.5, 0.5, 0.5)
    intensity: IntensityNorm = IntensityNorm.ZSCORE

    #: Percentile clip bounds for PERCENTILE / ZSCORE foreground estimation.
    clip_percentiles: tuple[float, float] = (0.5, 99.5)

    #: HU window for CT_WINDOW: (centre, width).
    ct_window: tuple[float, float] = (400.0, 1800.0)

    #: Exclude background from z-score statistics. For MR, background air
    #: dominates the histogram and drags the mean towards zero.
    zscore_foreground_only: bool = True

    #: Crop to a detected ROI around the joint before resampling.
    crop_to_roi: bool = True
    roi_margin_mm: float = 15.0

    #: Pad the result so every axis is at least this large (patch-based models
    #: need the input to be no smaller than one patch).
    min_shape: tuple[int, int, int] | None = None

    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["intensity"] = self.intensity.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreprocessingSpec:
        data = dict(payload)
        if "intensity" in data:
            data["intensity"] = IntensityNorm(data["intensity"])
        for key in ("target_spacing_mm", "clip_percentiles", "ct_window", "min_shape"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)


#: Spec for the primary 3D MR segmentation task (femur/tibia/cartilage/meniscus).
#: 0.5 mm isotropic keeps cartilage — which is 2-3 mm thick — resolvable across
#: several voxels; at 1 mm a thinned cartilage layer spans barely one voxel and
#: thickness measurements quantise badly.
MR_3D_SEGMENTATION_SPEC = PreprocessingSpec(
    name="mr_3d_seg_v1",
    orientation="LPS",
    target_spacing_mm=(0.5, 0.5, 0.5),
    intensity=IntensityNorm.ZSCORE,
    crop_to_roi=True,
    roi_margin_mm=15.0,
    min_shape=(96, 96, 96),
)

#: Spec for CT bone segmentation and implant planning. Coarser spacing is
#: acceptable because cortical bone edges are high-contrast, and the HU window is
#: preserved rather than z-scored.
CT_BONE_SPEC = PreprocessingSpec(
    name="ct_bone_v1",
    orientation="LPS",
    target_spacing_mm=(0.6, 0.6, 0.6),
    intensity=IntensityNorm.CT_WINDOW,
    ct_window=(400.0, 1800.0),
    crop_to_roi=True,
    roi_margin_mm=20.0,
    min_shape=(96, 96, 96),
)


@dataclass
class PreprocessedVolume:
    """A preprocessed volume plus everything needed to invert the process."""

    volume: Volume
    spec: PreprocessingSpec
    source_summary: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    provenance: Provenance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "source": self.source_summary,
            "result": self.volume.summary(),
            "steps": self.steps,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


# ---------------------------------------------------------------------------
# Intensity
# ---------------------------------------------------------------------------


def normalize_intensity(
    volume: Volume, spec: PreprocessingSpec = PreprocessingSpec()
) -> tuple[Volume, dict[str, Any]]:
    """Apply the spec's intensity normalisation. Label volumes pass through."""
    if volume.is_label:
        return volume, {"operation": "normalize_intensity", "skipped": "label volume"}

    method = spec.intensity
    arr = volume.array.astype(np.float32, copy=True)
    detail: dict[str, Any] = {"operation": "normalize_intensity", "method": method.value}

    if method is IntensityNorm.NONE:
        return volume.with_array(arr), detail

    if method is IntensityNorm.CT_WINDOW:
        centre, width = spec.ct_window
        lo, hi = centre - width / 2.0, centre + width / 2.0
        if volume.modality not in (Modality.CT, Modality.UNKNOWN):
            log.warning(
                "CT windowing applied to a non-CT volume; intensities are not "
                "calibrated in Hounsfield units",
                fields={"modality": volume.modality.value},
            )
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / max(hi - lo, 1e-6)
        detail.update({"window_low_hu": lo, "window_high_hu": hi})
        return volume.with_array(arr), detail

    lo_p, hi_p = spec.clip_percentiles
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise PreprocessingError("cannot normalise a volume with no finite voxels")
    lo = float(np.percentile(finite, lo_p))
    hi = float(np.percentile(finite, hi_p))
    if hi - lo < 1e-6:
        raise PreprocessingError(
            f"intensity range collapsed after percentile clipping "
            f"({lo_p}%={lo:.4f}, {hi_p}%={hi:.4f}); the image carries no contrast"
        )
    arr = np.clip(arr, lo, hi)
    detail.update({"clip_low": lo, "clip_high": hi})

    if method is IntensityNorm.PERCENTILE:
        arr = (arr - lo) / (hi - lo)
        return volume.with_array(arr), detail

    # ZSCORE
    if spec.zscore_foreground_only:
        # Otsu-free foreground estimate: everything above the midpoint between
        # the image minimum and its mean. Air in MR sits far below this.
        threshold = float(arr.min() + (arr.mean() - arr.min()) * 0.5)
        foreground = arr[arr > threshold]
        if foreground.size < max(1000, arr.size // 1000):
            foreground = arr.reshape(-1)
            detail["foreground_fallback"] = "too few foreground voxels; used all voxels"
        detail["foreground_threshold"] = threshold
    else:
        foreground = arr.reshape(-1)

    mean = float(foreground.mean())
    std = float(foreground.std())
    if std < 1e-6:
        raise PreprocessingError("intensity standard deviation is zero; cannot z-score")
    arr = (arr - mean) / std
    detail.update({"mean": mean, "std": std})
    return volume.with_array(arr), detail


# ---------------------------------------------------------------------------
# ROI detection
# ---------------------------------------------------------------------------


def detect_roi(volume: Volume, *, percentile: float = 60.0) -> NDArray[np.bool_]:
    """Coarse foreground mask used to crop away air and padding.

    This is deliberately crude — it is an ROI for cropping, not a segmentation.
    Its only job is to shrink the field of view enough that patch-based 3D
    training is tractable, without clipping anatomy.
    """
    from scipy import ndimage

    arr = volume.array.astype(np.float32, copy=False)
    if volume.modality is Modality.CT:
        # Everything denser than fat. Robust because CT is calibrated.
        mask = arr > -200.0
    else:
        threshold = float(np.percentile(arr, percentile))
        mask = arr > threshold

    if not mask.any():
        raise PreprocessingError("ROI detection found no foreground voxels")

    # Close small holes, then keep the largest component: scanner table, coils and
    # skin folds produce satellite blobs that would otherwise inflate the ROI.
    structure = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
    labelled, n = ndimage.label(mask, structure=structure)
    if n > 1:
        sizes = ndimage.sum(mask, labelled, range(1, n + 1))
        mask = labelled == (int(np.argmax(sizes)) + 1)
    return np.asarray(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Grid resampling
# ---------------------------------------------------------------------------


def resample_like(
    source: Volume,
    reference: Volume,
    *,
    order: int | None = None,
    cval: float = 0.0,
) -> Volume:
    """Resample ``source`` onto ``reference``'s exact voxel grid.

    This is how a mask predicted on a cropped/resampled volume is mapped back to
    the original acquisition grid, and how a mask from one modality is brought
    into another's grid. The mapping is derived from the two volumes' geometry,
    so it stays correct through arbitrary chains of crop/resample/reorient.
    """
    from scipy import ndimage

    interp_order = 0 if source.is_label else (3 if order is None else order)

    # reference voxel index -> mm -> source voxel index
    ref_to_src = source.image_to_voxel @ reference.voxel_to_image
    matrix = ref_to_src.matrix[:3, :3]
    offset = ref_to_src.matrix[:3, 3]

    src = source.array
    working = src if source.is_label else src.astype(np.float32, copy=False)
    out = ndimage.affine_transform(
        working,
        matrix,
        offset=offset,
        output_shape=reference.shape,
        order=interp_order,
        mode="constant",
        cval=cval,
        prefilter=interp_order > 1,
    )
    if source.is_label:
        out = out.astype(src.dtype)

    from dataclasses import replace

    return replace(
        source,
        array=np.ascontiguousarray(out),
        spacing=reference.spacing,
        origin=reference.origin,
        direction=reference.direction,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def preprocess(
    volume: Volume,
    spec: PreprocessingSpec = MR_3D_SEGMENTATION_SPEC,
    *,
    roi_mask: NDArray[np.bool_] | None = None,
) -> PreprocessedVolume:
    """Run the full preprocessing pipeline for ``spec``.

    Order matters: reorient, then crop, then resample, then normalise
    intensities. Cropping before resampling avoids interpolating voxels that are
    about to be discarded; normalising last means the statistics are computed on
    exactly the voxels the model will see.
    """
    source_summary = volume.summary()
    steps: list[dict[str, Any]] = []
    current = volume

    # 1. Canonical orientation.
    if spec.orientation:
        before = current.orientation
        current = current.reorient(spec.orientation)
        steps.append(
            {"operation": "reorient", "from": before, "to": current.orientation}
        )

    # 2. ROI crop.
    if spec.crop_to_roi and not current.is_label:
        try:
            mask = roi_mask if roi_mask is not None else detect_roi(current)
            before_shape = current.shape
            current = current.crop_to_mask(mask, margin_mm=spec.roi_margin_mm)
            steps.append(
                {
                    "operation": "crop_to_roi",
                    "from_shape": list(before_shape),
                    "to_shape": list(current.shape),
                    "margin_mm": spec.roi_margin_mm,
                }
            )
        except PreprocessingError as exc:
            # Cropping is an optimisation. Failing it must not fail the case.
            log.warning("ROI crop skipped", fields={"reason": str(exc)})
            steps.append({"operation": "crop_to_roi", "skipped": str(exc)})

    # 3. Resample.
    if spec.target_spacing_mm is not None:
        before_spacing = current.spacing.tolist()
        before_shape = current.shape
        current = current.resample(spec.target_spacing_mm)
        steps.append(
            {
                "operation": "resample",
                "from_spacing_mm": [round(s, 4) for s in before_spacing],
                "to_spacing_mm": [round(float(s), 4) for s in current.spacing],
                "from_shape": list(before_shape),
                "to_shape": list(current.shape),
                "interpolation": "nearest" if current.is_label else "cubic",
            }
        )

    # 4. Intensity normalisation.
    #
    # Before padding, deliberately: padding introduces constant voxels that are
    # not part of the acquisition, and including them in the percentile/mean/std
    # estimates shifts the normalisation by an amount that depends on how much
    # padding happened to be needed.
    current, detail = normalize_intensity(current, spec)
    steps.append(detail)

    # 5. Minimum shape padding, with a background-consistent pad value.
    if spec.min_shape is not None:
        before_shape = current.shape
        pad_value = 0.0 if current.is_label else float(np.min(current.array))
        current = current.pad_to(spec.min_shape, cval=pad_value)
        if current.shape != before_shape:
            steps.append(
                {
                    "operation": "pad",
                    "from_shape": list(before_shape),
                    "to_shape": list(current.shape),
                    "pad_value": round(pad_value, 6),
                }
            )

    if not current.is_label and current.array.dtype != np.dtype(spec.dtype):
        current = current.with_array(current.array.astype(spec.dtype))

    if not np.all(np.isfinite(current.array.astype(np.float64, copy=False))):
        raise PreprocessingError(
            "preprocessing produced non-finite voxels; refusing to pass this "
            "volume to a model"
        )

    provenance = Provenance.for_component(
        "imaging.preprocess",
        parameters={"spec": spec.to_dict(), "steps": steps},
    )
    result = PreprocessedVolume(
        volume=current.with_metadata(preprocessing_spec=spec.name),
        spec=spec,
        source_summary=source_summary,
        steps=steps,
        provenance=provenance,
    )
    log.info(
        "preprocessing complete",
        fields={
            "spec": spec.name,
            "shape": list(current.shape),
            "spacing": [round(float(s), 3) for s in current.spacing],
        },
    )
    return result


def spec_for_modality(modality: Modality) -> PreprocessingSpec:
    if modality is Modality.CT:
        return CT_BONE_SPEC
    if modality is Modality.MR:
        return MR_3D_SEGMENTATION_SPEC
    raise ValidationError(
        f"no volumetric preprocessing spec for modality {modality.value}; "
        "radiographs use the 2D pipeline"
    )
