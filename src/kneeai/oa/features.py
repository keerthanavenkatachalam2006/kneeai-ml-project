"""Osteoarthritis features, computed from a segmentation alone.

Every feature here is derivable from the five structures the segmentation model
produces — femur, tibia, femoral cartilage, medial and lateral tibial cartilage —
and from nothing else. That constraint is deliberate and it is the whole design:
an uploaded scan arrives with no age, no sex, no BMI and no radiologist report, so
a feature that needs any of those cannot be computed at the moment the question is
actually asked. Training on them would produce a model that scores well offline
and cannot run in the product.

Two further rules follow from what osteoarthritis actually is:

**Cartilage volume is normalised by bone volume.** Absolute cartilage volume is
dominated by body size — a large healthy knee carries more cartilage than a small
arthritic one. The ratio to the underlying bone removes that and leaves the
quantity of interest.

**Absent cartilage is a measurement, not a missing value.** Full-thickness loss is
the defining feature of advanced disease, so a compartment with zero predicted
cartilage voxels is encoded as zero thickness rather than dropped. The one case
that must not be read that way is a segmentation that found *no* cartilage
anywhere, which is a model failure wearing the costume of end-stage disease;
:attr:`OaFeatures.segmentation_usable` separates the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from kneeai.core.anatomy import Structure
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger
from kneeai.measure.soft_tissue import sheet_thickness

__all__ = ["FEATURE_NAMES", "OaFeatures", "extract_features"]

log = get_logger("oa.features")

#: The feature vector, in a fixed order. The order is part of the trained model's
#: contract: coefficients are stored positionally, so reordering this silently
#: rewires the classifier.
FEATURE_NAMES: tuple[str, ...] = (
    "femoral_cartilage_thickness_mm",
    "medial_tibial_cartilage_thickness_mm",
    "lateral_tibial_cartilage_thickness_mm",
    "femoral_cartilage_volume_ratio",
    "medial_tibial_cartilage_volume_ratio",
    "lateral_tibial_cartilage_volume_ratio",
    "medial_lateral_thickness_ratio",
)

_CARTILAGE = (
    Structure.FEMORAL_CARTILAGE,
    Structure.MEDIAL_TIBIAL_CARTILAGE,
    Structure.LATERAL_TIBIAL_CARTILAGE,
)


@dataclass
class OaFeatures:
    """The feature vector plus the raw quantities it was derived from."""

    values: dict[str, float]

    #: Per-structure volumes in mm³, before normalisation. Reported so the
    #: interface can show what the ratios were computed from.
    volumes_mm3: dict[str, float] = field(default_factory=dict)

    #: Thickness estimator detail per cartilage structure.
    thickness_detail: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: Compartments with cartilage declared by the label map but no voxels found.
    full_thickness_loss: list[str] = field(default_factory=list)

    #: False when no cartilage was found anywhere, which indicates a
    #: segmentation failure rather than end-stage disease.
    segmentation_usable: bool = True

    notes: list[str] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        """The feature vector in :data:`FEATURE_NAMES` order."""
        return np.asarray([self.values[name] for name in FEATURE_NAMES], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": {k: round(float(v), 6) for k, v in self.values.items()},
            "volumes_mm3": {k: round(float(v), 1) for k, v in self.volumes_mm3.items()},
            "thickness_detail": self.thickness_detail,
            "full_thickness_loss": list(self.full_thickness_loss),
            "segmentation_usable": self.segmentation_usable,
            "notes": list(self.notes),
        }


def _thickness(mask: Volume, structure: Structure) -> tuple[float, dict[str, Any] | None]:
    """Mean cartilage thickness in mm, or 0.0 when the structure has no voxels.

    Uses :func:`kneeai.measure.soft_tissue.sheet_thickness`, the same estimator
    the measurement stage reports, so the number in the OA panel and the number in
    the measurements table are the same quantity computed the same way.
    """
    if structure not in (mask.label_map or {}):
        return float("nan"), None
    if not mask.binary_mask(structure).any():
        return 0.0, None
    try:
        stats = sheet_thickness(mask, structure)
    except Exception as exc:  # noqa: BLE001 - a thickness failure must not abort assessment
        log.warning(
            "thickness estimation failed",
            fields={"structure": structure.value, "error": str(exc)},
        )
        return float("nan"), None
    return float(stats.mean_from_volume_area), stats.to_dict()


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return float("nan")
    return float(numerator / denominator)


def extract_features(mask: Volume) -> OaFeatures:
    """Compute the OA feature vector from a label volume.

    Works on a predicted mask or a ground-truth mask; nothing here knows or cares
    which, so the same code path is used for training and for inference.
    """
    if not mask.is_label or mask.label_map is None:
        raise ValueError("extract_features requires a label Volume")

    volumes = {
        s.value: mask.structure_volume_mm3(s)
        for s in mask.label_map
        if s is not Structure.BACKGROUND
    }

    femur_volume = volumes.get(Structure.FEMUR.value, 0.0)
    tibia_volume = volumes.get(Structure.TIBIA.value, 0.0)

    thickness: dict[Structure, float] = {}
    detail: dict[str, dict[str, Any]] = {}
    absent: list[str] = []
    for structure in _CARTILAGE:
        value, stats = _thickness(mask, structure)
        thickness[structure] = value
        if stats is not None:
            detail[structure.value] = stats
        if value == 0.0:
            absent.append(structure.value)

    notes: list[str] = []
    usable = True
    if len(absent) == len(_CARTILAGE):
        usable = False
        notes.append(
            "No cartilage voxels were found in any compartment. This is far more "
            "likely to be a segmentation failure than genuine total cartilage loss, "
            "so it is reported as an unusable segmentation rather than as end-stage "
            "osteoarthritis."
        )
    elif absent:
        notes.append(
            "Full-thickness cartilage loss recorded for: "
            + ", ".join(Structure(a).display_name for a in absent)
            + ". Confirm against the slice view before relying on it — the tibial "
            "cartilages are the weakest structures this model segments."
        )

    if femur_volume <= 0 or tibia_volume <= 0:
        usable = False
        notes.append(
            "Femur or tibia was not segmented, so cartilage volume cannot be "
            "normalised for body size and the assessment cannot be computed."
        )

    medial = thickness[Structure.MEDIAL_TIBIAL_CARTILAGE]
    lateral = thickness[Structure.LATERAL_TIBIAL_CARTILAGE]

    values: dict[str, float] = {
        "femoral_cartilage_thickness_mm": thickness[Structure.FEMORAL_CARTILAGE],
        "medial_tibial_cartilage_thickness_mm": medial,
        "lateral_tibial_cartilage_thickness_mm": lateral,
        # Scaled by 1000 purely to keep the magnitudes comparable to the
        # thicknesses, which keeps the standardiser well conditioned.
        "femoral_cartilage_volume_ratio": 1000.0
        * _ratio(volumes.get(Structure.FEMORAL_CARTILAGE.value, 0.0), femur_volume),
        "medial_tibial_cartilage_volume_ratio": 1000.0
        * _ratio(volumes.get(Structure.MEDIAL_TIBIAL_CARTILAGE.value, 0.0), tibia_volume),
        "lateral_tibial_cartilage_volume_ratio": 1000.0
        * _ratio(volumes.get(Structure.LATERAL_TIBIAL_CARTILAGE.value, 0.0), tibia_volume),
        # Medial-predominant loss is the common pattern, so the asymmetry carries
        # signal the two thicknesses do not carry on their own.
        "medial_lateral_thickness_ratio": (
            _ratio(medial, lateral) if lateral > 0 else 0.0 if medial == 0.0 else float("nan")
        ),
    }

    return OaFeatures(
        values=values,
        volumes_mm3=volumes,
        thickness_detail=detail,
        full_thickness_loss=absent,
        segmentation_usable=usable,
        notes=notes,
    )


def features_from_mapping(payload: Mapping[str, Any]) -> np.ndarray:
    """Build a feature vector from a stored ``values`` mapping."""
    return np.asarray([float(payload[name]) for name in FEATURE_NAMES], dtype=np.float64)
