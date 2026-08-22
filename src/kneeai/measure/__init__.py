"""Clinical measurement engine (PRD §13, §17, §18, §86).

    >>> from kneeai.measure import measure_all, MeasurementName
    >>> measurements = measure_all(mask, meshes, landmarks, frame)
    >>> measurements.value(MeasurementName.FEMORAL_AP_SIZE)
    >>> print(measurements.summary())

Every measurement records the landmarks and structures it depends on, so
:meth:`~kneeai.measure.types.MeasurementSet.depending_on` answers "what does this
landmark correction invalidate?" without guesswork.
"""

from __future__ import annotations

from kneeai.measure.alignment import ALIGNMENT_REQUIREMENTS, hka_angle_deg, measure_alignment
from kneeai.measure.bone import (
    distal_femoral_region,
    fit_articular_plane,
    joint_line_axis,
    measure_femur,
    measure_tibia,
    plateau_region,
    posterior_slope_deg,
)
from kneeai.measure.engine import measure_all, recompute_affected
from kneeai.measure.soft_tissue import (
    ThicknessStats,
    measure_cartilage,
    measure_meniscus,
    sheet_thickness,
)
from kneeai.measure.types import (
    REFERENCE_RANGES,
    Measurement,
    MeasurementName,
    MeasurementSet,
    make_measurement,
)

__all__ = [
    # entry point
    "measure_all",
    "recompute_affected",
    # types
    "Measurement",
    "MeasurementName",
    "MeasurementSet",
    "make_measurement",
    "REFERENCE_RANGES",
    # bone
    "measure_femur",
    "measure_tibia",
    "distal_femoral_region",
    "plateau_region",
    "fit_articular_plane",
    "posterior_slope_deg",
    "joint_line_axis",
    # alignment
    "measure_alignment",
    "hka_angle_deg",
    "ALIGNMENT_REQUIREMENTS",
    # soft tissue
    "measure_cartilage",
    "measure_meniscus",
    "sheet_thickness",
    "ThicknessStats",
]
