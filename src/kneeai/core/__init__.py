"""Core domain layer.

Contains the vocabulary and invariants the rest of the platform is built on:

- :mod:`kneeai.core.anatomy` — structures, sides, modalities, landmark names
- :mod:`kneeai.core.frames` — frame-tagged coordinate transforms (PRD §71/§72)
- :mod:`kneeai.core.geometry` — planes, axes, fits, projections
- :mod:`kneeai.core.units` — quantities with units and uncertainty
- :mod:`kneeai.core.provenance` — traceability records (PRD §118)
- :mod:`kneeai.core.case_state` — case lifecycle state machine (PRD §8)
- :mod:`kneeai.core.errors` — exception hierarchy

Nothing in this package imports torch, SimpleITK, FastAPI or SQLAlchemy. The
domain model stays installable and testable without the heavy stack.
"""

from __future__ import annotations

from kneeai.core.anatomy import (
    BONE_STRUCTURES,
    CARTILAGE_STRUCTURES,
    DEFAULT_LABEL_MAP,
    MENISCUS_STRUCTURES,
    LandmarkName,
    Modality,
    Side,
    Structure,
    StructureGroup,
    StudyType,
    invert_label_map,
    label_map_for,
)
from kneeai.core.case_state import CaseState, CaseStateMachine, StateTransition
from kneeai.core.errors import (
    FrameMismatchError,
    GuardrailViolation,
    KneeAIError,
    LandmarkError,
    MeasurementError,
    PlanningError,
    SafetyViolation,
    TransformError,
    ValidationError,
)
from kneeai.core.frames import (
    NAVIGATION_CHAIN,
    Frame,
    FramedPoints,
    Transform,
    TransformGraph,
    as_points,
    rigid_from_correspondences,
    rotation_between_vectors,
    rotation_from_axis_angle,
    rotation_matrix,
)
from kneeai.core.geometry import (
    Axis,
    OrientedBox,
    Plane,
    Sphere,
    angle_between_deg,
    bounding_extent,
    extreme_point,
    fit_line,
    fit_plane,
    fit_sphere,
    principal_axes,
    project_onto_plane,
    signed_angle_in_plane_deg,
    unit,
)
from kneeai.core.provenance import ArtifactRef, Origin, Provenance, new_id, now_utc
from kneeai.core.units import Quantity, Unit, deg, mm, mm3, percent, ratio

__all__ = [
    # anatomy
    "Side",
    "Modality",
    "StudyType",
    "Structure",
    "StructureGroup",
    "LandmarkName",
    "BONE_STRUCTURES",
    "CARTILAGE_STRUCTURES",
    "MENISCUS_STRUCTURES",
    "DEFAULT_LABEL_MAP",
    "label_map_for",
    "invert_label_map",
    # frames
    "Frame",
    "Transform",
    "FramedPoints",
    "TransformGraph",
    "NAVIGATION_CHAIN",
    "as_points",
    "rotation_matrix",
    "rotation_from_axis_angle",
    "rotation_between_vectors",
    "rigid_from_correspondences",
    # geometry
    "Plane",
    "Axis",
    "Sphere",
    "OrientedBox",
    "unit",
    "angle_between_deg",
    "signed_angle_in_plane_deg",
    "project_onto_plane",
    "fit_plane",
    "fit_line",
    "fit_sphere",
    "principal_axes",
    "extreme_point",
    "bounding_extent",
    # units
    "Quantity",
    "Unit",
    "mm",
    "deg",
    "mm3",
    "percent",
    "ratio",
    # provenance
    "Provenance",
    "ArtifactRef",
    "Origin",
    "new_id",
    "now_utc",
    # case state
    "CaseState",
    "CaseStateMachine",
    "StateTransition",
    # errors
    "KneeAIError",
    "ValidationError",
    "FrameMismatchError",
    "TransformError",
    "LandmarkError",
    "MeasurementError",
    "PlanningError",
    "SafetyViolation",
    "GuardrailViolation",
]
