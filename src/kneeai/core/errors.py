"""Exception hierarchy.

The distinction that matters clinically is between *recoverable* problems (bad
input, missing landmark) and *safety* problems. Safety failures must never be
caught and downgraded into a warning: PRD §88 requires the system to fail safe.
"""

from __future__ import annotations

__all__ = [
    "KneeAIError",
    "ConfigurationError",
    "ValidationError",
    "DicomError",
    "DicomValidationError",
    "PreprocessingError",
    "SegmentationError",
    "ReconstructionError",
    "LandmarkError",
    "MeasurementError",
    "ImplantError",
    "PlanningError",
    "SimulationError",
    "FrameMismatchError",
    "TransformError",
    "SafetyViolation",
    "PlanNotApprovedError",
    "GuardrailViolation",
    "DatasetError",
    "LeakageError",
]


class KneeAIError(Exception):
    """Base class for all platform errors."""


# --- configuration / input -------------------------------------------------


class ConfigurationError(KneeAIError):
    """Invalid or missing configuration."""


class ValidationError(KneeAIError):
    """Input failed a validation rule."""


# --- imaging ---------------------------------------------------------------


class DicomError(KneeAIError):
    """Generic DICOM handling failure."""


class DicomValidationError(DicomError, ValidationError):
    """A study failed DICOM validation and must not enter the AI pipeline."""


class PreprocessingError(KneeAIError):
    """Preprocessing could not produce a usable volume."""


# --- AI / geometry ---------------------------------------------------------


class SegmentationError(KneeAIError):
    """Segmentation failed or produced an unusable mask."""


class ReconstructionError(KneeAIError):
    """Surface reconstruction failed or produced invalid topology."""


class LandmarkError(KneeAIError):
    """A required anatomical landmark could not be established."""


class MeasurementError(KneeAIError):
    """A clinical measurement could not be computed from available inputs."""


# --- coordinate systems ----------------------------------------------------


class TransformError(KneeAIError):
    """Invalid transform (non-invertible, non-rigid where rigid required...)."""


class FrameMismatchError(TransformError):
    """Attempted to combine geometry expressed in different coordinate frames.

    This is the error that enforces PRD §71: image, world, mesh, planning,
    tracking and robot coordinates are never mixed implicitly.
    """


# --- implants / planning / simulation --------------------------------------


class ImplantError(KneeAIError):
    """Implant catalog or geometry problem."""


class PlanningError(KneeAIError):
    """The planner could not produce a valid plan."""


class SimulationError(KneeAIError):
    """Simulation failed or detected an impossible configuration."""


# --- safety ----------------------------------------------------------------


class SafetyViolation(KneeAIError):
    """A safety constraint was violated.

    Never catch this to continue execution. The only valid responses are: abort
    the motion, surface it to the surgeon, and log it.
    """


class PlanNotApprovedError(SafetyViolation):
    """Execution was attempted with a plan that carries no surgeon approval."""


class GuardrailViolation(KneeAIError):
    """The LLM layer attempted something outside its permitted envelope.

    Examples: emitting a number absent from the structured payload, attempting
    to mutate a locked plan, or producing an actuator command (PRD §41, §116).
    """


# --- datasets --------------------------------------------------------------


class DatasetError(KneeAIError):
    """Dataset registry / manifest problem."""


class LeakageError(DatasetError):
    """The same patient appears in more than one split (PRD §63, §115)."""
