"""Anatomical landmark detection and the patient reference frame (PRD §14, §36).

    >>> from kneeai.landmarks import detect_landmarks_and_frame
    >>> landmarks, frame = detect_landmarks_and_frame(meshes, external={
    ...     LandmarkName.FEMORAL_HEAD_CENTER: hip_centre_from_long_leg_film,
    ... })

The geometric detector is the deterministic baseline. A learned heatmap model
(PRD §36) plugs in behind the same :class:`LandmarkSet` interface, so the
measurement, planning and simulation layers do not change when it arrives.
"""

from __future__ import annotations

from kneeai.landmarks.frame import (
    IMAGE_ANTERIOR,
    IMAGE_INFERIOR,
    IMAGE_LEFT,
    IMAGE_POSTERIOR,
    IMAGE_RIGHT,
    IMAGE_SUPERIOR,
    PatientFrame,
    build_patient_frame,
)
from kneeai.landmarks.geometric import (
    DEFAULT_CONFIG,
    DetectionConfig,
    detect_landmarks,
    detect_landmarks_and_frame,
    detect_shaft_axis,
)
from kneeai.landmarks.types import Landmark, LandmarkSet

__all__ = [
    "Landmark",
    "LandmarkSet",
    "PatientFrame",
    "build_patient_frame",
    "DetectionConfig",
    "DEFAULT_CONFIG",
    "detect_landmarks",
    "detect_landmarks_and_frame",
    "detect_shaft_axis",
    "IMAGE_LEFT",
    "IMAGE_RIGHT",
    "IMAGE_ANTERIOR",
    "IMAGE_POSTERIOR",
    "IMAGE_SUPERIOR",
    "IMAGE_INFERIOR",
]
