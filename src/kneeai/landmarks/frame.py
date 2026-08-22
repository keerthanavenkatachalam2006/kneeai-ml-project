"""The patient anatomical frame (PRD §71).

Frame definition
----------------
The PATIENT frame is right-handed with::

    +X -> the patient's RIGHT
    +Y -> ANTERIOR
    +Z -> SUPERIOR (proximal)

Origin at the knee centre.

Why patient-right rather than "lateral": a frame defined as
(lateral, anterior, superior) is right-handed for one knee and left-handed for the
other, which quietly breaks every rotation and cross product. So the frame stays
side-independent and side-dependent directions are exposed explicitly as
:attr:`PatientFrame.lateral` and :attr:`PatientFrame.medial`, resolved from
:class:`~kneeai.core.anatomy.Side`.

The IMAGE frame is DICOM patient space, where +x is the patient's left,
+y posterior and +z superior — by definition of the standard, independent of how
the patient was positioned in the scanner. Those directions are therefore known
exactly and are used as the seed; bone geometry then refines them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import LandmarkName, Side
from kneeai.core.errors import LandmarkError, ValidationError
from kneeai.core.frames import Frame, Transform, rotation_matrix
from kneeai.core.geometry import Axis, Plane, angle_between_deg, unit
from kneeai.landmarks.types import LandmarkSet

__all__ = [
    "PatientFrame",
    "build_patient_frame",
    "IMAGE_RIGHT",
    "IMAGE_ANTERIOR",
    "IMAGE_SUPERIOR",
]

#: Anatomical directions in the IMAGE (DICOM LPS) frame. Fixed by the standard.
IMAGE_LEFT = np.array([1.0, 0.0, 0.0])
IMAGE_RIGHT = np.array([-1.0, 0.0, 0.0])
IMAGE_POSTERIOR = np.array([0.0, 1.0, 0.0])
IMAGE_ANTERIOR = np.array([0.0, -1.0, 0.0])
IMAGE_SUPERIOR = np.array([0.0, 0.0, 1.0])
IMAGE_INFERIOR = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class PatientFrame:
    """An anatomical reference frame derived from landmarks."""

    origin: NDArray[np.float64]
    right: NDArray[np.float64]
    anterior: NDArray[np.float64]
    superior: NDArray[np.float64]
    side: Side
    source_frame: Frame = Frame.IMAGE

    #: Which construction was used, e.g. "mechanical_axis+TEA" or "shaft+TEA".
    #: The two are not interchangeable: a frame built on the femoral shaft axis
    #: rather than the mechanical axis differs by the anatomical-mechanical angle
    #: (typically 5-7°), so every angle measured in it is offset by that amount.
    method: str = ""

    #: Angle between the two axes used to build the frame. Near-parallel axes make
    #: the derived third axis unstable.
    axis_angle_deg: float = 90.0

    def __post_init__(self) -> None:
        for name in ("origin", "right", "anterior", "superior"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.size != 3:
                raise ValidationError(f"PatientFrame.{name} must be a 3-vector")
            object.__setattr__(self, name, value)
        for name in ("right", "anterior", "superior"):
            object.__setattr__(self, name, unit(getattr(self, name)))

        R = self.rotation
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValidationError("patient frame axes are not orthonormal")
        if float(np.linalg.det(R)) < 0:
            raise ValidationError(
                "patient frame is left-handed; axis construction is wrong"
            )

    # --- basis -------------------------------------------------------------
    @property
    def rotation(self) -> NDArray[np.float64]:
        """Columns are the patient axes expressed in the source frame."""
        return np.column_stack([self.right, self.anterior, self.superior])

    @property
    def inferior(self) -> NDArray[np.float64]:
        return -self.superior

    @property
    def posterior(self) -> NDArray[np.float64]:
        return -self.anterior

    @property
    def left(self) -> NDArray[np.float64]:
        return -self.right

    @property
    def lateral(self) -> NDArray[np.float64]:
        """Away from the midline, for this knee."""
        if self.side is Side.RIGHT:
            return self.right
        if self.side is Side.LEFT:
            return self.left
        raise LandmarkError(
            "lateral direction is undefined until the operative side is known"
        )

    @property
    def medial(self) -> NDArray[np.float64]:
        return -self.lateral

    # --- transforms --------------------------------------------------------
    @property
    def transform(self) -> Transform:
        """IMAGE (or source) -> PATIENT."""
        return Transform.from_basis(
            self.origin,
            self.rotation,
            source=self.source_frame,
            target=Frame.PATIENT,
            label=f"patient_frame[{self.method}]",
        )

    @property
    def inverse_transform(self) -> Transform:
        return self.transform.inverse()

    def to_patient(self, points: ArrayLike) -> NDArray[np.float64]:
        return self.transform.apply(points)

    def to_source(self, points: ArrayLike) -> NDArray[np.float64]:
        return self.inverse_transform.apply(points)

    # --- anatomical planes -------------------------------------------------
    def coronal_plane(self, at: ArrayLike | None = None) -> Plane:
        """Frontal plane. Varus/valgus angles are measured in it."""
        return Plane(
            self.origin if at is None else np.asarray(at, dtype=np.float64),
            self.anterior,
            frame=self.source_frame,
            label="coronal",
        )

    def sagittal_plane(self, at: ArrayLike | None = None) -> Plane:
        """Plane in which flexion, extension and tibial slope are measured."""
        return Plane(
            self.origin if at is None else np.asarray(at, dtype=np.float64),
            self.right,
            frame=self.source_frame,
            label="sagittal",
        )

    def axial_plane(self, at: ArrayLike | None = None) -> Plane:
        """Transverse plane. Component rotation is measured in it."""
        return Plane(
            self.origin if at is None else np.asarray(at, dtype=np.float64),
            self.superior,
            frame=self.source_frame,
            label="axial",
        )

    def mediolateral_axis(self) -> Axis:
        return Axis(self.origin, self.right, frame=self.source_frame, label="mediolateral")

    def anteroposterior_axis(self) -> Axis:
        return Axis(self.origin, self.anterior, frame=self.source_frame, label="anteroposterior")

    def superoinferior_axis(self) -> Axis:
        return Axis(self.origin, self.superior, frame=self.source_frame, label="superoinferior")

    # --- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": [round(float(v), 4) for v in self.origin],
            "right": [round(float(v), 6) for v in self.right],
            "anterior": [round(float(v), 6) for v in self.anterior],
            "superior": [round(float(v), 6) for v in self.superior],
            "side": self.side.value,
            "source_frame": self.source_frame.value,
            "method": self.method,
            "axis_angle_deg": round(self.axis_angle_deg, 3),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PatientFrame:
        return cls(
            origin=np.asarray(payload["origin"], dtype=np.float64),
            right=np.asarray(payload["right"], dtype=np.float64),
            anterior=np.asarray(payload["anterior"], dtype=np.float64),
            superior=np.asarray(payload["superior"], dtype=np.float64),
            side=Side(payload["side"]),
            source_frame=Frame(payload.get("source_frame", "image")),
            method=str(payload.get("method", "")),
            axis_angle_deg=float(payload.get("axis_angle_deg", 90.0)),
        )


# ---------------------------------------------------------------------------


def build_patient_frame(
    landmarks: LandmarkSet,
    *,
    prefer_mechanical_axis: bool = True,
    min_axis_angle_deg: float = 45.0,
) -> PatientFrame:
    """Construct the patient frame from a landmark set.

    Longitudinal axis, in order of preference:

    1. **Femoral mechanical axis** (hip centre -> femoral knee centre). This is
       the reference every alignment measurement in TKA is defined against, so it
       is used whenever the hip centre is available.
    2. **Femoral shaft axis**, when the hip is outside the field of view (the
       usual case for a knee MR). The frame then differs from the mechanical frame
       by the anatomical-mechanical angle, and :attr:`PatientFrame.method` records
       that so downstream angles can be reported with the right reference.

    Transverse axis: the surgical transepicondylar axis (medial to lateral
    epicondyle), orthogonalised against the longitudinal axis.

    Args:
        min_axis_angle_deg: reject the construction if the two source axes are
            closer than this. A near-parallel pair makes the derived anterior
            axis numerically unstable, and an unstable AP axis silently corrupts
            component rotation.
    """
    if landmarks.side is Side.UNKNOWN:
        raise LandmarkError(
            "the operative side must be known before building a patient frame; "
            "medial and lateral are otherwise undefined"
        )
    if landmarks.frame is not Frame.IMAGE:
        raise LandmarkError(
            f"patient frame construction expects landmarks in the IMAGE frame, got "
            f"'{landmarks.frame.value}'"
        )

    knee_centre = landmarks.position(LandmarkName.FEMORAL_KNEE_CENTER)

    # --- longitudinal axis ---
    if prefer_mechanical_axis and landmarks.has(LandmarkName.FEMORAL_HEAD_CENTER):
        hip = landmarks.position(LandmarkName.FEMORAL_HEAD_CENTER)
        superior_seed = hip - knee_centre
        axis_source = "mechanical_axis"
    elif landmarks.has(LandmarkName.FEMORAL_CANAL_ENTRY):
        superior_seed = landmarks.position(LandmarkName.FEMORAL_CANAL_ENTRY) - knee_centre
        axis_source = "shaft_axis"
    else:
        raise LandmarkError(
            "cannot establish a longitudinal axis: neither the femoral head centre "
            "(mechanical axis) nor a femoral canal entry point (shaft axis) is "
            "available. Supply the hip centre from a long-leg radiograph, or "
            "detect the shaft axis from the femur mesh."
        )

    superior = unit(superior_seed)
    # Sanity: the derived superior axis must actually point proximally. A sign
    # error here inverts every subsequent angle.
    if float(np.dot(superior, IMAGE_SUPERIOR)) < 0:
        raise LandmarkError(
            f"the derived longitudinal axis points distally "
            f"(dot with image superior = {float(np.dot(superior, IMAGE_SUPERIOR)):.3f}). "
            "Check that the femoral head / canal landmark is proximal to the knee centre."
        )

    # --- transverse axis: surgical transepicondylar axis ---
    medial_epi = landmarks.position(LandmarkName.MEDIAL_EPICONDYLE)
    lateral_epi = landmarks.position(LandmarkName.LATERAL_EPICONDYLE)
    tea = medial_epi - lateral_epi

    axis_angle = angle_between_deg(tea, superior)
    axis_angle = min(axis_angle, 180.0 - axis_angle)
    if axis_angle < min_axis_angle_deg:
        raise LandmarkError(
            f"the transepicondylar axis is only {axis_angle:.1f}° from the "
            f"longitudinal axis (minimum {min_axis_angle_deg}°). One of the "
            "epicondylar landmarks is almost certainly misplaced; the derived "
            "anteroposterior axis would be unreliable."
        )

    # Orient the transverse axis to the patient's right.
    right_seed = tea if float(np.dot(tea, IMAGE_RIGHT)) > 0 else -tea

    # Superior is primary (it is the clinical reference), transverse is
    # orthogonalised against it.
    basis = rotation_matrix(z=superior, x=right_seed)
    right = basis[:, 0]
    anterior = basis[:, 1]
    superior_final = basis[:, 2]

    # Verify the derived anterior axis actually points anteriorly. If it does not,
    # the epicondylar landmarks are swapped medial-for-lateral.
    if float(np.dot(anterior, IMAGE_ANTERIOR)) < 0:
        raise LandmarkError(
            "the derived anteroposterior axis points posteriorly, which means the "
            "medial and lateral epicondyle landmarks are swapped for a "
            f"{landmarks.side.value} knee."
        )

    return PatientFrame(
        origin=knee_centre,
        right=right,
        anterior=anterior,
        superior=superior_final,
        side=landmarks.side,
        source_frame=Frame.IMAGE,
        method=f"{axis_source}+TEA",
        axis_angle_deg=axis_angle,
    )
