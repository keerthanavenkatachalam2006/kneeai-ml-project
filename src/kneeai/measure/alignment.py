"""Limb alignment measurements (PRD §13, §25).

HKA, mLDFA, MPTA and JLCA are the four numbers that determine the coronal
alignment strategy for a TKA. All four are **coronal-plane** angles, and all four
are **signed**.

Sign conventions used throughout, stated once:

- **HKA**: positive = *varus* (knee centre lateral to the hip-ankle line).
  Reported as deviation from a straight limb, so 0° is neutral. A knee reported
  as +8° is 8° varus.
- **mLDFA**: the angle, measured laterally, between the femoral mechanical axis
  and the distal femoral joint line. Typical 85-90°; below 85° is a valgus-shaped
  distal femur.
- **MPTA**: the angle, measured medially, between the tibial mechanical axis and
  the tibial joint line. Typical 85-90°; below 85° is a varus-shaped proximal
  tibia.
- **JLCA**: the angle between the femoral and tibial joint lines. Positive =
  convergence opening laterally, which is the pattern in a varus knee with medial
  soft-tissue contracture.

Why these need the hip and ankle: HKA, mLDFA and MPTA are all defined against
mechanical axes, which need the femoral head and ankle centres. A knee MR does
not contain them (PRD §14). When they are absent the alignment measurements are
marked unavailable with that reason, rather than silently substituting the shaft
axis — which would shift mLDFA by the anatomical-mechanical angle, typically
5-7°, and quietly change the planned distal femoral resection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import LandmarkName, Side, Structure
from kneeai.core.errors import MeasurementError
from kneeai.core.geometry import project_onto_plane, signed_angle_in_plane_deg, unit
from kneeai.core.units import Unit
from kneeai.landmarks.frame import PatientFrame
from kneeai.landmarks.types import LandmarkSet
from kneeai.logging import get_logger
from kneeai.measure.bone import joint_line_axis
from kneeai.measure.types import Measurement, MeasurementName, make_measurement

__all__ = ["measure_alignment", "hka_angle_deg", "ALIGNMENT_REQUIREMENTS"]

log = get_logger("measure.alignment")

#: Landmarks each alignment measurement needs, used to explain what is missing.
ALIGNMENT_REQUIREMENTS: dict[MeasurementName, tuple[LandmarkName, ...]] = {
    MeasurementName.HKA_ANGLE: (
        LandmarkName.FEMORAL_HEAD_CENTER,
        LandmarkName.FEMORAL_KNEE_CENTER,
        LandmarkName.ANKLE_CENTER,
    ),
    MeasurementName.MLDFA: (
        LandmarkName.FEMORAL_HEAD_CENTER,
        LandmarkName.FEMORAL_KNEE_CENTER,
        LandmarkName.DISTAL_MEDIAL_CONDYLE,
        LandmarkName.DISTAL_LATERAL_CONDYLE,
    ),
    MeasurementName.MPTA: (
        LandmarkName.TIBIAL_KNEE_CENTER,
        LandmarkName.ANKLE_CENTER,
        LandmarkName.MEDIAL_PLATEAU_LOW,
        LandmarkName.LATERAL_PLATEAU_LOW,
    ),
    MeasurementName.JLCA: (
        LandmarkName.DISTAL_MEDIAL_CONDYLE,
        LandmarkName.DISTAL_LATERAL_CONDYLE,
        LandmarkName.MEDIAL_PLATEAU_LOW,
        LandmarkName.LATERAL_PLATEAU_LOW,
    ),
}


def _missing(landmarks: LandmarkSet, name: MeasurementName) -> list[str]:
    return [n.value for n in ALIGNMENT_REQUIREMENTS[name] if n not in landmarks]


def hka_angle_deg(
    hip: NDArray[np.float64],
    knee: NDArray[np.float64],
    ankle: NDArray[np.float64],
    frame: PatientFrame,
) -> float:
    """Signed hip-knee-ankle deviation in the coronal plane. Positive = varus.

    Computed as the angle between the femoral mechanical axis (hip to knee) and
    the tibial mechanical axis (knee to ankle), both projected into the coronal
    plane, so a straight limb reads 0°.

    The sign comes from which side of the hip-ankle line the knee centre falls on:
    lateral is varus. That test is side-dependent, which is why the patient
    frame's :attr:`~kneeai.landmarks.frame.PatientFrame.lateral` is used rather
    than a fixed axis.
    """
    normal = frame.anterior  # coronal plane normal

    femoral = project_onto_plane(knee - hip, normal)
    tibial = project_onto_plane(ankle - knee, normal)
    if float(np.linalg.norm(femoral)) < 20.0 or float(np.linalg.norm(tibial)) < 20.0:
        raise MeasurementError(
            "the mechanical axis segments are shorter than 20 mm in the coronal "
            "plane; HKA cannot be measured meaningfully"
        )

    deviation = float(
        np.degrees(
            np.arccos(
                max(
                    -1.0,
                    min(1.0, float(np.dot(unit(femoral), unit(tibial)))),
                )
            )
        )
    )

    # Sign: project the knee centre's offset from the hip-ankle line onto the
    # lateral direction. Lateral offset = varus.
    hip_to_ankle = project_onto_plane(ankle - hip, normal)
    hip_to_knee = project_onto_plane(knee - hip, normal)
    along = unit(hip_to_ankle)
    perpendicular = hip_to_knee - float(np.dot(hip_to_knee, along)) * along
    lateral_offset = float(np.dot(perpendicular, frame.lateral))

    return deviation if lateral_offset >= 0 else -deviation


def measure_alignment(
    landmarks: LandmarkSet,
    frame: PatientFrame,
) -> tuple[list[Measurement], dict[str, str]]:
    """Compute coronal alignment measurements.

    Returns ``(measurements, unavailable)`` where ``unavailable`` maps a
    measurement name to the reason it could not be computed.
    """
    out: list[Measurement] = []
    unavailable: dict[str, str] = {}

    external_note = (
        "The femoral head and ankle centres lie outside a knee MR or CT field of "
        "view. Supply them from a long-leg alignment radiograph."
    )

    # --- HKA ---
    missing = _missing(landmarks, MeasurementName.HKA_ANGLE)
    if missing:
        unavailable[MeasurementName.HKA_ANGLE.value] = (
            f"missing landmarks: {', '.join(missing)}. {external_note}"
        )
    else:
        hip = landmarks.position(LandmarkName.FEMORAL_HEAD_CENTER)
        knee = landmarks.position(LandmarkName.FEMORAL_KNEE_CENTER)
        ankle = landmarks.position(LandmarkName.ANKLE_CENTER)
        hka = hka_angle_deg(hip, knee, ankle, frame)
        out.append(
            make_measurement(
                MeasurementName.HKA_ANGLE,
                hka,
                Unit.DEG,
                method=(
                    "coronal-plane angle between the femoral mechanical axis "
                    "(femoral head centre to knee centre) and the tibial "
                    "mechanical axis (knee centre to ankle centre), expressed as "
                    "deviation from a straight limb"
                ),
                landmarks=ALIGNMENT_REQUIREMENTS[MeasurementName.HKA_ANGLE],
                sign_convention=(
                    "positive = varus (knee centre lateral to the hip-ankle line); "
                    "negative = valgus; 0 = neutral mechanical alignment"
                ),
            )
        )
        out.append(
            make_measurement(
                MeasurementName.FEMORAL_MECHANICAL_AXIS_LENGTH,
                float(np.linalg.norm(knee - hip)),
                Unit.MM,
                method="distance from the femoral head centre to the knee centre",
                landmarks=(
                    LandmarkName.FEMORAL_HEAD_CENTER,
                    LandmarkName.FEMORAL_KNEE_CENTER,
                ),
            )
        )
        out.append(
            make_measurement(
                MeasurementName.TIBIAL_MECHANICAL_AXIS_LENGTH,
                float(np.linalg.norm(ankle - knee)),
                Unit.MM,
                method="distance from the knee centre to the ankle centre",
                landmarks=(
                    LandmarkName.FEMORAL_KNEE_CENTER,
                    LandmarkName.ANKLE_CENTER,
                ),
            )
        )

    # --- mLDFA ---
    missing = _missing(landmarks, MeasurementName.MLDFA)
    if missing:
        unavailable[MeasurementName.MLDFA.value] = (
            f"missing landmarks: {', '.join(missing)}. {external_note}"
        )
    else:
        hip = landmarks.position(LandmarkName.FEMORAL_HEAD_CENTER)
        knee = landmarks.position(LandmarkName.FEMORAL_KNEE_CENTER)
        femoral_axis = project_onto_plane(hip - knee, frame.anterior)
        joint_line = joint_line_axis(landmarks, frame, femoral=True)
        # Measured on the lateral side: the angle from the lateral limb of the
        # joint line up to the proximal mechanical axis.
        lateral_limb = -joint_line if _points_medially(joint_line, frame) else joint_line
        angle = _acute_angle_between(femoral_axis, lateral_limb)
        out.append(
            make_measurement(
                MeasurementName.MLDFA,
                angle,
                Unit.DEG,
                method=(
                    "coronal-plane angle measured on the lateral side between the "
                    "femoral mechanical axis and the distal femoral joint line "
                    "(distal condylar points)"
                ),
                landmarks=ALIGNMENT_REQUIREMENTS[MeasurementName.MLDFA],
                structures=(Structure.FEMUR,),
                sign_convention=(
                    "typical 85-90 degrees; below 85 indicates a valgus-shaped "
                    "distal femur, above 90 a varus-shaped one"
                ),
            )
        )

    # --- MPTA ---
    missing = _missing(landmarks, MeasurementName.MPTA)
    if missing:
        unavailable[MeasurementName.MPTA.value] = (
            f"missing landmarks: {', '.join(missing)}. {external_note}"
        )
    else:
        tibial_knee = landmarks.position(LandmarkName.TIBIAL_KNEE_CENTER)
        ankle = landmarks.position(LandmarkName.ANKLE_CENTER)
        tibial_axis = project_onto_plane(ankle - tibial_knee, frame.anterior)
        joint_line = joint_line_axis(landmarks, frame, femoral=False)
        medial_limb = joint_line if _points_medially(joint_line, frame) else -joint_line
        # MPTA is conventionally the angle between the joint line and the
        # *proximal* direction of the mechanical axis, measured medially.
        angle = _acute_angle_between(-tibial_axis, medial_limb)
        out.append(
            make_measurement(
                MeasurementName.MPTA,
                angle,
                Unit.DEG,
                method=(
                    "coronal-plane angle measured on the medial side between the "
                    "tibial mechanical axis and the proximal tibial joint line "
                    "(compartment low points)"
                ),
                landmarks=ALIGNMENT_REQUIREMENTS[MeasurementName.MPTA],
                structures=(Structure.TIBIA,),
                sign_convention=(
                    "typical 85-90 degrees; below 85 indicates a varus-shaped "
                    "proximal tibia"
                ),
            )
        )

    # --- JLCA ---
    missing = _missing(landmarks, MeasurementName.JLCA)
    if missing:
        unavailable[MeasurementName.JLCA.value] = (
            f"missing landmarks: {', '.join(missing)}"
        )
    else:
        femoral_line = joint_line_axis(landmarks, frame, femoral=True)
        tibial_line = joint_line_axis(landmarks, frame, femoral=False)
        # Signed in the coronal plane: positive when the joint space opens
        # laterally, the pattern seen in varus with medial contracture.
        signed = signed_angle_in_plane_deg(tibial_line, femoral_line, frame.anterior)
        # Normalise to the acute convergence angle, preserving the sign.
        magnitude = abs(signed)
        if magnitude > 90.0:
            magnitude = 180.0 - magnitude
        oriented = magnitude * (1.0 if signed >= 0 else -1.0)
        # Express relative to the operative side so "opens laterally" is
        # consistent between knees.
        oriented *= frame.side.sign if frame.side is not Side.UNKNOWN else 1.0
        out.append(
            make_measurement(
                MeasurementName.JLCA,
                oriented,
                Unit.DEG,
                method=(
                    "coronal-plane angle between the distal femoral joint line and "
                    "the proximal tibial joint line"
                ),
                landmarks=ALIGNMENT_REQUIREMENTS[MeasurementName.JLCA],
                structures=(Structure.FEMUR, Structure.TIBIA),
                sign_convention=(
                    "positive = joint space converges medially / opens laterally, "
                    "the pattern in a varus knee with medial soft-tissue "
                    "contracture; negative = opens medially"
                ),
            )
        )

    if unavailable:
        log.info(
            "some alignment measurements are unavailable",
            fields={"unavailable": sorted(unavailable)},
        )
    return out, unavailable


# ---------------------------------------------------------------------------


def _points_medially(vector: NDArray[np.float64], frame: PatientFrame) -> bool:
    return float(np.dot(vector, frame.medial)) > 0


def _acute_angle_between(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Angle in (0, 180] between two vectors, without the ambiguity of a sign.

    Used for mLDFA and MPTA, which are conventionally reported as the angle on a
    named side (lateral for mLDFA, medial for MPTA) and are therefore always
    positive and close to 90°.
    """
    cos = float(np.dot(unit(a), unit(b)))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))
