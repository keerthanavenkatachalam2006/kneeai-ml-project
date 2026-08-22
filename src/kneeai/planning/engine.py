"""The planning engine: strategy + anatomy + implant -> resection planes (PRD §25, §26).

Geometry conventions used throughout
------------------------------------
Cut plane orientations are built from **projected angles**, not from a sequence of
Euler rotations::

    n_proximal = normalise(superior + tan(coronal) * medial + tan(sagittal) * posterior)

This is deliberate. Measuring back from such a normal gives
``atan2(medial_component, superior_component) == coronal`` exactly, and likewise
for the sagittal angle — so the number planned is precisely the number the
measurement engine reads off the plan, with no ordering ambiguity. Composing two
Euler rotations instead makes the resulting projected angles depend on the order
they were applied, which is how a 3° cut ends up measuring 2.8°.

Signs, stated once:

- **tibial coronal**: positive = varus. The cut surface falls away medially, so its
  upward normal tilts medially (a surface's upward normal tilts towards downhill).
- **tibial sagittal**: positive = posterior slope. Upward normal tilts posteriorly.
- **femoral coronal**: positive = valgus. The distal cut falls away laterally, so
  its proximal normal tilts laterally.
- **femoral sagittal**: positive = component flexion. The proximal normal tilts
  posteriorly.

Gap definition
--------------
Gaps are reported as **bone gap minus construct thickness**, so 0 mm means the
planned bone resection exactly accommodates the implant, positive means loose and
negative means tight. These are *geometric* gaps from the planned cuts. They are
not soft-tissue tension: ligament behaviour is a research module (PRD §29), and
this engine does not pretend to predict it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import LandmarkName, Side, Structure
from kneeai.core.errors import PlanningError
from kneeai.core.frames import Frame, Transform, rotation_from_axis_angle
from kneeai.core.geometry import Axis, Plane, unit
from kneeai.core.provenance import Origin, Provenance
from kneeai.implants.catalog import ImplantConfiguration
from kneeai.implants.matching import ConfigurationFit
from kneeai.landmarks.frame import PatientFrame
from kneeai.landmarks.types import LandmarkSet
from kneeai.logging import get_logger
from kneeai.measure.bone import fit_articular_plane, posterior_slope_deg
from kneeai.measure.types import MeasurementName, MeasurementSet
from kneeai.planning.plan import (
    FEMORAL_CUTS,
    ComponentPlacement,
    CutName,
    PredictedOutcome,
    ResectionCut,
    SurgicalPlan,
)
from kneeai.planning.strategy import AlignmentStrategy, ResectionTargets, targets_for
from kneeai.recon.mesh import MeshSet

__all__ = [
    "PlanningInputs",
    "plan_resections",
    "cut_plane_normal",
    "projected_angles",
    "DEFAULT_CHAMFER_OFFSET_MM",
]

log = get_logger("planning.engine")

#: Offset of each chamfer plane from the corner formed by its two adjacent cuts.
#: Implant-specific in reality — it is set by the manufacturer's cut-box geometry.
#: Used only when the component does not declare its own.
DEFAULT_CHAMFER_OFFSET_MM = 7.0


@dataclass
class PlanningInputs:
    """Everything the planner needs, gathered in one place."""

    meshes: MeshSet
    landmarks: LandmarkSet
    frame: PatientFrame
    measurements: MeasurementSet
    configuration: ImplantConfiguration
    targets: ResectionTargets
    case_id: str = ""
    fit: ConfigurationFit | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.frame.side is Side.UNKNOWN:
            raise PlanningError("planning requires a known operative side")
        if self.configuration.side not in (Side.UNKNOWN, self.frame.side):
            raise PlanningError(
                f"implant configuration is for a {self.configuration.side.value} knee "
                f"but the case is {self.frame.side.value}"
            )
        if self.landmarks.frame is not self.frame.source_frame:
            raise PlanningError(
                f"landmarks are in '{self.landmarks.frame.value}' but the patient frame "
                f"was built from '{self.frame.source_frame.value}'"
            )


# ---------------------------------------------------------------------------
# Plane construction
# ---------------------------------------------------------------------------


def cut_plane_normal(
    frame: PatientFrame,
    *,
    coronal_deg: float,
    sagittal_deg: float,
    coronal_towards_medial: bool,
) -> NDArray[np.float64]:
    """Proximal-facing cut normal from projected coronal and sagittal angles.

    Args:
        coronal_towards_medial: True for the tibia, where positive coronal (varus)
            tilts the normal medially. False for the femur, where positive coronal
            (valgus) tilts it laterally.

    The returned normal points **proximally**. Callers flip it for cuts whose
    retained fragment is distal.
    """
    coronal_dir = frame.medial if coronal_towards_medial else frame.lateral
    return unit(
        frame.superior
        + math.tan(math.radians(coronal_deg)) * coronal_dir
        + math.tan(math.radians(sagittal_deg)) * frame.posterior
    )


def projected_angles(
    normal: ArrayLike,
    frame: PatientFrame,
    *,
    coronal_towards_medial: bool,
) -> tuple[float, float]:
    """Inverse of :func:`cut_plane_normal`: recover ``(coronal_deg, sagittal_deg)``."""
    n = unit(normal)
    if float(np.dot(n, frame.superior)) < 0:
        n = -n
    coronal_dir = frame.medial if coronal_towards_medial else frame.lateral
    superior_component = float(np.dot(n, frame.superior))
    coronal = math.degrees(
        math.atan2(float(np.dot(n, coronal_dir)), superior_component)
    )
    sagittal = math.degrees(
        math.atan2(float(np.dot(n, frame.posterior)), superior_component)
    )
    return coronal, sagittal


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------


def plan_resections(inputs: PlanningInputs) -> SurgicalPlan:
    """Produce a complete surgical plan.

    Every cut records the reference it was measured from and the angles that
    produced it, so a surgeon reading the plan can see not just where the saw goes
    but why.
    """
    frame = inputs.frame
    targets = inputs.targets
    landmarks = inputs.landmarks
    configuration = inputs.configuration

    plan = SurgicalPlan(
        case_id=inputs.case_id,
        side=frame.side,
        strategy=targets.strategy,
        targets=targets,
        configuration_id=configuration.configuration_id,
        configuration_label=configuration.label,
        femoral_size=configuration.femoral.size_label,
        tibial_size=configuration.tibial_baseplate.size_label,
        insert_thickness_mm=(
            configuration.tibial_insert.thickness_mm
            if configuration.tibial_insert
            else None
        ),
        label=inputs.label,
        origin=(
            Origin.SURGEON_EDITED_AI
            if landmarks.surgeon_edited()
            else Origin.DETERMINISTIC
        ),
        fit_summary=(
            {
                "combined_score": round(inputs.fit.score, 2),
                "femoral": inputs.fit.femoral_fit.to_dict(),
                "tibial": inputs.fit.tibial_fit.to_dict(),
            }
            if inputs.fit
            else {}
        ),
    )

    tibial_angles = _plan_tibial_cut(inputs, plan)
    femoral_angles = _plan_femoral_cuts(inputs, plan)
    _predict_outcome(inputs, plan, tibial_angles, femoral_angles)
    _validate(inputs, plan)

    plan.provenance = Provenance.for_component(
        "planning.engine",
        origin=plan.origin,
        parameters={
            "strategy": targets.strategy.value,
            "targets": targets.to_dict(),
            "configuration_id": configuration.configuration_id,
            "side": frame.side.value,
            "patient_frame_method": frame.method,
            "tibial_angles": tibial_angles,
            "femoral_angles": femoral_angles,
            "landmark_mean_confidence": round(landmarks.mean_confidence, 4),
            "surgeon_edited_landmarks": [
                lm.name.value for lm in landmarks.surgeon_edited()
            ],
        },
        notes=(
            "Deterministic planning. Cut orientations built from projected coronal "
            "and sagittal angles; every strategy clamp is recorded on the plan."
        ),
    )

    log.info(
        "plan generated",
        fields={
            "plan_id": plan.plan_id,
            "strategy": targets.strategy.value,
            "cuts": sorted(c.value for c in plan.cuts),
            "violations": len(plan.violations),
            "warnings": len(plan.warnings),
            "clamps": len(plan.constraint_notes),
            "residual_hka_deg": plan.predicted.residual_hka_deg,
        },
    )
    return plan


# ---------------------------------------------------------------------------
# Tibia
# ---------------------------------------------------------------------------


def _plan_tibial_cut(inputs: PlanningInputs, plan: SurgicalPlan) -> dict[str, float]:
    frame = inputs.frame
    targets = inputs.targets
    landmarks = inputs.landmarks
    measurements = inputs.measurements

    # --- coronal angle ---
    if targets.strategy.follows_native_anatomy:
        # Native tibial obliquity, expressed as varus relative to the mechanical
        # axis. MPTA is measured medially, so varus = 90 - MPTA.
        mpta = measurements.get(MeasurementName.MPTA)
        if mpta is not None:
            native_varus = 90.0 - mpta.value
            source = f"native tibial obliquity (MPTA {mpta.value:.1f}°)"
        else:
            native_varus = targets.tibial_varus_deg
            source = "strategy target (MPTA unavailable)"
            plan.warnings.append(
                "the medial proximal tibial angle could not be measured, so the "
                "tibial coronal cut falls back to the strategy target rather than "
                "the patient's native obliquity. Supply a long-leg radiograph for a "
                "kinematic plan."
            )
        coronal, note = targets.clamp("tibial_varus", native_varus)
    else:
        coronal = targets.tibial_varus_deg
        note = None
        source = f"{targets.strategy.display_name} target"

    if note:
        plan.constraint_notes.append(note)

    # --- sagittal angle ---
    if targets.match_native_slope:
        native_slope = measurements.get(MeasurementName.POSTERIOR_TIBIAL_SLOPE)
        if native_slope is not None:
            slope_value = native_slope.value
            slope_source = f"native posterior slope ({slope_value:.1f}°)"
        else:
            slope_value = targets.tibial_slope_deg
            slope_source = "strategy target (native slope unavailable)"
        sagittal, slope_note = targets.clamp("tibial_slope", slope_value)
    else:
        sagittal = targets.tibial_slope_deg
        slope_note = None
        slope_source = f"{targets.strategy.display_name} target"
    if slope_note:
        plan.constraint_notes.append(slope_note)

    # --- reference point and depth ---
    reference_point, reference_name = _tibial_reference(landmarks, frame, plan)

    proximal_normal = cut_plane_normal(
        frame,
        coronal_deg=coronal,
        sagittal_deg=sagittal,
        coronal_towards_medial=True,
    )
    depth = targets.tibial_resection_mm
    plane_point = reference_point - depth * proximal_normal

    # Retained tibia is distal to the cut, so the normal points distally.
    plane = Plane(
        plane_point,
        -proximal_normal,
        frame=frame.source_frame,
        label="tibial_resection",
    )
    plan.add_cut(
        ResectionCut(
            name=CutName.TIBIAL,
            plane=plane,
            depth_mm=depth,
            reference=(
                f"{depth:.1f} mm below the {reference_name}; coronal from {source}, "
                f"sagittal from {slope_source}"
            ),
            angles_deg={"coronal_varus": coronal, "sagittal_slope": sagittal},
        )
    )
    return {"coronal_varus": coronal, "sagittal_slope": sagittal, "depth_mm": depth}


def _tibial_reference(
    landmarks: LandmarkSet, frame: PatientFrame, plan: SurgicalPlan
) -> tuple[NDArray[np.float64], str]:
    """Reference point for tibial resection depth: the less worn compartment.

    Referencing the worn compartment would under-resect: the worn side has already
    lost height, so measuring 2 mm below it removes less bone than intended and
    leaves the joint line too proximal.
    """
    if landmarks.has(
        LandmarkName.MEDIAL_PLATEAU_LOW, LandmarkName.LATERAL_PLATEAU_LOW
    ):
        medial = landmarks.position(LandmarkName.MEDIAL_PLATEAU_LOW)
        lateral = landmarks.position(LandmarkName.LATERAL_PLATEAU_LOW)
        medial_height = float(np.dot(medial, frame.superior))
        lateral_height = float(np.dot(lateral, frame.superior))
        if medial_height >= lateral_height:
            return medial, "medial plateau (the less worn compartment)"
        return lateral, "lateral plateau (the less worn compartment)"

    plan.warnings.append(
        "compartment low points are unavailable, so the tibial resection depth is "
        "referenced from the tibial knee centre rather than the less worn "
        "compartment. Confirm the resection depth before approval."
    )
    return landmarks.position(LandmarkName.TIBIAL_KNEE_CENTER), "tibial knee centre"


# ---------------------------------------------------------------------------
# Femur
# ---------------------------------------------------------------------------


def _plan_femoral_cuts(inputs: PlanningInputs, plan: SurgicalPlan) -> dict[str, float]:
    frame = inputs.frame
    targets = inputs.targets
    landmarks = inputs.landmarks
    measurements = inputs.measurements
    femoral = inputs.configuration.femoral

    # --- coronal ---
    if targets.strategy.follows_native_anatomy:
        mldfa = measurements.get(MeasurementName.MLDFA)
        if mldfa is not None:
            # mLDFA is measured laterally; below 90° is a valgus-shaped distal femur.
            native_valgus = 90.0 - mldfa.value
            coronal_source = f"native distal femoral obliquity (mLDFA {mldfa.value:.1f}°)"
        else:
            native_valgus = targets.femoral_valgus_deg
            coronal_source = "strategy target (mLDFA unavailable)"
            plan.warnings.append(
                "the mechanical lateral distal femoral angle could not be measured, "
                "so the distal femoral coronal cut falls back to the strategy target."
            )
        coronal, note = targets.clamp("femoral_valgus", native_valgus)
    else:
        coronal = targets.femoral_valgus_deg
        note = None
        coronal_source = f"{targets.strategy.display_name} target"
    if note:
        plan.constraint_notes.append(note)

    # --- sagittal (component flexion) ---
    sagittal, flexion_note = targets.clamp(
        "femoral_flexion", targets.femoral_flexion_deg
    )
    if flexion_note:
        plan.constraint_notes.append(flexion_note)

    # --- rotation ---
    rotation, rotation_source = _femoral_rotation(inputs, plan)

    # --- component axes ---
    proximal = cut_plane_normal(
        frame,
        coronal_deg=coronal,
        sagittal_deg=sagittal,
        coronal_towards_medial=False,
    )
    ml_axis, ap_axis = _component_axes(inputs, proximal, rotation)

    # --- distal cut ---
    distal_reference, distal_name = _femoral_distal_reference(landmarks, frame, plan)
    distal_depth = targets.distal_femoral_resection_mm
    distal_point = distal_reference + distal_depth * proximal
    distal_plane = Plane(
        distal_point, proximal, frame=frame.source_frame, label="distal_femoral_resection"
    )
    plan.add_cut(
        ResectionCut(
            name=CutName.DISTAL_FEMORAL,
            plane=distal_plane,
            depth_mm=distal_depth,
            reference=(
                f"{distal_depth:.1f} mm proximal to the {distal_name}; coronal from "
                f"{coronal_source}"
            ),
            angles_deg={
                "coronal_valgus": coronal,
                "sagittal_flexion": sagittal,
                "external_rotation": rotation,
            },
        )
    )

    # --- posterior cut ---
    posterior_reference, posterior_name = _femoral_posterior_reference(
        landmarks, ap_axis, plan
    )
    posterior_depth = femoral.posterior_offset_mm
    # Retained bone is anterior to the posterior cut.
    posterior_point = posterior_reference + posterior_depth * ap_axis
    posterior_plane = Plane(
        posterior_point,
        ap_axis,
        frame=frame.source_frame,
        label="posterior_femoral_resection",
    )
    plan.add_cut(
        ResectionCut(
            name=CutName.POSTERIOR_FEMORAL,
            plane=posterior_plane,
            depth_mm=posterior_depth,
            reference=(
                f"{posterior_depth:.1f} mm anterior to the {posterior_name}, matching "
                f"the component's posterior condylar thickness; rotation from "
                f"{rotation_source}"
            ),
            angles_deg={"external_rotation": rotation},
        )
    )

    # --- anterior cut ---
    # Placed at the component's *internal* cut-box depth, not its external envelope:
    # the anterior flange sits in front of the cut, and the posterior condylar
    # section behind the posterior cut. Using the envelope directly would put the
    # anterior cut a centimetre in front of the anterior cortex.
    envelope = femoral.envelope_ap_mm or femoral.ap_mm
    internal_ap = femoral.internal_ap_mm
    anterior_point = posterior_point + internal_ap * ap_axis
    anterior_plane = Plane(
        anterior_point,
        -ap_axis,
        frame=frame.source_frame,
        label="anterior_femoral_resection",
    )
    plan.add_cut(
        ResectionCut(
            name=CutName.ANTERIOR_FEMORAL,
            plane=anterior_plane,
            depth_mm=0.0,
            reference=(
                f"{internal_ap:.1f} mm anterior to the posterior cut — the size "
                f"{femoral.size_label} component's {envelope:.1f} mm envelope less its "
                f"{femoral.posterior_offset_mm:.1f} mm posterior condylar thickness and "
                f"{femoral.anterior_flange_thickness_mm:.1f} mm anterior flange"
            ),
            angles_deg={"external_rotation": rotation},
        )
    )

    # --- chamfers ---
    chamfer_offset = float(
        femoral.metadata.get("chamfer_offset_mm", DEFAULT_CHAMFER_OFFSET_MM)
    )
    plan.add_cut(
        _chamfer_cut(
            CutName.ANTERIOR_CHAMFER,
            distal_plane,
            anterior_plane,
            chamfer_offset,
            frame.source_frame,
        )
    )
    plan.add_cut(
        _chamfer_cut(
            CutName.POSTERIOR_CHAMFER,
            distal_plane,
            posterior_plane,
            chamfer_offset,
            frame.source_frame,
        )
    )

    # --- component placements ---
    rotation_matrix_component = np.column_stack([ml_axis, ap_axis, proximal])
    plan.add_placement(
        ComponentPlacement(
            component_id=femoral.component_id,
            structure=Structure.FEMUR,
            size_label=femoral.size_label,
            transform=Transform.from_rotation_translation(
                rotation_matrix_component,
                distal_point,
                source=Frame.IMPLANT,
                target=frame.source_frame,
                label=f"femoral_placement[{femoral.component_id}]",
            ),
            parameters={
                "coronal_valgus_deg": coronal,
                "sagittal_flexion_deg": sagittal,
                "external_rotation_deg": rotation,
                "distal_resection_mm": distal_depth,
                "posterior_resection_mm": posterior_depth,
            },
        )
    )

    tibial_cut = plan.cut(CutName.TIBIAL)
    baseplate = inputs.configuration.tibial_baseplate
    tibial_proximal = -tibial_cut.plane.normal
    tibial_ml, tibial_ap = _tibial_component_axes(inputs, tibial_proximal)
    plan.add_placement(
        ComponentPlacement(
            component_id=baseplate.component_id,
            structure=Structure.TIBIA,
            size_label=baseplate.size_label,
            transform=Transform.from_rotation_translation(
                np.column_stack([tibial_ml, tibial_ap, tibial_proximal]),
                tibial_cut.plane.point,
                source=Frame.IMPLANT,
                target=frame.source_frame,
                label=f"tibial_placement[{baseplate.component_id}]",
            ),
            parameters={
                "coronal_varus_deg": tibial_cut.angles_deg.get("coronal_varus", 0.0),
                "sagittal_slope_deg": tibial_cut.angles_deg.get("sagittal_slope", 0.0),
                "resection_mm": tibial_cut.depth_mm,
            },
        )
    )

    return {
        "coronal_valgus": coronal,
        "sagittal_flexion": sagittal,
        "external_rotation": rotation,
        "distal_depth_mm": distal_depth,
        "posterior_depth_mm": posterior_depth,
        "envelope_ap_mm": envelope,
    }


def _femoral_rotation(
    inputs: PlanningInputs, plan: SurgicalPlan
) -> tuple[float, str]:
    """External rotation of the femoral component, relative to the posterior condyles."""
    targets = inputs.targets
    measurements = inputs.measurements

    if targets.reference_transepicondylar_axis:
        twist = measurements.get(MeasurementName.CONDYLAR_TWIST_ANGLE)
        if twist is not None:
            # Aligning the component to the transepicondylar axis means rotating it
            # externally from the posterior condylar axis by the measured twist.
            value = twist.value
            source = (
                f"the measured transepicondylar axis (condylar twist {value:.1f}°)"
            )
            rotation, note = targets.clamp("femoral_rotation", value)
            if note:
                plan.constraint_notes.append(note)
            return rotation, source
        plan.warnings.append(
            "the condylar twist angle could not be measured, so femoral rotation "
            "falls back to the fixed strategy target rather than the patient's own "
            "transepicondylar axis."
        )

    rotation, note = targets.clamp(
        "femoral_rotation", targets.femoral_external_rotation_deg
    )
    if note:
        plan.constraint_notes.append(note)
    return rotation, f"the fixed {targets.strategy.display_name} rotation target"


def _component_axes(
    inputs: PlanningInputs,
    proximal: NDArray[np.float64],
    rotation_deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Femoral component mediolateral and anteroposterior axes.

    Built from the posterior condylar axis rotated externally by ``rotation_deg``
    about the component's proximal axis. The posterior condylar axis is the
    intraoperative reference a surgeon actually rotates from, so planning in those
    terms means the planned number is the number set on the cutting block.
    """
    frame = inputs.frame
    landmarks = inputs.landmarks

    if landmarks.has(
        LandmarkName.POSTERIOR_MEDIAL_CONDYLE, LandmarkName.POSTERIOR_LATERAL_CONDYLE
    ):
        medial = landmarks.position(LandmarkName.POSTERIOR_MEDIAL_CONDYLE)
        lateral = landmarks.position(LandmarkName.POSTERIOR_LATERAL_CONDYLE)
        seed = medial - lateral
    else:
        seed = frame.medial

    # Orient towards the patient's right so the component frame matches the patient
    # frame's handedness.
    if float(np.dot(seed, frame.right)) < 0:
        seed = -seed
    seed = seed - float(np.dot(seed, proximal)) * proximal
    if float(np.linalg.norm(seed)) < 1e-6:
        raise PlanningError(
            "the posterior condylar axis is parallel to the femoral axis; component "
            "rotation cannot be established"
        )
    ml_axis = unit(seed)

    # External rotation: for a right knee the component rotates so its lateral side
    # moves anteriorly. The sense is side-dependent, so it is resolved from the
    # frame's own lateral direction rather than from a fixed sign.
    sense = 1.0 if frame.side is Side.RIGHT else -1.0
    ml_axis = unit(
        rotation_from_axis_angle(proximal, sense * rotation_deg) @ ml_axis
    )

    ap_axis = unit(np.cross(proximal, ml_axis))
    if float(np.dot(ap_axis, frame.anterior)) < 0:
        ap_axis = -ap_axis
        ml_axis = unit(np.cross(ap_axis, proximal))
    return ml_axis, ap_axis


def _tibial_component_axes(
    inputs: PlanningInputs, proximal: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tibial baseplate axes, rotationally referenced to the tibial tuberosity.

    The medial third of the tibial tuberosity is the standard rotational landmark
    for the tibial component. When it is unavailable the patient frame's
    anteroposterior axis is used instead, and the plan says so.
    """
    frame = inputs.frame
    landmarks = inputs.landmarks

    if landmarks.has(LandmarkName.TIBIAL_TUBEROSITY, LandmarkName.TIBIAL_KNEE_CENTER):
        centre = landmarks.position(LandmarkName.TIBIAL_KNEE_CENTER)
        tuberosity = landmarks.position(LandmarkName.TIBIAL_TUBEROSITY)
        seed = tuberosity - centre
    else:
        seed = frame.anterior

    seed = seed - float(np.dot(seed, proximal)) * proximal
    if float(np.linalg.norm(seed)) < 1e-6:
        seed = frame.anterior - float(np.dot(frame.anterior, proximal)) * proximal
    ap_axis = unit(seed)
    if float(np.dot(ap_axis, frame.anterior)) < 0:
        ap_axis = -ap_axis
    ml_axis = unit(np.cross(ap_axis, proximal))
    if float(np.dot(ml_axis, frame.right)) < 0:
        ml_axis = -ml_axis
        ap_axis = unit(np.cross(proximal, ml_axis))
    return ml_axis, ap_axis


def _femoral_distal_reference(
    landmarks: LandmarkSet, frame: PatientFrame, plan: SurgicalPlan
) -> tuple[NDArray[np.float64], str]:
    """Distal femoral reference: the less worn (more distal) condyle."""
    if landmarks.has(
        LandmarkName.DISTAL_MEDIAL_CONDYLE, LandmarkName.DISTAL_LATERAL_CONDYLE
    ):
        medial = landmarks.position(LandmarkName.DISTAL_MEDIAL_CONDYLE)
        lateral = landmarks.position(LandmarkName.DISTAL_LATERAL_CONDYLE)
        if float(np.dot(medial, frame.superior)) <= float(
            np.dot(lateral, frame.superior)
        ):
            return medial, "distal medial condyle (the less worn condyle)"
        return lateral, "distal lateral condyle (the less worn condyle)"
    raise PlanningError(
        "the distal femoral cut requires both distal condylar landmarks"
    )


def _femoral_posterior_reference(
    landmarks: LandmarkSet, ap_axis: NDArray[np.float64], plan: SurgicalPlan
) -> tuple[NDArray[np.float64], str]:
    """Posterior femoral reference: the less worn (more posterior) condyle."""
    if landmarks.has(
        LandmarkName.POSTERIOR_MEDIAL_CONDYLE, LandmarkName.POSTERIOR_LATERAL_CONDYLE
    ):
        medial = landmarks.position(LandmarkName.POSTERIOR_MEDIAL_CONDYLE)
        lateral = landmarks.position(LandmarkName.POSTERIOR_LATERAL_CONDYLE)
        # More posterior means a smaller projection onto the anterior axis.
        if float(np.dot(medial, ap_axis)) <= float(np.dot(lateral, ap_axis)):
            return medial, "posterior medial condyle (the less worn condyle)"
        return lateral, "posterior lateral condyle (the less worn condyle)"
    raise PlanningError(
        "the posterior femoral cut requires both posterior condylar landmarks"
    )


def _chamfer_cut(
    name: CutName,
    distal: Plane,
    adjacent: Plane,
    offset_mm: float,
    frame: Frame,
) -> ResectionCut:
    """Chamfer plane bisecting the distal cut and one of the anterior/posterior cuts.

    The chamfer removes the sharp corner where two cuts meet. Its normal is the
    bisector of the two retained-side normals, and it is offset from the corner into
    the bone so that it actually cuts something: a bisector plane through the corner
    line itself would remove nothing.
    """
    bisector = unit(distal.normal + adjacent.normal)

    # A point on the corner line: intersect the two planes with a third plane
    # through the distal cut's point, spanned so the solution is well conditioned.
    third_normal = unit(np.cross(distal.normal, adjacent.normal))
    matrix = np.vstack([distal.normal, adjacent.normal, third_normal])
    rhs = np.array(
        [
            distal.offset,
            adjacent.offset,
            float(np.dot(third_normal, distal.point)),
        ]
    )
    try:
        corner = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise PlanningError(
            f"{name.display_name}: the adjacent cut planes are parallel, so no corner "
            "exists to chamfer"
        ) from exc

    plane_point = corner + offset_mm * bisector
    return ResectionCut(
        name=name,
        plane=Plane(plane_point, bisector, frame=frame, label=name.value),
        depth_mm=offset_mm,
        reference=(
            f"bisects the distal and {adjacent.label.replace('_', ' ')} cuts, offset "
            f"{offset_mm:.1f} mm from their corner (implant-specific cut-box geometry)"
        ),
        notes=(
            "chamfer offset is a component property; the default is used when the "
            "catalog entry does not declare one"
        ),
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _predict_outcome(
    inputs: PlanningInputs,
    plan: SurgicalPlan,
    tibial_angles: Mapping[str, float],
    femoral_angles: Mapping[str, float],
) -> None:
    frame = inputs.frame
    landmarks = inputs.landmarks
    configuration = inputs.configuration

    tibial_varus = float(tibial_angles["coronal_varus"])
    femoral_valgus = float(femoral_angles["coronal_valgus"])

    # Residual coronal alignment. A varus tibial cut adds varus; a valgus femoral
    # cut subtracts it. Assumes the gaps are balanced, which the gap figures below
    # let the surgeon check.
    residual_hka = tibial_varus - femoral_valgus

    # Joint line: resecting more distal femur than the component replaces moves the
    # joint line proximally.
    joint_line_change = (
        femoral_angles["distal_depth_mm"] - configuration.femoral.distal_thickness_mm
    )

    construct = configuration.total_construct_thickness_mm
    tibial_plane = plan.cut(CutName.TIBIAL).plane
    distal_plane = plan.cut(CutName.DISTAL_FEMORAL).plane
    posterior_plane = plan.cut(CutName.POSTERIOR_FEMORAL).plane

    medial_probe, lateral_probe = _compartment_probes(landmarks, frame)

    extension_medial = _gap(
        tibial_plane, distal_plane, medial_probe, frame.superior, construct
    )
    extension_lateral = _gap(
        tibial_plane, distal_plane, lateral_probe, frame.superior, construct
    )

    # Flexion gap: at 90° of flexion the posterior condyles face the tibial cut, so
    # the posterior femoral cut is rotated about the flexion axis before measuring.
    # Rotating the plane is exact; approximating the flexion gap from the posterior
    # resection depth alone ignores the component's own geometry.
    flexion_medial, flexion_lateral = _flexion_gaps(
        inputs, tibial_plane, posterior_plane, medial_probe, lateral_probe, construct
    )

    plan.predicted = PredictedOutcome(
        residual_hka_deg=residual_hka,
        tibial_coronal_deg=tibial_varus,
        tibial_slope_deg=float(tibial_angles["sagittal_slope"]),
        femoral_coronal_deg=femoral_valgus,
        femoral_rotation_deg=float(femoral_angles["external_rotation"]),
        femoral_flexion_deg=float(femoral_angles["sagittal_flexion"]),
        joint_line_change_mm=joint_line_change,
        extension_gap_medial_mm=extension_medial,
        extension_gap_lateral_mm=extension_lateral,
        flexion_gap_medial_mm=flexion_medial,
        flexion_gap_lateral_mm=flexion_lateral,
        medial_resection_mm=_resection_at(tibial_plane, medial_probe, frame.superior),
        lateral_resection_mm=_resection_at(tibial_plane, lateral_probe, frame.superior),
    )


def _compartment_probes(
    landmarks: LandmarkSet, frame: PatientFrame
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Points at which the medial and lateral gaps are evaluated."""
    if landmarks.has(
        LandmarkName.MEDIAL_PLATEAU_LOW, LandmarkName.LATERAL_PLATEAU_LOW
    ):
        return (
            landmarks.position(LandmarkName.MEDIAL_PLATEAU_LOW),
            landmarks.position(LandmarkName.LATERAL_PLATEAU_LOW),
        )
    centre = landmarks.position(LandmarkName.TIBIAL_KNEE_CENTER)
    return centre + 18.0 * frame.medial, centre + 18.0 * frame.lateral


def _gap(
    lower: Plane,
    upper: Plane,
    probe: ArrayLike,
    direction: ArrayLike,
    construct_mm: float,
) -> float:
    """Space between two planes along ``direction`` at ``probe``, minus the construct.

    Zero means the planned bone resection exactly accommodates the implant;
    positive is loose, negative is tight.
    """
    d = unit(direction)
    p = np.asarray(probe, dtype=np.float64).reshape(3)
    lower_point = _ray_plane(p, d, lower)
    upper_point = _ray_plane(p, d, upper)
    bone_gap = float(np.dot(upper_point - lower_point, d))
    return bone_gap - construct_mm


def _resection_at(plane: Plane, probe: ArrayLike, direction: ArrayLike) -> float:
    """Depth of bone removed at ``probe``, measured along ``direction``.

    The tibial plane's normal points distally (into retained bone), so a probe on
    the original articular surface sits on the discarded side and its signed
    distance is negative; the magnitude is the resection depth at that point.
    """
    p = np.asarray(probe, dtype=np.float64).reshape(3)
    return abs(float(plane.signed_distance(p)))  # type: ignore[arg-type]


def _ray_plane(origin: NDArray[np.float64], direction: NDArray[np.float64], plane: Plane) -> NDArray[np.float64]:
    denominator = float(np.dot(plane.normal, direction))
    if abs(denominator) < 1e-9:
        raise PlanningError(
            f"the probe direction is parallel to the {plane.label or 'cut'} plane, so "
            "the gap at this point is undefined"
        )
    t = (plane.offset - float(np.dot(plane.normal, origin))) / denominator
    return origin + t * direction


def _flexion_gaps(
    inputs: PlanningInputs,
    tibial_plane: Plane,
    posterior_plane: Plane,
    medial_probe: NDArray[np.float64],
    lateral_probe: NDArray[np.float64],
    construct_mm: float,
) -> tuple[float | None, float | None]:
    """Medial and lateral flexion gaps, by rotating the femur to 90° of flexion."""
    frame = inputs.frame
    landmarks = inputs.landmarks

    if not landmarks.has(
        LandmarkName.MEDIAL_EPICONDYLE, LandmarkName.LATERAL_EPICONDYLE
    ):
        return None, None

    # Flexion axis: the transepicondylar axis through the knee centre.
    medial_epi = landmarks.position(LandmarkName.MEDIAL_EPICONDYLE)
    lateral_epi = landmarks.position(LandmarkName.LATERAL_EPICONDYLE)
    axis_direction = unit(medial_epi - lateral_epi)
    if float(np.dot(axis_direction, frame.right)) < 0:
        axis_direction = -axis_direction
    pivot = (medial_epi + lateral_epi) / 2.0

    # Flexion rotates the distal femur posteriorly. About an axis pointing to the
    # patient's right, that is a negative rotation by the right-hand rule (the
    # anterior direction sweeps towards superior), so the sign is verified rather
    # than assumed.
    for angle in (-90.0, 90.0):
        rotation = rotation_from_axis_angle(axis_direction, angle)
        rotated_normal = rotation @ posterior_plane.normal
        # After a correct 90° flexion the posterior cut faces proximally.
        if float(np.dot(rotated_normal, frame.superior)) > 0.9:
            break
    else:  # pragma: no cover - defensive
        return None, None

    transform = Transform.from_rotation_translation(
        rotation,
        pivot - rotation @ pivot,
        source=posterior_plane.frame,
        target=posterior_plane.frame,
        label="flexion_90",
    )
    flexed = posterior_plane.transform(transform)

    try:
        medial = _gap(tibial_plane, flexed, medial_probe, frame.superior, construct_mm)
        lateral = _gap(tibial_plane, flexed, lateral_probe, frame.superior, construct_mm)
    except PlanningError:
        return None, None
    return medial, lateral


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(inputs: PlanningInputs, plan: SurgicalPlan) -> None:
    """Check the plan against safety limits and target envelopes."""
    from kneeai.config import get_settings

    settings = get_settings()
    targets = inputs.targets
    predicted = plan.predicted

    missing = plan.missing_cuts()
    if missing:
        plan.violations.append(
            f"the plan is incomplete: missing {[c.display_name for c in missing]}"
        )

    # --- resection depth limits ---
    tibial = plan.cut(CutName.TIBIAL)
    if tibial.depth_mm > targets.max_tibial_resection_mm:
        plan.violations.append(
            f"tibial resection {tibial.depth_mm:.1f} mm exceeds the strategy limit of "
            f"{targets.max_tibial_resection_mm:.1f} mm; deeper cuts move into "
            "metaphyseal bone that may not support the baseplate"
        )
    distal = plan.cut(CutName.DISTAL_FEMORAL)
    if distal.depth_mm > targets.max_distal_femoral_resection_mm:
        plan.violations.append(
            f"distal femoral resection {distal.depth_mm:.1f} mm exceeds the strategy "
            f"limit of {targets.max_distal_femoral_resection_mm:.1f} mm"
        )
    for cut in plan.cuts.values():
        if cut.depth_mm > settings.max_resection_depth_mm:
            plan.violations.append(
                f"{cut.name.display_name} resection {cut.depth_mm:.1f} mm exceeds the "
                f"platform-wide safety ceiling of {settings.max_resection_depth_mm:.1f} mm"
            )

    # --- residual alignment ---
    if predicted.residual_hka_deg is not None:
        low, high = targets.residual_hka_envelope_deg
        if not low <= predicted.residual_hka_deg <= high:
            plan.violations.append(
                f"predicted residual alignment {predicted.residual_hka_deg:+.1f}° is "
                f"outside the {targets.strategy.display_name} envelope "
                f"[{low:+.1f}°, {high:+.1f}°]"
            )

    # --- gap balance ---
    imbalance = predicted.coronal_gap_imbalance_mm
    if imbalance is not None and abs(imbalance) > targets.max_gap_mismatch_mm:
        plan.warnings.append(
            f"predicted medial-lateral extension gap imbalance is {imbalance:+.1f} mm "
            f"(limit {targets.max_gap_mismatch_mm:.1f} mm). Soft-tissue balancing or a "
            "cut adjustment will be required; the platform does not model ligament "
            "tension."
        )
    mismatch = predicted.flexion_extension_mismatch_mm
    if mismatch is not None and abs(mismatch) > targets.max_gap_mismatch_mm:
        direction = "looser in flexion" if mismatch > 0 else "tighter in flexion"
        plan.warnings.append(
            f"predicted flexion-extension mismatch is {mismatch:+.1f} mm "
            f"({direction}). Consider femoral component size or posterior resection."
        )

    # --- anterior notching ---
    _check_anterior_notch(inputs, plan)

    # --- carry implant-fit violations onto the plan ---
    if inputs.fit is not None:
        for violation in inputs.fit.violations:
            plan.violations.append(f"implant fit — {violation}")
        for note in inputs.fit.notes:
            plan.warnings.append(f"implant fit — {note}")

    # --- landmark confidence ---
    low_confidence = inputs.landmarks.low_confidence(0.5)
    if low_confidence:
        plan.warnings.append(
            "the following landmarks were detected with low confidence and should be "
            "checked before approval: "
            + ", ".join(f"{lm.name.display_name} ({lm.confidence:.2f})" for lm in low_confidence)
        )


#: Anterior flange standoff beyond which patellofemoral overstuffing is flagged.
ANTERIOR_PROMINENCE_WARN_MM = 2.0

#: Depth by which the anterior cut may sit behind the cortex before notching is
#: called. A small step is expected and is filled by the flange; a deep one is a
#: stress riser.
ANTERIOR_NOTCH_LIMIT_MM = 2.0


def _check_anterior_notch(inputs: PlanningInputs, plan: SurgicalPlan) -> None:
    """Check the anterior femoral cut against the anterior cortex.

    Two opposite failure modes, and they need opposite signs:

    - **Notching.** The component's anterior surface sits *behind* the cortex, so the
      cut has removed cortex that the flange does not replace, leaving a step. This
      is a stress riser associated with periprosthetic fracture — a violation.
    - **Anterior prominence.** The component's anterior surface stands *proud* of the
      cortex, overstuffing the patellofemoral joint and restricting flexion — a
      warning, since it is a tension problem rather than a structural one.

    The comparison is made against the component's outer anterior surface, not
    against the cut itself: the cut is expected to sit behind the cortex by the
    flange thickness, and comparing the cut directly would flag every correctly
    sized component.
    """
    landmarks = inputs.landmarks
    if not landmarks.has(LandmarkName.ANTERIOR_FEMORAL_CORTEX):
        plan.warnings.append(
            "the anterior femoral cortex landmark is unavailable, so the plan cannot "
            "be checked for anterior notching or prominence. Verify on the 3D model."
        )
        return

    femoral = inputs.configuration.femoral
    cortex = landmarks.position(LandmarkName.ANTERIOR_FEMORAL_CORTEX)
    anterior_cut = plan.cut(CutName.ANTERIOR_FEMORAL)

    # The cut normal points posteriorly (into retained bone), so the outward normal
    # of the component's anterior surface is its negation.
    anterior_direction = -anterior_cut.plane.normal
    flange_surface = anterior_cut.plane.point + (
        femoral.anterior_flange_thickness_mm * anterior_direction
    )

    # Positive: the component's anterior surface stands proud of the cortex.
    # Negative: it sits behind the cortex, leaving an unfilled step.
    prominence = float(np.dot(flange_surface - cortex, anterior_direction))

    if prominence < -ANTERIOR_NOTCH_LIMIT_MM:
        plan.violations.append(
            f"the femoral component's anterior surface sits {abs(prominence):.1f} mm "
            "behind the anterior cortex, leaving an unfilled step in the cortex "
            "(anterior notch). This is a stress riser; consider a larger femoral "
            "component or more component flexion."
        )
    elif prominence > ANTERIOR_PROMINENCE_WARN_MM:
        plan.warnings.append(
            f"the femoral component's anterior surface stands {prominence:.1f} mm "
            "proud of the anterior cortex, which overstuffs the patellofemoral joint "
            "and may restrict flexion. Consider a smaller component or more flexion."
        )
    else:
        log.debug(
            "anterior flange sits flush with the cortex",
            fields={"prominence_mm": round(prominence, 2)},
        )
