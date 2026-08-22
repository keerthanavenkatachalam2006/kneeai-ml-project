"""Level 2 kinematic simulation: flexion sweep, range of motion, interference (PRD §73).

Scope, stated plainly
---------------------
This is a **rigid-body geometric** flexion sweep. The femoral construct is rotated
through a range of flexion angles about a defined flexion axis and, at each angle,
the geometry is checked for interference and the joint gap is measured.

It is **not** a kinematic model of a knee. It does not model ligament tension,
articular contact mechanics, patellar tracking, or the coupled rotation and
translation a real knee exhibits through flexion. Those need soft tissue and contact
mechanics, which PRD §29 assigns to a research module. Reporting a "range of motion"
from geometry alone would overstate what has been computed, so the output is named
for what it is: the flexion range over which the planned geometry does not
interfere.

Flexion model
-------------
Rotation is about the **transepicondylar axis**, held fixed. A real knee's flexion
axis migrates posteriorly through flexion and the femur rolls back on the tibia; a
fixed axis is a first-order approximation that is adequate for detecting gross
interference and is honest about being one. The limitation is recorded in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import LandmarkName, Side, Structure
from kneeai.core.errors import SimulationError
from kneeai.core.frames import Frame, Transform, rotation_from_axis_angle
from kneeai.core.geometry import Axis, unit
from kneeai.core.provenance import Provenance
from kneeai.landmarks.frame import PatientFrame
from kneeai.landmarks.types import LandmarkSet
from kneeai.logging import get_logger
from kneeai.recon.mesh import AnatomicalMesh
from kneeai.simulation.geometric import (
    COLLISION_TOLERANCE_MM,
    Finding,
    Severity,
    SimulationReport,
)

__all__ = [
    "FlexionSample",
    "KinematicReport",
    "KinematicConfig",
    "flexion_axis",
    "simulate_flexion",
]

log = get_logger("simulation.kinematic")


@dataclass(frozen=True)
class KinematicConfig:
    """Flexion sweep parameters."""

    #: Flexion angles to evaluate, in degrees. 0 is full extension.
    angles_deg: tuple[float, ...] = tuple(float(a) for a in range(0, 145, 5))

    #: Hyperextension angles to check, as negative flexion.
    include_hyperextension: bool = True
    hyperextension_deg: float = -5.0

    #: Interpenetration above this counts as interference.
    interference_tolerance_mm: float = COLLISION_TOLERANCE_MM

    #: Sample this many vertices per mesh for the sweep. Full meshes have tens of
    #: thousands of vertices and the sweep evaluates dozens of poses; subsampling
    #: keeps it interactive. Vertices are taken on a fixed stride so the result is
    #: deterministic.
    max_vertices_per_mesh: int = 1500

    #: Margin added to the tibial construct's bounding box when pre-filtering
    #: femoral points. Points outside it cannot be within interference range, and
    #: skipping them removes most of the femoral shaft from the distance queries —
    #: which is where nearly all the cost is.
    broad_phase_margin_mm: float = 6.0

    def all_angles(self) -> tuple[float, ...]:
        angles = list(self.angles_deg)
        if self.include_hyperextension:
            angles.insert(0, self.hyperextension_deg)
        return tuple(sorted(set(angles)))


@dataclass(frozen=True)
class FlexionSample:
    """Geometry at one flexion angle."""

    angle_deg: float

    #: Maximum interpenetration between the femoral and tibial constructs.
    interpenetration_mm: float

    #: Closest approach between the two constructs. Zero when they touch or overlap.
    clearance_mm: float

    #: Closest approach between the femoral and tibial constructs within each
    #: compartment. Named "clearance" rather than "gap" because it is a minimum
    #: separation, not the vertical joint space a spacer block would measure.
    medial_clearance_mm: float | None = None
    lateral_clearance_mm: float | None = None

    @property
    def interferes(self) -> bool:
        return self.interpenetration_mm > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_deg": round(self.angle_deg, 1),
            "interpenetration_mm": round(self.interpenetration_mm, 3),
            "clearance_mm": round(self.clearance_mm, 3),
            "medial_clearance_mm": (
                round(self.medial_clearance_mm, 3)
                if self.medial_clearance_mm is not None
                else None
            ),
            "lateral_clearance_mm": (
                round(self.lateral_clearance_mm, 3)
                if self.lateral_clearance_mm is not None
                else None
            ),
            "interferes": self.interferes,
        }


@dataclass
class KinematicReport:
    """Result of a flexion sweep."""

    plan_id: str
    side: Side
    samples: list[FlexionSample] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    flexion_axis_label: str = ""
    provenance: Provenance | None = None

    #: Stated limitations, carried into the clinical report so a reader is not left
    #: to infer what the simulation did and did not model.
    limitations: list[str] = field(default_factory=list)

    def add(
        self, code: str, severity: Severity, message: str, **detail: Any
    ) -> KinematicReport:
        self.findings.append(
            Finding(code=code, severity=severity, message=message, detail=detail)
        )
        return self

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.FAILURE]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def interference_free_range_deg(self) -> tuple[float, float] | None:
        """Contiguous flexion range containing 0° in which nothing interferes.

        Reported as a range around extension rather than as a maximum, because a
        construct that interferes at 10° and again at 120° has two separate problems
        and a single "maximum flexion" number would hide the first.
        """
        clear = [s for s in self.samples if not s.interferes]
        if not clear:
            return None
        ordered = sorted(self.samples, key=lambda s: s.angle_deg)
        zero_index = min(
            range(len(ordered)), key=lambda i: abs(ordered[i].angle_deg)
        )
        if ordered[zero_index].interferes:
            return None
        low = high = zero_index
        while low > 0 and not ordered[low - 1].interferes:
            low -= 1
        while high < len(ordered) - 1 and not ordered[high + 1].interferes:
            high += 1
        return ordered[low].angle_deg, ordered[high].angle_deg

    @property
    def first_interference_deg(self) -> float | None:
        interfering = [s.angle_deg for s in self.samples if s.interferes]
        return min(interfering, key=abs) if interfering else None

    def to_dict(self) -> dict[str, Any]:
        clear_range = self.interference_free_range_deg
        return {
            "plan_id": self.plan_id,
            "side": self.side.value,
            "passed": self.passed,
            "flexion_axis": self.flexion_axis_label,
            "interference_free_range_deg": (
                [round(v, 1) for v in clear_range] if clear_range else None
            ),
            "first_interference_deg": self.first_interference_deg,
            "samples": [s.to_dict() for s in self.samples],
            "findings": [f.to_dict() for f in self.findings],
            "limitations": list(self.limitations),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    def summary(self) -> str:
        clear_range = self.interference_free_range_deg
        lines = [
            f"Flexion sweep for plan {self.plan_id} ({self.side.value} knee): "
            + ("no interference" if self.passed else f"{len(self.failures)} finding(s)")
        ]
        if clear_range:
            lines.append(
                f"  interference-free flexion range: {clear_range[0]:.0f}° to "
                f"{clear_range[1]:.0f}°"
            )
        else:
            lines.append("  the construct interferes at every angle evaluated")
        if self.first_interference_deg is not None:
            lines.append(
                f"  first interference at {self.first_interference_deg:.0f}° of flexion"
            )
        lines.append(f"  flexion axis: {self.flexion_axis_label}")
        if self.samples:
            lines.append(
                f"  {'angle':>7}{'penetration':>13}{'clearance':>12}"
                f"{'medial':>10}{'lateral':>10}"
            )
            for sample in self.samples:
                medial = (
                    f"{sample.medial_clearance_mm:.2f}"
                    if sample.medial_clearance_mm is not None
                    else "-"
                )
                lateral = (
                    f"{sample.lateral_clearance_mm:.2f}"
                    if sample.lateral_clearance_mm is not None
                    else "-"
                )
                lines.append(
                    f"  {sample.angle_deg:>6.0f}°{sample.interpenetration_mm:>13.2f}"
                    f"{sample.clearance_mm:>12.2f}{medial:>10}{lateral:>10}"
                )
        for finding in self.findings:
            if finding.severity is not Severity.INFO:
                lines.append(f"  {finding}")
        if self.limitations:
            lines.append("  limitations of this simulation:")
            lines.extend(f"    - {limit}" for limit in self.limitations)
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def flexion_axis(landmarks: LandmarkSet, frame: PatientFrame) -> tuple[Axis, str]:
    """Flexion axis: the transepicondylar axis through the epicondylar midpoint."""
    if landmarks.has(LandmarkName.MEDIAL_EPICONDYLE, LandmarkName.LATERAL_EPICONDYLE):
        medial = landmarks.position(LandmarkName.MEDIAL_EPICONDYLE)
        lateral = landmarks.position(LandmarkName.LATERAL_EPICONDYLE)
        direction = unit(medial - lateral)
        if float(np.dot(direction, frame.right)) < 0:
            direction = -direction
        pivot = (medial + lateral) / 2.0
        return (
            Axis(pivot, direction, frame=landmarks.frame, label="transepicondylar"),
            "transepicondylar axis (fixed)",
        )

    raise SimulationError(
        "a flexion sweep requires both epicondylar landmarks to define the flexion "
        "axis"
    )


def _flexion_transform(axis: Axis, angle_deg: float, frame: PatientFrame) -> Transform:
    """Rigid transform flexing the femur by ``angle_deg`` about ``axis``.

    The sign is resolved by checking which direction moves the distal femur
    posteriorly, rather than assumed from a right-hand-rule convention that would be
    side-dependent and easy to get backwards.
    """
    probe = axis.origin - 40.0 * frame.superior  # a point on the distal femur
    for sign in (1.0, -1.0):
        rotation = rotation_from_axis_angle(axis.direction, sign * angle_deg)
        moved = rotation @ (probe - axis.origin) + axis.origin
        displacement = moved - probe
        if angle_deg == 0.0 or float(np.dot(displacement, frame.posterior)) >= 0:
            return Transform.from_rotation_translation(
                rotation,
                axis.origin - rotation @ axis.origin,
                source=axis.frame,
                target=axis.frame,
                label=f"flexion_{angle_deg:+.0f}",
            )
    raise SimulationError(  # pragma: no cover - defensive
        "could not determine the flexion direction from the frame"
    )


def _subsample(mesh: AnatomicalMesh, limit: int) -> NDArray[np.float64]:
    vertices = mesh.vertices
    if vertices.shape[0] <= limit:
        return vertices
    stride = max(1, vertices.shape[0] // limit)
    return vertices[::stride]


def simulate_flexion(
    geometric: SimulationReport,
    landmarks: LandmarkSet,
    frame: PatientFrame,
    *,
    config: KinematicConfig | None = None,
) -> KinematicReport:
    """Sweep the femoral construct through flexion, checking for interference.

    Requires a :class:`~kneeai.simulation.geometric.SimulationReport` that produced
    resected bone and placed component solids.
    """
    config = config or KinematicConfig()
    report = KinematicReport(plan_id=geometric.plan_id, side=geometric.side)
    report.limitations = [
        "Rigid-body geometry only: ligament tension, articular contact mechanics and "
        "patellar tracking are not modelled.",
        "The flexion axis is held fixed at the transepicondylar axis. A real knee's "
        "flexion axis migrates posteriorly and the femur rolls back on the tibia, so "
        "contact positions in deep flexion are approximate.",
        "The reported range is the flexion range over which the planned geometry does "
        "not interfere. It is not a prediction of the range of motion the patient "
        "will achieve.",
    ]
    report.limitations.append(
        "Component surfaces are procedurally generated reference geometry of the right "
        "shape and scale, not licensed manufacturer surfaces. Condylar radii are "
        "approximations, so contact positions are indicative rather than exact."
    )
    if geometric.approximated_components:
        report.limitations.append(
            "These components were reduced to extruded footprint prisms, which have no "
            "articular curvature at all, so deep-flexion contact is not represented "
            "for them: " + ", ".join(geometric.approximated_components)
        )

    try:
        axis, axis_label = flexion_axis(landmarks, frame)
    except SimulationError as exc:
        report.add("no_flexion_axis", Severity.FAILURE, str(exc))
        return report
    report.flexion_axis_label = axis_label

    # --- collect the moving and fixed geometry ---
    femoral_parts = [
        mesh
        for mesh in geometric.component_meshes.values()
        if mesh.structure is Structure.FEMUR
    ]
    if Structure.FEMUR in geometric.resected_meshes:
        femoral_parts.append(geometric.resected_meshes[Structure.FEMUR])

    tibial_parts = [
        mesh
        for mesh in geometric.component_meshes.values()
        if mesh.structure is Structure.TIBIA
    ]
    if Structure.TIBIA in geometric.resected_meshes:
        tibial_parts.append(geometric.resected_meshes[Structure.TIBIA])

    if not femoral_parts or not tibial_parts:
        report.add(
            "insufficient_geometry",
            Severity.FAILURE,
            "the flexion sweep needs both a femoral and a tibial construct from the "
            "geometric simulation.",
            femoral_parts=len(femoral_parts),
            tibial_parts=len(tibial_parts),
        )
        return report

    # The tibia is fixed, so its solids are the reference for distance queries.
    tibial_solids = [m for m in tibial_parts if m.is_watertight]
    if not tibial_solids:
        report.add(
            "tibial_construct_not_solid",
            Severity.WARNING,
            "no watertight tibial solid is available, so interpenetration depth cannot "
            "be measured; only clearance is reported.",
        )

    # All femoral surface points, subsampled once and pooled: the sweep only ever
    # needs the union, and pooling avoids repeating the per-mesh overhead at every
    # angle.
    femoral_points = np.vstack(
        [_subsample(mesh, config.max_vertices_per_mesh) for mesh in femoral_parts]
    )

    # Broad phase: the axis-aligned box the tibial construct occupies, expanded.
    # Anything outside cannot interfere, and the femoral shaft is entirely outside.
    tibial_bounds = np.vstack([solid.bounds for solid in tibial_solids])
    box_min = tibial_bounds.min(axis=0) - config.broad_phase_margin_mm
    box_max = tibial_bounds.max(axis=0) + config.broad_phase_margin_mm

    # Proximity queries are built once per solid rather than per call: constructing
    # one rebuilds trimesh's internal acceleration structures.
    import trimesh

    queries = [
        trimesh.proximity.ProximityQuery(solid.mesh) for solid in tibial_solids
    ]

    medial_probe, lateral_probe = _gap_probes(landmarks, frame)
    total_queried = 0

    for angle in config.all_angles():
        transform = _flexion_transform(axis, angle, frame)
        moved = transform.apply(femoral_points)

        in_box = np.all((moved >= box_min) & (moved <= box_max), axis=1)
        near = moved[in_box]
        total_queried += int(near.shape[0])

        penetration = 0.0
        clearance = float("inf")
        medial: float | None = None
        lateral: float | None = None

        if near.shape[0]:
            medial_mask = _within_compartment(near, medial_probe, frame)
            lateral_mask = _within_compartment(near, lateral_probe, frame)

            for query in queries:
                try:
                    signed = np.asarray(query.signed_distance(near), dtype=np.float64)
                except Exception as exc:  # noqa: BLE001
                    log.debug(
                        "distance query failed during flexion sweep",
                        fields={"angle": angle, "error": str(exc)},
                    )
                    continue
                inside = signed[signed > 0]
                if inside.size:
                    penetration = max(penetration, float(inside.max()))
                separation = -signed
                if separation.size:
                    clearance = min(clearance, float(separation.min()))
                if medial_mask.any():
                    value = float(separation[medial_mask].min())
                    medial = value if medial is None else min(medial, value)
                if lateral_mask.any():
                    value = float(separation[lateral_mask].min())
                    lateral = value if lateral is None else min(lateral, value)

        report.samples.append(
            FlexionSample(
                angle_deg=angle,
                interpenetration_mm=(
                    penetration if penetration > config.interference_tolerance_mm else 0.0
                ),
                clearance_mm=0.0 if not math.isfinite(clearance) else clearance,
                medial_clearance_mm=medial,
                lateral_clearance_mm=lateral,
            )
        )

    log.debug(
        "flexion sweep broad phase",
        fields={
            "femoral_points": int(femoral_points.shape[0]),
            "points_queried_total": total_queried,
            "angles": len(report.samples),
        },
    )

    _assess(report, config)

    report.provenance = Provenance.for_component(
        "simulation.kinematic",
        parameters={
            "plan_id": geometric.plan_id,
            "angles_deg": list(config.all_angles()),
            "flexion_axis": axis_label,
            "interference_tolerance_mm": config.interference_tolerance_mm,
            "max_vertices_per_mesh": config.max_vertices_per_mesh,
            "approximated_components": list(geometric.approximated_components),
        },
        notes="; ".join(report.limitations),
    )

    clear_range = report.interference_free_range_deg
    log.info(
        "flexion sweep complete",
        fields={
            "plan_id": geometric.plan_id,
            "angles": len(report.samples),
            "interference_free_range_deg": (
                [round(v, 1) for v in clear_range] if clear_range else None
            ),
            "first_interference_deg": report.first_interference_deg,
        },
    )
    return report


def _gap_probes(
    landmarks: LandmarkSet, frame: PatientFrame
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if landmarks.has(
        LandmarkName.MEDIAL_PLATEAU_LOW, LandmarkName.LATERAL_PLATEAU_LOW
    ):
        return (
            landmarks.position(LandmarkName.MEDIAL_PLATEAU_LOW),
            landmarks.position(LandmarkName.LATERAL_PLATEAU_LOW),
        )
    centre = landmarks.position(LandmarkName.TIBIAL_KNEE_CENTER)
    return centre + 18.0 * frame.medial, centre + 18.0 * frame.lateral


#: Radius of the cylinder around each compartment probe within which clearance is
#: measured. Roughly the size of the loaded contact region.
COMPARTMENT_PROBE_RADIUS_MM = 12.0


def _within_compartment(
    points: NDArray[np.float64],
    probe: NDArray[np.float64],
    frame: PatientFrame,
    *,
    radius_mm: float = COMPARTMENT_PROBE_RADIUS_MM,
) -> NDArray[np.bool_]:
    """Mask selecting points inside a proximal cylinder around a compartment probe.

    Compartment clearance is reported as a *minimum separation* within this cylinder
    rather than as a vertical distance. A vertical measurement — lowest femoral point
    minus highest tibial point — is unreliable, because at some flexion angles the
    cylinder catches part of the femoral shaft but not the condyle, and the "lowest
    femoral point" is then tens of millimetres from the joint.
    """
    offsets = points - probe
    radial = np.linalg.norm(
        offsets - np.outer(offsets @ frame.superior, frame.superior), axis=1
    )
    return radial <= radius_mm


def _assess(report: KinematicReport, config: KinematicConfig) -> None:
    """Turn the sweep into findings."""
    if not report.samples:  # pragma: no cover - defensive
        return

    extension = min(report.samples, key=lambda s: abs(s.angle_deg))
    if extension.interferes:
        report.add(
            "interference_in_extension",
            Severity.FAILURE,
            f"the construct interferes by {extension.interpenetration_mm:.2f} mm at "
            f"{extension.angle_deg:.0f}° of flexion. The components cannot be seated as "
            "planned.",
            angle_deg=extension.angle_deg,
            depth_mm=round(extension.interpenetration_mm, 3),
        )

    clear_range = report.interference_free_range_deg
    if clear_range is None:
        report.add(
            "no_clear_range",
            Severity.FAILURE,
            "the construct interferes at every flexion angle evaluated.",
        )
        return

    low, high = clear_range
    if high < 110.0:
        report.add(
            "limited_flexion_geometry",
            Severity.WARNING,
            f"the planned geometry interferes beyond {high:.0f}° of flexion. Since "
            "articular curvature and soft tissue are not modelled, treat this as a "
            "prompt to inspect the 3D model rather than as a predicted range of "
            "motion.",
            clear_to_deg=high,
        )

    hyperextension = [s for s in report.samples if s.angle_deg < 0]
    if hyperextension and any(s.interferes for s in hyperextension):
        report.add(
            "interference_in_hyperextension",
            Severity.WARNING,
            "the construct interferes in hyperextension. Check the posterior "
            "condylar resection and the tibial slope.",
        )

    # The clearance trend through flexion is more informative than any single angle.
    clearances = [
        (s.angle_deg, s.medial_clearance_mm, s.lateral_clearance_mm)
        for s in report.samples
        if s.medial_clearance_mm is not None and s.lateral_clearance_mm is not None
    ]
    if len(clearances) >= 2:
        first_angle, first_medial, first_lateral = clearances[0]
        mid = [c for c in clearances if 80.0 <= c[0] <= 100.0]
        if mid:
            _, mid_medial, mid_lateral = mid[0]
            medial_change = float(mid_medial) - float(first_medial)  # type: ignore[arg-type]
            lateral_change = float(mid_lateral) - float(first_lateral)  # type: ignore[arg-type]
            report.add(
                "clearance_trend",
                Severity.INFO,
                f"between {first_angle:.0f}° and 90° of flexion the medial compartment "
                f"clearance changes by {medial_change:+.1f} mm and the lateral by "
                f"{lateral_change:+.1f} mm.",
                medial_change_mm=round(medial_change, 3),
                lateral_change_mm=round(lateral_change, 3),
            )

        asymmetry = [
            (angle, float(medial) - float(lateral))
            for angle, medial, lateral in clearances
        ]
        worst_angle, worst = max(asymmetry, key=lambda item: abs(item[1]))
        if abs(worst) > 3.0:
            tighter = "lateral" if worst > 0 else "medial"
            report.add(
                "clearance_asymmetry",
                Severity.WARNING,
                f"the compartments differ in clearance by {abs(worst):.1f} mm at "
                f"{worst_angle:.0f}° of flexion, with the {tighter} compartment "
                "tighter. Check the coronal cuts and expect soft-tissue balancing.",
                angle_deg=worst_angle,
                difference_mm=round(worst, 3),
            )
