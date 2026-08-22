"""Approved surgical plan -> abstract robot command sequence (PRD §77, §80).

What this is: a translation from a surgeon-approved plan into a sequence of
high-level, inspectable tool poses, plus a keep-out boundary.

What this is **not**: joint angles, inverse kinematics, servo setpoints, or
anything a controller could execute. There is no arm model here and no driver
import. The output is data describing *where a tool would go*, in millimetres, in
the image frame — for visualisation and review only.

Why the split matters. PRD §80 forbids a path from a network's output to an
actuator. The plan already went through deterministic planning and a human
approval gate before reaching this module, and what leaves this module still is
not executable. A real system would add a controller that re-validates every pose
against its own kinematic and boundary limits before moving; this produces the
*input* to that validation, not a substitute for it.

Poses are tool-frame targets:

- ``position`` — tool tip, mm, image frame
- ``axis`` — unit vector the tool points **along**, pointing into the bone
- ``normal`` — the cut plane normal the pose was derived from

The approach vector is the cut normal reversed, so the tool descends onto the
plane from the free side rather than emerging from inside the bone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kneeai.core.errors import PlanningError
from kneeai.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "RobotAction",
    "ToolPose",
    "RobotCommand",
    "SafetyBoundary",
    "CommandSequence",
    "build_command_sequence",
    "DEFAULT_STANDOFF_MM",
    "DEFAULT_BOUNDARY_MARGIN_MM",
]

#: How far above the cut plane the tool holds before descending.
DEFAULT_STANDOFF_MM = 40.0

#: Margin added around the operative anatomy to form the keep-out boundary.
DEFAULT_BOUNDARY_MARGIN_MM = 15.0


class RobotAction(str, Enum):
    """The abstract command vocabulary.

    Coarse on purpose. Each entry is a phase a surgeon would recognise, not a
    motion primitive, so that a reviewer can read the sequence and see what the
    system intends without reading kinematics.
    """

    MOVE_TO_APPROACH_POSITION = "MOVE_TO_APPROACH_POSITION"
    ALIGN_TO_TARGET_AXIS = "ALIGN_TO_TARGET_AXIS"
    VERIFY_OPERATING_BOUNDARY = "VERIFY_OPERATING_BOUNDARY"
    POSITION_SURGICAL_TOOL = "POSITION_SURGICAL_TOOL"
    SIMULATE_RESECTION = "SIMULATE_RESECTION"
    VERIFY_RESECTION = "VERIFY_RESECTION"
    RETURN_TO_SAFE_POSITION = "RETURN_TO_SAFE_POSITION"


def _unit(v: Iterable[float]) -> NDArray[np.float64]:
    a = np.asarray(list(v), dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise PlanningError("cannot normalise a zero-length direction vector")
    return a / n


@dataclass(frozen=True)
class ToolPose:
    """A tool-tip target in the image frame. Millimetres."""

    position: tuple[float, float, float]
    axis: tuple[float, float, float]
    normal: tuple[float, float, float] | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "position_mm": [round(v, 3) for v in self.position],
            "axis": [round(v, 5) for v in self.axis],
            "frame": "image",
            "units": "mm",
        }
        if self.normal is not None:
            payload["plane_normal"] = [round(v, 5) for v in self.normal]
        if self.label:
            payload["label"] = self.label
        return payload


@dataclass(frozen=True)
class SafetyBoundary:
    """Keep-out volume around the operative anatomy.

    An axis-aligned box in the image frame, derived from the mesh bounds of the
    structures being cut plus a margin. Deliberately crude: a real haptic boundary
    is a surface, not a box, and calling this one a haptic boundary would overstate
    it. Its job here is to give the reviewer and the viewer something concrete to
    check poses against.
    """

    min_mm: tuple[float, float, float]
    max_mm: tuple[float, float, float]
    margin_mm: float
    derived_from: tuple[str, ...]

    def contains(self, point: Sequence[float]) -> bool:
        p = np.asarray(point, dtype=np.float64)
        return bool(np.all(p >= np.asarray(self.min_mm)) and np.all(p <= np.asarray(self.max_mm)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "axis_aligned_box",
            "min_mm": [round(v, 3) for v in self.min_mm],
            "max_mm": [round(v, 3) for v in self.max_mm],
            "size_mm": [round(b - a, 3) for a, b in zip(self.min_mm, self.max_mm)],
            "margin_mm": self.margin_mm,
            "derived_from": list(self.derived_from),
            "frame": "image",
            "note": (
                "Axis-aligned bounding box plus margin. A review aid, not a haptic "
                "constraint surface."
            ),
        }


@dataclass(frozen=True)
class RobotCommand:
    """One abstract action. Never executed; only inspected and drawn."""

    sequence: int
    action: RobotAction
    target: str
    status: str = "READY_FOR_SIMULATION"
    pose: ToolPose | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sequence": self.sequence,
            "action": self.action.value,
            "target": self.target,
            "status": self.status,
        }
        if self.pose is not None:
            payload["pose"] = self.pose.to_dict()
        if self.parameters:
            payload["parameters"] = dict(self.parameters)
        if self.rationale:
            payload["rationale"] = self.rationale
        return payload


@dataclass
class CommandSequence:
    """The full sequence for one plan, plus the boundary it was checked against."""

    case_id: str
    plan_id: str
    commands: list[RobotCommand] = field(default_factory=list)
    boundary: SafetyBoundary | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_commands(self) -> int:
        return len(self.commands)

    def poses_outside_boundary(self) -> list[RobotCommand]:
        """Commands whose tool pose sits outside the keep-out box.

        Approach and retract poses are *expected* to sit outside — that is what
        standoff means — so only cutting-phase poses are checked.
        """
        if self.boundary is None:
            return []
        checked = {
            RobotAction.POSITION_SURGICAL_TOOL,
            RobotAction.SIMULATE_RESECTION,
        }
        return [
            c
            for c in self.commands
            if c.action in checked
            and c.pose is not None
            and not self.boundary.contains(c.pose.position)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "plan_id": self.plan_id,
            "n_commands": self.n_commands,
            "frame": "image",
            "units": "mm",
            "executable": False,
            "hardware_control": False,
            "note": (
                "Abstract tool poses for visualisation and review. Not joint angles, "
                "not servo setpoints, and not executable by any controller."
            ),
            "safety_boundary": self.boundary.to_dict() if self.boundary else None,
            "commands": [c.to_dict() for c in self.commands],
            "warnings": list(self.warnings),
        }


def _boundary_from_manifest(
    manifest: Mapping[str, Any] | None,
    structures: Iterable[str],
    margin_mm: float,
) -> SafetyBoundary | None:
    """Bounding box over the named structures' meshes, expanded by a margin."""
    if not manifest:
        return None
    wanted = {s for s in structures}
    lows: list[NDArray[np.float64]] = []
    highs: list[NDArray[np.float64]] = []
    used: list[str] = []
    for entry in manifest.get("meshes", []) or []:
        if entry.get("structure") not in wanted:
            continue
        bounds = entry.get("bounds_mm")
        if not bounds or len(bounds) != 2:
            continue
        lows.append(np.asarray(bounds[0], dtype=np.float64))
        highs.append(np.asarray(bounds[1], dtype=np.float64))
        used.append(str(entry["structure"]))
    if not lows:
        return None
    lo = np.min(np.vstack(lows), axis=0) - margin_mm
    hi = np.max(np.vstack(highs), axis=0) + margin_mm
    return SafetyBoundary(
        min_mm=tuple(float(v) for v in lo),  # type: ignore[arg-type]
        max_mm=tuple(float(v) for v in hi),  # type: ignore[arg-type]
        margin_mm=margin_mm,
        derived_from=tuple(sorted(set(used))),
    )


def build_command_sequence(
    plan: Mapping[str, Any],
    *,
    mesh_manifest: Mapping[str, Any] | None = None,
    simulation: Mapping[str, Any] | None = None,
    standoff_mm: float = DEFAULT_STANDOFF_MM,
    boundary_margin_mm: float = DEFAULT_BOUNDARY_MARGIN_MM,
) -> CommandSequence:
    """Translate an approved plan into an abstract command sequence.

    Args:
        plan: a serialised :class:`~kneeai.planning.SurgicalPlan`.
        mesh_manifest: the case's ``meshes/manifest.json``, used for the boundary.
        simulation: the geometric simulation result, used to annotate verification.

    The caller is responsible for having obtained surgeon approval first. This
    function does not check for it — :class:`~kneeai.robotics.safety.SimulationController`
    is what enforces that, and duplicating the gate here would create two places
    to get it wrong.
    """
    cuts: Sequence[Mapping[str, Any]] = plan.get("cuts", []) or []
    if not cuts:
        raise PlanningError("plan defines no resection cuts; nothing to sequence")

    case_id = str(plan.get("case_id", "unknown"))
    plan_id = str(plan.get("plan_id", "unknown"))
    structures = [str(c.get("structure")) for c in cuts if c.get("structure")]
    boundary = _boundary_from_manifest(mesh_manifest, structures, boundary_margin_mm)

    seq = CommandSequence(case_id=case_id, plan_id=plan_id, boundary=boundary)
    if boundary is None:
        seq.warnings.append(
            "No mesh manifest supplied, so no keep-out boundary was derived. "
            "Boundary verification is reported as SKIPPED rather than PASSED."
        )

    n = 0

    def emit(
        action: RobotAction,
        target: str,
        *,
        pose: ToolPose | None = None,
        parameters: Mapping[str, Any] | None = None,
        rationale: str = "",
        status: str = "READY_FOR_SIMULATION",
    ) -> None:
        nonlocal n
        n += 1
        seq.commands.append(
            RobotCommand(
                sequence=n,
                action=action,
                target=target,
                status=status,
                pose=pose,
                parameters=dict(parameters or {}),
                rationale=rationale,
            )
        )

    # Home pose: above the whole operative field, on the boundary's superior face.
    if boundary is not None:
        home = (
            (boundary.min_mm[0] + boundary.max_mm[0]) / 2.0,
            (boundary.min_mm[1] + boundary.max_mm[1]) / 2.0,
            boundary.max_mm[2] + standoff_mm,
        )
    else:
        first_point = np.asarray(cuts[0]["plane"]["point"], dtype=np.float64)
        home = (float(first_point[0]), float(first_point[1]), float(first_point[2] + standoff_mm))

    emit(
        RobotAction.VERIFY_OPERATING_BOUNDARY,
        target="operative_field",
        parameters={
            "boundary": boundary.to_dict() if boundary else None,
            "result": "PASS" if boundary is not None else "SKIPPED",
        },
        rationale=(
            "Keep-out volume established before any tool motion is planned."
            if boundary is not None
            else "No anatomy bounds available; boundary check cannot be performed."
        ),
        status="READY_FOR_SIMULATION" if boundary is not None else "SKIPPED",
    )

    for cut in cuts:
        name = str(cut.get("name", "cut"))
        structure = str(cut.get("structure", "unknown"))
        plane = cut.get("plane") or {}
        try:
            point = np.asarray(plane["point"], dtype=np.float64)
            normal = _unit(plane["normal"])
        except (KeyError, TypeError, PlanningError) as exc:
            seq.warnings.append(f"cut '{name}' skipped: unusable plane ({exc})")
            continue

        # Descend along -normal, i.e. approach from the side the normal points to.
        approach_axis = -normal
        approach_point = point + normal * standoff_mm

        emit(
            RobotAction.MOVE_TO_APPROACH_POSITION,
            target=f"{structure}:{name}",
            pose=ToolPose(
                position=tuple(float(v) for v in approach_point),  # type: ignore[arg-type]
                axis=tuple(float(v) for v in approach_axis),  # type: ignore[arg-type]
                normal=tuple(float(v) for v in normal),  # type: ignore[arg-type]
                label=f"{name}_approach",
            ),
            parameters={"standoff_mm": standoff_mm},
            rationale=(
                f"Hold {standoff_mm:.0f} mm clear of the {name} plane along its "
                "normal, so the descent is onto the plane from the free side."
            ),
        )

        angles = cut.get("angles_deg") or {}
        emit(
            RobotAction.ALIGN_TO_TARGET_AXIS,
            target=f"{structure}:{name}",
            pose=ToolPose(
                position=tuple(float(v) for v in approach_point),  # type: ignore[arg-type]
                axis=tuple(float(v) for v in approach_axis),  # type: ignore[arg-type]
                normal=tuple(float(v) for v in normal),  # type: ignore[arg-type]
                label=f"{name}_aligned",
            ),
            parameters={
                "plane_normal": [round(float(v), 5) for v in normal],
                "coronal_deg": angles.get("coronal_varus"),
                "sagittal_deg": angles.get("sagittal_slope"),
                "axial_deg": angles.get("axial_rotation"),
            },
            rationale=cut.get("reference", "") or "Align the tool to the planned plane normal.",
        )

        in_boundary = boundary.contains(point) if boundary else None
        emit(
            RobotAction.POSITION_SURGICAL_TOOL,
            target=f"{structure}:{name}",
            pose=ToolPose(
                position=tuple(float(v) for v in point),  # type: ignore[arg-type]
                axis=tuple(float(v) for v in approach_axis),  # type: ignore[arg-type]
                normal=tuple(float(v) for v in normal),  # type: ignore[arg-type]
                label=f"{name}_contact",
            ),
            parameters={
                "depth_mm": cut.get("depth_mm"),
                "inside_boundary": in_boundary,
            },
            rationale="Tool tip at the plane, on the planned resection depth.",
        )

        emit(
            RobotAction.SIMULATE_RESECTION,
            target=f"{structure}:{name}",
            pose=ToolPose(
                position=tuple(float(v) for v in point),  # type: ignore[arg-type]
                axis=tuple(float(v) for v in approach_axis),  # type: ignore[arg-type]
                normal=tuple(float(v) for v in normal),  # type: ignore[arg-type]
                label=f"{name}_resection",
            ),
            parameters={
                "depth_mm": cut.get("depth_mm"),
                "planned_resected_volume_mm3": cut.get("resected_volume_mm3"),
                "execution": "DIGITAL_SIMULATION_ONLY",
            },
            rationale=(
                "Virtual resection along the planned plane. No material is removed "
                "and no hardware is commanded."
            ),
        )

    resected = (simulation or {}).get("resected_volume_mm3")
    sim_passed = (simulation or {}).get("passed")
    emit(
        RobotAction.VERIFY_RESECTION,
        target="resected_anatomy",
        parameters={
            "geometric_simulation_passed": sim_passed,
            "resected_volume_mm3": resected,
            "seating_gap_mm": (simulation or {}).get("seating_gap_mm"),
            "interpenetration_mm": (simulation or {}).get("interpenetration_mm"),
            "result": (
                "PASS" if sim_passed else ("FAIL" if simulation is not None else "SKIPPED")
            ),
        },
        rationale=(
            "Compare the simulated resection against the plan using "
            "kneeai.simulation, which is what actually evaluates the cuts."
        ),
        status="READY_FOR_SIMULATION" if simulation is not None else "SKIPPED",
    )

    emit(
        RobotAction.RETURN_TO_SAFE_POSITION,
        target="home",
        pose=ToolPose(
            position=home,
            axis=(0.0, 0.0, 1.0),
            label="home",
        ),
        rationale="Withdraw clear of the operative field.",
    )

    outside = seq.poses_outside_boundary()
    if outside:
        seq.warnings.append(
            f"{len(outside)} cutting-phase pose(s) fall outside the keep-out "
            f"boundary: {', '.join(c.pose.label for c in outside if c.pose)}. "
            "This indicates a plane placed away from the segmented anatomy."
        )

    log.info(
        "robot command sequence built",
        fields={
            "case": case_id,
            "plan": plan_id,
            "commands": seq.n_commands,
            "cuts": len(cuts),
            "warnings": len(seq.warnings),
        },
    )
    return seq
