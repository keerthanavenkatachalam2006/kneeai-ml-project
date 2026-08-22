"""Machine-readable surgical plan for a future robotic interface (PRD §76, §77).

Assembles one self-contained document from the case artefacts:

``patient_case``       identity, laterality, and the provenance of the geometry
``anatomy``            structures, volumes, and the measurements that drove the plan
``implant``            selected configuration, sizes, and why it was chosen
``alignment``          strategy and targets, with unavailable values named
``cuts``               resection planes with point, normal, depth and angles
``rotations``          component orientation angles per cut
``offsets``            resection depths and insert thickness
``safety_constraints`` keep-out boundary, approval record, and the hard limits

Two design decisions worth stating.

**Unavailable values are named, not omitted.** A robot interface that receives a
plan with no ``hka_angle`` key cannot tell whether alignment was measured and
found normal, or never measurable at all. The second is the truth for a knee MRI —
the field of view holds neither the femoral head nor the ankle — so the document
carries an explicit ``unavailable`` list with the reason attached.

**The document is inert.** It contains no command, no trajectory and no
executable field. :mod:`kneeai.robotics.commands` derives poses from it separately,
and those are not executable either. A future controller would consume this
document, re-derive its own motion under its own limits, and re-validate
everything. This is the interface, not the actuator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from kneeai.logging import get_logger
from kneeai.robotics.commands import CommandSequence
from kneeai.robotics.safety import SIMULATION_ONLY_BANNER, SimulationController
from kneeai.version import PLATFORM_VERSION

log = get_logger(__name__)

__all__ = ["RobotPlan", "build_robot_plan", "write_robot_plan"]

SCHEMA_VERSION = "1"

#: Limits a consuming controller must enforce for itself. Advisory here: this
#: document cannot enforce anything, and saying otherwise would be the exact
#: overstatement PRD §80 warns about.
HARD_LIMITS: dict[str, Any] = {
    "autonomous_execution_permitted": False,
    "hardware_control_present": False,
    "surgeon_approval_required_before_any_motion": True,
    "llm_output_in_control_path": False,
    "max_deviation_from_planned_plane_mm": 1.0,
    "max_deviation_from_planned_angle_deg": 1.0,
    "controller_must_revalidate_every_pose": True,
    "note": (
        "Advisory limits for a future controller. This document enforces nothing; "
        "it declares what a controller is expected to enforce."
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RobotPlan:
    """The assembled document."""

    patient_case: dict[str, Any] = field(default_factory=dict)
    anatomy: dict[str, Any] = field(default_factory=dict)
    implant: dict[str, Any] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    cuts: list[dict[str, Any]] = field(default_factory=list)
    rotations: dict[str, Any] = field(default_factory=dict)
    offsets: dict[str, Any] = field(default_factory=dict)
    safety_constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kneeai.robot_plan",
            "schema_version": SCHEMA_VERSION,
            "platform_version": PLATFORM_VERSION,
            "generated_at": _now(),
            "banner": SIMULATION_ONLY_BANNER,
            "executable": False,
            "patient_case": self.patient_case,
            "anatomy": self.anatomy,
            "implant": self.implant,
            "alignment": self.alignment,
            "cuts": self.cuts,
            "rotations": self.rotations,
            "offsets": self.offsets,
            "safety_constraints": self.safety_constraints,
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Flatten the cuts to one row each, for spreadsheet review."""
        rows = []
        for c in self.cuts:
            plane = c.get("plane", {})
            angles = c.get("angles_deg", {}) or {}
            rows.append(
                {
                    "case_id": self.patient_case.get("case_id"),
                    "side": self.patient_case.get("side"),
                    "cut": c.get("name"),
                    "structure": c.get("structure"),
                    "depth_mm": c.get("depth_mm"),
                    "point_x_mm": (plane.get("point") or [None] * 3)[0],
                    "point_y_mm": (plane.get("point") or [None] * 3)[1],
                    "point_z_mm": (plane.get("point") or [None] * 3)[2],
                    "normal_x": (plane.get("normal") or [None] * 3)[0],
                    "normal_y": (plane.get("normal") or [None] * 3)[1],
                    "normal_z": (plane.get("normal") or [None] * 3)[2],
                    "coronal_deg": angles.get("coronal_varus"),
                    "sagittal_deg": angles.get("sagittal_slope"),
                    "axial_deg": angles.get("axial_rotation"),
                    "reference": c.get("reference"),
                }
            )
        return rows


def _measurement_lookup(measurements: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten the serialised measurement set into ``name -> record``.

    Accepts both the list form and a name-keyed mapping, because the engine
    serialises a list while some consumers index by name.
    """
    if not measurements:
        return {}
    items = measurements.get("measurements")
    out: dict[str, Any] = {}
    if isinstance(items, Mapping):
        iterable: Sequence[Any] = list(items.values())
    elif isinstance(items, Sequence):
        iterable = items
    else:
        return {}
    for m in iterable:
        if not isinstance(m, Mapping):
            continue
        name = m.get("name")
        if name:
            out[str(name)] = m
    return out


def _quantity(record: Mapping[str, Any] | None) -> tuple[Any, str | None]:
    """Pull ``(value, unit)`` out of a measurement record.

    The value lives under ``quantity`` — ``{"quantity": {"value": .., "unit": ..}}``
    — because the engine carries units and uncertainty alongside the number rather
    than as a bare float. Reading ``record["value"]`` silently yields ``None`` for
    every measurement, which looks like "not measured" rather than "read wrongly".
    """
    if not record:
        return None, None
    q = record.get("quantity")
    if isinstance(q, Mapping):
        return q.get("value"), q.get("unit")
    return record.get("value"), record.get("unit")


def _unavailable_names(measurements: Mapping[str, Any] | None) -> list[str]:
    """Names of measurements the engine could not compute.

    Serialised as a mapping of ``name -> reason``; older payloads used a list.
    """
    raw = (measurements or {}).get("unavailable")
    if isinstance(raw, Mapping):
        return sorted(str(k) for k in raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(v) for v in raw]
    return []


def _unavailable_reasons(measurements: Mapping[str, Any] | None) -> dict[str, str]:
    raw = (measurements or {}).get("unavailable")
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def build_robot_plan(
    *,
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
    measurements: Mapping[str, Any] | None = None,
    implants: Mapping[str, Any] | None = None,
    simulation: Mapping[str, Any] | None = None,
    mesh_manifest: Mapping[str, Any] | None = None,
    commands: CommandSequence | None = None,
    controller: SimulationController | None = None,
) -> RobotPlan:
    """Assemble the document from a case's artefacts."""
    seg = result.get("segmentation", {}) or {}
    geom = result.get("geometry", {}) or {}
    conf = result.get("confidence", {}) or {}
    lookup = _measurement_lookup(measurements)

    def value_of(name: str) -> Any:
        value, _unit = _quantity(lookup.get(name))
        return value

    unavailable = _unavailable_names(measurements)
    unavailable_detail = _unavailable_reasons(measurements)

    robot = RobotPlan()

    robot.patient_case = {
        "case_id": result.get("case_id"),
        "plan_id": plan.get("plan_id"),
        "plan_version": plan.get("version"),
        "side": geom.get("side"),
        # Carried prominently: animating a phantom is fine, presenting it as a
        # patient is not.
        "geometry_source": result.get("source"),
        "model": result.get("model"),
        "image_geometry": {
            "spacing_mm": geom.get("spacing_mm"),
            "image_shape": geom.get("image_shape"),
            "orientation": geom.get("orientation"),
        },
    }

    robot.anatomy = {
        "structures_present": seg.get("structures_present", []),
        "structures_unavailable": seg.get("structures_expected_but_missing", []),
        "label_map": seg.get("label_map", {}),
        "volumes_mm3": conf.get("volume_mm3", {}),
        "segmentation_confidence": conf.get("per_structure", {}),
        "key_dimensions_mm": {
            "femoral_ml_width": value_of("femoral_ml_width"),
            "femoral_ap_size": value_of("femoral_ap_size"),
            "transepicondylar_width": value_of("transepicondylar_width"),
            "tibial_ml_width": value_of("tibial_ml_width"),
            "tibial_ap_depth": value_of("tibial_ap_depth"),
        },
    }

    selected = (implants or {}).get("selected") or {}
    robot.implant = {
        "configuration_id": plan.get("configuration_id"),
        "configuration_label": plan.get("configuration_label"),
        "femoral_size": plan.get("femoral_size"),
        "tibial_size": plan.get("tibial_size"),
        "insert_thickness_mm": plan.get("insert_thickness_mm"),
        "selection_rationale": (implants or {}).get("rationale"),
        "runner_up": (implants or {}).get("runner_up"),
        "fit_metrics": {
            k: selected.get(k)
            for k in ("fit_score", "coverage_pct", "max_overhang_mm", "undercoverage_mm")
            if k in selected
        },
        "catalog_disclaimer": (implants or {}).get(
            "disclaimer",
            "Reference (non-clinical) implant geometry. Not real manufacturer data.",
        ),
        "placements": plan.get("placements", []),
    }

    robot.alignment = {
        "strategy": plan.get("strategy"),
        "targets": plan.get("targets"),
        "measured": {
            "posterior_tibial_slope_deg": value_of("posterior_tibial_slope"),
            "condylar_twist_angle_deg": value_of("condylar_twist_angle"),
            "jlca_deg": value_of("jlca"),
            "hka_angle_deg": value_of("hka_angle"),
            "mldfa_deg": value_of("mldfa"),
            "mpta_deg": value_of("mpta"),
        },
        "unavailable": unavailable,
        "unavailable_detail": unavailable_detail,
        "unavailable_reason": (
            "Mechanical-axis measurements (HKA, MLDFA, MPTA) require the femoral "
            "head and ankle centres. A knee MRI field of view contains neither, so "
            "these cannot be computed from this study alone. A long-leg radiograph "
            "or full-leg CT is required."
            if any(n.startswith(("hka", "mldfa", "mpta")) for n in unavailable)
            else None
        ),
    }

    robot.cuts = [dict(c) for c in (plan.get("cuts") or [])]

    robot.rotations = {
        str(c.get("name")): {
            "coronal_deg": (c.get("angles_deg") or {}).get("coronal_varus"),
            "sagittal_deg": (c.get("angles_deg") or {}).get("sagittal_slope"),
            "axial_deg": (c.get("angles_deg") or {}).get("axial_rotation"),
            "plane_normal": (c.get("plane") or {}).get("normal"),
        }
        for c in robot.cuts
    }

    robot.offsets = {
        "resection_depth_mm": {
            str(c.get("name")): c.get("depth_mm") for c in robot.cuts
        },
        "insert_thickness_mm": plan.get("insert_thickness_mm"),
        "total_resected_volume_mm3": (simulation or {}).get("resected_volume_mm3"),
        "seating_gap_mm": (simulation or {}).get("seating_gap_mm"),
        "interpenetration_mm": (simulation or {}).get("interpenetration_mm"),
    }

    robot.safety_constraints = {
        "hard_limits": HARD_LIMITS,
        "keep_out_boundary": (
            commands.boundary.to_dict() if commands and commands.boundary else None
        ),
        "approval": {
            "required": True,
            "state": controller.state.value if controller else "not_started",
            "approver": controller.approver if controller else None,
            "approved_at": (
                controller.approved_at.isoformat()
                if controller and controller.approved_at
                else None
            ),
        },
        "safety_checks": (
            controller.safety_report.to_dict()
            if controller and controller.safety_report
            else None
        ),
        "geometric_simulation": {
            "passed": (simulation or {}).get("passed"),
            "n_failures": (simulation or {}).get("n_failures"),
            "n_warnings": (simulation or {}).get("n_warnings"),
            "findings": (simulation or {}).get("findings", []),
        },
        "command_sequence_warnings": list(commands.warnings) if commands else [],
    }

    robot.metadata = {
        "n_cuts": len(robot.cuts),
        "n_commands": commands.n_commands if commands else 0,
        "mesh_count": len((mesh_manifest or {}).get("meshes", []) or []),
        "clinical_status": "RESEARCH / SIMULATION PROTOTYPE — NOT FOR CLINICAL USE",
        "provenance": plan.get("provenance") or result.get("provenance"),
    }

    log.info(
        "robot plan assembled",
        fields={
            "case": robot.patient_case.get("case_id"),
            "plan": robot.patient_case.get("plan_id"),
            "cuts": len(robot.cuts),
            "source": robot.patient_case.get("geometry_source"),
        },
    )
    return robot


def write_robot_plan(
    robot: RobotPlan,
    out_dir: str | os.PathLike[str],
    *,
    commands: CommandSequence | None = None,
    write_csv: bool = True,
) -> dict[str, Path]:
    """Write ``robot_plan.json``, optionally ``robot_commands.json`` and a CSV."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    plan_path = directory / "robot_plan.json"
    plan_path.write_text(robot.to_json())
    written["robot_plan"] = plan_path

    if commands is not None:
        cmd_path = directory / "robot_commands.json"
        cmd_path.write_text(json.dumps(commands.to_dict(), indent=2, default=str))
        written["robot_commands"] = cmd_path

    if write_csv:
        import csv

        rows = robot.to_csv_rows()
        if rows:
            csv_path = directory / "robot_cuts.csv"
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            written["robot_cuts_csv"] = csv_path

    return written
