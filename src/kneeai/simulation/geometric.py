"""Level 1 geometric simulation: virtual cuts, implant placement, collision (PRD §28, §73).

What this module establishes, before anything moves:

1. **The cuts are executable.** Each plane actually intersects its bone, removes a
   plausible volume, and leaves a valid solid behind.
2. **The components fit the cut bone.** Placed component geometry does not
   interpenetrate retained bone, and does not float away from the cut surface.
3. **Nothing collides that should not.** Femoral and tibial constructs do not
   intersect each other in extension.

PRD §88 requires the system to fail safe on invalid anatomy and impossible implant
placement. So every check here reports a *finding* rather than raising: an
impossible plan produces a simulation report full of failures, which is exactly the
information the surgeon needs, whereas an exception produces a blank screen.

Component solids
----------------
The reference catalog ships footprints, not full 3D surfaces. Where a component has
no mesh, this module extrudes its footprint into a prism of the component's
thickness. That is a genuine approximation and it is labelled as one throughout:
a prism captures the footprint boundary and thickness (which is what overhang and
gap checks need) but not the articular curvature (which is what a real
collision/kinematics check needs). :attr:`SimulationReport.approximated_components`
lists every component simulated this way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import Side, Structure
from kneeai.core.errors import SimulationError
from kneeai.core.frames import Frame, Transform
from kneeai.core.geometry import Plane
from kneeai.core.provenance import Provenance
from kneeai.implants.catalog import ComponentType, ImplantComponent, ImplantConfiguration
from kneeai.landmarks.frame import PatientFrame
from kneeai.logging import get_logger
from kneeai.planning.plan import FEMORAL_CUTS, CutName, SurgicalPlan
from kneeai.recon.mesh import AnatomicalMesh, MeshSet

__all__ = [
    "Severity",
    "Finding",
    "CutResult",
    "SimulationReport",
    "simulate_plan",
    "apply_cuts",
    "component_solid",
    "COLLISION_TOLERANCE_MM",
]

log = get_logger("simulation.geometric")

#: Interpenetration below this is attributed to mesh discretisation rather than a
#: genuine collision. At 0.5 mm voxels a marching-cubes surface is accurate to a
#: few tenths of a millimetre, so a tighter threshold would report noise.
COLLISION_TOLERANCE_MM = 0.4


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    FAILURE = "failure"

    @property
    def blocks(self) -> bool:
        return self is Severity.FAILURE


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "detail": dict(self.detail),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


@dataclass
class CutResult:
    """Outcome of applying one cut."""

    name: CutName
    succeeded: bool
    resected_volume_mm3: float = 0.0
    retained_volume_mm3: float = 0.0
    cut_area_mm2: float = 0.0
    reason: str = ""

    @property
    def resected_fraction(self) -> float:
        total = self.resected_volume_mm3 + self.retained_volume_mm3
        return self.resected_volume_mm3 / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "succeeded": self.succeeded,
            "resected_volume_mm3": round(self.resected_volume_mm3, 1),
            "retained_volume_mm3": round(self.retained_volume_mm3, 1),
            "resected_fraction": round(self.resected_fraction, 4),
            "cut_area_mm2": round(self.cut_area_mm2, 1),
            "reason": self.reason,
        }


@dataclass
class SimulationReport:
    """Result of a Level 1 geometric simulation."""

    plan_id: str
    side: Side
    findings: list[Finding] = field(default_factory=list)
    cuts: dict[CutName, CutResult] = field(default_factory=dict)

    #: Resected bone volume per structure.
    resected_volume_mm3: dict[str, float] = field(default_factory=dict)

    #: Components simulated as extruded footprint prisms rather than real surfaces.
    approximated_components: list[str] = field(default_factory=list)

    #: Maximum interpenetration depth found, per pair.
    interpenetration_mm: dict[str, float] = field(default_factory=dict)

    #: Gap between each component's seating surface and its bone cut.
    seating_gap_mm: dict[str, float] = field(default_factory=dict)

    resected_meshes: dict[Structure, AnatomicalMesh] = field(default_factory=dict)
    component_meshes: dict[str, AnatomicalMesh] = field(default_factory=dict)
    provenance: Provenance | None = None

    def add(
        self, code: str, severity: Severity, message: str, **detail: Any
    ) -> SimulationReport:
        self.findings.append(
            Finding(code=code, severity=severity, message=message, detail=detail)
        )
        return self

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.FAILURE]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "side": self.side.value,
            "passed": self.passed,
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
            "cuts": [c.to_dict() for c in self.cuts.values()],
            "resected_volume_mm3": {
                k: round(v, 1) for k, v in self.resected_volume_mm3.items()
            },
            "approximated_components": list(self.approximated_components),
            "interpenetration_mm": {
                k: round(v, 3) for k, v in self.interpenetration_mm.items()
            },
            "seating_gap_mm": {k: round(v, 3) for k, v in self.seating_gap_mm.items()},
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    def summary(self) -> str:
        lines = [
            f"Geometric simulation of plan {self.plan_id} ({self.side.value} knee): "
            + ("PASSED" if self.passed else f"{len(self.failures)} FAILURE(S)")
        ]
        if self.cuts:
            lines.append("  cuts:")
            for name in (CutName.TIBIAL, *FEMORAL_CUTS, CutName.PATELLAR):
                result = self.cuts.get(name)
                if result is None:
                    continue
                status = "ok" if result.succeeded else f"FAILED ({result.reason})"
                lines.append(
                    f"    {name.display_name:<20} removed "
                    f"{result.resected_volume_mm3:8.1f} mm³ "
                    f"({result.resected_fraction * 100:5.1f}%)  {status}"
                )
        if self.resected_volume_mm3:
            lines.append("  total bone removed:")
            for structure, volume in sorted(self.resected_volume_mm3.items()):
                lines.append(f"    {structure:<20}{volume:10.1f} mm³")
        if self.seating_gap_mm:
            lines.append("  component seating:")
            for name, gap in sorted(self.seating_gap_mm.items()):
                lines.append(f"    {name:<40}{gap:+7.2f} mm")
        if self.interpenetration_mm:
            lines.append("  interpenetration:")
            for pair, depth in sorted(self.interpenetration_mm.items()):
                lines.append(f"    {pair:<40}{depth:7.2f} mm")
        if self.approximated_components:
            lines.append(
                "  note: the following components were simulated as extruded "
                "footprint prisms because the catalog provides no 3D surface — "
                "articular geometry is not represented:"
            )
            lines.extend(f"    - {c}" for c in self.approximated_components)
        for finding in self.findings:
            if finding.severity is not Severity.INFO:
                lines.append(f"  {finding}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cutting
# ---------------------------------------------------------------------------


def apply_cuts(
    mesh: AnatomicalMesh,
    cuts: Sequence[tuple[CutName, Plane]],
    report: SimulationReport,
) -> AnatomicalMesh | None:
    """Apply a sequence of planar cuts to a bone, keeping the retained fragment.

    Cuts are applied one at a time and each is validated, so a report can say
    *which* cut failed rather than only that the preparation did not work.
    """
    original_volume = mesh.volume_mm3
    if original_volume <= 0:
        report.add(
            "bone_not_watertight",
            Severity.FAILURE,
            f"the {mesh.structure.display_name} surface is not watertight, so virtual "
            "cutting cannot compute resected volume. Reconstruct the bone before "
            "simulating.",
            structure=mesh.structure.value,
        )
        return None

    current = mesh
    for name, plane in cuts:
        before = current.volume_mm3
        try:
            cut_mesh = current.slice_plane(plane, cap=True, keep_positive=True)
        except Exception as exc:  # noqa: BLE001 - trimesh raises various types
            report.cuts[name] = CutResult(
                name=name,
                succeeded=False,
                reason=f"the plane does not usably intersect the bone: {exc}",
            )
            report.add(
                "cut_failed",
                Severity.FAILURE,
                f"the {name.display_name} cut could not be applied: {exc}",
                cut=name.value,
            )
            continue

        after = cut_mesh.volume_mm3
        removed = before - after

        if after <= 0:
            report.cuts[name] = CutResult(
                name=name,
                succeeded=False,
                reason="the cut removed the entire bone",
            )
            report.add(
                "cut_removes_everything",
                Severity.FAILURE,
                f"the {name.display_name} cut removes the entire "
                f"{mesh.structure.display_name}. The plane is on the wrong side of the "
                "bone, or its normal is inverted.",
                cut=name.value,
            )
            continue

        if removed <= 1.0:
            # A cut that removes nothing is usually a plane placed outside the bone.
            report.cuts[name] = CutResult(
                name=name,
                succeeded=True,
                resected_volume_mm3=max(0.0, removed),
                retained_volume_mm3=after,
                reason="the cut removed almost no bone",
            )
            report.add(
                "cut_removes_nothing",
                Severity.WARNING,
                f"the {name.display_name} cut removes only "
                f"{max(0.0, removed):.1f} mm³ of bone. The plane may lie outside the "
                "bone, in which case the corresponding component surface has nothing "
                "to seat against.",
                cut=name.value,
                removed_mm3=round(max(0.0, removed), 2),
            )
            current = cut_mesh
            continue

        report.cuts[name] = CutResult(
            name=name,
            succeeded=True,
            resected_volume_mm3=removed,
            retained_volume_mm3=after,
            cut_area_mm2=_cut_face_area(current, plane),
        )
        current = cut_mesh

    report.resected_volume_mm3[mesh.structure.value] = original_volume - current.volume_mm3

    quality = current.quality()
    if not quality.is_watertight:
        report.add(
            "resected_bone_not_solid",
            Severity.WARNING,
            f"after cutting, the {mesh.structure.display_name} is no longer watertight, "
            "so its remaining volume and any collision test against it are unreliable.",
            structure=mesh.structure.value,
        )
    return current


def _cut_face_area(mesh: AnatomicalMesh, plane: Plane) -> float:
    """Area of the cross-section the cut exposes."""
    try:
        polygons = mesh.section_polygons(plane)
    except Exception:  # noqa: BLE001
        return 0.0
    total = 0.0
    for polygon in polygons:
        x, y = polygon[:, 0], polygon[:, 1]
        total += 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return total


# ---------------------------------------------------------------------------
# Component solids
# ---------------------------------------------------------------------------


def component_solid(
    component: ImplantComponent,
    placement: Transform,
    *,
    frame: Frame = Frame.IMAGE,
) -> tuple[AnatomicalMesh, bool]:
    """Build a placed 3D solid for a component.

    Returns ``(mesh, approximated)``. ``approximated`` is True when the solid was
    extruded from the footprint because the catalog has no surface for the
    component.

    Extrusion direction is **away from the bone** — a component sits on its cut and
    its material occupies the space the bone used to. For the tibial baseplate that
    is proximal; for the femoral component the distal-cut footprint extrudes
    distally.
    """
    import trimesh

    if component.mesh_path:
        loaded = AnatomicalMesh.load(
            component.mesh_path,
            structure=_component_structure(component),
            frame=Frame.IMPLANT,
        )
        return loaded.transform(_retarget(placement, loaded.frame, frame)), False

    if not component.has_footprint:
        raise SimulationError(
            f"{component.component_id} has neither a mesh nor a footprint, so no solid "
            "can be built for simulation"
        )

    thickness = component.thickness_mm
    if thickness <= 0:
        raise SimulationError(
            f"{component.component_id} declares zero thickness, so it cannot be "
            "extruded into a solid"
        )

    try:
        prism = trimesh.creation.extrude_polygon(
            component.footprint_polygon(), thickness
        )
    except ValueError as exc:
        # trimesh ships no polygon triangulator of its own.
        raise SimulationError(
            f"cannot extrude {component.component_id} into a solid: no polygon "
            "triangulation backend is available. Install one with:\n"
            "    pip install mapbox-earcut\n"
            f"(underlying error: {exc})"
        ) from exc

    # extrude_polygon builds from z=0 to z=+thickness in the component design frame,
    # whose +Z is proximal. Which way the material extends depends on which side of
    # its cut the component sits:
    #
    #   femoral  — the retained femur is proximal to the distal cut, so the component
    #              occupies the space distal to it (where the condyle was): -Z.
    #   tibial   — the retained tibia is distal to the plateau cut, so the baseplate
    #              and insert sit proximal to it: +Z.
    #
    # Getting this backwards buries the baseplate inside the tibia, which then reads
    # as a 10 mm gap between the constructs and hides any genuine interference.
    if _extrudes_proximally(component):
        offset = 0.0
    else:
        offset = -thickness
    prism.apply_translation([0.0, 0.0, offset])

    local = AnatomicalMesh(
        structure=_component_structure(component),
        mesh=prism,
        frame=Frame.IMPLANT,
        metadata={
            "component_id": component.component_id,
            "approximation": "extruded footprint prism",
        },
    )
    return local.transform(_retarget(placement, Frame.IMPLANT, frame)), True


def _extrudes_proximally(component: ImplantComponent) -> bool:
    """Whether the component's material sits proximal to its seating plane.

    Tibial components sit on top of the plateau cut (proximal); femoral and
    patellar components occupy the space the resected bone vacated, which is distal
    to / behind their seating cut.
    """
    return component.component_type in (
        ComponentType.TIBIAL_BASEPLATE,
        ComponentType.TIBIAL_INSERT,
    )


def _component_structure(component: ImplantComponent) -> Structure:
    if component.component_type is ComponentType.FEMORAL:
        return Structure.FEMUR
    if component.component_type is ComponentType.PATELLAR:
        return Structure.PATELLA
    return Structure.TIBIA


def _retarget(placement: Transform, source: Frame, target: Frame) -> Transform:
    if placement.source is not source:
        raise SimulationError(
            f"component placement maps from '{placement.source.value}' but the "
            f"component mesh is in '{source.value}'"
        )
    if placement.target is not target:
        raise SimulationError(
            f"component placement maps to '{placement.target.value}' but the "
            f"simulation runs in '{target.value}'"
        )
    return placement


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def simulate_plan(
    plan: SurgicalPlan,
    meshes: MeshSet,
    configuration: ImplantConfiguration,
    frame: PatientFrame,
    *,
    check_collisions: bool = True,
) -> SimulationReport:
    """Run the Level 1 geometric simulation for a plan."""
    report = SimulationReport(plan_id=plan.plan_id, side=plan.side)

    missing = plan.missing_cuts()
    if missing:
        report.add(
            "plan_incomplete",
            Severity.FAILURE,
            "the plan is missing cut(s) "
            f"{[c.display_name for c in missing]}, so the preparation cannot be "
            "simulated.",
            missing=[c.value for c in missing],
        )

    # --- tibia ---
    if CutName.TIBIAL in plan.cuts and Structure.TIBIA in meshes:
        tibia = apply_cuts(
            meshes.require(Structure.TIBIA),
            [(CutName.TIBIAL, plan.cut(CutName.TIBIAL).plane)],
            report,
        )
        if tibia is not None:
            report.resected_meshes[Structure.TIBIA] = tibia

    # --- femur ---
    femoral_cuts = [
        (name, plan.cut(name).plane) for name in FEMORAL_CUTS if name in plan.cuts
    ]
    if femoral_cuts and Structure.FEMUR in meshes:
        femur = apply_cuts(meshes.require(Structure.FEMUR), femoral_cuts, report)
        if femur is not None:
            report.resected_meshes[Structure.FEMUR] = femur

    # --- component solids ---
    for structure, component in (
        (Structure.FEMUR, configuration.femoral),
        (Structure.TIBIA, configuration.tibial_baseplate),
    ):
        if structure not in plan.placements:
            continue
        placement = plan.placement(structure)
        try:
            solid, approximated = component_solid(
                component, placement.transform, frame=frame.source_frame
            )
        except SimulationError as exc:
            report.add(
                "component_solid_failed",
                Severity.FAILURE,
                f"a solid could not be built for {component.component_id}: {exc}",
                component=component.component_id,
            )
            continue
        report.component_meshes[component.component_id] = solid
        if approximated:
            report.approximated_components.append(component.component_id)

    # --- seating ---
    _check_seating(plan, report, configuration, frame)

    # --- collisions ---
    if check_collisions:
        _check_collisions(report, frame)

    # --- bone-stock sanity ---
    _check_bone_stock(report)

    report.provenance = Provenance.for_component(
        "simulation.geometric",
        origin=plan.origin,
        parameters={
            "plan_id": plan.plan_id,
            "plan_version": plan.version,
            "configuration_id": configuration.configuration_id,
            "cuts_applied": sorted(c.value for c in report.cuts),
            "collision_tolerance_mm": COLLISION_TOLERANCE_MM,
            "approximated_components": list(report.approximated_components),
        },
        notes=(
            "Level 1 geometric simulation: planar cuts, component placement and "
            "static interpenetration. Soft tissue and articular contact are not "
            "modelled."
        ),
    )

    log.info(
        "geometric simulation complete",
        fields={
            "plan_id": plan.plan_id,
            "passed": report.passed,
            "failures": len(report.failures),
            "warnings": len(report.warnings),
            "cuts": len(report.cuts),
        },
    )
    return report


def _check_seating(
    plan: SurgicalPlan,
    report: SimulationReport,
    configuration: ImplantConfiguration,
    frame: PatientFrame,
) -> None:
    """Verify each component sits on its cut rather than floating or sinking.

    Checked by comparing the placement origin against the cut plane it should lie
    on. A component that does not sit on its cut indicates the placement transform
    and the cut plane were derived inconsistently — a class of error that produces a
    plausible-looking 3D view and a construct that cannot be assembled.
    """
    pairs = (
        (Structure.TIBIA, CutName.TIBIAL, configuration.tibial_baseplate),
        (Structure.FEMUR, CutName.DISTAL_FEMORAL, configuration.femoral),
    )
    for structure, cut_name, component in pairs:
        if structure not in plan.placements or cut_name not in plan.cuts:
            continue
        placement = plan.placement(structure)
        plane = plan.cut(cut_name).plane
        origin = placement.transform.translation
        offset = float(plane.signed_distance(origin))  # type: ignore[arg-type]
        report.seating_gap_mm[component.component_id] = offset
        if abs(offset) > 0.5:
            report.add(
                "component_not_seated",
                Severity.FAILURE,
                f"{component.component_id} sits {offset:+.2f} mm off its "
                f"{cut_name.display_name} cut plane. The placement transform and the "
                "cut plane were derived inconsistently.",
                component=component.component_id,
                cut=cut_name.value,
                offset_mm=round(offset, 3),
            )


def _check_collisions(report: SimulationReport, frame: PatientFrame) -> None:
    """Static interpenetration checks between components and retained bone."""
    checks: list[tuple[str, AnatomicalMesh, AnatomicalMesh]] = []

    for component_id, component_mesh in report.component_meshes.items():
        for structure, bone in report.resected_meshes.items():
            # A component is expected to occupy the space of the bone it replaces, so
            # only the *opposite* bone is a genuine collision partner.
            if component_mesh.structure is structure:
                continue
            checks.append((f"{component_id} vs {structure.value}", component_mesh, bone))

    component_ids = sorted(report.component_meshes)
    for i, a in enumerate(component_ids):
        for b in component_ids[i + 1 :]:
            checks.append(
                (f"{a} vs {b}", report.component_meshes[a], report.component_meshes[b])
            )

    for label, first, second in checks:
        depth = _interpenetration_depth(first, second)
        if depth is None:
            continue
        report.interpenetration_mm[label] = depth
        if depth > COLLISION_TOLERANCE_MM:
            report.add(
                "interpenetration",
                Severity.FAILURE,
                f"{label} interpenetrate by up to {depth:.2f} mm. In extension these "
                "surfaces must not overlap.",
                pair=label,
                depth_mm=round(depth, 3),
            )


def _interpenetration_depth(
    first: AnatomicalMesh, second: AnatomicalMesh
) -> float | None:
    """Maximum depth by which ``second``'s vertices lie inside ``first``.

    Uses signed distance rather than a boolean intersection: booleans on
    marching-cubes meshes are slow and fragile, whereas the maximum penetration
    depth is both cheaper and more informative than a yes/no answer.
    """
    if not first.is_watertight:
        return None
    try:
        signed = first.signed_distance_to(second.vertices)
    except Exception as exc:  # noqa: BLE001 - proximity queries can fail on odd meshes
        log.debug(
            "interpenetration query failed",
            fields={"error": str(exc), "a": first.structure.value},
        )
        return None
    inside = signed[signed > 0]
    return float(inside.max()) if inside.size else 0.0


def _check_bone_stock(report: SimulationReport) -> None:
    """Flag implausible resection volumes.

    An unexpectedly large resection usually means a cut plane is misplaced rather
    than that a genuinely huge resection was intended, and it is worth catching
    before the plan reaches an operating theatre.
    """
    for structure, volume in report.resected_volume_mm3.items():
        retained = next(
            (
                m.volume_mm3
                for s, m in report.resected_meshes.items()
                if s.value == structure
            ),
            0.0,
        )
        total = volume + retained
        if total <= 0:
            continue
        fraction = volume / total
        if fraction > 0.35:
            report.add(
                "excessive_resection",
                Severity.FAILURE,
                f"the plan removes {fraction * 100:.1f}% of the visible "
                f"{structure} ({volume:.0f} mm³). A primary knee replacement removes a "
                "small fraction of the bone in the field of view; a cut plane is "
                "almost certainly misplaced.",
                structure=structure,
                resected_fraction=round(fraction, 4),
            )
        elif fraction > 0.22:
            report.add(
                "large_resection",
                Severity.WARNING,
                f"the plan removes {fraction * 100:.1f}% of the visible {structure}. "
                "Confirm the resection depths.",
                structure=structure,
                resected_fraction=round(fraction, 4),
            )
