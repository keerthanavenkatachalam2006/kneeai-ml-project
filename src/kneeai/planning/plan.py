"""Surgical plan records (PRD §25, §26, §93).

A plan is data, not behaviour. It holds the resections, the chosen implant, the
predicted outcome, and the full trace of how it was produced — and it is the thing
the surgeon edits, compares, approves and (eventually) executes.

Two invariants are enforced here rather than by convention:

**A plan is immutable once locked.** :meth:`SurgicalPlan.lock` returns a locked
copy; every mutating method on a locked plan raises. Approval locks the plan, and
re-opening it (PRD §8) creates a new version rather than editing the approved one.
So the artefact a surgeon approved is exactly the artefact that can be exported.

**Approval carries identity.** :meth:`SurgicalPlan.approve` requires a surgeon
identifier and records the timestamp. A plan cannot become approved by having a
flag set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import Side, Structure
from kneeai.core.errors import PlanningError, ValidationError
from kneeai.core.frames import Frame, Transform
from kneeai.core.geometry import Plane
from kneeai.core.provenance import Origin, Provenance, new_id, now_utc
from kneeai.core.units import Quantity, Unit
from kneeai.planning.strategy import AlignmentStrategy, ResectionTargets

__all__ = [
    "CutName",
    "ResectionCut",
    "ComponentPlacement",
    "PredictedOutcome",
    "SurgicalPlan",
    "PlanEdit",
]


class CutName(str, Enum):
    """The bone cuts of a primary TKA.

    The femur takes five planar cuts; the tibia one. Naming them explicitly means a
    plan can be validated for completeness rather than trusting that a list of
    planes is the right list.
    """

    TIBIAL = "tibial"
    DISTAL_FEMORAL = "distal_femoral"
    ANTERIOR_FEMORAL = "anterior_femoral"
    POSTERIOR_FEMORAL = "posterior_femoral"
    ANTERIOR_CHAMFER = "anterior_chamfer"
    POSTERIOR_CHAMFER = "posterior_chamfer"
    PATELLAR = "patellar"

    @property
    def bone(self) -> Structure:
        if self is CutName.TIBIAL:
            return Structure.TIBIA
        if self is CutName.PATELLAR:
            return Structure.PATELLA
        return Structure.FEMUR

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


#: Cuts required for a complete femoral preparation.
FEMORAL_CUTS: tuple[CutName, ...] = (
    CutName.DISTAL_FEMORAL,
    CutName.ANTERIOR_FEMORAL,
    CutName.POSTERIOR_FEMORAL,
    CutName.ANTERIOR_CHAMFER,
    CutName.POSTERIOR_CHAMFER,
)


@dataclass(frozen=True)
class ResectionCut:
    """One planar bone cut.

    The plane normal points **into the bone that is retained**, matching the
    convention in :mod:`kneeai.core.geometry`. That makes the sign of a signed
    distance immediately meaningful: negative means the point is in the fragment
    being removed.
    """

    name: CutName
    plane: Plane

    #: Resection depth measured from the reference landmark, in millimetres.
    depth_mm: float = 0.0

    #: What the depth was measured from, in words.
    reference: str = ""

    #: Angular parameters that produced this plane, for display and for editing.
    #: Keys are free-form but conventionally include ``coronal_deg``,
    #: ``sagittal_deg`` and ``rotation_deg``.
    angles_deg: Mapping[str, float] = field(default_factory=dict)

    #: Predicted volume of bone removed by this cut, when computed.
    resected_volume_mm3: float | None = None

    notes: str = ""

    def __post_init__(self) -> None:
        if self.plane.frame is Frame.VOXEL:
            raise ValidationError(
                f"{self.name.value} cut plane is in voxel space; resection planes "
                "must be in a physical frame"
            )

    @property
    def structure(self) -> Structure:
        return self.name.bone

    def offset(self, delta_mm: float) -> ResectionCut:
        """Move the cut along its normal. Positive removes more bone."""
        return replace(
            self,
            plane=self.plane.offset_by(delta_mm),
            depth_mm=self.depth_mm + delta_mm,
            notes=(f"{self.notes} adjusted {delta_mm:+.1f} mm".strip()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "structure": self.structure.value,
            "plane": self.plane.to_dict(),
            "depth_mm": round(self.depth_mm, 3),
            "reference": self.reference,
            "angles_deg": {k: round(float(v), 3) for k, v in self.angles_deg.items()},
            "resected_volume_mm3": (
                round(self.resected_volume_mm3, 1)
                if self.resected_volume_mm3 is not None
                else None
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResectionCut:
        return cls(
            name=CutName(payload["name"]),
            plane=Plane.from_dict(payload["plane"]),
            depth_mm=float(payload.get("depth_mm", 0.0)),
            reference=str(payload.get("reference", "")),
            angles_deg=dict(payload.get("angles_deg", {})),
            resected_volume_mm3=payload.get("resected_volume_mm3"),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class ComponentPlacement:
    """Where a component sits, as a transform plus the parameters that produced it."""

    component_id: str
    structure: Structure

    #: IMPLANT -> IMAGE (or PLANNING) transform.
    transform: Transform

    #: Named placement parameters in the patient frame, for display and editing:
    #: ``coronal_deg``, ``sagittal_deg``, ``rotation_deg``,
    #: ``mediolateral_mm``, ``anteroposterior_mm``, ``proximodistal_mm``.
    parameters: Mapping[str, float] = field(default_factory=dict)

    size_label: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "structure": self.structure.value,
            "size_label": self.size_label,
            "transform": self.transform.to_dict(),
            "parameters": {k: round(float(v), 3) for k, v in self.parameters.items()},
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComponentPlacement:
        return cls(
            component_id=str(payload["component_id"]),
            structure=Structure(payload["structure"]),
            transform=Transform.from_dict(payload["transform"]),
            parameters=dict(payload.get("parameters", {})),
            size_label=str(payload.get("size_label", "")),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class PredictedOutcome:
    """What the plan is predicted to achieve.

    These are geometric predictions from the planned cuts, not clinical outcome
    predictions. Distinguishing the two matters: "this plan yields 1.2° residual
    varus" is a computation, whereas "this patient will be satisfied" is not
    something the platform claims (PRD §17).
    """

    residual_hka_deg: float | None = None
    tibial_coronal_deg: float | None = None
    tibial_slope_deg: float | None = None
    femoral_coronal_deg: float | None = None
    femoral_rotation_deg: float | None = None
    femoral_flexion_deg: float | None = None
    joint_line_change_mm: float | None = None
    extension_gap_medial_mm: float | None = None
    extension_gap_lateral_mm: float | None = None
    flexion_gap_medial_mm: float | None = None
    flexion_gap_lateral_mm: float | None = None
    medial_resection_mm: float | None = None
    lateral_resection_mm: float | None = None

    @property
    def coronal_gap_imbalance_mm(self) -> float | None:
        if self.extension_gap_medial_mm is None or self.extension_gap_lateral_mm is None:
            return None
        return self.extension_gap_medial_mm - self.extension_gap_lateral_mm

    @property
    def flexion_extension_mismatch_mm(self) -> float | None:
        """Mean flexion gap minus mean extension gap.

        A positive value means the flexion gap is looser than the extension gap,
        which manifests as mid-flexion instability; negative means flexion is tight
        and restricts range of motion. Both matter, and they are different problems,
        so the sign is preserved rather than reporting a magnitude.
        """
        gaps = (
            self.extension_gap_medial_mm,
            self.extension_gap_lateral_mm,
            self.flexion_gap_medial_mm,
            self.flexion_gap_lateral_mm,
        )
        if any(g is None for g in gaps):
            return None
        extension = (self.extension_gap_medial_mm + self.extension_gap_lateral_mm) / 2.0  # type: ignore[operator]
        flexion = (self.flexion_gap_medial_mm + self.flexion_gap_lateral_mm) / 2.0  # type: ignore[operator]
        return float(flexion - extension)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in asdict(self).items()
        }
        payload["coronal_gap_imbalance_mm"] = (
            round(self.coronal_gap_imbalance_mm, 3)
            if self.coronal_gap_imbalance_mm is not None
            else None
        )
        payload["flexion_extension_mismatch_mm"] = (
            round(self.flexion_extension_mismatch_mm, 3)
            if self.flexion_extension_mismatch_mm is not None
            else None
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PredictedOutcome:
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})


@dataclass(frozen=True)
class PlanEdit:
    """A recorded surgeon modification (PRD §32: AI vs surgeon-confirmed)."""

    at: datetime
    actor: str
    field_path: str
    previous: Any
    updated: Any
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "actor": self.actor,
            "field": self.field_path,
            "previous": self.previous,
            "updated": self.updated,
            "reason": self.reason,
        }


@dataclass
class SurgicalPlan:
    """A complete, versioned surgical plan."""

    plan_id: str = field(default_factory=lambda: new_id("plan"))
    case_id: str = ""
    side: Side = Side.UNKNOWN
    version: int = 1

    strategy: AlignmentStrategy = AlignmentStrategy.RESTRICTED_KINEMATIC
    targets: ResectionTargets | None = None

    cuts: dict[CutName, ResectionCut] = field(default_factory=dict)
    placements: dict[Structure, ComponentPlacement] = field(default_factory=dict)

    configuration_id: str = ""
    configuration_label: str = ""
    femoral_size: str = ""
    tibial_size: str = ""
    insert_thickness_mm: float | None = None

    predicted: PredictedOutcome = field(default_factory=PredictedOutcome)

    #: Fit metrics for the chosen configuration, carried forward from the matcher
    #: so the plan is self-contained for reporting.
    fit_summary: Mapping[str, Any] = field(default_factory=dict)

    #: Where the strategy envelope constrained the plan, in words.
    constraint_notes: list[str] = field(default_factory=list)

    #: Hard problems with the plan. A plan with violations must not be approved.
    violations: list[str] = field(default_factory=list)

    #: Advisory concerns that do not block approval.
    warnings: list[str] = field(default_factory=list)

    origin: Origin = Origin.DETERMINISTIC
    provenance: Provenance | None = None

    edits: list[PlanEdit] = field(default_factory=list)
    created_at: datetime = field(default_factory=now_utc)

    approved: bool = False
    approved_by: str = ""
    approved_at: datetime | None = None
    approval_note: str = ""

    locked: bool = False

    #: Free-text label so a surgeon can compare "Plan A" against "Plan B" (PRD §27).
    label: str = ""

    # ------------------------------------------------------------------
    # Mutation guards
    # ------------------------------------------------------------------
    def _require_unlocked(self, operation: str) -> None:
        if self.locked:
            raise PlanningError(
                f"cannot {operation}: plan {self.plan_id} (version {self.version}) is "
                "locked. Create a new version with revise() — an approved plan is the "
                "record of what was approved and must not change underneath it."
            )

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def add_cut(self, cut: ResectionCut) -> SurgicalPlan:
        self._require_unlocked("add a cut")
        self.cuts[cut.name] = cut
        return self

    def add_placement(self, placement: ComponentPlacement) -> SurgicalPlan:
        self._require_unlocked("add a component placement")
        self.placements[placement.structure] = placement
        return self

    def cut(self, name: CutName) -> ResectionCut:
        found = self.cuts.get(name)
        if found is None:
            raise PlanningError(
                f"plan {self.plan_id} has no {name.display_name} cut. Present: "
                f"{sorted(c.value for c in self.cuts)}"
            )
        return found

    def placement(self, structure: Structure) -> ComponentPlacement:
        found = self.placements.get(structure)
        if found is None:
            raise PlanningError(
                f"plan {self.plan_id} has no {structure.display_name} component "
                f"placement. Present: {sorted(s.value for s in self.placements)}"
            )
        return found

    @property
    def is_complete(self) -> bool:
        """Whether every cut a primary TKA requires is present."""
        return not self.missing_cuts()

    def missing_cuts(self) -> list[CutName]:
        required = [CutName.TIBIAL, *FEMORAL_CUTS]
        return [c for c in required if c not in self.cuts]

    @property
    def approvable(self) -> bool:
        return not self.violations and self.is_complete

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def record_edit(
        self,
        field_path: str,
        previous: Any,
        updated: Any,
        *,
        actor: str,
        reason: str = "",
    ) -> SurgicalPlan:
        """Log a surgeon modification and mark the plan surgeon-edited."""
        self._require_unlocked("edit")
        if not actor:
            raise ValidationError("plan edits must record who made them")
        self.edits.append(
            PlanEdit(
                at=now_utc(),
                actor=actor,
                field_path=field_path,
                previous=previous,
                updated=updated,
                reason=reason,
            )
        )
        # Origin becomes surgeon-edited so the report separates this from a purely
        # AI-derived plan (PRD §32).
        if self.origin in (Origin.AI, Origin.DETERMINISTIC):
            self.origin = Origin.SURGEON_EDITED_AI
        return self

    def adjust_cut(
        self,
        name: CutName,
        delta_mm: float,
        *,
        actor: str,
        reason: str = "",
    ) -> SurgicalPlan:
        """Move a cut along its normal, recording the edit."""
        self._require_unlocked("adjust a cut")
        existing = self.cut(name)
        updated = existing.offset(delta_mm)
        self.record_edit(
            f"cuts.{name.value}.depth_mm",
            round(existing.depth_mm, 3),
            round(updated.depth_mm, 3),
            actor=actor,
            reason=reason or f"cut moved {delta_mm:+.1f} mm",
        )
        self.cuts[name] = updated
        return self

    # ------------------------------------------------------------------
    # Versioning, approval, locking
    # ------------------------------------------------------------------
    def revise(self, *, actor: str, reason: str) -> SurgicalPlan:
        """Create an unlocked next version of this plan.

        Used when a surgeon wants to change an approved plan. The approved version
        is left intact; the new version starts unapproved with the approval history
        cleared, because an approval applies to a specific plan and does not carry
        forward.
        """
        if not actor or not reason:
            raise ValidationError("revising a plan requires an actor and a reason")
        clone = SurgicalPlan(
            case_id=self.case_id,
            side=self.side,
            version=self.version + 1,
            strategy=self.strategy,
            targets=self.targets,
            cuts=dict(self.cuts),
            placements=dict(self.placements),
            configuration_id=self.configuration_id,
            configuration_label=self.configuration_label,
            femoral_size=self.femoral_size,
            tibial_size=self.tibial_size,
            insert_thickness_mm=self.insert_thickness_mm,
            predicted=self.predicted,
            fit_summary=dict(self.fit_summary),
            constraint_notes=list(self.constraint_notes),
            violations=list(self.violations),
            warnings=list(self.warnings),
            origin=Origin.SURGEON_EDITED_AI,
            provenance=self.provenance,
            edits=list(self.edits),
            label=self.label,
        )
        clone.edits.append(
            PlanEdit(
                at=now_utc(),
                actor=actor,
                field_path="version",
                previous=self.version,
                updated=clone.version,
                reason=reason,
            )
        )
        return clone

    def approve(self, *, surgeon: str, note: str = "") -> SurgicalPlan:
        """Record surgeon approval and lock the plan.

        Refuses when the plan is incomplete or carries violations. An approval on a
        plan with an unresolved violation would make the violation part of the
        approved record, which is the opposite of what approval is for.
        """
        self._require_unlocked("approve")
        if not surgeon:
            raise ValidationError("approval requires the approving surgeon's identity")
        if self.violations:
            raise PlanningError(
                f"plan {self.plan_id} cannot be approved with unresolved "
                f"violation(s):\n  - " + "\n  - ".join(self.violations)
            )
        missing = self.missing_cuts()
        if missing:
            raise PlanningError(
                f"plan {self.plan_id} is incomplete; missing cut(s): "
                f"{[c.display_name for c in missing]}"
            )
        self.approved = True
        self.approved_by = surgeon
        self.approved_at = now_utc()
        self.approval_note = note
        self.locked = True
        return self

    def lock(self) -> SurgicalPlan:
        """Freeze the plan without approving it (e.g. before export for review)."""
        self.locked = True
        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "case_id": self.case_id,
            "side": self.side.value,
            "version": self.version,
            "label": self.label,
            "strategy": self.strategy.value,
            "targets": self.targets.to_dict() if self.targets else None,
            "cuts": [c.to_dict() for c in self.cuts.values()],
            "placements": [p.to_dict() for p in self.placements.values()],
            "configuration_id": self.configuration_id,
            "configuration_label": self.configuration_label,
            "femoral_size": self.femoral_size,
            "tibial_size": self.tibial_size,
            "insert_thickness_mm": self.insert_thickness_mm,
            "predicted": self.predicted.to_dict(),
            "fit_summary": dict(self.fit_summary),
            "constraint_notes": list(self.constraint_notes),
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "is_complete": self.is_complete,
            "approvable": self.approvable,
            "origin": self.origin.value,
            "created_at": self.created_at.isoformat(),
            "edits": [e.to_dict() for e in self.edits],
            "surgeon_approved": self.approved,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approval_note": self.approval_note,
            "locked": self.locked,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SurgicalPlan:
        plan = cls(
            plan_id=str(payload.get("plan_id", new_id("plan"))),
            case_id=str(payload.get("case_id", "")),
            side=Side(payload.get("side", "unknown")),
            version=int(payload.get("version", 1)),
            label=str(payload.get("label", "")),
            strategy=AlignmentStrategy(payload.get("strategy", "restricted_kinematic")),
            targets=(
                ResectionTargets.from_dict(payload["targets"])
                if payload.get("targets")
                else None
            ),
            configuration_id=str(payload.get("configuration_id", "")),
            configuration_label=str(payload.get("configuration_label", "")),
            femoral_size=str(payload.get("femoral_size", "")),
            tibial_size=str(payload.get("tibial_size", "")),
            insert_thickness_mm=payload.get("insert_thickness_mm"),
            predicted=PredictedOutcome.from_dict(payload.get("predicted", {})),
            fit_summary=dict(payload.get("fit_summary", {})),
            constraint_notes=list(payload.get("constraint_notes", [])),
            violations=list(payload.get("violations", [])),
            warnings=list(payload.get("warnings", [])),
            origin=Origin(payload.get("origin", "deterministic")),
            approved=bool(payload.get("surgeon_approved", False)),
            approved_by=str(payload.get("approved_by", "")),
            approval_note=str(payload.get("approval_note", "")),
            locked=bool(payload.get("locked", False)),
        )
        for item in payload.get("cuts", []):
            plan.cuts[CutName(item["name"])] = ResectionCut.from_dict(item)
        for item in payload.get("placements", []):
            placement = ComponentPlacement.from_dict(item)
            plan.placements[placement.structure] = placement
        if payload.get("created_at"):
            plan.created_at = datetime.fromisoformat(payload["created_at"])
        if payload.get("approved_at"):
            plan.approved_at = datetime.fromisoformat(payload["approved_at"])
        if payload.get("provenance"):
            plan.provenance = Provenance.from_dict(payload["provenance"])
        return plan

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"Surgical plan {self.label or self.plan_id} "
            f"(v{self.version}, {self.side.value} knee)",
            f"  strategy      : {self.strategy.display_name}",
            f"  implant       : {self.configuration_label or '<none selected>'}",
        ]
        if self.insert_thickness_mm is not None:
            lines.append(f"  insert        : {self.insert_thickness_mm:.0f} mm")
        lines.append("  resections:")
        for name in (CutName.TIBIAL, *FEMORAL_CUTS, CutName.PATELLAR):
            existing = self.cuts.get(name)
            if existing is None:
                continue
            angles = ", ".join(
                f"{k.replace('_', ' ')} {v:+.1f}°" for k, v in existing.angles_deg.items()
            )
            lines.append(
                f"    {name.display_name:<20} {existing.depth_mm:6.1f} mm"
                + (f"   [{angles}]" if angles else "")
                + (f"   ref: {existing.reference}" if existing.reference else "")
            )
        predicted = self.predicted
        lines.append("  predicted:")
        for label, value, unit in (
            ("residual HKA", predicted.residual_hka_deg, "°"),
            ("tibial coronal", predicted.tibial_coronal_deg, "°"),
            ("tibial slope", predicted.tibial_slope_deg, "°"),
            ("femoral coronal", predicted.femoral_coronal_deg, "°"),
            ("femoral rotation", predicted.femoral_rotation_deg, "°"),
            ("joint line change", predicted.joint_line_change_mm, " mm"),
            ("extension gap medial", predicted.extension_gap_medial_mm, " mm"),
            ("extension gap lateral", predicted.extension_gap_lateral_mm, " mm"),
            ("flexion-extension mismatch", predicted.flexion_extension_mismatch_mm, " mm"),
        ):
            if value is not None:
                lines.append(f"    {label:<28} {value:+7.2f}{unit}")
        if self.constraint_notes:
            lines.append("  strategy envelope applied:")
            lines.extend(f"    - {n}" for n in self.constraint_notes)
        if self.warnings:
            lines.append("  warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        if self.violations:
            lines.append("  VIOLATIONS (block approval):")
            lines.extend(f"    - {v}" for v in self.violations)
        lines.append(
            f"  status        : "
            + (
                f"approved by {self.approved_by} at "
                f"{self.approved_at.isoformat() if self.approved_at else '?'}"
                if self.approved
                else ("locked, awaiting approval" if self.locked else "draft")
            )
        )
        if self.edits:
            lines.append(f"  surgeon edits : {len(self.edits)}")
        return "\n".join(lines)
