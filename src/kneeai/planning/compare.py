"""Plan comparison (PRD §27).

PRD §27 is emphatic: *the system should not hide alternatives.* A surgeon
comparing Plan A against Plan B needs to see where they differ and what each
difference costs, not a single recommendation with the reasoning discarded.

So comparison here is field-by-field and symmetric. It reports what differs, by how
much, and whether the difference is clinically meaningful — using per-field
thresholds, because 0.3° of tibial slope and 3 mm of resection depth are not
comparable magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from kneeai.core.errors import ValidationError
from kneeai.logging import get_logger
from kneeai.planning.plan import CutName, SurgicalPlan

__all__ = [
    "FieldDifference",
    "PlanComparison",
    "compare_plans",
    "compare_many",
    "SIGNIFICANCE_THRESHOLDS",
]

log = get_logger("planning.compare")

#: Difference below which a field is treated as equivalent between two plans.
#:
#: These are "would this change what the surgeon does?" thresholds, not measurement
#: precision. A 0.5° difference in planned tibial slope is below the precision of
#: any cutting guide; a 1 mm difference in resection depth is not.
SIGNIFICANCE_THRESHOLDS: dict[str, float] = {
    "residual_hka_deg": 1.0,
    "tibial_coronal_deg": 1.0,
    "tibial_slope_deg": 1.0,
    "femoral_coronal_deg": 1.0,
    "femoral_rotation_deg": 1.0,
    "femoral_flexion_deg": 1.0,
    "joint_line_change_mm": 1.0,
    "extension_gap_medial_mm": 1.0,
    "extension_gap_lateral_mm": 1.0,
    "flexion_gap_medial_mm": 1.0,
    "flexion_gap_lateral_mm": 1.0,
    "coronal_gap_imbalance_mm": 1.0,
    "flexion_extension_mismatch_mm": 1.0,
    "medial_resection_mm": 1.0,
    "lateral_resection_mm": 1.0,
}

#: Higher is better for none of the compared fields — they are all "closer to zero
#: is better" except gaps, which target zero too. So a single rule applies: the plan
#: with the smaller magnitude is preferable, which is why comparison reports
#: magnitudes rather than declaring a winner per field.
_TARGET_ZERO_FIELDS = frozenset(SIGNIFICANCE_THRESHOLDS)


@dataclass(frozen=True)
class FieldDifference:
    """One field's value in each plan, and whether the difference matters."""

    field_name: str
    unit: str
    value_a: float | str | None
    value_b: float | str | None
    delta: float | None = None
    threshold: float | None = None

    @property
    def significant(self) -> bool:
        if self.delta is None:
            return self.value_a != self.value_b
        if self.threshold is None:
            return abs(self.delta) > 1e-9
        return abs(self.delta) > self.threshold

    @property
    def closer_to_target(self) -> str | None:
        """Which plan is closer to zero, for target-zero fields."""
        if (
            self.field_name not in _TARGET_ZERO_FIELDS
            or not isinstance(self.value_a, float)
            or not isinstance(self.value_b, float)
        ):
            return None
        if abs(abs(self.value_a) - abs(self.value_b)) < 1e-9:
            return None
        return "A" if abs(self.value_a) < abs(self.value_b) else "B"

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "unit": self.unit,
            "a": (
                round(self.value_a, 3) if isinstance(self.value_a, float) else self.value_a
            ),
            "b": (
                round(self.value_b, 3) if isinstance(self.value_b, float) else self.value_b
            ),
            "delta": round(self.delta, 3) if self.delta is not None else None,
            "significant": self.significant,
            "closer_to_target": self.closer_to_target,
        }


@dataclass
class PlanComparison:
    """A symmetric comparison between two plans."""

    label_a: str
    label_b: str
    differences: list[FieldDifference] = field(default_factory=list)
    cut_deltas: dict[str, float] = field(default_factory=dict)

    #: Differences in implant selection.
    implant_differences: list[str] = field(default_factory=list)

    #: Violations and warnings unique to each plan.
    only_a_violations: list[str] = field(default_factory=list)
    only_b_violations: list[str] = field(default_factory=list)
    only_a_warnings: list[str] = field(default_factory=list)
    only_b_warnings: list[str] = field(default_factory=list)

    @property
    def significant_differences(self) -> list[FieldDifference]:
        return [d for d in self.differences if d.significant]

    @property
    def equivalent(self) -> bool:
        """Whether the two plans are clinically equivalent."""
        return (
            not self.significant_differences
            and not self.implant_differences
            and not self.only_a_violations
            and not self.only_b_violations
        )

    def summary(self) -> str:
        lines = [f"Comparison: {self.label_a} vs {self.label_b}"]
        if self.equivalent:
            lines.append(
                "  The two plans are clinically equivalent: no field differs by more "
                "than its significance threshold, and both select the same implant."
            )
            return "\n".join(lines)

        if self.implant_differences:
            lines.append("  implant:")
            lines.extend(f"    {d}" for d in self.implant_differences)

        significant = self.significant_differences
        if significant:
            lines.append(
                f"  {'field':<32}{'A':>10}{'B':>10}{'delta':>10}  favours"
            )
            for difference in significant:
                a = (
                    f"{difference.value_a:.2f}"
                    if isinstance(difference.value_a, float)
                    else str(difference.value_a)[:10]
                )
                b = (
                    f"{difference.value_b:.2f}"
                    if isinstance(difference.value_b, float)
                    else str(difference.value_b)[:10]
                )
                delta = (
                    f"{difference.delta:+.2f}" if difference.delta is not None else "-"
                )
                favours = difference.closer_to_target or "-"
                lines.append(
                    f"  {difference.field_name:<32}{a:>10}{b:>10}{delta:>10}  {favours}"
                )

        if self.cut_deltas:
            lines.append("  resection depth differences (B minus A):")
            for name, delta in sorted(self.cut_deltas.items()):
                lines.append(f"    {name:<24}{delta:+8.2f} mm")

        for label, violations in (
            (f"only in {self.label_a}", self.only_a_violations),
            (f"only in {self.label_b}", self.only_b_violations),
        ):
            if violations:
                lines.append(f"  violations {label}:")
                lines.extend(f"    - {v}" for v in violations)

        for label, warnings in (
            (f"only in {self.label_a}", self.only_a_warnings),
            (f"only in {self.label_b}", self.only_b_warnings),
        ):
            if warnings:
                lines.append(f"  warnings {label}:")
                lines.extend(f"    - {w}" for w in warnings)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "equivalent": self.equivalent,
            "differences": [d.to_dict() for d in self.differences],
            "significant_differences": [
                d.to_dict() for d in self.significant_differences
            ],
            "cut_deltas": {k: round(v, 3) for k, v in self.cut_deltas.items()},
            "implant_differences": list(self.implant_differences),
            "only_a_violations": list(self.only_a_violations),
            "only_b_violations": list(self.only_b_violations),
            "only_a_warnings": list(self.only_a_warnings),
            "only_b_warnings": list(self.only_b_warnings),
        }


_UNITS: dict[str, str] = {
    **{k: "deg" for k in SIGNIFICANCE_THRESHOLDS if k.endswith("_deg")},
    **{k: "mm" for k in SIGNIFICANCE_THRESHOLDS if k.endswith("_mm")},
}


def compare_plans(plan_a: SurgicalPlan, plan_b: SurgicalPlan) -> PlanComparison:
    """Compare two plans field by field."""
    if plan_a.side is not plan_b.side:
        raise ValidationError(
            f"cannot compare a {plan_a.side.value} plan with a {plan_b.side.value} plan"
        )

    label_a = plan_a.label or f"{plan_a.plan_id[:12]} v{plan_a.version}"
    label_b = plan_b.label or f"{plan_b.plan_id[:12]} v{plan_b.version}"
    comparison = PlanComparison(label_a=label_a, label_b=label_b)

    predicted_a = plan_a.predicted.to_dict()
    predicted_b = plan_b.predicted.to_dict()

    for name in SIGNIFICANCE_THRESHOLDS:
        a_value = predicted_a.get(name)
        b_value = predicted_b.get(name)
        if a_value is None and b_value is None:
            continue
        delta = (
            float(b_value) - float(a_value)
            if isinstance(a_value, (int, float)) and isinstance(b_value, (int, float))
            else None
        )
        comparison.differences.append(
            FieldDifference(
                field_name=name,
                unit=_UNITS.get(name, ""),
                value_a=float(a_value) if isinstance(a_value, (int, float)) else a_value,
                value_b=float(b_value) if isinstance(b_value, (int, float)) else b_value,
                delta=delta,
                threshold=SIGNIFICANCE_THRESHOLDS[name],
            )
        )

    # --- strategy ---
    if plan_a.strategy is not plan_b.strategy:
        comparison.differences.append(
            FieldDifference(
                field_name="strategy",
                unit="",
                value_a=plan_a.strategy.display_name,
                value_b=plan_b.strategy.display_name,
            )
        )

    # --- cuts ---
    for name in set(plan_a.cuts) | set(plan_b.cuts):
        a_cut = plan_a.cuts.get(name)
        b_cut = plan_b.cuts.get(name)
        if a_cut is None:
            comparison.implant_differences.append(
                f"{name.display_name} cut is present only in {label_b}"
            )
            continue
        if b_cut is None:
            comparison.implant_differences.append(
                f"{name.display_name} cut is present only in {label_a}"
            )
            continue
        delta = b_cut.depth_mm - a_cut.depth_mm
        if abs(delta) > 1e-6:
            comparison.cut_deltas[name.value] = delta

    # --- implant ---
    if plan_a.configuration_label != plan_b.configuration_label:
        comparison.implant_differences.append(
            f"implant: {plan_a.configuration_label or '<none>'} vs "
            f"{plan_b.configuration_label or '<none>'}"
        )
    if plan_a.femoral_size != plan_b.femoral_size:
        comparison.implant_differences.append(
            f"femoral size: {plan_a.femoral_size or '?'} vs {plan_b.femoral_size or '?'}"
        )
    if plan_a.tibial_size != plan_b.tibial_size:
        comparison.implant_differences.append(
            f"tibial size: {plan_a.tibial_size or '?'} vs {plan_b.tibial_size or '?'}"
        )
    if plan_a.insert_thickness_mm != plan_b.insert_thickness_mm:
        comparison.implant_differences.append(
            f"insert thickness: {plan_a.insert_thickness_mm} vs "
            f"{plan_b.insert_thickness_mm} mm"
        )

    # --- violations and warnings ---
    a_violations, b_violations = set(plan_a.violations), set(plan_b.violations)
    comparison.only_a_violations = sorted(a_violations - b_violations)
    comparison.only_b_violations = sorted(b_violations - a_violations)
    a_warnings, b_warnings = set(plan_a.warnings), set(plan_b.warnings)
    comparison.only_a_warnings = sorted(a_warnings - b_warnings)
    comparison.only_b_warnings = sorted(b_warnings - a_warnings)

    log.info(
        "plans compared",
        fields={
            "a": label_a,
            "b": label_b,
            "significant_differences": len(comparison.significant_differences),
            "equivalent": comparison.equivalent,
        },
    )
    return comparison


def compare_many(plans: Sequence[SurgicalPlan]) -> dict[str, Any]:
    """Build a comparison table across several plans (PRD §27: Plan A/B/C).

    Returns a table suitable for direct rendering: one row per predicted field, one
    column per plan, plus the per-plan violation and warning counts.
    """
    if len(plans) < 2:
        raise ValidationError("comparing plans requires at least two plans")
    sides = {p.side for p in plans}
    if len(sides) > 1:
        raise ValidationError(
            f"cannot compare plans from different knees: {sorted(s.value for s in sides)}"
        )

    labels = [
        p.label or f"{p.plan_id[:12]} v{p.version}" for p in plans
    ]
    predictions = [p.predicted.to_dict() for p in plans]

    rows: list[dict[str, Any]] = []
    for name in SIGNIFICANCE_THRESHOLDS:
        values = [pred.get(name) for pred in predictions]
        if all(v is None for v in values):
            continue
        numeric = [v for v in values if isinstance(v, (int, float))]
        spread = (max(numeric) - min(numeric)) if len(numeric) > 1 else 0.0
        rows.append(
            {
                "field": name,
                "unit": _UNITS.get(name, ""),
                "values": [
                    round(float(v), 3) if isinstance(v, (int, float)) else None
                    for v in values
                ],
                "spread": round(float(spread), 3),
                "significant": spread > SIGNIFICANCE_THRESHOLDS[name],
                "best_index": (
                    int(np.argmin([abs(float(v)) for v in numeric]))
                    if len(numeric) == len(values) and numeric
                    else None
                ),
            }
        )

    return {
        "labels": labels,
        "plans": [
            {
                "label": label,
                "plan_id": plan.plan_id,
                "version": plan.version,
                "strategy": plan.strategy.display_name,
                "implant": plan.configuration_label,
                "femoral_size": plan.femoral_size,
                "tibial_size": plan.tibial_size,
                "insert_thickness_mm": plan.insert_thickness_mm,
                "n_violations": len(plan.violations),
                "n_warnings": len(plan.warnings),
                "n_clamps": len(plan.constraint_notes),
                "approvable": plan.approvable,
                "approved": plan.approved,
            }
            for label, plan in zip(labels, plans)
        ],
        "rows": rows,
        "note": (
            "Every candidate is shown. The platform does not suppress alternatives "
            "(PRD §27); 'best_index' marks the plan closest to the target for each "
            "field and is not a recommendation."
        ),
    }
