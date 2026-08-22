"""Deterministic plan optimisation (PRD §40).

PRD §40 requires the optimiser to be **deterministic and inspectable**. That rules
out a stochastic search: the same inputs must produce the same plan, and a surgeon
must be able to see what was traded against what.

So this is a bounded grid search over a small number of clinically meaningful
parameters, with an explicit, weighted objective. Every candidate evaluated is
recorded, so the optimiser's decision can be audited — and the objective's terms
are reported alongside the result rather than collapsed into one number.

What it does *not* do: it does not search for the best implant (that is the
matcher's job), it does not adjust anything outside the strategy's envelope, and it
never returns a plan with a constraint violation in preference to one without.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np

from kneeai.core.errors import PlanningError
from kneeai.core.provenance import Provenance
from kneeai.logging import get_logger
from kneeai.planning.engine import PlanningInputs, plan_resections
from kneeai.planning.plan import SurgicalPlan
from kneeai.planning.strategy import ResectionTargets

__all__ = [
    "ObjectiveWeights",
    "OptimizationCandidate",
    "OptimizationResult",
    "SearchSpace",
    "optimize_plan",
]

log = get_logger("planning.optimizer")


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for the planning objective.

    Residual alignment is weighted highest because it is the primary determinant of
    load distribution and the parameter with the strongest outcome evidence. Gap
    balance follows: an imbalanced knee is unstable regardless of alignment. Joint
    line and bone preservation are secondary but real — a proximalised joint line
    changes patellar tracking, and unnecessary resection removes bone that a
    revision would need.
    """

    residual_alignment: float = 4.0
    coronal_gap_balance: float = 3.0
    flexion_extension_balance: float = 2.0
    joint_line_preservation: float = 1.5
    bone_preservation: float = 1.0

    #: Penalty added per strategy clamp. Small: a clamp is the strategy working as
    #: intended, not a defect — but an unclamped plan is preferable where one exists.
    clamp_penalty: float = 0.5

    #: Penalty per advisory warning.
    warning_penalty: float = 1.0

    def to_dict(self) -> dict[str, float]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class SearchSpace:
    """Parameters the optimiser may vary, and the values it may try.

    Deliberately small. Each axis is a parameter a surgeon would actually adjust,
    expressed as an offset from the strategy's target, and the grid is coarse enough
    that the whole search is a few dozen plan evaluations.
    """

    #: Offsets applied to the tibial coronal target, in degrees.
    tibial_varus_offsets: tuple[float, ...] = (-1.0, 0.0, 1.0)

    #: Offsets applied to the tibial slope target, in degrees.
    tibial_slope_offsets: tuple[float, ...] = (-2.0, 0.0, 2.0)

    #: Offsets applied to the tibial resection depth, in millimetres.
    tibial_resection_offsets: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0)

    #: Offsets applied to the distal femoral resection depth, in millimetres.
    distal_femoral_offsets: tuple[float, ...] = (-1.0, 0.0, 1.0)

    #: Offsets applied to the femoral coronal target, in degrees.
    femoral_valgus_offsets: tuple[float, ...] = (0.0,)

    @property
    def size(self) -> int:
        return (
            len(self.tibial_varus_offsets)
            * len(self.tibial_slope_offsets)
            * len(self.tibial_resection_offsets)
            * len(self.distal_femoral_offsets)
            * len(self.femoral_valgus_offsets)
        )

    def combinations(self) -> Iterable[tuple[float, float, float, float, float]]:
        return itertools.product(
            self.tibial_varus_offsets,
            self.tibial_slope_offsets,
            self.tibial_resection_offsets,
            self.distal_femoral_offsets,
            self.femoral_valgus_offsets,
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {k: list(v) for k, v in asdict(self).items()}


@dataclass
class OptimizationCandidate:
    """One evaluated plan, with its objective breakdown."""

    offsets: dict[str, float]
    cost: float
    terms: dict[str, float]
    acceptable: bool
    n_violations: int
    n_warnings: int
    n_clamps: int
    residual_hka_deg: float | None
    plan: SurgicalPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "offsets": {k: round(v, 3) for k, v in self.offsets.items()},
            "cost": round(self.cost, 4),
            "terms": {k: round(v, 4) for k, v in self.terms.items()},
            "acceptable": self.acceptable,
            "n_violations": self.n_violations,
            "n_warnings": self.n_warnings,
            "n_clamps": self.n_clamps,
            "residual_hka_deg": (
                round(self.residual_hka_deg, 3)
                if self.residual_hka_deg is not None
                else None
            ),
        }


@dataclass
class OptimizationResult:
    """The chosen plan plus the full search trace."""

    plan: SurgicalPlan
    baseline: SurgicalPlan
    chosen: OptimizationCandidate
    baseline_candidate: OptimizationCandidate
    candidates: list[OptimizationCandidate] = field(default_factory=list)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    provenance: Provenance | None = None

    @property
    def improved(self) -> bool:
        return self.chosen.cost < self.baseline_candidate.cost - 1e-9

    def explanation(self) -> list[str]:
        """Why the optimiser chose this plan over the strategy defaults."""
        lines = [
            f"Evaluated {len(self.candidates)} candidate plans over "
            f"{self.search_space.size} parameter combinations.",
        ]
        if not self.improved:
            lines.append(
                "The strategy defaults already minimise the objective; no adjustment "
                "was applied."
            )
        else:
            changes = [
                f"{name.replace('_', ' ')} {value:+.1f}"
                for name, value in self.chosen.offsets.items()
                if abs(value) > 1e-9
            ]
            lines.append(
                "Adjusted from the strategy defaults: "
                + (", ".join(changes) if changes else "no change")
            )
            lines.append(
                f"Objective cost {self.baseline_candidate.cost:.3f} -> "
                f"{self.chosen.cost:.3f}"
            )
        lines.append("Objective breakdown for the chosen plan:")
        for name, value in sorted(self.chosen.terms.items()):
            lines.append(f"  {name.replace('_', ' '):<32} {value:8.4f}")
        if self.chosen.n_violations:
            lines.append(
                f"WARNING: the chosen plan still carries {self.chosen.n_violations} "
                "constraint violation(s). No candidate in the search space satisfied "
                "every constraint — the implant selection or the strategy may need to "
                "change."
            )
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen.to_dict(),
            "baseline": self.baseline_candidate.to_dict(),
            "improved": self.improved,
            "weights": self.weights.to_dict(),
            "search_space": self.search_space.to_dict(),
            "n_candidates": len(self.candidates),
            "explanation": self.explanation(),
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------------------


def _evaluate(
    plan: SurgicalPlan,
    offsets: dict[str, float],
    weights: ObjectiveWeights,
    targets: ResectionTargets,
) -> OptimizationCandidate:
    """Score a plan. Lower cost is better."""
    predicted = plan.predicted
    terms: dict[str, float] = {}

    # --- residual alignment: squared, so 4° is much worse than two knees at 2° ---
    if predicted.residual_hka_deg is not None:
        terms["residual_alignment"] = weights.residual_alignment * (
            predicted.residual_hka_deg**2
        )
    else:
        # No alignment prediction is itself a cost: a plan whose alignment cannot be
        # predicted must not win over one whose can.
        terms["residual_alignment"] = weights.residual_alignment * 9.0

    # --- coronal gap balance ---
    imbalance = predicted.coronal_gap_imbalance_mm
    terms["coronal_gap_balance"] = (
        weights.coronal_gap_balance * (imbalance**2)
        if imbalance is not None
        else weights.coronal_gap_balance * 4.0
    )

    # --- flexion / extension balance ---
    mismatch = predicted.flexion_extension_mismatch_mm
    terms["flexion_extension_balance"] = (
        weights.flexion_extension_balance * (mismatch**2)
        if mismatch is not None
        else weights.flexion_extension_balance * 4.0
    )

    # --- joint line ---
    joint_line = predicted.joint_line_change_mm
    terms["joint_line_preservation"] = (
        weights.joint_line_preservation * (joint_line**2)
        if joint_line is not None
        else 0.0
    )

    # --- bone preservation: total resected depth beyond the strategy targets ---
    excess = max(
        0.0,
        sum(c.depth_mm for c in plan.cuts.values())
        - (targets.tibial_resection_mm + targets.distal_femoral_resection_mm),
    )
    terms["bone_preservation"] = weights.bone_preservation * excess

    # --- soft penalties ---
    terms["clamps"] = weights.clamp_penalty * len(plan.constraint_notes)
    terms["warnings"] = weights.warning_penalty * len(plan.warnings)

    # --- violations dominate ---
    # A large fixed cost per violation rather than a hard filter, so the optimiser
    # still returns the least-bad plan when no candidate is fully acceptable — and
    # says so — instead of failing.
    terms["violations"] = 1000.0 * len(plan.violations)

    return OptimizationCandidate(
        offsets=dict(offsets),
        cost=float(sum(terms.values())),
        terms=terms,
        acceptable=not plan.violations,
        n_violations=len(plan.violations),
        n_warnings=len(plan.warnings),
        n_clamps=len(plan.constraint_notes),
        residual_hka_deg=predicted.residual_hka_deg,
        plan=plan,
    )


def optimize_plan(
    inputs: PlanningInputs,
    *,
    weights: ObjectiveWeights | None = None,
    search_space: SearchSpace | None = None,
    keep_all_plans: bool = False,
) -> OptimizationResult:
    """Search a bounded grid of resection adjustments for the lowest-cost plan.

    Args:
        keep_all_plans: retain every candidate's full plan object. Off by default
            because a full grid holds dozens of plans; the candidate summaries are
            enough for auditing.
    """
    weights = weights or ObjectiveWeights()
    space = search_space or SearchSpace()
    base_targets = inputs.targets

    baseline_plan = plan_resections(inputs)
    baseline_offsets = {
        "tibial_varus_deg": 0.0,
        "tibial_slope_deg": 0.0,
        "tibial_resection_mm": 0.0,
        "distal_femoral_resection_mm": 0.0,
        "femoral_valgus_deg": 0.0,
    }
    baseline_candidate = _evaluate(
        baseline_plan, baseline_offsets, weights, base_targets
    )

    candidates: list[OptimizationCandidate] = []
    best = baseline_candidate
    best_plan = baseline_plan

    for varus, slope, tibial_depth, femoral_depth, valgus in space.combinations():
        offsets = {
            "tibial_varus_deg": varus,
            "tibial_slope_deg": slope,
            "tibial_resection_mm": tibial_depth,
            "distal_femoral_resection_mm": femoral_depth,
            "femoral_valgus_deg": valgus,
        }
        if all(abs(v) < 1e-9 for v in offsets.values()):
            candidate = replace(baseline_candidate, offsets=dict(offsets))
            candidates.append(
                candidate if keep_all_plans else replace(candidate, plan=None)
            )
            continue

        adjusted = base_targets.with_overrides(
            tibial_varus_deg=base_targets.tibial_varus_deg + varus,
            tibial_slope_deg=base_targets.tibial_slope_deg + slope,
            tibial_resection_mm=base_targets.tibial_resection_mm + tibial_depth,
            distal_femoral_resection_mm=(
                base_targets.distal_femoral_resection_mm + femoral_depth
            ),
            femoral_valgus_deg=base_targets.femoral_valgus_deg + valgus,
        )
        try:
            plan = plan_resections(replace(inputs, targets=adjusted))
        except PlanningError as exc:
            log.debug(
                "candidate rejected",
                fields={"offsets": offsets, "reason": str(exc)},
            )
            continue

        candidate = _evaluate(plan, offsets, weights, base_targets)
        if candidate.cost < best.cost:
            best = candidate
            best_plan = plan
        candidates.append(candidate if keep_all_plans else replace(candidate, plan=None))

    if not candidates:  # pragma: no cover - defensive
        raise PlanningError("no candidate plans could be evaluated")

    chosen = replace(best, plan=None)
    # Distinguish the optimised plan from the plan it was derived from, so a
    # side-by-side comparison does not show two columns with the same heading.
    base_label = inputs.label or best_plan.label
    if base_label and best_plan is not baseline_plan:
        best_plan.label = f"{base_label} — optimised"
    elif base_label:
        best_plan.label = f"{base_label} — optimised (no change)"
    best_plan.provenance = Provenance.for_component(
        "planning.optimizer",
        origin=best_plan.origin,
        parameters={
            "weights": weights.to_dict(),
            "search_space": space.to_dict(),
            "n_candidates_evaluated": len(candidates),
            "chosen_offsets": chosen.offsets,
            "chosen_cost": round(chosen.cost, 4),
            "baseline_cost": round(baseline_candidate.cost, 4),
            "objective_terms": {k: round(v, 4) for k, v in chosen.terms.items()},
        },
        notes=(
            "Deterministic bounded grid search. Every candidate and its objective "
            "breakdown is retained for audit (PRD §40)."
        ),
    )

    result = OptimizationResult(
        plan=best_plan,
        baseline=baseline_plan,
        chosen=chosen,
        baseline_candidate=replace(baseline_candidate, plan=None),
        candidates=candidates,
        weights=weights,
        search_space=space,
        provenance=best_plan.provenance,
    )

    log.info(
        "plan optimised",
        fields={
            "candidates": len(candidates),
            "baseline_cost": round(baseline_candidate.cost, 3),
            "chosen_cost": round(chosen.cost, 3),
            "improved": result.improved,
            "offsets": {k: v for k, v in chosen.offsets.items() if abs(v) > 1e-9},
            "residual_hka_deg": chosen.residual_hka_deg,
            "violations": chosen.n_violations,
        },
    )
    return result
