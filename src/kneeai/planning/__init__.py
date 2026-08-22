"""Surgical planning: strategy -> resection planes -> editable, approvable plan.

PRD §25-§27 and §40.

    >>> from kneeai.planning import PlanningInputs, plan_resections, targets_for
    >>> from kneeai.planning import AlignmentStrategy, optimize_plan, compare_plans
    >>>
    >>> inputs = PlanningInputs(
    ...     meshes=meshes, landmarks=landmarks, frame=frame,
    ...     measurements=measurements, configuration=best_fit.configuration,
    ...     targets=targets_for(AlignmentStrategy.RESTRICTED_KINEMATIC),
    ...     fit=best_fit, label="Plan A",
    ... )
    >>> plan = plan_resections(inputs)
    >>> print(plan.summary())
    >>> result = optimize_plan(inputs)
    >>> print(compare_plans(plan, result.plan).summary())
    >>> result.plan.approve(surgeon="dr-x", note="agreed")

A plan is immutable once approved: :meth:`SurgicalPlan.approve` locks it, and
changing an approved plan requires :meth:`SurgicalPlan.revise`, which produces a
new unapproved version.
"""

from __future__ import annotations

from kneeai.planning.compare import (
    SIGNIFICANCE_THRESHOLDS,
    FieldDifference,
    PlanComparison,
    compare_many,
    compare_plans,
)
from kneeai.planning.engine import (
    DEFAULT_CHAMFER_OFFSET_MM,
    PlanningInputs,
    cut_plane_normal,
    plan_resections,
    projected_angles,
)
from kneeai.planning.optimizer import (
    ObjectiveWeights,
    OptimizationCandidate,
    OptimizationResult,
    SearchSpace,
    optimize_plan,
)
from kneeai.planning.plan import (
    FEMORAL_CUTS,
    ComponentPlacement,
    CutName,
    PlanEdit,
    PredictedOutcome,
    ResectionCut,
    SurgicalPlan,
)
from kneeai.planning.strategy import (
    ADJUSTED_MECHANICAL_TARGETS,
    KINEMATIC_TARGETS,
    MECHANICAL_TARGETS,
    RESTRICTED_KINEMATIC_TARGETS,
    AlignmentStrategy,
    ResectionTargets,
    SurgeonPreferences,
    targets_for,
)

__all__ = [
    # strategy
    "AlignmentStrategy",
    "ResectionTargets",
    "SurgeonPreferences",
    "targets_for",
    "MECHANICAL_TARGETS",
    "KINEMATIC_TARGETS",
    "RESTRICTED_KINEMATIC_TARGETS",
    "ADJUSTED_MECHANICAL_TARGETS",
    # plan records
    "CutName",
    "FEMORAL_CUTS",
    "ResectionCut",
    "ComponentPlacement",
    "PredictedOutcome",
    "PlanEdit",
    "SurgicalPlan",
    # engine
    "PlanningInputs",
    "plan_resections",
    "cut_plane_normal",
    "projected_angles",
    "DEFAULT_CHAMFER_OFFSET_MM",
    # optimiser
    "ObjectiveWeights",
    "SearchSpace",
    "OptimizationCandidate",
    "OptimizationResult",
    "optimize_plan",
    # comparison
    "FieldDifference",
    "PlanComparison",
    "compare_plans",
    "compare_many",
    "SIGNIFICANCE_THRESHOLDS",
]
