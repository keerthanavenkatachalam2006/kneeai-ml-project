"""Simulation safety state machine (PRD §79, §80, §117, §119).

This module exists to make one property structural rather than aspirational: a
robot simulation cannot start unless a surgeon has explicitly approved the plan.
Not "should not" — *cannot*. :meth:`SimulationController.start` raises unless the
machine has passed through :attr:`SimulationState.APPROVED_FOR_SIMULATION`, and
the only route into that state is :meth:`record_surgeon_approval`, which requires
a named approver.

The lifecycle::

    PLANNING
        ↓ submit_for_approval()
    AWAITING_SURGEON_APPROVAL
        ↓ record_surgeon_approval(approver=...)
    APPROVED_FOR_SIMULATION
        ↓ run_safety_checks()
    SAFETY_CHECK
        ↓ start()
    SIMULATION_RUNNING
        ↓ complete()
    SIMULATION_COMPLETE

``emergency_stop()`` is reachable from any active state and is terminal for that
run. It stops the *simulation*; there is no hardware in this system to stop.

Nothing here talks to an actuator, and nothing here can be made to. The module
imports no driver, opens no socket and emits no motion command — it produces
state transitions and an audit trail. PRD §117 requires that the path from a
model to a motor be broken by a human decision, and the break is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from kneeai.core.errors import GuardrailViolation
from kneeai.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "SimulationState",
    "SafetyCheck",
    "SafetyReport",
    "StateTransition",
    "SimulationController",
    "SIMULATION_ONLY_BANNER",
]

#: Must be displayed wherever simulation output is shown (PRD §80).
SIMULATION_ONLY_BANNER = "SIMULATION ONLY — NO HARDWARE CONTROL"


class SimulationState(str, Enum):
    """States of a single simulation run."""

    PLANNING = "planning"
    AWAITING_SURGEON_APPROVAL = "awaiting_surgeon_approval"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    SAFETY_CHECK = "safety_check"
    SIMULATION_RUNNING = "simulation_running"
    SIMULATION_COMPLETE = "simulation_complete"
    EMERGENCY_STOPPED = "emergency_stopped"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (
            SimulationState.SIMULATION_COMPLETE,
            SimulationState.EMERGENCY_STOPPED,
            SimulationState.REJECTED,
        )

    @property
    def allows_motion(self) -> bool:
        """Whether simulated motion may be emitted in this state.

        Only one state qualifies. Anything reading this to decide whether to move
        is asking the right question.
        """
        return self is SimulationState.SIMULATION_RUNNING


#: The only transitions permitted. Anything absent here is rejected.
_ALLOWED: dict[SimulationState, frozenset[SimulationState]] = {
    SimulationState.PLANNING: frozenset(
        {SimulationState.AWAITING_SURGEON_APPROVAL, SimulationState.REJECTED}
    ),
    SimulationState.AWAITING_SURGEON_APPROVAL: frozenset(
        {
            SimulationState.APPROVED_FOR_SIMULATION,
            SimulationState.REJECTED,
            SimulationState.PLANNING,
        }
    ),
    SimulationState.APPROVED_FOR_SIMULATION: frozenset(
        {SimulationState.SAFETY_CHECK, SimulationState.EMERGENCY_STOPPED}
    ),
    SimulationState.SAFETY_CHECK: frozenset(
        {
            SimulationState.SIMULATION_RUNNING,
            SimulationState.REJECTED,
            SimulationState.EMERGENCY_STOPPED,
        }
    ),
    SimulationState.SIMULATION_RUNNING: frozenset(
        {SimulationState.SIMULATION_COMPLETE, SimulationState.EMERGENCY_STOPPED}
    ),
    SimulationState.SIMULATION_COMPLETE: frozenset(),
    SimulationState.EMERGENCY_STOPPED: frozenset(),
    SimulationState.REJECTED: frozenset(),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class StateTransition:
    """One state change, retained for audit."""

    from_state: SimulationState
    to_state: SimulationState
    at: datetime
    actor: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "at": self.at.isoformat(),
            "actor": self.actor,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SafetyCheck:
    """One pre-simulation gate."""

    name: str
    passed: bool
    detail: str
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "blocking": self.blocking,
        }


@dataclass
class SafetyReport:
    """Outcome of the pre-simulation gates."""

    checks: list[SafetyCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(c.blocking and not c.passed for c in self.checks)

    @property
    def blocking_failures(self) -> list[SafetyCheck]:
        return [c for c in self.checks if c.blocking and not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_blocking_failures": len(self.blocking_failures),
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class SimulationController:
    """Guards the lifecycle of one simulation run."""

    case_id: str
    plan_id: str
    state: SimulationState = SimulationState.PLANNING
    history: list[StateTransition] = field(default_factory=list)
    approver: str | None = None
    approved_at: datetime | None = None
    safety_report: SafetyReport | None = None

    # --- internals -------------------------------------------------------

    def _transition(
        self, to_state: SimulationState, *, actor: str, reason: str = ""
    ) -> None:
        allowed = _ALLOWED.get(self.state, frozenset())
        if to_state not in allowed:
            raise GuardrailViolation(
                f"illegal simulation transition {self.state.value} -> {to_state.value}. "
                f"Permitted from {self.state.value}: "
                f"{sorted(s.value for s in allowed) or 'none (terminal state)'}"
            )
        transition = StateTransition(
            from_state=self.state, to_state=to_state, at=_now(), actor=actor, reason=reason
        )
        self.history.append(transition)
        self.state = to_state
        log.info(
            "simulation state changed",
            fields={
                "case": self.case_id,
                "plan": self.plan_id,
                "from": transition.from_state.value,
                "to": to_state.value,
                "actor": actor,
            },
        )

    # --- lifecycle -------------------------------------------------------

    def submit_for_approval(self, *, actor: str = "system") -> None:
        self._transition(
            SimulationState.AWAITING_SURGEON_APPROVAL,
            actor=actor,
            reason="plan complete, awaiting surgeon review",
        )

    def record_surgeon_approval(self, *, approver: str, justification: str = "") -> None:
        """The mandatory human gate.

        An empty approver is refused rather than defaulted. An audit trail whose
        approver is "system" or "" records that nobody took responsibility, which
        is worse than having no trail at all because it looks like one.
        """
        if not approver or not approver.strip():
            raise GuardrailViolation(
                "surgeon approval requires a named approver; PRD §80 makes the "
                "human decision the break in the AI-to-actuator path, and an "
                "anonymous approval is not a break"
            )
        self.approver = approver.strip()
        self.approved_at = _now()
        self._transition(
            SimulationState.APPROVED_FOR_SIMULATION,
            actor=self.approver,
            reason=justification or "surgeon approved plan for simulation",
        )

    def reject(self, *, actor: str, reason: str) -> None:
        self._transition(SimulationState.REJECTED, actor=actor, reason=reason)

    def run_safety_checks(self, report: SafetyReport, *, actor: str = "system") -> SafetyReport:
        """Record the pre-simulation gates. Blocking failures move to REJECTED."""
        self._transition(SimulationState.SAFETY_CHECK, actor=actor, reason="running safety checks")
        self.safety_report = report
        if not report.passed:
            names = ", ".join(c.name for c in report.blocking_failures)
            self._transition(
                SimulationState.REJECTED,
                actor=actor,
                reason=f"blocking safety failures: {names}",
            )
        return report

    def start(self, *, actor: str = "system") -> None:
        """Begin simulated motion. Refuses without approval and passed checks."""
        if self.approver is None:
            raise GuardrailViolation(
                "simulation cannot start without recorded surgeon approval "
                f"(state={self.state.value})"
            )
        if self.safety_report is None or not self.safety_report.passed:
            raise GuardrailViolation(
                "simulation cannot start before safety checks pass "
                f"(state={self.state.value})"
            )
        self._transition(
            SimulationState.SIMULATION_RUNNING,
            actor=actor,
            reason=f"approved by {self.approver}",
        )

    def complete(self, *, actor: str = "system", summary: str = "") -> None:
        self._transition(
            SimulationState.SIMULATION_COMPLETE, actor=actor, reason=summary or "simulation finished"
        )

    def emergency_stop(self, *, actor: str, reason: str = "operator pressed stop") -> None:
        """Halt the simulation. Terminal.

        Stops a *simulation*. There is no hardware attached to this system, and
        this method is not a substitute for a hardware E-stop in a system that has
        one.
        """
        if self.state.is_terminal:
            return
        self.history.append(
            StateTransition(
                from_state=self.state,
                to_state=SimulationState.EMERGENCY_STOPPED,
                at=_now(),
                actor=actor,
                reason=reason,
            )
        )
        self.state = SimulationState.EMERGENCY_STOPPED
        log.warning(
            "simulation emergency stop",
            fields={"case": self.case_id, "plan": self.plan_id, "actor": actor, "reason": reason},
        )

    # --- reporting -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "plan_id": self.plan_id,
            "state": self.state.value,
            "is_terminal": self.state.is_terminal,
            "allows_motion": self.state.allows_motion,
            "approver": self.approver,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "safety_report": self.safety_report.to_dict() if self.safety_report else None,
            "history": [t.to_dict() for t in self.history],
            "banner": SIMULATION_ONLY_BANNER,
            "hardware_control": False,
        }


def build_safety_report(
    *,
    plan: Mapping[str, Any],
    simulation: Mapping[str, Any] | None,
    measurements: Mapping[str, Any] | None = None,
    source: str = "unknown",
) -> SafetyReport:
    """Derive the pre-simulation gates from the case artefacts.

    These are checks on *data completeness and internal consistency*, not clinical
    safety assessments. They answer "is this plan coherent enough to animate",
    which is a much smaller question than "is this plan safe for a patient", and
    the naming keeps the two apart.
    """
    checks: list[SafetyCheck] = []

    cuts: Sequence[Mapping[str, Any]] = plan.get("cuts", []) or []
    checks.append(
        SafetyCheck(
            name="plan_has_cuts",
            passed=bool(cuts),
            detail=f"{len(cuts)} resection cut(s) defined",
        )
    )

    well_formed = [
        c
        for c in cuts
        if isinstance(c.get("plane"), Mapping)
        and len(c["plane"].get("point", ())) == 3
        and len(c["plane"].get("normal", ())) == 3
    ]
    checks.append(
        SafetyCheck(
            name="cut_planes_well_formed",
            passed=len(well_formed) == len(cuts) and bool(cuts),
            detail=f"{len(well_formed)}/{len(cuts)} cuts carry a 3D point and normal",
        )
    )

    frames = {
        c["plane"].get("frame")
        for c in well_formed
        if isinstance(c.get("plane"), Mapping)
    }
    checks.append(
        SafetyCheck(
            name="cut_planes_share_one_frame",
            passed=len(frames) <= 1,
            detail=(
                f"all cuts in frame '{next(iter(frames))}'"
                if len(frames) == 1
                else f"cuts span multiple frames: {sorted(f for f in frames if f)}"
            ),
        )
    )

    checks.append(
        SafetyCheck(
            name="implant_placements_present",
            passed=bool(plan.get("placements")),
            detail=f"{len(plan.get('placements', []) or [])} component placement(s)",
        )
    )

    if simulation is not None:
        checks.append(
            SafetyCheck(
                name="geometric_simulation_passed",
                passed=bool(simulation.get("passed")),
                detail=(
                    f"{simulation.get('n_failures', 0)} failure(s), "
                    f"{simulation.get('n_warnings', 0)} warning(s) from "
                    "kneeai.simulation"
                ),
            )
        )
        gap = simulation.get("seating_gap_mm")
        pen = simulation.get("interpenetration_mm")
        checks.append(
            SafetyCheck(
                name="component_seating_reported",
                passed=gap is not None or pen is not None,
                detail=f"seating gap {gap} mm, interpenetration {pen} mm",
                blocking=False,
            )
        )
    else:
        checks.append(
            SafetyCheck(
                name="geometric_simulation_passed",
                passed=False,
                detail="no simulation result available for this plan",
            )
        )

    # Not a blocker, but it must be visible: animating a phantom is fine, and
    # presenting it as a patient is not.
    checks.append(
        SafetyCheck(
            name="segmentation_source_declared",
            passed=source in ("ai_prediction", "ground_truth", "phantom", "mock"),
            detail=f"source = {source}",
            blocking=False,
        )
    )

    if measurements is not None:
        # Serialised as a mapping of name -> reason; older payloads used a list.
        raw_unavailable = measurements.get("unavailable") or {}
        unavailable = (
            sorted(raw_unavailable)
            if isinstance(raw_unavailable, Mapping)
            else [str(u) for u in raw_unavailable]
        )
        checks.append(
            SafetyCheck(
                name="alignment_measurements_available",
                passed=not any(
                    str(u).startswith(("hka", "mldfa", "mpta")) for u in unavailable
                ),
                detail=(
                    "mechanical-axis alignment unavailable from this study "
                    f"({len(unavailable)} measurement(s) could not be computed); "
                    "a knee MRI contains neither the femoral head nor the ankle"
                    if unavailable
                    else "all alignment measurements available"
                ),
                blocking=False,
            )
        )

    return SafetyReport(checks=checks)
