"""Case lifecycle state machine (PRD §8, §91).

The state machine is explicit and validated rather than implied by which fields
happen to be populated. Two properties matter clinically:

1. A case cannot reach ``APPROVED`` without passing through ``SURGEON_REVIEW``.
2. Once ``APPROVED``, the plan is locked. Re-planning requires an explicit,
   audited transition back to ``PLANNING`` which invalidates the approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from kneeai.core.errors import ValidationError
from kneeai.core.provenance import now_utc

__all__ = ["CaseState", "StateTransition", "CaseStateMachine", "TERMINAL_STATES"]


class CaseState(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    VALIDATION_FAILED = "validation_failed"
    PREPROCESSING = "preprocessing"
    AI_PROCESSING = "ai_processing"
    AI_FAILED = "ai_failed"
    RECONSTRUCTION = "reconstruction"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    SURGEON_REVIEW = "surgeon_review"
    APPROVED = "approved"
    EXPORTED = "exported"
    REJECTED = "rejected"
    ARCHIVED = "archived"

    @property
    def is_failure(self) -> bool:
        return self in (CaseState.VALIDATION_FAILED, CaseState.AI_FAILED, CaseState.REJECTED)

    @property
    def is_locked(self) -> bool:
        """States in which the surgical plan must not be mutated."""
        return self in (CaseState.APPROVED, CaseState.EXPORTED, CaseState.ARCHIVED)


TERMINAL_STATES: frozenset[CaseState] = frozenset({CaseState.ARCHIVED})


#: Allowed transitions. Anything absent here is rejected.
_ALLOWED: dict[CaseState, frozenset[CaseState]] = {
    CaseState.CREATED: frozenset({CaseState.UPLOADED, CaseState.ARCHIVED}),
    CaseState.UPLOADED: frozenset({CaseState.VALIDATING, CaseState.ARCHIVED}),
    CaseState.VALIDATING: frozenset(
        {CaseState.PREPROCESSING, CaseState.VALIDATION_FAILED, CaseState.ARCHIVED}
    ),
    # A failed validation can be retried after the study is corrected/re-uploaded.
    CaseState.VALIDATION_FAILED: frozenset({CaseState.UPLOADED, CaseState.ARCHIVED}),
    CaseState.PREPROCESSING: frozenset(
        {CaseState.AI_PROCESSING, CaseState.AI_FAILED, CaseState.ARCHIVED}
    ),
    CaseState.AI_PROCESSING: frozenset(
        {CaseState.RECONSTRUCTION, CaseState.AI_FAILED, CaseState.ARCHIVED}
    ),
    CaseState.AI_FAILED: frozenset({CaseState.PREPROCESSING, CaseState.ARCHIVED}),
    CaseState.RECONSTRUCTION: frozenset(
        {CaseState.ANALYSIS, CaseState.AI_FAILED, CaseState.ARCHIVED}
    ),
    CaseState.ANALYSIS: frozenset({CaseState.PLANNING, CaseState.AI_FAILED, CaseState.ARCHIVED}),
    CaseState.PLANNING: frozenset({CaseState.SURGEON_REVIEW, CaseState.ARCHIVED}),
    # Review can loop back to planning (surgeon edits) any number of times.
    CaseState.SURGEON_REVIEW: frozenset(
        {CaseState.PLANNING, CaseState.APPROVED, CaseState.REJECTED, CaseState.ARCHIVED}
    ),
    # Re-opening an approved case is permitted but invalidates the approval.
    CaseState.APPROVED: frozenset(
        {CaseState.EXPORTED, CaseState.PLANNING, CaseState.ARCHIVED}
    ),
    CaseState.EXPORTED: frozenset({CaseState.PLANNING, CaseState.ARCHIVED}),
    CaseState.REJECTED: frozenset({CaseState.PLANNING, CaseState.ARCHIVED}),
    CaseState.ARCHIVED: frozenset(),
}

#: Transitions that void any existing surgeon approval.
_APPROVAL_VOIDING: frozenset[tuple[CaseState, CaseState]] = frozenset(
    {
        (CaseState.APPROVED, CaseState.PLANNING),
        (CaseState.EXPORTED, CaseState.PLANNING),
    }
)


@dataclass(frozen=True)
class StateTransition:
    from_state: CaseState
    to_state: CaseState
    at: datetime
    actor: str
    reason: str = ""
    voids_approval: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "at": self.at.isoformat(),
            "actor": self.actor,
            "reason": self.reason,
            "voids_approval": self.voids_approval,
        }


@dataclass
class CaseStateMachine:
    """Tracks a case's state and the full transition history.

    History is append-only. It is the audit trail required by PRD §8/§32 and is
    what lets a report state exactly when a plan was approved and by whom.
    """

    state: CaseState = CaseState.CREATED
    history: list[StateTransition] = field(default_factory=list)

    def can(self, to_state: CaseState) -> bool:
        return to_state in _ALLOWED.get(self.state, frozenset())

    def allowed_next(self) -> frozenset[CaseState]:
        return _ALLOWED.get(self.state, frozenset())

    def transition(
        self, to_state: CaseState, *, actor: str, reason: str = ""
    ) -> StateTransition:
        if not actor:
            raise ValidationError("state transitions must record an actor for the audit trail")
        if self.state in TERMINAL_STATES:
            raise ValidationError(f"case is in terminal state {self.state.value}")
        if not self.can(to_state):
            allowed = sorted(s.value for s in self.allowed_next())
            raise ValidationError(
                f"illegal transition {self.state.value} -> {to_state.value}; "
                f"allowed: {allowed or ['<none>']}"
            )
        voids = (self.state, to_state) in _APPROVAL_VOIDING
        tr = StateTransition(
            from_state=self.state,
            to_state=to_state,
            at=now_utc(),
            actor=actor,
            reason=reason,
            voids_approval=voids,
        )
        self.history.append(tr)
        self.state = to_state
        return tr

    def require(self, *states: CaseState) -> None:
        if self.state not in states:
            names = ", ".join(s.value for s in states)
            raise ValidationError(
                f"operation requires case state in [{names}] but case is {self.state.value}"
            )

    @property
    def approval_voided(self) -> bool:
        """True if the case was ever re-opened after approval and not re-approved."""
        for tr in reversed(self.history):
            if tr.to_state is CaseState.APPROVED:
                return False
            if tr.voids_approval:
                return True
        return False

    def approved_at(self) -> datetime | None:
        for tr in reversed(self.history):
            if tr.to_state is CaseState.APPROVED:
                return tr.at
        return None

    def approved_by(self) -> str | None:
        for tr in reversed(self.history):
            if tr.to_state is CaseState.APPROVED:
                return tr.actor
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "history": [t.to_dict() for t in self.history],
        }
