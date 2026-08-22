"""Alignment strategies and planning targets (PRD §25, §26, §40).

A TKA plan is a set of resection planes. What determines those planes is the
**alignment philosophy** — the rule for where the joint line should end up. The
three in mainstream use produce materially different cuts on the same knee, so the
strategy is an explicit, recorded input to the planner rather than a hard-coded
default.

Mechanical alignment
    Both components perpendicular to their mechanical axes, targeting a neutral
    hip-knee-ankle angle. Predictable, decades of survivorship data, but it
    changes the joint line in a knee that was never neutral, and around a fifth of
    patients report the knee feels unnatural.

Kinematic alignment
    Restore the patient's pre-arthritic joint line by resurfacing: resect exactly
    what the implant replaces. Reproduces native kinematics but can leave
    substantial residual deformity, and there is no long-term survivorship
    evidence for extreme cases.

Restricted kinematic alignment
    Kinematic, but clamped inside a safe envelope of joint-line obliquity and
    residual HKA. This is the default here: it follows the patient's anatomy where
    that anatomy is within a range the literature supports, and falls back towards
    mechanical where it is not — and it **reports** when a clamp was applied, so
    the surgeon knows the plan is a compromise rather than a pure kinematic one.

Nothing in this module decides which strategy is correct for a patient. That is a
surgeon's judgement; the planner's job is to execute a chosen strategy faithfully
and to be explicit about where the strategy's own limits bound the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from kneeai.core.errors import ValidationError

__all__ = [
    "AlignmentStrategy",
    "ResectionTargets",
    "SurgeonPreferences",
    "MECHANICAL_TARGETS",
    "KINEMATIC_TARGETS",
    "RESTRICTED_KINEMATIC_TARGETS",
    "targets_for",
]


class AlignmentStrategy(str, Enum):
    MECHANICAL = "mechanical"
    KINEMATIC = "kinematic"
    RESTRICTED_KINEMATIC = "restricted_kinematic"

    #: Adjusted mechanical: tibia perpendicular, femur allowed slight valgus.
    ADJUSTED_MECHANICAL = "adjusted_mechanical"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def follows_native_anatomy(self) -> bool:
        return self in (
            AlignmentStrategy.KINEMATIC,
            AlignmentStrategy.RESTRICTED_KINEMATIC,
        )


@dataclass(frozen=True)
class ResectionTargets:
    """Angular and depth targets that define a resection plan.

    All angles are in degrees and all depths in millimetres. Coronal angles use a
    single convention throughout: **positive is varus** for the tibia and
    **positive is valgus** for the femur, matching how each is described
    intraoperatively. The sign convention is carried in the plan record so a
    displayed number is never ambiguous.
    """

    strategy: AlignmentStrategy

    # --- tibial coronal ---
    #: Target tibial component coronal angle. 0 = perpendicular to the tibial
    #: mechanical axis. Positive = varus.
    tibial_varus_deg: float = 0.0
    #: How far the planner may deviate from the target to follow native anatomy.
    tibial_varus_envelope_deg: tuple[float, float] = (0.0, 0.0)

    # --- tibial sagittal ---
    #: Target posterior slope of the tibial cut.
    tibial_slope_deg: float = 3.0
    tibial_slope_envelope_deg: tuple[float, float] = (0.0, 7.0)
    #: When true, reproduce the patient's measured slope instead of the target.
    match_native_slope: bool = False

    # --- tibial depth ---
    #: Resection depth below the reference (less worn) compartment's articular
    #: surface. 9 mm is the thinnest tibial construct — 3 mm baseplate plus 6 mm
    #: insert — so the cut restores the joint line rather than raising it. This is
    #: the same rule the distal femoral cut follows, and it is measured from the
    #: unworn side; the "2 mm" figure quoted in the literature is the same plane
    #: measured from the *worn* compartment instead.
    tibial_resection_mm: float = 9.0
    #: Minimum bone left under the baseplate. Cutting deeper than the proximal
    #: metaphysis supports risks subsidence.
    max_tibial_resection_mm: float = 14.0

    # --- femoral coronal ---
    #: Target distal femoral coronal angle. 0 = perpendicular to the femoral
    #: mechanical axis. Positive = valgus.
    femoral_valgus_deg: float = 0.0
    femoral_valgus_envelope_deg: tuple[float, float] = (0.0, 0.0)

    # --- femoral rotation ---
    #: Target external rotation of the femoral component relative to the posterior
    #: condylar axis. 3° is the classical value; referencing the
    #: transepicondylar axis directly sets this from the measured condylar twist.
    femoral_external_rotation_deg: float = 3.0
    #: Derive rotation from the measured transepicondylar axis rather than using
    #: the fixed target.
    reference_transepicondylar_axis: bool = True
    femoral_rotation_envelope_deg: tuple[float, float] = (0.0, 7.0)

    # --- femoral sagittal ---
    #: Flexion of the femoral component in the sagittal plane. A few degrees of
    #: flexion avoids anterior notching without overstuffing.
    femoral_flexion_deg: float = 3.0
    femoral_flexion_envelope_deg: tuple[float, float] = (0.0, 7.0)

    # --- femoral depth ---
    #: Distal femoral resection from the reference condyle.
    distal_femoral_resection_mm: float = 9.0
    max_distal_femoral_resection_mm: float = 14.0

    # --- residual alignment limits ---
    #: Acceptable residual hip-knee-ankle angle after the plan. Positive = varus.
    residual_hka_envelope_deg: tuple[float, float] = (-3.0, 3.0)

    # --- gaps ---
    #: Target extension and flexion gap, i.e. the space the insert must fill.
    target_extension_gap_mm: float = 0.0
    #: Acceptable flexion-extension gap mismatch.
    max_gap_mismatch_mm: float = 2.0

    notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "tibial_varus_envelope_deg",
            "tibial_slope_envelope_deg",
            "femoral_valgus_envelope_deg",
            "femoral_rotation_envelope_deg",
            "femoral_flexion_envelope_deg",
            "residual_hka_envelope_deg",
        ):
            low, high = getattr(self, name)
            if low > high:
                raise ValidationError(
                    f"{name} is inverted: ({low}, {high}). The first element must be "
                    "the lower bound."
                )
        if self.max_tibial_resection_mm < self.tibial_resection_mm:
            raise ValidationError(
                "max_tibial_resection_mm is below the target tibial resection"
            )
        if self.max_distal_femoral_resection_mm < self.distal_femoral_resection_mm:
            raise ValidationError(
                "max_distal_femoral_resection_mm is below the target distal resection"
            )

    def clamp(self, name: str, value: float) -> tuple[float, str | None]:
        """Clamp a value into its envelope, returning ``(value, note)``.

        The note is non-``None`` whenever a clamp was applied, so the plan can say
        "your anatomy suggested 6° of tibial varus; the strategy's envelope limited
        the plan to 3°" instead of silently substituting a different number.
        """
        envelope = getattr(self, f"{name}_envelope_deg", None)
        if envelope is None:
            raise ValidationError(f"no envelope defined for {name!r}")
        low, high = envelope
        if value < low:
            return low, (
                f"{name.replace('_', ' ')} was limited to {low:+.1f}° by the "
                f"{self.strategy.display_name} envelope (native geometry suggested "
                f"{value:+.1f}°)"
            )
        if value > high:
            return high, (
                f"{name.replace('_', ' ')} was limited to {high:+.1f}° by the "
                f"{self.strategy.display_name} envelope (native geometry suggested "
                f"{value:+.1f}°)"
            )
        return value, None

    def with_overrides(self, **overrides: Any) -> ResectionTargets:
        unknown = set(overrides) - {f.name for f in self.__dataclass_fields__.values()}
        if unknown:
            raise ValidationError(f"unknown resection target(s): {sorted(unknown)}")
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        payload["strategy"] = self.strategy.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResectionTargets:
        data = dict(payload)
        data["strategy"] = AlignmentStrategy(data["strategy"])
        for key, value in list(data.items()):
            if key.endswith("_envelope_deg") and value is not None:
                data[key] = tuple(value)
        return cls(**data)


#: Both components perpendicular to their mechanical axes. Zero envelope: the
#: whole point of mechanical alignment is that it does not follow deformity.
MECHANICAL_TARGETS = ResectionTargets(
    strategy=AlignmentStrategy.MECHANICAL,
    tibial_varus_deg=0.0,
    tibial_varus_envelope_deg=(0.0, 0.0),
    tibial_slope_deg=3.0,
    tibial_slope_envelope_deg=(0.0, 7.0),
    femoral_valgus_deg=0.0,
    femoral_valgus_envelope_deg=(0.0, 0.0),
    femoral_external_rotation_deg=3.0,
    reference_transepicondylar_axis=True,
    residual_hka_envelope_deg=(-3.0, 3.0),
    notes=(
        "Neutral mechanical alignment. Predictable and well evidenced, but alters "
        "the joint line in a knee that was not natively neutral."
    ),
)

#: Resurfacing: reproduce the native joint line. Wide envelopes because the whole
#: point is to follow the patient's own geometry.
KINEMATIC_TARGETS = ResectionTargets(
    strategy=AlignmentStrategy.KINEMATIC,
    tibial_varus_deg=0.0,
    tibial_varus_envelope_deg=(-10.0, 10.0),
    tibial_slope_deg=3.0,
    tibial_slope_envelope_deg=(0.0, 12.0),
    match_native_slope=True,
    femoral_valgus_deg=0.0,
    femoral_valgus_envelope_deg=(-10.0, 10.0),
    femoral_external_rotation_deg=0.0,
    reference_transepicondylar_axis=True,
    femoral_rotation_envelope_deg=(-5.0, 10.0),
    residual_hka_envelope_deg=(-10.0, 10.0),
    notes=(
        "Restores the native joint line by resurfacing. Reproduces native "
        "kinematics; may leave substantial residual deformity, for which long-term "
        "survivorship evidence is limited."
    ),
)

#: The default. Kinematic intent inside a safe envelope.
#:
#: The bounds follow the commonly cited restricted-kinematic limits: joint-line
#: obliquity within a few degrees either side of perpendicular, and residual
#: coronal alignment within 3° of neutral.
RESTRICTED_KINEMATIC_TARGETS = ResectionTargets(
    strategy=AlignmentStrategy.RESTRICTED_KINEMATIC,
    tibial_varus_deg=0.0,
    tibial_varus_envelope_deg=(-3.0, 3.0),
    tibial_slope_deg=3.0,
    tibial_slope_envelope_deg=(0.0, 7.0),
    match_native_slope=True,
    femoral_valgus_deg=0.0,
    femoral_valgus_envelope_deg=(-3.0, 3.0),
    femoral_external_rotation_deg=0.0,
    reference_transepicondylar_axis=True,
    femoral_rotation_envelope_deg=(-3.0, 6.0),
    residual_hka_envelope_deg=(-3.0, 3.0),
    notes=(
        "Kinematic intent, constrained to a joint-line obliquity and residual "
        "alignment envelope supported by published outcomes. Every clamp applied is "
        "reported on the plan."
    ),
)

#: Tibia perpendicular, femur permitted slight valgus to reduce medial release.
ADJUSTED_MECHANICAL_TARGETS = ResectionTargets(
    strategy=AlignmentStrategy.ADJUSTED_MECHANICAL,
    tibial_varus_deg=0.0,
    tibial_varus_envelope_deg=(0.0, 2.0),
    tibial_slope_deg=3.0,
    tibial_slope_envelope_deg=(0.0, 7.0),
    femoral_valgus_deg=0.0,
    femoral_valgus_envelope_deg=(-2.0, 0.0),
    residual_hka_envelope_deg=(-3.0, 3.0),
    notes=(
        "Mechanical alignment with a small allowance for residual varus, to reduce "
        "the extent of medial soft-tissue release required."
    ),
)

_TARGETS: dict[AlignmentStrategy, ResectionTargets] = {
    AlignmentStrategy.MECHANICAL: MECHANICAL_TARGETS,
    AlignmentStrategy.KINEMATIC: KINEMATIC_TARGETS,
    AlignmentStrategy.RESTRICTED_KINEMATIC: RESTRICTED_KINEMATIC_TARGETS,
    AlignmentStrategy.ADJUSTED_MECHANICAL: ADJUSTED_MECHANICAL_TARGETS,
}


def targets_for(strategy: AlignmentStrategy) -> ResectionTargets:
    """Default targets for a strategy."""
    try:
        return _TARGETS[strategy]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValidationError(f"no default targets for strategy {strategy}") from exc


@dataclass(frozen=True)
class SurgeonPreferences:
    """Per-surgeon defaults (PRD §20: surgeon preferences are a planner input).

    Kept separate from :class:`ResectionTargets` because these are habits and
    product choices rather than alignment philosophy, and they persist across
    cases while targets are per-case.
    """

    surgeon_id: str = ""
    strategy: AlignmentStrategy = AlignmentStrategy.RESTRICTED_KINEMATIC

    #: Preferred implant families, in order.
    preferred_families: tuple[str, ...] = ()

    #: Preferred bearing type, if the surgeon has one.
    preferred_bearing: str | None = None

    #: Overrides applied on top of the strategy's default targets.
    target_overrides: Mapping[str, Any] = field(default_factory=dict)

    #: Whether to resurface the patella by default.
    resurface_patella: bool = False

    notes: str = ""

    def resolve_targets(self) -> ResectionTargets:
        base = targets_for(self.strategy)
        return base.with_overrides(**dict(self.target_overrides)) if self.target_overrides else base

    def to_dict(self) -> dict[str, Any]:
        return {
            "surgeon_id": self.surgeon_id,
            "strategy": self.strategy.value,
            "preferred_families": list(self.preferred_families),
            "preferred_bearing": self.preferred_bearing,
            "target_overrides": dict(self.target_overrides),
            "resurface_patella": self.resurface_patella,
            "notes": self.notes,
        }
