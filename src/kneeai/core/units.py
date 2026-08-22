"""Units and quantity handling.

The platform works in **millimetres**, **degrees**, **cubic millimetres** and
**seconds** everywhere internally. Conversion happens only at IO boundaries
(DICOM parsing, report rendering, robot SDK).

Mixed units are the second most common source of quietly wrong clinical numbers
after mixed coordinate frames, so quantities that leave the numeric core and
enter a report or an API response are wrapped in :class:`Quantity`, which
carries its unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from kneeai.core.errors import ValidationError

__all__ = ["Unit", "Quantity", "mm", "deg", "mm3", "percent", "ratio", "seconds"]


class Unit(str, Enum):
    MM = "mm"
    MM2 = "mm^2"
    MM3 = "mm^3"
    DEG = "deg"
    RAD = "rad"
    PERCENT = "%"
    RATIO = ""
    SECONDS = "s"
    NEWTON = "N"
    NEWTON_METRE = "Nm"
    MM_PER_S = "mm/s"
    DEG_PER_S = "deg/s"
    MM_PER_S2 = "mm/s^2"
    COUNT = "count"
    HU = "HU"

    @property
    def is_angle(self) -> bool:
        return self in (Unit.DEG, Unit.RAD)


#: Conversions into the platform's canonical unit for each dimension.
_TO_CANONICAL: dict[tuple[Unit, Unit], float] = {
    (Unit.RAD, Unit.DEG): 180.0 / math.pi,
    (Unit.DEG, Unit.RAD): math.pi / 180.0,
    (Unit.RATIO, Unit.PERCENT): 100.0,
    (Unit.PERCENT, Unit.RATIO): 0.01,
}


@dataclass(frozen=True)
class Quantity:
    """A number with a unit, optional uncertainty, and optional reference range.

    ``uncertainty`` is a 1-sigma estimate in the same unit. It is populated where
    the platform can honestly estimate it (registration residual, landmark
    localisation error propagated through a measurement) and left ``None``
    otherwise. It is never invented: a missing uncertainty is more useful to a
    surgeon than a fabricated one.
    """

    value: float
    unit: Unit
    uncertainty: float | None = None
    reference_low: float | None = None
    reference_high: float | None = None

    def __post_init__(self) -> None:
        v = float(self.value)
        if not math.isfinite(v):
            raise ValidationError(f"quantity value must be finite, got {self.value}")
        object.__setattr__(self, "value", v)
        if self.uncertainty is not None:
            u = float(self.uncertainty)
            if u < 0:
                raise ValidationError("uncertainty must be non-negative")
            object.__setattr__(self, "uncertainty", u)

    def to(self, unit: Unit) -> Quantity:
        if unit is self.unit:
            return self
        factor = _TO_CANONICAL.get((self.unit, unit))
        if factor is None:
            raise ValidationError(f"no conversion defined from {self.unit.value} to {unit.value}")
        return Quantity(
            self.value * factor,
            unit,
            uncertainty=None if self.uncertainty is None else self.uncertainty * factor,
            reference_low=None if self.reference_low is None else self.reference_low * factor,
            reference_high=None if self.reference_high is None else self.reference_high * factor,
        )

    @property
    def in_reference_range(self) -> bool | None:
        """``None`` when no reference range is defined for this measurement."""
        if self.reference_low is None and self.reference_high is None:
            return None
        if self.reference_low is not None and self.value < self.reference_low:
            return False
        if self.reference_high is not None and self.value > self.reference_high:
            return False
        return True

    def format(self, decimals: int = 1) -> str:
        sep = "" if self.unit in (Unit.PERCENT, Unit.DEG, Unit.RATIO) else " "
        text = f"{self.value:.{decimals}f}{sep}{self.unit.value}".rstrip()
        if self.uncertainty is not None:
            text += f" (±{self.uncertainty:.{decimals}f})"
        return text

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"value": self.value, "unit": self.unit.value}
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty
        if self.reference_low is not None:
            payload["reference_low"] = self.reference_low
        if self.reference_high is not None:
            payload["reference_high"] = self.reference_high
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> Quantity:
        return cls(
            value=float(payload["value"]),
            unit=Unit(payload.get("unit", "")),
            uncertainty=payload.get("uncertainty"),
            reference_low=payload.get("reference_low"),
            reference_high=payload.get("reference_high"),
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.format()


# --- convenience constructors ---------------------------------------------


def mm(value: float, uncertainty: float | None = None, **kw: float | None) -> Quantity:
    return Quantity(value, Unit.MM, uncertainty=uncertainty, **kw)  # type: ignore[arg-type]


def deg(value: float, uncertainty: float | None = None, **kw: float | None) -> Quantity:
    return Quantity(value, Unit.DEG, uncertainty=uncertainty, **kw)  # type: ignore[arg-type]


def mm3(value: float, uncertainty: float | None = None, **kw: float | None) -> Quantity:
    return Quantity(value, Unit.MM3, uncertainty=uncertainty, **kw)  # type: ignore[arg-type]


def percent(value: float, uncertainty: float | None = None, **kw: float | None) -> Quantity:
    return Quantity(value, Unit.PERCENT, uncertainty=uncertainty, **kw)  # type: ignore[arg-type]


def ratio(value: float, uncertainty: float | None = None) -> Quantity:
    return Quantity(value, Unit.RATIO, uncertainty=uncertainty)


def seconds(value: float) -> Quantity:
    return Quantity(value, Unit.SECONDS)
