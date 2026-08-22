"""Measurement value objects (PRD §17, §32, §86).

A :class:`Measurement` is not just a number. It carries:

- the :class:`~kneeai.core.units.Quantity` (value, unit, uncertainty, reference range)
- the **landmarks it was derived from**, so a surgeon who moves a landmark can see
  exactly which measurements that invalidates
- the **method**, in words, so the number can be argued with
- the :class:`~kneeai.core.provenance.Origin`, so a report can separate
  AI-derived from surgeon-confirmed values (PRD §32)

Reference ranges are included for planning context only. They are typical adult
values from the orthopaedic literature, not diagnostic thresholds, and the
platform never converts an out-of-range measurement into a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from kneeai.core.anatomy import LandmarkName, Side, Structure
from kneeai.core.errors import MeasurementError, ValidationError
from kneeai.core.provenance import Origin, Provenance
from kneeai.core.units import Quantity, Unit

__all__ = ["MeasurementName", "Measurement", "MeasurementSet", "REFERENCE_RANGES"]


class MeasurementName(str, Enum):
    """Stable measurement identifiers.

    These strings appear in the database, the API and reports. Renaming one is a
    schema migration.
    """

    # --- femur: sizing ---
    FEMORAL_ML_WIDTH = "femoral_ml_width"
    FEMORAL_AP_SIZE = "femoral_ap_size"
    FEMORAL_MEDIAL_AP = "femoral_medial_ap"
    FEMORAL_LATERAL_AP = "femoral_lateral_ap"
    FEMORAL_ASPECT_RATIO = "femoral_aspect_ratio"
    TRANSEPICONDYLAR_WIDTH = "transepicondylar_width"

    # --- femur: rotation and asymmetry ---
    CONDYLAR_TWIST_ANGLE = "condylar_twist_angle"
    DISTAL_CONDYLAR_OFFSET = "distal_condylar_offset"
    POSTERIOR_CONDYLAR_OFFSET = "posterior_condylar_offset"
    FEMORAL_ANATOMICAL_MECHANICAL_ANGLE = "femoral_anatomical_mechanical_angle"

    # --- tibia: sizing ---
    TIBIAL_ML_WIDTH = "tibial_ml_width"
    TIBIAL_AP_DEPTH = "tibial_ap_depth"
    TIBIAL_MEDIAL_AP = "tibial_medial_ap"
    TIBIAL_LATERAL_AP = "tibial_lateral_ap"
    TIBIAL_ASPECT_RATIO = "tibial_aspect_ratio"

    # --- tibia: geometry ---
    POSTERIOR_TIBIAL_SLOPE = "posterior_tibial_slope"
    MEDIAL_PLATEAU_DEPTH = "medial_plateau_depth"
    LATERAL_PLATEAU_DEPTH = "lateral_plateau_depth"
    PLATEAU_HEIGHT_DIFFERENCE = "plateau_height_difference"

    # --- alignment ---
    HKA_ANGLE = "hka_angle"
    MLDFA = "mldfa"
    MPTA = "mpta"
    JLCA = "jlca"
    FEMORAL_MECHANICAL_AXIS_LENGTH = "femoral_mechanical_axis_length"
    TIBIAL_MECHANICAL_AXIS_LENGTH = "tibial_mechanical_axis_length"

    # --- cartilage ---
    FEMORAL_CARTILAGE_VOLUME = "femoral_cartilage_volume"
    FEMORAL_CARTILAGE_MEAN_THICKNESS = "femoral_cartilage_mean_thickness"
    MEDIAL_TIBIAL_CARTILAGE_VOLUME = "medial_tibial_cartilage_volume"
    MEDIAL_TIBIAL_CARTILAGE_MEAN_THICKNESS = "medial_tibial_cartilage_mean_thickness"
    LATERAL_TIBIAL_CARTILAGE_VOLUME = "lateral_tibial_cartilage_volume"
    LATERAL_TIBIAL_CARTILAGE_MEAN_THICKNESS = "lateral_tibial_cartilage_mean_thickness"
    PATELLAR_CARTILAGE_VOLUME = "patellar_cartilage_volume"
    PATELLAR_CARTILAGE_MEAN_THICKNESS = "patellar_cartilage_mean_thickness"
    CARTILAGE_THICKNESS_ASYMMETRY = "cartilage_thickness_asymmetry"

    # --- meniscus ---
    MEDIAL_MENISCUS_VOLUME = "medial_meniscus_volume"
    LATERAL_MENISCUS_VOLUME = "lateral_meniscus_volume"
    MEDIAL_MENISCUS_EXTRUSION = "medial_meniscus_extrusion"
    LATERAL_MENISCUS_EXTRUSION = "lateral_meniscus_extrusion"
    MEDIAL_MENISCUS_HEIGHT = "medial_meniscus_height"
    LATERAL_MENISCUS_HEIGHT = "lateral_meniscus_height"
    MEDIAL_MENISCUS_COVERAGE = "medial_meniscus_coverage"
    LATERAL_MENISCUS_COVERAGE = "lateral_meniscus_coverage"

    @property
    def display_name(self) -> str:
        overrides = {
            MeasurementName.HKA_ANGLE: "Hip-Knee-Ankle Angle",
            MeasurementName.MLDFA: "Mechanical Lateral Distal Femoral Angle",
            MeasurementName.MPTA: "Medial Proximal Tibial Angle",
            MeasurementName.JLCA: "Joint Line Convergence Angle",
            MeasurementName.FEMORAL_ML_WIDTH: "Femoral Mediolateral Width",
            MeasurementName.FEMORAL_AP_SIZE: "Femoral Anteroposterior Size",
            MeasurementName.TIBIAL_ML_WIDTH: "Tibial Mediolateral Width",
            MeasurementName.TIBIAL_AP_DEPTH: "Tibial Anteroposterior Depth",
        }
        if self in overrides:
            return overrides[self]
        return self.value.replace("_", " ").title()


#: Typical adult reference ranges, for planning context only.
#:
#: Sources are the standard orthopaedic ranges used in TKA planning. They are
#: deliberately wide. They are NOT diagnostic thresholds, and the platform does
#: not classify a patient from them (PRD §17: no diagnosis claim without
#: clinical validation).
REFERENCE_RANGES: dict[MeasurementName, tuple[float | None, float | None]] = {
    MeasurementName.FEMORAL_ML_WIDTH: (55.0, 90.0),
    MeasurementName.FEMORAL_AP_SIZE: (48.0, 75.0),
    MeasurementName.FEMORAL_ASPECT_RATIO: (1.05, 1.45),
    MeasurementName.TIBIAL_ML_WIDTH: (58.0, 90.0),
    MeasurementName.TIBIAL_AP_DEPTH: (38.0, 60.0),
    MeasurementName.CONDYLAR_TWIST_ANGLE: (0.0, 10.0),
    MeasurementName.POSTERIOR_TIBIAL_SLOPE: (0.0, 15.0),
    MeasurementName.HKA_ANGLE: (-3.0, 3.0),
    MeasurementName.MLDFA: (85.0, 90.0),
    MeasurementName.MPTA: (85.0, 90.0),
    MeasurementName.JLCA: (0.0, 3.0),
    MeasurementName.MEDIAL_MENISCUS_EXTRUSION: (None, 3.0),
    MeasurementName.LATERAL_MENISCUS_EXTRUSION: (None, 3.0),
}


@dataclass(frozen=True)
class Measurement:
    """A single clinical measurement with full traceability."""

    name: MeasurementName
    quantity: Quantity
    method: str

    #: Landmarks this value depends on. Moving any of them invalidates it.
    landmarks: tuple[LandmarkName, ...] = ()

    #: Structures whose segmentation this value depends on.
    structures: tuple[Structure, ...] = ()

    origin: Origin = Origin.DETERMINISTIC

    #: Sign convention, where the measurement is signed. Stated explicitly
    #: because "5 degrees of varus" and "-5 degrees" are the same fact only if
    #: everyone agrees which way is positive.
    sign_convention: str = ""

    #: Set when the value was computed but should not be trusted, with the reason.
    caveat: str = ""

    def __post_init__(self) -> None:
        if not self.method:
            raise ValidationError(
                f"measurement {self.name.value} must state how it was derived"
            )

    @property
    def value(self) -> float:
        return self.quantity.value

    @property
    def unit(self) -> Unit:
        return self.quantity.unit

    @property
    def in_reference_range(self) -> bool | None:
        return self.quantity.in_reference_range

    @property
    def is_reliable(self) -> bool:
        return not self.caveat

    def with_caveat(self, caveat: str) -> Measurement:
        return replace(self, caveat=caveat)

    def format(self, decimals: int = 1) -> str:
        return self.quantity.format(decimals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "display_name": self.name.display_name,
            "quantity": self.quantity.to_dict(),
            "formatted": self.format(),
            "method": self.method,
            "landmarks": [lm.value for lm in self.landmarks],
            "structures": [s.value for s in self.structures],
            "origin": self.origin.value,
            "sign_convention": self.sign_convention,
            "in_reference_range": self.in_reference_range,
            "caveat": self.caveat,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Measurement:
        return cls(
            name=MeasurementName(payload["name"]),
            quantity=Quantity.from_dict(payload["quantity"]),
            method=str(payload["method"]),
            landmarks=tuple(LandmarkName(v) for v in payload.get("landmarks", [])),
            structures=tuple(Structure(v) for v in payload.get("structures", [])),
            origin=Origin(payload.get("origin", "deterministic")),
            sign_convention=str(payload.get("sign_convention", "")),
            caveat=str(payload.get("caveat", "")),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Measurement({self.name.value}={self.format()})"


@dataclass
class MeasurementSet:
    """All measurements for one case."""

    measurements: dict[MeasurementName, Measurement] = field(default_factory=dict)
    side: Side = Side.UNKNOWN
    provenance: Provenance | None = None

    #: Measurements that could not be computed, and why. Recorded rather than
    #: silently omitted: "the platform did not report tibial slope" and "the
    #: platform could not measure tibial slope because the plateau plane fit
    #: failed" are very different messages to a surgeon.
    unavailable: dict[str, str] = field(default_factory=dict)

    def add(self, measurement: Measurement) -> MeasurementSet:
        self.measurements[measurement.name] = measurement
        return self

    def add_all(self, measurements: Iterable[Measurement]) -> MeasurementSet:
        for m in measurements:
            self.add(m)
        return self

    def mark_unavailable(self, name: MeasurementName | str, reason: str) -> MeasurementSet:
        key = name.value if isinstance(name, MeasurementName) else str(name)
        self.unavailable[key] = reason
        return self

    # --- access ---
    def __contains__(self, name: object) -> bool:
        return name in self.measurements

    def __len__(self) -> int:
        return len(self.measurements)

    def __iter__(self) -> Iterator[Measurement]:
        return iter(self.measurements.values())

    def __getitem__(self, name: MeasurementName) -> Measurement:
        return self.require(name)

    def get(self, name: MeasurementName) -> Measurement | None:
        return self.measurements.get(name)

    def require(self, name: MeasurementName) -> Measurement:
        found = self.measurements.get(name)
        if found is None:
            reason = self.unavailable.get(name.value)
            detail = f": {reason}" if reason else ""
            raise MeasurementError(
                f"measurement {name.value} is not available for this case{detail}"
            )
        return found

    def value(self, name: MeasurementName, default: float | None = None) -> float:
        found = self.measurements.get(name)
        if found is None:
            if default is None:
                return float(self.require(name).value)  # raises with context
            return default
        return float(found.value)

    def values(self) -> dict[str, float]:
        """Plain name -> value mapping, for comparison and export."""
        return {name.value: m.value for name, m in self.measurements.items()}

    def units(self) -> dict[str, str]:
        return {name.value: m.unit.value for name, m in self.measurements.items()}

    # --- quality ---
    def depending_on(self, landmark: LandmarkName) -> list[Measurement]:
        """Measurements invalidated by a change to ``landmark``.

        Drives the dashboard behaviour when a surgeon adjusts a landmark: the
        affected measurements are the ones that must be recomputed and re-shown,
        rather than recomputing everything and losing track of what moved.
        """
        return [m for m in self.measurements.values() if landmark in m.landmarks]

    def with_caveats(self) -> list[Measurement]:
        return [m for m in self.measurements.values() if m.caveat]

    def out_of_reference_range(self) -> list[Measurement]:
        return [m for m in self.measurements.values() if m.in_reference_range is False]

    # --- serialisation ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "n_measurements": len(self.measurements),
            "measurements": [
                m.to_dict()
                for m in sorted(self.measurements.values(), key=lambda x: x.name.value)
            ],
            "unavailable": dict(self.unavailable),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MeasurementSet:
        items = [Measurement.from_dict(m) for m in payload.get("measurements", [])]
        return cls(
            measurements={m.name: m for m in items},
            side=Side(payload.get("side", "unknown")),
            unavailable=dict(payload.get("unavailable", {})),
            provenance=(
                Provenance.from_dict(payload["provenance"])
                if payload.get("provenance")
                else None
            ),
        )

    def summary(self) -> str:
        lines = [f"MeasurementSet ({self.side.value} knee, {len(self.measurements)} values)"]
        for m in sorted(self.measurements.values(), key=lambda x: x.name.value):
            flag = ""
            if m.in_reference_range is False:
                flag = "  [outside typical range]"
            if m.caveat:
                flag += f"  [! {m.caveat}]"
            lines.append(f"  {m.name.display_name:<44} {m.format():>16}{flag}")
        if self.unavailable:
            lines.append("  unavailable:")
            for key, reason in sorted(self.unavailable.items()):
                lines.append(f"    {key}: {reason}")
        return "\n".join(lines)


def make_measurement(
    name: MeasurementName,
    value: float,
    unit: Unit,
    *,
    method: str,
    landmarks: Iterable[LandmarkName] = (),
    structures: Iterable[Structure] = (),
    uncertainty: float | None = None,
    origin: Origin = Origin.DETERMINISTIC,
    sign_convention: str = "",
    caveat: str = "",
) -> Measurement:
    """Build a measurement, attaching its reference range automatically."""
    if not np.isfinite(value):
        raise MeasurementError(
            f"{name.value} computed as a non-finite value; the inputs are degenerate"
        )
    low, high = REFERENCE_RANGES.get(name, (None, None))
    return Measurement(
        name=name,
        quantity=Quantity(
            value=float(value),
            unit=unit,
            uncertainty=uncertainty,
            reference_low=low,
            reference_high=high,
        ),
        method=method,
        landmarks=tuple(landmarks),
        structures=tuple(structures),
        origin=origin,
        sign_convention=sign_convention,
        caveat=caveat,
    )
