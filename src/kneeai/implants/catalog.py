"""Implant catalog schema (PRD §21).

An implant in this system is defined by the geometry that determines fit, not by a
marketing size label. Each component carries:

- its **footprint polygon** in the component's own design frame, which is what the
  coverage and overhang calculations run on
- its **resection geometry** (thickness, posterior offset, cut plane relationships)
- the **configurations it is allowed to be combined into**, because femoral and
  tibial sizes are not freely mixable in a real system

Regulatory position
-------------------
The bundled catalog is **synthetic reference geometry** for development and
testing. It is not any manufacturer's implant, its dimensions are interpolated
from published size ranges, and it must never be used to plan an operation on a
patient. :class:`ImplantCatalog` marks it ``clinical_use_permitted=False`` and
:mod:`kneeai.config` refuses to run in production with non-clinical implants
enabled.

Loading authorised geometry means supplying a manufacturer catalog file with real
footprints under a licence that permits it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import Side
from kneeai.core.errors import ImplantError, ValidationError
from kneeai.core.frames import Frame
from kneeai.logging import get_logger

__all__ = [
    "ComponentType",
    "FixationType",
    "BearingType",
    "ImplantComponent",
    "ImplantConfiguration",
    "ImplantFamily",
    "ImplantCatalog",
    "load_catalog",
]

log = get_logger("implants.catalog")


class ComponentType(str, Enum):
    FEMORAL = "femoral"
    TIBIAL_BASEPLATE = "tibial_baseplate"
    TIBIAL_INSERT = "tibial_insert"
    PATELLAR = "patellar"


class FixationType(str, Enum):
    CEMENTED = "cemented"
    CEMENTLESS = "cementless"
    HYBRID = "hybrid"


class BearingType(str, Enum):
    CRUCIATE_RETAINING = "cruciate_retaining"
    POSTERIOR_STABILISED = "posterior_stabilised"
    MEDIAL_CONGRUENT = "medial_congruent"
    CONSTRAINED = "constrained"
    MOBILE_BEARING = "mobile_bearing"


@dataclass(frozen=True)
class ImplantComponent:
    """One implant component and the geometry that determines its fit.

    The design frame convention matches the platform's patient frame axes so a
    component can be placed by a rigid transform without an axis remap:
    ``+X`` right, ``+Y`` anterior, ``+Z`` superior, origin at the component's
    reference point (the centre of its resection interface).

    ``footprint`` is the component's outline projected onto its seating plane, as
    an ``(N, 2)`` polygon in ``(x, y)`` millimetres. Coverage and overhang are
    computed against this polygon, which is why it is required rather than
    optional: a component described only by "size 4" cannot be fitted to a
    patient's bone.
    """

    component_id: str
    family_id: str
    component_type: ComponentType
    size_label: str

    #: Outline on the seating plane, in the component design frame.
    footprint: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 2)))

    #: Nominal mediolateral and anteroposterior dimensions, for display and for
    #: the coarse size shortlist before full polygon fitting.
    ml_mm: float = 0.0
    ap_mm: float = 0.0

    #: Intentional cutouts in the footprint — the posterior cruciate notch, keel
    #: slots, peg reliefs — as polygons in the design frame.
    #:
    #: These exist so uncovered bone *inside a deliberate relief* is not scored as
    #: undercoverage. A tibial baseplate's PCL notch leaves ~13 mm of plateau
    #: uncovered by design; counting that as a fit defect would penalise every
    #: baseplate equally and swamp the real signal, which is peripheral rim
    #: coverage.
    relief_regions: tuple[NDArray[np.float64], ...] = ()

    #: Sagittal anteroposterior envelope, for femoral components: the distance from
    #: the anterior flange to the posterior condyle, measured on the component's
    #: *outer* surface.
    #:
    #: Distinct from :attr:`ap_mm`, which is the depth of the *seating footprint* on
    #: the distal cut. For a femoral component these differ by nearly twofold, and
    #: it is the envelope — not the footprint depth — that determines anterior
    #: notching and flexion-gap fit.
    envelope_ap_mm: float | None = None

    #: Thickness of the anterior flange, for femoral components.
    #:
    #: Needed to place the anterior femoral cut. The cut sits *behind* the
    #: component's outer anterior surface by this thickness, so the internal cut-box
    #: depth is ``envelope_ap - posterior_offset - anterior_flange_thickness``.
    #: Placing the cut at the full envelope depth instead puts it a centimetre in
    #: front of the anterior cortex, which reads as gross overstuffing.
    anterior_flange_thickness_mm: float = 0.0

    @property
    def internal_ap_mm(self) -> float:
        """Depth of the cut box, from the posterior cut to the anterior cut."""
        if self.envelope_ap_mm is None:
            raise ImplantError(
                f"{self.component_id} has no anteroposterior envelope, so its internal "
                "cut-box depth is undefined"
            )
        depth = (
            self.envelope_ap_mm
            - self.posterior_offset_mm
            - self.anterior_flange_thickness_mm
        )
        if depth <= 0:
            raise ImplantError(
                f"{self.component_id}: posterior offset "
                f"({self.posterior_offset_mm} mm) plus anterior flange thickness "
                f"({self.anterior_flange_thickness_mm} mm) is not smaller than the "
                f"anteroposterior envelope ({self.envelope_ap_mm} mm)"
            )
        return depth

    #: Material thickness at the articular surface. Determines how much bone must
    #: be resected to restore the joint line.
    thickness_mm: float = 0.0

    #: Femoral components only: the depth of the posterior condylar offset, and the
    #: distal thickness. Both feed the resection calculation.
    posterior_offset_mm: float = 0.0
    distal_thickness_mm: float = 0.0

    #: Side-specific components (most femoral components are). ``UNKNOWN`` means
    #: the component is symmetric and fits either knee.
    side: Side = Side.UNKNOWN

    #: Sizes of the same type this one may be combined with, by size label. Empty
    #: means unconstrained.
    compatible_sizes: tuple[str, ...] = ()

    #: Optional path to a full 3D surface for simulation and collision testing.
    mesh_path: str | None = None

    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValidationError("implant components require a component_id")
        footprint = np.asarray(self.footprint, dtype=np.float64)
        if footprint.size:
            footprint = np.atleast_2d(footprint)
            if footprint.ndim != 2 or footprint.shape[1] != 2:
                raise ValidationError(
                    f"{self.component_id}: footprint must be (N, 2), got {footprint.shape}"
                )
            if footprint.shape[0] < 3:
                raise ValidationError(
                    f"{self.component_id}: a footprint needs at least 3 vertices"
                )
        object.__setattr__(self, "footprint", footprint)

        reliefs: list[NDArray[np.float64]] = []
        for index, region in enumerate(self.relief_regions):
            polygon = np.atleast_2d(np.asarray(region, dtype=np.float64))
            if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
                raise ValidationError(
                    f"{self.component_id}: relief region {index} must be an "
                    f"(N>=3, 2) polygon, got {polygon.shape}"
                )
            reliefs.append(polygon)
        object.__setattr__(self, "relief_regions", tuple(reliefs))

        if self.envelope_ap_mm is not None and self.envelope_ap_mm <= 0:
            raise ValidationError(
                f"{self.component_id}: envelope_ap_mm must be positive when set"
            )

        if self.ml_mm <= 0 or self.ap_mm <= 0:
            raise ValidationError(
                f"{self.component_id}: mediolateral and anteroposterior dimensions "
                "must be positive"
            )
        if self.thickness_mm < 0:
            raise ValidationError(f"{self.component_id}: thickness cannot be negative")

    @property
    def has_footprint(self) -> bool:
        return bool(self.footprint.size)

    @property
    def aspect_ratio(self) -> float:
        return self.ml_mm / self.ap_mm

    @property
    def footprint_area_mm2(self) -> float:
        """Polygon area via the shoelace formula."""
        if not self.has_footprint:
            return 0.0
        x, y = self.footprint[:, 0], self.footprint[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def footprint_polygon(self) -> Any:
        """Shapely polygon of the footprint, in the component design frame."""
        from shapely.geometry import Polygon

        if not self.has_footprint:
            raise ImplantError(
                f"{self.component_id} has no footprint polygon, so coverage and "
                "overhang cannot be computed. Supply authorised component geometry."
            )
        polygon = Polygon(self.footprint)
        if not polygon.is_valid:
            # A self-intersecting outline usually means the vertices are unordered.
            repaired = polygon.buffer(0)
            if repaired.is_empty:
                raise ImplantError(
                    f"{self.component_id}: footprint polygon is invalid and could "
                    "not be repaired; check the vertex ordering"
                )
            log.warning(
                "footprint polygon repaired via buffer(0)",
                fields={"component": self.component_id},
            )
            return repaired
        return polygon

    def relief_polygons(self) -> list[Any]:
        """Shapely polygons for the intentional reliefs, in the design frame."""
        from shapely.geometry import Polygon

        out: list[Any] = []
        for region in self.relief_regions:
            polygon = Polygon(region)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                out.append(polygon)
        return out

    def mirrored(self) -> ImplantComponent:
        """Mirror across the sagittal plane, producing the opposite-side component.

        Used only for symmetric or notionally-mirrored reference geometry. A real
        left femoral component is a distinct manufactured part, not a mirrored
        right one, so this is not a substitute for the real left-side catalog
        entry — it exists so the phantom and tests can exercise both sides.
        """
        def flip(polygon: NDArray[np.float64]) -> NDArray[np.float64]:
            mirrored = polygon.copy()
            mirrored[:, 0] *= -1.0
            # Reverse the winding so the polygon keeps a consistent orientation.
            return mirrored[::-1]

        return replace(
            self,
            component_id=f"{self.component_id}-mirrored",
            footprint=flip(self.footprint) if self.has_footprint else self.footprint,
            relief_regions=tuple(flip(r) for r in self.relief_regions),
            side=self.side.opposite(),
            notes=f"{self.notes} (mirrored from {self.component_id})".strip(),
        )

    def fits_side(self, side: Side) -> bool:
        return self.side is Side.UNKNOWN or self.side is side

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "family_id": self.family_id,
            "component_type": self.component_type.value,
            "size_label": self.size_label,
            "footprint": self.footprint.tolist(),
            "relief_regions": [r.tolist() for r in self.relief_regions],
            "ml_mm": self.ml_mm,
            "ap_mm": self.ap_mm,
            "envelope_ap_mm": self.envelope_ap_mm,
            "anterior_flange_thickness_mm": self.anterior_flange_thickness_mm,
            "thickness_mm": self.thickness_mm,
            "posterior_offset_mm": self.posterior_offset_mm,
            "distal_thickness_mm": self.distal_thickness_mm,
            "side": self.side.value,
            "compatible_sizes": list(self.compatible_sizes),
            "mesh_path": self.mesh_path,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImplantComponent:
        return cls(
            component_id=str(payload["component_id"]),
            family_id=str(payload["family_id"]),
            component_type=ComponentType(payload["component_type"]),
            size_label=str(payload["size_label"]),
            footprint=np.asarray(payload.get("footprint", []), dtype=np.float64),
            relief_regions=tuple(
                np.asarray(r, dtype=np.float64) for r in payload.get("relief_regions", [])
            ),
            ml_mm=float(payload["ml_mm"]),
            ap_mm=float(payload["ap_mm"]),
            envelope_ap_mm=(
                float(payload["envelope_ap_mm"])
                if payload.get("envelope_ap_mm") is not None
                else None
            ),
            anterior_flange_thickness_mm=float(
                payload.get("anterior_flange_thickness_mm", 0.0)
            ),
            thickness_mm=float(payload.get("thickness_mm", 0.0)),
            posterior_offset_mm=float(payload.get("posterior_offset_mm", 0.0)),
            distal_thickness_mm=float(payload.get("distal_thickness_mm", 0.0)),
            side=Side(payload.get("side", "unknown")),
            compatible_sizes=tuple(payload.get("compatible_sizes", ())),
            mesh_path=payload.get("mesh_path"),
            notes=str(payload.get("notes", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ImplantComponent({self.component_id}, {self.component_type.value}, "
            f"size={self.size_label}, {self.ml_mm:.0f}x{self.ap_mm:.0f} mm)"
        )


@dataclass(frozen=True)
class ImplantConfiguration:
    """A permitted combination of components — what actually gets implanted.

    Femoral and tibial sizes are not freely mixable in a real system, and neither
    are all inserts with all baseplates. The catalog enumerates the permitted
    combinations rather than leaving the planner to guess, because an impossible
    configuration that reaches the operating theatre is a wasted case.
    """

    configuration_id: str
    family_id: str
    femoral: ImplantComponent
    tibial_baseplate: ImplantComponent
    tibial_insert: ImplantComponent | None = None
    patellar: ImplantComponent | None = None
    bearing: BearingType = BearingType.CRUCIATE_RETAINING
    fixation: FixationType = FixationType.CEMENTED
    side: Side = Side.UNKNOWN
    notes: str = ""

    def __post_init__(self) -> None:
        if self.femoral.component_type is not ComponentType.FEMORAL:
            raise ValidationError(
                f"{self.configuration_id}: femoral slot holds a "
                f"{self.femoral.component_type.value} component"
            )
        if self.tibial_baseplate.component_type is not ComponentType.TIBIAL_BASEPLATE:
            raise ValidationError(
                f"{self.configuration_id}: tibial slot holds a "
                f"{self.tibial_baseplate.component_type.value} component"
            )
        if (
            self.femoral.compatible_sizes
            and self.tibial_baseplate.size_label not in self.femoral.compatible_sizes
        ):
            raise ValidationError(
                f"{self.configuration_id}: femoral size {self.femoral.size_label} is "
                f"not compatible with tibial size {self.tibial_baseplate.size_label} "
                f"(permitted: {list(self.femoral.compatible_sizes)})"
            )

    @property
    def components(self) -> list[ImplantComponent]:
        return [
            c
            for c in (
                self.femoral,
                self.tibial_baseplate,
                self.tibial_insert,
                self.patellar,
            )
            if c is not None
        ]

    @property
    def total_construct_thickness_mm(self) -> float:
        """Combined thickness the resection must accommodate.

        Distal femoral thickness plus baseplate plus insert. This is the amount of
        bone and cartilage that must be removed for the reconstructed joint line to
        sit where it did before, which is the quantity the resection planner works
        back from.
        """
        return (
            self.femoral.distal_thickness_mm
            + self.tibial_baseplate.thickness_mm
            + (self.tibial_insert.thickness_mm if self.tibial_insert else 0.0)
        )

    #: Short codes for bearing types, used in configuration labels.
    _BEARING_CODES = {
        BearingType.CRUCIATE_RETAINING: "CR",
        BearingType.POSTERIOR_STABILISED: "PS",
        BearingType.MEDIAL_CONGRUENT: "MC",
        BearingType.CONSTRAINED: "CCK",
        BearingType.MOBILE_BEARING: "MB",
    }

    @property
    def label(self) -> str:
        """Human-readable identifier.

        Includes the bearing type: two configurations with identical component sizes
        but different bearings are different implants, and a label that cannot
        distinguish them produces a candidate list with apparent duplicates.
        """
        parts = [
            self.family_id,
            f"F{self.femoral.size_label}",
            f"T{self.tibial_baseplate.size_label}",
        ]
        if self.tibial_insert:
            parts.append(f"I{self.tibial_insert.size_label}")
        parts.append(self._BEARING_CODES.get(self.bearing, self.bearing.value))
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "family_id": self.family_id,
            "label": self.label,
            "femoral": self.femoral.component_id,
            "tibial_baseplate": self.tibial_baseplate.component_id,
            "tibial_insert": (
                self.tibial_insert.component_id if self.tibial_insert else None
            ),
            "patellar": self.patellar.component_id if self.patellar else None,
            "bearing": self.bearing.value,
            "fixation": self.fixation.value,
            "side": self.side.value,
            "total_construct_thickness_mm": round(self.total_construct_thickness_mm, 2),
            "notes": self.notes,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ImplantConfiguration({self.label}, {self.bearing.value})"


@dataclass
class ImplantFamily:
    """A product family: a set of sizes plus their permitted combinations."""

    family_id: str
    manufacturer: str
    model_name: str
    components: dict[str, ImplantComponent] = field(default_factory=dict)
    bearings: tuple[BearingType, ...] = (BearingType.CRUCIATE_RETAINING,)
    fixations: tuple[FixationType, ...] = (FixationType.CEMENTED,)

    #: Whether this family's geometry is authorised for clinical planning.
    clinical_use_permitted: bool = False

    #: Regulatory clearances this geometry is covered by, if any.
    regulatory_status: str = "not cleared - synthetic reference geometry"

    notes: str = ""

    def add(self, component: ImplantComponent) -> ImplantFamily:
        if component.family_id != self.family_id:
            raise ValidationError(
                f"component {component.component_id} belongs to family "
                f"{component.family_id!r}, not {self.family_id!r}"
            )
        if component.component_id in self.components:
            raise ValidationError(
                f"component {component.component_id} is already in family {self.family_id}"
            )
        self.components[component.component_id] = component
        return self

    def by_type(
        self, component_type: ComponentType, *, side: Side | None = None
    ) -> list[ImplantComponent]:
        found = [
            c for c in self.components.values() if c.component_type is component_type
        ]
        if side is not None:
            found = [c for c in found if c.fits_side(side)]
        return sorted(found, key=lambda c: (c.ml_mm, c.ap_mm))

    def sizes(self, component_type: ComponentType) -> list[str]:
        return [c.size_label for c in self.by_type(component_type)]

    def configurations(
        self,
        *,
        side: Side,
        bearing: BearingType | None = None,
        fixation: FixationType | None = None,
    ) -> list[ImplantConfiguration]:
        """Enumerate every permitted configuration for a side."""
        femoral = self.by_type(ComponentType.FEMORAL, side=side)
        tibial = self.by_type(ComponentType.TIBIAL_BASEPLATE, side=side)
        inserts = self.by_type(ComponentType.TIBIAL_INSERT, side=side)
        bearings = (bearing,) if bearing else self.bearings
        fixations = (fixation,) if fixation else self.fixations

        out: list[ImplantConfiguration] = []
        for f in femoral:
            for t in tibial:
                if f.compatible_sizes and t.size_label not in f.compatible_sizes:
                    continue
                matching_inserts = [
                    i
                    for i in inserts
                    if not i.compatible_sizes or t.size_label in i.compatible_sizes
                ] or [None]
                for insert in matching_inserts:
                    for b in bearings:
                        for fx in fixations:
                            suffix = f"-{insert.size_label}" if insert else ""
                            out.append(
                                ImplantConfiguration(
                                    configuration_id=(
                                        f"{self.family_id}-{f.size_label}-"
                                        f"{t.size_label}{suffix}-{b.value}-{fx.value}"
                                    ),
                                    family_id=self.family_id,
                                    femoral=f,
                                    tibial_baseplate=t,
                                    tibial_insert=insert,
                                    bearing=b,
                                    fixation=fx,
                                    side=side,
                                )
                            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "manufacturer": self.manufacturer,
            "model_name": self.model_name,
            "bearings": [b.value for b in self.bearings],
            "fixations": [f.value for f in self.fixations],
            "clinical_use_permitted": self.clinical_use_permitted,
            "regulatory_status": self.regulatory_status,
            "notes": self.notes,
            "components": [c.to_dict() for c in self.components.values()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImplantFamily:
        family = cls(
            family_id=str(payload["family_id"]),
            manufacturer=str(payload["manufacturer"]),
            model_name=str(payload["model_name"]),
            bearings=tuple(BearingType(b) for b in payload.get("bearings", ["cruciate_retaining"])),
            fixations=tuple(FixationType(f) for f in payload.get("fixations", ["cemented"])),
            clinical_use_permitted=bool(payload.get("clinical_use_permitted", False)),
            regulatory_status=str(payload.get("regulatory_status", "unknown")),
            notes=str(payload.get("notes", "")),
        )
        for item in payload.get("components", []):
            family.add(ImplantComponent.from_dict(item))
        return family


@dataclass
class ImplantCatalog:
    """A collection of implant families available to the planner."""

    catalog_id: str
    families: dict[str, ImplantFamily] = field(default_factory=dict)
    version: str = "0"
    source: str = ""

    def add(self, family: ImplantFamily) -> ImplantCatalog:
        if family.family_id in self.families:
            raise ValidationError(f"family {family.family_id} is already in the catalog")
        self.families[family.family_id] = family
        return self

    def __iter__(self) -> Iterator[ImplantFamily]:
        return iter(self.families.values())

    def __len__(self) -> int:
        return len(self.families)

    def family(self, family_id: str) -> ImplantFamily:
        found = self.families.get(family_id)
        if found is None:
            raise ImplantError(
                f"implant family {family_id!r} is not in catalog {self.catalog_id!r}. "
                f"Available: {sorted(self.families)}"
            )
        return found

    def component(self, component_id: str) -> ImplantComponent:
        for family in self.families.values():
            if component_id in family.components:
                return family.components[component_id]
        raise ImplantError(f"component {component_id!r} is not in this catalog")

    @property
    def clinical_use_permitted(self) -> bool:
        """True only if every family is authorised for clinical planning."""
        return bool(self.families) and all(
            f.clinical_use_permitted for f in self.families.values()
        )

    def require_clinical_use(self) -> ImplantCatalog:
        """Assert the catalog may be used to plan a real operation."""
        unauthorised = [
            f.family_id for f in self.families.values() if not f.clinical_use_permitted
        ]
        if unauthorised:
            raise ImplantError(
                "the following implant families are not authorised for clinical "
                f"planning: {unauthorised}. The bundled reference geometry is "
                "synthetic and dimensionally approximate; planning a real operation "
                "requires authorised manufacturer geometry (PRD §21)."
            )
        return self

    def all_configurations(
        self,
        *,
        side: Side,
        family_ids: Iterable[str] | None = None,
        bearing: BearingType | None = None,
    ) -> list[ImplantConfiguration]:
        selected = (
            [self.family(fid) for fid in family_ids]
            if family_ids is not None
            else list(self.families.values())
        )
        out: list[ImplantConfiguration] = []
        for family in selected:
            out.extend(family.configurations(side=side, bearing=bearing))
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "version": self.version,
            "source": self.source,
            "clinical_use_permitted": self.clinical_use_permitted,
            "families": [
                {
                    "family_id": f.family_id,
                    "manufacturer": f.manufacturer,
                    "model": f.model_name,
                    "clinical_use_permitted": f.clinical_use_permitted,
                    "regulatory_status": f.regulatory_status,
                    "femoral_sizes": f.sizes(ComponentType.FEMORAL),
                    "tibial_sizes": f.sizes(ComponentType.TIBIAL_BASEPLATE),
                    "insert_sizes": f.sizes(ComponentType.TIBIAL_INSERT),
                }
                for f in self.families.values()
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "version": self.version,
            "source": self.source,
            "families": [f.to_dict() for f in self.families.values()],
        }

    def write(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImplantCatalog:
        catalog = cls(
            catalog_id=str(payload["catalog_id"]),
            version=str(payload.get("version", "0")),
            source=str(payload.get("source", "")),
        )
        for item in payload.get("families", []):
            catalog.add(ImplantFamily.from_dict(item))
        return catalog


def load_catalog(reference: str | os.PathLike[str] | None = None) -> ImplantCatalog:
    """Load an implant catalog.

    ``reference`` may be:

    - ``"builtin:reference-v0"`` — the bundled synthetic reference catalog
    - a path to a JSON catalog file
    - ``None`` — read from ``KNEEAI_IMPLANT_CATALOG``
    """
    if reference is None:
        from kneeai.config import get_settings

        reference = get_settings().implant_catalog

    text = str(reference)
    if text.startswith("builtin:"):
        name = text.split(":", 1)[1]
        if name in ("reference-v0", "reference"):
            from kneeai.implants.reference import build_reference_catalog

            return build_reference_catalog()
        raise ImplantError(
            f"unknown builtin catalog {name!r}; available: reference-v0"
        )

    path = Path(text)
    if not path.exists():
        raise ImplantError(f"implant catalog not found: {path}")
    catalog = ImplantCatalog.from_dict(json.loads(path.read_text()))
    log.info(
        "implant catalog loaded",
        fields={
            "catalog": catalog.catalog_id,
            "families": len(catalog),
            "clinical": catalog.clinical_use_permitted,
        },
    )
    return catalog
