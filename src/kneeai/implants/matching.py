"""Deterministic geometric implant matching (PRD §22).

    Patient geometry -> candidate implant -> registration -> coverage ->
    overhang -> undercoverage -> fit score -> ranked candidates

PRD §22 is explicit that this must **not** start as a black-box model, and the
reason is the explainability requirement in §2.2: the surgeon has to be shown *why*
a size was recommended. So every number here is computed from geometry, every score
component is reported with its weight and contribution, and the whole thing is
reproducible from the inputs.

Why overhang dominates the score
--------------------------------
Component overhang — implant extending beyond the bone edge — is weighted far more
heavily than undercoverage. Overhang irritates soft tissue and is a recognised
cause of persistent pain after knee replacement, particularly on the medial tibia.
Modest undercoverage is well tolerated. Weighting them equally would produce a
matcher that happily proposes an oversized component to maximise coverage, which
is the wrong failure direction.

The future ML ranker (PRD §23, §39) sits *behind* this, not instead of it: it
reorders candidates that already satisfy these geometric constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import Side, Structure
from kneeai.core.errors import ImplantError
from kneeai.core.frames import Frame, Transform, rotation_from_axis_angle
from kneeai.core.provenance import Provenance
from kneeai.core.units import Unit
from kneeai.implants.catalog import (
    ComponentType,
    ImplantCatalog,
    ImplantComponent,
    ImplantConfiguration,
)
from kneeai.implants.footprint import BoneFootprint
from kneeai.logging import get_logger

__all__ = [
    "FitConfig",
    "FitMethod",
    "FemoralDimensions",
    "ScoreTerm",
    "ImplantFit",
    "ConfigurationFit",
    "fit_component",
    "fit_femoral_component",
    "rank_components",
    "rank_femoral_components",
    "match_configurations",
]

log = get_logger("implants.matching")


@dataclass(frozen=True)
class FitConfig:
    """Thresholds, weights and the placement search envelope.

    Every threshold is stated here rather than buried in the scoring code, so a
    deployment can review and adjust the clinical policy in one place.
    """

    # --- hard constraints ---
    #: Overhang beyond this is a constraint violation, not merely a low score.
    max_overhang_mm: float = 3.0
    #: Coverage below this is a violation.
    min_coverage_pct: float = 70.0

    # --- scoring targets ---
    #: Coverage at which the coverage term saturates. Deliberately below 100%:
    #: a component covering the entire cut surface is almost certainly overhanging
    #: somewhere, because bone cross-sections are not implant-shaped.
    target_coverage_pct: float = 92.0
    #: Overhang up to this is treated as free (within surgical tolerance).
    overhang_tolerance_mm: float = 0.5
    #: Undercoverage up to this is treated as free.
    undercoverage_tolerance_mm: float = 2.0
    #: Undercoverage at which the term scores zero.
    undercoverage_limit_mm: float = 6.0

    # --- femoral anteroposterior envelope, asymmetric ---
    #: Undersizing tolerated before the anterior-notching penalty starts.
    femoral_ap_undersize_tolerance_mm: float = 1.0
    #: Undersizing at which the fit is rejected. Tighter than the oversize limit,
    #: because notching the anterior femoral cortex is a structural risk while mild
    #: overstuffing is a soft-tissue tension problem the surgeon can compensate for.
    femoral_ap_undersize_limit_mm: float = 4.0
    #: Oversizing tolerated before the flexion-gap penalty starts.
    femoral_ap_oversize_tolerance_mm: float = 2.0
    #: Oversizing at which the fit is rejected.
    femoral_ap_oversize_limit_mm: float = 6.0

    # --- score weights ---
    weight_overhang: float = 40.0
    weight_coverage: float = 30.0
    weight_undercoverage: float = 15.0
    weight_dimensional: float = 15.0

    # --- placement search ---
    #: In-plane translation search half-range, in millimetres. Surgeons routinely
    #: shift a baseplate a few millimetres to avoid medial overhang; this models
    #: that freedom.
    max_translation_mm: float = 4.0
    #: In-plane rotation search half-range, in degrees. Kept small because
    #: rotation is set by anatomical references (transepicondylar axis, tibial
    #: tuberosity), not chosen to improve coverage.
    max_rotation_deg: float = 3.0
    #: Grid resolution for the coarse search, before local refinement.
    coarse_translation_steps: int = 5
    coarse_rotation_steps: int = 3
    refine: bool = True

    # --- shortlist ---
    #: Only components within this many millimetres of the bone's mediolateral and
    #: anteroposterior extents are fully evaluated. Keeps the search bounded
    #: without discarding plausible candidates.
    shortlist_tolerance_mm: float = 12.0

    def __post_init__(self) -> None:
        if self.target_coverage_pct <= self.min_coverage_pct:
            raise ImplantError(
                "target_coverage_pct must exceed min_coverage_pct"
            )
        if self.undercoverage_limit_mm <= self.undercoverage_tolerance_mm:
            raise ImplantError(
                "undercoverage_limit_mm must exceed undercoverage_tolerance_mm"
            )

    @property
    def total_weight(self) -> float:
        return (
            self.weight_overhang
            + self.weight_coverage
            + self.weight_undercoverage
            + self.weight_dimensional
        )


@dataclass(frozen=True)
class ScoreTerm:
    """One component of the fit score, with everything needed to explain it."""

    name: str
    metric_value: float
    metric_unit: Unit
    score_0_100: float
    weight: float
    explanation: str

    @property
    def contribution(self) -> float:
        return self.score_0_100 * self.weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": round(self.metric_value, 3),
            "unit": self.metric_unit.value,
            "score": round(self.score_0_100, 2),
            "weight": self.weight,
            "weighted_contribution": round(self.contribution, 2),
            "explanation": self.explanation,
        }


class FitMethod(str, Enum):
    """How a component's fit was assessed.

    ``FOOTPRINT`` compares 2D polygons on a single planar cut. Correct for the
    tibial baseplate, which seats entirely on the plateau resection.

    ``DIMENSIONAL`` compares the component's mediolateral width and sagittal
    anteroposterior envelope against the bone's. Correct for the femoral component,
    which is prepared with five cuts and does not seat on one plane — its fit is
    governed by mediolateral overhang at the condyles and by whether its
    anteroposterior envelope matches the bone (too small notches the anterior
    cortex, too large overstuffs the flexion gap). Polygon-matching a femoral
    component against a single cut compares two different dimensions and produces
    a meaningless overhang figure.
    """

    FOOTPRINT = "footprint"
    DIMENSIONAL = "dimensional"


@dataclass
class FemoralDimensions:
    """The femoral measurements that determine component size.

    Taken from the measurement engine rather than from a resection cross-section,
    for the reason given in :class:`FitMethod`.
    """

    ml_width_mm: float
    ap_size_mm: float
    side: Side
    medial_ap_mm: float | None = None
    lateral_ap_mm: float | None = None
    provisional: bool = False

    @property
    def aspect_ratio(self) -> float:
        return self.ml_width_mm / max(self.ap_size_mm, 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ml_width_mm": round(self.ml_width_mm, 2),
            "ap_size_mm": round(self.ap_size_mm, 2),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "medial_ap_mm": (
                round(self.medial_ap_mm, 2) if self.medial_ap_mm is not None else None
            ),
            "lateral_ap_mm": (
                round(self.lateral_ap_mm, 2) if self.lateral_ap_mm is not None else None
            ),
            "side": self.side.value,
        }

    @classmethod
    def from_measurements(cls, measurements: Any, side: Side) -> FemoralDimensions:
        """Build from a :class:`~kneeai.measure.types.MeasurementSet`."""
        from kneeai.measure.types import MeasurementName

        return cls(
            ml_width_mm=measurements.value(MeasurementName.FEMORAL_ML_WIDTH),
            ap_size_mm=measurements.value(MeasurementName.FEMORAL_AP_SIZE),
            side=side,
            medial_ap_mm=(
                measurements.get(MeasurementName.FEMORAL_MEDIAL_AP).value
                if measurements.get(MeasurementName.FEMORAL_MEDIAL_AP)
                else None
            ),
            lateral_ap_mm=(
                measurements.get(MeasurementName.FEMORAL_LATERAL_AP).value
                if measurements.get(MeasurementName.FEMORAL_LATERAL_AP)
                else None
            ),
        )


@dataclass
class ImplantFit:
    """How well one component fits a patient's bone."""

    component: ImplantComponent
    method: FitMethod
    score: float

    #: Dimensional comparison, present for both methods.
    ml_mismatch_mm: float = 0.0
    ap_mismatch_mm: float = 0.0
    aspect_ratio_mismatch: float = 0.0

    #: Overhang per side implied by the mediolateral mismatch, assuming the
    #: component is centred. The clinically meaningful figure: a component 4 mm
    #: wider than the bone overhangs 2 mm on each side, not 4 mm.
    ml_overhang_per_side_mm: float = 0.0

    #: Polygon metrics. NaN when ``method`` is ``DIMENSIONAL``.
    coverage_pct: float = float("nan")
    overhang_max_mm: float = float("nan")
    overhang_mean_mm: float = float("nan")
    overhang_area_mm2: float = float("nan")
    overhang_perimeter_pct: float = float("nan")
    undercoverage_max_mm: float = float("nan")
    undercoverage_area_mm2: float = float("nan")
    relief_area_mm2: float = float("nan")

    #: Placement, for footprint fits.
    footprint: BoneFootprint | None = None
    translation_mm: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0
    placement: Transform | None = None

    terms: list[ScoreTerm] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return not self.violations

    @property
    def size_label(self) -> str:
        return self.component.size_label

    @property
    def provisional(self) -> bool:
        return bool(self.footprint and self.footprint.provisional)

    def rationale(self) -> list[str]:
        """Human-readable explanation, in the form PRD §2.2 asks for."""
        component_dims = f"{self.component.ml_mm:.0f} mm mediolateral"
        if self.component.envelope_ap_mm is not None:
            component_dims += (
                f", {self.component.envelope_ap_mm:.0f} mm anteroposterior envelope"
            )
        else:
            component_dims += f" x {self.component.ap_mm:.0f} mm anteroposterior"

        lines = [
            f"{self.component.component_type.value.replace('_', ' ').title()} "
            f"size {self.component.size_label} ({component_dims})",
        ]

        if self.method is FitMethod.FOOTPRINT and self.footprint is not None:
            lines.append(
                f"  bone cross-section: {self.footprint.ml_extent_mm:.1f} x "
                f"{self.footprint.ap_extent_mm:.1f} mm, "
                f"{self.footprint.area_mm2:.0f} mm²"
            )
            lines.append(
                f"  coverage {self.coverage_pct:.1f}%  "
                f"max overhang {self.overhang_max_mm:.1f} mm  "
                f"max uncovered rim {self.undercoverage_max_mm:.1f} mm"
            )
            lines.append(
                f"  placement: shifted {self.translation_mm[0]:+.1f} mm "
                f"mediolateral, {self.translation_mm[1]:+.1f} mm anteroposterior, "
                f"rotated {self.rotation_deg:+.1f}°"
            )
        else:
            lines.append(
                f"  mediolateral {self.ml_mismatch_mm:+.1f} mm vs bone "
                f"({self.ml_overhang_per_side_mm:.1f} mm overhang per side), "
                f"anteroposterior {self.ap_mismatch_mm:+.1f} mm vs bone"
            )

        lines.append(f"  fit score {self.score:.1f} / 100")
        lines.extend(f"    {t.name}: {t.explanation}" for t in self.terms)
        if self.violations:
            lines.append("  constraint violations:")
            lines.extend(f"    - {v}" for v in self.violations)
        for note in self.notes:
            lines.append(f"  note: {note}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        def clean(value: float) -> float | None:
            return None if not np.isfinite(value) else round(float(value), 3)

        return {
            "component_id": self.component.component_id,
            "size_label": self.component.size_label,
            "component_type": self.component.component_type.value,
            "method": self.method.value,
            "component_ml_mm": self.component.ml_mm,
            "component_ap_mm": self.component.ap_mm,
            "component_envelope_ap_mm": self.component.envelope_ap_mm,
            "coverage_pct": clean(self.coverage_pct),
            "overhang_max_mm": clean(self.overhang_max_mm),
            "overhang_mean_mm": clean(self.overhang_mean_mm),
            "overhang_area_mm2": clean(self.overhang_area_mm2),
            "overhang_perimeter_pct": clean(self.overhang_perimeter_pct),
            "undercoverage_max_mm": clean(self.undercoverage_max_mm),
            "undercoverage_area_mm2": clean(self.undercoverage_area_mm2),
            "relief_area_mm2": clean(self.relief_area_mm2),
            "ml_mismatch_mm": round(self.ml_mismatch_mm, 3),
            "ap_mismatch_mm": round(self.ap_mismatch_mm, 3),
            "ml_overhang_per_side_mm": round(self.ml_overhang_per_side_mm, 3),
            "aspect_ratio_mismatch": round(self.aspect_ratio_mismatch, 4),
            "translation_mm": [round(v, 3) for v in self.translation_mm],
            "rotation_deg": round(self.rotation_deg, 3),
            "score": round(self.score, 2),
            "acceptable": self.acceptable,
            "provisional": self.provisional,
            "terms": [t.to_dict() for t in self.terms],
            "violations": list(self.violations),
            "notes": list(self.notes),
            "placement": self.placement.to_dict() if self.placement else None,
        }


# ---------------------------------------------------------------------------
# Single-component fitting
# ---------------------------------------------------------------------------


def fit_component(
    footprint: BoneFootprint,
    component: ImplantComponent,
    config: FitConfig = FitConfig(),
    *,
    placement: tuple[float, float, float] | None = None,
) -> ImplantFit:
    """Fit one component to one bone footprint and score the result.

    ``placement`` pins the in-plane offset as ``(dx_mm, dy_mm, rotation_deg)`` relative
    to the seeded centroid alignment, instead of searching for the best one. That is
    what the surgeon's adjustment controls use: the same scoring, at a position they
    chose, so the coverage and overhang figures they see belong to the placement they
    are looking at rather than to the optimiser's.
    """
    from shapely.affinity import rotate as shapely_rotate
    from shapely.affinity import translate as shapely_translate

    if not component.fits_side(footprint.side):
        raise ImplantError(
            f"component {component.component_id} is a {component.side.value} part but "
            f"the footprint is from a {footprint.side.value} knee"
        )

    bone = footprint.shapely()
    implant_base = component.footprint_polygon()

    # Start with the component centred on the bone cross-section's centroid.
    bone_centroid = np.asarray(bone.centroid.coords[0], dtype=np.float64)
    implant_centroid = np.asarray(implant_base.centroid.coords[0], dtype=np.float64)
    seed_shift = bone_centroid - implant_centroid

    def place(dx: float, dy: float, theta_deg: float) -> Any:
        moved = implant_base
        if abs(theta_deg) > 1e-9:
            moved = shapely_rotate(
                moved, theta_deg, origin=tuple(implant_centroid), use_radians=False
            )
        return shapely_translate(
            moved, xoff=seed_shift[0] + dx, yoff=seed_shift[1] + dy
        )

    def objective(dx: float, dy: float, theta: float) -> float:
        placed = place(dx, dy, theta)
        overhang_area = placed.difference(bone).area
        uncovered_area = bone.difference(placed).area
        # Overhang is penalised an order of magnitude more heavily than
        # undercoverage, for the clinical reason given in the module docstring.
        return 10.0 * overhang_area + 1.0 * uncovered_area

    if placement is None:
        dx, dy, theta = _search_placement(objective, config)
    else:
        dx, dy, theta = (float(placement[0]), float(placement[1]), float(placement[2]))

    placed = place(dx, dy, theta)
    reliefs = [
        shapely_translate(
            shapely_rotate(r, theta, origin=tuple(implant_centroid), use_radians=False)
            if abs(theta) > 1e-9
            else r,
            xoff=seed_shift[0] + dx,
            yoff=seed_shift[1] + dy,
        )
        for r in component.relief_polygons()
    ]

    metrics = _footprint_metrics(bone, placed, reliefs, footprint, component)
    terms, score = _score_footprint(metrics, config)
    violations = _footprint_violations(metrics, config)

    placement = _placement_transform(
        footprint, seed_shift + np.array([dx, dy]), theta, component
    )

    fit = ImplantFit(
        component=component,
        method=FitMethod.FOOTPRINT,
        footprint=footprint,
        placement=placement,
        translation_mm=(float(dx), float(dy)),
        rotation_deg=float(theta),
        score=score,
        terms=terms,
        violations=violations,
        **metrics,
    )
    if footprint.provisional:
        fit.notes.append(
            "evaluated against a provisional resection plane; the final plan may "
            "change these numbers"
        )
    if placement is not None:
        fit.notes.append(
            "scored at a manually specified placement, not at the optimiser's"
        )
    return fit


# ---------------------------------------------------------------------------
# Femoral dimensional fitting
# ---------------------------------------------------------------------------


def fit_femoral_component(
    dimensions: FemoralDimensions,
    component: ImplantComponent,
    config: FitConfig = FitConfig(),
) -> ImplantFit:
    """Fit a femoral component by dimension rather than by footprint polygon.

    Scored on three clinically distinct failure modes:

    - **Mediolateral overhang.** A component wider than the condyles overhangs and
      irritates the collateral structures. Assessed per side.
    - **Anteroposterior undersizing.** A component whose envelope is shorter than
      the bone's requires the anterior cut to be moved posteriorly, risking a notch
      in the anterior femoral cortex — a stress riser and a recognised cause of
      periprosthetic fracture.
    - **Anteroposterior oversizing.** A component larger than the bone overstuffs
      the flexion gap and restricts flexion.

    The two anteroposterior failures are penalised **asymmetrically**, because they
    are not equally bad: notching is a structural risk, whereas mild overstuffing is
    a tension problem the surgeon can address with the tibial resection. Treating
    them symmetrically would bias the recommendation towards the more harmful side.
    """
    if not component.fits_side(dimensions.side):
        raise ImplantError(
            f"component {component.component_id} is a {component.side.value} part but "
            f"the patient's knee is {dimensions.side.value}"
        )
    if component.component_type is not ComponentType.FEMORAL:
        raise ImplantError(
            f"fit_femoral_component received a {component.component_type.value} "
            "component"
        )

    envelope = component.envelope_ap_mm
    if envelope is None:
        raise ImplantError(
            f"{component.component_id} has no envelope_ap_mm, so its anteroposterior "
            "fit cannot be assessed. Femoral components must declare their sagittal "
            "envelope."
        )

    ml_mismatch = component.ml_mm - dimensions.ml_width_mm
    ap_mismatch = envelope - dimensions.ap_size_mm
    ml_overhang_per_side = max(0.0, ml_mismatch / 2.0)

    terms: list[ScoreTerm] = []

    # --- mediolateral overhang ---
    if ml_overhang_per_side <= config.overhang_tolerance_mm:
        ml_score = 100.0
        ml_text = (
            f"mediolateral overhang {ml_overhang_per_side:.1f} mm per side is within "
            f"the {config.overhang_tolerance_mm:.1f} mm tolerance"
        )
    else:
        span = max(config.max_overhang_mm - config.overhang_tolerance_mm, 1e-6)
        ml_score = max(
            0.0,
            100.0
            * (1.0 - (ml_overhang_per_side - config.overhang_tolerance_mm) / span),
        )
        ml_text = (
            f"component is {ml_mismatch:+.1f} mm wider than the condyles, "
            f"overhanging {ml_overhang_per_side:.1f} mm per side "
            f"(limit {config.max_overhang_mm:.1f} mm)"
        )
    terms.append(
        ScoreTerm(
            "mediolateral_overhang",
            ml_overhang_per_side,
            Unit.MM,
            ml_score,
            config.weight_overhang,
            ml_text,
        )
    )

    # --- anteroposterior fit, asymmetric ---
    if ap_mismatch < -config.femoral_ap_undersize_tolerance_mm:
        deficit = abs(ap_mismatch) - config.femoral_ap_undersize_tolerance_mm
        ap_score = max(0.0, 100.0 * (1.0 - deficit / config.femoral_ap_undersize_limit_mm))
        ap_text = (
            f"component anteroposterior envelope is {abs(ap_mismatch):.1f} mm shorter "
            "than the femur, which risks notching the anterior cortex"
        )
    elif ap_mismatch > config.femoral_ap_oversize_tolerance_mm:
        excess = ap_mismatch - config.femoral_ap_oversize_tolerance_mm
        ap_score = max(0.0, 100.0 * (1.0 - excess / config.femoral_ap_oversize_limit_mm))
        ap_text = (
            f"component anteroposterior envelope is {ap_mismatch:.1f} mm longer than "
            "the femur, which will tighten the flexion gap"
        )
    else:
        ap_score = 100.0
        ap_text = (
            f"anteroposterior envelope is within tolerance ({ap_mismatch:+.1f} mm "
            "relative to the femur)"
        )
    terms.append(
        ScoreTerm(
            "anteroposterior_envelope",
            ap_mismatch,
            Unit.MM,
            ap_score,
            config.weight_coverage + config.weight_undercoverage,
            ap_text,
        )
    )

    # --- aspect ratio ---
    aspect_mismatch = component.ml_mm / envelope - dimensions.aspect_ratio
    aspect_score = float(np.clip(100.0 * (1.0 - abs(aspect_mismatch) / 0.25), 0.0, 100.0))
    terms.append(
        ScoreTerm(
            "aspect_ratio",
            aspect_mismatch,
            Unit.RATIO,
            aspect_score,
            config.weight_dimensional,
            f"component aspect ratio {component.ml_mm / envelope:.2f} versus the "
            f"femur's {dimensions.aspect_ratio:.2f}; a mismatch here is what forces a "
            "choice between mediolateral overhang and anteroposterior compromise",
        )
    )

    score = sum(t.contribution for t in terms) / sum(t.weight for t in terms)

    violations: list[str] = []
    if ml_overhang_per_side > config.max_overhang_mm:
        violations.append(
            f"mediolateral overhang {ml_overhang_per_side:.1f} mm per side exceeds the "
            f"{config.max_overhang_mm:.1f} mm limit"
        )
    if ap_mismatch < -config.femoral_ap_undersize_limit_mm:
        violations.append(
            f"anteroposterior envelope is {abs(ap_mismatch):.1f} mm short of the "
            f"femur, beyond the {config.femoral_ap_undersize_limit_mm:.1f} mm limit "
            "for anterior notching risk"
        )
    if ap_mismatch > config.femoral_ap_oversize_limit_mm:
        violations.append(
            f"anteroposterior envelope is {ap_mismatch:.1f} mm longer than the femur, "
            f"beyond the {config.femoral_ap_oversize_limit_mm:.1f} mm limit for "
            "flexion-gap overstuffing"
        )

    fit = ImplantFit(
        component=component,
        method=FitMethod.DIMENSIONAL,
        score=float(score),
        ml_mismatch_mm=float(ml_mismatch),
        ap_mismatch_mm=float(ap_mismatch),
        ml_overhang_per_side_mm=float(ml_overhang_per_side),
        aspect_ratio_mismatch=float(aspect_mismatch),
        terms=terms,
        violations=violations,
    )
    fit.notes.append(
        "assessed dimensionally: a femoral component is prepared with five cuts and "
        "does not seat on a single plane, so mediolateral overhang and "
        "anteroposterior envelope govern its fit rather than a single-plane footprint"
    )
    if dimensions.provisional:
        fit.notes.append("based on provisional femoral dimensions")
    return fit


def _search_placement(
    objective: Any, config: FitConfig
) -> tuple[float, float, float]:
    """Coarse grid search followed by optional local refinement.

    A grid first, then refinement, rather than starting a gradient-free optimiser
    from a single seed: the objective is piecewise-linear in the placement (polygon
    intersection areas) and has flat regions, so a local method from one start can
    stall in a plateau. The coarse grid is small enough to be cheap and makes the
    result deterministic.
    """
    span = config.max_translation_mm
    rotation_span = config.max_rotation_deg
    translations = np.linspace(-span, span, max(1, config.coarse_translation_steps))
    rotations = (
        np.linspace(-rotation_span, rotation_span, max(1, config.coarse_rotation_steps))
        if rotation_span > 0
        else np.zeros(1)
    )

    best_value = float("inf")
    best = (0.0, 0.0, 0.0)
    for theta in rotations:
        for dx in translations:
            for dy in translations:
                value = objective(float(dx), float(dy), float(theta))
                if value < best_value:
                    best_value = value
                    best = (float(dx), float(dy), float(theta))

    if not config.refine:
        return best

    from scipy.optimize import minimize

    result = minimize(
        lambda p: objective(float(p[0]), float(p[1]), float(p[2])),
        x0=np.asarray(best, dtype=np.float64),
        method="Powell",
        bounds=[
            (-span, span),
            (-span, span),
            (-rotation_span, rotation_span),
        ],
        options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 200},
    )
    if result.success and float(result.fun) <= best_value:
        return (float(result.x[0]), float(result.x[1]), float(result.x[2]))
    return best


def _footprint_metrics(
    bone: Any,
    placed: Any,
    reliefs: Sequence[Any],
    footprint: BoneFootprint,
    component: ImplantComponent,
) -> dict[str, float]:
    """Coverage, overhang and undercoverage for a placed component.

    Coverage and undercoverage are measured against the bone *excluding the
    component's intentional relief regions* — the posterior cruciate notch and keel
    slots. Bone left exposed inside a designed relief is not a fit defect; counting
    it would make every baseplate look as though it left 13 mm of plateau uncovered
    and would drown out the peripheral rim coverage that actually varies with size.

    Overhang is measured against the **full** bone outline: implant material outside
    the bone edge is a problem regardless of reliefs.
    """
    from shapely.ops import unary_union

    bone_area = float(bone.area)
    if bone_area <= 0:
        raise ImplantError("bone footprint has zero area")

    relief_union = unary_union(list(reliefs)) if reliefs else None
    if relief_union is not None and not relief_union.is_empty:
        effective_bone = bone.difference(relief_union)
        relief_area = float(bone.intersection(relief_union).area)
    else:
        effective_bone = bone
        relief_area = 0.0

    effective_area = float(effective_bone.area)
    if effective_area <= 0:  # pragma: no cover - defensive
        raise ImplantError(
            "the component's relief regions cover the entire bone cross-section"
        )

    coverage_pct = 100.0 * float(placed.intersection(effective_bone).area) / effective_area

    overhang_region = placed.difference(bone)
    uncovered_region = effective_bone.difference(placed)

    overhang_max, overhang_mean, overhang_perimeter_pct = _protrusion_distances(
        overhang_region, bone, placed
    )
    undercoverage_max, _mean, _perimeter = _protrusion_distances(
        uncovered_region, placed, effective_bone
    )

    ml_mismatch = component.ml_mm - footprint.ml_extent_mm
    return {
        "coverage_pct": coverage_pct,
        "overhang_max_mm": overhang_max,
        "overhang_mean_mm": overhang_mean,
        "overhang_area_mm2": float(overhang_region.area),
        "overhang_perimeter_pct": overhang_perimeter_pct,
        "undercoverage_max_mm": undercoverage_max,
        "undercoverage_area_mm2": float(uncovered_region.area),
        "relief_area_mm2": relief_area,
        "ml_mismatch_mm": ml_mismatch,
        "ap_mismatch_mm": component.ap_mm - footprint.ap_extent_mm,
        "ml_overhang_per_side_mm": max(0.0, ml_mismatch / 2.0),
        "aspect_ratio_mismatch": component.aspect_ratio - footprint.aspect_ratio,
    }


def _protrusion_distances(
    region: Any, reference: Any, protruding: Any, *, max_samples: int = 400
) -> tuple[float, float, float]:
    """Max and mean protrusion depth, plus the fraction of perimeter involved.

    Depth is measured from each protruding boundary point to the reference
    boundary, which is the quantity a surgeon cares about ("how many millimetres
    is it hanging over?") rather than the protruding area.
    """
    from shapely.geometry import Point

    if region.is_empty or region.area <= 1e-9:
        return 0.0, 0.0, 0.0

    geometries = list(region.geoms) if region.geom_type == "MultiPolygon" else [region]
    points: list[tuple[float, float]] = []
    for geometry in geometries:
        coords = list(geometry.exterior.coords)
        if len(coords) > max_samples:
            step = max(1, len(coords) // max_samples)
            coords = coords[::step]
        points.extend((float(x), float(y)) for x, y in coords)

    boundary = reference.boundary
    distances = [boundary.distance(Point(p)) for p in points]
    if not distances:  # pragma: no cover - defensive
        return 0.0, 0.0, 0.0

    perimeter = float(protruding.length)
    involved = sum(float(g.length) for g in geometries)
    # The protruding region's boundary is partly shared with the reference, so the
    # shared portion is subtracted to estimate the genuinely protruding arc.
    protruding_arc = max(0.0, involved - float(region.intersection(boundary).length))
    perimeter_pct = 100.0 * protruding_arc / perimeter if perimeter > 0 else 0.0

    return (
        float(max(distances)),
        float(np.mean(distances)),
        float(min(100.0, perimeter_pct)),
    )


def _score_footprint(
    metrics: dict[str, float], config: FitConfig
) -> tuple[list[ScoreTerm], float]:
    """Turn footprint metrics into an explainable 0-100 score."""
    terms: list[ScoreTerm] = []

    # --- overhang ---
    overhang = metrics["overhang_max_mm"]
    if overhang <= config.overhang_tolerance_mm:
        overhang_score = 100.0
        overhang_text = (
            f"maximum overhang {overhang:.1f} mm is within the "
            f"{config.overhang_tolerance_mm:.1f} mm surgical tolerance"
        )
    else:
        span = max(config.max_overhang_mm - config.overhang_tolerance_mm, 1e-6)
        overhang_score = max(
            0.0, 100.0 * (1.0 - (overhang - config.overhang_tolerance_mm) / span)
        )
        overhang_text = (
            f"maximum overhang {overhang:.1f} mm exceeds the "
            f"{config.overhang_tolerance_mm:.1f} mm tolerance "
            f"(limit {config.max_overhang_mm:.1f} mm)"
        )
    terms.append(
        ScoreTerm(
            "overhang",
            overhang,
            Unit.MM,
            overhang_score,
            config.weight_overhang,
            overhang_text,
        )
    )

    # --- coverage ---
    coverage = metrics["coverage_pct"]
    span = config.target_coverage_pct - config.min_coverage_pct
    coverage_score = float(
        np.clip(100.0 * (coverage - config.min_coverage_pct) / span, 0.0, 100.0)
    )
    terms.append(
        ScoreTerm(
            "coverage",
            coverage,
            Unit.PERCENT,
            coverage_score,
            config.weight_coverage,
            f"covers {coverage:.1f}% of the resected surface "
            f"(target {config.target_coverage_pct:.0f}%, "
            f"minimum {config.min_coverage_pct:.0f}%)",
        )
    )

    # --- undercoverage ---
    undercoverage = metrics["undercoverage_max_mm"]
    if undercoverage <= config.undercoverage_tolerance_mm:
        under_score = 100.0
        under_text = (
            f"maximum uncovered rim {undercoverage:.1f} mm is within the "
            f"{config.undercoverage_tolerance_mm:.1f} mm tolerance"
        )
    else:
        span = max(
            config.undercoverage_limit_mm - config.undercoverage_tolerance_mm, 1e-6
        )
        under_score = max(
            0.0,
            100.0 * (1.0 - (undercoverage - config.undercoverage_tolerance_mm) / span),
        )
        under_text = (
            f"up to {undercoverage:.1f} mm of the resected rim is left uncovered"
        )
    terms.append(
        ScoreTerm(
            "undercoverage",
            undercoverage,
            Unit.MM,
            under_score,
            config.weight_undercoverage,
            under_text,
        )
    )

    # --- dimensional match ---
    ml_error = abs(metrics["ml_mismatch_mm"])
    ap_error = abs(metrics["ap_mismatch_mm"])
    combined = float(np.hypot(ml_error, ap_error))
    # 8 mm combined dimensional error scores zero: that is roughly two size steps.
    dimensional_score = float(np.clip(100.0 * (1.0 - combined / 8.0), 0.0, 100.0))
    terms.append(
        ScoreTerm(
            "dimensional_match",
            combined,
            Unit.MM,
            dimensional_score,
            config.weight_dimensional,
            f"component is {metrics['ml_mismatch_mm']:+.1f} mm mediolateral and "
            f"{metrics['ap_mismatch_mm']:+.1f} mm anteroposterior relative to the "
            "bone cross-section",
        )
    )

    total = sum(t.contribution for t in terms) / config.total_weight
    return terms, float(total)


def _footprint_violations(metrics: dict[str, float], config: FitConfig) -> list[str]:
    out: list[str] = []
    if metrics["overhang_max_mm"] > config.max_overhang_mm:
        out.append(
            f"overhang {metrics['overhang_max_mm']:.1f} mm exceeds the "
            f"{config.max_overhang_mm:.1f} mm limit"
        )
    if metrics["coverage_pct"] < config.min_coverage_pct:
        out.append(
            f"coverage {metrics['coverage_pct']:.1f}% is below the "
            f"{config.min_coverage_pct:.0f}% minimum"
        )
    return out


def _placement_transform(
    footprint: BoneFootprint,
    translation_2d: NDArray[np.float64],
    rotation_deg: float,
    component: ImplantComponent,
) -> Transform:
    """Build the IMPLANT -> IMAGE transform for a placed component.

    The component's design axes map onto the footprint basis: design ``+X`` to the
    footprint's ``u`` (patient-right), design ``+Y`` to ``v`` (anterior), design
    ``+Z`` to the plane normal. In-plane rotation is applied about the normal.
    """
    basis = np.column_stack([footprint.u_axis, footprint.v_axis, footprint.plane.normal])
    rotation = basis @ rotation_from_axis_angle([0.0, 0.0, 1.0], rotation_deg)
    origin = (
        footprint.origin
        + float(translation_2d[0]) * footprint.u_axis
        + float(translation_2d[1]) * footprint.v_axis
    )
    return Transform.from_rotation_translation(
        rotation,
        origin,
        source=Frame.IMPLANT,
        target=footprint.plane.frame,
        label=f"placement[{component.component_id}]",
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_femoral_components(
    dimensions: FemoralDimensions,
    components: Iterable[ImplantComponent],
    config: FitConfig = FitConfig(),
    *,
    limit: int | None = None,
) -> list[ImplantFit]:
    """Fit and rank femoral components against a patient's femoral dimensions."""
    candidates = [c for c in components if c.fits_side(dimensions.side)]
    if not candidates:
        raise ImplantError(f"no femoral components for a {dimensions.side.value} knee")

    fits = [fit_femoral_component(dimensions, c, config) for c in candidates]
    fits.sort(key=lambda f: (not f.acceptable, -f.score))
    log.info(
        "femoral components ranked",
        fields={
            "evaluated": len(fits),
            "acceptable": sum(1 for f in fits if f.acceptable),
            "best": fits[0].size_label if fits else None,
            "best_score": round(fits[0].score, 1) if fits else None,
        },
    )
    return fits[:limit] if limit else fits


def rank_components(
    footprint: BoneFootprint,
    components: Iterable[ImplantComponent],
    config: FitConfig = FitConfig(),
    *,
    limit: int | None = None,
) -> list[ImplantFit]:
    """Fit and rank candidate components against one bone footprint.

    Components far from the bone's dimensions are shortlisted out before the
    expensive polygon fitting. Candidates that violate a hard constraint are still
    returned — ranked below acceptable ones — because PRD §27 requires that
    alternatives are not hidden from the surgeon.
    """
    candidates = [c for c in components if c.fits_side(footprint.side)]
    if not candidates:
        raise ImplantError(
            f"no candidate components for a {footprint.side.value} knee"
        )

    tolerance = config.shortlist_tolerance_mm
    shortlist = [
        c
        for c in candidates
        if abs(c.ml_mm - footprint.ml_extent_mm) <= tolerance
        and abs(c.ap_mm - footprint.ap_extent_mm) <= tolerance
    ]
    if not shortlist:
        # Rather than fail, fall back to the dimensionally nearest few and say so.
        ranked_by_distance = sorted(
            candidates,
            key=lambda c: np.hypot(
                c.ml_mm - footprint.ml_extent_mm, c.ap_mm - footprint.ap_extent_mm
            ),
        )
        shortlist = ranked_by_distance[:3]
        log.warning(
            "no component is within the shortlist tolerance of this bone; "
            "evaluating the dimensionally nearest candidates",
            fields={
                "bone_ml_mm": round(footprint.ml_extent_mm, 1),
                "bone_ap_mm": round(footprint.ap_extent_mm, 1),
                "tolerance_mm": tolerance,
                "evaluating": [c.component_id for c in shortlist],
            },
        )

    fits = [fit_component(footprint, c, config) for c in shortlist]
    # Acceptable candidates first, then by score. A violating candidate with a high
    # raw score must not outrank an acceptable one.
    fits.sort(key=lambda f: (not f.acceptable, -f.score))

    log.info(
        "components ranked",
        fields={
            "structure": footprint.structure.value,
            "evaluated": len(fits),
            "acceptable": sum(1 for f in fits if f.acceptable),
            "best": fits[0].component.size_label if fits else None,
            "best_score": round(fits[0].score, 1) if fits else None,
        },
    )
    return fits[:limit] if limit else fits


@dataclass
class ConfigurationFit:
    """A complete implantable configuration, with both component fits."""

    configuration: ImplantConfiguration
    femoral_fit: ImplantFit
    tibial_fit: ImplantFit
    score: float
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return not self.violations

    def rationale(self) -> list[str]:
        lines = [
            f"{self.configuration.label}  —  combined fit score {self.score:.1f} / 100",
            f"  bearing: {self.configuration.bearing.value.replace('_', ' ')}, "
            f"fixation: {self.configuration.fixation.value}",
            f"  construct thickness: "
            f"{self.configuration.total_construct_thickness_mm:.1f} mm",
            "",
            "  Femoral component",
        ]
        lines.extend(f"  {line}" for line in self.femoral_fit.rationale())
        lines.extend(["", "  Tibial component"])
        lines.extend(f"  {line}" for line in self.tibial_fit.rationale())
        if self.violations:
            lines.append("")
            lines.append("  configuration-level violations:")
            lines.extend(f"    - {v}" for v in self.violations)
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.to_dict(),
            "score": round(self.score, 2),
            "acceptable": self.acceptable,
            "femoral_fit": self.femoral_fit.to_dict(),
            "tibial_fit": self.tibial_fit.to_dict(),
            "violations": list(self.violations),
            "notes": list(self.notes),
        }


def match_configurations(
    femoral_dimensions: FemoralDimensions,
    tibial_footprint: BoneFootprint,
    catalog: ImplantCatalog,
    *,
    side: Side,
    config: FitConfig = FitConfig(),
    family_ids: Iterable[str] | None = None,
    limit: int = 10,
    require_clinical_catalog: bool = False,
) -> tuple[list[ConfigurationFit], Provenance]:
    """Rank complete implant configurations against a patient's geometry.

    The femoral component is assessed dimensionally and the tibial baseplate by
    footprint polygon — see :class:`FitMethod` for why the two differ.

    Component fits are computed once per component and reused across every
    configuration containing it: the fit of a femoral size does not depend on which
    insert thickness accompanies it, and the tibial polygon optimisation is the
    expensive part.
    """
    if require_clinical_catalog:
        catalog.require_clinical_use()

    if femoral_dimensions.side is not side:
        raise ImplantError(
            f"femoral dimensions are for a {femoral_dimensions.side.value} knee but "
            f"matching was requested for {side.value}"
        )
    if tibial_footprint.side is not side:
        raise ImplantError(
            f"tibial footprint is from a {tibial_footprint.side.value} knee but "
            f"matching was requested for {side.value}"
        )

    configurations = catalog.all_configurations(side=side, family_ids=family_ids)
    if not configurations:
        raise ImplantError(
            f"the catalog contains no configurations for a {side.value} knee"
        )

    femoral_cache: dict[str, ImplantFit] = {}
    tibial_cache: dict[str, ImplantFit] = {}

    def femoral_fit(component: ImplantComponent) -> ImplantFit:
        if component.component_id not in femoral_cache:
            femoral_cache[component.component_id] = fit_femoral_component(
                femoral_dimensions, component, config
            )
        return femoral_cache[component.component_id]

    def tibial_fit(component: ImplantComponent) -> ImplantFit:
        if component.component_id not in tibial_cache:
            tibial_cache[component.component_id] = fit_component(
                tibial_footprint, component, config
            )
        return tibial_cache[component.component_id]

    tolerance = config.shortlist_tolerance_mm

    def femoral_shortlisted(component: ImplantComponent) -> bool:
        envelope = component.envelope_ap_mm or component.ap_mm
        return (
            abs(component.ml_mm - femoral_dimensions.ml_width_mm) <= tolerance
            and abs(envelope - femoral_dimensions.ap_size_mm) <= tolerance
        )

    def tibial_shortlisted(component: ImplantComponent) -> bool:
        return (
            abs(component.ml_mm - tibial_footprint.ml_extent_mm) <= tolerance
            and abs(component.ap_mm - tibial_footprint.ap_extent_mm) <= tolerance
        )

    shortlisted = [
        c
        for c in configurations
        if femoral_shortlisted(c.femoral) and tibial_shortlisted(c.tibial_baseplate)
    ]
    if not shortlisted:
        log.warning(
            "no configuration is dimensionally close to this patient; evaluating all "
            "configurations",
            fields={
                "femoral_bone_mm": [
                    round(femoral_dimensions.ml_width_mm, 1),
                    round(femoral_dimensions.ap_size_mm, 1),
                ],
                "tibial_bone_mm": [
                    round(tibial_footprint.ml_extent_mm, 1),
                    round(tibial_footprint.ap_extent_mm, 1),
                ],
                "n_configurations": len(configurations),
            },
        )
        shortlisted = configurations

    results: list[ConfigurationFit] = []
    for configuration in shortlisted:
        f_fit = femoral_fit(configuration.femoral)
        t_fit = tibial_fit(configuration.tibial_baseplate)
        violations = [f"femoral: {v}" for v in f_fit.violations]
        violations.extend(f"tibial: {v}" for v in t_fit.violations)

        # The femoral and tibial components must be dimensionally consistent with
        # each other, not merely each acceptable against its own bone. A large
        # mediolateral step between them produces a construct that overhangs the
        # tibia relative to the femur.
        ml_step = abs(configuration.femoral.ml_mm - configuration.tibial_baseplate.ml_mm)
        notes: list[str] = []
        if ml_step > 10.0:
            violations.append(
                f"femoral and tibial mediolateral widths differ by {ml_step:.1f} mm, "
                "which is outside the range these components are intended to be "
                "combined across"
            )
        elif ml_step > 6.0:
            notes.append(
                f"femoral and tibial mediolateral widths differ by {ml_step:.1f} mm; "
                "confirm this combination is acceptable"
            )

        # Combined score: the worse component dominates, because a construct is only
        # as good as its poorer-fitting half. A plain mean would let an excellent
        # femoral fit mask a tibial component that overhangs.
        combined = 0.5 * min(f_fit.score, t_fit.score) + 0.5 * (
            0.5 * (f_fit.score + t_fit.score)
        )
        results.append(
            ConfigurationFit(
                configuration=configuration,
                femoral_fit=f_fit,
                tibial_fit=t_fit,
                score=float(combined),
                violations=violations,
                notes=notes,
            )
        )

    results.sort(key=lambda r: (not r.acceptable, -r.score))

    provenance = Provenance.for_component(
        "implants.matcher",
        parameters={
            "catalog_id": catalog.catalog_id,
            "catalog_clinical_use_permitted": catalog.clinical_use_permitted,
            "side": side.value,
            "configurations_available": len(configurations),
            "configurations_evaluated": len(shortlisted),
            "femoral_method": FitMethod.DIMENSIONAL.value,
            "femoral_dimensions": femoral_dimensions.to_dict(),
            "tibial_method": FitMethod.FOOTPRINT.value,
            "tibial_footprint": tibial_footprint.to_dict(),
            "fit_config": {
                "max_overhang_mm": config.max_overhang_mm,
                "min_coverage_pct": config.min_coverage_pct,
                "target_coverage_pct": config.target_coverage_pct,
                "weights": {
                    "overhang": config.weight_overhang,
                    "coverage": config.weight_coverage,
                    "undercoverage": config.weight_undercoverage,
                    "dimensional": config.weight_dimensional,
                },
                "max_translation_mm": config.max_translation_mm,
                "max_rotation_deg": config.max_rotation_deg,
            },
        },
        notes=(
            "Deterministic geometric matching. No machine-learned ranking is "
            "applied; every number is derived from the patient's own resected "
            "cross-section and the component footprint."
        ),
    )

    log.info(
        "configurations matched",
        fields={
            "evaluated": len(results),
            "acceptable": sum(1 for r in results if r.acceptable),
            "best": results[0].configuration.label if results else None,
            "best_score": round(results[0].score, 1) if results else None,
        },
    )
    return results[:limit], provenance
