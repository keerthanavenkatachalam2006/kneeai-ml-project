"""Synthetic reference implant geometry for development and testing.

**This is not a real implant.** The dimensions are plausible — they span the size
ranges published for contemporary primary total knee systems — but the footprints
are generated procedurally, not measured from any manufactured component. The
catalog is marked ``clinical_use_permitted=False`` and
:meth:`~kneeai.implants.catalog.ImplantCatalog.require_clinical_use` will refuse
it.

Its purpose is to let the matching, planning and simulation layers be built and
tested end-to-end before a manufacturer licence is in place (PRD §21, §22). The
matcher's behaviour — how coverage, overhang and fit score respond to a patient's
geometry — is exercised properly by geometry of the right *shape and scale*, which
is what this provides.

Footprint construction
----------------------
Both footprints are superellipses rather than ellipses. A real tibial baseplate
outline is fuller than an ellipse — closer to a rounded rectangle — and using an
ellipse would systematically under-report coverage at the corners of the resected
plateau, which is exactly where overhang matters clinically.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import Side
from kneeai.implants.catalog import (
    BearingType,
    ComponentType,
    FixationType,
    ImplantCatalog,
    ImplantComponent,
    ImplantFamily,
)

__all__ = [
    "build_reference_catalog",
    "REFERENCE_FEMORAL_SIZES",
    "REFERENCE_TIBIAL_SIZES",
    "REFERENCE_INSERT_THICKNESSES",
    "femoral_footprint",
    "tibial_footprint",
]

REFERENCE_FAMILY_ID = "REF-PRIMARY-CR"

#: (size label, mediolateral mm, anteroposterior mm). Spans the range covered by
#: contemporary primary femoral components.
REFERENCE_FEMORAL_SIZES: tuple[tuple[str, float, float], ...] = (
    ("1", 55.0, 50.0),
    ("2", 58.5, 53.0),
    ("3", 62.0, 56.0),
    ("4", 66.0, 59.0),
    ("5", 70.0, 62.0),
    ("6", 74.0, 65.0),
    ("7", 78.0, 68.0),
    ("8", 82.0, 71.0),
)

#: (size label, mediolateral mm, anteroposterior mm) for tibial baseplates.
#: Tibial baseplates are proportionally wider and shallower than femoral
#: components, which is why femoral and tibial sizing cannot be a single number.
REFERENCE_TIBIAL_SIZES: tuple[tuple[str, float, float], ...] = (
    ("1", 58.0, 39.0),
    ("2", 62.0, 41.5),
    ("3", 66.0, 44.0),
    ("4", 70.0, 46.5),
    ("5", 74.0, 49.0),
    ("6", 78.0, 51.5),
    ("7", 82.0, 54.0),
)

#: Polyethylene insert thicknesses, in millimetres.
REFERENCE_INSERT_THICKNESSES: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0, 14.0)

#: Metal baseplate thickness.
BASEPLATE_THICKNESS_MM = 3.0

#: Femoral distal and posterior condylar material thickness.
FEMORAL_DISTAL_THICKNESS_MM = 9.0
FEMORAL_POSTERIOR_OFFSET_MM = 9.0

#: Anterior flange thickness. Thinner than the condylar sections, as it is on real
#: components — the flange only has to resurface the trochlea.
FEMORAL_FLANGE_THICKNESS_MM = 3.0

#: Depth of the femoral component's distal-cut seating footprint, as a fraction of
#: its sagittal anteroposterior envelope. The distal cut spans from behind the
#: anterior chamfer to in front of the posterior chamfer, so it covers a little
#: over half the component's overall anteroposterior extent.
FEMORAL_SEATING_AP_FRACTION = 0.55

#: How many sizes either side a femoral component may be paired with. Real systems
#: permit limited mixing; unlimited mixing would let the matcher propose
#: combinations that cannot be assembled.
SIZE_MIX_TOLERANCE = 1


def _superellipse(
    a: float, b: float, exponent: float, n: int = 128
) -> NDArray[np.float64]:
    """Superellipse outline. ``exponent`` 2 is an ellipse; higher is more rectangular."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    power = 2.0 / exponent
    cos_t, sin_t = np.cos(t), np.sin(t)
    x = a * np.sign(cos_t) * np.abs(cos_t) ** power
    y = b * np.sign(sin_t) * np.abs(sin_t) ** power
    return np.column_stack([x, y])


def _notch_polygon(
    centre: tuple[float, float], radius: float, segments: int = 48
) -> NDArray[np.float64]:
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    return np.column_stack(
        [centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)]
    )


def _subtract_notch(
    outline: NDArray[np.float64],
    centre: tuple[float, float],
    radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remove a rounded notch, returning ``(exterior_ring, notch_polygon)``.

    The notch is returned so it can be registered as an intentional relief region
    on the component: bone left uncovered inside it is by design, not a fit defect.
    """
    from shapely.geometry import Polygon

    polygon = Polygon(outline)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    notch = _notch_polygon(centre, radius)
    cut = polygon.difference(Polygon(notch))
    if cut.geom_type == "MultiPolygon":
        cut = max(cut.geoms, key=lambda g: g.area)
    ring = np.asarray(cut.exterior.coords, dtype=np.float64)[:-1, :2]
    return ring, notch


def femoral_footprint(
    ml_mm: float, seating_ap_mm: float, side: Side
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Femoral component footprint on the **distal cut**, in the design frame.

    ``seating_ap_mm`` is the depth of the distal resection footprint, which is much
    less than the component's sagittal anteroposterior envelope — the distal cut is
    only one of the five femoral cuts.

    Returns ``(outline, intercondylar_notch)``.
    """
    a, b = ml_mm / 2.0, seating_ap_mm / 2.0
    outline = _superellipse(a, b, exponent=3.0)

    # Taper the posterior half: the posterior condyles are narrower than the
    # epicondylar width, so a uniform rectangle would overstate posterior coverage.
    posterior = outline[:, 1] < 0
    taper = 1.0 - 0.12 * (np.abs(outline[posterior, 1]) / b)
    outline[posterior, 0] *= taper

    # Slight medial/lateral asymmetry: the medial condyle is larger.
    medial_side = (outline[:, 0] * _medial_sign(side)) > 0
    outline[medial_side, 0] *= 1.02

    return _subtract_notch(outline, centre=(0.0, -b * 0.9), radius=ml_mm * 0.145)


def tibial_footprint(
    ml_mm: float, ap_mm: float, side: Side
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Tibial baseplate footprint, in the design frame.

    Asymmetric D-shape with a posterior cruciate notch — the outline family used by
    most contemporary baseplates.

    Returns ``(outline, pcl_notch)``.
    """
    a, b = ml_mm / 2.0, ap_mm / 2.0
    outline = _superellipse(a, b, exponent=2.5)

    # Medial lobe wider and more circular, lateral lobe narrower: the standard
    # asymmetry of the proximal tibia.
    medial_side = (outline[:, 0] * _medial_sign(side)) > 0
    outline[medial_side, 0] *= 1.04
    outline[~medial_side, 0] *= 0.96

    return _subtract_notch(outline, centre=(0.0, -b * 1.02), radius=ml_mm * 0.19)


def _medial_sign(side: Side) -> float:
    """Sign of the design-frame X axis that points medially, for a given knee.

    The design frame's +X is the *patient's right* (it matches the platform's
    patient frame — see :mod:`kneeai.landmarks.frame`). A right knee's medial side
    therefore faces the patient's left, i.e. -X; a left knee's medial side faces
    +X. Getting this backwards mirrors every asymmetric component, which is a
    wrong-implant error that a symmetric test fixture would never catch.
    """
    return -1.0 if side is Side.RIGHT else 1.0


def _compatible_tibial_sizes(femoral_index: int) -> tuple[str, ...]:
    lo = max(0, femoral_index - SIZE_MIX_TOLERANCE)
    hi = min(len(REFERENCE_TIBIAL_SIZES) - 1, femoral_index + SIZE_MIX_TOLERANCE)
    return tuple(REFERENCE_TIBIAL_SIZES[i][0] for i in range(lo, hi + 1))


def build_reference_catalog() -> ImplantCatalog:
    """Build the bundled non-clinical reference catalog, for both sides."""
    family = ImplantFamily(
        family_id=REFERENCE_FAMILY_ID,
        manufacturer="KneeAI Reference (synthetic)",
        model_name="Primary CR/PS Reference Geometry v0",
        bearings=(BearingType.CRUCIATE_RETAINING, BearingType.POSTERIOR_STABILISED),
        fixations=(FixationType.CEMENTED,),
        clinical_use_permitted=False,
        regulatory_status=(
            "NOT CLEARED. Synthetic geometry generated for software development and "
            "testing. Must not be used to plan treatment of any patient."
        ),
        notes=(
            "Dimensions span published primary TKA size ranges; footprints are "
            "procedurally generated superellipses, not measured components."
        ),
    )

    for side in (Side.RIGHT, Side.LEFT):
        suffix = "R" if side is Side.RIGHT else "L"

        for index, (label, ml, envelope_ap) in enumerate(REFERENCE_FEMORAL_SIZES):
            seating_ap = envelope_ap * FEMORAL_SEATING_AP_FRACTION
            outline, notch = femoral_footprint(ml, seating_ap, side)
            family.add(
                ImplantComponent(
                    component_id=f"{REFERENCE_FAMILY_ID}-FEM-{label}-{suffix}",
                    family_id=REFERENCE_FAMILY_ID,
                    component_type=ComponentType.FEMORAL,
                    size_label=label,
                    footprint=outline,
                    relief_regions=(notch,),
                    ml_mm=ml,
                    ap_mm=seating_ap,
                    envelope_ap_mm=envelope_ap,
                    anterior_flange_thickness_mm=FEMORAL_FLANGE_THICKNESS_MM,
                    thickness_mm=FEMORAL_DISTAL_THICKNESS_MM,
                    distal_thickness_mm=FEMORAL_DISTAL_THICKNESS_MM,
                    posterior_offset_mm=FEMORAL_POSTERIOR_OFFSET_MM,
                    side=side,
                    compatible_sizes=_compatible_tibial_sizes(index),
                    notes="synthetic reference geometry",
                )
            )

        for label, ml, ap in REFERENCE_TIBIAL_SIZES:
            outline, notch = tibial_footprint(ml, ap, side)
            family.add(
                ImplantComponent(
                    component_id=f"{REFERENCE_FAMILY_ID}-TIB-{label}-{suffix}",
                    family_id=REFERENCE_FAMILY_ID,
                    component_type=ComponentType.TIBIAL_BASEPLATE,
                    size_label=label,
                    footprint=outline,
                    relief_regions=(notch,),
                    ml_mm=ml,
                    ap_mm=ap,
                    thickness_mm=BASEPLATE_THICKNESS_MM,
                    side=side,
                    notes="synthetic reference geometry",
                )
            )

        # Inserts are sized to the baseplate they sit on; thickness is the variable.
        for label, ml, ap in REFERENCE_TIBIAL_SIZES:
            insert_outline, insert_notch = tibial_footprint(ml * 0.97, ap * 0.97, side)
            for thickness in REFERENCE_INSERT_THICKNESSES:
                family.add(
                    ImplantComponent(
                        component_id=(
                            f"{REFERENCE_FAMILY_ID}-INS-{label}-"
                            f"{thickness:.0f}-{suffix}"
                        ),
                        family_id=REFERENCE_FAMILY_ID,
                        component_type=ComponentType.TIBIAL_INSERT,
                        size_label=f"{label}/{thickness:.0f}",
                        footprint=insert_outline,
                        relief_regions=(insert_notch,),
                        ml_mm=ml * 0.97,
                        ap_mm=ap * 0.97,
                        thickness_mm=thickness,
                        side=side,
                        compatible_sizes=(label,),
                        notes="synthetic reference geometry",
                    )
                )

    catalog = ImplantCatalog(
        catalog_id="reference-v0",
        version="0",
        source="builtin:reference-v0 (synthetic, non-clinical)",
    )
    catalog.add(family)
    return catalog
