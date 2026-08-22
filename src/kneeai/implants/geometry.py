"""Solid geometry for the reference implant components.

Why this module exists
----------------------
The reference catalog (:mod:`kneeai.implants.reference`) describes each component by
its *footprint* — a 2D outline plus a thickness. That is all the fit scoring needs:
coverage and overhang are footprint properties. It is not enough for anything that
looks at the component in three dimensions.

Before this module, :func:`kneeai.simulation.geometric.component_solid` extruded the
footprint into a flat prism. For the tibial baseplate a prism is a fair likeness. For
the femoral component it is not: a real femoral component is a thin shell wrapped
around five planar cuts, with an anterior flange, a trochlear groove, two condylar
lobes and an open intercondylar space between them. Extruding its distal footprint
produces a 67 x 33 x 12 mm slab lying on the distal cut — which is why the viewer
showed a plate floating under the femur rather than a knee replacement, and why the
anterior and chamfer cuts had no component surface to correspond to.

The solids here are built from the *same* parameters the planner uses to position the
cuts — ``envelope_ap_mm``, ``posterior_offset_mm``, ``distal_thickness_mm``,
``anterior_flange_thickness_mm`` and the chamfer offset — so a component seats on the
planned cuts by construction rather than by coincidence.

**These are still not real implants.** They are procedurally generated geometry of the
right shape, scale and topology, for a catalog that is explicitly marked
``clinical_use_permitted=False``. No manufacturer surface is reproduced or implied.

Design frame
------------
Every solid is returned in the component design frame that
:class:`~kneeai.core.frames.Frame` calls ``IMPLANT``, which the planner's placement
transform maps into the image frame:

* ``+X`` mediolateral, towards the patient's right
* ``+Y`` anterior
* ``+Z`` proximal
* the origin lies **on the seating cut** — the distal femoral cut for a femoral
  component, the tibial cut for a baseplate — at the centre of the bone's
  cross-section there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from kneeai.core.errors import ImplantError
from kneeai.implants.catalog import ComponentType, ImplantComponent
from kneeai.logging import get_logger

__all__ = [
    "component_surface",
    "femoral_solid",
    "baseplate_solid",
    "insert_solid",
    "DEFAULT_CHAMFER_OFFSET_MM",
]

log = get_logger("implants.geometry")

#: Distance of each chamfer plane from the corner it cuts off. Must match
#: :data:`kneeai.planning.engine.DEFAULT_CHAMFER_OFFSET_MM`, otherwise the component's
#: chamfer faces will not lie on the planned chamfer cuts.
DEFAULT_CHAMFER_OFFSET_MM = 7.0

#: Height of the posterior condylar section above the distal cut, as a fraction of the
#: component's anteroposterior envelope. Real posterior condyles reach roughly to the
#: level of the epicondyles.
_POSTERIOR_HEIGHT_FRACTION = 0.30

#: Height of the anterior flange above the distal cut, as a fraction of the envelope.
#: The flange is the tallest part of the component; it has to resurface the trochlea.
_FLANGE_HEIGHT_FRACTION = 0.44

#: Half-width of the open intercondylar space, as a fraction of the mediolateral width.
_NOTCH_WIDTH_FRACTION = 0.30

#: Trochlear groove radius and depth, as fractions of the mediolateral width. The
#: groove has to stay shallower than the flange is thick.
_GROOVE_RADIUS_FRACTION = 0.24
_GROOVE_DEPTH_FRACTION = 0.030

#: How rectangular the transverse (mediolateral x anteroposterior) envelope is. 2 is an
#: ellipse; the condylar outline of a femoral component is fuller than that but still
#: clearly rounded at the corners.
_TRANSVERSE_EXPONENT = 3.0

#: Tibial keel length and taper, relative to the baseplate's anteroposterior depth.
_KEEL_LENGTH_FRACTION = 0.78
_KEEL_TIP_SCALE = 0.42
_KEEL_FIN_WIDTH_FRACTION = 0.16

#: Circumferential resolution used when lofting round features.
_ARC_SEGMENTS = 40


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def component_surface(component: ImplantComponent) -> Any | None:
    """Build a solid for ``component`` in the design frame.

    Returns a :class:`trimesh.Trimesh`, or ``None`` when this module has no model for
    the component's type — callers should fall back to extruding the footprint and
    report the approximation.
    """
    builders = {
        ComponentType.FEMORAL: femoral_solid,
        ComponentType.TIBIAL_BASEPLATE: baseplate_solid,
        ComponentType.TIBIAL_INSERT: insert_solid,
    }
    builder = builders.get(component.component_type)
    if builder is None:
        return None
    try:
        solid = builder(component)
    except ImplantError:
        raise
    except Exception as exc:  # noqa: BLE001 - CSG can fail on degenerate parameters
        log.warning(
            "could not build a solid for this component; the caller will fall back "
            "to an extruded footprint",
            fields={"component_id": component.component_id, "error": str(exc)},
        )
        return None

    if solid is None:
        return None
    if not solid.is_watertight:
        log.warning(
            "generated component solid is not watertight; interpenetration checks "
            "will skip it",
            fields={"component_id": component.component_id, "faces": len(solid.faces)},
        )
    return solid


# ---------------------------------------------------------------------------
# Femoral component
# ---------------------------------------------------------------------------


def femoral_solid(component: ImplantComponent) -> Any:
    """A five-cut femoral shell with an anterior flange and open intercondylar space.

    Construction, in order:

    1. The **inner** (bone-facing) profile in the sagittal plane is the five-cut box
       itself: posterior cut, posterior chamfer, distal cut, anterior chamfer,
       anterior cut. This is taken straight from the component's envelope and chamfer
       offset, so it coincides with the planner's cut planes.
    2. Each segment of that profile is thickened by *its own* wall thickness — 9 mm at
       the condyles, 3 mm at the flange — and the results unioned. A single offset
       would give a uniform wall; real components are thick where they carry load and
       thin where they only resurface. Round joins produce the smooth external
       "J-curve" between the segments.
    3. The retained-bone region is subtracted, leaving a shell.
    4. The shell is swept mediolaterally, then intersected with a rounded transverse
       envelope so the condyles taper at their medial and lateral borders instead of
       ending in square corners.
    5. The intercondylar space and the trochlear groove are subtracted.
    """
    import trimesh
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    envelope = component.envelope_ap_mm or component.ap_mm
    if not envelope or envelope <= 0:
        raise ImplantError(
            f"{component.component_id}: a femoral solid needs an anteroposterior "
            "envelope"
        )
    ml = float(component.ml_mm)
    if ml <= 0:
        raise ImplantError(f"{component.component_id}: mediolateral width must be positive")

    distal_t = float(component.distal_thickness_mm) or 9.0
    posterior_t = float(component.posterior_offset_mm) or distal_t
    flange_t = float(component.anterior_flange_thickness_mm) or 3.0
    chamfer = float(component.metadata.get("chamfer_offset_mm", DEFAULT_CHAMFER_OFFSET_MM))

    box_ap = component.internal_ap_mm  # posterior cut face to anterior cut face
    half = box_ap / 2.0
    step = chamfer * np.sqrt(2.0)  # each chamfer's projection onto the AP axis
    if 2.0 * step >= box_ap:
        raise ImplantError(
            f"{component.component_id}: chamfer offset {chamfer} mm is too large for a "
            f"{box_ap:.1f} mm cut box"
        )

    flange_h = _FLANGE_HEIGHT_FRACTION * envelope
    posterior_h = _POSTERIOR_HEIGHT_FRACTION * envelope

    # --- 1. inner profile, posterior end first, in (anterior, proximal) ---
    inner = [
        (-half, posterior_h),
        (-half, step),
        (-half + step, 0.0),
        (half - step, 0.0),
        (half, step),
        (half, flange_h),
    ]
    # Wall thickness per segment of that polyline.
    thickness = [
        posterior_t,  # posterior cut face
        posterior_t,  # posterior chamfer
        distal_t,  # distal cut
        0.5 * (distal_t + flange_t),  # anterior chamfer
        flange_t,  # anterior cut face
    ]

    bands = [
        LineString([inner[i], inner[i + 1]]).buffer(
            thickness[i], cap_style=1, join_style=1, quad_segs=8
        )
        for i in range(len(inner) - 1)
    ]
    band = unary_union(bands)

    # --- 2. remove the half that lies inside the retained bone ---
    # The retained femur is the region enclosed by the five cut faces themselves, so
    # the inner polyline *is* its boundary; closing it proximally with a tall lid
    # completes the polygon. Using a plain rectangle here instead severs the anterior
    # flange band from the rest of the shell, because the flange lies entirely
    # anterior of the anterior cut face.
    reach = max(flange_h, posterior_h) + max(thickness) + 10.0
    bone = Polygon([*inner, (half, reach), (-half, reach)])
    shell_2d = band.difference(bone)
    if shell_2d.geom_type == "MultiPolygon":
        shell_2d = max(shell_2d.geoms, key=lambda g: g.area)
    if shell_2d.is_empty:
        raise ImplantError(f"{component.component_id}: femoral shell profile is empty")
    # Unioning round buffers leaves near-duplicate vertices that make the extruded
    # prism non-manifold, which the CSG backend then refuses.
    import shapely

    shell_2d = shapely.set_precision(shell_2d, 1e-4)
    if shell_2d.geom_type == "MultiPolygon":
        shell_2d = max(shell_2d.geoms, key=lambda g: g.area)

    # --- 3. sweep mediolaterally ---
    # extrude_polygon builds a prism along +Z from a polygon in XY. The profile is in
    # (anterior, proximal), so extruding it and then mapping XYZ -> (Z, X, Y) puts the
    # sweep on the mediolateral axis and the profile on (anterior, proximal).
    swept = trimesh.creation.extrude_polygon(shell_2d, ml)
    verts = np.asarray(swept.vertices, dtype=np.float64)
    # (x, y, z) -> (z, x, y) is a cyclic permutation, so it preserves handedness and
    # the face winding must be left alone. Reversing it here turns the solid inside
    # out, and manifold3d then rejects it with "not all meshes are volumes".
    swept = trimesh.Trimesh(
        vertices=np.column_stack([verts[:, 2] - ml / 2.0, verts[:, 0], verts[:, 1]]),
        faces=np.asarray(swept.faces),
        process=False,
    )

    # --- 4. taper the medial and lateral borders ---
    # The envelope has to span the shell's full anteroposterior reach, which is
    # asymmetric: 9 mm of posterior condyle behind the posterior cut but only a 3 mm
    # flange in front of the anterior one. Centring it on zero clips the condyles.
    y_post = -(half + posterior_t) - 1.0
    y_ant = (half + flange_t) + 1.0
    outline = _superellipse(ml / 2.0, (y_ant - y_post) / 2.0, _TRANSVERSE_EXPONENT)
    outline[:, 1] += (y_ant + y_post) / 2.0
    z_lo = -(distal_t + posterior_t) - 5.0
    z_hi = flange_h + flange_t + 5.0
    envelope_solid = trimesh.creation.extrude_polygon(Polygon(outline), z_hi - z_lo)
    envelope_solid.apply_translation([0.0, 0.0, z_lo])
    solid = trimesh.boolean.intersection([swept, envelope_solid])

    # --- 5. intercondylar space ---
    notch_half = _NOTCH_WIDTH_FRACTION * ml / 2.0
    # Open from the posterior border forward to just behind the trochlear groove, so
    # the two condylar lobes are separated but the flange stays continuous.
    notch_front = -0.06 * box_ap
    notch = _rounded_box(
        x=(-notch_half, notch_half),
        y=(-half - posterior_t - 5.0, notch_front),
        z=(z_lo - 5.0, 0.0),
        radius=min(notch_half * 0.6, 6.0),
    )
    solid = trimesh.boolean.difference([solid, notch])

    # --- 6. trochlear groove ---
    groove_r = _GROOVE_RADIUS_FRACTION * ml
    groove_d = _GROOVE_DEPTH_FRACTION * ml
    if groove_d >= flange_t * 0.75:
        groove_d = flange_t * 0.5
    groove_height = flange_h + flange_t + 20.0
    groove = trimesh.creation.cylinder(radius=groove_r, height=groove_height, sections=48)
    # Axis along +Z (proximodistal), centred in the midline, pushed anteriorly so it
    # bites only ``groove_d`` into the flange's outer surface.
    groove.apply_translation(
        [0.0, half + flange_t + groove_r - groove_d, groove_height / 2.0 - 12.0]
    )
    solid = trimesh.boolean.difference([solid, groove])

    solid.metadata.update(
        {
            "component_id": component.component_id,
            "construction": "procedural five-cut shell",
            "wall_thickness_mm": {
                "distal": distal_t,
                "posterior": posterior_t,
                "anterior_flange": flange_t,
            },
            "cut_box_ap_mm": round(box_ap, 2),
            "chamfer_offset_mm": chamfer,
        }
    )
    return solid


# ---------------------------------------------------------------------------
# Tibial baseplate
# ---------------------------------------------------------------------------


def baseplate_solid(component: ImplantComponent) -> Any:
    """A tibial baseplate: the footprint plate plus a tapered cruciform keel.

    The keel occupies bone below the tibial cut. That is not an error and it is not
    flagged as one: :func:`kneeai.simulation.geometric._check_collisions` only tests a
    component against the *opposite* bone, because a component is expected to occupy
    the space of the bone it replaces. The keel slot is a preparation step, recorded in
    the robotic plan rather than in a planar cut.
    """
    import trimesh
    from shapely.geometry import Polygon

    if not component.has_footprint:
        raise ImplantError(f"{component.component_id}: a baseplate solid needs a footprint")

    plate_t = float(component.thickness_mm) or 3.0
    footprint = component.footprint_polygon()
    plate = trimesh.creation.extrude_polygon(footprint, plate_t)

    ap = float(component.ap_mm) or 45.0
    ml = float(component.ml_mm) or 70.0
    keel_len = _KEEL_LENGTH_FRACTION * ap
    fin = _KEEL_FIN_WIDTH_FRACTION * ap / 2.0

    cruciform = Polygon(
        [
            (-fin, -ap * 0.30),
            (fin, -ap * 0.30),
            (fin, -fin),
            (ml * 0.20, -fin),
            (ml * 0.20, fin),
            (fin, fin),
            (fin, ap * 0.30),
            (-fin, ap * 0.30),
            (-fin, fin),
            (-ml * 0.20, fin),
            (-ml * 0.20, -fin),
            (-fin, -fin),
        ]
    ).buffer(1.2, quad_segs=4)

    keel = trimesh.creation.extrude_polygon(cruciform, keel_len)
    kv = np.asarray(keel.vertices, dtype=np.float64)
    # extrude_polygon builds 0..keel_len along +Z; the keel goes distally, and tapers
    # towards its tip so it can be introduced into a broached slot.
    depth = kv[:, 2] / keel_len
    scale = 1.0 - (1.0 - _KEEL_TIP_SCALE) * depth
    # Negating z mirrors the solid, which does reverse handedness, so the winding has
    # to be flipped to keep the normals outward.
    keel = trimesh.Trimesh(
        vertices=np.column_stack([kv[:, 0] * scale, kv[:, 1] * scale, -kv[:, 2]]),
        faces=np.asarray(keel.faces)[:, ::-1],
        process=False,
    )

    solid = trimesh.boolean.union([plate, keel])
    solid.metadata.update(
        {
            "component_id": component.component_id,
            "construction": "procedural plate with tapered cruciform keel",
            "plate_thickness_mm": plate_t,
            "keel_length_mm": round(keel_len, 2),
        }
    )
    return solid


# ---------------------------------------------------------------------------
# Tibial insert
# ---------------------------------------------------------------------------


def insert_solid(component: ImplantComponent) -> Any:
    """A polyethylene insert with medial and lateral dished articular surfaces.

    The dishes are spherical, of a radius in the range femoral condyles are machined
    to. They are what makes the insert readable as a bearing rather than a spacer, and
    they set where the femoral condyles come to rest in extension.
    """
    import trimesh

    if not component.has_footprint:
        raise ImplantError(f"{component.component_id}: an insert solid needs a footprint")

    thickness = float(component.thickness_mm)
    if thickness <= 0:
        raise ImplantError(f"{component.component_id}: insert thickness must be positive")

    # An insert shares the baseplate's placement transform, so its own geometry has to
    # carry the offset that lifts it onto the plate. Without this it is modelled at the
    # tibial cut, occupying the same space as the baseplate.
    seats_on = float(component.metadata.get("seats_on_baseplate_mm", 0.0))

    block = trimesh.creation.extrude_polygon(component.footprint_polygon(), thickness)
    if seats_on:
        block.apply_translation([0.0, 0.0, seats_on])

    ml = float(component.ml_mm) or 70.0
    ap = float(component.ap_mm) or 45.0
    # Dish depth is a fraction of thickness so a thin insert is not perforated.
    dish_depth = min(2.4, thickness * 0.30)
    dish_r = 0.42 * ml
    offset = 0.26 * ml

    top = seats_on + thickness
    cutters = []
    for x in (-offset, offset):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=dish_r)
        sphere.apply_translation([x, -0.02 * ap, top - dish_depth + dish_r])
        cutters.append(sphere)

    solid = trimesh.boolean.difference([block, *cutters])
    solid.metadata.update(
        {
            "component_id": component.component_id,
            "construction": "procedural insert with spherical dishes",
            "dish_radius_mm": round(dish_r, 2),
            "dish_depth_mm": round(dish_depth, 2),
        }
    )
    return solid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _superellipse(a: float, b: float, exponent: float, n: int = 96) -> NDArray[np.float64]:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    power = 2.0 / exponent
    cos_t, sin_t = np.cos(t), np.sin(t)
    return np.column_stack(
        [
            a * np.sign(cos_t) * np.abs(cos_t) ** power,
            b * np.sign(sin_t) * np.abs(sin_t) ** power,
        ]
    )


def _rounded_box(
    *,
    x: tuple[float, float],
    y: tuple[float, float],
    z: tuple[float, float],
    radius: float,
) -> Any:
    """An axis-aligned box with its vertical edges rounded.

    Used for the intercondylar space: a square-edged cut there leaves a visibly
    machined slot, whereas real components blend the intercondylar walls into the
    condyles.
    """
    import trimesh
    from shapely.geometry import box as shapely_box

    profile = shapely_box(x[0], y[0], x[1], y[1])
    if radius > 0:
        profile = profile.buffer(-radius, quad_segs=6).buffer(radius, quad_segs=6)
        if profile.is_empty:
            profile = shapely_box(x[0], y[0], x[1], y[1])
    solid = trimesh.creation.extrude_polygon(profile, z[1] - z[0])
    solid.apply_translation([0.0, 0.0, z[0]])
    return solid
