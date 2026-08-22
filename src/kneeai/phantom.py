"""Synthetic phantom knee generator.

Purpose: exercise the entire pipeline — segmentation post-processing, surface
reconstruction, landmarking, measurement, implant matching, planning, simulation
and safety validation — with **known ground truth** and **no patient data**.

Every anatomical feature is defined analytically in :class:`PhantomParams`, so
the expected landmark positions and clinical measurements are derived from the
same numbers that built the mask. That makes it possible to assert that the
measurement engine recovers, say, a 5.0° posterior tibial slope rather than
merely producing "a plausible number".

The phantom is anatomically *schematic*, not realistic. It is a correctness
fixture. A model trained on it would learn nothing useful, and its intensities
do not model any real MR sequence.

Coordinate convention (matches DICOM LPS)::

    +x -> patient left      +y -> posterior      +z -> superior

Medial/lateral are resolved from :class:`~kneeai.core.anatomy.Side`: for a right
knee medial is +x, for a left knee medial is -x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import LandmarkName, Modality, Side, Structure
from kneeai.core.frames import Frame
from kneeai.core.provenance import Origin, Provenance
from kneeai.imaging.volume import Volume

__all__ = ["PhantomParams", "PhantomKnee", "build_phantom", "PHANTOM_LABEL_MAP"]


#: Label map used by the phantom. Matches DEFAULT_LABEL_MAP ordering.
PHANTOM_LABEL_MAP: dict[Structure, int] = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
    Structure.PATELLA: 6,
    Structure.PATELLAR_CARTILAGE: 7,
    Structure.MEDIAL_MENISCUS: 8,
    Structure.LATERAL_MENISCUS: 9,
    Structure.FIBULA: 10,
}


@dataclass(frozen=True)
class PhantomParams:
    """Every dimension of the phantom, in millimetres and degrees.

    Defaults approximate an adult knee: ~76 mm femoral ML width, ~64 mm femoral
    AP, ~72 mm tibial ML, 5° posterior tibial slope, 2.5 mm cartilage.
    """

    side: Side = Side.RIGHT
    spacing_mm: float = 0.8

    # --- field of view (millimetres, IMAGE frame) ---
    x_range: tuple[float, float] = (-62.0, 62.0)
    y_range: tuple[float, float] = (-64.0, 42.0)
    z_range: tuple[float, float] = (-72.0, 72.0)

    # --- femoral condyles (medial / lateral) ---
    #: (x_offset_from_midline, y_centre, z_centre)
    #:
    #: The anteroposterior and superoinferior radii are kept close to each other,
    #: and the epicondyles are placed so the transepicondylar axis sits at a
    #: comparable distance from the distal and posterior condylar surfaces. That
    #: makes the phantom a *balanced* knee: the flexion and extension gaps a planner
    #: computes come out similar. Elongating the condyles anteroposteriorly instead
    #: produces a real several-millimetre flexion-gap tightness, which is correct
    #: behaviour from the planner but noise in a fixture meant to exercise the
    #: normal case.
    medial_condyle_centre: tuple[float, float, float] = (20.0, 0.0, 22.0)
    medial_condyle_radii: tuple[float, float, float] = (16.0, 24.0, 22.0)
    lateral_condyle_centre: tuple[float, float, float] = (-20.0, -2.0, 22.0)
    lateral_condyle_radii: tuple[float, float, float] = (15.0, 25.0, 21.0)

    # --- femoral epicondyles (define the transepicondylar axis) ---
    #: Positioned so the epicondyles are the widest part of the distal femur, as
    #: they are anatomically: epicondylar width slightly exceeds condylar width.
    #: If they did not protrude, "the most medial point of the distal femur" would
    #: land on the condylar equator instead of the epicondyle, and the phantom
    #: would silently be testing the wrong landmark.
    medial_epicondyle: tuple[float, float, float] = (29.0, -2.0, 27.0)
    lateral_epicondyle: tuple[float, float, float] = (-29.0, -4.0, 26.0)
    epicondyle_radius: float = 8.0

    # --- anterior femur (trochlea / anterior flange) ---
    #: Kept proximal to the condylar articular surface. An anterior flange
    #: extending below the condyles would become the distal-most femoral point and
    #: displace the distal condylar landmarks.
    anterior_block_centre: tuple[float, float, float] = (0.0, -12.0, 26.0)
    anterior_block_radii: tuple[float, float, float] = (24.0, 24.0, 22.0)

    # --- femoral shaft ---
    femoral_shaft_xy: tuple[float, float] = (0.0, -6.0)
    femoral_shaft_radius: float = 14.0
    femoral_shaft_z: tuple[float, float] = (30.0, 72.0)
    femoral_flare_radius: float = 24.0
    femoral_flare_z: float = 44.0

    # --- intercondylar notch (subtracted) ---
    notch_half_width: float = 9.0
    notch_y_min: float = 4.0
    notch_z_max: float = 20.0

    # --- tibial plateau ---
    plateau_centre_xy: tuple[float, float] = (0.0, -4.0)
    plateau_semi_axes: tuple[float, float] = (36.0, 25.0)
    plateau_top_z: float = -6.0
    posterior_slope_deg: float = 5.0
    plateau_bottom_z: float = -30.0

    #: Depth of the articular concavity in each compartment (medial, lateral).
    #: Real plateaus are dished. A perfectly flat plateau would make "the deepest
    #: point of the compartment" degenerate — on a sloped plane the deepest point
    #: is wherever the posterior rim happens to be — so the landmark it is meant
    #: to validate would be ill-posed.
    plateau_concavity_depth_mm: tuple[float, float] = (4.0, 3.0)
    #: Distance of each concavity centre from the midline.
    plateau_concavity_offset_mm: float = 18.0
    #: In-plane radius of the concavity.
    plateau_concavity_radius_mm: float = 15.0

    # --- tibial shaft ---
    tibial_shaft_xy: tuple[float, float] = (0.0, -8.0)
    tibial_shaft_radius: float = 13.0
    tibial_shaft_z: tuple[float, float] = (-72.0, -25.0)
    tibial_flare_radius: float = 24.0
    tibial_flare_z: float = -45.0

    #: Tibial tuberosity: (x, y, z) centre and radii.
    tuberosity_centre: tuple[float, float, float] = (0.0, -30.0, -26.0)
    tuberosity_radii: tuple[float, float, float] = (12.0, 8.0, 14.0)

    # --- fibula ---
    fibula_xy: tuple[float, float] = (-30.0, 6.0)
    fibula_radius: float = 8.0
    fibula_z: tuple[float, float] = (-72.0, -14.0)

    # --- patella ---
    patella_centre: tuple[float, float, float] = (0.0, -48.0, 22.0)
    patella_radii: tuple[float, float, float] = (22.0, 11.0, 20.0)

    # --- soft tissue thicknesses ---
    femoral_cartilage_mm: float = 2.5
    tibial_cartilage_mm: float = 2.5
    patellar_cartilage_mm: float = 3.0
    #: Peripheral height of the meniscal wedge. Chosen so each meniscus lands in
    #: the 2500-3500 mm³ range reported for adult menisci, which makes the
    #: phantom a usable check on the meniscus volume measurement.
    meniscus_height_mm: float = 5.0
    meniscus_inner_fraction: float = 0.45

    #: Simulate medial compartment cartilage loss (0 = none, 1 = full-thickness).
    #: Lets the phantom represent a varus osteoarthritic knee, which is the
    #: population the platform is aimed at.
    medial_cartilage_loss: float = 0.0

    # --- alignment reference points, as a long-leg film would supply them ---
    #: Femoral head centre. Not visible in a knee MR field of view (PRD §14), so
    #: it is provided as an external landmark exactly as it would be clinically.
    #:
    #: Chosen to produce roughly 4° of varus, a plausible TKA candidate. Note that
    #: the resulting mLDFA and MPTA are *not* physiological: the phantom's joint
    #: lines are perpendicular to its own bone axes, so the whole coronal
    #: deformity sits in the mechanical axes rather than being distributed between
    #: femoral and tibial joint-line obliquity as it is in a real varus knee. HKA
    #: and the sizing measurements are validated against truth; mLDFA and MPTA are
    #: exercised for correctness of computation, not for anatomical realism.
    hip_centre: tuple[float, float, float] = (34.0, -6.0, 400.0)
    ankle_centre: tuple[float, float, float] = (-6.0, -6.0, -390.0)

    # --- synthetic image appearance ---
    intensity_background: float = 25.0
    intensity_marrow: float = 130.0
    intensity_cortex: float = 45.0
    intensity_cartilage: float = 205.0
    intensity_meniscus: float = 42.0
    noise_sigma: float = 10.0
    bias_field_amplitude: float = 0.18
    cortex_thickness_mm: float = 1.6
    seed: int = 20260822

    @property
    def medial_sign(self) -> float:
        """+1 when medial is +x (right knee), -1 for a left knee."""
        return 1.0 if self.side is Side.RIGHT else -1.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        payload["side"] = self.side.value
        return payload


# ---------------------------------------------------------------------------
# Implicit geometry helpers. Each returns a boolean array over the grid.
# ---------------------------------------------------------------------------


def _ellipsoid(
    X: NDArray[np.float32],
    Y: NDArray[np.float32],
    Z: NDArray[np.float32],
    centre: tuple[float, float, float],
    radii: tuple[float, float, float],
    *,
    inflate: float = 0.0,
) -> NDArray[np.bool_]:
    cx, cy, cz = centre
    rx, ry, rz = (r + inflate for r in radii)
    return (
        ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2
    ) <= 1.0


def _tapered_cylinder(
    X: NDArray[np.float32],
    Y: NDArray[np.float32],
    Z: NDArray[np.float32],
    centre_xy: tuple[float, float],
    z_range: tuple[float, float],
    radius: float,
    *,
    flare_radius: float | None = None,
    flare_z: float | None = None,
) -> NDArray[np.bool_]:
    """Cylinder along z, optionally flaring towards the metaphysis."""
    cx, cy = centre_xy
    z_lo, z_hi = min(z_range), max(z_range)
    radial = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    if flare_radius is None or flare_z is None:
        local_radius = np.full_like(radial, radius)
    else:
        # Linear taper between the shaft radius and the flare radius.
        if flare_z > (z_lo + z_hi) / 2.0:  # flare towards +z
            t = np.clip((flare_z - Z) / max(flare_z - z_lo, 1e-6), 0.0, 1.0)
        else:  # flare towards -z
            t = np.clip((Z - flare_z) / max(z_hi - flare_z, 1e-6), 0.0, 1.0)
        local_radius = radius + (flare_radius - radius) * (1.0 - t)
    return (radial <= local_radius) & (Z >= z_lo) & (Z <= z_hi)


def _plateau_top(Y: NDArray[np.float32], params: PhantomParams) -> NDArray[np.float32]:
    """Height of the tibial plateau surface, sloping posteriorly."""
    _cx, cy = params.plateau_centre_xy
    slope = math.tan(math.radians(params.posterior_slope_deg))
    return params.plateau_top_z - slope * (Y - cy)


def _plateau_radial(
    X: NDArray[np.float32], Y: NDArray[np.float32], params: PhantomParams
) -> NDArray[np.float32]:
    """Normalised radius within the plateau ellipse (1.0 at the rim)."""
    cx, cy = params.plateau_centre_xy
    ax, ay = params.plateau_semi_axes
    return np.sqrt(((X - cx) / ax) ** 2 + ((Y - cy) / ay) ** 2)


# ---------------------------------------------------------------------------


@dataclass
class PhantomKnee:
    """A generated phantom: image, mask, and the ground truth used to build it."""

    image: Volume
    mask: Volume
    params: PhantomParams
    landmarks: dict[LandmarkName, NDArray[np.float64]] = field(default_factory=dict)
    truth: dict[str, float] = field(default_factory=dict)

    @property
    def side(self) -> Side:
        return self.params.side

    def landmark(self, name: LandmarkName) -> NDArray[np.float64]:
        if name not in self.landmarks:
            raise KeyError(
                f"phantom does not define {name.value}; available: "
                f"{sorted(k.value for k in self.landmarks)}"
            )
        return self.landmarks[name]

    def structure_volumes_mm3(self) -> dict[str, float]:
        return {
            s.value: round(self.mask.structure_volume_mm3(s), 1)
            for s in self.mask.structures_present()
        }

    def summary(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "spacing_mm": self.params.spacing_mm,
            "shape": list(self.mask.shape),
            "structures": self.structure_volumes_mm3(),
            "truth": {k: round(v, 3) for k, v in self.truth.items()},
        }


def build_phantom(params: PhantomParams | None = None, **overrides: Any) -> PhantomKnee:
    """Generate a phantom knee.

    Args:
        params: full parameter set. When omitted, defaults are used.
        **overrides: convenience overrides applied to the defaults, e.g.
            ``build_phantom(side=Side.LEFT, spacing_mm=1.0)``.
    """
    from dataclasses import replace as _replace

    p = params or PhantomParams()
    if overrides:
        p = _replace(p, **overrides)

    sign = p.medial_sign
    sp = float(p.spacing_mm)
    if sp <= 0:
        raise ValueError("spacing_mm must be positive")

    # --- grid ---
    xs = np.arange(p.x_range[0], p.x_range[1], sp, dtype=np.float32)
    ys = np.arange(p.y_range[0], p.y_range[1], sp, dtype=np.float32)
    zs = np.arange(p.z_range[0], p.z_range[1], sp, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    def mirror(centre: tuple[float, float, float]) -> tuple[float, float, float]:
        """Apply the medial/lateral sign to a centre's x coordinate."""
        return (centre[0] * sign, centre[1], centre[2])

    # ---------------- femur ----------------
    medial_condyle = _ellipsoid(X, Y, Z, mirror(p.medial_condyle_centre), p.medial_condyle_radii)
    lateral_condyle = _ellipsoid(
        X, Y, Z, mirror(p.lateral_condyle_centre), p.lateral_condyle_radii
    )
    anterior_block = _ellipsoid(
        X, Y, Z, mirror(p.anterior_block_centre), p.anterior_block_radii
    )
    femoral_shaft = _tapered_cylinder(
        X,
        Y,
        Z,
        (p.femoral_shaft_xy[0] * sign, p.femoral_shaft_xy[1]),
        p.femoral_shaft_z,
        p.femoral_shaft_radius,
        flare_radius=p.femoral_flare_radius,
        flare_z=p.femoral_flare_z,
    )
    medial_epi = _ellipsoid(
        X,
        Y,
        Z,
        mirror(p.medial_epicondyle),
        (p.epicondyle_radius,) * 3,
    )
    lateral_epi = _ellipsoid(
        X, Y, Z, mirror(p.lateral_epicondyle), (p.epicondyle_radius,) * 3
    )

    femur = (
        medial_condyle
        | lateral_condyle
        | anterior_block
        | femoral_shaft
        | medial_epi
        | lateral_epi
    )
    # Intercondylar notch: removes bone between the condyles, posteriorly.
    notch = (
        (np.abs(X) <= p.notch_half_width)
        & (Y >= p.notch_y_min)
        & (Z <= p.notch_z_max)
    )
    femur = femur & ~notch

    # ---------------- tibia ----------------
    plateau_top = _plateau_top(Y, p)
    plateau_radial = _plateau_radial(X, Y, p)
    plateau = (plateau_radial <= 1.0) & (Z <= plateau_top) & (Z >= p.plateau_bottom_z)

    # Dish each compartment by subtracting a large-radius sphere whose lower pole
    # sits `depth` below the plateau plane.
    concavity = np.zeros_like(plateau)
    a = p.plateau_concavity_radius_mm
    for depth, x_sign in zip(p.plateau_concavity_depth_mm, (1.0, -1.0)):
        if depth <= 0:
            continue
        sphere_radius = (a * a + depth * depth) / (2.0 * depth)
        centre = (
            x_sign * p.plateau_concavity_offset_mm * sign,
            p.plateau_centre_xy[1],
            p.plateau_top_z + sphere_radius - depth,
        )
        concavity |= _ellipsoid(X, Y, Z, centre, (sphere_radius,) * 3)
    plateau = plateau & ~concavity

    tibial_shaft = _tapered_cylinder(
        X,
        Y,
        Z,
        (p.tibial_shaft_xy[0] * sign, p.tibial_shaft_xy[1]),
        p.tibial_shaft_z,
        p.tibial_shaft_radius,
        flare_radius=p.tibial_flare_radius,
        flare_z=p.tibial_flare_z,
    )
    tuberosity = _ellipsoid(X, Y, Z, mirror(p.tuberosity_centre), p.tuberosity_radii)
    tibia = plateau | tibial_shaft | tuberosity

    # ---------------- fibula ----------------
    fibula = _tapered_cylinder(
        X,
        Y,
        Z,
        (p.fibula_xy[0] * sign, p.fibula_xy[1]),
        p.fibula_z,
        p.fibula_radius,
    )
    fibula = fibula & ~tibia

    # ---------------- patella ----------------
    patella = _ellipsoid(X, Y, Z, mirror(p.patella_centre), p.patella_radii)
    patella = patella & ~femur

    bone = femur | tibia | patella | fibula

    # ---------------- femoral cartilage ----------------
    # Shell outside the condyles, over the articular (distal/posterior) surface
    # only: cartilage does not extend onto the shaft or the epicondyles.
    condyle_shell = (
        _ellipsoid(X, Y, Z, mirror(p.medial_condyle_centre), p.medial_condyle_radii,
                   inflate=p.femoral_cartilage_mm)
        | _ellipsoid(X, Y, Z, mirror(p.lateral_condyle_centre), p.lateral_condyle_radii,
                     inflate=p.femoral_cartilage_mm)
    )
    articular_zone = Z <= (p.medial_condyle_centre[2] + 6.0)
    femoral_cartilage = condyle_shell & ~bone & articular_zone & ~notch

    if p.medial_cartilage_loss > 0.0:
        # Full- or partial-thickness loss over the medial weight-bearing zone.
        medial_side = (X * sign) > 6.0
        weight_bearing = (Z <= 4.0) & (np.abs(Y - p.medial_condyle_centre[1]) < 18.0)
        loss_zone = medial_side & weight_bearing
        if p.medial_cartilage_loss >= 1.0:
            femoral_cartilage = femoral_cartilage & ~loss_zone
        else:
            # Thin the layer rather than removing it.
            kept = _ellipsoid(
                X,
                Y,
                Z,
                mirror(p.medial_condyle_centre),
                p.medial_condyle_radii,
                inflate=p.femoral_cartilage_mm * (1.0 - p.medial_cartilage_loss),
            )
            femoral_cartilage = femoral_cartilage & ~(loss_zone & ~kept)

    # ---------------- tibial cartilage ----------------
    # Fills the articular concavity and adds a layer above the plateau plane, so
    # cartilage is thickest where the plateau is dished — as it is anatomically.
    max_dish = max(p.plateau_concavity_depth_mm) if p.plateau_concavity_depth_mm else 0.0
    tibial_cartilage_band = (
        (plateau_radial <= 1.0)
        & ~bone
        & (Z > plateau_top - max_dish - 1.0)
        & (Z <= plateau_top + p.tibial_cartilage_mm)
    )
    medial_tibial_cartilage = tibial_cartilage_band & ((X * sign) > 3.0)
    lateral_tibial_cartilage = tibial_cartilage_band & ((X * sign) < -3.0)

    if p.medial_cartilage_loss >= 1.0:
        medial_tibial_cartilage = np.zeros_like(medial_tibial_cartilage)
    elif p.medial_cartilage_loss > 0.0:
        kept_height = p.tibial_cartilage_mm * (1.0 - p.medial_cartilage_loss)
        medial_tibial_cartilage = medial_tibial_cartilage & (
            Z <= plateau_top + kept_height
        )

    # ---------------- patellar cartilage ----------------
    patellar_cartilage = (
        _ellipsoid(X, Y, Z, mirror(p.patella_centre), p.patella_radii,
                   inflate=p.patellar_cartilage_mm)
        & ~bone
        & (Y > p.patella_centre[1])  # posterior (articular) surface only
    )

    # ---------------- menisci ----------------
    # Peripheral wedge: thin at the inner edge, full height at the rim.
    inner = p.meniscus_inner_fraction
    wedge_fraction = np.clip((plateau_radial - inner) / max(1.0 - inner, 1e-6), 0.0, 1.0)
    wedge_height = p.meniscus_height_mm * wedge_fraction
    # Sits above the tibial cartilage layer rather than overlapping it, so the two
    # structures are unambiguously separable in the label map.
    meniscus_floor = plateau_top + p.tibial_cartilage_mm
    meniscus_band = (
        (plateau_radial <= 1.02)
        & (plateau_radial >= inner)
        & (Z > meniscus_floor)
        & (Z <= meniscus_floor + wedge_height)
        & ~bone
    )
    medial_meniscus = meniscus_band & ((X * sign) > 2.0)
    lateral_meniscus = meniscus_band & ((X * sign) < -2.0)

    # ---------------- assemble the label volume ----------------
    labels = np.zeros(X.shape, dtype=np.uint8)
    # Painted in ascending priority so bone always wins over soft tissue.
    for structure, binary in (
        (Structure.MEDIAL_MENISCUS, medial_meniscus),
        (Structure.LATERAL_MENISCUS, lateral_meniscus),
        (Structure.FEMORAL_CARTILAGE, femoral_cartilage),
        (Structure.MEDIAL_TIBIAL_CARTILAGE, medial_tibial_cartilage),
        (Structure.LATERAL_TIBIAL_CARTILAGE, lateral_tibial_cartilage),
        (Structure.PATELLAR_CARTILAGE, patellar_cartilage),
        (Structure.FIBULA, fibula),
        (Structure.PATELLA, patella),
        (Structure.TIBIA, tibia),
        (Structure.FEMUR, femur),
    ):
        labels[binary] = PHANTOM_LABEL_MAP[structure]

    origin = np.array([xs[0], ys[0], zs[0]], dtype=np.float64)
    spacing = np.array([sp, sp, sp], dtype=np.float64)

    provenance = Provenance.for_component(
        "imaging.preprocess",
        origin=Origin.DEFAULT,
        parameters={"operation": "build_phantom", "params": p.to_dict()},
        notes=(
            "Synthetic phantom knee. Schematic geometry for pipeline validation; "
            "not derived from any patient and not representative of real anatomy."
        ),
    )

    mask = Volume(
        array=labels,
        spacing=spacing,
        origin=origin,
        direction=np.eye(3),
        modality=Modality.MR,
        side=p.side,
        is_label=True,
        label_map=PHANTOM_LABEL_MAP,
        metadata={"phantom": True, "side": p.side.value},
        provenance=provenance,
    )

    image = _synthesise_image(labels, bone, p, mask)

    landmarks = _truth_landmarks(p)
    truth = _truth_measurements(p, landmarks)

    return PhantomKnee(
        image=image, mask=mask, params=p, landmarks=landmarks, truth=truth
    )


# ---------------------------------------------------------------------------


def _synthesise_image(
    labels: NDArray[np.uint8],
    bone: NDArray[np.bool_],
    p: PhantomParams,
    reference: Volume,
) -> Volume:
    """Build an MR-like intensity volume from the label map.

    Includes a dark cortical rim, a smooth multiplicative bias field and Gaussian
    noise, because those three are what the preprocessing stage exists to handle:
    a perfectly uniform synthetic image would let a broken normalisation step pass.
    """
    from scipy import ndimage

    rng = np.random.default_rng(p.seed)
    image = np.full(labels.shape, p.intensity_background, dtype=np.float32)

    cartilage_labels = {
        PHANTOM_LABEL_MAP[s]
        for s in (
            Structure.FEMORAL_CARTILAGE,
            Structure.MEDIAL_TIBIAL_CARTILAGE,
            Structure.LATERAL_TIBIAL_CARTILAGE,
            Structure.PATELLAR_CARTILAGE,
        )
    }
    meniscus_labels = {
        PHANTOM_LABEL_MAP[s]
        for s in (Structure.MEDIAL_MENISCUS, Structure.LATERAL_MENISCUS)
    }

    image[bone] = p.intensity_marrow
    image[np.isin(labels, list(cartilage_labels))] = p.intensity_cartilage
    image[np.isin(labels, list(meniscus_labels))] = p.intensity_meniscus

    # Cortical rim: erode the bone mask and treat the shell as low signal.
    erosion_iterations = max(1, int(round(p.cortex_thickness_mm / p.spacing_mm)))
    structure = ndimage.generate_binary_structure(3, 1)
    interior = ndimage.binary_erosion(bone, structure=structure, iterations=erosion_iterations)
    image[bone & ~interior] = p.intensity_cortex

    # Smooth multiplicative bias field (coil inhomogeneity).
    if p.bias_field_amplitude > 0:
        coarse = rng.normal(0.0, 1.0, size=(6, 6, 6)).astype(np.float32)
        zoom = [s / c for s, c in zip(labels.shape, coarse.shape)]
        field = ndimage.zoom(coarse, zoom, order=3)[
            : labels.shape[0], : labels.shape[1], : labels.shape[2]
        ]
        # zoom can come up short by a voxel; pad by edge replication.
        pad = [(0, s - f) for s, f in zip(labels.shape, field.shape)]
        field = np.pad(field, pad, mode="edge")
        spread = float(np.abs(field).max()) or 1.0
        image *= 1.0 + p.bias_field_amplitude * (field / spread)

    if p.noise_sigma > 0:
        image += rng.normal(0.0, p.noise_sigma, size=labels.shape).astype(np.float32)

    image = np.clip(image, 0.0, None)

    from dataclasses import replace as _replace

    return _replace(
        reference,
        array=image.astype(np.float32),
        is_label=False,
        label_map=None,
        metadata={"phantom": True, "synthetic_intensities": True},
    )


def _truth_landmarks(p: PhantomParams) -> dict[LandmarkName, NDArray[np.float64]]:
    """Ground-truth landmark positions, derived from the phantom's parameters."""
    sign = p.medial_sign
    slope = math.tan(math.radians(p.posterior_slope_deg))
    _cx, plateau_cy = p.plateau_centre_xy
    ax, ay = p.plateau_semi_axes

    def point(x: float, y: float, z: float) -> NDArray[np.float64]:
        return np.array([x, y, z], dtype=np.float64)

    med_c = p.medial_condyle_centre
    lat_c = p.lateral_condyle_centre
    med_r = p.medial_condyle_radii
    lat_r = p.lateral_condyle_radii

    # Distal-most and posterior-most points of each condyle ellipsoid.
    distal_medial = point(med_c[0] * sign, med_c[1], med_c[2] - med_r[2])
    distal_lateral = point(lat_c[0] * sign, lat_c[1], lat_c[2] - lat_r[2])
    posterior_medial = point(med_c[0] * sign, med_c[1] + med_r[1], med_c[2])
    posterior_lateral = point(lat_c[0] * sign, lat_c[1] + lat_r[1], lat_c[2])

    medial_epicondyle = point(
        p.medial_epicondyle[0] * sign + p.epicondyle_radius * sign,
        p.medial_epicondyle[1],
        p.medial_epicondyle[2],
    )
    lateral_epicondyle = point(
        p.lateral_epicondyle[0] * sign - p.epicondyle_radius * sign,
        p.lateral_epicondyle[1],
        p.lateral_epicondyle[2],
    )

    # Plateau low points: the bottom of each compartment's articular concavity.
    # Both concavity centres sit at the plateau's AP centre, where the sloping
    # plateau plane is exactly at plateau_top_z, so the depths subtract directly.
    depth_medial, depth_lateral = p.plateau_concavity_depth_mm
    medial_plateau_low = point(
        p.plateau_concavity_offset_mm * sign, plateau_cy, p.plateau_top_z - depth_medial
    )
    lateral_plateau_low = point(
        -p.plateau_concavity_offset_mm * sign, plateau_cy, p.plateau_top_z - depth_lateral
    )
    medial_plateau_edge = point(ax * sign, plateau_cy, p.plateau_top_z)
    lateral_plateau_edge = point(-ax * sign, plateau_cy, p.plateau_top_z)

    return {
        LandmarkName.FEMORAL_HEAD_CENTER: point(*p.hip_centre),
        LandmarkName.ANKLE_CENTER: point(*p.ankle_centre),
        LandmarkName.MEDIAL_EPICONDYLE: medial_epicondyle,
        LandmarkName.LATERAL_EPICONDYLE: lateral_epicondyle,
        LandmarkName.DISTAL_MEDIAL_CONDYLE: distal_medial,
        LandmarkName.DISTAL_LATERAL_CONDYLE: distal_lateral,
        LandmarkName.POSTERIOR_MEDIAL_CONDYLE: posterior_medial,
        LandmarkName.POSTERIOR_LATERAL_CONDYLE: posterior_lateral,
        LandmarkName.FEMORAL_KNEE_CENTER: (medial_epicondyle + lateral_epicondyle) / 2.0,
        # INTERCONDYLAR_NOTCH is deliberately absent. The phantom carves the fossa
        # as a rectangular notch, so its "roof" is a flat plane with no unique
        # apex — there is no single correct answer to compare a detector against.
        # Asserting agreement on an ill-posed quantity would be a test that passes
        # for the wrong reason.
        LandmarkName.TIBIAL_KNEE_CENTER: point(0.0, plateau_cy, p.plateau_top_z),
        LandmarkName.MEDIAL_PLATEAU_LOW: medial_plateau_low,
        LandmarkName.LATERAL_PLATEAU_LOW: lateral_plateau_low,
        LandmarkName.MEDIAL_PLATEAU_EDGE: medial_plateau_edge,
        LandmarkName.LATERAL_PLATEAU_EDGE: lateral_plateau_edge,
        LandmarkName.TIBIAL_TUBEROSITY: point(
            p.tuberosity_centre[0] * sign,
            p.tuberosity_centre[1] - p.tuberosity_radii[1],
            p.tuberosity_centre[2],
        ),
        LandmarkName.ANTERIOR_FEMORAL_CORTEX: point(
            0.0,
            p.anterior_block_centre[1] - p.anterior_block_radii[1],
            p.anterior_block_centre[2],
        ),
    }


def _truth_measurements(
    p: PhantomParams, landmarks: dict[LandmarkName, NDArray[np.float64]]
) -> dict[str, float]:
    """Expected clinical measurements, computed independently from the parameters.

    Written from the phantom's defining numbers rather than by calling the
    measurement engine, so a test comparing the two is a genuine cross-check
    rather than a tautology.
    """
    sign = p.medial_sign

    # Femoral ML: widest extent across the condyles/epicondyles.
    femoral_ml = max(
        (p.medial_condyle_centre[0] + p.medial_condyle_radii[0])
        + (abs(p.lateral_condyle_centre[0]) + p.lateral_condyle_radii[0]),
        (p.medial_epicondyle[0] + p.epicondyle_radius)
        + (abs(p.lateral_epicondyle[0]) + p.epicondyle_radius),
    )

    # Femoral AP: posterior condyle to anterior cortex.
    posterior_extent = max(
        p.medial_condyle_centre[1] + p.medial_condyle_radii[1],
        p.lateral_condyle_centre[1] + p.lateral_condyle_radii[1],
    )
    anterior_extent = min(
        p.anterior_block_centre[1] - p.anterior_block_radii[1],
        p.patella_centre[1] + p.patella_radii[1],  # patella is separate; ignore
    )
    femoral_ap = posterior_extent - (
        p.anterior_block_centre[1] - p.anterior_block_radii[1]
    )

    tibial_ml = 2.0 * p.plateau_semi_axes[0]
    tibial_ap = 2.0 * p.plateau_semi_axes[1]

    # Transepicondylar vs posterior condylar axis angle, in the axial plane.
    tea = (
        landmarks[LandmarkName.MEDIAL_EPICONDYLE]
        - landmarks[LandmarkName.LATERAL_EPICONDYLE]
    )
    pca = (
        landmarks[LandmarkName.POSTERIOR_MEDIAL_CONDYLE]
        - landmarks[LandmarkName.POSTERIOR_LATERAL_CONDYLE]
    )
    tea_xy = np.array([tea[0], tea[1]])
    pca_xy = np.array([pca[0], pca[1]])
    cos = float(
        np.dot(tea_xy, pca_xy) / (np.linalg.norm(tea_xy) * np.linalg.norm(pca_xy))
    )
    condylar_twist = math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    # Hip-knee-ankle angle. Computed in the coronal (x-z) plane, which is where a
    # long-leg film measures it.
    hip = landmarks[LandmarkName.FEMORAL_HEAD_CENTER]
    ankle = landmarks[LandmarkName.ANKLE_CENTER]
    knee = landmarks[LandmarkName.FEMORAL_KNEE_CENTER]
    femoral_mech = np.array([knee[0] - hip[0], knee[2] - hip[2]])
    tibial_mech = np.array([ankle[0] - knee[0], ankle[2] - knee[2]])
    cos_hka = float(
        np.dot(femoral_mech, tibial_mech)
        / (np.linalg.norm(femoral_mech) * np.linalg.norm(tibial_mech))
    )
    deviation = math.degrees(math.acos(max(-1.0, min(1.0, cos_hka))))
    # Sign: positive = varus. The knee centre lying lateral to the hip-ankle line
    # is varus.
    hip_to_ankle = np.array([ankle[0] - hip[0], ankle[2] - hip[2]])
    hip_to_knee = np.array([knee[0] - hip[0], knee[2] - hip[2]])
    cross = hip_to_ankle[0] * hip_to_knee[1] - hip_to_ankle[1] * hip_to_knee[0]
    varus_sign = 1.0 if (cross * sign) < 0 else -1.0

    return {
        "femoral_ml_mm": float(femoral_ml),
        "femoral_ap_mm": float(femoral_ap),
        "tibial_ml_mm": float(tibial_ml),
        "tibial_ap_mm": float(tibial_ap),
        "posterior_tibial_slope_deg": float(p.posterior_slope_deg),
        "condylar_twist_deg": float(condylar_twist),
        "hka_deviation_deg": float(deviation),
        "hka_varus_deg": float(deviation * varus_sign),
        "femoral_cartilage_thickness_mm": float(p.femoral_cartilage_mm),
        "tibial_cartilage_thickness_mm": float(p.tibial_cartilage_mm),
        "aspect_ratio_femur": float(femoral_ml / femoral_ap),
    }
