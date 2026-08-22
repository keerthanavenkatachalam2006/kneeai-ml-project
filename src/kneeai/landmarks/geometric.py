"""Geometric landmark detection from reconstructed bone surfaces (PRD §14, §36).

This is the deterministic baseline. It derives landmarks from mesh geometry using
the same definitions a surgeon uses — most distal point of the medial condyle,
most posterior point of the lateral condyle, deepest point of the medial plateau —
rather than learning them. Three reasons it comes first:

1. It is inspectable. Every landmark records the rule that produced it, so a
   surgeon disagreeing with a point can see why the platform chose it.
2. It needs no training data, which does not yet exist at scale with landmark
   labels.
3. It gives the eventual heatmap-regression model (PRD §36) something to be
   measured against.

Robustness
----------
Clinical landmarks are defined as extreme points, and a single stray voxel in a
segmentation produces a spike on the reconstructed surface that would then define
the landmark. Every extreme-point query here therefore uses a percentile
threshold and returns the centroid of the extreme *cluster*, plus the spread of
that cluster — which becomes the landmark's uncertainty rather than being
discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import LandmarkName, Side, Structure
from kneeai.core.errors import LandmarkError
from kneeai.core.frames import Frame
from kneeai.core.geometry import Axis, fit_line, fit_plane, unit
from kneeai.core.provenance import Origin, Provenance
from kneeai.landmarks.frame import (
    IMAGE_ANTERIOR,
    IMAGE_RIGHT,
    IMAGE_SUPERIOR,
    PatientFrame,
    build_patient_frame,
)
from kneeai.landmarks.types import Landmark, LandmarkSet
from kneeai.logging import get_logger
from kneeai.recon.mesh import AnatomicalMesh, MeshSet

__all__ = ["DetectionConfig", "detect_landmarks", "detect_shaft_axis"]

log = get_logger("landmarks.geometric")


@dataclass(frozen=True)
class DetectionConfig:
    """Region definitions, as fractions of bone length along the shaft axis."""

    #: Height of the condylar region, as a multiple of the distal femoral width.
    #:
    #: Scaled to the bone rather than expressed as a fraction of the visible bone
    #: length, deliberately. A fractional definition means the condylar region
    #: changes size when the scan happens to include more femoral shaft, so
    #: landmarks would shift with the field of view rather than with the anatomy.
    condyle_height_to_width_ratio: float = 0.62

    #: Thickness of the distal slab used to measure the condylar width.
    width_probe_slab_mm: float = 15.0

    #: Band, as a fraction of condylar height, used to locate the epicondyles.
    #: The epicondyles sit at roughly mid-condylar height; searching the whole
    #: condyle instead picks the widest point, which is not the same landmark.
    epicondyle_band: tuple[float, float] = (0.32, 0.78)

    #: Band used for the posterior condylar points.
    posterior_condyle_band: tuple[float, float] = (0.20, 0.72)

    #: Plausible range for adult distal femoral width. A measurement outside this
    #: means the width probe failed (wrong mediolateral direction, or a fragmented
    #: mesh), so the region falls back to a fraction of visible bone length.
    plausible_condyle_width_mm: tuple[float, float] = (45.0, 105.0)

    #: Fallback condylar region height, as a fraction of visible bone length, used
    #: only when the width estimate is implausible.
    fallback_condyle_fraction: float = 0.40

    #: Hard cap on the condylar region as a fraction of visible bone length.
    max_condyle_fraction: float = 0.75

    #: Band, as a fraction of condylar height, used to locate the anterior femoral
    #: cortex — the reference for anterior notching. It sits proximal to the
    #: trochlear groove, which is where the anterior femoral cut exits the bone.
    anterior_cortex_band: tuple[float, float] = (0.45, 1.0)

    #: Proximal tibia (plateau) as a fraction of tibial length from the proximal end.
    tibial_plateau_fraction: float = 0.14

    #: Minimum superior component of a vertex normal for it to count as part of
    #: the plateau's articular surface.
    plateau_normal_threshold: float = 0.35

    #: Half-width of the midline strip, in millimetres, used for midline landmarks.
    midline_half_width_mm: float = 7.0

    #: Band, as a fraction of condylar height, searched for the notch roof.
    notch_band: tuple[float, float] = (0.0, 0.62)

    #: Fraction of the plateau's radial extent treated as central articular
    #: surface. The deepest point of a compartment is sought here rather than over
    #: the whole plateau: including the rim means the "deepest point" can land on
    #: the posterior edge of a sloping plateau, or on a peripheral osteophyte,
    #: neither of which is the resection reference a surgeon uses.
    plateau_central_fraction: float = 0.72

    #: Percentile for robust extreme-point queries.
    extreme_percentile: float = 99.0

    #: Cluster spread, in millimetres, at which confidence reaches its floor.
    confidence_spread_scale_mm: float = 8.0

    #: Fraction of shaft length used for shaft-axis fitting.
    shaft_fraction: float = 0.45

    #: One refinement pass: re-derive the epicondyles once the mediolateral axis
    #: has been improved from the posterior condylar axis.
    refine_iterations: int = 1


DEFAULT_CONFIG = DetectionConfig()


# ---------------------------------------------------------------------------
# Robust extreme-point machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Extreme:
    point: NDArray[np.float64]
    spread_mm: float
    n_selected: int
    method: str

    def confidence(self, scale_mm: float) -> float:
        """Map cluster spread to a confidence in [0.25, 1.0].

        A tight cluster means the extremum is well defined. A broad one means the
        surface is flat there and "the most distal point" is genuinely ambiguous —
        which is worth telling the surgeon rather than hiding.
        """
        raw = 1.0 - (self.spread_mm / max(scale_mm, 1e-6))
        return float(min(1.0, max(0.25, raw)))


def _robust_extreme(
    points: NDArray[np.float64],
    direction: ArrayLike,
    *,
    percentile: float = 99.0,
    label: str = "",
) -> _Extreme:
    """Centroid of the vertices furthest along ``direction``, plus their spread."""
    if points.size == 0:
        raise LandmarkError(f"no candidate vertices for {label or 'extreme point'}")
    d = unit(direction)
    projections = points @ d
    threshold = float(np.percentile(projections, percentile))
    selected = points[projections >= threshold]
    if selected.shape[0] < 3:
        order = np.argsort(-projections)[: min(5, points.shape[0])]
        selected = points[order]
    centroid = selected.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.sum((selected - centroid) ** 2, axis=1))))
    return _Extreme(
        point=centroid,
        spread_mm=spread,
        n_selected=int(selected.shape[0]),
        method=f"robust_extreme({label}, p{percentile:g}, n={selected.shape[0]})",
    )


def _condylar_region(
    vertices: NDArray[np.float64],
    superior: NDArray[np.float64],
    medial_dir: NDArray[np.float64],
    config: DetectionConfig,
) -> tuple[NDArray[np.bool_], float, float]:
    """Select the distal femoral (condylar) region, scaled to the bone's own width.

    Returns ``(mask, region_height_mm, distal_reference_projection)``.

    The width is measured on a thin slab at the distal pole and the region height
    is a fixed multiple of it. That keeps every downstream band — epicondylar,
    posterior condylar, anterior cortex — anchored to anatomy rather than to how
    much femoral shaft the scan happened to include.
    """
    from kneeai.core.geometry import bounding_extent

    proj = vertices @ superior
    distal = float(proj.min())
    bone_length = float(proj.max()) - distal

    slab = vertices[proj <= distal + config.width_probe_slab_mm]
    if slab.shape[0] < 20:
        raise LandmarkError(
            "too few vertices near the distal femoral pole to estimate the "
            "condylar width"
        )
    width = bounding_extent(slab, medial_dir)
    width_lo, width_hi = config.plausible_condyle_width_mm

    if width_lo <= width <= width_hi:
        height = config.condyle_height_to_width_ratio * width
    else:
        height = config.fallback_condyle_fraction * bone_length
        log.warning(
            "distal femoral width is outside the plausible range; falling back to "
            "a fraction of visible bone length for the condylar region",
            fields={
                "measured_width_mm": round(width, 1),
                "plausible_range_mm": [width_lo, width_hi],
                "fallback_height_mm": round(height, 1),
            },
        )

    max_height = config.max_condyle_fraction * bone_length
    if max_height > 0:
        height = min(height, max_height)

    return (proj <= distal + height), height, distal


def _band(
    points: NDArray[np.float64],
    direction: ArrayLike,
    low: float,
    high: float,
    *,
    reference: float | None = None,
    span: float | None = None,
) -> NDArray[np.bool_]:
    """Boolean mask selecting points whose projection lies in a fractional band."""
    d = unit(direction)
    proj = points @ d
    base = float(proj.min()) if reference is None else reference
    extent = float(proj.max() - proj.min()) if span is None else span
    if extent <= 0:
        return np.ones(points.shape[0], dtype=bool)
    normalised = (proj - base) / extent
    return (normalised >= low) & (normalised <= high)


# ---------------------------------------------------------------------------
# Shaft axis
# ---------------------------------------------------------------------------


def detect_shaft_axis(
    mesh: AnatomicalMesh,
    *,
    superior_hint: ArrayLike = IMAGE_SUPERIOR,
    fraction: float = 0.45,
    from_proximal: bool = True,
    n_sections: int = 14,
    min_section_vertices: int = 12,
) -> tuple[Axis, float]:
    """Fit the diaphyseal (shaft) axis of a long bone.

    Method: divide the selected region into slabs perpendicular to
    ``superior_hint``, take the centroid of each slab's surface vertices, and fit
    a line through those centroids.

    A principal-component fit on the raw vertices does **not** work here, and the
    reason is worth stating: the mesh is a hollow shell, not a solid. For a
    cylindrical shell of length *L* and radius *r*, the axial variance is
    ``L²/12`` while each radial variance is ``r²/2``. For a femoral diaphysis
    visible over ~35 mm at ~14 mm radius those are 102 and 98 — indistinguishable.
    PCA then returns an essentially arbitrary radial direction. Slab centroids lie
    *on* the axis regardless of shell radius, so the line fit is well conditioned.

    Returns ``(axis, straightness)``. Straightness is ``1 - rms_residual / span``
    of the centroid line fit: 1.0 for a perfectly straight shaft, falling as the
    metaphyseal flare or a bowed diaphysis pulls the centroids off a line.

    The axis is always oriented proximally.
    """
    vertices = mesh.vertices
    hint = unit(superior_hint)
    proj = vertices @ hint
    lo, hi = float(proj.min()), float(proj.max())
    span = hi - lo
    if span <= 0:  # pragma: no cover - degenerate
        raise LandmarkError(f"{mesh.structure.value} mesh has zero extent along the hint axis")

    if from_proximal:
        band_lo, band_hi = hi - fraction * span, hi
    else:
        band_lo, band_hi = lo, lo + fraction * span

    in_band = (proj >= band_lo) & (proj <= band_hi)
    selected = vertices[in_band]
    selected_proj = proj[in_band]
    if selected.shape[0] < 50:
        raise LandmarkError(
            f"only {selected.shape[0]} vertices fall in the "
            f"{'proximal' if from_proximal else 'distal'} "
            f"{fraction:.0%} of the {mesh.structure.value}; too little shaft is "
            "visible to fit an axis"
        )

    edges = np.linspace(band_lo, band_hi, n_sections + 1)
    centroids: list[NDArray[np.float64]] = []
    for start, end in zip(edges[:-1], edges[1:]):
        slab = selected[(selected_proj >= start) & (selected_proj < end)]
        if slab.shape[0] >= min_section_vertices:
            centroids.append(slab.mean(axis=0))

    if len(centroids) < 4:
        raise LandmarkError(
            f"only {len(centroids)} usable cross-sections in the "
            f"{mesh.structure.value}; cannot fit a shaft axis"
        )

    centroid_array = np.vstack(centroids)
    axis, rms = fit_line(
        centroid_array, frame=mesh.frame, label=f"{mesh.structure.value}_shaft_axis"
    )

    direction = axis.direction
    if float(np.dot(direction, hint)) < 0:
        direction = -direction
        axis = Axis(axis.origin, direction, frame=mesh.frame, label=axis.label)

    centroid_span = float(
        (centroid_array @ direction).max() - (centroid_array @ direction).min()
    )
    straightness = (
        float(max(0.0, min(1.0, 1.0 - rms / centroid_span))) if centroid_span > 1e-6 else 0.0
    )
    return axis, straightness


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


def detect_landmarks(
    meshes: MeshSet,
    *,
    side: Side | None = None,
    external: Mapping[LandmarkName, ArrayLike] | None = None,
    config: DetectionConfig = DEFAULT_CONFIG,
) -> LandmarkSet:
    """Detect anatomical landmarks from reconstructed bone surfaces.

    Args:
        meshes: reconstructed surfaces, in the IMAGE frame. Femur and tibia are
            required.
        side: operative side. Taken from the meshes when not given.
        external: landmarks supplied from outside the volumetric study — most
            importantly the femoral head and ankle centres, which lie outside a
            knee MR field of view and come from a long-leg radiograph (PRD §14).
            These are marked ``Origin.IMPORTED``.
    """
    resolved_side = side or meshes.side
    if resolved_side is Side.UNKNOWN:
        raise LandmarkError(
            "the operative side must be known before landmark detection; medial "
            "and lateral cannot otherwise be distinguished"
        )
    if meshes.frame is not Frame.IMAGE:
        raise LandmarkError(
            f"landmark detection expects meshes in the IMAGE frame, got "
            f"'{meshes.frame.value}'"
        )

    femur = meshes.require(Structure.FEMUR)
    tibia = meshes.require(Structure.TIBIA)

    found: list[Landmark] = []

    def emit(
        name: LandmarkName,
        extreme: _Extreme | NDArray[np.float64],
        *,
        confidence: float | None = None,
        method: str = "",
    ) -> NDArray[np.float64]:
        if isinstance(extreme, _Extreme):
            position = extreme.point
            conf = (
                confidence
                if confidence is not None
                else extreme.confidence(config.confidence_spread_scale_mm)
            )
            how = method or extreme.method
            uncertainty = extreme.spread_mm
        else:
            position = np.asarray(extreme, dtype=np.float64).reshape(3)
            conf = 1.0 if confidence is None else confidence
            how = method
            uncertainty = None
        found.append(
            Landmark(
                name=name,
                position=position,
                frame=Frame.IMAGE,
                confidence=conf,
                origin=Origin.DETERMINISTIC,
                method=how,
                uncertainty_mm=uncertainty,
            )
        )
        return position

    # ---------------- externally supplied landmarks ----------------
    for name, position in (external or {}).items():
        found.append(
            Landmark(
                name=name,
                position=np.asarray(position, dtype=np.float64).reshape(3),
                frame=Frame.IMAGE,
                confidence=1.0,
                origin=Origin.IMPORTED,
                method="supplied externally (e.g. long-leg radiograph)",
            )
        )

    # ---------------- femur ----------------
    femoral_axis, femoral_straightness = _safe_shaft_axis(femur, config)
    superior = femoral_axis.direction

    # Proximal end of the visible femoral shaft: the mechanical-axis fallback used
    # by build_patient_frame when the hip centre is unavailable.
    femur_vertices = femur.vertices
    proximal = _robust_extreme(
        femur_vertices, superior, percentile=99.5, label="femoral shaft proximal end"
    )
    canal_entry = femoral_axis.closest_point(proximal.point)
    emit(
        LandmarkName.FEMORAL_CANAL_ENTRY,
        canal_entry,
        confidence=min(1.0, max(0.3, femoral_straightness)),
        method=(
            f"proximal end of the fitted femoral shaft axis "
            f"(straightness {femoral_straightness:.3f})"
        ),
    )

    # Provisional mediolateral direction from the side; refined below.
    medial_dir = IMAGE_RIGHT * (-resolved_side.sign)
    lateral_dir = -medial_dir

    condyle_mask, condyle_height, distal_reference = _condylar_region(
        femur_vertices, superior, medial_dir, config
    )
    condyles = femur_vertices[condyle_mask]
    if condyles.shape[0] < 100:
        raise LandmarkError(
            f"only {condyles.shape[0]} femoral vertices lie within {condyle_height:.0f} mm "
            "of the distal femoral pole; the distal femur appears to be outside the "
            "field of view"
        )

    posterior_dir = -_anterior_direction(superior, medial_dir)

    for iteration in range(config.refine_iterations + 1):
        medial_half, lateral_half = _split_compartments(condyles, medial_dir)

        posterior_band_med = medial_half[
            _band(
                medial_half,
                superior,
                *config.posterior_condyle_band,
                reference=distal_reference,
                span=condyle_height,
            )
        ]
        posterior_band_lat = lateral_half[
            _band(
                lateral_half,
                superior,
                *config.posterior_condyle_band,
                reference=distal_reference,
                span=condyle_height,
            )
        ]
        post_med = _robust_extreme(
            posterior_band_med,
            posterior_dir,
            percentile=config.extreme_percentile,
            label="posterior medial condyle",
        )
        post_lat = _robust_extreme(
            posterior_band_lat,
            posterior_dir,
            percentile=config.extreme_percentile,
            label="posterior lateral condyle",
        )

        if iteration < config.refine_iterations:
            # Refine the mediolateral direction using the posterior condylar axis,
            # which is a real anatomical axis rather than an image-frame guess.
            refined = post_med.point - post_lat.point
            if float(np.linalg.norm(refined)) > 20.0:
                candidate = unit(refined - float(np.dot(unit(refined), superior)) * superior)
                # Only accept the refinement if it agrees with the side-derived
                # direction. A large disagreement means a condyle was mis-split.
                if float(np.dot(candidate, medial_dir)) > 0.9:
                    medial_dir = candidate
                    lateral_dir = -medial_dir
                    posterior_dir = -_anterior_direction(superior, medial_dir)
                else:
                    log.warning(
                        "posterior condylar axis disagrees with the side-derived "
                        "mediolateral direction; keeping the side-derived axis",
                        fields={
                            "agreement": round(float(np.dot(candidate, medial_dir)), 4)
                        },
                    )

    anterior_dir = _anterior_direction(superior, medial_dir)

    emit(LandmarkName.POSTERIOR_MEDIAL_CONDYLE, post_med)
    emit(LandmarkName.POSTERIOR_LATERAL_CONDYLE, post_lat)

    inferior = -superior
    distal_med = _robust_extreme(
        medial_half, inferior, percentile=config.extreme_percentile,
        label="distal medial condyle",
    )
    distal_lat = _robust_extreme(
        lateral_half, inferior, percentile=config.extreme_percentile,
        label="distal lateral condyle",
    )
    emit(LandmarkName.DISTAL_MEDIAL_CONDYLE, distal_med)
    emit(LandmarkName.DISTAL_LATERAL_CONDYLE, distal_lat)

    # Epicondyles, restricted to mid-condylar height.
    epi_band_med = medial_half[
        _band(
            medial_half,
            superior,
            *config.epicondyle_band,
            reference=distal_reference,
            span=condyle_height,
        )
    ]
    epi_band_lat = lateral_half[
        _band(
            lateral_half,
            superior,
            *config.epicondyle_band,
            reference=distal_reference,
            span=condyle_height,
        )
    ]
    med_epi = _robust_extreme(
        epi_band_med, medial_dir, percentile=config.extreme_percentile,
        label="medial epicondyle",
    )
    lat_epi = _robust_extreme(
        epi_band_lat, lateral_dir, percentile=config.extreme_percentile,
        label="lateral epicondyle",
    )
    med_epi_point = emit(LandmarkName.MEDIAL_EPICONDYLE, med_epi)
    lat_epi_point = emit(LandmarkName.LATERAL_EPICONDYLE, lat_epi)

    emit(
        LandmarkName.FEMORAL_KNEE_CENTER,
        (med_epi_point + lat_epi_point) / 2.0,
        confidence=min(
            med_epi.confidence(config.confidence_spread_scale_mm),
            lat_epi.confidence(config.confidence_spread_scale_mm),
        ),
        method="midpoint of the surgical transepicondylar axis",
    )

    # Intercondylar notch roof: the most proximal bone at the posterior midline,
    # bounded in height so the search cannot climb onto the anterior flange or the
    # supracondylar shaft, both of which sit at the midline and are more proximal
    # than the notch.
    notch_region = condyles[
        _band(
            condyles,
            superior,
            *config.notch_band,
            reference=distal_reference,
            span=condyle_height,
        )
    ]
    midline = _midline_strip(
        notch_region, medial_dir, med_epi_point, lat_epi_point, config
    )
    if midline.shape[0] >= 5:
        ap_projection = condyles @ posterior_dir
        ap_midpoint = (float(ap_projection.min()) + float(ap_projection.max())) / 2.0
        posterior_midline = midline[(midline @ posterior_dir) > ap_midpoint]
        if posterior_midline.shape[0] >= 5:
            notch = _robust_extreme(
                posterior_midline,
                superior,
                percentile=96.0,
                label="intercondylar notch roof",
            )
            # Capped at 0.5 regardless of cluster tightness. The intercondylar
            # fossa is a saddle, not an extremum, so "the most superior point of
            # the posterior midline" is an approximation of the roof rather than a
            # definition of it — the value drifts with where the condyles are
            # judged to end. Nothing downstream in this build consumes it; it is
            # emitted as a visual reference, and its confidence says so.
            emit(
                LandmarkName.INTERCONDYLAR_NOTCH,
                notch,
                confidence=min(
                    0.5, notch.confidence(config.confidence_spread_scale_mm)
                ),
                method=(
                    f"{notch.method} — approximate: the fossa is a saddle, not an "
                    "extremum. Confirm before using as a reference."
                ),
            )

    # Anterior femoral cortex, above the trochlea: the reference for anterior
    # notching when the femoral component is planned.
    cortex_mask = _band(
        femur_vertices,
        superior,
        config.anterior_cortex_band[0],
        config.anterior_cortex_band[1],
        reference=distal_reference,
        span=condyle_height,
    )
    cortex_region = femur_vertices[cortex_mask]
    if cortex_region.shape[0] >= 20:
        cortex_midline = _midline_strip(
            cortex_region, medial_dir, med_epi_point, lat_epi_point, config
        )
        target = cortex_midline if cortex_midline.shape[0] >= 10 else cortex_region
        cortex = _robust_extreme(
            target, anterior_dir, percentile=98.0, label="anterior femoral cortex"
        )
        emit(LandmarkName.ANTERIOR_FEMORAL_CORTEX, cortex)

    # ---------------- tibia ----------------
    tibial_axis, tibial_straightness = _safe_shaft_axis(
        tibia, config, from_proximal=False, superior_hint=superior
    )
    tibia_vertices = tibia.vertices
    tibial_superior = tibial_axis.direction

    plateau_mask = _band(
        tibia_vertices, tibial_superior, 1.0 - config.tibial_plateau_fraction, 1.0
    )
    plateau = tibia_vertices[plateau_mask]
    if plateau.shape[0] < 100:
        raise LandmarkError(
            f"only {plateau.shape[0]} tibial vertices lie in the proximal "
            f"{config.tibial_plateau_fraction:.0%} of the bone; the plateau appears "
            "to be outside the field of view"
        )

    # Articular surface: plateau vertices whose normal points proximally.
    plateau_surface = _upward_facing(tibia, plateau_mask, tibial_superior, config)
    if plateau_surface.shape[0] < 50:
        log.warning(
            "few upward-facing plateau vertices found; falling back to the full "
            "plateau region for plane fitting",
            fields={"n_upward": int(plateau_surface.shape[0])},
        )
        plateau_surface = plateau

    plateau_plane, plateau_rms = fit_plane(
        plateau_surface, frame=Frame.IMAGE, label="tibial_plateau"
    )

    # Deepest point of each compartment, restricted to the central articular
    # surface so the peripheral rim cannot win.
    central = _plateau_central_mask(
        plateau_surface, medial_dir, anterior_dir, config.plateau_central_fraction
    )
    central_surface = plateau_surface[central]
    if central_surface.shape[0] < 40:
        log.warning(
            "central plateau region is sparse; searching the full articular surface",
            fields={"n_central": int(central_surface.shape[0])},
        )
        central_surface = plateau_surface

    central_medial, central_lateral = _split_compartments(central_surface, medial_dir)
    low_med = _robust_extreme(
        central_medial, -tibial_superior, percentile=97.0,
        label="deepest point of the central medial plateau",
    )
    low_lat = _robust_extreme(
        central_lateral, -tibial_superior, percentile=97.0,
        label="deepest point of the central lateral plateau",
    )
    emit(LandmarkName.MEDIAL_PLATEAU_LOW, low_med)
    emit(LandmarkName.LATERAL_PLATEAU_LOW, low_lat)

    # Rim landmarks are taken on the articular surface, not the whole proximal
    # region: the widest point of the metaphysis below the joint line is not the
    # plateau rim.
    edge_med = _robust_extreme(
        plateau_surface, medial_dir, percentile=config.extreme_percentile,
        label="medial plateau rim",
    )
    edge_lat = _robust_extreme(
        plateau_surface, lateral_dir, percentile=config.extreme_percentile,
        label="lateral plateau rim",
    )
    emit(LandmarkName.MEDIAL_PLATEAU_EDGE, edge_med)
    emit(LandmarkName.LATERAL_PLATEAU_EDGE, edge_lat)

    # Tibial knee centre: centre of the plateau's mediolateral and anteroposterior
    # extents, projected onto the fitted plateau plane.
    ml_centre = (edge_med.point + edge_lat.point) / 2.0
    ap_anterior = _robust_extreme(plateau, anterior_dir, percentile=98.0, label="anterior plateau rim")
    ap_posterior = _robust_extreme(plateau, -anterior_dir, percentile=98.0, label="posterior plateau rim")
    ap_centre = (ap_anterior.point + ap_posterior.point) / 2.0
    centre_seed = np.array(
        [
            float(np.dot(ml_centre, medial_dir)) * medial_dir
            + float(np.dot(ap_centre, anterior_dir)) * anterior_dir
            + float(np.dot(ml_centre, tibial_superior)) * tibial_superior
        ]
    ).reshape(3)
    emit(
        LandmarkName.TIBIAL_KNEE_CENTER,
        plateau_plane.project(centre_seed),
        confidence=0.9,
        method="centre of the plateau extents, projected onto the fitted plateau plane",
    )
    emit(LandmarkName.ANTERIOR_TIBIAL_CORTEX, ap_anterior)
    emit(LandmarkName.POSTERIOR_TIBIAL_CORTEX, ap_posterior)

    # Tibial tuberosity: most anterior point 15-45 mm distal to the plateau. It is
    # the rotational reference for the tibial component.
    plateau_top = float((tibia_vertices @ tibial_superior).max())
    tibial_proj = tibia_vertices @ tibial_superior
    tuberosity_region = tibia_vertices[
        (tibial_proj <= plateau_top - 15.0) & (tibial_proj >= plateau_top - 45.0)
    ]
    if tuberosity_region.shape[0] >= 20:
        tuberosity = _robust_extreme(
            tuberosity_region, anterior_dir, percentile=98.0, label="tibial tuberosity"
        )
        emit(LandmarkName.TIBIAL_TUBEROSITY, tuberosity)

    provenance = Provenance.for_component(
        "landmarks.geometric",
        parameters={
            "side": resolved_side.value,
            "femoral_shaft_straightness": round(femoral_straightness, 4),
            "tibial_shaft_straightness": round(tibial_straightness, 4),
            "plateau_plane_rms_mm": round(plateau_rms, 4),
            "mediolateral_source": "posterior_condylar_axis_refined",
            "external_landmarks": sorted(k.value for k in (external or {})),
            "condyle_region_height_mm": round(condyle_height, 2),
            "config": {
                "condyle_height_to_width_ratio": config.condyle_height_to_width_ratio,
                "tibial_plateau_fraction": config.tibial_plateau_fraction,
                "extreme_percentile": config.extreme_percentile,
            },
        },
    )

    landmark_set = LandmarkSet.from_iterable(
        found, side=resolved_side, provenance=provenance
    )
    log.info(
        "landmarks detected",
        fields={
            "count": len(landmark_set),
            "mean_confidence": round(landmark_set.mean_confidence, 3),
            "low_confidence": [lm.name.value for lm in landmark_set.low_confidence()],
        },
    )
    return landmark_set


def detect_landmarks_and_frame(
    meshes: MeshSet,
    *,
    side: Side | None = None,
    external: Mapping[LandmarkName, ArrayLike] | None = None,
    config: DetectionConfig = DEFAULT_CONFIG,
) -> tuple[LandmarkSet, PatientFrame]:
    """Detect landmarks and build the patient frame in one step."""
    landmarks = detect_landmarks(meshes, side=side, external=external, config=config)
    frame = build_patient_frame(landmarks)
    return landmarks, frame


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _safe_shaft_axis(
    mesh: AnatomicalMesh,
    config: DetectionConfig,
    *,
    from_proximal: bool = True,
    superior_hint: ArrayLike = IMAGE_SUPERIOR,
) -> tuple[Axis, float]:
    """Fit a shaft axis, falling back to the image superior axis if the fit is poor."""
    try:
        axis, straightness = detect_shaft_axis(
            mesh,
            superior_hint=superior_hint,
            fraction=config.shaft_fraction,
            from_proximal=from_proximal,
        )
    except LandmarkError as exc:
        log.warning(
            "shaft axis fit failed; using the image superior axis instead",
            fields={"structure": mesh.structure.value, "reason": str(exc)},
        )
        return (
            Axis(
                mesh.centroid,
                unit(superior_hint),
                frame=mesh.frame,
                label=f"{mesh.structure.value}_axis_fallback",
            ),
            0.0,
        )

    # A shaft axis wildly divergent from the anatomical superior direction means
    # the PCA locked onto the metaphyseal flare rather than the diaphysis.
    from kneeai.core.geometry import angle_between_deg

    divergence = angle_between_deg(axis.direction, superior_hint)
    if divergence > 30.0:
        log.warning(
            "fitted shaft axis diverges strongly from the anatomical superior "
            "direction; using the anatomical direction instead",
            fields={
                "structure": mesh.structure.value,
                "divergence_deg": round(divergence, 2),
            },
        )
        return (
            Axis(
                mesh.centroid,
                unit(superior_hint),
                frame=mesh.frame,
                label=f"{mesh.structure.value}_axis_fallback",
            ),
            0.0,
        )
    return axis, straightness


def _anterior_direction(
    superior: NDArray[np.float64], medial: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Anterior direction implied by a superior and a medial direction."""
    candidate = np.cross(medial, superior)
    candidate = candidate - float(np.dot(candidate, superior)) * superior
    candidate = unit(candidate)
    return candidate if float(np.dot(candidate, IMAGE_ANTERIOR)) > 0 else -candidate


def _split_compartments(
    points: NDArray[np.float64], medial_dir: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a point set into medial and lateral halves about its own midline.

    The midline is the midpoint of the mediolateral extent rather than the
    centroid: the centroid is pulled by whichever compartment carries more
    surface area, which in a worn knee is systematically the less-affected side.
    """
    proj = points @ medial_dir
    midline = (float(proj.min()) + float(proj.max())) / 2.0
    medial = points[proj > midline]
    lateral = points[proj <= midline]
    if medial.shape[0] < 10 or lateral.shape[0] < 10:
        raise LandmarkError(
            "failed to separate the medial and lateral compartments "
            f"({medial.shape[0]} medial / {lateral.shape[0]} lateral vertices). "
            "The mediolateral direction is probably wrong for this side."
        )
    return medial, lateral


def _plateau_central_mask(
    points: NDArray[np.float64],
    medial_dir: NDArray[np.float64],
    anterior_dir: NDArray[np.float64],
    fraction: float,
) -> NDArray[np.bool_]:
    """Select the central portion of the plateau, as a normalised elliptical radius.

    The plateau is treated as an ellipse in the mediolateral/anteroposterior
    plane, sized from its own extents. Points with a normalised radius above
    ``fraction`` are the peripheral rim and are excluded.
    """
    ml = points @ medial_dir
    ap = points @ anterior_dir
    ml_centre = (float(ml.min()) + float(ml.max())) / 2.0
    ap_centre = (float(ap.min()) + float(ap.max())) / 2.0
    ml_semi = max((float(ml.max()) - float(ml.min())) / 2.0, 1e-6)
    ap_semi = max((float(ap.max()) - float(ap.min())) / 2.0, 1e-6)
    radius = np.sqrt(((ml - ml_centre) / ml_semi) ** 2 + ((ap - ap_centre) / ap_semi) ** 2)
    return radius <= fraction


def _midline_strip(
    points: NDArray[np.float64],
    medial_dir: NDArray[np.float64],
    medial_point: NDArray[np.float64],
    lateral_point: NDArray[np.float64],
    config: DetectionConfig,
) -> NDArray[np.float64]:
    """Points within a narrow strip about the mediolateral midline."""
    centre = (float(np.dot(medial_point, medial_dir)) + float(np.dot(lateral_point, medial_dir))) / 2.0
    proj = points @ medial_dir
    return points[np.abs(proj - centre) <= config.midline_half_width_mm]


def _upward_facing(
    mesh: AnatomicalMesh,
    region_mask: NDArray[np.bool_],
    superior: NDArray[np.float64],
    config: DetectionConfig,
) -> NDArray[np.float64]:
    """Vertices in a region whose surface normal points proximally."""
    normals = np.asarray(mesh.mesh.vertex_normals, dtype=np.float64)
    if normals.shape[0] != mesh.n_vertices:  # pragma: no cover - defensive
        return mesh.vertices[region_mask]
    facing = (normals @ superior) >= config.plateau_normal_threshold
    return mesh.vertices[region_mask & facing]
