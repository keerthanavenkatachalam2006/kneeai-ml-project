"""Cartilage and meniscus measurements (PRD §17, §18).

PRD §18 insists on separating three questions, and this module keeps them apart:

- **Segmentation** — where is it? Answered upstream, in :mod:`kneeai.ml`.
- **Quantification** — how large, how thick, how far extruded? Answered here.
- **Pathology classification** — is there an abnormality? *Not* answered here. This
  module reports geometry. It does not grade cartilage loss or diagnose a
  meniscal tear.

Thickness estimation
--------------------
Two independent estimators are computed, because they fail differently:

1. **Volume over articular area.** For a thin sheet, total surface area is
   approximately twice the articular area, so ``2 x volume / area`` estimates mean
   thickness. Robust and unbiased in the interior; slightly overestimates for a
   small plate whose rim area is not negligible.
2. **Distance-transform local maxima.** Inside the mask, the Euclidean distance to
   the nearest boundary reaches half the local thickness on the sheet's
   mid-surface. Local maxima of that field therefore sample the thickness
   distribution directly, which gives a maximum and percentiles that the
   volume/area estimator cannot.

Reporting both, and reporting them as separate numbers, is deliberate: when they
disagree the cartilage is not sheet-like, and that disagreement is information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import Side, Structure
from kneeai.core.errors import MeasurementError
from kneeai.core.units import Unit
from kneeai.imaging.volume import Volume
from kneeai.landmarks.frame import PatientFrame
from kneeai.logging import get_logger
from kneeai.measure.bone import plateau_region
from kneeai.measure.types import Measurement, MeasurementName, make_measurement
from kneeai.recon.mesh import AnatomicalMesh, MeshSet

__all__ = [
    "ThicknessStats",
    "sheet_thickness",
    "measure_cartilage",
    "measure_meniscus",
    "PROJECTION_RESOLUTION_MM",
]

log = get_logger("measure.soft_tissue")

#: Raster resolution for axial projections (coverage, footprint area).
PROJECTION_RESOLUTION_MM = 0.5


@dataclass(frozen=True)
class ThicknessStats:
    """Thickness distribution of a thin structure, in millimetres."""

    #: 2 x volume / surface area. The primary mean-thickness estimate.
    mean_from_volume_area: float

    #: Statistics of the distance-transform local-maxima samples.
    mean_from_distance: float
    median_from_distance: float
    p95_from_distance: float
    max_from_distance: float
    n_samples: int

    volume_mm3: float
    surface_area_mm2: float

    @property
    def estimators_agree(self) -> bool:
        """Whether the two estimators agree within 25%.

        Disagreement means the structure is not sheet-like — it may be fragmented,
        or the mask may include a blob that is not cartilage.
        """
        if self.mean_from_volume_area <= 0:
            return False
        ratio = self.mean_from_distance / self.mean_from_volume_area
        return 0.75 <= ratio <= 1.25

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_mm": round(self.mean_from_volume_area, 3),
            "mean_from_distance_mm": round(self.mean_from_distance, 3),
            "median_from_distance_mm": round(self.median_from_distance, 3),
            "p95_from_distance_mm": round(self.p95_from_distance, 3),
            "max_from_distance_mm": round(self.max_from_distance, 3),
            "n_thickness_samples": self.n_samples,
            "volume_mm3": round(self.volume_mm3, 1),
            "surface_area_mm2": round(self.surface_area_mm2, 1),
            "estimators_agree": self.estimators_agree,
        }


def sheet_thickness(
    mask: Volume,
    structure: Structure,
    mesh: AnatomicalMesh | None = None,
    *,
    rim_exclusion_mm: float = 0.75,
) -> ThicknessStats:
    """Estimate the thickness distribution of a thin structure.

    Args:
        rim_exclusion_mm: distance-transform samples below this are discarded.
            Local maxima also occur along the sheet's *rim ridge*, where the
            distance to the boundary is small by construction; including them
            drags the distribution down and makes every cartilage plate look
            thinner than it is.
    """
    from scipy import ndimage

    binary = mask.binary_mask(structure)
    n_voxels = int(binary.sum())
    if n_voxels == 0:
        raise MeasurementError(
            f"{structure.display_name} has no voxels; thickness is undefined"
        )
    volume_mm3 = float(n_voxels) * mask.voxel_volume_mm3

    # --- estimator 1: volume / articular area ---
    if mesh is not None:
        area = mesh.area_mm2
    else:
        # Surface area from the mask boundary, using the marching-cubes surface.
        from skimage import measure as skmeasure

        padded = np.pad(binary.astype(np.float32), 1)
        verts, faces, _n, _v = skmeasure.marching_cubes(
            padded, level=0.5, spacing=tuple(float(s) for s in mask.spacing)
        )
        area = float(skmeasure.mesh_surface_area(verts, faces))

    mean_from_volume_area = 2.0 * volume_mm3 / area if area > 0 else float("nan")

    # --- estimator 2: distance-transform local maxima ---
    edt = np.asarray(
        ndimage.distance_transform_edt(binary, sampling=tuple(float(s) for s in mask.spacing)),
        dtype=np.float64,
    )
    local_max = edt >= ndimage.maximum_filter(edt, size=3) - 1e-9
    candidates = binary & local_max & (edt > rim_exclusion_mm)
    samples = 2.0 * edt[candidates]

    if samples.size == 0:
        log.warning(
            "no interior thickness samples survived rim exclusion; the structure is "
            "thinner than the exclusion distance",
            fields={
                "structure": structure.value,
                "rim_exclusion_mm": rim_exclusion_mm,
                "voxels": n_voxels,
            },
        )
        samples = 2.0 * edt[binary & local_max]
    if samples.size == 0:  # pragma: no cover - defensive
        samples = np.asarray([float("nan")])

    return ThicknessStats(
        mean_from_volume_area=float(mean_from_volume_area),
        mean_from_distance=float(np.nanmean(samples)),
        median_from_distance=float(np.nanmedian(samples)),
        p95_from_distance=float(np.nanpercentile(samples, 95)),
        max_from_distance=float(np.nanmax(samples)),
        n_samples=int(samples.size),
        volume_mm3=volume_mm3,
        surface_area_mm2=float(area),
    )


# ---------------------------------------------------------------------------
# Cartilage
# ---------------------------------------------------------------------------

_CARTILAGE_MEASUREMENTS: dict[Structure, tuple[MeasurementName, MeasurementName]] = {
    Structure.FEMORAL_CARTILAGE: (
        MeasurementName.FEMORAL_CARTILAGE_VOLUME,
        MeasurementName.FEMORAL_CARTILAGE_MEAN_THICKNESS,
    ),
    Structure.MEDIAL_TIBIAL_CARTILAGE: (
        MeasurementName.MEDIAL_TIBIAL_CARTILAGE_VOLUME,
        MeasurementName.MEDIAL_TIBIAL_CARTILAGE_MEAN_THICKNESS,
    ),
    Structure.LATERAL_TIBIAL_CARTILAGE: (
        MeasurementName.LATERAL_TIBIAL_CARTILAGE_VOLUME,
        MeasurementName.LATERAL_TIBIAL_CARTILAGE_MEAN_THICKNESS,
    ),
    Structure.PATELLAR_CARTILAGE: (
        MeasurementName.PATELLAR_CARTILAGE_VOLUME,
        MeasurementName.PATELLAR_CARTILAGE_MEAN_THICKNESS,
    ),
}


def measure_cartilage(
    mask: Volume,
    meshes: MeshSet,
) -> tuple[list[Measurement], dict[str, str], dict[str, Any]]:
    """Cartilage volumes and thicknesses per compartment.

    Returns ``(measurements, unavailable, detail)`` where ``detail`` holds the full
    thickness distributions for the report and the dashboard.
    """
    out: list[Measurement] = []
    unavailable: dict[str, str] = {}
    detail: dict[str, Any] = {}

    present = set(mask.structures_present())

    for structure, (volume_name, thickness_name) in _CARTILAGE_MEASUREMENTS.items():
        if structure not in present:
            reason = (
                f"{structure.display_name} was not segmented in this study. In an "
                "end-stage osteoarthritic knee this may reflect genuine "
                "full-thickness loss rather than a segmentation failure — check the "
                "images before concluding either."
            )
            unavailable[volume_name.value] = reason
            unavailable[thickness_name.value] = reason
            continue

        try:
            stats = sheet_thickness(mask, structure, meshes.get(structure))
        except MeasurementError as exc:
            unavailable[thickness_name.value] = str(exc)
            continue

        detail[structure.value] = stats.to_dict()

        out.append(
            make_measurement(
                volume_name,
                stats.volume_mm3,
                Unit.MM3,
                method=(
                    f"voxel count of the {structure.display_name} mask multiplied by "
                    f"the voxel volume ({mask.voxel_volume_mm3:.4f} mm³)"
                ),
                structures=(structure,),
            )
        )

        caveat = ""
        if not stats.estimators_agree:
            caveat = (
                f"the two thickness estimators disagree "
                f"({stats.mean_from_volume_area:.2f} mm from volume/area vs "
                f"{stats.mean_from_distance:.2f} mm from the distance transform); "
                "the structure is not sheet-like, so treat the mean thickness as "
                "unreliable and inspect the segmentation"
            )
        out.append(
            make_measurement(
                thickness_name,
                stats.mean_from_volume_area,
                Unit.MM,
                method=(
                    "mean thickness estimated as twice the cartilage volume divided "
                    "by its surface area (thin-sheet approximation); "
                    f"distance-transform median {stats.median_from_distance:.2f} mm, "
                    f"maximum {stats.max_from_distance:.2f} mm"
                ),
                structures=(structure,),
                caveat=caveat,
            )
        )

    # --- medial vs lateral asymmetry ---
    medial = detail.get(Structure.MEDIAL_TIBIAL_CARTILAGE.value)
    lateral = detail.get(Structure.LATERAL_TIBIAL_CARTILAGE.value)
    if medial and lateral:
        medial_mm = float(medial["mean_mm"])
        lateral_mm = float(lateral["mean_mm"])
        total = medial_mm + lateral_mm
        if total > 1e-6:
            asymmetry = 100.0 * (lateral_mm - medial_mm) / total
            out.append(
                make_measurement(
                    MeasurementName.CARTILAGE_THICKNESS_ASYMMETRY,
                    asymmetry,
                    Unit.PERCENT,
                    method=(
                        "normalised difference between lateral and medial tibial "
                        "cartilage mean thickness, as a percentage of their sum"
                    ),
                    structures=(
                        Structure.MEDIAL_TIBIAL_CARTILAGE,
                        Structure.LATERAL_TIBIAL_CARTILAGE,
                    ),
                    sign_convention=(
                        "positive means the lateral cartilage is thicker than the "
                        "medial, the pattern in medial compartment disease"
                    ),
                )
            )

    return out, unavailable, detail


# ---------------------------------------------------------------------------
# Meniscus
# ---------------------------------------------------------------------------

_MENISCUS_MEASUREMENTS: dict[
    Structure, tuple[MeasurementName, MeasurementName, MeasurementName, MeasurementName]
] = {
    Structure.MEDIAL_MENISCUS: (
        MeasurementName.MEDIAL_MENISCUS_VOLUME,
        MeasurementName.MEDIAL_MENISCUS_EXTRUSION,
        MeasurementName.MEDIAL_MENISCUS_HEIGHT,
        MeasurementName.MEDIAL_MENISCUS_COVERAGE,
    ),
    Structure.LATERAL_MENISCUS: (
        MeasurementName.LATERAL_MENISCUS_VOLUME,
        MeasurementName.LATERAL_MENISCUS_EXTRUSION,
        MeasurementName.LATERAL_MENISCUS_HEIGHT,
        MeasurementName.LATERAL_MENISCUS_COVERAGE,
    ),
}


def measure_meniscus(
    mask: Volume,
    meshes: MeshSet,
    frame: PatientFrame,
) -> tuple[list[Measurement], dict[str, str], dict[str, Any]]:
    """Meniscus volume, extrusion, height and tibial coverage."""
    out: list[Measurement] = []
    unavailable: dict[str, str] = {}
    detail: dict[str, Any] = {}

    present = set(mask.structures_present())
    if Structure.TIBIA not in meshes:
        for structure in _MENISCUS_MEASUREMENTS:
            for name in _MENISCUS_MEASUREMENTS[structure]:
                unavailable[name.value] = (
                    "the tibia surface is required as the reference for meniscal "
                    "extrusion and coverage"
                )
        return out, unavailable, detail

    tibia = meshes.require(Structure.TIBIA)
    plateau = plateau_region(tibia, frame)

    for structure, (
        volume_name,
        extrusion_name,
        height_name,
        coverage_name,
    ) in _MENISCUS_MEASUREMENTS.items():
        if structure not in present:
            reason = f"{structure.display_name} was not segmented in this study"
            for name in (volume_name, extrusion_name, height_name, coverage_name):
                unavailable[name.value] = reason
            continue

        is_medial = structure is Structure.MEDIAL_MENISCUS
        outward = frame.medial if is_medial else frame.lateral

        # --- volume, from the mask ---
        volume_mm3 = mask.structure_volume_mm3(structure)
        out.append(
            make_measurement(
                volume_name,
                volume_mm3,
                Unit.MM3,
                method=(
                    f"voxel count of the {structure.display_name} mask multiplied by "
                    f"the voxel volume ({mask.voxel_volume_mm3:.4f} mm³)"
                ),
                structures=(structure,),
            )
        )

        mesh = meshes.get(structure)
        if mesh is None:
            for name in (extrusion_name, height_name, coverage_name):
                unavailable[name.value] = (
                    f"the {structure.display_name} surface could not be "
                    "reconstructed, so geometric measurements are unavailable"
                )
            continue

        vertices = mesh.vertices

        # --- extrusion ---
        # How far the meniscus projects beyond the tibial plateau rim, measured
        # outward along the mediolateral axis. Measured against the plateau rim on
        # the same side, not against the midline, so a wide tibia does not read as
        # extrusion.
        compartment_plateau = plateau[
            (plateau @ outward) > float(np.median(plateau @ outward))
        ]
        if compartment_plateau.shape[0] < 10:
            unavailable[extrusion_name.value] = (
                "too few plateau vertices on this side to establish the rim"
            )
        else:
            # Robust extremes on both, so a single stray vertex on either surface
            # cannot manufacture or hide extrusion.
            rim = float(np.percentile(compartment_plateau @ outward, 99.5))
            meniscus_edge = float(np.percentile(vertices @ outward, 99.5))
            extrusion = max(0.0, meniscus_edge - rim)
            out.append(
                make_measurement(
                    extrusion_name,
                    extrusion,
                    Unit.MM,
                    method=(
                        "distance by which the meniscus extends beyond the tibial "
                        "plateau rim along the mediolateral axis (99.5th percentile "
                        "of both surfaces, clamped at zero)"
                    ),
                    structures=(structure, Structure.TIBIA),
                    sign_convention=(
                        "zero means the meniscus is contained within the plateau "
                        "rim; values above ~3 mm are conventionally described as "
                        "major extrusion"
                    ),
                )
            )

        # --- height ---
        # Local height per axial column, not the global superoinferior extent.
        # The plateau slopes posteriorly by several degrees, so the global extent
        # of a meniscus lying on it includes that drop across its whole
        # anteroposterior span — inflating a 5 mm wedge to ~12 mm. Clinically the
        # height is read off a coronal slice as the height of the meniscal
        # cross-section, which is a local quantity.
        height = _local_height(vertices, frame)
        out.append(
            make_measurement(
                height_name,
                height,
                Unit.MM,
                method=(
                    "90th percentile of the per-column superoinferior extent of the "
                    f"meniscus surface, rasterised at {PROJECTION_RESOLUTION_MM} mm "
                    "in axial projection (a local wedge height, not the global "
                    "superoinferior extent)"
                ),
                structures=(structure,),
            )
        )

        # --- coverage ---
        try:
            meniscus_area = _axial_footprint_area(vertices, frame)
            plateau_area = _axial_footprint_area(compartment_plateau, frame)
            if plateau_area <= 0:
                raise MeasurementError("compartment footprint area is zero")
            coverage = 100.0 * meniscus_area / plateau_area
            detail[structure.value] = {
                "meniscus_footprint_mm2": round(meniscus_area, 1),
                "compartment_footprint_mm2": round(plateau_area, 1),
                "volume_mm3": round(volume_mm3, 1),
                "height_mm": round(height, 2),
            }
            out.append(
                make_measurement(
                    coverage_name,
                    coverage,
                    Unit.PERCENT,
                    method=(
                        "meniscal footprint area divided by the compartment's "
                        "plateau footprint area, both rasterised in axial "
                        f"projection at {PROJECTION_RESOLUTION_MM} mm"
                    ),
                    structures=(structure, Structure.TIBIA),
                    caveat=(
                        "coverage above 100% indicates the meniscal footprint "
                        "extends beyond the bony compartment, which is consistent "
                        "with extrusion"
                        if coverage > 100.0
                        else ""
                    ),
                )
            )
        except MeasurementError as exc:
            unavailable[coverage_name.value] = str(exc)

    return out, unavailable, detail


def _local_height(
    points: NDArray[np.float64],
    frame: PatientFrame,
    *,
    resolution_mm: float = PROJECTION_RESOLUTION_MM,
    percentile: float = 90.0,
    min_points_per_column: int = 4,
) -> float:
    """Per-column superoinferior extent, as a percentile across columns.

    Columns with too few surface points are dropped: a column clipped by the
    structure's edge samples only one wall and reports a spuriously small height.
    """
    if points.shape[0] < 20:
        raise MeasurementError("too few points to estimate a local height")

    u = points @ frame.right
    v = points @ frame.anterior
    z = points @ frame.superior

    iu = ((u - float(u.min())) / resolution_mm).astype(int)
    iv = ((v - float(v.min())) / resolution_mm).astype(int)
    nv = int(iv.max()) + 1
    keys = iu * nv + iv

    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    z_sorted = z[order]
    boundaries = np.flatnonzero(np.diff(keys_sorted)) + 1
    groups = np.split(z_sorted, boundaries)

    heights = [
        float(g.max() - g.min()) for g in groups if g.size >= min_points_per_column
    ]
    if not heights:
        raise MeasurementError(
            "no axial column contained enough surface points to measure a height"
        )
    return float(np.percentile(np.asarray(heights), percentile))


def _axial_footprint_area(
    points: NDArray[np.float64],
    frame: PatientFrame,
    *,
    resolution_mm: float = PROJECTION_RESOLUTION_MM,
) -> float:
    """Area of a point set's projection onto the axial plane, in mm².

    Rasterised rather than computed as a polygon union. A meniscus is a C-shape,
    so its convex hull overestimates badly, and a triangle-by-triangle polygon
    union over ~10⁴ faces is slow. Marking the cells hit by surface vertices and
    then closing single-cell gaps is fast, and at 0.5 mm the vertex density of a
    marching-cubes surface is several points per cell.
    """
    from scipy import ndimage

    if points.shape[0] < 10:
        raise MeasurementError("too few points to rasterise a footprint")

    u = points @ frame.right
    v = points @ frame.anterior
    u_min, v_min = float(u.min()), float(v.min())
    nu = int(np.ceil((float(u.max()) - u_min) / resolution_mm)) + 2
    nv = int(np.ceil((float(v.max()) - v_min) / resolution_mm)) + 2
    if nu < 2 or nv < 2:
        raise MeasurementError("footprint has no extent in the axial plane")

    grid = np.zeros((nu, nv), dtype=bool)
    iu = np.clip(((u - u_min) / resolution_mm).astype(int), 0, nu - 1)
    iv = np.clip(((v - v_min) / resolution_mm).astype(int), 0, nv - 1)
    grid[iu, iv] = True

    # Close single-cell gaps left by vertex sampling, then fill enclosed holes.
    # The C-shaped opening of a meniscus is wider than the closing kernel, so it
    # survives — which is the point.
    grid = ndimage.binary_closing(grid, structure=np.ones((3, 3), bool))
    grid = ndimage.binary_fill_holes(grid)
    return float(grid.sum()) * resolution_mm * resolution_mm
