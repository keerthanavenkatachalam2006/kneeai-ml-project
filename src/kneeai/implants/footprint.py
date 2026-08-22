"""Bone resection footprints — the 2D geometry implant matching runs on (PRD §22).

An implant seats on a planar bone cut. Whether it fits is therefore a 2D question:
how does the component's outline compare with the outline of the resected bone
surface?

This module produces that bone outline. It sections the bone mesh with the
resection plane and expresses the resulting polygon in a basis aligned with the
patient frame — ``u`` along patient-right, ``v`` anterior — which is exactly the
basis the implant footprints are defined in. The two polygons are then directly
comparable without any further axis reasoning, which is the point: every
coordinate conversion happens here, once.

Provisional planes
------------------
Matching happens *before* planning in the build order (PRD §XXII steps 13-14 come
before 15), so the matcher needs a resection plane before the planner has produced
one. The provisional planes here are the conventional defaults — a tibial cut
referenced off the less-worn compartment, a distal femoral cut perpendicular to
the mechanical axis — and they are clearly labelled provisional so a plan never
silently inherits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import LandmarkName, Side, Structure
from kneeai.core.errors import ImplantError, PlanningError
from kneeai.core.frames import Frame
from kneeai.core.geometry import Plane, unit
from kneeai.landmarks.frame import PatientFrame
from kneeai.landmarks.types import LandmarkSet
from kneeai.logging import get_logger
from kneeai.measure.bone import fit_articular_plane
from kneeai.recon.mesh import AnatomicalMesh, MeshSet

__all__ = [
    "BoneFootprint",
    "resection_footprint",
    "provisional_tibial_plane",
    "provisional_distal_femoral_plane",
    "DEFAULT_TIBIAL_RESECTION_MM",
    "DEFAULT_DISTAL_FEMORAL_RESECTION_MM",
]

log = get_logger("implants.footprint")

#: Conventional tibial resection depth below the *less worn* compartment's
#: articular surface. Referencing the less-worn side is standard: referencing the
#: worn side would under-resect and leave the joint line too high.
DEFAULT_TIBIAL_RESECTION_MM = 2.0

#: Conventional distal femoral resection, measured from the distal-most condyle.
#: Matches the femoral component's distal thickness, so the joint line is restored.
DEFAULT_DISTAL_FEMORAL_RESECTION_MM = 9.0


@dataclass(frozen=True)
class BoneFootprint:
    """The outline of a bone's cross-section at a resection plane.

    Attributes:
        polygon: ``(N, 2)`` outline in millimetres, in the ``(u, v)`` basis.
        plane: the resection plane, in the source (IMAGE) frame.
        u_axis / v_axis: in-plane basis vectors in the source frame. ``u`` is
            patient-right, ``v`` is anterior.
        origin: 3D point corresponding to ``(0, 0)`` in the 2D polygon.
    """

    structure: Structure
    polygon: NDArray[np.float64]
    plane: Plane
    u_axis: NDArray[np.float64]
    v_axis: NDArray[np.float64]
    origin: NDArray[np.float64]
    side: Side
    source: str = ""
    provisional: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        poly = np.atleast_2d(np.asarray(self.polygon, dtype=np.float64))
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            raise ImplantError(
                f"{self.structure.value} footprint must be an (N>=3, 2) polygon, got "
                f"{poly.shape}"
            )
        object.__setattr__(self, "polygon", poly)

    # --- geometry ---
    def shapely(self) -> Any:
        from shapely.geometry import Polygon

        polygon = Polygon(self.polygon)
        if not polygon.is_valid:
            repaired = polygon.buffer(0)
            if repaired.is_empty:
                raise ImplantError(
                    f"{self.structure.value} resection footprint is not a valid "
                    "polygon and could not be repaired"
                )
            if repaired.geom_type == "MultiPolygon":
                repaired = max(repaired.geoms, key=lambda g: g.area)
            return repaired
        return polygon

    @property
    def area_mm2(self) -> float:
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def ml_extent_mm(self) -> float:
        return float(self.polygon[:, 0].max() - self.polygon[:, 0].min())

    @property
    def ap_extent_mm(self) -> float:
        return float(self.polygon[:, 1].max() - self.polygon[:, 1].min())

    @property
    def aspect_ratio(self) -> float:
        return self.ml_extent_mm / max(self.ap_extent_mm, 1e-6)

    @property
    def centroid_2d(self) -> NDArray[np.float64]:
        return np.asarray(self.shapely().centroid.coords[0], dtype=np.float64)

    def to_3d(self, uv: ArrayLike) -> NDArray[np.float64]:
        """Lift 2D in-plane coordinates back into the source frame."""
        q = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        return self.origin + q[:, 0:1] * self.u_axis + q[:, 1:2] * self.v_axis

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure.value,
            "side": self.side.value,
            "polygon": np.round(self.polygon, 3).tolist(),
            "plane": self.plane.to_dict(),
            "u_axis": self.u_axis.tolist(),
            "v_axis": self.v_axis.tolist(),
            "origin": self.origin.tolist(),
            "area_mm2": round(self.area_mm2, 1),
            "ml_extent_mm": round(self.ml_extent_mm, 2),
            "ap_extent_mm": round(self.ap_extent_mm, 2),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "source": self.source,
            "provisional": self.provisional,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------


def resection_footprint(
    mesh: AnatomicalMesh,
    plane: Plane,
    frame: PatientFrame,
    *,
    source: str = "",
    provisional: bool = False,
) -> BoneFootprint:
    """Section ``mesh`` with ``plane`` and return the resected outline.

    The 2D basis is fixed to (patient-right, anterior) so the result is directly
    comparable with an implant footprint. When the plane is tilted — posterior
    slope, varus/valgus — the basis vectors are those directions projected into the
    plane, which is correct: the component seats on the cut, not on a horizontal.
    """
    if plane.frame is not mesh.frame:
        raise ImplantError(
            f"resection plane is in '{plane.frame.value}' but the "
            f"{mesh.structure.value} mesh is in '{mesh.frame.value}'"
        )

    u, v, _n = plane.basis(frame.right)
    # Confirm the basis came out anatomically oriented. plane.basis puts u along
    # the projected hint and v = n x u; with n superior that makes v anterior, but
    # a strongly tilted plane can flip it, and a flipped v axis mirrors the
    # footprint anteroposteriorly.
    if float(np.dot(v, frame.anterior)) < 0:
        v = -v
        u = -u  # keep the basis right-handed with respect to the plane normal

    polygons = _section_polygons(mesh, plane, u, v)
    if not polygons:
        raise ImplantError(
            f"the {mesh.structure.value} does not intersect the resection plane "
            f"({plane.label or 'unlabelled'}); the plane is outside the bone"
        )
    if len(polygons) > 1:
        areas = [_polygon_area(p) for p in polygons]
        log.info(
            "resection plane produced multiple cross-sections; keeping the largest",
            fields={
                "structure": mesh.structure.value,
                "n_sections": len(polygons),
                "areas_mm2": [round(a, 1) for a in areas],
            },
        )
        polygons = [polygons[int(np.argmax(areas))]]

    return BoneFootprint(
        structure=mesh.structure,
        polygon=polygons[0],
        plane=plane,
        u_axis=u,
        v_axis=v,
        origin=plane.point,
        side=frame.side,
        source=source or f"section of {mesh.structure.value} at {plane.label}",
        provisional=provisional,
        metadata={"n_cross_sections": len(polygons)},
    )


def _section_polygons(
    mesh: AnatomicalMesh,
    plane: Plane,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Cross-section outlines in the explicit ``(u, v)`` basis."""
    section = mesh.mesh.section(plane_origin=plane.point, plane_normal=plane.normal)
    if section is None:
        return []

    matrix = np.eye(4)
    matrix[:3, :3] = np.vstack([u, v, plane.normal])
    matrix[:3, 3] = -matrix[:3, :3] @ plane.point

    try:
        planar, _to_3d = section.to_planar(to_2D=matrix, check=False)
    except Exception as exc:  # noqa: BLE001 - trimesh raises various types
        raise ImplantError(
            f"the {mesh.structure.value} cross-section could not be flattened: {exc}"
        ) from exc

    out: list[NDArray[np.float64]] = []
    for polygon in planar.polygons_full:
        coords = np.asarray(polygon.exterior.coords, dtype=np.float64)[:, :2]
        if coords.shape[0] >= 4:
            out.append(coords[:-1])  # drop the duplicated closing vertex
    return out


def _polygon_area(polygon: NDArray[np.float64]) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# Provisional resection planes
# ---------------------------------------------------------------------------


def provisional_tibial_plane(
    meshes: MeshSet,
    landmarks: LandmarkSet,
    frame: PatientFrame,
    *,
    resection_mm: float = DEFAULT_TIBIAL_RESECTION_MM,
    preserve_slope: bool = True,
) -> Plane:
    """A provisional tibial resection plane, for sizing before planning.

    Referenced from the **less worn** compartment: the deeper of the two
    compartment low points is the worn one, so the shallower (more proximal) one is
    used as the reference. Referencing the worn compartment instead would place the
    cut too distal and over-resect.

    ``preserve_slope=True`` keeps the patient's own posterior slope. Setting it
    ``False`` cuts perpendicular to the tibial mechanical axis in the sagittal
    plane, which is the alternative convention.
    """
    tibia = meshes.require(Structure.TIBIA)
    articular_plane, rms = fit_articular_plane(tibia, frame)

    have_both = landmarks.has(
        LandmarkName.MEDIAL_PLATEAU_LOW, LandmarkName.LATERAL_PLATEAU_LOW
    )
    if have_both:
        medial_low = landmarks.position(LandmarkName.MEDIAL_PLATEAU_LOW)
        lateral_low = landmarks.position(LandmarkName.LATERAL_PLATEAU_LOW)
        medial_height = float(np.dot(medial_low, frame.superior))
        lateral_height = float(np.dot(lateral_low, frame.superior))
        if medial_height >= lateral_height:
            reference_point, reference_name = medial_low, "medial"
        else:
            reference_point, reference_name = lateral_low, "lateral"
    else:
        reference_point = articular_plane.point
        reference_name = "fitted articular plane centroid"
        log.warning(
            "compartment low points unavailable; referencing the provisional tibial "
            "cut from the fitted plateau plane instead of the less worn compartment"
        )

    normal = articular_plane.normal
    if not preserve_slope:
        # Perpendicular to the tibial axis in the sagittal plane, keeping the
        # coronal component of the fitted plane.
        normal = unit(
            normal - float(np.dot(normal, frame.anterior)) * frame.anterior
        )

    # The plane passes through the reference point, offset distally by the
    # resection depth. Normal points proximally (into retained bone is *below*
    # the cut for a tibial resection, so the normal points at the discarded part;
    # flip it so it points into the bone that is kept).
    plane_point = reference_point - resection_mm * unit(normal)
    plane = Plane(
        plane_point,
        -unit(normal),
        frame=articular_plane.frame,
        label="provisional_tibial_resection",
    )

    log.info(
        "provisional tibial resection plane",
        fields={
            "reference": reference_name,
            "resection_mm": resection_mm,
            "preserve_slope": preserve_slope,
            "articular_plane_rms_mm": round(rms, 3),
        },
    )
    return plane


def provisional_distal_femoral_plane(
    meshes: MeshSet,
    landmarks: LandmarkSet,
    frame: PatientFrame,
    *,
    resection_mm: float = DEFAULT_DISTAL_FEMORAL_RESECTION_MM,
) -> Plane:
    """A provisional distal femoral resection plane, perpendicular to the axis.

    Referenced from the **less worn** (more distal) condyle, for the same reason as
    the tibial cut. Oriented perpendicular to the frame's longitudinal axis, which
    is the mechanical axis when the hip centre is available.
    """
    if not landmarks.has(
        LandmarkName.DISTAL_MEDIAL_CONDYLE, LandmarkName.DISTAL_LATERAL_CONDYLE
    ):
        raise PlanningError(
            "a provisional distal femoral cut needs both distal condylar landmarks"
        )
    medial = landmarks.position(LandmarkName.DISTAL_MEDIAL_CONDYLE)
    lateral = landmarks.position(LandmarkName.DISTAL_LATERAL_CONDYLE)
    medial_height = float(np.dot(medial, frame.superior))
    lateral_height = float(np.dot(lateral, frame.superior))
    reference, reference_name = (
        (medial, "medial") if medial_height <= lateral_height else (lateral, "lateral")
    )

    plane_point = reference + resection_mm * frame.superior
    plane = Plane(
        plane_point,
        frame.superior,
        frame=Frame.IMAGE,
        label="provisional_distal_femoral_resection",
    )
    log.info(
        "provisional distal femoral resection plane",
        fields={
            "reference_condyle": reference_name,
            "resection_mm": resection_mm,
            "axis": frame.method,
        },
    )
    return plane
