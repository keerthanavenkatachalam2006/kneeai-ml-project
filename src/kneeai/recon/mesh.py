"""Frame-aware anatomical meshes.

A bare ``trimesh.Trimesh`` has vertices but no notion of which coordinate frame
they are in, which structure they represent, or where they came from. Every
downstream consumer — landmarking, measurement, implant fitting, simulation —
needs all three, so :class:`AnatomicalMesh` carries them and refuses operations
that would mix frames (PRD §70, §71).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import Side, Structure
from kneeai.core.errors import FrameMismatchError, ReconstructionError, ValidationError
from kneeai.core.frames import Frame, FramedPoints, Transform
from kneeai.core.geometry import Plane, extreme_point
from kneeai.core.provenance import ArtifactRef, Provenance
from kneeai.logging import get_logger

__all__ = ["AnatomicalMesh", "MeshQuality", "MeshSet"]

log = get_logger("recon.mesh")

_EXPORT_FORMATS = {".stl", ".ply", ".obj", ".glb", ".gltf", ".off"}


@dataclass(frozen=True)
class MeshQuality:
    """Topology and quality assessment of a reconstructed surface (PRD §15)."""

    n_vertices: int
    n_faces: int
    n_components: int
    is_watertight: bool
    is_winding_consistent: bool
    volume_mm3: float
    area_mm2: float
    euler_number: int
    n_degenerate_faces: int
    n_duplicate_vertices: int
    min_edge_mm: float
    max_edge_mm: float
    #: Ratio of surface area to the area of a sphere of equal volume. 1.0 is a
    #: sphere; a staircased marching-cubes surface scores noticeably higher.
    sphericity_ratio: float
    issues: tuple[str, ...] = ()

    #: Whether a single connected component was expected for this structure.
    single_component_expected: bool = True

    @property
    def is_valid_for_planning(self) -> bool:
        """Whether this surface can be used for volumes, cuts and collision tests.

        Requires a watertight surface, because mesh volume is undefined on an open
        surface and boolean cut operations produce nonsense on one. A single
        component is required only where anatomy implies one.
        """
        if not self.is_watertight or self.volume_mm3 <= 0:
            return False
        return self.n_components == 1 or not self.single_component_expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "n_components": self.n_components,
            "is_watertight": self.is_watertight,
            "is_winding_consistent": self.is_winding_consistent,
            "volume_mm3": round(self.volume_mm3, 2),
            "area_mm2": round(self.area_mm2, 2),
            "euler_number": self.euler_number,
            "n_degenerate_faces": self.n_degenerate_faces,
            "n_duplicate_vertices": self.n_duplicate_vertices,
            "min_edge_mm": round(self.min_edge_mm, 4),
            "max_edge_mm": round(self.max_edge_mm, 4),
            "sphericity_ratio": round(self.sphericity_ratio, 3),
            "single_component_expected": self.single_component_expected,
            "valid_for_planning": self.is_valid_for_planning,
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class AnatomicalMesh:
    """A triangulated anatomical surface with its frame and provenance."""

    structure: Structure
    mesh: Any  # trimesh.Trimesh
    frame: Frame = Frame.IMAGE
    side: Side = Side.UNKNOWN
    provenance: Provenance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not hasattr(self.mesh, "vertices") or not hasattr(self.mesh, "faces"):
            raise ValidationError(
                "AnatomicalMesh requires a trimesh.Trimesh-like object with "
                "vertices and faces"
            )
        if len(self.mesh.vertices) == 0:
            raise ReconstructionError(
                f"{self.structure.value} mesh has no vertices; the mask was probably empty"
            )

    # --- basic properties -------------------------------------------------
    @property
    def vertices(self) -> NDArray[np.float64]:
        return np.asarray(self.mesh.vertices, dtype=np.float64)

    @property
    def faces(self) -> NDArray[np.int64]:
        return np.asarray(self.mesh.faces, dtype=np.int64)

    @property
    def n_vertices(self) -> int:
        return int(len(self.mesh.vertices))

    @property
    def n_faces(self) -> int:
        return int(len(self.mesh.faces))

    @property
    def framed_vertices(self) -> FramedPoints:
        return FramedPoints(self.vertices, frame=self.frame)

    @property
    def centroid(self) -> NDArray[np.float64]:
        return np.asarray(self.mesh.centroid, dtype=np.float64)

    @property
    def center_mass(self) -> NDArray[np.float64]:
        """Volumetric centre of mass. Only meaningful on a watertight mesh."""
        if not self.mesh.is_watertight:
            return self.centroid
        return np.asarray(self.mesh.center_mass, dtype=np.float64)

    @property
    def bounds(self) -> NDArray[np.float64]:
        """``[[min_x, min_y, min_z], [max_x, max_y, max_z]]`` in millimetres."""
        return np.asarray(self.mesh.bounds, dtype=np.float64)

    @property
    def extents_mm(self) -> NDArray[np.float64]:
        return np.asarray(self.mesh.extents, dtype=np.float64)

    @property
    def volume_mm3(self) -> float:
        """Enclosed volume. Zero for a non-watertight surface, not a guess."""
        if not self.mesh.is_watertight:
            return 0.0
        return abs(float(self.mesh.volume))

    @property
    def area_mm2(self) -> float:
        return float(self.mesh.area)

    @property
    def is_watertight(self) -> bool:
        return bool(self.mesh.is_watertight)

    # --- frame handling ---------------------------------------------------
    def require_frame(self, frame: Frame, context: str = "") -> AnatomicalMesh:
        if self.frame is not frame:
            where = f" ({context})" if context else ""
            raise FrameMismatchError(
                f"{self.structure.value} mesh is in '{self.frame.value}' but "
                f"'{frame.value}' is required{where}"
            )
        return self

    def transform(self, t: Transform) -> AnatomicalMesh:
        """Apply a frame transform, producing a mesh tagged with the target frame."""
        if t.source is not self.frame:
            raise FrameMismatchError(
                f"{self.structure.value} mesh is in '{self.frame.value}' but the "
                f"transform maps from '{t.source.value}'"
            )
        moved = self.mesh.copy()
        moved.apply_transform(t.matrix)
        return replace(self, mesh=moved, frame=t.target)

    # --- geometry queries -------------------------------------------------
    def extreme_point(
        self, direction: ArrayLike, *, percentile: float | None = 99.5
    ) -> NDArray[np.float64]:
        """Most extreme surface point along ``direction``.

        Defaults to a robust percentile rather than the single furthest vertex:
        a marching-cubes surface derived from a mask with one stray voxel has a
        spike, and clinical landmarks must not be defined by it.
        """
        return extreme_point(self.vertices, direction, percentile=percentile)

    def extent_along(self, direction: ArrayLike) -> float:
        from kneeai.core.geometry import bounding_extent

        return bounding_extent(self.vertices, direction)

    def section(self, plane: Plane) -> Any | None:
        """Cross-section as a ``trimesh.Path3D``, or ``None`` if no intersection."""
        plane_frame = plane.frame
        if plane_frame is not self.frame:
            raise FrameMismatchError(
                f"plane is in '{plane_frame.value}' but the mesh is in "
                f"'{self.frame.value}'"
            )
        return self.mesh.section(plane_origin=plane.point, plane_normal=plane.normal)

    def section_polygons(self, plane: Plane) -> list[NDArray[np.float64]]:
        """In-plane 2D polygons of the cross-section, in millimetres.

        This is the operation implant footprint matching depends on: the resected
        bone surface becomes a 2D polygon, which is compared against the implant's
        footprint polygon (PRD §22).
        """
        section = self.section(plane)
        if section is None:
            return []
        try:
            planar, _to_3d = section.to_planar(
                to_2D=self._plane_to_2d_matrix(plane), check=False
            )
        except Exception as exc:  # noqa: BLE001 - trimesh raises various types here
            log.warning(
                "cross-section could not be planarised",
                fields={"structure": self.structure.value, "error": str(exc)},
            )
            return []
        polygons: list[NDArray[np.float64]] = []
        for polygon in planar.polygons_full:
            polygons.append(np.asarray(polygon.exterior.coords, dtype=np.float64)[:, :2])
        return polygons

    def _plane_to_2d_matrix(self, plane: Plane) -> NDArray[np.float64]:
        """4x4 matrix mapping the plane into z=0, with an anatomical in-plane basis."""
        hint = self.metadata.get("ap_axis")
        u, v, n = plane.basis(hint)
        matrix = np.eye(4)
        matrix[:3, :3] = np.vstack([u, v, n])
        matrix[:3, 3] = -matrix[:3, :3] @ plane.point
        return matrix

    def slice_plane(
        self, plane: Plane, *, cap: bool = True, keep_positive: bool = True
    ) -> AnatomicalMesh:
        """Cut the mesh with a plane, keeping one side.

        With the resection-plane convention (normal pointing into retained bone),
        ``keep_positive=True`` returns the bone that remains after the cut.
        ``cap=True`` closes the cut face so the result stays watertight and its
        volume remains defined.
        """
        if plane.frame is not self.frame:
            raise FrameMismatchError(
                f"plane frame '{plane.frame.value}' != mesh frame '{self.frame.value}'"
            )
        normal = plane.normal if keep_positive else -plane.normal
        cut = self.mesh.slice_plane(
            plane_origin=plane.point, plane_normal=normal, cap=cap
        )
        if cut is None or len(cut.vertices) == 0:
            raise ReconstructionError(
                f"cutting {self.structure.value} with plane {plane.label!r} removed "
                "the entire mesh; the plane does not intersect the bone"
            )
        return replace(self, mesh=cut)

    def contains(self, points: ArrayLike) -> NDArray[np.bool_]:
        """Point-in-mesh test. Requires a watertight surface."""
        if not self.is_watertight:
            raise ReconstructionError(
                f"{self.structure.value} mesh is not watertight; inside/outside is "
                "undefined"
            )
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return np.asarray(self.mesh.contains(p), dtype=bool)

    def distance_to(self, points: ArrayLike) -> NDArray[np.float64]:
        """Unsigned distance from each point to the surface, in millimetres."""
        import trimesh

        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        proximity = trimesh.proximity.ProximityQuery(self.mesh)
        return np.abs(np.asarray(proximity.signed_distance(p), dtype=np.float64))

    def signed_distance_to(self, points: ArrayLike) -> NDArray[np.float64]:
        """Signed distance; positive inside the mesh (trimesh convention)."""
        import trimesh

        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        proximity = trimesh.proximity.ProximityQuery(self.mesh)
        return np.asarray(proximity.signed_distance(p), dtype=np.float64)

    def intersects(self, other: AnatomicalMesh) -> bool:
        """Whether two meshes overlap. Used for implant/bone collision (PRD §28)."""
        import trimesh

        if other.frame is not self.frame:
            raise FrameMismatchError(
                f"cannot test intersection between meshes in '{self.frame.value}' "
                f"and '{other.frame.value}'"
            )
        manager = trimesh.collision.CollisionManager()
        manager.add_object("a", self.mesh)
        manager.add_object("b", other.mesh)
        return bool(manager.in_collision_internal())

    # --- quality ----------------------------------------------------------
    @property
    def expects_single_component(self) -> bool:
        """Whether this structure should form one connected surface.

        Bones and menisci should. Cartilage should not: the intercondylar notch
        separates the medial and lateral femoral cartilage into two genuinely
        disconnected plates, and full-thickness wear splits a plate further. A
        component count of 2 on femoral cartilage is anatomy, not a defect.
        """
        from kneeai.core.anatomy import CARTILAGE_STRUCTURES

        return self.structure not in CARTILAGE_STRUCTURES

    def quality(self, *, expect_single_component: bool | None = None) -> MeshQuality:
        import trimesh

        mesh = self.mesh
        issues: list[str] = []
        single_expected = (
            self.expects_single_component
            if expect_single_component is None
            else expect_single_component
        )

        try:
            n_components = len(mesh.split(only_watertight=False))
        except Exception:  # noqa: BLE001
            n_components = 1

        edges = mesh.edges_unique_length
        min_edge = float(edges.min()) if len(edges) else 0.0
        max_edge = float(edges.max()) if len(edges) else 0.0

        degenerate = int(np.count_nonzero(~mesh.nondegenerate_faces()))
        duplicates = int(len(mesh.vertices) - len(trimesh.grouping.unique_rows(mesh.vertices)[0]))

        volume = abs(float(mesh.volume)) if mesh.is_watertight else 0.0
        area = float(mesh.area)
        if volume > 0 and area > 0:
            equivalent_radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
            sphere_area = 4.0 * np.pi * equivalent_radius**2
            sphericity = area / sphere_area
        else:
            sphericity = float("nan")

        if not mesh.is_watertight:
            issues.append(
                "surface is not watertight: volume is undefined and boolean cuts "
                "will fail"
            )
        if n_components > 1 and single_expected:
            issues.append(
                f"{n_components} disconnected components; only the largest should "
                "represent this structure"
            )
        if degenerate:
            issues.append(f"{degenerate} degenerate (zero-area) faces")
        if not mesh.is_winding_consistent:
            issues.append("inconsistent face winding; normals point both ways")
        if volume <= 0 and mesh.is_watertight:
            issues.append("watertight surface with non-positive volume (inverted normals)")

        return MeshQuality(
            n_vertices=self.n_vertices,
            n_faces=self.n_faces,
            n_components=n_components,
            is_watertight=bool(mesh.is_watertight),
            is_winding_consistent=bool(mesh.is_winding_consistent),
            volume_mm3=volume,
            area_mm2=area,
            euler_number=int(mesh.euler_number),
            n_degenerate_faces=degenerate,
            n_duplicate_vertices=duplicates,
            min_edge_mm=min_edge,
            max_edge_mm=max_edge,
            sphericity_ratio=sphericity,
            issues=tuple(issues),
            single_component_expected=single_expected,
        )

    def require_valid_for_planning(self) -> AnatomicalMesh:
        q = self.quality()
        if not q.is_valid_for_planning:
            raise ReconstructionError(
                f"{self.structure.display_name} surface is not usable for planning:\n  - "
                + "\n  - ".join(q.issues or ["unknown topology problem"])
            )
        return self

    # --- IO ---------------------------------------------------------------
    def export(self, path: str | Path, *, write_sidecar: bool = True) -> ArtifactRef:
        """Export the mesh, plus a sidecar recording its frame and provenance.

        The sidecar exists because STL/PLY/OBJ carry no frame information. A mesh
        file without it is geometry with no stated coordinate system, which is
        precisely what PRD §70 requires meshes to preserve.
        """
        import json

        target = Path(path)
        if target.suffix.lower() not in _EXPORT_FORMATS:
            raise ValidationError(
                f"unsupported mesh format {target.suffix!r}; expected one of "
                f"{sorted(_EXPORT_FORMATS)}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        self.mesh.export(str(target))

        if write_sidecar:
            sidecar = target.with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "structure": self.structure.value,
                        "frame": self.frame.value,
                        "side": self.side.value,
                        "units": "mm",
                        "quality": self.quality().to_dict(),
                        "metadata": {
                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in self.metadata.items()
                        },
                        "provenance": (
                            self.provenance.to_dict() if self.provenance else None
                        ),
                    },
                    indent=2,
                    default=str,
                )
            )
        return ArtifactRef.from_path(
            "mesh", target, note=f"{self.structure.value} in {self.frame.value}"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        structure: Structure | None = None,
        frame: Frame | None = None,
        side: Side | None = None,
    ) -> AnatomicalMesh:
        """Load a mesh, using its sidecar for frame/structure when available."""
        import json

        import trimesh

        source = Path(path)
        if not source.exists():
            raise ValidationError(f"mesh file not found: {source}")

        meta: dict[str, Any] = {}
        sidecar = source.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
            except json.JSONDecodeError:
                log.warning("mesh sidecar is not valid JSON", fields={"path": str(sidecar)})

        resolved_structure = structure or (
            Structure(meta["structure"]) if meta.get("structure") else None
        )
        if resolved_structure is None:
            raise ValidationError(
                f"cannot determine which structure {source.name} represents; pass "
                "structure= explicitly or provide a sidecar"
            )
        resolved_frame = frame or (Frame(meta["frame"]) if meta.get("frame") else None)
        if resolved_frame is None:
            raise ValidationError(
                f"cannot determine the coordinate frame of {source.name}; pass "
                "frame= explicitly. Loading geometry with an assumed frame is how "
                "coordinate errors enter a plan (PRD §71)."
            )

        loaded = trimesh.load(str(source), process=False, force="mesh")
        return cls(
            structure=resolved_structure,
            mesh=loaded,
            frame=resolved_frame,
            side=side or Side(meta.get("side", "unknown")),
            metadata=meta.get("metadata", {}),
            provenance=(
                Provenance.from_dict(meta["provenance"]) if meta.get("provenance") else None
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "structure": self.structure.value,
            "frame": self.frame.value,
            "side": self.side.value,
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "volume_mm3": round(self.volume_mm3, 1),
            "area_mm2": round(self.area_mm2, 1),
            "extents_mm": [round(float(e), 1) for e in self.extents_mm],
            "watertight": self.is_watertight,
        }

    def with_metadata(self, **extra: Any) -> AnatomicalMesh:
        merged = dict(self.metadata)
        merged.update(extra)
        return replace(self, metadata=merged)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AnatomicalMesh({self.structure.value}, {self.n_faces} faces, "
            f"frame={self.frame.value}, watertight={self.is_watertight})"
        )


class MeshSet(dict):
    """A case's reconstructed surfaces, keyed by structure.

    A dict subclass with frame checking and convenience accessors, so callers can
    write ``meshes.femur`` and get a clear error if the femur was never
    reconstructed.
    """

    def __init__(self, meshes: Iterable[AnatomicalMesh] = ()) -> None:
        super().__init__({m.structure: m for m in meshes})

    # --- named accessors ---
    def require(self, structure: Structure) -> AnatomicalMesh:
        mesh = self.get(structure)
        if mesh is None:
            available = sorted(s.value for s in self)
            raise ReconstructionError(
                f"{structure.display_name} surface is not available for this case. "
                f"Reconstructed: {available or ['<none>']}"
            )
        return mesh

    @property
    def femur(self) -> AnatomicalMesh:
        return self.require(Structure.FEMUR)

    @property
    def tibia(self) -> AnatomicalMesh:
        return self.require(Structure.TIBIA)

    @property
    def patella(self) -> AnatomicalMesh:
        return self.require(Structure.PATELLA)

    @property
    def frame(self) -> Frame:
        frames = {m.frame for m in self.values()}
        if len(frames) > 1:
            raise FrameMismatchError(
                f"mesh set contains mixed frames: {sorted(f.value for f in frames)}"
            )
        if not frames:
            raise ReconstructionError("mesh set is empty")
        return frames.pop()

    @property
    def side(self) -> Side:
        sides = {m.side for m in self.values() if m.side is not Side.UNKNOWN}
        if len(sides) > 1:
            raise ValidationError(
                f"mesh set contains mixed sides: {sorted(s.value for s in sides)}"
            )
        return sides.pop() if sides else Side.UNKNOWN

    def transform(self, t: Transform) -> MeshSet:
        return MeshSet(m.transform(t) for m in self.values())

    def summary(self) -> dict[str, Any]:
        return {s.value: m.summary() for s, m in sorted(self.items(), key=lambda kv: kv[0].value)}

    def quality_report(self) -> dict[str, Any]:
        return {
            s.value: m.quality().to_dict()
            for s, m in sorted(self.items(), key=lambda kv: kv[0].value)
        }

    def export_all(self, directory: str | Path, *, extension: str = ".stl") -> dict[Structure, ArtifactRef]:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            structure: mesh.export(out_dir / f"{structure.value}{extension}")
            for structure, mesh in self.items()
        }
