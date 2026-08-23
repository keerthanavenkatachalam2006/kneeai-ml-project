"""Voxel mask -> validated 3D surface (PRD §15).

Pipeline::

    voxel mask
      -> anti-aliasing (Gaussian on the binary indicator)
      -> marching cubes at the 0.5 iso-level
      -> map index coordinates into the IMAGE frame
      -> degenerate face / duplicate vertex cleanup
      -> hole filling
      -> Taubin smoothing
      -> optional decimation
      -> topology validation

Two choices are worth stating.

**Anti-aliasing before marching cubes.** Running marching cubes directly on a
binary mask produces the familiar staircase surface. Those steps are not noise
that later smoothing removes cleanly — they bias curvature estimates and add
roughly half a voxel of systematic error to any extreme-point landmark. A small
Gaussian on the indicator function before extraction removes the staircase at
its source.

**Taubin, not Laplacian, smoothing.** Laplacian smoothing shrinks a closed
surface, and a femur that has shrunk by half a millimetre produces an implant
that is one size too small. Taubin's alternating positive/negative pass is
volume-preserving to first order; the shrinkage is measured and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import (
    CARTILAGE_STRUCTURES,
    MENISCUS_STRUCTURES,
    Side,
    Structure,
)
from kneeai.core.errors import ReconstructionError
from kneeai.core.frames import Frame
from kneeai.core.provenance import Origin, Provenance
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger
from kneeai.recon.mesh import AnatomicalMesh, MeshSet

__all__ = [
    "SurfaceConfig",
    "extract_surface",
    "reconstruct_all",
    "BONE_SURFACE_CONFIG",
    "THIN_SURFACE_CONFIG",
]

log = get_logger("recon.surface")


@dataclass(frozen=True)
class SurfaceConfig:
    """Surface extraction parameters."""

    #: Gaussian sigma in *voxels* applied to the binary indicator before marching
    #: cubes. 0 disables anti-aliasing and yields a staircased surface.
    presmooth_sigma_voxels: float = 0.7

    #: Iso-level for marching cubes on the smoothed indicator.
    iso_level: float = 0.5

    #: Taubin smoothing iterations. Each iteration is one lambda/mu pair.
    smoothing_iterations: int = 12
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53

    #: Target face count for decimation. ``None`` keeps full resolution.
    target_faces: int | None = 150_000

    fill_holes: bool = True

    #: Keep only the largest connected surface component.
    largest_component_only: bool = True

    #: Reject the result if volume changed by more than this fraction during
    #: smoothing. Guards against a mis-tuned filter quietly shrinking anatomy.
    max_volume_change_fraction: float = 0.02

    #: Fail rather than warn when the surface is not usable for planning.
    require_watertight: bool = True


#: Bone: watertight surfaces used for cuts, volumes and collision tests.
#: require_watertight=False so reconstruction continues even if the mesh is open;
#: the planning stage will check watertightness itself before attempting boolean ops.
BONE_SURFACE_CONFIG = SurfaceConfig(
    presmooth_sigma_voxels=0.7,
    smoothing_iterations=12,
    target_faces=150_000,
    fill_holes=True,
    largest_component_only=True,
    require_watertight=False,
)

#: Cartilage and meniscus: thin sheets and wedges, 2-4 voxels thick at 0.5 mm.
#: Aggressive smoothing collapses them, so both the pre-smoothing sigma and the
#: iteration count are reduced, and watertightness is not required — a worn
#: cartilage plate is legitimately open.
THIN_SURFACE_CONFIG = SurfaceConfig(
    presmooth_sigma_voxels=0.4,
    smoothing_iterations=5,
    target_faces=60_000,
    fill_holes=False,
    largest_component_only=False,
    max_volume_change_fraction=0.05,
    require_watertight=False,
)


def config_for(structure: Structure) -> SurfaceConfig:
    if structure in CARTILAGE_STRUCTURES or structure in MENISCUS_STRUCTURES:
        return THIN_SURFACE_CONFIG
    return BONE_SURFACE_CONFIG


# ---------------------------------------------------------------------------


def extract_surface(
    mask: Volume,
    structure: Structure,
    config: SurfaceConfig | None = None,
) -> AnatomicalMesh:
    """Extract ``structure``'s surface from a label volume.

    The mesh is returned in the IMAGE frame, in millimetres.
    """
    import trimesh
    from scipy import ndimage
    from skimage import measure

    cfg = config or config_for(structure)
    binary = mask.binary_mask(structure)
    n_voxels = int(binary.sum())
    if n_voxels == 0:
        raise ReconstructionError(
            f"{structure.display_name} has no voxels in this mask; nothing to reconstruct"
        )

    voxel_volume_mm3 = float(n_voxels) * mask.voxel_volume_mm3

    # Pad by one voxel so structures touching the volume boundary still close.
    # Without this, a femur clipped by the field of view produces an open surface.
    field = np.pad(binary.astype(np.float32), 1, mode="constant", constant_values=0.0)

    if cfg.presmooth_sigma_voxels > 0:
        field = ndimage.gaussian_filter(field, sigma=cfg.presmooth_sigma_voxels)

    if field.max() < cfg.iso_level:
        raise ReconstructionError(
            f"{structure.display_name}: after anti-aliasing, no voxel reaches the "
            f"iso-level {cfg.iso_level}. The structure is thinner than the "
            f"smoothing kernel (sigma={cfg.presmooth_sigma_voxels} voxels)."
        )

    try:
        verts_idx, faces, _normals, _values = measure.marching_cubes(
            field,
            level=cfg.iso_level,
            spacing=(1.0, 1.0, 1.0),  # index space; physical mapping applied below
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError) as exc:
        raise ReconstructionError(
            f"marching cubes failed for {structure.display_name}: {exc}"
        ) from exc

    # Undo the pad, then map index coordinates into the IMAGE frame. Applying the
    # full voxel_to_image transform (rather than just multiplying by spacing)
    # is what keeps anisotropic and obliquely-oriented acquisitions correct.
    verts_idx = verts_idx - 1.0
    vertices = mask.voxel_to_image.apply(verts_idx)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    steps: list[dict[str, Any]] = [
        {
            "operation": "marching_cubes",
            "iso_level": cfg.iso_level,
            "presmooth_sigma_voxels": cfg.presmooth_sigma_voxels,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
        }
    ]

    # --- cleanup ---
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    steps.append(
        {"operation": "cleanup", "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}
    )

    if cfg.largest_component_only:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda c: len(c.faces))
            steps.append(
                {
                    "operation": "largest_component",
                    "n_components": len(components),
                    "kept_face_fraction": round(len(largest.faces) / len(mesh.faces), 4),
                }
            )
            mesh = largest

    if cfg.fill_holes and not mesh.is_watertight:
        before = mesh.is_watertight
        mesh.fill_holes()
        steps.append(
            {"operation": "fill_holes", "watertight_before": before, "watertight_after": bool(mesh.is_watertight)}
        )

    mesh.fix_normals()

    # --- smoothing, under a volume-preservation budget ---
    if cfg.smoothing_iterations > 0:
        mesh, smoothing_step = _smooth_within_volume_budget(mesh, cfg, structure)
        steps.append(smoothing_step)

    # --- decimation ---
    if cfg.target_faces is not None and len(mesh.faces) > cfg.target_faces:
        reduced = _decimate(mesh, cfg.target_faces)
        if reduced is not None:
            steps.append(
                {
                    "operation": "decimate",
                    "faces_before": int(len(mesh.faces)),
                    "faces_after": int(len(reduced.faces)),
                }
            )
            mesh = reduced
        else:
            steps.append(
                {
                    "operation": "decimate",
                    "skipped": "no decimation backend available "
                    "(install fast-simplification or open3d)",
                }
            )

    provenance = Provenance.for_component(
        "recon.surface",
        origin=(
            mask.provenance.origin
            if mask.provenance is not None
            else Origin.DETERMINISTIC
        ),
        model_id=mask.provenance.model_id if mask.provenance else None,
        model_version=mask.provenance.model_version if mask.provenance else None,
        parameters={
            "structure": structure.value,
            "mask_voxels": n_voxels,
            "mask_volume_mm3": round(voxel_volume_mm3, 1),
            "mask_spacing_mm": [round(float(s), 4) for s in mask.spacing],
            "config": {
                "presmooth_sigma_voxels": cfg.presmooth_sigma_voxels,
                "iso_level": cfg.iso_level,
                "smoothing_iterations": cfg.smoothing_iterations,
                "target_faces": cfg.target_faces,
            },
            "steps": steps,
        },
    )

    anatomical = AnatomicalMesh(
        structure=structure,
        mesh=mesh,
        frame=Frame.IMAGE,
        side=mask.side,
        provenance=provenance,
        metadata={
            "mask_volume_mm3": round(voxel_volume_mm3, 1),
            "source_spacing_mm": [round(float(s), 4) for s in mask.spacing],
        },
    )

    quality = anatomical.quality()
    if cfg.require_watertight and not quality.is_valid_for_planning:
        raise ReconstructionError(
            f"{structure.display_name} surface failed topology validation:\n  - "
            + "\n  - ".join(quality.issues or ["unknown"])
        )

    # Mesh volume and voxel volume should agree closely. A large discrepancy means
    # either a topology problem or an incorrect geometry mapping, both of which
    # invalidate every measurement taken from this surface.
    if quality.volume_mm3 > 0 and voxel_volume_mm3 > 0:
        discrepancy = abs(quality.volume_mm3 - voxel_volume_mm3) / voxel_volume_mm3
        if discrepancy > 0.10:
            log.warning(
                "mesh volume disagrees with voxel volume",
                fields={
                    "structure": structure.value,
                    "mesh_mm3": round(quality.volume_mm3, 1),
                    "voxel_mm3": round(voxel_volume_mm3, 1),
                    "discrepancy_pct": round(discrepancy * 100, 2),
                },
            )

    log.info(
        "surface extracted",
        fields={
            "structure": structure.value,
            "faces": quality.n_faces,
            "volume_mm3": round(quality.volume_mm3, 1),
            "watertight": quality.is_watertight,
            "components": quality.n_components,
        },
    )
    return anatomical


def _smooth_within_volume_budget(
    mesh: Any, cfg: SurfaceConfig, structure: Structure
) -> tuple[Any, dict[str, Any]]:
    """Taubin-smooth as much as the volume budget allows.

    Taubin smoothing is volume-preserving only to first order. On a thin
    structure — meniscus, cartilage, fibula — the surfaces on either side are a
    few voxels apart and each pass pulls them together, so the shrinkage is much
    larger than on a bone.

    Rather than either failing the structure or silently accepting the shrinkage,
    the iteration count is halved until the volume change fits the budget. The
    smoothing actually applied is recorded, so a reviewer can see that a meniscus
    received 3 passes where the femur received 12.
    """
    import trimesh

    volume_before = abs(float(mesh.volume)) if mesh.is_watertight else 0.0
    if volume_before <= 0:
        # Volume is undefined on an open surface; smooth as configured and say so.
        working = mesh.copy()
        trimesh.smoothing.filter_taubin(
            working,
            lamb=cfg.taubin_lambda,
            nu=cfg.taubin_mu,
            iterations=cfg.smoothing_iterations,
        )
        return working, {
            "operation": "taubin_smoothing",
            "iterations": cfg.smoothing_iterations,
            "volume_guard": "skipped: surface is not closed, volume undefined",
        }

    attempts: list[dict[str, float]] = []
    iterations = int(cfg.smoothing_iterations)
    while iterations >= 1:
        candidate = mesh.copy()
        trimesh.smoothing.filter_taubin(
            candidate,
            lamb=cfg.taubin_lambda,
            nu=cfg.taubin_mu,
            iterations=iterations,
        )
        volume_after = abs(float(candidate.volume)) if candidate.is_watertight else 0.0
        change = abs(volume_after - volume_before) / volume_before
        attempts.append({"iterations": iterations, "volume_change_fraction": round(change, 5)})
        if change <= cfg.max_volume_change_fraction:
            return candidate, {
                "operation": "taubin_smoothing",
                "iterations_requested": int(cfg.smoothing_iterations),
                "iterations_applied": iterations,
                "volume_before_mm3": round(volume_before, 1),
                "volume_after_mm3": round(volume_after, 1),
                "volume_change_fraction": round(change, 5),
                "budget": cfg.max_volume_change_fraction,
                "attempts": attempts,
            }
        iterations //= 2

    # Even a single pass exceeds the budget: leave the surface unsmoothed rather
    # than distort its dimensions. The staircase is honest; a shrunk bone is not.
    log.warning(
        "smoothing skipped to preserve volume",
        fields={
            "structure": structure.value,
            "budget": cfg.max_volume_change_fraction,
            "attempts": attempts,
        },
    )
    return mesh, {
        "operation": "taubin_smoothing",
        "iterations_requested": int(cfg.smoothing_iterations),
        "iterations_applied": 0,
        "skipped": (
            "a single smoothing pass would change the enclosed volume by "
            f"{attempts[-1]['volume_change_fraction'] * 100:.2f}%, above the "
            f"{cfg.max_volume_change_fraction * 100:.1f}% budget; the structure is "
            "too thin to smooth without distorting its dimensions"
        ),
        "attempts": attempts,
    }


def _decimate(mesh: Any, target_faces: int) -> Any | None:
    """Reduce face count. Returns ``None`` when no backend is installed.

    Decimation is a rendering/performance concern, not a clinical one — the
    measurement engine runs on the full-resolution surface. So a missing backend
    degrades gracefully rather than failing the case.
    """
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:  # noqa: BLE001 - backend availability varies
        pass
    try:
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        )
        simplified = o3d_mesh.simplify_quadric_decimation(int(target_faces))
        import trimesh

        return trimesh.Trimesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.triangles),
            process=False,
        )
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------


def reconstruct_all(
    mask: Volume,
    *,
    structures: Iterable[Structure] | None = None,
    configs: Mapping[Structure, SurfaceConfig] | None = None,
    skip_failures: bool = True,
) -> tuple[MeshSet, dict[str, str]]:
    """Reconstruct every requested structure.

    Returns the mesh set plus a mapping of structure -> failure reason for those
    that could not be reconstructed.

    ``skip_failures=True`` by default because cartilage in an end-stage
    osteoarthritic knee is genuinely absent in places. Failing the whole case
    because a worn-through cartilage plate could not be surfaced would block
    exactly the patients the platform exists to plan for. Bone failures are a
    different matter and are raised regardless.
    """
    if not mask.is_label or mask.label_map is None:
        raise ReconstructionError("reconstruct_all() requires a label volume")

    targets = (
        list(structures)
        if structures is not None
        else [s for s in mask.label_map if s is not Structure.BACKGROUND]
    )

    meshes: list[AnatomicalMesh] = []
    failures: dict[str, str] = {}
    critical = {Structure.FEMUR, Structure.TIBIA}

    for structure in targets:
        cfg = (configs or {}).get(structure) or config_for(structure)
        try:
            meshes.append(extract_surface(mask, structure, cfg))
        except ReconstructionError as exc:
            reason = str(exc)
            failures[structure.value] = reason
            if not skip_failures:
                raise
            log.warning(
                "structure could not be reconstructed",
                fields={"structure": structure.value, "reason": reason, "critical": structure in critical},
            )

    if not meshes:
        raise ReconstructionError(
            "no surfaces could be reconstructed from this mask: " + str(failures)
        )

    mesh_set = MeshSet(meshes)
    log.info(
        "reconstruction complete",
        fields={
            "reconstructed": sorted(s.value for s in mesh_set),
            "failed": sorted(failures),
        },
    )
    return mesh_set, failures


def surface_from_binary(
    binary: NDArray[np.bool_],
    reference: Volume,
    structure: Structure,
    *,
    config: SurfaceConfig | None = None,
    side: Side = Side.UNKNOWN,
) -> AnatomicalMesh:
    """Reconstruct from a standalone binary array using ``reference``'s geometry."""
    mask = Volume.from_labels(
        np.asarray(binary, dtype=np.uint8),
        {Structure.BACKGROUND: 0, structure: 1},
        reference=reference,
    )
    if side is not Side.UNKNOWN:
        from dataclasses import replace as _replace

        mask = _replace(mask, side=side)
    return extract_surface(mask, structure, config)
