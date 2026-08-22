"""3D reconstruction: voxel masks -> validated anatomical surfaces (PRD §15, §70).

    >>> from kneeai.recon import reconstruct_all
    >>> meshes, failures = reconstruct_all(mask)
    >>> meshes.femur.volume_mm3
    """

from __future__ import annotations

from kneeai.recon.mesh import AnatomicalMesh, MeshQuality, MeshSet
from kneeai.recon.surface import (
    BONE_SURFACE_CONFIG,
    THIN_SURFACE_CONFIG,
    SurfaceConfig,
    config_for,
    extract_surface,
    reconstruct_all,
    surface_from_binary,
)

__all__ = [
    "AnatomicalMesh",
    "MeshQuality",
    "MeshSet",
    "SurfaceConfig",
    "BONE_SURFACE_CONFIG",
    "THIN_SURFACE_CONFIG",
    "config_for",
    "extract_surface",
    "reconstruct_all",
    "surface_from_binary",
]
