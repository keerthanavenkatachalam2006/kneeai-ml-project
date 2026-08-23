"""
3D Reconstruction Module.
Converts segmentation masks into 3D meshes using marching cubes.
Exports to GLB/OBJ for web visualization.
"""
import io
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from .config import LABELS_MAP, LABEL_COLORS, MESH_OUTPUT_DIR


class MeshReconstructor:
    """Converts voxel segmentation masks to 3D surface meshes."""

    def __init__(self, smoothing_sigma: float = 1.0, step_size: int = 1):
        """
        Args:
            smoothing_sigma: Gaussian smoothing applied before marching cubes.
            step_size: Step size for marching cubes (higher = faster, coarser).
        """
        self.smoothing_sigma = smoothing_sigma
        self.step_size = step_size

    def extract_mesh(
        self,
        label_volume: np.ndarray,
        label_id: int,
        voxel_size: tuple,
        smooth: bool = True,
    ) -> Optional[trimesh.Trimesh]:
        """
        Extract a 3D mesh for a specific label from the segmentation volume.

        Args:
            label_volume: 3D numpy array with integer labels.
            label_id: Which label to extract (1-5 for OAIZIB-CM).
            voxel_size: Physical voxel dimensions in mm (z, y, x).
            smooth: Whether to apply Gaussian smoothing before meshing.

        Returns:
            trimesh.Trimesh object or None if label not present.
        """
        # Create binary mask for this label
        binary = (label_volume == label_id).astype(np.float32)

        if binary.sum() == 0:
            return None

        # Smooth the binary mask for smoother surfaces
        if smooth:
            binary = gaussian_filter(binary, sigma=self.smoothing_sigma)

        # Marching cubes to extract isosurface
        try:
            verts, faces, normals, _ = marching_cubes(
                binary,
                level=0.5,
                spacing=voxel_size,
                step_size=self.step_size,
            )
        except Exception:
            return None

        # Create trimesh object
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
        )

        # Apply color
        color = LABEL_COLORS.get(label_id, (0.5, 0.5, 0.5, 1.0))
        color_rgba = np.array([int(c * 255) for c in color], dtype=np.uint8)
        mesh.visual.face_colors = np.tile(color_rgba, (len(faces), 1))

        return mesh

    def reconstruct_all_structures(
        self,
        label_volume: np.ndarray,
        voxel_size: tuple,
        labels: Optional[list[int]] = None,
    ) -> dict[int, trimesh.Trimesh]:
        """
        Reconstruct meshes for all (or specified) structures.

        Returns dict mapping label_id -> trimesh.Trimesh.
        """
        if labels is None:
            labels = [1, 2, 3, 4, 5, 6]

        meshes = {}
        for label_id in labels:
            mesh = self.extract_mesh(label_volume, label_id, voxel_size)
            if mesh is not None:
                meshes[label_id] = mesh

        return meshes

    def create_combined_scene(
        self,
        meshes: dict[int, trimesh.Trimesh],
    ) -> trimesh.Scene:
        """Create a trimesh Scene with all structures as named nodes."""
        scene = trimesh.Scene()
        for label_id, mesh in meshes.items():
            name = LABELS_MAP.get(label_id, f"Label_{label_id}")
            scene.add_geometry(mesh, node_name=name)
        return scene

    def export_glb(
        self,
        meshes: dict[int, trimesh.Trimesh],
        output_path: Optional[Path] = None,
    ) -> bytes:
        """
        Export all meshes as a single GLB (binary glTF) file.
        This is the format used by Three.js in the frontend.

        Returns the GLB bytes (and optionally saves to file).
        """
        scene = self.create_combined_scene(meshes)
        glb_data = scene.export(file_type="glb")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(glb_data)

        return glb_data

    def export_obj(
        self,
        mesh: trimesh.Trimesh,
        output_path: Path,
    ) -> None:
        """Export a single mesh as OBJ."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path), file_type="obj")

    def export_case(
        self,
        case_id: int,
        label_volume: np.ndarray,
        voxel_size: tuple,
    ) -> dict:
        """
        Full reconstruction pipeline for a case.
        Extracts all structures, saves GLB, returns metadata.
        """
        meshes = self.reconstruct_all_structures(label_volume, voxel_size)

        output_path = MESH_OUTPUT_DIR / f"case_{case_id:03d}.glb"
        glb_data = self.export_glb(meshes, output_path)

        # Compute mesh statistics
        stats = {}
        for label_id, mesh in meshes.items():
            stats[LABELS_MAP[label_id]] = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
                "surface_area_mm2": float(mesh.area),
                "bounds_mm": mesh.bounds.tolist(),
                "is_watertight": mesh.is_watertight,
            }

        return {
            "case_id": case_id,
            "glb_path": str(output_path),
            "glb_size_bytes": len(glb_data),
            "structures": stats,
            "num_structures": len(meshes),
        }

    def get_mesh_for_viewer(
        self,
        label_volume: np.ndarray,
        voxel_size: tuple,
        labels: Optional[list[int]] = None,
        decimate_target: Optional[int] = None,
    ) -> bytes:
        """
        Generate GLB optimized for web viewing.
        Optionally decimates meshes to reduce polygon count.
        """
        meshes = self.reconstruct_all_structures(label_volume, voxel_size, labels)

        if decimate_target:
            for label_id in meshes:
                mesh = meshes[label_id]
                if len(mesh.faces) > decimate_target:
                    ratio = decimate_target / len(mesh.faces)
                    meshes[label_id] = mesh.simplify_quadric_decimation(
                        int(len(mesh.faces) * ratio)
                    )

        return self.export_glb(meshes)
