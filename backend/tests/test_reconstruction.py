"""
Quantitative tests for the 3D reconstruction module.
Validates mesh generation, physical scaling, and export integrity.
"""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

import numpy as np
import trimesh
import pytest

from app.reconstruction import MeshReconstructor


@pytest.fixture
def reconstructor():
    return MeshReconstructor(smoothing_sigma=0.5, step_size=1)


# ─── Mesh Generation Tests ──────────────────────────────────────────────────

class TestMeshExtraction:
    """Verify basic mesh extraction from label volumes."""

    def test_empty_label_returns_none(self, reconstructor):
        """If the label is not present, return None."""
        label = np.zeros((20, 20, 20), dtype=np.uint8)
        mesh = reconstructor.extract_mesh(label, label_id=1, voxel_size=(1.0, 1.0, 1.0))
        assert mesh is None

    def test_sphere_produces_mesh(self, reconstructor):
        """A filled sphere should produce a valid mesh."""
        label = np.zeros((40, 40, 40), dtype=np.uint8)
        center = np.array([20, 20, 20])
        radius = 10
        for z in range(40):
            for y in range(40):
                for x in range(40):
                    if np.linalg.norm(np.array([z, y, x]) - center) <= radius:
                        label[z, y, x] = 1

        mesh = reconstructor.extract_mesh(label, label_id=1, voxel_size=(1.0, 1.0, 1.0))

        assert mesh is not None
        assert len(mesh.vertices) > 100
        assert len(mesh.faces) > 100
        assert mesh.is_watertight  # Sphere should be closed

    def test_cube_mesh_bounds(self, reconstructor):
        """A 10-voxel cube with 1mm spacing should have bounds near 10mm."""
        label = np.zeros((20, 20, 20), dtype=np.uint8)
        label[5:15, 5:15, 5:15] = 1
        voxel_size = (1.0, 1.0, 1.0)

        # Use minimal smoothing so bounds are closer to exact
        recon = MeshReconstructor(smoothing_sigma=0.3, step_size=1)
        mesh = recon.extract_mesh(label, label_id=1, voxel_size=voxel_size, smooth=True)

        assert mesh is not None
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        size = bounds[1] - bounds[0]

        # Each dimension should be approximately 10mm (with some smoothing margin)
        for dim_size in size:
            assert 8.0 < dim_size < 12.0, f"Mesh dimension {dim_size}mm not close to 10mm"


class TestPhysicalScaling:
    """Verify that voxel spacing is correctly applied to mesh vertices."""

    def test_isotropic_spacing(self, reconstructor):
        """With 2mm isotropic spacing, a 5-voxel cube → ~10mm mesh."""
        label = np.zeros((15, 15, 15), dtype=np.uint8)
        label[5:10, 5:10, 5:10] = 1  # 5×5×5 cube
        voxel_size = (2.0, 2.0, 2.0)

        recon = MeshReconstructor(smoothing_sigma=0.3, step_size=1)
        mesh = recon.extract_mesh(label, label_id=1, voxel_size=voxel_size, smooth=True)

        assert mesh is not None
        size = mesh.bounds[1] - mesh.bounds[0]
        for dim_size in size:
            assert 8.0 < dim_size < 12.0, f"Dimension {dim_size}mm (expected ~10mm)"

    def test_anisotropic_spacing(self, reconstructor):
        """
        With spacing (0.7, 0.3646, 0.3646) and 10 voxels each:
        Expected: axis0 ≈ 7mm, axis1 ≈ 3.6mm, axis2 ≈ 3.6mm.
        """
        label = np.zeros((20, 20, 20), dtype=np.uint8)
        label[5:15, 5:15, 5:15] = 1  # 10×10×10
        voxel_size = (0.7, 0.3646, 0.3646)

        recon = MeshReconstructor(smoothing_sigma=0.3, step_size=1)
        mesh = recon.extract_mesh(label, label_id=1, voxel_size=voxel_size, smooth=True)

        assert mesh is not None
        size = mesh.bounds[1] - mesh.bounds[0]

        # axis 0 (first coord of vertices) should span ~7mm
        assert 5.5 < size[0] < 8.5, f"Axis 0 size {size[0]}mm (expected ~7mm)"
        # axes 1,2 should span ~3.6mm
        assert 2.5 < size[1] < 4.7, f"Axis 1 size {size[1]}mm (expected ~3.6mm)"
        assert 2.5 < size[2] < 4.7, f"Axis 2 size {size[2]}mm (expected ~3.6mm)"

    def test_mesh_volume_approximates_voxel_volume(self, reconstructor):
        """
        For a solid sphere, mesh volume should approximate voxel-counted volume.
        Tolerance: within 20% (smoothing and discretization cause differences).
        """
        label = np.zeros((50, 50, 50), dtype=np.uint8)
        center = np.array([25, 25, 25])
        radius = 15
        for z in range(50):
            for y in range(50):
                for x in range(50):
                    if np.linalg.norm(np.array([z, y, x]) - center) <= radius:
                        label[z, y, x] = 1

        voxel_size = (1.0, 1.0, 1.0)
        voxel_volume = label.sum() * np.prod(voxel_size)

        recon = MeshReconstructor(smoothing_sigma=0.5, step_size=1)
        mesh = recon.extract_mesh(label, label_id=1, voxel_size=voxel_size)

        assert mesh is not None
        if mesh.is_watertight:
            mesh_volume = abs(mesh.volume)
            ratio = mesh_volume / voxel_volume
            assert 0.7 < ratio < 1.3, (
                f"Mesh volume {mesh_volume:.0f} vs voxel volume {voxel_volume:.0f} "
                f"(ratio {ratio:.2f})"
            )


class TestGLBExport:
    """Verify GLB export produces valid files."""

    def test_export_produces_bytes(self, reconstructor):
        """GLB export should return non-empty bytes."""
        label = np.zeros((20, 20, 20), dtype=np.uint8)
        label[5:15, 5:15, 5:15] = 1
        label[5:15, 5:15, 12:18] = 2
        voxel_size = (1.0, 1.0, 1.0)

        meshes = reconstructor.reconstruct_all_structures(label, voxel_size)
        glb_data = reconstructor.export_glb(meshes)

        assert isinstance(glb_data, bytes)
        assert len(glb_data) > 100
        # GLB magic number: 'glTF' = 0x46546C67
        assert glb_data[:4] == b'glTF'

    def test_multiple_structures_in_scene(self, reconstructor):
        """Multiple labels should produce multiple meshes in the scene."""
        label = np.zeros((30, 30, 30), dtype=np.uint8)
        label[5:15, 5:15, 5:15] = 1   # Femur
        label[5:15, 5:15, 16:25] = 3  # Tibia
        label[14:16, 5:15, 5:15] = 2  # Femoral cartilage (thin layer)
        voxel_size = (1.0, 1.0, 1.0)

        meshes = reconstructor.reconstruct_all_structures(label, voxel_size)
        assert len(meshes) >= 2  # At least femur and tibia

    def test_export_case_saves_file(self, reconstructor, tmp_path):
        """export_case should save a .glb file to disk."""
        import app.config as config
        original_dir = config.MESH_OUTPUT_DIR
        config.MESH_OUTPUT_DIR = tmp_path

        label = np.zeros((20, 20, 20), dtype=np.uint8)
        label[5:15, 5:15, 5:15] = 1
        voxel_size = (1.0, 1.0, 1.0)

        result = reconstructor.export_case(999, label, voxel_size)

        config.MESH_OUTPUT_DIR = original_dir

        assert result["num_structures"] >= 1
        assert result["glb_size_bytes"] > 0


class TestRealDataReconstruction:
    """Integration tests with real OAIZIB-CM data."""

    @pytest.fixture
    def real_case(self):
        from app.data_loader import OAIZIBDataLoader
        loader = OAIZIBDataLoader()
        return loader.load_case(405)

    def test_all_structures_reconstructed(self, reconstructor, real_case):
        """All 5 structures should produce meshes."""
        meshes = reconstructor.reconstruct_all_structures(
            real_case["label"], real_case["voxel_size"]
        )
        assert len(meshes) == 5, f"Expected 5 structures, got {len(meshes)}"

    def test_mesh_vertices_in_physical_space(self, reconstructor, real_case):
        """Mesh vertices should be in mm, bounded by image physical extent."""
        meshes = reconstructor.reconstruct_all_structures(
            real_case["label"], real_case["voxel_size"]
        )

        voxel_size = real_case["voxel_size"]
        shape = real_case["shape"]
        max_physical = np.array(shape) * np.array(voxel_size)

        for label_id, mesh in meshes.items():
            # All vertices should be within the image physical bounds
            assert mesh.vertices.min() >= -1.0, (
                f"Label {label_id}: negative vertex coordinates"
            )
            for dim in range(3):
                assert mesh.vertices[:, dim].max() <= max_physical[dim] + 1.0, (
                    f"Label {label_id}: vertex exceeds physical bounds on axis {dim}"
                )

    def test_bone_mesh_larger_than_cartilage(self, reconstructor, real_case):
        """Bone meshes should have more vertices/faces than cartilage."""
        meshes = reconstructor.reconstruct_all_structures(
            real_case["label"], real_case["voxel_size"]
        )

        femur_faces = len(meshes[1].faces) if 1 in meshes else 0
        fc_faces = len(meshes[2].faces) if 2 in meshes else 0

        assert femur_faces > fc_faces, "Femur should have more faces than femoral cartilage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
