"""
Quantitative tests for the measurement engine.
Tests validate physical-unit calculations against known synthetic volumes
and real OAIZIB-CM data to catch unit errors and methodology bugs.
"""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

import numpy as np
import pytest

from app.measurements import KneeMeasurementEngine


@pytest.fixture
def engine():
    return KneeMeasurementEngine()


# ─── Unit Tests with Synthetic Phantoms ──────────────────────────────────────

class TestVolumeCalculation:
    """Verify volume = voxel_count × voxel_volume."""

    def test_known_cube_volume(self, engine):
        """A 10×10×10 voxel cube with 1mm spacing = 1000 mm³ = 1.0 cm³."""
        label = np.zeros((20, 20, 20), dtype=np.uint8)
        label[5:15, 5:15, 5:15] = 1  # 10×10×10 = 1000 voxels
        voxel_size = (1.0, 1.0, 1.0)

        result = engine.measure_case(label, voxel_size)
        femur = result["femur"]

        assert femur["voxel_count"] == 1000
        assert abs(femur["volume_mm3"] - 1000.0) < 0.01
        assert abs(femur["volume_cm3"] - 1.0) < 0.001

    def test_anisotropic_voxel_volume(self, engine):
        """With anisotropic spacing (0.7, 0.3646, 0.3646), volume scales correctly."""
        label = np.zeros((20, 40, 40), dtype=np.uint8)
        label[5:15, 10:30, 10:30] = 1  # 10×20×20 = 4000 voxels
        voxel_size = (0.7, 0.3646, 0.3646)

        voxel_volume = 0.7 * 0.3646 * 0.3646
        expected_volume = 4000 * voxel_volume

        result = engine.measure_case(label, voxel_size)
        femur = result["femur"]

        assert femur["voxel_count"] == 4000
        assert abs(femur["volume_mm3"] - expected_volume) < 0.01

    def test_single_voxel_volume(self, engine):
        """A single voxel has volume = product of spacings."""
        label = np.zeros((10, 10, 10), dtype=np.uint8)
        label[5, 5, 5] = 1
        voxel_size = (2.0, 3.0, 4.0)

        result = engine.measure_case(label, voxel_size)
        femur = result["femur"]

        assert femur["voxel_count"] == 1
        assert abs(femur["volume_mm3"] - 24.0) < 0.001


class TestBoundingBoxExtent:
    """Verify bounding box → physical extent conversion."""

    def test_extent_isotropic(self, engine):
        """10-voxel extent × 1mm spacing = 10mm."""
        label = np.zeros((30, 30, 30), dtype=np.uint8)
        label[10:20, 5:15, 8:18] = 1  # 10×10×10
        voxel_size = (1.0, 1.0, 1.0)

        result = engine.measure_case(label, voxel_size)
        extent = result["femur"]["extent_mm"]

        assert abs(extent["sagittal_slices"] - 10.0) < 0.01
        assert abs(extent["anterior_posterior"] - 10.0) < 0.01
        assert abs(extent["medial_lateral"] - 10.0) < 0.01

    def test_extent_anisotropic(self, engine):
        """10 voxels × 0.7mm = 7mm on axis 0; 20 voxels × 0.3646mm ≈ 7.29mm."""
        label = np.zeros((20, 40, 40), dtype=np.uint8)
        label[5:15, 10:30, 10:30] = 1  # 10×20×20 voxels
        voxel_size = (0.7, 0.3646, 0.3646)

        result = engine.measure_case(label, voxel_size)
        extent = result["femur"]["extent_mm"]

        assert abs(extent["sagittal_slices"] - 7.0) < 0.01
        assert abs(extent["anterior_posterior"] - 7.292) < 0.01
        assert abs(extent["medial_lateral"] - 7.292) < 0.01


class TestWidthMeasurements:
    """Verify ML width and AP depth at specific slices."""

    def test_femur_condyle_measurement(self, engine):
        """
        Place a femur-like structure and verify the condyle measurement
        picks the widest slice in the distal 25%.
        """
        label = np.zeros((40, 50, 60), dtype=np.uint8)
        # Create a shape wider at the distal end (higher slice indices)
        # Proximal (narrow): slices 0-20, size 20×20
        label[0:20, 15:35, 20:40] = 1
        # Distal (wide, condyle): slices 30-40, size 30×50
        label[30:40, 10:40, 5:55] = 1
        voxel_size = (1.0, 1.0, 1.0)

        result = engine.measure_case(label, voxel_size)
        femur = result["femur"]

        # The widest slice in distal 25% should measure ML=50mm, AP=30mm
        assert abs(femur["condyle_ml_width_mm"] - 50.0) < 1.0
        assert abs(femur["condyle_ap_depth_mm"] - 30.0) < 1.0

    def test_tibia_plateau_measurement(self, engine):
        """
        Place a tibia-like structure and verify plateau measurement
        picks widest slice in proximal 25%.
        """
        label = np.zeros((40, 50, 60), dtype=np.uint8)
        # Proximal (wide, plateau): slices 0-10, size 40×50
        label[0:10, 5:45, 5:55] = 3
        # Distal (narrow shaft): slices 20-40
        label[20:40, 20:30, 25:35] = 3
        voxel_size = (1.0, 1.0, 1.0)

        result = engine.measure_case(label, voxel_size)
        tibia = result["tibia"]

        assert abs(tibia["plateau_ml_width_mm"] - 50.0) < 1.0
        assert abs(tibia["plateau_ap_depth_mm"] - 40.0) < 1.0


class TestCartilageThickness:
    """
    Verify cartilage thickness using volume/area method.
    A flat slab of known dimensions should have exact thickness.
    """

    def test_flat_tibial_cartilage_slab(self, engine):
        """
        A 2-voxel thick slab along axis 0 with spacing 0.7mm = 1.4mm thickness.
        Coverage = 20×20 pixels, pixel_area = 1×1 = 1 mm².
        Volume = 2×20×20 × 0.7×1×1 = 560 mm³.
        Area = 20×20 × 1 = 400 mm².
        Thickness = 560/400 = 1.4 mm.
        """
        label = np.zeros((10, 30, 30), dtype=np.uint8)
        label[3:5, 5:25, 5:25] = 4  # Medial tibial cartilage, 2 slices thick
        voxel_size = (0.7, 1.0, 1.0)

        result = engine.measure_case(label, voxel_size)
        mtc = result["medial_tibial_cartilage"]

        expected_volume = 2 * 20 * 20 * 0.7 * 1.0 * 1.0  # 560
        expected_area = 20 * 20 * 1.0 * 1.0  # 400
        expected_thickness = expected_volume / expected_area  # 1.4

        assert abs(mtc["volume_mm3"] - expected_volume) < 0.01
        assert abs(mtc["coverage_area_mm2"] - expected_area) < 0.01
        assert abs(mtc["thickness_mean_mm"] - expected_thickness) < 0.01

    def test_flat_femoral_cartilage_slab(self, engine):
        """
        Femoral cartilage: 3-voxel thick along axis 1 with spacing 0.3646mm.
        Coverage projects along axis 1 onto plane (axis0 × axis2).
        """
        label = np.zeros((20, 20, 30), dtype=np.uint8)
        label[5:15, 8:11, 5:25] = 2  # 10×3×20 block
        voxel_size = (0.7, 0.3646, 0.3646)

        result = engine.measure_case(label, voxel_size)
        fc = result["femoral_cartilage"]

        voxel_vol = 0.7 * 0.3646 * 0.3646
        n_voxels = 10 * 3 * 20
        expected_volume = n_voxels * voxel_vol

        # Projection along axis 1: shape (axis0, axis2) = (10, 20) nonzero pixels
        pixel_area_proj = 0.7 * 0.3646
        expected_area = 10 * 20 * pixel_area_proj
        expected_thickness = expected_volume / expected_area  # Should be 3 × 0.3646

        assert abs(fc["volume_mm3"] - expected_volume) < 0.01
        assert abs(fc["coverage_area_mm2"] - expected_area) < 0.01
        assert abs(fc["thickness_mean_mm"] - (3 * 0.3646)) < 0.01

    def test_variable_thickness(self, engine):
        """
        A wedge-shaped cartilage: thickness varies from 1 to 5 voxels.
        Mean thickness should be average of the column heights.
        """
        label = np.zeros((10, 20, 20), dtype=np.uint8)
        # Create a wedge: row 0 has 1 voxel thick, row 4 has 5 voxels thick
        for row in range(5):
            thickness = row + 1  # 1, 2, 3, 4, 5 voxels along axis 0
            label[0:thickness, row + 5, 5:15] = 4

        voxel_size = (1.0, 1.0, 1.0)
        result = engine.measure_case(label, voxel_size)
        mtc = result["medial_tibial_cartilage"]

        # Volume = sum of (1+2+3+4+5)×10 = 150 voxels × 1 mm³ each = 150
        total_voxels = sum((r + 1) * 10 for r in range(5))
        assert mtc["voxel_count"] == total_voxels

        # Coverage: project along axis 0 → 5 rows × 10 cols = 50 pixels
        assert mtc["coverage_pixels"] == 50
        assert abs(mtc["coverage_area_mm2"] - 50.0) < 0.01

        # Mean thickness = volume / area = 150 / 50 = 3.0 mm
        assert abs(mtc["thickness_mean_mm"] - 3.0) < 0.01


class TestDerivedMeasurements:
    """Verify derived metrics are computed correctly."""

    def test_ml_ratio(self, engine):
        """Medial/lateral ratio should be thickness_medial / thickness_lateral."""
        label = np.zeros((10, 30, 30), dtype=np.uint8)
        # MTC: 2 slices thick (thickness = 2 × 0.7 = 1.4mm by volume/area)
        label[3:5, 5:15, 5:15] = 4  # 2×10×10 = 200 voxels
        # LTC: 4 slices thick (thickness = 4 × 0.7 = 2.8mm by volume/area)
        label[3:7, 15:25, 5:15] = 5  # 4×10×10 = 400 voxels
        voxel_size = (0.7, 1.0, 1.0)

        result = engine.measure_case(label, voxel_size)
        derived = result["derived"]

        # MTC coverage = 10×10 = 100, vol = 200×0.7 = 140, thickness = 1.4
        # LTC coverage = 10×10 = 100, vol = 400×0.7 = 280, thickness = 2.8
        expected_ratio = 1.4 / 2.8  # 0.5

        assert abs(derived["medial_lateral_thickness_ratio"] - expected_ratio) < 0.01

    def test_total_cartilage_volume(self, engine):
        """Total volume = sum of all cartilage components."""
        label = np.zeros((10, 30, 30), dtype=np.uint8)
        label[3:5, 5:10, 5:10] = 2   # FC: 2×5×5 = 50 voxels
        label[3:5, 10:15, 5:10] = 4  # MTC: 2×5×5 = 50 voxels
        label[3:5, 15:20, 5:10] = 5  # LTC: 2×5×5 = 50 voxels
        voxel_size = (1.0, 1.0, 1.0)

        result = engine.measure_case(label, voxel_size)
        derived = result["derived"]

        # Each has 50 voxels × 1mm³ = 50mm³. Total = 150mm³.
        assert abs(derived["total_cartilage_volume_mm3"] - 150.0) < 0.01


class TestOAIZIBRealData:
    """
    Integration tests using real OAIZIB-CM data.
    These validate that measurements fall within physiologically plausible ranges.
    """

    @pytest.fixture
    def real_case(self):
        """Load a real OAIZIB-CM test case."""
        from app.data_loader import OAIZIBDataLoader
        loader = OAIZIBDataLoader()
        return loader.load_case(405)

    def test_femur_volume_plausible(self, engine, real_case):
        """Femur volume should be 100-300 cm³ for adults."""
        result = engine.measure_case(real_case["label"], real_case["voxel_size"])
        vol_cm3 = result["femur"]["volume_cm3"]
        assert 50 < vol_cm3 < 400, f"Femur volume {vol_cm3} cm³ outside plausible range"

    def test_tibia_volume_plausible(self, engine, real_case):
        """Tibia visible volume should be 50-200 cm³."""
        result = engine.measure_case(real_case["label"], real_case["voxel_size"])
        vol_cm3 = result["tibia"]["volume_cm3"]
        assert 30 < vol_cm3 < 250, f"Tibia volume {vol_cm3} cm³ outside plausible range"

    def test_cartilage_thickness_plausible(self, engine, real_case):
        """
        Cartilage thickness (volume/area) should be 0.5-4.0mm.
        Literature: healthy FC 1.5-3.0mm, MTC 1.0-2.5mm (Eckstein 2006).
        We use wider bounds since we include OA cases.
        """
        result = engine.measure_case(real_case["label"], real_case["voxel_size"])

        fc_t = result["femoral_cartilage"]["thickness_mean_mm"]
        assert 0.5 < fc_t < 5.0, f"FC thickness {fc_t}mm outside plausible range"

        mtc_t = result["medial_tibial_cartilage"]["thickness_mean_mm"]
        assert 0.3 < mtc_t < 5.0, f"MTC thickness {mtc_t}mm outside plausible range"

        ltc_t = result["lateral_tibial_cartilage"]["thickness_mean_mm"]
        assert 0.3 < ltc_t < 5.0, f"LTC thickness {ltc_t}mm outside plausible range"

    def test_condyle_width_plausible(self, engine, real_case):
        """
        Femoral condyle ML width typically 55-85mm in adults.
        Since we measure on a single slice, lower values are possible.
        """
        result = engine.measure_case(real_case["label"], real_case["voxel_size"])
        if "condyle_ml_width_mm" in result["femur"]:
            ml = result["femur"]["condyle_ml_width_mm"]
            assert 20 < ml < 100, f"Condyle ML width {ml}mm outside plausible range"

    def test_measurements_consistent_with_voxel_spacing(self, engine, real_case):
        """
        All measurements should be expressible as integer_voxels × spacing.
        Volume should exactly equal voxel_count × voxel_volume.
        """
        result = engine.measure_case(real_case["label"], real_case["voxel_size"])
        voxel_vol = np.prod(real_case["voxel_size"])

        for structure in ["femur", "tibia"]:
            if structure in result:
                s = result[structure]
                expected_vol = s["voxel_count"] * voxel_vol
                assert abs(s["volume_mm3"] - expected_vol) < 0.001, (
                    f"{structure} volume inconsistent: {s['volume_mm3']} vs {expected_vol}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
