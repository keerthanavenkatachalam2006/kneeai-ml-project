"""
Anatomical Measurement Engine.
Extracts physical dimensions from segmentation masks using voxel spacing.
Computes clinically relevant knee measurements for implant planning.

Methodology Notes:
- Volume: Sum of labeled voxels × voxel_volume (exact for discrete masks).
- Bone widths: Bounding-box extent of the mask at a specific slice level,
  converted to mm via voxel spacing. This is a simplified approximation;
  clinical measurement uses specific anatomical landmarks.
- Cartilage thickness: Computed as volume / coverage_area, which gives the
  mean thickness averaged over the articular surface projection. This avoids
  the axis-alignment bias of column-counting methods.
- All measurements use the NIfTI voxel spacing from the image header.
  OAIZIB-CM voxel size: (0.7, 0.3646, 0.3646) mm for (axis0, axis1, axis2).
  Axis 0 = sagittal slices, Axis 1 = anterior-posterior, Axis 2 = medial-lateral.

Limitations:
- Width/depth measurements are bounding-box approximations, not landmark-based.
- Cartilage thickness is a volumetric average, not a point-wise surface normal
  measurement (which would require mesh-based distance computation).
- Measurements are from the full MRI field of view, not from a surgical cut plane.
"""
import numpy as np
from scipy.ndimage import center_of_mass
from typing import Optional

from .config import LABELS_MAP


class KneeMeasurementEngine:
    """
    Extracts anatomical measurements from knee segmentation masks.

    All measurements are returned in millimeters using voxel spacing metadata.
    Voxel size order: (axis0_spacing, axis1_spacing, axis2_spacing) in mm.

    For OAIZIB-CM (verified from NIfTI affine):
      axis0 = Left→Right direction (ML, 0.7 mm spacing, 160 slices)
      axis1 = Posterior→Anterior direction (AP, 0.3646 mm spacing)
      axis2 = Inferior→Superior direction (SI, 0.3646 mm spacing)
    """

    def __init__(self):
        pass

    def _estimate_joint_line_si(self, label_volume: np.ndarray) -> Optional[int]:
        """
        Estimate the SI slice index of the knee joint line.

        Uses cartilage labels (2=femoral cartilage, 4=MTC, 5=LTC) as the
        most reliable indicator of the joint. Their SI center is the joint line.
        If no cartilage is present, falls back to the transition between femur
        and tibia density along SI.

        Returns:
            SI slice index of the joint line, or None if undetermined.
        """
        # Cartilage-based joint line (most robust)
        cartilage_mask = (label_volume == 2) | (label_volume == 4) | (label_volume == 5)
        if cartilage_mask.sum() > 100:
            cart_coords = np.argwhere(cartilage_mask)
            return int(np.median(cart_coords[:, 2]))

        # Fallback: find where femur and tibia SI distributions overlap
        femur_si = (label_volume == 1).sum(axis=(0, 1))
        tibia_si = (label_volume == 3).sum(axis=(0, 1))

        # Joint line = where femur density decreases and tibia increases
        # Use the crossover point
        femur_slices = np.where(femur_si > 0)[0]
        tibia_slices = np.where(tibia_si > 0)[0]

        if len(femur_slices) > 0 and len(tibia_slices) > 0:
            femur_center = np.median(femur_slices)
            tibia_center = np.median(tibia_slices)
            return int((femur_center + tibia_center) / 2)

        return None

    def measure_case(
        self,
        label_volume: np.ndarray,
        voxel_size: tuple,
    ) -> dict:
        """
        Compute all measurements for a segmented knee case.

        Args:
            label_volume: 3D array with integer labels (0-5).
            voxel_size: (axis0_spacing, axis1_spacing, axis2_spacing) in mm.

        Returns:
            Dictionary of measurements with units.
        """
        measurements = {}

        # Compute joint line estimate for robust bone measurements.
        # The joint line is where femur and tibia meet along the SI axis.
        # Even with noisy predictions, the cartilage labels (2,4,5) indicate
        # the joint region. If cartilage is present, its SI center is the joint.
        joint_si = self._estimate_joint_line_si(label_volume)

        # Femur measurements
        femur_mask = (label_volume == 1).astype(np.uint8)
        if femur_mask.sum() > 0:
            measurements["femur"] = self._measure_bone(
                femur_mask, voxel_size, "femur", joint_si=joint_si
            )

        # Tibia measurements
        tibia_mask = (label_volume == 3).astype(np.uint8)
        if tibia_mask.sum() > 0:
            measurements["tibia"] = self._measure_bone(
                tibia_mask, voxel_size, "tibia", joint_si=joint_si
            )

        # Femoral cartilage
        fc_mask = (label_volume == 2).astype(np.uint8)
        if fc_mask.sum() > 0:
            measurements["femoral_cartilage"] = self._measure_cartilage(
                fc_mask, voxel_size, "femoral"
            )

        # Medial tibial cartilage
        mtc_mask = (label_volume == 4).astype(np.uint8)
        if mtc_mask.sum() > 0:
            measurements["medial_tibial_cartilage"] = self._measure_cartilage(
                mtc_mask, voxel_size, "medial_tibial"
            )

        # Lateral tibial cartilage
        ltc_mask = (label_volume == 5).astype(np.uint8)
        if ltc_mask.sum() > 0:
            measurements["lateral_tibial_cartilage"] = self._measure_cartilage(
                ltc_mask, voxel_size, "lateral_tibial"
            )

        # Medial meniscus (label 6, from meniscus estimation module)
        meniscus_mask = (label_volume == 6).astype(np.uint8)
        if meniscus_mask.sum() > 0:
            measurements["medial_meniscus"] = self._measure_meniscus(
                meniscus_mask, voxel_size
            )

        # Derived measurements
        measurements["derived"] = self._compute_derived(measurements, voxel_size)

        return measurements

    def _measure_bone(
        self, mask: np.ndarray, voxel_size: tuple, bone_name: str,
        joint_si: Optional[int] = None,
    ) -> dict:
        """Measure bone dimensions from binary mask."""
        coords = np.argwhere(mask > 0)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        # Physical dimensions (bounding box extent)
        extent_voxels = max_coords - min_coords + 1
        extent_mm = extent_voxels * np.array(voxel_size)

        # Volume: count × voxel_volume
        volume_voxels = int(mask.sum())
        voxel_volume_mm3 = float(np.prod(voxel_size))
        volume_mm3 = volume_voxels * voxel_volume_mm3

        # Center of mass in physical coordinates
        com = center_of_mass(mask)
        com_mm = np.array(com) * np.array(voxel_size)

        # Width measurements at clinically relevant slices
        widths = self._measure_widths_at_slices(mask, voxel_size, bone_name, joint_si)

        return {
            "extent_mm": {
                "medial_lateral": float(extent_mm[0]),
                "anterior_posterior": float(extent_mm[1]),
                "superior_inferior": float(extent_mm[2]),
            },
            "volume_mm3": float(volume_mm3),
            "volume_cm3": float(volume_mm3 / 1000),
            "center_of_mass_mm": com_mm.tolist(),
            "voxel_count": volume_voxels,
            "bounding_box_voxels": {
                "min": min_coords.tolist(),
                "max": max_coords.tolist(),
            },
            **widths,
        }

    def _measure_widths_at_slices(
        self, mask: np.ndarray, voxel_size: tuple, bone_name: str,
        joint_si: Optional[int] = None,
    ) -> dict:
        """
        Measure ML width and AP depth at clinically relevant anatomical regions.

        OAIZIB-CM NIfTI orientation (from affine, verified):
          - Axis 0 = Left→Right (ML direction, 0.7mm spacing, 160 slices)
          - Axis 1 = Posterior→Anterior (AP direction, 0.3646mm spacing)
          - Axis 2 = Inferior→Superior (SI direction, 0.3646mm spacing)

        Strategy (robust to noisy AI predictions):
          - Use the joint line (from cartilage center) as anatomical anchor.
          - For femur: condyle is NEAR and BELOW the joint line.
            Search for densest slice in range [joint_si - 30, joint_si + 5].
          - For tibia: plateau is NEAR and BELOW the joint line.
            Search for densest slice in range [joint_si - 30, joint_si].
          - If no joint line, fall back to density-weighted median approach.
          - Within the measurement slice, use 2-98th percentile to trim noise.

        Expected ranges (literature):
          - Femoral condyle ML: 55-80mm
          - Femoral condyle AP: 55-75mm
          - Tibial plateau ML: 60-85mm
          - Tibial plateau AP: 40-55mm
        """
        results = {}

        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return results

        si_projection = mask.sum(axis=(0, 1))

        # Determine search range for the measurement slice
        if joint_si is not None:
            # Joint-anchored search (most robust)
            # The condyle and plateau are both NEAR the joint line
            # Femur condyle: just above the joint (higher SI = more superior)
            # Tibia plateau: just below the joint (lower SI = more inferior)
            margin = 30  # search ±30 slices (~11mm) from joint

            if bone_name == "femur":
                # Condyle is at/above the joint line
                search_start = max(0, joint_si - 5)
                search_end = min(mask.shape[2] - 1, joint_si + margin)
            elif bone_name == "tibia":
                # Plateau is at/below the joint line
                search_start = max(0, joint_si - margin)
                search_end = min(mask.shape[2] - 1, joint_si + 5)
            else:
                return results
        else:
            # Fallback: use median-based approach
            si_median = np.median(coords[:, 2])
            if bone_name == "femur":
                search_start = int(si_median)
                search_end = mask.shape[2] - 1
            elif bone_name == "tibia":
                search_start = 0
                search_end = int(si_median)
            else:
                return results

        # Find densest slice in the search range
        best_area = 0
        best_slice_idx = (search_start + search_end) // 2
        for s in range(search_start, search_end + 1):
            area = si_projection[s]
            if area > best_area:
                best_area = area
                best_slice_idx = s

        if best_area == 0:
            return results

        widest_slice = mask[:, :, best_slice_idx]
        slice_coords = np.argwhere(widest_slice > 0)

        if len(slice_coords) == 0:
            return results

        # Measure extent with percentile trimming
        if len(slice_coords) > 50:
            ml_p2 = np.percentile(slice_coords[:, 0], 2)
            ml_p98 = np.percentile(slice_coords[:, 0], 98)
            ap_p2 = np.percentile(slice_coords[:, 1], 2)
            ap_p98 = np.percentile(slice_coords[:, 1], 98)
            ml_voxels = int(ml_p98 - ml_p2 + 1)
            ap_voxels = int(ap_p98 - ap_p2 + 1)
        else:
            ml_voxels = slice_coords[:, 0].max() - slice_coords[:, 0].min() + 1
            ap_voxels = slice_coords[:, 1].max() - slice_coords[:, 1].min() + 1

        ml_mm = float(ml_voxels * voxel_size[0])
        ap_mm = float(ap_voxels * voxel_size[1])

        prefix = "condyle" if bone_name == "femur" else "plateau"
        results[f"{prefix}_ml_width_mm"] = ml_mm
        results[f"{prefix}_ap_depth_mm"] = ap_mm
        results[f"{prefix}_measurement_slice_idx"] = int(best_slice_idx)
        results[f"{prefix}_cross_section_area_mm2"] = float(
            best_area * voxel_size[0] * voxel_size[1]
        )

        return results

    def _measure_cartilage(
        self, mask: np.ndarray, voxel_size: tuple, region: str
    ) -> dict:
        """
        Measure cartilage thickness and coverage.

        Thickness methodology: volume / projected_surface_area.
        This gives the mean thickness averaged over the articular surface,
        which is the standard volumetric approach used in OA research
        (Eckstein et al., 2006). It avoids axis-alignment bias that occurs
        when measuring column heights along a fixed axis.

        For tibial cartilage: project onto the axial plane (axis 1 × axis 2).
        For femoral cartilage: project onto the sagittal-axial plane (axis 0 × axis 2).
        """
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return {
                "thickness_mean_mm": 0.0,
                "volume_mm3": 0.0,
                "volume_cm3": 0.0,
                "coverage_area_mm2": 0.0,
            }

        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        extent_voxels = max_coords - min_coords + 1
        extent_mm = extent_voxels * np.array(voxel_size)

        # Volume: exact count of labeled voxels × voxel volume
        volume_voxels = int(mask.sum())
        voxel_volume_mm3 = float(np.prod(voxel_size))
        volume_mm3 = volume_voxels * voxel_volume_mm3

        # Coverage area: number of pixels in the surface projection × pixel area
        if "tibial" in region:
            # Tibial cartilage: project along axis 0 (superior view, axial plane)
            projection = mask.max(axis=0)
            pixel_area = voxel_size[1] * voxel_size[2]
        else:
            # Femoral cartilage: project along axis 1 (radial/depth direction)
            projection = mask.max(axis=1)
            pixel_area = voxel_size[0] * voxel_size[2]

        coverage_pixels = int(projection.sum())
        coverage_area_mm2 = coverage_pixels * pixel_area

        # Mean thickness = volume / coverage_area
        # This is the standard volumetric thickness estimate
        if coverage_area_mm2 > 0:
            thickness_mean_mm = volume_mm3 / coverage_area_mm2
        else:
            thickness_mean_mm = 0.0

        # Per-column thickness for distribution stats (along projection axis)
        thickness_stats = self._compute_thickness_distribution(
            mask, voxel_size, region
        )

        return {
            "thickness_mean_mm": float(thickness_mean_mm),
            "thickness_max_mm": thickness_stats["max"],
            "thickness_min_mm": thickness_stats["min"],
            "thickness_std_mm": thickness_stats["std"],
            "thickness_method": "volume_over_area",
            "volume_mm3": float(volume_mm3),
            "volume_cm3": float(volume_mm3 / 1000),
            "coverage_area_mm2": float(coverage_area_mm2),
            "coverage_pixels": coverage_pixels,
            "voxel_count": volume_voxels,
            "extent_mm": {
                "sagittal_slices": float(extent_mm[0]),
                "anterior_posterior": float(extent_mm[1]),
                "medial_lateral": float(extent_mm[2]),
            },
        }

    def _measure_meniscus(
        self, mask: np.ndarray, voxel_size: tuple
    ) -> dict:
        """
        Measure medial meniscus dimensions.

        Computes:
        - Volume
        - Mean thickness (volume / coverage area, same as cartilage method)
        - Regional thickness: anterior, body (middle), posterior thirds

        The meniscus is divided into thirds along the AP axis for regional
        thickness reporting, which is the standard clinical approach.
        """
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return {"thickness_mean_mm": 0.0, "volume_mm3": 0.0}

        # Volume
        volume_voxels = int(mask.sum())
        voxel_volume_mm3 = float(np.prod(voxel_size))
        volume_mm3 = volume_voxels * voxel_volume_mm3

        # Coverage area: project along slice axis (axis 0)
        projection = mask.max(axis=0)
        pixel_area = voxel_size[1] * voxel_size[2]
        coverage_pixels = int(projection.sum())
        coverage_area_mm2 = coverage_pixels * pixel_area

        # Mean thickness
        if coverage_area_mm2 > 0:
            thickness_mean_mm = volume_mm3 / coverage_area_mm2
        else:
            thickness_mean_mm = 0.0

        # Regional thickness (anterior / body / posterior thirds)
        # Divide along axis 1 (AP direction)
        min_ap = coords[:, 1].min()
        max_ap = coords[:, 1].max()
        ap_range = max_ap - min_ap + 1

        if ap_range > 3:
            third = ap_range // 3
            regions = {
                "anterior": (min_ap, min_ap + third),
                "body": (min_ap + third, min_ap + 2 * third),
                "posterior": (min_ap + 2 * third, max_ap + 1),
            }

            regional_thickness = {}
            for region_name, (start, end) in regions.items():
                region_mask = mask.copy()
                region_mask[:, :start, :] = 0
                region_mask[:, end:, :] = 0

                region_voxels = region_mask.sum()
                if region_voxels > 0:
                    region_proj = region_mask.max(axis=0)
                    region_area = region_proj.sum() * pixel_area
                    region_vol = region_voxels * voxel_volume_mm3
                    if region_area > 0:
                        regional_thickness[f"{region_name}_mm"] = round(
                            region_vol / region_area, 2
                        )
                    else:
                        regional_thickness[f"{region_name}_mm"] = 0.0
                else:
                    regional_thickness[f"{region_name}_mm"] = 0.0
        else:
            regional_thickness = {
                "anterior_mm": thickness_mean_mm,
                "body_mm": thickness_mean_mm,
                "posterior_mm": thickness_mean_mm,
            }

        # Extent
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        extent_voxels = max_coords - min_coords + 1
        extent_mm = extent_voxels * np.array(voxel_size)

        return {
            "thickness_mean_mm": float(thickness_mean_mm),
            "regional_thickness": regional_thickness,
            "volume_mm3": float(volume_mm3),
            "volume_cm3": float(volume_mm3 / 1000),
            "coverage_area_mm2": float(coverage_area_mm2),
            "voxel_count": volume_voxels,
            "extent_mm": {
                "sagittal_slices": float(extent_mm[0]),
                "anterior_posterior": float(extent_mm[1]),
                "medial_lateral": float(extent_mm[2]),
            },
            "method": "anatomical_estimation",
            "disclaimer": (
                "Meniscus segmentation is based on anatomical priors from "
                "adjacent structures, not a dedicated trained model. Measurements "
                "should be interpreted with caution."
            ),
        }

    def _compute_thickness_distribution(
        self, mask: np.ndarray, voxel_size: tuple, region: str
    ) -> dict:
        """
        Compute per-column thickness distribution for min/max/std.

        Each 'column' is a 1D line through the volume along the thickness
        direction. The thickness at that column = number_of_voxels × spacing.

        This is an approximation (assumes thickness is axis-aligned), but
        provides useful distribution statistics. The mean thickness reported
        by the parent function uses the more accurate volume/area method.
        """
        if "tibial" in region:
            # Thickness direction = axis 0 (slice direction, 0.7mm)
            spacing = voxel_size[0]
            # Sum along axis 0 gives column heights
            column_heights = mask.sum(axis=0)
        else:
            # Femoral cartilage: thickness along axis 1 (AP direction, 0.3646mm)
            spacing = voxel_size[1]
            column_heights = mask.sum(axis=1)

        # Only consider columns that have cartilage
        nonzero = column_heights[column_heights > 0]

        if len(nonzero) == 0:
            return {"max": 0.0, "min": 0.0, "std": 0.0}

        thicknesses_mm = nonzero * spacing

        return {
            "max": float(thicknesses_mm.max()),
            "min": float(thicknesses_mm.min()),
            "std": float(thicknesses_mm.std()),
        }

    def _compute_derived(self, measurements: dict, voxel_size: tuple) -> dict:
        """Compute derived measurements from individual structures."""
        derived = {}

        # Cartilage thickness ratio (medial vs lateral)
        mtc = measurements.get("medial_tibial_cartilage", {})
        ltc = measurements.get("lateral_tibial_cartilage", {})

        if mtc.get("thickness_mean_mm", 0) > 0 and ltc.get("thickness_mean_mm", 0) > 0:
            derived["medial_lateral_thickness_ratio"] = (
                mtc["thickness_mean_mm"] / ltc["thickness_mean_mm"]
            )

        # Joint space width estimate (gap between femur and tibia)
        fc = measurements.get("femoral_cartilage", {})
        if fc.get("thickness_mean_mm", 0) > 0 and mtc.get("thickness_mean_mm", 0) > 0:
            derived["estimated_medial_joint_space_mm"] = (
                fc["thickness_mean_mm"] + mtc["thickness_mean_mm"]
            )

        # Total cartilage volume
        total_cart_vol = sum(
            measurements.get(k, {}).get("volume_mm3", 0)
            for k in ["femoral_cartilage", "medial_tibial_cartilage", "lateral_tibial_cartilage"]
        )
        derived["total_cartilage_volume_mm3"] = total_cart_vol
        derived["total_cartilage_volume_cm3"] = total_cart_vol / 1000

        return derived
