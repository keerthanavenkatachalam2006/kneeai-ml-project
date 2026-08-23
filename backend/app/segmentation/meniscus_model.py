"""
Meniscus Segmentation Module.

Strategy: Anatomical-prior based meniscus estimation from existing bone/cartilage
segmentation. The medial meniscus sits between the femoral condyle and the tibial
plateau in the medial compartment, adjacent to the medial tibial cartilage.

This uses a geometric approach:
1. Locate the medial tibial cartilage (label 4) from the SegResNet prediction
2. The medial meniscus is the C-shaped structure immediately peripheral
   (lateral edge) to the medial tibial cartilage, wedged between femur and tibia
3. Use the gap between femur and tibia in the medial compartment to estimate
   meniscus location

This is NOT a learned meniscus model — it's a geometry-based estimation.
When a proper pretrained meniscus model becomes available (e.g., from IWOAI challenge
weights or MRAnnotator), it should replace this module via the same interface.

The output adds label 6 = Medial Meniscus to the segmentation volume.
"""
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_closing,
    gaussian_filter,
    label as connected_components,
)
from typing import Optional


MENISCUS_LABEL = 6


class MeniscusEstimator:
    """
    Estimates medial meniscus location from existing knee segmentation.
    
    Uses anatomical priors:
    - Meniscus sits on the tibial plateau, peripheral to the articular cartilage
    - It occupies the joint gap between femur and tibia
    - It is wedge-shaped (thicker at periphery, thinner at inner edge)
    - It is C-shaped when viewed from above (axial)
    
    This produces a reasonable meniscus mask for thickness measurement
    and 3D visualization, though it is NOT as accurate as a dedicated
    trained model.
    """

    def __init__(self):
        # Label IDs from OAIZIB-CM
        self.FEMUR = 1
        self.FEMORAL_CART = 2
        self.TIBIA = 3
        self.MEDIAL_TIBIAL_CART = 4
        self.LATERAL_TIBIAL_CART = 5

    def estimate_meniscus(
        self,
        label_volume: np.ndarray,
        image_volume: Optional[np.ndarray] = None,
        voxel_size: tuple = (0.7, 0.3646, 0.3646),
    ) -> np.ndarray:
        """
        Estimate medial meniscus location from existing segmentation.

        Args:
            label_volume: 3D array with labels 0-5 from SegResNet prediction.
            image_volume: Original MRI volume (optional, for intensity-based refinement).
            voxel_size: Voxel dimensions in mm.

        Returns:
            Updated label volume with label 6 = Medial Meniscus added.
        """
        result = label_volume.copy()

        # Extract relevant masks
        femur = (label_volume == self.FEMUR)
        tibia = (label_volume == self.TIBIA)
        medial_cart = (label_volume == self.MEDIAL_TIBIAL_CART)

        # Check we have the required structures
        if medial_cart.sum() == 0 or femur.sum() == 0 or tibia.sum() == 0:
            print("  [Meniscus] Warning: Missing required structures, skipping estimation")
            return result

        # Step 1: Find the joint space (gap between femur and tibia)
        joint_gap = self._find_joint_gap(femur, tibia, voxel_size)

        # Step 2: Find the medial region (where medial tibial cartilage is)
        medial_region = self._find_medial_region(medial_cart, voxel_size)

        # Step 3: Estimate meniscus as the peripheral part of the joint gap
        # in the medial compartment, excluding existing cartilage
        meniscus_mask = self._estimate_meniscus_region(
            joint_gap, medial_region, medial_cart, femur, tibia, label_volume, voxel_size
        )

        # Step 4: Refine using intensity if MRI available
        if image_volume is not None:
            meniscus_mask = self._refine_with_intensity(
                meniscus_mask, image_volume, label_volume
            )

        # Step 5: Clean up the mask
        meniscus_mask = self._postprocess(meniscus_mask)

        # Only add meniscus where there's currently background
        meniscus_mask = meniscus_mask & (label_volume == 0)

        # Add to label volume
        result[meniscus_mask] = MENISCUS_LABEL

        return result

    def _find_joint_gap(
        self, femur: np.ndarray, tibia: np.ndarray, voxel_size: tuple
    ) -> np.ndarray:
        """
        Find the space between femur and tibia.
        Dilate both bones and find their overlap region (approximates the joint).
        """
        # Use iterations-based dilation (much faster than large structuring elements)
        # ~5mm dilation: iterations = ceil(5 / min_spacing)
        iterations = max(2, int(5.0 / min(voxel_size)))

        femur_dilated = binary_dilation(femur, iterations=iterations)
        tibia_dilated = binary_dilation(tibia, iterations=iterations)

        # Joint gap = overlap of dilated bones minus actual bones
        joint_region = femur_dilated & tibia_dilated
        joint_gap = joint_region & ~femur & ~tibia

        return joint_gap

    def _find_medial_region(
        self, medial_cart: np.ndarray, voxel_size: tuple
    ) -> np.ndarray:
        """
        Define the medial compartment region based on medial tibial cartilage location.
        Expand it peripherally to capture where the meniscus would be.
        """
        # ~12mm dilation around cartilage to define the medial zone
        iterations = max(3, int(12.0 / min(voxel_size)))
        medial_region = binary_dilation(medial_cart, iterations=iterations)
        return medial_region

    def _estimate_meniscus_region(
        self,
        joint_gap: np.ndarray,
        medial_region: np.ndarray,
        medial_cart: np.ndarray,
        femur: np.ndarray,
        tibia: np.ndarray,
        label_volume: np.ndarray,
        voxel_size: tuple,
    ) -> np.ndarray:
        """
        Combine joint gap and medial region to estimate meniscus location.
        The meniscus is in the joint gap, in the medial compartment,
        peripheral to the articular cartilage.
        """
        # Meniscus candidate = joint gap ∩ medial region
        candidate = joint_gap & medial_region

        # Exclude existing labeled structures
        existing = (label_volume > 0)
        candidate = candidate & ~existing

        # The meniscus is peripheral to the cartilage, not central
        # Erode medial cartilage slightly and exclude its center
        cart_core = binary_erosion(medial_cart, iterations=2)
        # Peripheral zone = dilated cartilage minus core
        cart_peripheral = binary_dilation(medial_cart, iterations=4) & ~cart_core

        # Meniscus should be in the peripheral zone of the cartilage within the joint gap
        meniscus = candidate & cart_peripheral

        # Also add: parts of joint gap that are directly adjacent to tibia surface
        # in the medial compartment (the meniscus rests on the tibial plateau)
        tibia_surface = binary_dilation(tibia, iterations=1) & ~tibia
        on_tibia = candidate & tibia_surface & medial_region
        meniscus = meniscus | on_tibia

        return meniscus

    def _refine_with_intensity(
        self,
        meniscus_mask: np.ndarray,
        image: np.ndarray,
        label_volume: np.ndarray,
    ) -> np.ndarray:
        """
        Refine meniscus estimation using MRI intensity.
        Meniscus appears dark (low signal) on DESS MRI due to its
        fibrocartilaginous composition, while cartilage appears bright.
        """
        if meniscus_mask.sum() == 0:
            return meniscus_mask

        # Get intensity statistics in the candidate region
        candidate_intensities = image[meniscus_mask]
        if len(candidate_intensities) == 0:
            return meniscus_mask

        # Meniscus is typically darker than cartilage on DESS
        # Get cartilage intensity as reference
        cart_mask = (label_volume == self.MEDIAL_TIBIAL_CART) | (label_volume == self.FEMORAL_CART)
        if cart_mask.sum() > 0:
            cart_intensity = np.median(image[cart_mask])
            # Meniscus should be darker than cartilage
            # Keep only voxels that are at most 80% of cartilage brightness
            threshold = cart_intensity * 0.8
            intensity_filter = image < threshold
            meniscus_mask = meniscus_mask & intensity_filter

        return meniscus_mask

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Clean up the meniscus mask: close holes, remove small components."""
        if mask.sum() == 0:
            return mask

        # Morphological closing to fill small gaps
        mask = binary_closing(mask, iterations=2)

        # Keep only the largest connected component
        labeled, num_features = connected_components(mask)
        if num_features > 1:
            sizes = [0] + [(labeled == i).sum() for i in range(1, num_features + 1)]
            largest = np.argmax(sizes)
            mask = (labeled == largest)

        # Remove very small predictions (noise)
        if mask.sum() < 100:
            mask = np.zeros_like(mask, dtype=bool)

        return mask


def add_meniscus_to_segmentation(
    label_volume: np.ndarray,
    image_volume: Optional[np.ndarray] = None,
    voxel_size: tuple = (0.7, 0.3646, 0.3646),
) -> np.ndarray:
    """
    Convenience function: add meniscus estimation to an existing segmentation.
    
    Args:
        label_volume: 3D array with labels 0-5 from SegResNet.
        image_volume: Original MRI (optional, improves accuracy).
        voxel_size: Voxel spacing in mm.
    
    Returns:
        Label volume with label 6 = Medial Meniscus added.
    """
    estimator = MeniscusEstimator()
    return estimator.estimate_meniscus(label_volume, image_volume, voxel_size)
