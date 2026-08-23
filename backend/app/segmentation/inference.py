"""
Inference module for the trained SegResNet model.
Uses sliding window inference on full volumes.
Integrates with the existing ground-truth pipeline as a drop-in replacement.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.amp import autocast

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    Spacing,
    EnsureType,
    AsDiscrete,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.config import MODELS_DIR
from app.segmentation.model import create_segresnet, NUM_CLASSES
from app.segmentation.dataset import TARGET_SPACING, PATCH_SIZE


class KneeSegmentationModel:
    """
    Inference wrapper for the trained SegResNet knee segmentation model.

    Usage:
        model = KneeSegmentationModel()
        model.load()
        prediction = model.predict(image_array, voxel_size)

    The output is a label volume compatible with the existing measurement,
    reconstruction, and implant matching pipeline.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or (MODELS_DIR / "segresnet_knee" / "best_model.pth")
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[torch.nn.Module] = None
        self._loaded = False

    def load(self) -> bool:
        """Load the trained model weights."""
        if not self.model_path.exists():
            print(f"Model not found at {self.model_path}")
            return False

        self.model = create_segresnet().to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._loaded = True
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        voxel_size: tuple,
        use_amp: bool = True,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """
        Run segmentation on a full 3D MRI volume.

        Args:
            image: 3D numpy array (raw MRI, shape like (160, 384, 384)).
            voxel_size: Original voxel spacing (axis0, axis1, axis2) in mm.
            use_amp: Use mixed precision for faster inference.
            overlap: Sliding window overlap ratio.

        Returns:
            Label volume (same shape as input) with integer labels 0-5.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Preprocess: add batch and channel dims, normalize
        # Shape: (1, 1, D, H, W)
        tensor = torch.from_numpy(image.astype(np.float32))
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Normalize
        nonzero_mask = tensor > 0
        if nonzero_mask.sum() > 0:
            mean = tensor[nonzero_mask].mean()
            std = tensor[nonzero_mask].std()
            tensor = (tensor - mean) / (std + 1e-8)

        tensor = tensor.to(self.device)

        # Sliding window inference
        with autocast("cuda", enabled=use_amp):
            output = sliding_window_inference(
                tensor,
                roi_size=PATCH_SIZE,
                sw_batch_size=2,
                predictor=self.model,
                overlap=overlap,
            )

        # Convert to label map
        prediction = output.argmax(dim=1).squeeze(0)  # (D, H, W)
        label_volume = prediction.cpu().numpy().astype(np.uint8)

        return label_volume

    def predict_with_confidence(
        self,
        image: np.ndarray,
        voxel_size: tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run segmentation and return both labels and confidence map.

        Returns:
            (label_volume, confidence_map) where confidence is the max softmax
            probability at each voxel (0-1 range).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        tensor = torch.from_numpy(image.astype(np.float32))
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        nonzero_mask = tensor > 0
        if nonzero_mask.sum() > 0:
            mean = tensor[nonzero_mask].mean()
            std = tensor[nonzero_mask].std()
            tensor = (tensor - mean) / (std + 1e-8)

        tensor = tensor.to(self.device)

        with torch.no_grad(), autocast("cuda", enabled=True):
            output = sliding_window_inference(
                tensor,
                roi_size=PATCH_SIZE,
                sw_batch_size=2,
                predictor=self.model,
                overlap=0.5,
            )

        # Softmax → confidence
        probs = torch.softmax(output, dim=1).squeeze(0)  # (C, D, H, W)
        confidence = probs.max(dim=0).values.cpu().numpy()  # (D, H, W)
        labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

        return labels, confidence


def evaluate_model_vs_ground_truth(
    model: KneeSegmentationModel,
    image: np.ndarray,
    ground_truth: np.ndarray,
    voxel_size: tuple,
) -> dict:
    """
    Compare AI prediction against ground-truth segmentation.
    Computes per-class Dice scores and volume differences.

    This is the bridge between the AI model and the reference baseline pipeline.
    """
    from app.config import LABELS_MAP

    prediction = model.predict(image, voxel_size)

    results = {"per_class": {}, "overall": {}}
    all_dices = []

    for label_id in range(1, NUM_CLASSES):
        pred_mask = (prediction == label_id)
        gt_mask = (ground_truth == label_id)

        # Dice coefficient
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()

        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = 2.0 * intersection / union

        # Volume difference
        voxel_vol = np.prod(voxel_size)
        pred_vol = pred_mask.sum() * voxel_vol
        gt_vol = gt_mask.sum() * voxel_vol
        vol_diff_pct = ((pred_vol - gt_vol) / gt_vol * 100) if gt_vol > 0 else 0

        results["per_class"][LABELS_MAP[label_id]] = {
            "dice": round(dice, 4),
            "pred_volume_mm3": float(pred_vol),
            "gt_volume_mm3": float(gt_vol),
            "volume_diff_pct": round(vol_diff_pct, 1),
        }
        all_dices.append(dice)

    results["overall"]["mean_dice"] = round(np.mean(all_dices), 4)
    results["overall"]["min_dice"] = round(np.min(all_dices), 4)

    return results
