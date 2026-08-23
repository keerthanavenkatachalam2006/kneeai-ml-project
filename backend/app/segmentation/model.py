"""
MONAI SegResNet model definition for knee MRI segmentation.

Architecture: SegResNet (Myronenko, 2019) — a lightweight 3D encoder-decoder
with residual connections. Chosen for:
  - Strong performance on medical segmentation benchmarks
  - Low memory footprint compared to SwinUNETR / nnU-Net
  - Works well with limited VRAM (6GB RTX 4050)

Input: 3D MRI patches (single-channel)
Output: 6-class segmentation (background + 5 structures)

Label map:
  0: Background
  1: Femur
  2: Femoral Cartilage
  3: Tibia
  4: Medial Tibial Cartilage
  5: Lateral Tibial Cartilage
"""
import torch
from monai.networks.nets import SegResNet


NUM_CLASSES = 6  # Background + 5 structures


def create_segresnet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = NUM_CLASSES,
    init_filters: int = 16,
    blocks_down: tuple = (1, 2, 2, 4),
    blocks_up: tuple = (1, 1, 1),
    dropout_prob: float = 0.2,
) -> SegResNet:
    """
    Create SegResNet for knee MRI segmentation.

    With init_filters=16 and blocks_down=(1,2,2,4):
    - Total parameters: ~1.2M (fits comfortably in 6GB VRAM)
    - Feature maps: 16 → 32 → 64 → 128 at deepest level
    - Receptive field sufficient for knee structures

    Args:
        spatial_dims: 3 for volumetric segmentation.
        in_channels: 1 (grayscale MRI).
        out_channels: 6 (background + 5 structures).
        init_filters: Initial feature map count.
        blocks_down: Residual blocks per encoder level.
        blocks_up: Residual blocks per decoder level.
        dropout_prob: Dropout probability.

    Returns:
        SegResNet model instance.
    """
    model = SegResNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=dropout_prob,
    )
    return model


def get_model_summary(model: SegResNet) -> dict:
    """Get model parameter count and architecture summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "architecture": "SegResNet",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_params_M": round(total_params / 1e6, 2),
        "in_channels": 1,
        "out_channels": NUM_CLASSES,
    }
