"""Model builders (MONAI + PyTorch).

Requires the ``ml`` extra, which is installed in Colab rather than locally
(PRD §65/§120). Importing this module without torch raises a message that says
where to install it.

nnU-Net is the segmentation baseline (PRD §35). It is a self-configuring pipeline
rather than a bare architecture, so it is driven through
:mod:`kneeai.ml.nnunet` instead of being constructed here. The architectures in
this module are the comparison arm: MONAI models trained under the platform's own
loop, which is what makes ablations and per-structure loss weighting possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from kneeai.core.errors import ConfigurationError
from kneeai.ml.registry import Architecture

__all__ = [
    "ModelConfig",
    "build_model",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "count_parameters",
    "require_torch",
    "describe_device",
]

_TORCH_HINT = (
    "PyTorch/MONAI are not installed. They are intentionally excluded from the "
    "local install: training runs on Google Colab with a CUDA runtime "
    "(see notebooks/). To install locally anyway:\n"
    "    pip install -e '.[ml]'"
)


def require_torch() -> tuple[Any, Any]:
    """Import torch and monai, or raise with installation guidance."""
    try:
        import monai
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigurationError(f"{_TORCH_HINT}\n(underlying error: {exc})") from exc
    return torch, monai


def describe_device(device: str | None = None) -> dict[str, Any]:
    """Report the compute device, matching the PRD §66 CUDA verification step."""
    torch, monai = require_torch()
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    info: dict[str, Any] = {
        "device": resolved,
        "torch": torch.__version__,
        "monai": monai.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / 1024**3, 2),
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return info


@dataclass
class ModelConfig:
    """Architecture and optimisation configuration for a segmentation run."""

    architecture: Architecture = Architecture.SEGRESNET
    in_channels: int = 1
    out_channels: int = 10
    patch_size: tuple[int, int, int] = (96, 96, 96)
    spatial_dims: int = 3

    # --- capacity ---
    features: tuple[int, ...] = (32, 64, 128, 256, 320)
    dropout: float = 0.0

    #: Normalisation layer, recorded per model rather than hardcoded.
    #:
    #: This matters more than it looks. MONAI's ``SegResNet`` defaults to
    #: ``GroupNorm(8)``, which carries affine weight and bias per channel;
    #: PyTorch's ``InstanceNorm3d`` defaults to ``affine=False`` and carries
    #: neither. A checkpoint trained under one does not load into the other, and
    #: because the affine tensors are the *only* difference the failure surfaces
    #: as a pile of "unexpected" keys rather than a shape error.
    #:
    #: ``"instance"`` is the platform's historical default and is kept so existing
    #: checkpoints keep loading.
    norm: str = "instance"

    #: Group count for ``norm="group"``. Ignored otherwise.
    norm_groups: int = 8

    # --- transformer-specific ---
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    feature_size: int = 48

    # --- loss ---
    #: Dice + cross-entropy. Dice alone under-penalises small structures such as
    #: cartilage; CE alone is dominated by the background class.
    loss: str = "dice_ce"
    include_background_in_loss: bool = False

    #: Per-class weights, indexed by label value. Used to stop femur and tibia —
    #: two orders of magnitude larger than cartilage — from dominating the
    #: gradient.
    class_weights: tuple[float, ...] | None = None

    # --- optimisation ---
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    max_epochs: int = 300

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        payload["architecture"] = self.architecture.value
        return payload

    def resolved_norm(self) -> Any:
        """The normalisation argument in the form MONAI's layer factory expects.

        Kept as a plain name plus an integer on the dataclass rather than a
        ``(name, kwargs)`` tuple so it survives a JSON round trip through a
        checkpoint's ``model_config`` without turning into a list.
        """
        name = str(self.norm).lower()
        if name in ("group", "groupnorm"):
            return ("group", {"num_groups": int(self.norm_groups)})
        if name in ("instance_affine", "instance-affine"):
            return ("instance", {"affine": True})
        if name in ("batch", "batchnorm"):
            return "batch"
        if name in ("instance", "instancenorm"):
            return "instance"
        raise ConfigurationError(
            f"unknown norm {self.norm!r}; choose from "
            "'instance', 'instance_affine', 'group', 'batch'"
        )


def build_model(config: ModelConfig) -> Any:
    """Instantiate the configured architecture."""
    torch, _monai = require_torch()
    from monai.networks.nets import (
        AttentionUnet,
        SegResNet,
        SwinUNETR,
        UNet,
        UNETR,
    )

    arch = config.architecture
    if arch is Architecture.NNUNET:
        raise ConfigurationError(
            "nnU-Net is a self-configuring pipeline, not a plain architecture. "
            "Drive it through kneeai.ml.nnunet (export the dataset, then run "
            "nnUNetv2_plan_and_preprocess / nnUNetv2_train)."
        )

    if arch is Architecture.UNET:
        n_levels = len(config.features)
        return UNet(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.features,
            strides=(2,) * (n_levels - 1),
            num_res_units=2,
            dropout=config.dropout,
            norm=config.resolved_norm(),
        )

    if arch is Architecture.SEGRESNET:
        # Default choice: strong on 3D medical segmentation, tolerant of the small
        # batch sizes a Colab GPU allows, and cheap enough to train in a session.
        return SegResNet(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=config.features[0],
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=config.dropout or None,
            norm=config.resolved_norm(),
        )

    if arch is Architecture.ATTENTION_UNET:
        n_levels = len(config.features)
        return AttentionUnet(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.features,
            strides=(2,) * (n_levels - 1),
            dropout=config.dropout,
        )

    if arch is Architecture.UNETR:
        return UNETR(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            img_size=config.patch_size,
            feature_size=config.feature_size // 3 or 16,
            hidden_size=config.hidden_size,
            mlp_dim=config.mlp_dim,
            num_heads=config.num_heads,
            norm_name="instance",
            dropout_rate=config.dropout,
        )

    if arch is Architecture.SWIN_UNETR:
        return SwinUNETR(
            img_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=config.feature_size,
            drop_rate=config.dropout,
            use_checkpoint=True,  # trades compute for memory; needed on a 16 GB GPU
        )

    raise ConfigurationError(
        f"architecture {arch.value!r} has no segmentation builder. "
        f"Supported: {[a.value for a in (Architecture.UNET, Architecture.SEGRESNET, Architecture.ATTENTION_UNET, Architecture.UNETR, Architecture.SWIN_UNETR)]}"
    )


def build_loss(config: ModelConfig) -> Any:
    """Build the loss function."""
    torch, _ = require_torch()
    from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, TverskyLoss

    weights = None
    if config.class_weights is not None:
        weights = torch.as_tensor(config.class_weights, dtype=torch.float32)
        if len(config.class_weights) != config.out_channels:
            raise ConfigurationError(
                f"class_weights has {len(config.class_weights)} entries but the "
                f"model has {config.out_channels} output channels"
            )

    common = {
        "include_background": config.include_background_in_loss,
        "to_onehot_y": True,
        "softmax": True,
    }
    name = config.loss.lower()
    if name == "dice_ce":
        # MONAI renamed this argument from ``ce_weight`` to ``weight``. Pick
        # whichever the installed version accepts so the loss works on both;
        # silently dropping the weights would leave cartilage under-trained with
        # no error to explain why.
        import inspect

        weight_kwarg = (
            "weight"
            if "weight" in inspect.signature(DiceCELoss.__init__).parameters
            else "ce_weight"
        )
        return DiceCELoss(
            **common, **{weight_kwarg: weights}, lambda_dice=1.0, lambda_ce=1.0
        )
    if name == "dice":
        return DiceLoss(**common)
    if name == "dice_focal":
        return DiceFocalLoss(**common, gamma=2.0)
    if name == "tversky":
        # alpha < beta penalises false negatives more heavily. Useful for thin
        # cartilage, which models tend to under-segment.
        return TverskyLoss(**common, alpha=0.3, beta=0.7)
    raise ConfigurationError(
        f"unknown loss {config.loss!r}; expected one of "
        "dice_ce, dice, dice_focal, tversky"
    )


def build_optimizer(model: Any, config: ModelConfig) -> Any:
    torch, _ = require_torch()
    name = config.optimizer.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.99,
            nesterov=True,
            weight_decay=config.weight_decay,
        )
    raise ConfigurationError(f"unknown optimizer {config.optimizer!r}")


def build_scheduler(optimizer: Any, config: ModelConfig) -> Any | None:
    torch, _ = require_torch()
    name = (config.scheduler or "none").lower()
    if name in ("none", ""):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.max_epochs)
        )
    if name == "poly":
        # nnU-Net's schedule. Included so the comparison arm can match it.
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1.0 - epoch / max(1, config.max_epochs)) ** 0.9,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=15
        )
    raise ConfigurationError(f"unknown scheduler {config.scheduler!r}")


def count_parameters(model: Any) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def estimate_activation_memory_gb(
    config: ModelConfig, batch_size: int = 2
) -> float:
    """Rough activation-memory estimate, to catch OOM before a run starts.

    Very approximate. Its purpose is to flag an obviously infeasible patch size
    at configuration time rather than 40 minutes into a Colab session.
    """
    voxels = 1
    for n in config.patch_size:
        voxels *= n
    # Sum of feature maps across a U-shaped encoder/decoder, halving spatially per
    # level, times 4 bytes, times ~3 for gradients and optimiser workspace.
    total = 0.0
    for level, channels in enumerate(config.features):
        level_voxels = voxels / (8**level)
        total += level_voxels * channels
    total *= 2  # decoder mirrors the encoder
    return round(total * batch_size * 4 * 3 / 1024**3, 2)
