"""
MONAI SegResNet Training Script for OAIZIB-CM Knee Segmentation.

Hardware target: RTX 4050 (6GB VRAM)
Strategy:
  - Patch-based training: 96×96×96 patches
  - Batch size: 2 (with 2 patches per sample from RandCropByPosNegLabel)
  - Mixed precision (AMP) to halve memory usage
  - Gradient accumulation if needed
  - Validation: sliding window inference on full volumes

Expected training time: ~2-4 hours for 100 epochs on 404 training cases.
"""
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from monai.data import DataLoader, CacheDataset, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.config import MODELS_DIR, PROCESSED_DATA
from app.segmentation.model import create_segresnet, get_model_summary, NUM_CLASSES
from app.segmentation.dataset import (
    prepare_data_list,
    extract_all_data,
    get_train_transforms,
    get_val_transforms,
    PATCH_SIZE,
)


# ─── Training Configuration ──────────────────────────────────────────────────

class TrainConfig:
    """Training hyperparameters optimized for RTX 4050 (6GB)."""
    # Data
    patch_size: tuple = PATCH_SIZE  # (64, 128, 128) — native resolution
    num_workers: int = 0  # Windows requires 0 (no fork)
    cache_rate: float = 0.0  # Don't cache in RAM (too many volumes)

    # Training
    max_epochs: int = 100
    batch_size: int = 1  # 1 volume → 2 patches via RandCropByPosNegLabel
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Model
    init_filters: int = 16
    blocks_down: tuple = (1, 2, 2, 4)
    blocks_up: tuple = (1, 1, 1)
    dropout: float = 0.2

    # Optimization
    use_amp: bool = True  # Mixed precision — essential for 6GB
    grad_accum_steps: int = 2  # Effective batch size = batch_size × 2 × grad_accum = 4

    # Validation
    val_interval: int = 5  # Validate every N epochs
    sw_batch_size: int = 2  # Sliding window batch size for inference
    sw_overlap: float = 0.5

    # Paths
    output_dir: Path = MODELS_DIR / "segresnet_knee"
    best_model_path: Path = MODELS_DIR / "segresnet_knee" / "best_model.pth"
    log_path: Path = MODELS_DIR / "segresnet_knee" / "training_log.json"


def train(config: TrainConfig = None):
    """Full training loop."""
    if config is None:
        config = TrainConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ─── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/5] Preparing data...")
    extract_dir = PROCESSED_DATA / "extracted"
    if not (extract_dir / "imagesTr").exists():
        print("  Extracting dataset from ZIPs (first run only)...")
        extract_all_data(extract_dir)

    train_files, val_files = prepare_data_list(extract_dir)
    print(f"  Training cases: {len(train_files)}")
    print(f"  Validation cases: {len(val_files)}")

    # Use first 10% of val for quick validation during training
    val_subset = val_files[:max(5, len(val_files) // 2)]
    print(f"  Validation subset (for speed): {len(val_subset)}")

    train_transforms = get_train_transforms(config.patch_size)
    val_transforms = get_val_transforms()

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_subset, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ─── Model ────────────────────────────────────────────────────────────────
    print("\n[2/5] Creating model...")
    model = create_segresnet(
        init_filters=config.init_filters,
        blocks_down=config.blocks_down,
        blocks_up=config.blocks_up,
        dropout_prob=config.dropout,
    ).to(device)

    summary = get_model_summary(model)
    print(f"  Architecture: {summary['architecture']}")
    print(f"  Parameters: {summary['total_params_M']}M")

    # ─── Loss & Optimizer ─────────────────────────────────────────────────────
    print("\n[3/5] Setting up training...")
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=config.use_amp)

    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])

    # ─── Training Loop ────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {config.max_epochs} epochs...")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Batch size: {config.batch_size} (effective: {config.batch_size * 2 * config.grad_accum_steps})")
    print(f"  Mixed precision: {config.use_amp}")
    print(f"  Validation every: {config.val_interval} epochs")
    print()

    best_metric = -1
    best_epoch = -1
    training_log = []
    start_time = time.time()

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0
        step = 0

        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            with autocast("cuda", enabled=config.use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss = loss / config.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.grad_accum_steps
            step += 1

        scheduler.step()
        epoch_loss /= max(step, 1)

        # Log
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        log_entry = {
            "epoch": epoch + 1,
            "loss": round(epoch_loss, 4),
            "lr": lr,
            "elapsed_min": round(elapsed / 60, 1),
        }

        # ─── Validation ───────────────────────────────────────────────────────
        if (epoch + 1) % config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    with autocast("cuda", enabled=config.use_amp):
                        val_outputs = sliding_window_inference(
                            val_inputs,
                            roi_size=config.patch_size,
                            sw_batch_size=config.sw_batch_size,
                            predictor=model,
                            overlap=config.sw_overlap,
                        )

                    val_outputs_list = decollate_batch(val_outputs)
                    val_labels_list = decollate_batch(val_labels)
                    val_outputs_onehot = [post_pred(i) for i in val_outputs_list]
                    val_labels_onehot = [post_label(i) for i in val_labels_list]

                    dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

            log_entry["val_dice"] = round(metric, 4)

            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                torch.save(model.state_dict(), config.best_model_path)
                log_entry["best"] = True

            print(
                f"  Epoch {epoch+1:3d}/{config.max_epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Val Dice: {metric:.4f} | "
                f"Best: {best_metric:.4f} (ep{best_epoch}) | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed/60:.0f}min"
            )
        else:
            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{config.max_epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed/60:.0f}min"
                )

        training_log.append(log_entry)

    # ─── Save Final ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Training complete!")
    print(f"  Best validation Dice: {best_metric:.4f} at epoch {best_epoch}")
    print(f"  Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Model saved: {config.best_model_path}")

    # Save training log
    with open(config.log_path, 'w') as f:
        json.dump({
            "config": {
                "max_epochs": config.max_epochs,
                "batch_size": config.batch_size,
                "patch_size": list(config.patch_size),
                "learning_rate": config.learning_rate,
                "init_filters": config.init_filters,
                "use_amp": config.use_amp,
            },
            "summary": summary,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "total_time_min": round((time.time() - start_time) / 60, 1),
            "log": training_log,
        }, f, indent=2)

    return best_metric, best_epoch


if __name__ == "__main__":
    train()
