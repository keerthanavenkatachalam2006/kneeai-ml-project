"""
Launch script for MONAI SegResNet training on OAIZIB-CM.

Usage:
    python scripts/train_segmentation.py

This will:
1. Extract all data from ZIPs (first run only, ~12GB disk space)
2. Train SegResNet for 150 epochs with patch-based training
3. Save the best model to models/segresnet_knee/best_model.pth
4. Log training metrics to models/segresnet_knee/training_log.json

Hardware: RTX 4050 (6GB VRAM)
Expected time: ~2-4 hours
"""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

from app.segmentation.train import train, TrainConfig

if __name__ == "__main__":
    config = TrainConfig()
    print("=" * 60)
    print("KneeAI - MONAI SegResNet Training")
    print("=" * 60)
    print(f"  Model: SegResNet (init_filters={config.init_filters})")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Epochs: {config.max_epochs}")
    print(f"  Batch size: {config.batch_size} (effective: {config.batch_size * 2 * config.grad_accum_steps})")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision: {config.use_amp}")
    print(f"  Output: {config.output_dir}")
    print("=" * 60)

    best_dice, best_epoch = train(config)

    print(f"\nFinal Result: Best Dice = {best_dice:.4f} at epoch {best_epoch}")
