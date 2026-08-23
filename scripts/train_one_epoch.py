"""
Run exactly 1 training epoch + 1 validation pass.
Reports loss, Dice scores per class, time, and memory usage.
"""
import sys
import time
sys.path.insert(0, r'E:\KneeAI\backend')

import torch
import numpy as np
from torch.amp import GradScaler, autocast
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

from app.config import MODELS_DIR, PROCESSED_DATA, LABELS_MAP
from app.segmentation.model import create_segresnet, get_model_summary, NUM_CLASSES
from app.segmentation.dataset import (
    prepare_data_list,
    get_train_transforms,
    get_val_transforms,
    PATCH_SIZE,
)

set_determinism(seed=42)
device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()

print("=" * 60)
print("KneeAI - Single Epoch Training + Validation")
print("=" * 60)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Patch size: {PATCH_SIZE}")

# Data
print("\nLoading data lists...")
train_files, val_files = prepare_data_list(PROCESSED_DATA / "extracted")
print(f"  Train: {len(train_files)}, Val: {len(val_files)}")

# Use subset for speed: 50 train, 5 val
train_subset = train_files[:50]
val_subset = val_files[:5]
print(f"  Using subset: {len(train_subset)} train, {len(val_subset)} val")

train_ds = Dataset(data=train_subset, transform=get_train_transforms())
val_ds = Dataset(data=val_subset, transform=get_val_transforms())

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=True)

# Model
model = create_segresnet().to(device)
summary = get_model_summary(model)
print(f"  Model: {summary['architecture']} ({summary['total_params_M']}M params)")

# Loss & optimizer
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scaler = GradScaler("cuda", enabled=True)

# ─── Train 1 epoch ───────────────────────────────────────────────────────────
print("\n--- Training (1 epoch, 50 cases) ---")
model.train()
epoch_loss = 0.0
step = 0
t0 = time.time()

for batch_data in train_loader:
    inputs = batch_data["image"].to(device)
    labels = batch_data["label"].to(device)

    optimizer.zero_grad()
    with autocast("cuda", enabled=True):
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    epoch_loss += loss.item()
    step += 1
    if step % 10 == 0:
        print(f"  Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

train_time = time.time() - t0
avg_loss = epoch_loss / max(step, 1)
print(f"\n  Epoch complete: {step} steps in {train_time:.0f}s ({train_time/step:.1f}s/step)")
print(f"  Average loss: {avg_loss:.4f}")
print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# ─── Validate ────────────────────────────────────────────────────────────────
print("\n--- Validation (5 cases, sliding window) ---")
model.eval()
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)

t1 = time.time()
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        val_inputs = val_data["image"].to(device)
        val_labels = val_data["label"].to(device)

        with autocast("cuda", enabled=True):
            val_outputs = sliding_window_inference(
                val_inputs,
                roi_size=PATCH_SIZE,
                sw_batch_size=2,
                predictor=model,
                overlap=0.5,
            )

        val_outputs_list = decollate_batch(val_outputs)
        val_labels_list = decollate_batch(val_labels)
        val_preds = [post_pred(x) for x in val_outputs_list]
        val_gts = [post_label(x) for x in val_labels_list]
        dice_metric(y_pred=val_preds, y=val_gts)
        print(f"  Validated case {i+1}/5")

val_time = time.time() - t1

# Per-class dice
per_class_dice = dice_metric.aggregate()
dice_metric.reset()

print(f"\n  Validation complete in {val_time:.0f}s")
print(f"\n--- Results ---")
print(f"  {'Structure':<25} {'Dice':>8}")
print(f"  {'-'*35}")
class_names = [LABELS_MAP[i] for i in range(1, NUM_CLASSES)]
for i, name in enumerate(class_names):
    d = per_class_dice[i].item()
    print(f"  {name:<25} {d:>8.4f}")

mean_dice = per_class_dice.mean().item()
print(f"  {'-'*35}")
print(f"  {'MEAN':<25} {mean_dice:>8.4f}")

print(f"\n--- Summary ---")
print(f"  Training loss: {avg_loss:.4f}")
print(f"  Mean Dice (5 structures): {mean_dice:.4f}")
print(f"  Train time: {train_time:.0f}s | Val time: {val_time:.0f}s")
print(f"  Peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# Save model checkpoint
ckpt_path = MODELS_DIR / "segresnet_knee" / "epoch_001.pth"
ckpt_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), ckpt_path)
print(f"  Checkpoint: {ckpt_path}")
print("=" * 60)
