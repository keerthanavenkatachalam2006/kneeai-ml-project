"""
Train 20 epochs on full dataset with validation every 5 epochs.
Expected time: ~50-60 minutes.
Expected Dice at epoch 20: 0.5-0.7 (model should start segmenting correctly).
"""
import sys
import time
import json
sys.path.insert(0, r'E:\KneeAI\backend')

import torch
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

from app.config import MODELS_DIR, PROCESSED_DATA, LABELS_MAP
from app.segmentation.model import create_segresnet, get_model_summary, NUM_CLASSES
from app.segmentation.dataset import (
    prepare_data_list,
    get_train_transforms,
    get_val_transforms,
    PATCH_SIZE,
)

if __name__ == "__main__":
    set_determinism(seed=42)
    device = torch.device("cuda")

    MAX_EPOCHS = 20
    VAL_INTERVAL = 5
    LR = 1e-3

    print("=" * 60)
    print(f"KneeAI - Training {MAX_EPOCHS} epochs (full dataset)")
    print("=" * 60)

    # Data
    train_files, val_files = prepare_data_list(PROCESSED_DATA / "extracted")
    val_subset = val_files[:10]  # 10 val cases for speed
    print(f"Train: {len(train_files)} | Val: {len(val_subset)}")

    train_ds = Dataset(data=train_files, transform=get_train_transforms())
    val_ds = Dataset(data=val_subset, transform=get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=True)

    # Model
    model = create_segresnet().to(device)
    print(f"Model: SegResNet ({get_model_summary(model)['total_params_M']}M params)")

    # Resume from latest checkpoint if exists
    start_epoch = 0
    for ep in range(MAX_EPOCHS, 0, -1):
        ckpt = MODELS_DIR / "segresnet_knee" / f"epoch_{ep:03d}.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
            start_epoch = ep
            print(f"Resumed from epoch {ep} checkpoint")
            break
    else:
        ckpt = MODELS_DIR / "segresnet_knee" / "epoch_001.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
            start_epoch = 1
            print(f"Resumed from epoch 1 checkpoint")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=True)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    best_dice = 0.0
    best_epoch = 0
    log = []
    t_start = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        # ─── Train ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        step = 0
        t_ep = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", unit="it")
        for batch_data in pbar:
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
        pbar.close()

        scheduler.step()
        avg_loss = epoch_loss / max(step, 1)
        ep_time = time.time() - t_ep
        lr = optimizer.param_groups[0]['lr']

        entry = {"epoch": epoch + 1, "loss": round(avg_loss, 4), "lr": round(lr, 6), "time_s": round(ep_time)}

        # ─── Validate ─────────────────────────────────────────────────────
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == MAX_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_in = val_data["image"].to(device)
                    val_lb = val_data["label"].to(device)
                    with autocast("cuda", enabled=True):
                        val_out = sliding_window_inference(
                            val_in, roi_size=PATCH_SIZE, sw_batch_size=2,
                            predictor=model, overlap=0.5,
                        )
                    pred_list = decollate_batch(val_out)
                    label_list = decollate_batch(val_lb)
                    dice_metric(
                        y_pred=[post_pred(x) for x in pred_list],
                        y=[post_label(x) for x in label_list],
                    )

            per_class = dice_metric.aggregate()
            dice_metric.reset()
            mean_dice = per_class.mean().item()

            entry["val_dice"] = round(mean_dice, 4)
            entry["per_class"] = {LABELS_MAP[i+1]: round(per_class[i].item(), 4) for i in range(5)}

            if mean_dice > best_dice:
                best_dice = mean_dice
                best_epoch = epoch + 1
                torch.save(model.state_dict(), MODELS_DIR / "segresnet_knee" / "best_model.pth")
                entry["best"] = True

            print(
                f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | Loss: {avg_loss:.4f} | "
                f"Dice: {mean_dice:.4f} | Best: {best_dice:.4f} (ep{best_epoch}) | "
                f"{ep_time:.0f}s"
            )
            for i in range(5):
                print(f"    {LABELS_MAP[i+1]:<25}: {per_class[i].item():.4f}")
        else:
            print(f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr:.5f} | {ep_time:.0f}s")

        log.append(entry)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), MODELS_DIR / "segresnet_knee" / f"epoch_{epoch+1:03d}.pth")

    # Save log
    total_time = time.time() - t_start
    log_data = {
        "epochs": MAX_EPOCHS,
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "total_time_min": round(total_time / 60, 1),
        "log": log,
    }
    log_path = MODELS_DIR / "segresnet_knee" / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: Best Dice = {best_dice:.4f} at epoch {best_epoch}")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Log: {log_path}")
    print(f"{'='*60}")
