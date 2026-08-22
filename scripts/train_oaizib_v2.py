"""OAIZIB-CM segmentation, run 2 — targeted at the cartilage failure of run 1.

Run 1 reached femur 0.913 / tibia 0.890 but only 0.455 / 0.423 on the tibial
cartilages. Three changes address that specifically, and nothing else changes, so
the comparison stays honest.

**1. Class-balanced cropping.** Run 1 used ``RandCropByPosNegLabeld``, which treats
every non-background voxel as equally worth sampling. Femur is 1.86 M voxels and
medial tibial cartilage is 19.5 k, so roughly 1 crop in 100 was centred on the
structure that was failing. ``RandCropByLabelClassesd`` with explicit per-class
ratios forces the sampler to centre crops on cartilage far more often.

**2. Cross-entropy class weights.** Dice already normalises per structure, but the
CE half of DiceCE does not, so it was dominated by bone. Weights are computed from
the actual voxel frequencies in the training split rather than guessed.

**3. 48 training patients instead of 30**, from the 60 cases already on disk. The
disk is full, so this is all the data available — ``imagesTr.zip`` is 9.3 GB and
will not fit.

Deliberately unchanged from run 1: architecture, 1.0 mm resampling, patch size,
optimiser, augmentation, seed, and the metric definition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from monai.utils import set_determinism

REPO = Path(__file__).resolve().parents[1]

STRUCTURES = [
    "femur",
    "femoral_cartilage",
    "tibia",
    "medial_tibial_cartilage",
    "lateral_tibial_cartilage",
]
N_CLASSES = len(STRUCTURES) + 1

# Sampling ratios per class: background, femur, fem cart, tibia, med tib, lat tib.
# The cartilages get the largest share because they are the structures that failed.
CROP_RATIOS = [1, 2, 4, 2, 5, 5]


def load_files(tag: str) -> tuple[list[dict], list[dict], dict]:
    manifest = json.loads((REPO / f"data/manifests/{tag}.json").read_text())
    records = {r["case_id"]: r for r in manifest["records"]}

    def pick(split: str) -> list[dict]:
        s = json.loads((REPO / f"data/splits/{tag}_{split}.json").read_text())
        return [
            {
                "image": str(REPO / records[c]["image_path"]),
                "label": str(REPO / records[c]["label_path"]),
                "case_id": c,
                "patient_id": records[c]["patient_id"],
            }
            for c in s["case_ids"]
        ]

    return pick("train"), pick("val"), manifest


def class_weights(files: list[dict], sample: int = 12) -> torch.Tensor:
    """Inverse-frequency CE weights, measured from the training labels.

    Capped so the rarest class cannot dominate the loss outright — an uncapped
    inverse frequency on a 100:1 imbalance produces gradients that drown the bone
    classes and destabilise early training.
    """
    import nibabel as nib

    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for f in files[:sample]:
        a = np.asarray(nib.load(f["label"]).dataobj)
        for c in range(N_CLASSES):
            counts[c] += float((a == c).sum())
    freq = counts / counts.sum()
    w = 1.0 / np.maximum(freq, 1e-8)
    w = w / w[1]  # normalise so femur == 1
    w = np.clip(w, 0.2, 25.0)
    w[0] = 0.2  # background barely matters, Dice excludes it anyway
    print("  voxel fractions :", np.round(freq, 5).tolist())
    print("  CE class weights:", np.round(w, 2).tolist())
    return torch.tensor(w, dtype=torch.float32)


def build_transforms(spacing: float, patch: int, n_samples: int):
    load = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(spacing,) * 3,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys="image", nonzero=True),
    ]
    train = Compose(
        [
            *load,
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(patch,) * 3,
                ratios=CROP_RATIOS,
                num_classes=N_CLASSES,
                num_samples=n_samples,
                allow_smaller=True,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandGaussianNoised(keys="image", prob=0.15, std=0.03),
            RandShiftIntensityd(keys="image", prob=0.20, offsets=0.10),
            RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.8, 1.3)),
        ]
    )
    return train, Compose(load)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="oaizib_60")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--out", default="artifacts/models/seg_oaizib_v2")
    args = ap.parse_args()

    set_determinism(seed=args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 78)
    print("OAIZIB-CM segmentation — RUN 2 (cartilage-targeted)")
    print("=" * 78)
    print(f"torch {torch.__version__}  monai {__import__('monai').__version__}  device {device}")

    train_files, val_files = load_files(args.tag)[:2]
    tp = {f["patient_id"] for f in train_files}
    vp = {f["patient_id"] for f in val_files}
    print(f"train {len(train_files)} cases / {len(tp)} patients")
    print(f"val   {len(val_files)} cases / {len(vp)} patients")
    print(f"leakage: {'FAIL ' + str(tp & vp) if tp & vp else 'PASS (0 shared patients)'}")
    print(f"changes vs run 1: 48 train (was 30), class-balanced crops, CE weights")
    print("-" * 78)

    print("measuring class frequencies from training labels ...")
    weights = class_weights(train_files).to(device)

    train_tf, val_tf = build_transforms(args.spacing, args.patch, args.samples)
    t0 = time.time()
    print("caching ...", flush=True)
    train_ds = CacheDataset(train_files, train_tf, cache_rate=1.0, num_workers=2, progress=False)
    val_ds = CacheDataset(val_files, val_tf, cache_rate=1.0, num_workers=2, progress=False)
    print(f"cached in {time.time()-t0:.0f}s")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=N_CLASSES,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
    ).to(device)
    print(f"UNet3D {sum(p.numel() for p in model.parameters())/1e6:.2f} M params")
    print("-" * 78)

    loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True, include_background=False, ce_weight=weights
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=N_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=N_CLASSES)])

    history: list[dict] = []
    best = -1.0
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = REPO / "artifacts/metrics/oaizib_v2_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep0, run, n = time.time(), 0.0, 0
        for batch in train_dl:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run += float(loss.item())
            n += 1
            print(f"  ep{epoch:02d} step {n:03d}/{len(train_dl)} loss {loss.item():.4f}", end="\r", flush=True)
        sched.step()
        row = {"epoch": epoch, "train_loss": run / max(n, 1), "seconds": time.time() - ep0}
        line = f"epoch {epoch:02d}/{args.epochs}  train_loss {row['train_loss']:.4f}  lr {opt.param_groups[0]['lr']:.2e}  {row['seconds']:.0f}s"

        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            metric.reset()
            with torch.no_grad():
                for batch in val_dl:
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    logits = sliding_window_inference(
                        x, (args.patch,) * 3, 1, model, overlap=0.25
                    )
                    metric(
                        y_pred=[post_pred(p) for p in decollate_batch(logits)],
                        y=[post_label(l) for l in decollate_batch(y)],
                    )
            per = metric.aggregate().cpu().numpy()
            mean = float(np.nanmean(per))
            row["val_dice_mean"] = mean
            row["val_dice_per_structure"] = {s: float(d) for s, d in zip(STRUCTURES, per)}
            line += f"  |  val_dice {mean:.4f}  [" + "  ".join(
                f"{s.split('_')[0][:4]}={d:.3f}" for s, d in zip(STRUCTURES, per)
            ) + "]"
            if mean > best:
                best = mean
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_dice_mean": mean,
                        "val_dice_per_structure": row["val_dice_per_structure"],
                        "structures": STRUCTURES,
                        "n_classes": N_CLASSES,
                        "spacing_mm": [args.spacing] * 3,
                        "patch": args.patch,
                        "run": "v2",
                    },
                    out_dir / "checkpoint.pt",
                )
                line += "  <- saved"

        print(" " * 110, end="\r")
        print(line, flush=True)
        history.append(row)
        hist_path.write_text(json.dumps(history, indent=2))

    print("-" * 78)
    print(f"best val mean Dice: {best:.4f}")
    print(f"checkpoint: {out_dir/'checkpoint.pt'}")


if __name__ == "__main__":
    main()
