"""Local fine-tuning run for OAIZIB-CM bone + cartilage segmentation.

Runs on Apple Silicon MPS. This is the *small-scale* variant of the Colab
recipe: the model, patch size and epoch count are sized for an 8 GB M1, not for
a 16 GB CUDA GPU. Everything else — data, splits, label map, loss, metrics — is
the real pipeline.

Deliberate choices, and why:

**Resample to 1.0 mm isotropic.** Native spacing is 0.7 x 0.365 x 0.365 mm, so a
volume is 160 x 384 x 384 = 23.6 M voxels. Thirty of those plus labels will not
fit in 8 GB of unified memory. At 1.0 mm the same anatomy is ~112 x 140 x 140,
which caches comfortably and keeps the physical field of view identical. The
target spacing is measured from the data, not guessed.

**Left/right mirroring is on.** Every OAIZIB-CM knee is a right knee. Without
mirroring the model would never see a left knee in any form. Medial and lateral
are anatomical rather than spatial designations, so mirroring across the
sagittal plane maps a right knee onto a correctly labelled left knee.

**Cartilage is expected to score far below bone.** Femoral cartilage is ~148 k
voxels against ~1.86 M for femur, and the tibial cartilages are ~20 k each. At
1.0 mm with 30 training cases the thin structures are genuinely hard. The
numbers this prints are the numbers; they are not tuned to look good.
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
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from monai.utils import set_determinism

REPO = Path(__file__).resolve().parents[1]

# OAIZIB-CM label map. Identical to core.anatomy.DEFAULT_LABEL_MAP for these
# five structures, which is asserted in datasets/oaizib.py.
STRUCTURES = [
    "femur",
    "femoral_cartilage",
    "tibia",
    "medial_tibial_cartilage",
    "lateral_tibial_cartilage",
]
N_CLASSES = len(STRUCTURES) + 1  # + background


def load_split(manifest_path: Path, split_dir: Path, tag: str) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text())
    records = {r["case_id"]: r for r in manifest["records"]}

    def pick(split_name: str) -> list[dict]:
        split = json.loads((split_dir / f"{tag}_{split_name}.json").read_text())
        out = []
        for case_id in split["case_ids"]:
            rec = records[case_id]
            out.append(
                {
                    "image": str(REPO / rec["image_path"]),
                    "label": str(REPO / rec["label_path"]),
                    "case_id": case_id,
                    "patient_id": rec["patient_id"],
                }
            )
        return out

    return pick("train"), {"val": pick("val"), "manifest": manifest}


def build_transforms(spacing: float, patch: int, n_samples: int):
    load = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(spacing, spacing, spacing),
            mode=("bilinear", "nearest"),
        ),
        # Percentile clipping before z-score: MR intensities have no absolute
        # scale, and a handful of bright voxels otherwise dominate the mean.
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys="image", nonzero=True),
    ]

    train = Compose(
        [
            *load,
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(patch, patch, patch),
                pos=3,
                neg=1,
                num_samples=n_samples,
                image_key="image",
                allow_smaller=True,
            ),
            # Sagittal mirror -> turns a right knee into a valid left knee.
            # After Orientationd(RAS) axis 0 is left-right.
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandGaussianNoised(keys="image", prob=0.15, std=0.03),
            RandShiftIntensityd(keys="image", prob=0.20, offsets=0.10),
            RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.8, 1.3)),
        ]
    )
    val = Compose(load)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--out", type=str, default="artifacts/models/seg_oaizib")
    args = ap.parse_args()

    set_determinism(seed=args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 74)
    print("OAIZIB-CM bone + cartilage segmentation — local fine-tune")
    print("=" * 74)
    print(f"torch {torch.__version__}   monai {__import__('monai').__version__}")
    print(f"device {device}   (MPS available: {torch.backends.mps.is_available()})")

    train_files, extra = load_split(
        REPO / "data/manifests/oaizib_40.json", REPO / "data/splits", "oaizib_40"
    )
    val_files = extra["val"]
    if args.limit_train:
        train_files = train_files[: args.limit_train]

    train_pat = {f["patient_id"] for f in train_files}
    val_pat = {f["patient_id"] for f in val_files}
    overlap = train_pat & val_pat
    print(f"train {len(train_files)} cases / {len(train_pat)} patients")
    print(f"val   {len(val_files)} cases / {len(val_pat)} patients")
    print(f"patient-level leakage check: {'FAIL ' + str(overlap) if overlap else 'PASS (0 shared patients)'}")
    print(f"resample {args.spacing} mm iso   patch {args.patch}^3   batch {args.batch}")
    print(f"structures: {', '.join(STRUCTURES)}")
    print("no meniscus / patella labels exist in OAIZIB-CM")
    print("-" * 74)

    train_tf, val_tf = build_transforms(args.spacing, args.patch, args.samples)

    t0 = time.time()
    print("caching volumes (resample happens once, here) ...", flush=True)
    train_ds = CacheDataset(train_files, transform=train_tf, cache_rate=1.0, num_workers=2)
    val_ds = CacheDataset(val_files, transform=val_tf, cache_rate=1.0, num_workers=2)
    print(f"cached in {time.time() - t0:.0f}s")

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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model UNet3D  {n_params/1e6:.2f} M params  out_channels={N_CLASSES}")
    print("-" * 74)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=N_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=N_CLASSES)])

    history: list[dict] = []
    best = -1.0
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep0 = time.time()
        running, nstep = 0.0, 0

        for batch in train_dl:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item())
            nstep += 1
            print(
                f"  epoch {epoch:02d}  step {nstep:03d}/{len(train_dl)}  loss {loss.item():.4f}",
                end="\r",
                flush=True,
            )

        sched.step()
        train_loss = running / max(nstep, 1)
        line = f"epoch {epoch:02d}/{args.epochs}  train_loss {train_loss:.4f}  lr {opt.param_groups[0]['lr']:.2e}  {time.time()-ep0:.0f}s"

        row = {"epoch": epoch, "train_loss": train_loss, "seconds": time.time() - ep0}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in val_dl:
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)
                    logits = sliding_window_inference(
                        x,
                        roi_size=(args.patch, args.patch, args.patch),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.25,
                    )
                    preds = [post_pred(p) for p in decollate_batch(logits)]
                    labels = [post_label(l) for l in decollate_batch(y)]
                    dice_metric(y_pred=preds, y=labels)
            per_struct = dice_metric.aggregate().cpu().numpy()
            mean_dice = float(np.nanmean(per_struct))
            row["val_dice_mean"] = mean_dice
            row["val_dice_per_structure"] = {
                s: float(d) for s, d in zip(STRUCTURES, per_struct)
            }
            line += f"  |  val_dice {mean_dice:.4f}"
            line += "  [" + "  ".join(
                f"{s.split('_')[0][:4]}={d:.3f}" for s, d in zip(STRUCTURES, per_struct)
            ) + "]"

            if mean_dice > best:
                best = mean_dice
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_dice_mean": mean_dice,
                        "val_dice_per_structure": row["val_dice_per_structure"],
                        "structures": STRUCTURES,
                        "n_classes": N_CLASSES,
                        "spacing_mm": [args.spacing] * 3,
                        "patch": args.patch,
                    },
                    out_dir / "checkpoint.pt",
                )
                line += "  <- saved"

        print(" " * 100, end="\r")
        print(line, flush=True)
        history.append(row)
        (REPO / "artifacts/metrics").mkdir(parents=True, exist_ok=True)
        (REPO / "artifacts/metrics/oaizib_history.json").write_text(
            json.dumps(history, indent=2)
        )

    print("-" * 74)
    print(f"best val mean Dice (excl. background): {best:.4f}")
    print(f"checkpoint: {out_dir/'checkpoint.pt'}")
    print(f"history:    artifacts/metrics/oaizib_history.json")


if __name__ == "__main__":
    main()
