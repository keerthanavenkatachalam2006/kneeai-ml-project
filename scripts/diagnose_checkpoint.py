"""Diagnose *how* a checkpoint's prediction disagrees with ground truth.

A mean Dice number says a prediction is wrong. It does not say why, and the
plausible causes call for completely different fixes:

* **Permuted labels** — the model is right about the anatomy and wrong about the
  numbering. Shows up as mass off the diagonal of the confusion matrix, and is
  fixed by remapping rather than retraining.
* **Under-segmentation** — the model predicts background where anatomy is, which
  points at an intensity distribution it never saw in training.
* **A scoring artefact** — the prediction is fine on the grid it was made on and is
  destroyed by resampling it back to the acquisition grid. This one is a bug in the
  evaluation, not in the model, so the two grids are scored separately below.

Kept deliberately memory-lean: the acquisition grid here is 160x384x384 and an
earlier version of this script was killed for holding several float copies of it
at once. Everything works in ``uint8`` and counts with ``bincount``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

from kneeai.core.anatomy import Structure
from kneeai.imaging.io import read_volume
from kneeai.imaging.preprocess import IntensityNorm, PreprocessingSpec, preprocess, resample_like
from kneeai.imaging.volume import Volume

REPO = Path(__file__).resolve().parents[1]

NAMES = {
    0: "background",
    1: "femur",
    2: "femoral_cart",
    3: "tibia",
    4: "med_tib_cart",
    5: "lat_tib_cart",
}
LABEL_MAP = {
    "background": 0,
    "femur": 1,
    "femoral_cartilage": 2,
    "tibia": 3,
    "medial_tibial_cartilage": 4,
    "lateral_tibial_cartilage": 5,
}
N = 6


def dice_table(truth: np.ndarray, pred: np.ndarray, label: str) -> None:
    print(f"\n--- Dice on the {label} ---")
    scores = []
    for value in range(1, N):
        t = truth == value
        p = pred == value
        denom = int(t.sum()) + int(p.sum())
        d = 2.0 * float(np.logical_and(t, p).sum()) / denom if denom else float("nan")
        scores.append(d)
        print(f"  {NAMES[value]:>14s}: {d:.4f}")
    finite = [s for s in scores if np.isfinite(s)]
    print(f"  {'MEAN':>14s}: {np.mean(finite):.4f}" if finite else "  no overlap")


def confusion(truth: np.ndarray, pred: np.ndarray) -> None:
    """Row-normalised confusion, computed with a single bincount."""
    flat = (truth.astype(np.int64) * N + pred.astype(np.int64)).ravel()
    counts = np.bincount(flat, minlength=N * N).reshape(N, N)

    print("\nconfusion matrix (rows = truth, cols = prediction), % of each truth row")
    corner = "truth \\ pred"
    print(f"{corner:>16s}" + "".join(f"{NAMES[c]:>15s}" for c in range(N)))
    for t in range(N):
        total = int(counts[t].sum())
        cells = "".join(
            f"{100.0 * counts[t, p] / total:14.2f}%" if total else f"{'-':>15s}"
            for p in range(N)
        )
        print(f"{NAMES[t]:>16s}{cells}")

    # Best label permutation: if the model is right about anatomy and wrong about
    # numbering, some off-diagonal assignment beats the diagonal.
    print("\nfor each truth label, the prediction label it most overlaps:")
    for t in range(1, N):
        row = counts[t].copy()
        best = int(np.argmax(row[1:]) + 1) if row[1:].sum() else 0
        share = 100.0 * row[best] / max(int(row.sum()), 1)
        flag = "  <-- MISMATCH" if best != t and best != 0 else ""
        print(f"  truth {NAMES[t]:>14s} -> pred {NAMES[best]:>14s} ({share:.1f}%){flag}")


def _mean_dice(truth: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for value in range(1, N):
        t = truth == value
        p = pred == value
        denom = int(t.sum()) + int(p.sum())
        if denom:
            scores.append(2.0 * float(np.logical_and(t, p).sum()) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def _flip_search(truth: np.ndarray, pred: np.ndarray) -> None:
    """Score the prediction against every axis flip of the ground truth.

    The point is to separate "the model is bad" from "the model is good and the
    axes disagree". An orientation mismatch between training and inference
    mirrors the volume, which swaps medial for lateral and leaves a moderate Dice
    that looks like mediocre accuracy. If any flip scores dramatically higher than
    the identity, the fix is a preprocessing change, not a better model.

    Only shape-preserving transforms are tried, so the axis swap is included only
    when two axes have equal length.
    """
    print("\n--- axis flip / swap search (ground truth transformed, prediction fixed) ---")
    candidates: list[tuple[str, np.ndarray]] = []
    for swap in (False, True):
        if swap and truth.shape[1] != truth.shape[2]:
            continue
        base = np.swapaxes(truth, 1, 2) if swap else truth
        for flips in range(8):
            axes = [a for a in range(3) if flips & (1 << a)]
            transformed = np.flip(base, axis=axes) if axes else base
            label = ("swap12 " if swap else "       ") + (
                "flip" + "".join(str(a) for a in axes) if axes else "identity"
            )
            candidates.append((label, transformed))

    scored = [(label, _mean_dice(t, pred)) for label, t in candidates]
    scored.sort(key=lambda x: (-x[1] if np.isfinite(x[1]) else 0.0))
    for label, score in scored:
        marker = "  <== BEST" if label == scored[0][0] else ""
        print(f"  {label:24s} mean dice {score:.4f}{marker}")

    best_label, best_score = scored[0]
    identity = next(s for lbl, s in scored if lbl.strip() == "identity")
    if best_score > identity + 0.15:
        print(
            f"\n  CONCLUSION: '{best_label.strip()}' beats identity by "
            f"{best_score - identity:.3f} Dice. The preprocessing orientation is "
            "wrong, not the weights."
        )
    else:
        print(
            f"\n  CONCLUSION: no flip beats identity meaningfully "
            f"(best {best_score:.4f} vs identity {identity:.4f}). The axes are "
            "consistent; the accuracy ceiling is the model or its intensity "
            "preprocessing."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_model.pth")
    ap.add_argument("--case", default="417")
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--orientation", default="RAS")
    ap.add_argument("--foreground-zscore", action="store_true")
    args = ap.parse_args()

    image = read_volume(REPO / f"data/raw/oaizib/imagesTs/oaizib_{args.case}_0000.nii.gz")
    truth_vol = read_volume(
        REPO / f"data/raw/oaizib/labelsTs/oaizib_{args.case}.nii.gz",
        is_label=True,
        label_map={Structure(n): v for n, v in LABEL_MAP.items()},
    )

    state = torch.load(REPO / args.checkpoint, map_location="cpu", weights_only=False)
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=None,
        norm=("group", {"num_groups": 8}),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"weights: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    spec = PreprocessingSpec(
        name="diagnose",
        orientation=args.orientation,
        target_spacing_mm=(args.spacing,) * 3,
        intensity=IntensityNorm.ZSCORE,
        zscore_foreground_only=args.foreground_zscore,
        crop_to_roi=False,
    )
    working = preprocess(image, spec).volume
    print(f"working grid: {working.shape} at {np.round(working.spacing, 4)} mm")

    tensor = (
        torch.from_numpy(np.ascontiguousarray(working.array, dtype=np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    print(
        f"input intensity: mean={float(tensor.mean()):.4f} std={float(tensor.std()):.4f} "
        f"min={float(tensor.min()):.3f} max={float(tensor.max()):.3f}",
        flush=True,
    )

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=(args.patch,) * 3,
            sw_batch_size=1,
            predictor=model,
            overlap=args.overlap,
            mode="gaussian",
            padding_mode="constant",
        )
        labels = logits.argmax(dim=1)[0].to(torch.uint8).numpy()
    del logits, tensor
    print("inference done", flush=True)

    predicted = Volume.from_labels(
        labels, {Structure(n): v for n, v in LABEL_MAP.items()}, reference=working
    )

    # --- score on the working grid: ground truth brought TO the prediction ---
    truth_on_working = np.asarray(
        resample_like(truth_vol, working).array, dtype=np.uint8
    )
    dice_table(truth_on_working, labels, f"working grid ({args.spacing} mm)")
    confusion(truth_on_working, labels)
    _flip_search(truth_on_working, labels)

    print(f"\npredicted label histogram (working grid): "
          f"{np.bincount(labels.ravel(), minlength=N).tolist()}")
    print(f"truth     label histogram (working grid): "
          f"{np.bincount(truth_on_working.ravel(), minlength=N).tolist()}")
    del truth_on_working

    # --- score on the acquisition grid: prediction brought back ---
    pred_on_source = np.asarray(resample_like(predicted, image).array, dtype=np.uint8)
    truth_source = np.asarray(truth_vol.array, dtype=np.uint8)
    dice_table(truth_source, pred_on_source, "acquisition grid (as the pipeline scores it)")

    voxel = float(np.prod(np.asarray(image.spacing, dtype=np.float64)))
    print("\nper-structure volume, acquisition grid")
    print(f"{'structure':>16s}{'truth mm3':>13s}{'pred mm3':>13s}{'pred/truth':>12s}")
    for value in range(1, N):
        tv = float(np.count_nonzero(truth_source == value)) * voxel
        pv = float(np.count_nonzero(pred_on_source == value)) * voxel
        ratio = pv / tv if tv > 0 else float("nan")
        print(f"{NAMES[value]:>16s}{tv:13.0f}{pv:13.0f}{ratio:12.2f}")

    anatomy = truth_source > 0
    missed = float(np.count_nonzero(anatomy & (pred_on_source == 0))) / max(
        int(anatomy.sum()), 1
    )
    print(f"\nanatomy predicted as background: {100 * missed:.1f}%")


if __name__ == "__main__":
    main()
