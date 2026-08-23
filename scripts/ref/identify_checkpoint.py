"""Identify the architecture and preprocessing behind an unlabelled checkpoint.

``best_model.pth`` arrived as a bare state dict with no metadata: no label map, no
``model_config``, no preprocessing spec. Its tensor shapes narrow the architecture
down but do not determine it — GroupNorm and affine InstanceNorm produce
identically shaped affine parameters, and nothing in the file records the target
spacing it was trained at.

Guessing either wrong is silent: the weights load, inference runs, and the mask is
degraded rather than absent. So this grid-searches the ambiguous axes and scores
each candidate by Dice against the OAIZIB-CM ground truth the model was trained
on. The winning combination is the one written into the registered checkpoint.

Usage::

    python scripts/identify_checkpoint.py --cases 417 418 --spacings 1.0 0.5
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

from kneeai.core.anatomy import Structure
from kneeai.imaging.io import read_volume
from kneeai.imaging.preprocess import (
    IntensityNorm,
    PreprocessingSpec,
    preprocess,
    resample_like,
)
from kneeai.imaging.volume import Volume

REPO = Path(__file__).resolve().parents[1]

STRUCTURES = {
    1: "femur",
    2: "femoral_cartilage",
    3: "tibia",
    4: "medial_tibial_cartilage",
    5: "lateral_tibial_cartilage",
}

LABEL_MAP = {"background": 0, **{v: k for k, v in STRUCTURES.items()}}

#: The two norm configurations that fit the checkpoint's tensor shapes exactly.
#: MONAI's SegResNet default is GroupNorm(8); an explicit affine InstanceNorm has
#: the same parameter shapes, so shape inspection alone cannot separate them.
NORM_CANDIDATES: dict[str, Any] = {
    "group8": ("group", {"num_groups": 8}),
    "instance_affine": ("instance", {"affine": True}),
}

#: Intensity normalisation variants, as (mode, foreground_only). MR intensity has
#: no absolute meaning, so which of these the model saw during training changes the
#: input distribution and therefore the output.
INTENSITY_CANDIDATES: dict[str, tuple[IntensityNorm, bool]] = {
    "zscore_fg": (IntensityNorm.ZSCORE, True),
    "zscore_all": (IntensityNorm.ZSCORE, False),
    "percentile": (IntensityNorm.PERCENTILE, True),
    "none": (IntensityNorm.NONE, True),
}


def dice(pred: np.ndarray, truth: np.ndarray, label: int) -> float:
    p = pred == label
    t = truth == label
    denom = int(p.sum()) + int(t.sum())
    if denom == 0:
        return float("nan")
    return 2.0 * float(np.logical_and(p, t).sum()) / float(denom)


def build(norm: Any) -> torch.nn.Module:
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=None,
        norm=norm,
    )


def run_case(
    model: torch.nn.Module,
    image: Volume,
    truth: np.ndarray,
    spec: PreprocessingSpec,
    patch: int,
    overlap: float,
) -> dict[str, float]:
    pre = preprocess(image, spec)
    working = pre.volume
    tensor = (
        torch.from_numpy(np.ascontiguousarray(working.array, dtype=np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=(patch, patch, patch),
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
            padding_mode="constant",
        )
        labels = logits.softmax(dim=1)[0].argmax(dim=0).numpy().astype(np.uint8)

    # Score on the acquisition grid, which is where measurements are taken.
    predicted = Volume.from_labels(
        labels,
        {Structure(name): value for name, value in LABEL_MAP.items()},
        reference=working,
    )
    on_source = resample_like(predicted, image)
    arr = np.asarray(on_source.array)

    scores = {name: dice(arr, truth, label) for label, name in STRUCTURES.items()}
    finite = [v for v in scores.values() if np.isfinite(v)]
    scores["MEAN"] = float(np.mean(finite)) if finite else float("nan")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_model.pth")
    ap.add_argument("--cases", nargs="+", default=["417"])
    ap.add_argument("--spacings", nargs="+", type=float, default=[1.0, 0.5])
    ap.add_argument("--orientations", nargs="+", default=["RAS"])
    ap.add_argument("--patches", nargs="+", type=int, default=[64])
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--norms", nargs="+", default=list(NORM_CANDIDATES))
    ap.add_argument(
        "--intensities",
        nargs="+",
        default=["zscore_fg"],
        choices=["zscore_fg", "zscore_all", "percentile", "none"],
        help="intensity normalisation variants to try",
    )
    ap.add_argument("--out", default="artifacts/checkpoint_identification.json")
    args = ap.parse_args()

    state = torch.load(REPO / args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    cases = []
    for case in args.cases:
        image_path = REPO / f"data/raw/oaizib/imagesTs/oaizib_{case}_0000.nii.gz"
        label_path = REPO / f"data/raw/oaizib/labelsTs/oaizib_{case}.nii.gz"
        if not image_path.exists() or not label_path.exists():
            raise SystemExit(f"case {case}: missing image or label")
        image = read_volume(image_path)
        truth = np.asarray(read_volume(label_path, is_label=True, label_map=None).array)
        cases.append((case, image, truth))

    results: list[dict[str, Any]] = []
    combos = list(
        itertools.product(
            args.norms, args.orientations, args.spacings, args.patches, args.intensities
        )
    )
    print(f"{len(combos)} candidate(s) x {len(cases)} case(s)\n", flush=True)

    for norm_name, orientation, spacing, patch, intensity in combos:
        model = build(NORM_CANDIDATES[norm_name])
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"{norm_name:16s} SKIP  missing={len(missing)} unexpected={len(unexpected)}")
            continue
        model.eval()

        spec = PreprocessingSpec(
            name=f"probe_{orientation}_{spacing}_{intensity}",
            orientation=orientation,
            target_spacing_mm=(spacing, spacing, spacing),
            intensity=INTENSITY_CANDIDATES[intensity][0],
            zscore_foreground_only=INTENSITY_CANDIDATES[intensity][1],
            crop_to_roi=False,
        )

        per_case = []
        t0 = time.perf_counter()
        for case, image, truth in cases:
            per_case.append(run_case(model, image, truth, spec, patch, args.overlap))
        elapsed = time.perf_counter() - t0

        mean = float(np.mean([c["MEAN"] for c in per_case]))
        record = {
            "norm": norm_name,
            "orientation": orientation,
            "spacing_mm": spacing,
            "patch": patch,
            "intensity": intensity,
            "dice_mean": round(mean, 4),
            "per_structure": {
                k: round(float(np.mean([c[k] for c in per_case])), 4)
                for k in STRUCTURES.values()
            },
            "seconds": round(elapsed, 1),
        }
        results.append(record)
        print(
            f"{norm_name:16s} {orientation} {spacing:>6}mm p{patch:<4d} "
            f"{intensity:11s} dice={mean:.4f}  ({elapsed:.0f}s)  "
            f"{record['per_structure']}",
            flush=True,
        )

    results.sort(key=lambda r: r["dice_mean"], reverse=True)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": args.cases, "results": results}, indent=2) + "\n")

    print(f"\nwritten: {out.relative_to(REPO)}")
    if results:
        best = results[0]
        print(
            f"BEST: norm={best['norm']} orientation={best['orientation']} "
            f"spacing={best['spacing_mm']}mm  dice={best['dice_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
