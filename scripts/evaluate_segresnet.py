"""Measure the registered segmentation model on the held-out validation split.

Two outputs, both needed for honesty elsewhere:

1. ``artifacts/segresnet_validation.json`` — Dice per structure, in the
   ``ModelMetrics.internal_validation`` shape, so ``register_best_model.py`` can
   attach real measured numbers to the model card instead of leaving it blank or
   inheriting someone else's.
2. ``artifacts/val_predictions/<case_id>/segmentation.nii.gz`` — the model's own
   masks. ``train_oa_classifier.py --predicted-masks`` scores the OA classifier on
   these, which is the only number that describes the system as it actually runs.
   Evaluating the OA classifier on expert annotation measures a product nobody
   ships.

Inference goes through :class:`kneeai.ml.infer.SegmentationPredictor` and the
registered model card rather than a hand-rolled forward pass, so this doubles as an
integration test of the path the API uses. If the card and the checkpoint disagree,
this fails here rather than on a user's upload.

The split comes from the dataset manifest and is patient-level. No case scored here
was trained on — assuming the checkpoint was trained on OAIZIB-CM at all, which is
itself unrecorded and stated as an assumption on the model card.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from kneeai.core.anatomy import Structure
from kneeai.imaging.io import read_volume, write_volume
from kneeai.ml.registry import ModelCard

REPO = Path(__file__).resolve().parents[1]

STRUCTURES = (
    Structure.FEMUR,
    Structure.FEMORAL_CARTILAGE,
    Structure.TIBIA,
    Structure.MEDIAL_TIBIAL_CARTILAGE,
    Structure.LATERAL_TIBIAL_CARTILAGE,
)

LABEL_MAP = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
}


def dice(pred: np.ndarray, truth: np.ndarray, label: int) -> float:
    p = pred == label
    t = truth == label
    denom = int(p.sum()) + int(t.sum())
    if denom == 0:
        return float("nan")
    return 2.0 * float(np.logical_and(p, t).sum()) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/oaizib_60.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N cases")
    ap.add_argument("--out", default="artifacts/segresnet_validation.json")
    ap.add_argument("--predictions", default="artifacts/val_predictions")
    ap.add_argument(
        "--card",
        default="artifacts/models/seg_oaizib/model_card.json",
        help="the card the inference adapter reads",
    )
    args = ap.parse_args()

    from kneeai.ml.infer import SegmentationPredictor

    card = ModelCard.from_dict(json.loads((REPO / args.card).read_text()))
    print(f"model       : {card.key} ({card.architecture.value})")
    print(f"preprocessing: {card.preprocessing.get('name')} "
          f"spacing={card.preprocessing.get('target_spacing_mm')} "
          f"orientation={card.preprocessing.get('orientation')}")
    print(f"patch       : {card.patch_size}")

    records = json.loads((REPO / args.manifest).read_text()).get("records", []) or []
    cases = [r for r in records if r.get("split") == args.split]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit(f"no cases in split {args.split!r}")
    print(f"cases       : {len(cases)} in split '{args.split}'\n")

    predictor = SegmentationPredictor(card)
    predictions_root = REPO / args.predictions

    per_case: list[dict[str, Any]] = []
    for i, record in enumerate(cases, 1):
        case_id = str(record["case_id"])
        image_path = REPO / str(record["image_path"])
        label_path = REPO / str(record["label_path"])
        if not image_path.exists() or not label_path.exists():
            print(f"[{i}/{len(cases)}] {case_id}: image or label missing, skipped")
            continue

        t0 = time.perf_counter()
        image = read_volume(image_path)
        result = predictor.predict(image)
        elapsed = time.perf_counter() - t0

        truth = np.asarray(
            read_volume(label_path, is_label=True, label_map=LABEL_MAP).array
        )
        pred = np.asarray(result.mask.array)

        scores = {s.value: dice(pred, truth, LABEL_MAP[s]) for s in STRUCTURES}
        finite = [v for v in scores.values() if np.isfinite(v)]
        mean = float(np.mean(finite)) if finite else float("nan")

        # Keep the mask: the OA classifier needs to be scored on it.
        target = predictions_root / case_id / "segmentation.nii.gz"
        write_volume(result.mask, target)

        per_case.append(
            {
                "case_id": case_id,
                "patient_id": record.get("patient_id"),
                "kl_grade": record.get("kl_grade"),
                "dice": {k: round(v, 4) for k, v in scores.items()},
                "dice_mean": round(mean, 4),
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"[{i}/{len(cases)}] {case_id}: mean dice {mean:.4f} ({elapsed:.0f}s)  "
            + "  ".join(f"{k.split('_')[0][:4]}={v:.3f}" for k, v in scores.items())
        )

    if not per_case:
        raise SystemExit("nothing was scored")

    def aggregate(structure: str) -> float:
        values = [c["dice"][structure] for c in per_case if np.isfinite(c["dice"][structure])]
        return round(float(np.mean(values)), 4) if values else float("nan")

    per_structure = {s.value: aggregate(s.value) for s in STRUCTURES}
    overall = round(float(np.mean([v for v in per_structure.values() if np.isfinite(v)])), 4)

    payload = {
        "internal_validation": {
            "n_cases": len(per_case),
            "n_patients": len({c["patient_id"] for c in per_case}),
            "dice_mean_excl_background": overall,
            "dice_per_structure": per_structure,
            "metric_definition": (
                f"Dice on the acquisition grid, computed through "
                f"kneeai.ml.infer.SegmentationPredictor with the registered card "
                f"{card.key}: {card.preprocessing.get('target_spacing_mm')} mm "
                f"{card.preprocessing.get('orientation')}, sliding window "
                f"{card.patch_size} at 0.5 overlap. Mean over the patient-level "
                f"'{args.split}' split of {args.manifest}, excluding background."
            ),
            "model": card.key,
        },
        "per_case": per_case,
    }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n--- {card.key} on {len(per_case)} held-out cases ---")
    for structure, value in per_structure.items():
        print(f"  {structure:28s} {value:.4f}")
    print(f"  {'MEAN':28s} {overall:.4f}")
    print(f"\nwritten     : {out.relative_to(REPO)}")
    print(f"predictions : {predictions_root.relative_to(REPO)}/<case_id>/segmentation.nii.gz")
    print(
        "\nNext: re-run scripts/register_best_model.py --activate to attach these "
        "numbers to the model card, and scripts/train_oa_classifier.py "
        f"--predicted-masks {args.predictions} to measure the OA classifier honestly."
    )


if __name__ == "__main__":
    main()
