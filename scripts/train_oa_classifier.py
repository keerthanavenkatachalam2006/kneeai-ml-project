"""Fit the osteoarthritis classifier against the cohort's Kellgren-Lawrence grades.

The labels already exist — OAIZIB-CM's ``info`` archive carries a KL grade per knee,
and ``data/manifests/*.json`` records it. Nothing new is annotated here. What this
script does is extract cartilage geometry from each knee's segmentation and fit a
regularised logistic regression to predict KL >= 2, the conventional threshold for
definite radiographic osteoarthritis.

Three things it does deliberately, because each is a way this could have been done
dishonestly:

**It uses the manifest's own patient-level split.** Train and validation share no
patient. A random split over knees would leak, since a patient's two knees are
correlated, and would inflate every number below.

**It evaluates on the model's own masks as well as on ground truth.** The classifier
consumes a segmentation, and in production that segmentation comes from a network
whose tibial cartilage Dice is poor. Scoring only against expert annotation measures
a system nobody will ever run. ``--predicted-masks`` points at a directory of
predicted masks and produces the second, honest number.

**It reports the failure.** If validation performance is no better than the majority
class, that is what gets written to the model card, and ``assess.py`` will carry it
into every prediction the interface renders.

Usage::

    python scripts/train_oa_classifier.py
    python scripts/train_oa_classifier.py --predicted-masks artifacts/val_predictions
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from kneeai.core.anatomy import Structure
from kneeai.imaging.io import read_volume
from kneeai.oa.classifier import (
    OA_KL_THRESHOLD,
    MODEL_FILENAME,
    OaClassifier,
    evaluate_binary,
    fit_logistic,
)
from kneeai.oa.features import FEATURE_NAMES, extract_features

REPO = Path(__file__).resolve().parents[1]

LABEL_MAP = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
}


def load_records(manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text())
    records = payload.get("records", []) or []
    usable = [r for r in records if r.get("kl_grade") is not None and r.get("label_path")]
    skipped = len(records) - len(usable)
    print(f"manifest {manifest.name}: {len(records)} records, {len(usable)} with a KL grade")
    if skipped:
        print(f"  {skipped} skipped for a missing KL grade — they cannot be trained on")
    return usable


def features_for(path: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        mask = read_volume(path, is_label=True, label_map=LABEL_MAP)
    except Exception as exc:  # noqa: BLE001
        print(f"    could not read {path.name}: {exc}")
        return None
    try:
        result = extract_features(mask)
    except Exception as exc:  # noqa: BLE001
        print(f"    feature extraction failed for {path.name}: {exc}")
        return None
    return result.vector(), result.to_dict()


def extract_all(
    records: list[dict[str, Any]], *, root: Path, predicted: Path | None
) -> dict[str, dict[str, Any]]:
    """Feature vectors per case, from ground truth and optionally predicted masks."""
    out: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(records, 1):
        case_id = str(record["case_id"])
        label_path = root / str(record["label_path"])
        if not label_path.exists():
            print(f"  [{i}/{len(records)}] {case_id}: label missing, skipped")
            continue

        t0 = time.perf_counter()
        truth = features_for(label_path)
        if truth is None:
            continue
        entry: dict[str, Any] = {
            "case_id": case_id,
            "patient_id": record.get("patient_id"),
            "kl_grade": int(record["kl_grade"]),
            "split": record.get("split", "train"),
            "truth_vector": truth[0].tolist(),
            "truth_detail": truth[1],
        }

        if predicted is not None:
            candidate = predicted / case_id / "segmentation.nii.gz"
            if not candidate.exists():
                candidate = predicted / f"{case_id}.nii.gz"
            if candidate.exists():
                pred = features_for(candidate)
                if pred is not None:
                    entry["predicted_vector"] = pred[0].tolist()

        out[case_id] = entry
        print(
            f"  [{i}/{len(records)}] {case_id}: KL {entry['kl_grade']} "
            f"({entry['split']}) {time.perf_counter() - t0:.1f}s"
            + ("  +predicted" if "predicted_vector" in entry else "")
        )
    return out


def fit_grade_regressor(design: np.ndarray, grades: np.ndarray, l2: float) -> np.ndarray:
    """Ridge least squares on the already-standardised design matrix."""
    n, d = design.shape
    gram = design.T @ design + l2 * np.eye(d)
    return np.linalg.solve(gram, design.T @ (grades - grades.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/oaizib_60.json")
    ap.add_argument("--predicted-masks", default=None)
    ap.add_argument("--l2", type=float, default=2.0)
    ap.add_argument("--iterations", type=int, default=40_000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cache", default="artifacts/oa_features.json")
    ap.add_argument("--refresh", action="store_true", help="ignore the feature cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = load_records(REPO / args.manifest)
    if not records:
        raise SystemExit("no usable records; cannot fit a classifier")

    predicted_dir = REPO / args.predicted_masks if args.predicted_masks else None
    cache_path = REPO / args.cache

    if cache_path.exists() and not args.refresh:
        cached = json.loads(cache_path.read_text())
        data = cached["cases"]
        print(f"loaded cached features for {len(data)} cases from {cache_path.name}")
        # Top up with predicted-mask features if they were not cached.
        if predicted_dir is not None and not any("predicted_vector" in v for v in data.values()):
            print("cache has no predicted-mask features; extracting those now")
            for case_id, entry in data.items():
                candidate = predicted_dir / case_id / "segmentation.nii.gz"
                if candidate.exists():
                    pred = features_for(candidate)
                    if pred is not None:
                        entry["predicted_vector"] = pred[0].tolist()
                        print(f"  {case_id}: +predicted")
    else:
        print(f"\nextracting features from {len(records)} segmentations")
        data = extract_all(records, root=REPO, predicted=predicted_dir)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"feature_names": list(FEATURE_NAMES), "cases": data}, indent=2) + "\n"
    )
    print(f"\nfeature cache: {cache_path.relative_to(REPO)}")

    entries = list(data.values())
    train = [e for e in entries if e["split"] == "train"]
    val = [e for e in entries if e["split"] == "val"]
    if not train or not val:
        raise SystemExit(f"need both splits; got {len(train)} train and {len(val)} val")

    def matrix(items: list[dict[str, Any]], key: str = "truth_vector") -> np.ndarray:
        return np.asarray([e[key] for e in items], dtype=np.float64)

    x_train = matrix(train)
    y_train = np.asarray([1.0 if e["kl_grade"] >= OA_KL_THRESHOLD else 0.0 for e in train])
    x_val = matrix(val)
    y_val = np.asarray([1.0 if e["kl_grade"] >= OA_KL_THRESHOLD else 0.0 for e in val])

    print(
        f"\ntrain: {len(train)} knees, {int(y_train.sum())} with KL >= {OA_KL_THRESHOLD}"
        f"\nval  : {len(val)} knees, {int(y_val.sum())} with KL >= {OA_KL_THRESHOLD}"
    )
    print(f"patient overlap between splits: "
          f"{len({e['patient_id'] for e in train} & {e['patient_id'] for e in val})}")

    model = fit_logistic(
        x_train,
        y_train,
        l2=args.l2,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )

    train_metrics = evaluate_binary(
        y_train, model.predict_proba(x_train), threshold=args.threshold
    )
    val_metrics = evaluate_binary(y_val, model.predict_proba(x_val), threshold=args.threshold)

    print("\n--- coefficients (standardised features) ---")
    order = np.argsort(-np.abs(model.coefficients))
    for i in order:
        print(f"  {FEATURE_NAMES[i]:42s} {model.coefficients[i]:+.4f}")
    print(f"  {'(intercept)':42s} {model.intercept:+.4f}")

    print("\n--- training split (in-sample; not a performance claim) ---")
    for k, v in train_metrics.to_dict().items():
        print(f"  {k}: {v}")
    print("\n--- validation split, GROUND-TRUTH masks ---")
    for k, v in val_metrics.to_dict().items():
        print(f"  {k}: {v}")

    metrics: dict[str, Any] = {
        "training": train_metrics.to_dict(),
        "validation": val_metrics.to_dict(),
    }

    # --- the number that actually matters: predicted masks ---
    predicted_val = [e for e in val if "predicted_vector" in e]
    if predicted_val:
        xp = matrix(predicted_val, "predicted_vector")
        yp = np.asarray(
            [1.0 if e["kl_grade"] >= OA_KL_THRESHOLD else 0.0 for e in predicted_val]
        )
        pred_metrics = evaluate_binary(yp, model.predict_proba(xp), threshold=args.threshold)
        metrics["validation_on_predicted_masks"] = pred_metrics.to_dict()
        print(
            f"\n--- validation split, MODEL'S OWN masks ({len(predicted_val)} cases) ---"
        )
        for k, v in pred_metrics.to_dict().items():
            print(f"  {k}: {v}")
        print(
            "\n  This is the production number. The gap against the ground-truth row "
            "above is the cost of imperfect cartilage segmentation."
        )
    else:
        metrics["validation_on_predicted_masks"] = None
        print(
            "\n--- validation on predicted masks: NOT MEASURED ---\n"
            "  No predicted masks were supplied, so the only metrics here describe "
            "the classifier running on expert annotation — which is not how it runs "
            "in the product. Pass --predicted-masks to measure it properly."
        )

    # --- severity: a ridge regressor on the KL grade itself ---
    design = model.design(x_train)
    grades = np.asarray([float(e["kl_grade"]) for e in train])
    grade_coefficients = fit_grade_regressor(design, grades, args.l2)
    grade_intercept = float(grades.mean())
    grade_pred = np.clip(design @ grade_coefficients + grade_intercept, 0.0, 4.0)
    val_design = model.design(x_val)
    val_grade_pred = np.clip(val_design @ grade_coefficients + grade_intercept, 0.0, 4.0)
    val_grades = np.asarray([float(e["kl_grade"]) for e in val])
    grade_mae = float(np.mean(np.abs(val_grade_pred - val_grades)))
    metrics["kl_grade_regression"] = {
        "train_mae": round(float(np.mean(np.abs(grade_pred - grades))), 4),
        "validation_mae": round(grade_mae, 4),
        "note": (
            "Mean absolute error in KL grades on the validation split. Used only for "
            "the severity band; the yes/no answer comes from the logistic model."
        ),
    }
    print(f"\nKL grade regression validation MAE: {grade_mae:.2f} grades")

    classifier = OaClassifier(
        binary=model,
        grade_coefficients=[float(c) for c in grade_coefficients],
        grade_intercept=grade_intercept,
        version="0.1.0",
        threshold=args.threshold,
        metrics=metrics,
        training={
            "n_train": len(train),
            "n_val": len(val),
            "n_train_positive": int(y_train.sum()),
            "n_val_positive": int(y_val.sum()),
            "kl_threshold": OA_KL_THRESHOLD,
            "l2": args.l2,
            "manifest": args.manifest,
            "split_strategy": "patient-level, from the dataset manifest",
            "feature_names": list(FEATURE_NAMES),
            "features_from": "ground-truth segmentations",
        },
    )

    out = Path(args.out) if args.out else REPO / "artifacts" / "models" / "oa_kl" / MODEL_FILENAME
    classifier.save(out)
    print(f"\nwritten: {out.relative_to(REPO)}")

    balanced = val_metrics.balanced_accuracy
    majority = max(y_val.mean(), 1 - y_val.mean())
    print(
        f"\nSanity check: validation balanced accuracy {balanced:.3f}; always predicting "
        f"the majority class would score {majority:.3f} accuracy "
        f"({0.5:.3f} balanced)."
    )
    if not np.isfinite(balanced) or balanced <= 0.55:
        print(
            "  WARNING: this classifier is close to useless on held-out data. That "
            "result is written to the card as-is rather than tuned away."
        )


if __name__ == "__main__":
    main()
