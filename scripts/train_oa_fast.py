"""Fast OA classifier training — uses simplified features from label volumes.

This is a streamlined version that:
1. Loads ground-truth labels (already extracted NIfTI)
2. Computes volume-based features (fast, no thickness estimation)
3. Fits the L2-regularised logistic regression
4. Saves the classifier to artifacts/models/oa_kl/oa_classifier.json

The full train_oa_classifier.py uses sheet_thickness which is slow (~10s/case).
This version uses volume ratios and basic geometry (~1s/case) for a quick fit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

# Add project paths
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "backend"))

from kneeai.oa.classifier import (
    OA_KL_THRESHOLD,
    MODEL_FILENAME,
    OaClassifier,
    evaluate_binary,
    fit_logistic,
)
from kneeai.oa.features import FEATURE_NAMES


# Label map
LABEL_MAP = {
    0: "background",
    1: "femur",
    2: "femoral_cartilage",
    3: "tibia",
    4: "medial_tibial_cartilage",
    5: "lateral_tibial_cartilage",
}


def compute_features_fast(label_path: Path, spacing: tuple[float, ...] = (0.7, 0.3646, 0.3646)) -> np.ndarray | None:
    """Compute the 7-feature vector from a label volume, quickly.
    
    Uses volume-based approximations for thickness (vol/surface_area).
    """
    try:
        nii = nib.load(str(label_path))
        labels = np.asarray(nii.dataobj, dtype=np.int16)
        voxel_spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
        if voxel_spacing[0] > 0:
            spacing = voxel_spacing
    except Exception:
        return None

    voxel_volume_mm3 = float(np.prod(spacing))

    # Compute volumes
    volumes = {}
    for label_idx, name in LABEL_MAP.items():
        if label_idx == 0:
            continue
        mask = labels == label_idx
        volumes[name] = float(mask.sum()) * voxel_volume_mm3

    femur_vol = volumes.get("femur", 0.0)
    tibia_vol = volumes.get("tibia", 0.0)
    fc_vol = volumes.get("femoral_cartilage", 0.0)
    mtc_vol = volumes.get("medial_tibial_cartilage", 0.0)
    ltc_vol = volumes.get("lateral_tibial_cartilage", 0.0)

    if femur_vol <= 0 or tibia_vol <= 0:
        return None

    # Approximate thickness from volume / estimated surface area
    # Surface area approximation: count boundary voxels and scale
    def approx_thickness(cart_mask: np.ndarray, bone_mask: np.ndarray) -> float:
        """Approximate mean thickness = volume / contact_surface_area."""
        if not cart_mask.any():
            return 0.0
        # Contact surface: cartilage voxels adjacent to bone
        from scipy.ndimage import binary_dilation
        dilated_bone = binary_dilation(bone_mask, iterations=1)
        contact = cart_mask & dilated_bone
        contact_count = contact.sum()
        if contact_count <= 0:
            # Fallback: use total cartilage surface
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(cart_mask, iterations=1)
            surface_count = max(cart_mask.sum() - eroded.sum(), 1)
            # Each surface voxel represents one face with area = product of 2 spacings
            avg_face_area = (spacing[0] * spacing[1] + spacing[1] * spacing[2] + spacing[0] * spacing[2]) / 3.0
            surface_area = surface_count * avg_face_area
        else:
            # Contact area
            avg_face_area = (spacing[0] * spacing[1] + spacing[1] * spacing[2] + spacing[0] * spacing[2]) / 3.0
            surface_area = contact_count * avg_face_area

        cart_volume = cart_mask.sum() * voxel_volume_mm3
        return cart_volume / max(surface_area, 1.0)

    femur_mask = labels == 1
    tibia_mask = labels == 3
    fc_mask = labels == 2
    mtc_mask = labels == 4
    ltc_mask = labels == 5

    fc_thick = approx_thickness(fc_mask, femur_mask)
    mtc_thick = approx_thickness(mtc_mask, tibia_mask)
    ltc_thick = approx_thickness(ltc_mask, tibia_mask)

    # Volume ratios (scaled by 1000 per feature spec)
    fc_ratio = 1000.0 * fc_vol / femur_vol if femur_vol > 0 else 0.0
    mtc_ratio = 1000.0 * mtc_vol / tibia_vol if tibia_vol > 0 else 0.0
    ltc_ratio = 1000.0 * ltc_vol / tibia_vol if tibia_vol > 0 else 0.0

    # Medial/lateral thickness ratio
    ml_ratio = mtc_thick / ltc_thick if ltc_thick > 0 else (0.0 if mtc_thick == 0.0 else float("nan"))

    # Feature vector in FEATURE_NAMES order:
    # femoral_cartilage_thickness_mm, medial_tibial_cartilage_thickness_mm,
    # lateral_tibial_cartilage_thickness_mm, femoral_cartilage_volume_ratio,
    # medial_tibial_cartilage_volume_ratio, lateral_tibial_cartilage_volume_ratio,
    # medial_lateral_thickness_ratio
    vector = np.array([
        fc_thick,
        mtc_thick,
        ltc_thick,
        fc_ratio,
        mtc_ratio,
        ltc_ratio,
        ml_ratio,
    ], dtype=np.float64)

    return vector


def main() -> None:
    manifest_path = REPO / "data" / "manifests" / "oaizib_60.json"
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest.get("records", []) if r.get("kl_grade") is not None and r.get("label_path")]

    print(f"Manifest: {len(records)} usable records with KL grades")

    # Extract features
    data = []
    t0 = time.perf_counter()
    for i, record in enumerate(records, 1):
        label_path = REPO / record["label_path"]
        if not label_path.exists():
            print(f"  [{i}/{len(records)}] {record['case_id']}: label missing, skipped")
            continue

        vector = compute_features_fast(label_path)
        if vector is None:
            print(f"  [{i}/{len(records)}] {record['case_id']}: feature extraction failed")
            continue

        data.append({
            "case_id": record["case_id"],
            "patient_id": record.get("patient_id"),
            "kl_grade": int(record["kl_grade"]),
            "split": record.get("split", "train"),
            "vector": vector.tolist(),
        })
        elapsed = time.perf_counter() - t0
        avg = elapsed / i
        print(f"  [{i}/{len(records)}] {record['case_id']}: KL {record['kl_grade']} ({record['split']}) "
              f"{elapsed:.1f}s (avg {avg:.1f}s/case)")

    print(f"\nExtracted features for {len(data)} cases in {time.perf_counter() - t0:.1f}s")

    # Split
    train = [e for e in data if e["split"] == "train"]
    val = [e for e in data if e["split"] == "val"]
    
    if not train or not val:
        print(f"ERROR: need both splits; got {len(train)} train and {len(val)} val")
        raise SystemExit(1)

    x_train = np.array([e["vector"] for e in train], dtype=np.float64)
    y_train = np.array([1.0 if e["kl_grade"] >= OA_KL_THRESHOLD else 0.0 for e in train])
    x_val = np.array([e["vector"] for e in val], dtype=np.float64)
    y_val = np.array([1.0 if e["kl_grade"] >= OA_KL_THRESHOLD else 0.0 for e in val])

    print(f"\ntrain: {len(train)} knees, {int(y_train.sum())} with KL >= {OA_KL_THRESHOLD}")
    print(f"val  : {len(val)} knees, {int(y_val.sum())} with KL >= {OA_KL_THRESHOLD}")
    print(f"patient overlap: {len({e['patient_id'] for e in train} & {e['patient_id'] for e in val})}")

    # Fit
    model = fit_logistic(
        x_train, y_train,
        l2=2.0,
        iterations=40000,
        learning_rate=0.05,
    )

    train_metrics = evaluate_binary(y_train, model.predict_proba(x_train), threshold=0.5)
    val_metrics = evaluate_binary(y_val, model.predict_proba(x_val), threshold=0.5)

    print("\n--- coefficients (standardised) ---")
    order = np.argsort(-np.abs(model.coefficients))
    for i in order:
        print(f"  {FEATURE_NAMES[i]:42s} {model.coefficients[i]:+.4f}")
    print(f"  {'(intercept)':42s} {model.intercept:+.4f}")

    print("\n--- training (in-sample) ---")
    for k, v in train_metrics.to_dict().items():
        print(f"  {k}: {v}")
    print("\n--- validation (ground-truth masks) ---")
    for k, v in val_metrics.to_dict().items():
        print(f"  {k}: {v}")

    metrics = {
        "training": train_metrics.to_dict(),
        "validation": val_metrics.to_dict(),
    }

    # Grade regression
    design = model.design(x_train)
    grades = np.array([float(e["kl_grade"]) for e in train])
    n, d = design.shape
    gram = design.T @ design + 2.0 * np.eye(d)
    grade_coefficients = np.linalg.solve(gram, design.T @ (grades - grades.mean()))
    grade_intercept = float(grades.mean())

    val_design = model.design(x_val)
    val_grades = np.array([float(e["kl_grade"]) for e in val])
    val_grade_pred = np.clip(val_design @ grade_coefficients + grade_intercept, 0.0, 4.0)
    grade_mae = float(np.mean(np.abs(val_grade_pred - val_grades)))
    metrics["kl_grade_regression"] = {
        "train_mae": round(float(np.mean(np.abs(np.clip(design @ grade_coefficients + grade_intercept, 0, 4) - grades))), 4),
        "validation_mae": round(grade_mae, 4),
    }
    print(f"\nKL grade regression validation MAE: {grade_mae:.2f} grades")

    # Save
    classifier = OaClassifier(
        binary=model,
        grade_coefficients=[float(c) for c in grade_coefficients],
        grade_intercept=grade_intercept,
        version="0.1.0",
        threshold=0.5,
        metrics=metrics,
        training={
            "n_train": len(train),
            "n_val": len(val),
            "n_train_positive": int(y_train.sum()),
            "n_val_positive": int(y_val.sum()),
            "kl_threshold": OA_KL_THRESHOLD,
            "l2": 2.0,
            "manifest": "data/manifests/oaizib_60.json",
            "split_strategy": "patient-level, from the dataset manifest",
            "feature_names": list(FEATURE_NAMES),
            "features_from": "ground-truth segmentations (fast volume-based thickness)",
        },
    )

    out = REPO / "artifacts" / "models" / "oa_kl" / MODEL_FILENAME
    classifier.save(out)
    print(f"\nSaved: {out.relative_to(REPO)}")

    balanced = val_metrics.balanced_accuracy
    majority = max(y_val.mean(), 1 - y_val.mean())
    print(f"\nSanity: val balanced acc {balanced:.3f}; majority-class baseline {majority:.3f} acc (0.500 balanced)")
    if balanced <= 0.55:
        print("  WARNING: classifier is near-useless on held-out data.")


if __name__ == "__main__":
    main()
