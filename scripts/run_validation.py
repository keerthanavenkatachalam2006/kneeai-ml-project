"""
Priority 6 — Lightweight validation: AI predictions vs ground truth.

Runs inference on N test cases, computes:
- Per-class Dice scores
- Volume errors
- Measurement errors (AI vs GT)
- Implant sizing agreement

Usage:
    python scripts/run_validation.py              # 10 cases
    python scripts/run_validation.py --n 50       # 50 cases
    python scripts/run_validation.py --all        # All 103 test cases
"""
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r'E:\KneeAI\backend')

from app.config import MODELS_DIR, PROCESSED_DATA, LABELS_MAP
from app.segmentation.inference import KneeSegmentationModel
from app.segmentation.meniscus_model import add_meniscus_to_segmentation
from app.data_loader import OAIZIBDataLoader
from app.measurements import KneeMeasurementEngine
from app.implant_database import ImplantMatcher


def compute_dice(pred: np.ndarray, gt: np.ndarray, label_id: int) -> float:
    """Compute Dice score for a single label."""
    pred_mask = (pred == label_id)
    gt_mask = (gt == label_id)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union


def compute_volume_error(pred: np.ndarray, gt: np.ndarray, label_id: int, voxel_size: tuple) -> dict:
    """Compute volume and volume error for a label."""
    voxel_vol = np.prod(voxel_size)
    pred_vol = (pred == label_id).sum() * voxel_vol / 1000  # cm³
    gt_vol = (gt == label_id).sum() * voxel_vol / 1000
    diff = pred_vol - gt_vol
    pct = (abs(diff) / gt_vol * 100) if gt_vol > 0 else 0
    return {"pred_cm3": pred_vol, "gt_cm3": gt_vol, "diff_cm3": diff, "error_pct": pct}


def get_top_implant(matches: dict, component: str) -> str:
    """Get the top implant candidate identifier."""
    match_list = matches.get(component, [])
    if match_list:
        m = match_list[0]
        return f"{m.implant.manufacturer}_{m.implant.system}_{m.implant.size}"
    return "none"


def run_validation(n_cases: int = 10):
    """Run validation on test cases."""

    output_dir = Path("E:/KneeAI/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KneeAI — Validation: AI vs Ground Truth")
    print("=" * 60)

    # Load model
    model = KneeSegmentationModel()
    if not model.load():
        print("ERROR: Model not found.")
        sys.exit(1)
    print(f"Model loaded: {model.model_path}")

    # Get test cases
    loader = OAIZIBDataLoader()
    metadata = loader.load_metadata()
    test_cases = metadata[metadata['split'] == 'test']['CMT-ID'].tolist()

    if n_cases < len(test_cases):
        test_cases = test_cases[:n_cases]

    print(f"Validating {len(test_cases)} test cases...\n")

    engine = KneeMeasurementEngine()
    matcher = ImplantMatcher()

    # Results accumulators
    all_dices = {label_id: [] for label_id in range(1, 6)}
    all_volumes = {label_id: [] for label_id in range(1, 6)}
    measurement_errors = {
        "femur_ml": [], "femur_ap": [],
        "tibia_ml": [], "tibia_ap": [],
        "fc_thickness": [], "mtc_thickness": [], "ltc_thickness": [],
    }
    implant_agreements = {"femoral": 0, "tibial": 0, "total": 0}
    case_results = []

    t_start = time.time()

    for i, case_id in enumerate(test_cases):
        try:
            case = loader.load_case(case_id)
        except Exception as e:
            print(f"  Case {case_id}: SKIP ({e})")
            continue

        image = case['image']
        gt_label = case['label'].astype(np.uint8)
        voxel_size = case['voxel_size']

        # AI prediction
        prediction = model.predict(image, voxel_size)

        # Per-class Dice
        case_dice = {}
        for label_id in range(1, 6):
            d = compute_dice(prediction, gt_label, label_id)
            all_dices[label_id].append(d)
            case_dice[LABELS_MAP[label_id]] = round(d, 4)

        mean_dice = np.mean([case_dice[LABELS_MAP[lid]] for lid in range(1, 6)])

        # Volume errors
        for label_id in range(1, 6):
            vol = compute_volume_error(prediction, gt_label, label_id, voxel_size)
            all_volumes[label_id].append(vol["error_pct"])

        # Measurement comparison
        ai_m = engine.measure_case(prediction, voxel_size)
        gt_m = engine.measure_case(gt_label, voxel_size)

        def safe_err(ai_dict, gt_dict, key):
            ai_val = ai_dict.get(key, 0)
            gt_val = gt_dict.get(key, 0)
            if gt_val > 0 and ai_val > 0:
                return abs(ai_val - gt_val)
            return None

        err = safe_err(ai_m.get('femur', {}), gt_m.get('femur', {}), 'condyle_ml_width_mm')
        if err is not None: measurement_errors["femur_ml"].append(err)

        err = safe_err(ai_m.get('femur', {}), gt_m.get('femur', {}), 'condyle_ap_depth_mm')
        if err is not None: measurement_errors["femur_ap"].append(err)

        err = safe_err(ai_m.get('tibia', {}), gt_m.get('tibia', {}), 'plateau_ml_width_mm')
        if err is not None: measurement_errors["tibia_ml"].append(err)

        err = safe_err(ai_m.get('tibia', {}), gt_m.get('tibia', {}), 'plateau_ap_depth_mm')
        if err is not None: measurement_errors["tibia_ap"].append(err)

        err = safe_err(ai_m.get('femoral_cartilage', {}), gt_m.get('femoral_cartilage', {}), 'thickness_mean_mm')
        if err is not None: measurement_errors["fc_thickness"].append(err)

        err = safe_err(ai_m.get('medial_tibial_cartilage', {}), gt_m.get('medial_tibial_cartilage', {}), 'thickness_mean_mm')
        if err is not None: measurement_errors["mtc_thickness"].append(err)

        err = safe_err(ai_m.get('lateral_tibial_cartilage', {}), gt_m.get('lateral_tibial_cartilage', {}), 'thickness_mean_mm')
        if err is not None: measurement_errors["ltc_thickness"].append(err)

        # Implant agreement
        ai_matches = matcher.match_all(ai_m, top_n=1)
        gt_matches = matcher.match_all(gt_m, top_n=1)

        implant_agreements["total"] += 1
        if get_top_implant(ai_matches, "femoral") == get_top_implant(gt_matches, "femoral"):
            implant_agreements["femoral"] += 1
        if get_top_implant(ai_matches, "tibial") == get_top_implant(gt_matches, "tibial"):
            implant_agreements["tibial"] += 1

        case_results.append({
            "case_id": case_id,
            "dice": case_dice,
            "mean_dice": round(mean_dice, 4),
        })

        # Progress
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(test_cases) - i - 1)
            print(f"  [{i+1}/{len(test_cases)}] Case {case_id}: Mean Dice={mean_dice:.3f} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # ─── Summary ──────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS ({len(case_results)} cases, {total_time:.0f}s)")
    print(f"{'='*60}")

    print(f"\n--- Dice Scores ---")
    print(f"  {'Structure':<25} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-'*50}")
    overall_dices = []
    for label_id in range(1, 6):
        dices = all_dices[label_id]
        if dices:
            mean = np.mean(dices)
            overall_dices.append(mean)
            print(f"  {LABELS_MAP[label_id]:<25} {mean:>7.4f} {np.std(dices):>7.4f} "
                  f"{np.min(dices):>7.4f} {np.max(dices):>7.4f}")
    print(f"  {'-'*50}")
    print(f"  {'OVERALL MEAN':<25} {np.mean(overall_dices):>7.4f}")

    print(f"\n--- Measurement Errors (MAE in mm) ---")
    for key, errors in measurement_errors.items():
        if errors:
            print(f"  {key:<20}: MAE={np.mean(errors):.2f}mm, "
                  f"Median={np.median(errors):.2f}mm (n={len(errors)})")

    print(f"\n--- Implant Size Agreement ---")
    total = implant_agreements["total"]
    if total > 0:
        print(f"  Femoral: {implant_agreements['femoral']}/{total} "
              f"({implant_agreements['femoral']/total*100:.0f}%)")
        print(f"  Tibial:  {implant_agreements['tibial']}/{total} "
              f"({implant_agreements['tibial']/total*100:.0f}%)")

    # ─── Save report ──────────────────────────────────────────────────────────
    report = {
        "model": "SegResNet (20 epochs)",
        "n_cases": len(case_results),
        "total_time_s": round(total_time),
        "dice_scores": {
            LABELS_MAP[lid]: {
                "mean": round(np.mean(all_dices[lid]), 4),
                "std": round(np.std(all_dices[lid]), 4),
                "min": round(np.min(all_dices[lid]), 4),
                "max": round(np.max(all_dices[lid]), 4),
            }
            for lid in range(1, 6) if all_dices[lid]
        },
        "overall_mean_dice": round(np.mean(overall_dices), 4),
        "measurement_mae_mm": {
            k: round(np.mean(v), 2) for k, v in measurement_errors.items() if v
        },
        "implant_agreement": implant_agreements,
        "per_case": case_results,
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # Human-readable summary
    summary_path = output_dir / "validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("KneeAI Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: SegResNet\n")
        f.write(f"Cases: {len(case_results)}\n")
        f.write(f"Time: {round(total_time)}s\n\n")
        f.write("Dice Scores:\n")
        for lid in range(1, 6):
            if all_dices[lid]:
                f.write(f"  {LABELS_MAP[lid]:<25}: {np.mean(all_dices[lid]):.4f} "
                       f"(+/-{np.std(all_dices[lid]):.4f})\n")
        f.write(f"  {'OVERALL MEAN':<25}: {np.mean(overall_dices):.4f}\n\n")
        f.write("Measurement MAE (mm):\n")
        for k, v in measurement_errors.items():
            if v:
                f.write(f"  {k:<20}: {np.mean(v):.2f} (median {np.median(v):.2f})\n")
        f.write(f"\nImplant Agreement:\n")
        if total > 0:
            f.write(f"  Femoral: {implant_agreements['femoral']}/{total} "
                   f"({implant_agreements['femoral']/total*100:.0f}%)\n")
            f.write(f"  Tibial:  {implant_agreements['tibial']}/{total} "
                   f"({implant_agreements['tibial']/total*100:.0f}%)\n")
    print(f"  Summary saved: {summary_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="KneeAI Validation")
    parser.add_argument("--n", type=int, default=10, help="Number of test cases")
    parser.add_argument("--all", action="store_true", help="Run on all 103 test cases")
    args = parser.parse_args()

    n = 103 if args.all else args.n
    run_validation(n)


if __name__ == "__main__":
    main()
