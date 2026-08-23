"""
Priority 2.1 — Full end-to-end pipeline:
MRI → AI Segmentation → Measurements → OA Analysis → Implant Matching → 3D Reconstruction

Usage:
    python scripts/run_full_pipeline.py                # Default: case 405
    python scripts/run_full_pipeline.py --case 410
    python scripts/run_full_pipeline.py --case 405 --compare-gt

The --compare-gt flag also runs measurements on ground-truth labels
and reports the difference (Priority 2.2 sanity check).
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, r'E:\KneeAI\backend')

from app.config import MODELS_DIR, PROCESSED_DATA, LABELS_MAP
from app.segmentation.inference import KneeSegmentationModel
from app.segmentation.meniscus_model import add_meniscus_to_segmentation, MENISCUS_LABEL
from app.data_loader import OAIZIBDataLoader
from app.measurements import KneeMeasurementEngine
from app.reconstruction import MeshReconstructor
from app.implant_database import ImplantMatcher
from app.oa_research import OAResearchEngine


def run_pipeline(case_id: int, compare_gt: bool = False):
    """Run the full pipeline on a single case."""

    output_dir = PROCESSED_DATA / "pipeline_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Load MRI ─────────────────────────────────────────────────────
    print("\n[1/6] Loading MRI data...")
    loader = OAIZIBDataLoader()
    case = loader.load_case(case_id)

    image = case['image']
    voxel_size = case['voxel_size']
    metadata = case['metadata']

    print(f"  Case {case_id}: {metadata.get('Gender_Label', '?')}, "
          f"Age {metadata.get('Age', '?')}, BMI {metadata.get('BMI', '?'):.1f}, "
          f"KL {metadata.get('KLGrade', '?')}")
    print(f"  Shape: {image.shape}, Voxel: {voxel_size}")

    # ─── Step 2: AI Segmentation ──────────────────────────────────────────────
    print("\n[2/6] Running AI segmentation...")
    model = KneeSegmentationModel()
    if not model.load():
        print("  ERROR: Model not found at", model.model_path)
        sys.exit(1)

    prediction = model.predict(image, voxel_size)

    # Check what was detected
    detected = {}
    for label_id in range(1, 6):
        count = (prediction == label_id).sum()
        detected[LABELS_MAP[label_id]] = count > 0
        status = "✓" if count > 0 else "✗ MISSING"
        print(f"  {LABELS_MAP[label_id]:<25}: {status} ({count} voxels)")

    # ─── Step 2b: Meniscus Estimation ─────────────────────────────────────────
    print("\n[2b/6] Estimating medial meniscus...")
    prediction = add_meniscus_to_segmentation(prediction, image, voxel_size)
    meniscus_count = (prediction == MENISCUS_LABEL).sum()
    detected["Medial Meniscus"] = meniscus_count > 0
    if meniscus_count > 0:
        print(f"  Medial Meniscus            : ✓ ({meniscus_count} voxels, "
              f"{meniscus_count * np.prod(voxel_size) / 1000:.2f} cm³)")
    else:
        print(f"  Medial Meniscus            : ✗ Could not estimate")

    # ─── Step 3: Measurements ─────────────────────────────────────────────────
    print("\n[3/6] Computing anatomical measurements (AI)...")
    engine = KneeMeasurementEngine()
    ai_measurements = engine.measure_case(prediction, voxel_size)

    _print_measurements(ai_measurements, prefix="  AI")

    # ─── Step 3b: Compare with ground truth (optional) ────────────────────────
    gt_measurements = None
    if compare_gt:
        print("\n[3b] Computing ground-truth measurements for comparison...")
        gt_label = case['label'].astype(np.uint8)
        gt_measurements = engine.measure_case(gt_label, voxel_size)
        _print_measurements(gt_measurements, prefix="  GT")
        _print_comparison(ai_measurements, gt_measurements)

    # ─── Step 4: OA Assessment ────────────────────────────────────────────────
    print("\n[4/6] OA morphometric assessment...")
    oa_engine = OAResearchEngine()
    oa_result = oa_engine.assess_oa_morphometrics(
        ai_measurements, clinical_kl=metadata.get('KLGrade')
    )
    print(f"  Morphometric severity: {oa_result['morphometric_severity']}")
    print(f"  Approximate KL range: {oa_result['approximate_kl_range']}")
    if oa_result.get('clinical_kl_grade') is not None:
        print(f"  Clinical KL (metadata): {oa_result['clinical_kl_grade']} — {oa_result.get('clinical_kl_description', '')}")
    for ind in oa_result.get('indicators', []):
        print(f"    • {ind}")

    # ─── Step 5: Implant Matching ─────────────────────────────────────────────
    print("\n[5/6] Matching to implant database...")
    matcher = ImplantMatcher()
    matches = matcher.match_all(ai_measurements, top_n=3)

    for component, match_list in matches.items():
        print(f"\n  {component.upper()} candidates:")
        for i, m in enumerate(match_list, 1):
            imp = m.implant
            print(f"    #{i}: {imp.manufacturer} {imp.system} Size {imp.size} "
                  f"— Score: {m.score:.0f}% ({m.fit_category})")
            print(f"        ML: {m.ml_mismatch_mm:+.1f}mm | AP: {m.ap_mismatch_mm:+.1f}mm")

    # ─── Step 6: 3D Reconstruction ────────────────────────────────────────────
    print("\n[6/6] Generating 3D mesh from AI prediction...")
    reconstructor = MeshReconstructor(smoothing_sigma=1.0, step_size=2)
    recon_result = reconstructor.export_case(case_id, prediction, voxel_size)
    print(f"  GLB: {recon_result['glb_path']} ({recon_result['glb_size_bytes']/1024:.0f} KB)")
    print(f"  Structures: {recon_result['num_structures']}")
    for name, stats in recon_result['structures'].items():
        print(f"    {name}: {stats['vertices']} verts, {stats['faces']} faces")

    # ─── Save Patient Profile JSON ────────────────────────────────────────────
    profile = _build_patient_profile(
        case_id, metadata, ai_measurements, oa_result, matches, recon_result, detected
    )
    profile_path = output_dir / f"case_{case_id:03d}_profile.json"
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"\n  Patient profile: {profile_path}")

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — Case {case_id}")
    print("=" * 60)

    return profile


def _print_measurements(m: dict, prefix: str = ""):
    """Print key measurements."""
    femur = m.get('femur', {})
    tibia = m.get('tibia', {})
    fc = m.get('femoral_cartilage', {})
    mtc = m.get('medial_tibial_cartilage', {})
    ltc = m.get('lateral_tibial_cartilage', {})
    meniscus = m.get('medial_meniscus', {})
    derived = m.get('derived', {})

    if femur:
        print(f"{prefix} Femur: ML={femur.get('condyle_ml_width_mm', 0):.1f}mm, "
              f"AP={femur.get('condyle_ap_depth_mm', 0):.1f}mm, "
              f"Vol={femur.get('volume_cm3', 0):.1f}cm³")
    if tibia:
        print(f"{prefix} Tibia: ML={tibia.get('plateau_ml_width_mm', 0):.1f}mm, "
              f"AP={tibia.get('plateau_ap_depth_mm', 0):.1f}mm, "
              f"Vol={tibia.get('volume_cm3', 0):.1f}cm³")
    if fc:
        print(f"{prefix} Fem. Cart: thickness={fc.get('thickness_mean_mm', 0):.2f}mm, "
              f"vol={fc.get('volume_cm3', 0):.3f}cm³")
    if mtc:
        print(f"{prefix} Med. Tib. Cart: thickness={mtc.get('thickness_mean_mm', 0):.2f}mm")
    if ltc:
        print(f"{prefix} Lat. Tib. Cart: thickness={ltc.get('thickness_mean_mm', 0):.2f}mm")
    if meniscus:
        rt = meniscus.get('regional_thickness', {})
        print(f"{prefix} Med. Meniscus: thickness={meniscus.get('thickness_mean_mm', 0):.2f}mm "
              f"(ant={rt.get('anterior_mm', 0):.1f}, body={rt.get('body_mm', 0):.1f}, "
              f"post={rt.get('posterior_mm', 0):.1f})")
    if derived:
        ratio = derived.get('medial_lateral_thickness_ratio')
        if ratio:
            print(f"{prefix} M/L ratio: {ratio:.2f}")


def _print_comparison(ai: dict, gt: dict):
    """Print AI vs GT measurement comparison."""
    print("\n  --- AI vs Ground Truth Comparison ---")

    comparisons = [
        ("Femur ML width", ai.get('femur', {}).get('condyle_ml_width_mm', 0),
         gt.get('femur', {}).get('condyle_ml_width_mm', 0), "mm"),
        ("Femur AP depth", ai.get('femur', {}).get('condyle_ap_depth_mm', 0),
         gt.get('femur', {}).get('condyle_ap_depth_mm', 0), "mm"),
        ("Tibia ML width", ai.get('tibia', {}).get('plateau_ml_width_mm', 0),
         gt.get('tibia', {}).get('plateau_ml_width_mm', 0), "mm"),
        ("Tibia AP depth", ai.get('tibia', {}).get('plateau_ap_depth_mm', 0),
         gt.get('tibia', {}).get('plateau_ap_depth_mm', 0), "mm"),
        ("Fem. Cart thickness", ai.get('femoral_cartilage', {}).get('thickness_mean_mm', 0),
         gt.get('femoral_cartilage', {}).get('thickness_mean_mm', 0), "mm"),
        ("Med. Tib. Cart thickness", ai.get('medial_tibial_cartilage', {}).get('thickness_mean_mm', 0),
         gt.get('medial_tibial_cartilage', {}).get('thickness_mean_mm', 0), "mm"),
        ("Lat. Tib. Cart thickness", ai.get('lateral_tibial_cartilage', {}).get('thickness_mean_mm', 0),
         gt.get('lateral_tibial_cartilage', {}).get('thickness_mean_mm', 0), "mm"),
    ]

    print(f"  {'Measurement':<28} {'AI':>8} {'GT':>8} {'Diff':>8} {'%Err':>7}")
    print(f"  {'-'*63}")
    for name, ai_val, gt_val, unit in comparisons:
        if gt_val > 0:
            diff = ai_val - gt_val
            pct = abs(diff) / gt_val * 100
            print(f"  {name:<28} {ai_val:>7.1f} {gt_val:>7.1f} {diff:>+7.1f} {pct:>6.1f}%")
        else:
            print(f"  {name:<28} {ai_val:>7.1f} {'N/A':>8} {'—':>8} {'—':>7}")


def _build_patient_profile(case_id, metadata, measurements, oa, implants, recon, detected):
    """Build the final patient profile JSON."""
    return {
        "case_id": case_id,
        "source": "AI_prediction",
        "demographics": {
            "age": metadata.get('Age'),
            "gender": metadata.get('Gender_Label'),
            "bmi": metadata.get('BMI'),
            "kl_grade": metadata.get('KLGrade'),
            "knee_side": metadata.get('side'),
        },
        "segmentation": {
            "model": "SegResNet (20 epochs, Dice=0.62)",
            "structures_detected": detected,
        },
        "measurements": {
            "femur": measurements.get('femur', {}),
            "tibia": measurements.get('tibia', {}),
            "femoral_cartilage": measurements.get('femoral_cartilage', {}),
            "medial_tibial_cartilage": measurements.get('medial_tibial_cartilage', {}),
            "lateral_tibial_cartilage": measurements.get('lateral_tibial_cartilage', {}),
            "medial_meniscus": measurements.get('medial_meniscus', {}),
            "derived": measurements.get('derived', {}),
        },
        "oa_assessment": oa,
        "implant_candidates": {
            component: [
                {
                    "rank": i + 1,
                    "manufacturer": m.implant.manufacturer,
                    "system": m.implant.system,
                    "size": m.implant.size,
                    "score": m.score,
                    "fit_category": m.fit_category,
                    "ml_mismatch_mm": m.ml_mismatch_mm,
                    "ap_mismatch_mm": m.ap_mismatch_mm,
                    "explanation": m.explanation,
                }
                for i, m in enumerate(match_list)
            ]
            for component, match_list in implants.items()
        },
        "reconstruction": {
            "glb_path": recon['glb_path'],
            "glb_size_kb": recon['glb_size_bytes'] / 1024,
            "structures": recon['structures'],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="KneeAI Full Pipeline")
    parser.add_argument("--case", type=int, default=405, help="Case ID (default: 405)")
    parser.add_argument("--compare-gt", action="store_true", help="Compare AI vs ground truth")
    args = parser.parse_args()

    print("=" * 60)
    print("KneeAI — Full Pipeline (MRI → AI → 3D → Implants)")
    print("=" * 60)

    run_pipeline(args.case, compare_gt=args.compare_gt)


if __name__ == "__main__":
    main()
