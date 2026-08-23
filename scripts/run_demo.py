"""
KneeAI — Complete PS Demonstration

Single command to run the full pipeline and produce all demo artifacts.

Usage:
    python scripts/run_demo.py                    # Default: case 405
    python scripts/run_demo.py --case 410
    python scripts/run_demo.py --case 405 --open  # Also open 3D viewer

Produces:
    results/demo/
    ├── patient_profile.json       (complete structured output)
    ├── patient_summary.txt        (human-readable explainable result)
    ├── knee_3d.glb                (3D model for viewer)
    ├── segmentation_overlay.png   (visual QC)
    └── viewer_instructions.txt    (how to open 3D viewer)
"""
import sys
import os
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, r'E:\KneeAI\backend')

from app.config import MODELS_DIR, PROCESSED_DATA, LABELS_MAP
from app.segmentation.inference import KneeSegmentationModel
from app.segmentation.meniscus_model import add_meniscus_to_segmentation, MENISCUS_LABEL
from app.data_loader import OAIZIBDataLoader
from app.measurements import KneeMeasurementEngine
from app.reconstruction import MeshReconstructor
from app.implant_database import ImplantMatcher
from app.oa_research import OAResearchEngine


def generate_patient_summary(case_id, meta, measurements, oa, implant_matches, detected):
    """Generate the human-readable explainable result."""
    lines = []
    lines.append("=" * 64)
    lines.append("  KNEEAI — AI-ASSISTED KNEE ANALYSIS REPORT")
    lines.append("  Prototype System | For Research/Clinical Review Only")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"  Case ID:    {case_id}")
    lines.append(f"  Age:        {meta.get('Age', 'N/A')}")
    lines.append(f"  Gender:     {meta.get('Gender_Label', 'N/A')}")
    lines.append(f"  BMI:        {meta.get('BMI', 'N/A'):.1f}" if meta.get('BMI') else "  BMI:        N/A")
    lines.append(f"  KL Grade:   {int(meta.get('KLGrade', -1))} (from clinical metadata)")
    lines.append(f"  Knee Side:  {meta.get('side', 'N/A')}")
    lines.append("")

    lines.append("-" * 64)
    lines.append("  AI SEGMENTATION RESULTS")
    lines.append("-" * 64)
    for struct, found in detected.items():
        status = "DETECTED" if found else "NOT DETECTED"
        source = "(AI model)" if struct != "Medial Meniscus" else "(anatomical estimation)"
        lines.append(f"    {struct:<28} {status}  {source}")
    lines.append("")

    lines.append("-" * 64)
    lines.append("  ANATOMICAL MEASUREMENTS")
    lines.append("-" * 64)
    f = measurements.get('femur', {})
    t = measurements.get('tibia', {})
    fc = measurements.get('femoral_cartilage', {})
    mtc = measurements.get('medial_tibial_cartilage', {})
    ltc = measurements.get('lateral_tibial_cartilage', {})
    men = measurements.get('medial_meniscus', {})

    if f:
        lines.append(f"    Femur condyle ML width:     {f.get('condyle_ml_width_mm', 0):.1f} mm")
        lines.append(f"    Femur condyle AP depth:     {f.get('condyle_ap_depth_mm', 0):.1f} mm")
    if t:
        lines.append(f"    Tibia plateau ML width:     {t.get('plateau_ml_width_mm', 0):.1f} mm")
        lines.append(f"    Tibia plateau AP depth:     {t.get('plateau_ap_depth_mm', 0):.1f} mm")
    if fc:
        lines.append(f"    Femoral cartilage thickness: {fc.get('thickness_mean_mm', 0):.2f} mm (mean)")
    if mtc:
        lines.append(f"    Medial tibial cart. thick.:  {mtc.get('thickness_mean_mm', 0):.2f} mm (mean)")
    if ltc:
        lines.append(f"    Lateral tibial cart. thick.: {ltc.get('thickness_mean_mm', 0):.2f} mm (mean)")
    if men:
        rt = men.get('regional_thickness', {})
        lines.append(f"    Medial meniscus thickness:   {men.get('thickness_mean_mm', 0):.2f} mm (mean)")
        lines.append(f"      Anterior: {rt.get('anterior_mm', 0):.1f} mm | "
                     f"Body: {rt.get('body_mm', 0):.1f} mm | "
                     f"Posterior: {rt.get('posterior_mm', 0):.1f} mm")
    lines.append("")

    lines.append("-" * 64)
    lines.append("  OA MORPHOMETRIC ASSESSMENT")
    lines.append("-" * 64)
    lines.append(f"    Severity:     {oa.get('morphometric_severity', 'N/A')}")
    lines.append(f"    Approx KL:    {oa.get('approximate_kl_range', 'N/A')}")
    if oa.get('clinical_kl_grade') is not None:
        lines.append(f"    Clinical KL:  {oa['clinical_kl_grade']} ({oa.get('clinical_kl_description', '')})")
    for ind in oa.get('indicators', []):
        lines.append(f"      - {ind}")
    if not oa.get('indicators'):
        lines.append("      No morphometric abnormalities detected.")
    lines.append("")

    lines.append("-" * 64)
    lines.append("  IMPLANT CANDIDATE RANKING")
    lines.append("-" * 64)
    for component, match_list in implant_matches.items():
        lines.append(f"\n    {component.upper()} COMPONENT:")
        for i, m in enumerate(match_list[:3], 1):
            imp = m.implant
            lines.append(f"      #{i}  {imp.manufacturer} {imp.system} Size {imp.size}")
            lines.append(f"          Fit score:    {m.score:.0f}%  ({m.fit_category})")
            lines.append(f"          ML mismatch:  {m.ml_mismatch_mm:+.1f} mm")
            lines.append(f"          AP mismatch:  {m.ap_mismatch_mm:+.1f} mm")

        # Best candidate explanation
        if match_list:
            best = match_list[0]
            lines.append(f"\n    RECOMMENDED CANDIDATE: {best.implant.manufacturer} "
                        f"{best.implant.system} Size {best.implant.size}")
            lines.append(f"    REASON: {best.explanation}")
    lines.append("")

    lines.append("-" * 64)
    lines.append("  DISCLAIMER")
    lines.append("-" * 64)
    lines.append("    This is a research prototype for AI-assisted anatomical")
    lines.append("    measurement and implant size matching. Results are for")
    lines.append("    clinical review only and do NOT constitute a surgical")
    lines.append("    recommendation. Clinical implant sizing requires")
    lines.append("    intraoperative measurements, surgeon judgment, and")
    lines.append("    manufacturer-specific templates.")
    lines.append("")
    lines.append("    Methodology: MONAI SegResNet trained on OAIZIB-CM (507 knees).")
    lines.append("    Meniscus: Anatomical estimation (not dedicated trained model).")
    lines.append("    Implant DB: Approximate dimensions from surgical technique guides.")
    lines.append("")
    lines.append("=" * 64)

    return "\n".join(lines)


def run_demo(case_id: int, open_viewer: bool = False):
    """Run the complete demonstration pipeline."""
    demo_dir = Path("E:/KneeAI/results/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  KNEEAI — COMPLETE PS DEMONSTRATION")
    print("=" * 64)

    # ─── Load ─────────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading case {case_id}...")
    loader = OAIZIBDataLoader()
    case = loader.load_case(case_id)
    image = case['image']
    voxel_size = case['voxel_size']
    meta = case['metadata']
    print(f"  Patient: {meta.get('Gender_Label')}, Age {meta.get('Age')}, "
          f"BMI {meta.get('BMI', 0):.1f}, KL {int(meta.get('KLGrade', -1))}")

    # ─── AI Segmentation ──────────────────────────────────────────────────
    print("\n[2/7] AI segmentation (SegResNet)...")
    model = KneeSegmentationModel()
    if not model.load():
        print("  ERROR: Model not found!")
        sys.exit(1)
    prediction = model.predict(image, voxel_size)

    detected = {}
    for lid in range(1, 6):
        cnt = (prediction == lid).sum()
        detected[LABELS_MAP[lid]] = cnt > 0
        print(f"  {LABELS_MAP[lid]:<28}: {'OK' if cnt > 0 else 'MISSING'} ({cnt} voxels)")

    # ─── Meniscus ─────────────────────────────────────────────────────────
    print("\n[3/7] Meniscus estimation (anatomical priors)...")
    prediction = add_meniscus_to_segmentation(prediction, image, voxel_size)
    men_cnt = (prediction == MENISCUS_LABEL).sum()
    detected["Medial Meniscus"] = men_cnt > 0
    print(f"  Medial Meniscus: {'OK' if men_cnt > 0 else 'FAILED'} ({men_cnt} voxels)")

    # ─── Measurements ─────────────────────────────────────────────────────
    print("\n[4/7] Anatomical measurements...")
    engine = KneeMeasurementEngine()
    measurements = engine.measure_case(prediction, voxel_size)

    f = measurements.get('femur', {})
    t = measurements.get('tibia', {})
    print(f"  Femur: ML={f.get('condyle_ml_width_mm',0):.1f}mm, AP={f.get('condyle_ap_depth_mm',0):.1f}mm")
    print(f"  Tibia: ML={t.get('plateau_ml_width_mm',0):.1f}mm, AP={t.get('plateau_ap_depth_mm',0):.1f}mm")

    # ─── OA Assessment ────────────────────────────────────────────────────
    print("\n[5/7] OA morphometric assessment...")
    oa_engine = OAResearchEngine()
    oa_result = oa_engine.assess_oa_morphometrics(measurements, meta.get('KLGrade'))
    print(f"  Severity: {oa_result['morphometric_severity']} (approx KL {oa_result['approximate_kl_range']})")

    # ─── Implant Matching ─────────────────────────────────────────────────
    print("\n[6/7] Implant matching...")
    matcher = ImplantMatcher()
    matches = matcher.match_all(measurements, top_n=3)
    for comp, mlist in matches.items():
        if mlist:
            best = mlist[0]
            print(f"  Best {comp}: {best.implant.manufacturer} {best.implant.system} "
                  f"Size {best.implant.size} (Score={best.score:.0f}%, {best.fit_category})")

    # ─── 3D Reconstruction ────────────────────────────────────────────────
    print("\n[7/7] 3D reconstruction...")
    reconstructor = MeshReconstructor(smoothing_sigma=1.0, step_size=2)
    meshes = reconstructor.reconstruct_all_structures(prediction, voxel_size)
    glb_path = demo_dir / "knee_3d.glb"
    reconstructor.export_glb(meshes, glb_path)
    print(f"  GLB: {glb_path} ({glb_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Structures: {len(meshes)}")

    # ─── Save outputs ─────────────────────────────────────────────────────
    print("\n  Saving demo outputs...")

    # Patient profile JSON
    profile = {
        "case_id": case_id,
        "source": "AI_prediction (SegResNet)",
        "demographics": {
            "age": meta.get('Age'),
            "gender": meta.get('Gender_Label'),
            "bmi": float(meta.get('BMI', 0)),
            "kl_grade": int(meta.get('KLGrade', -1)),
            "knee_side": meta.get('side'),
        },
        "segmentation": {
            "model": "SegResNet (MONAI)",
            "training_data": "OAIZIB-CM (404 cases)",
            "structures_detected": detected,
            "meniscus_method": "anatomical_estimation",
        },
        "measurements": measurements,
        "oa_assessment": oa_result,
        "implant_candidates": {
            comp: [
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
                for i, m in enumerate(mlist)
            ]
            for comp, mlist in matches.items()
        },
    }
    profile_path = demo_dir / "patient_profile.json"
    with open(profile_path, 'w') as fp:
        json.dump(profile, fp, indent=2, default=str)

    # Patient summary text
    summary = generate_patient_summary(case_id, meta, measurements, oa_result, matches, detected)
    summary_path = demo_dir / "patient_summary.txt"
    with open(summary_path, 'w') as fp:
        fp.write(summary)

    # Viewer instructions
    viewer_path = demo_dir / "viewer_instructions.txt"
    with open(viewer_path, 'w') as fp:
        fp.write("KneeAI 3D Viewer Instructions\n")
        fp.write("=" * 40 + "\n\n")
        fp.write("1. Open viewer/index.html in a web browser\n")
        fp.write("2. Click 'Load GLB file' and select: results/demo/knee_3d.glb\n")
        fp.write("3. Click 'Load patient profile JSON' and select: results/demo/patient_profile.json\n")
        fp.write("4. Use mouse to rotate/zoom the 3D model\n")
        fp.write("5. Toggle structures on/off with checkboxes\n")
        fp.write("6. View measurements and implant candidates in side panels\n")

    # Visualization
    try:
        from scripts.run_inference import save_slice_visualization
        save_slice_visualization(image, prediction, case_id, demo_dir)
    except Exception:
        pass

    print(f"\n  Outputs saved to: {demo_dir}/")
    print(f"    patient_profile.json")
    print(f"    patient_summary.txt")
    print(f"    knee_3d.glb")
    print(f"    viewer_instructions.txt")

    # Print the summary
    print("\n")
    print(summary)

    if open_viewer:
        viewer_html = Path("E:/KneeAI/viewer/index.html")
        if viewer_html.exists():
            os.startfile(str(viewer_html))
            print("\n  3D viewer opened in browser.")

    return profile


def main():
    parser = argparse.ArgumentParser(description="KneeAI Complete Demo")
    parser.add_argument("--case", type=int, default=405, help="Case ID")
    parser.add_argument("--open", action="store_true", help="Open 3D viewer")
    args = parser.parse_args()

    run_demo(args.case, open_viewer=args.open)


if __name__ == "__main__":
    main()
