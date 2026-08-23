"""Test the fixed measurement methodology on ground truth."""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')
import numpy as np
from app.data_loader import OAIZIBDataLoader
from app.measurements import KneeMeasurementEngine
from app.implant_database import ImplantMatcher

loader = OAIZIBDataLoader()
engine = KneeMeasurementEngine()
matcher = ImplantMatcher()

print("=" * 60)
print("MEASUREMENT FIX VERIFICATION (Ground Truth)")
print("=" * 60)
print("Expected ranges (literature):")
print("  Femoral condyle ML: 55-80mm")
print("  Femoral condyle AP: 55-75mm")
print("  Tibial plateau ML:  60-85mm")
print("  Tibial plateau AP:  40-55mm")
print()

for case_id in [405, 406, 407, 408, 410]:
    try:
        case = loader.load_case(case_id)
    except Exception as e:
        print(f"Case {case_id}: SKIP ({e})")
        continue

    gt = case['label'].astype(np.uint8)
    vs = case['voxel_size']
    meta = case['metadata']
    m = engine.measure_case(gt, vs)

    f = m.get('femur', {})
    t = m.get('tibia', {})

    fml = f.get('condyle_ml_width_mm', 0)
    fap = f.get('condyle_ap_depth_mm', 0)
    tml = t.get('plateau_ml_width_mm', 0)
    tap = t.get('plateau_ap_depth_mm', 0)

    # Check if in expected range
    fml_ok = "OK" if 50 <= fml <= 85 else "OUT OF RANGE"
    fap_ok = "OK" if 50 <= fap <= 80 else "OUT OF RANGE"
    tml_ok = "OK" if 55 <= tml <= 90 else "OUT OF RANGE"
    tap_ok = "OK" if 35 <= tap <= 60 else "OUT OF RANGE"

    gender = meta.get('Gender_Label', '?')
    print(f"Case {case_id} ({gender}):")
    print(f"  Femur ML: {fml:.1f}mm [{fml_ok}]  AP: {fap:.1f}mm [{fap_ok}]")
    print(f"  Tibia ML: {tml:.1f}mm [{tml_ok}]  AP: {tap:.1f}mm [{tap_ok}]")

    # Implant matching
    matches = matcher.match_all(m, top_n=1)
    for comp, mlist in matches.items():
        if mlist:
            best = mlist[0]
            imp = best.implant
            print(f"  Best {comp}: {imp.manufacturer} {imp.system} Size {imp.size} "
                  f"(Score={best.score:.0f}%, {best.fit_category})")
    print()

print("=" * 60)
