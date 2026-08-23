"""Quick test: load trained classifier and run on case 405."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "backend"))

from kneeai.oa.assess import load_classifier, assess_oa
from kneeai.imaging.io import read_volume
from kneeai.core.anatomy import Structure

# Load the trained classifier
clf = load_classifier()
print(f"Classifier: loaded={clf is not None}, version={clf.version if clf else None}")

# Load segmentation from existing case
seg_path = REPO / "artifacts" / "cases" / "oaizib_405" / "segmentation.nii.gz"
label_map = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
}
mask = read_volume(seg_path, is_label=True, label_map=label_map)

result = assess_oa(mask, case_id="oaizib_405", classifier=clf)
d = result.to_dict()
print(f"OA detected: {d['oa_detected']}")
print(f"Probability: {d['probability']}")
print(f"Severity: {d['severity']}")
print(f"Predicted KL: {d['predicted_kl_grade']}")
print(f"Warnings: {len(d['warnings'])}")
print(f"Available: {d['available']}")
if d.get("indicators"):
    print("\nTop indicators:")
    for ind in d["indicators"][:3]:
        print(f"  {ind['display_name']}: {ind['value']} {ind['unit']} -> {ind['direction']} ({ind['log_odds_contribution']:+.3f})")
