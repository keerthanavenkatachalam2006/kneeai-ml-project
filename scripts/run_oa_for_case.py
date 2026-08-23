"""Run OA assessment for a specific case that already has segmentation."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "backend"))

case_id = sys.argv[1] if len(sys.argv) > 1 else "oaizib_488"
case_dir = REPO / "artifacts" / "cases" / case_id

from kneeai.oa.assess import assess_oa, load_classifier
from kneeai.imaging.io import read_volume
from kneeai.core.anatomy import Structure

clf = load_classifier()
if clf is None:
    print("ERROR: No classifier found")
    sys.exit(1)

seg_path = case_dir / "segmentation.nii.gz"
if not seg_path.exists():
    print(f"ERROR: No segmentation at {seg_path}")
    sys.exit(1)

label_map = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
}

# Get KL grade from manifest
kl_grade = None
manifest = json.loads((REPO / "data" / "manifests" / "oaizib_60.json").read_text())
for r in manifest["records"]:
    if r["case_id"].replace("-", "_") == case_id:
        kl_grade = r.get("kl_grade")
        break

print(f"Case: {case_id}, Dataset KL grade: {kl_grade}")
mask = read_volume(seg_path, is_label=True, label_map=label_map)
result = assess_oa(mask, case_id=case_id, classifier=clf, dataset_kl_grade=kl_grade)

d = result.to_dict()
print(f"OA detected: {d['oa_detected']}")
print(f"Probability: {d['probability']}")
print(f"Severity: {d['severity']}")
print(f"Predicted KL: {d['predicted_kl_grade']}")

# Write oa.json
oa_path = case_dir / "oa.json"
# Handle NaN values
import math
def sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj

oa_path.write_text(json.dumps(sanitize(d), indent=2) + "\n")
print(f"Written: {oa_path}")
