"""Update case 405's oa.json with trained classifier results."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "backend"))

from kneeai.oa.assess import load_classifier, assess_oa
from kneeai.imaging.io import read_volume
from kneeai.core.anatomy import Structure

clf = load_classifier()
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
result = assess_oa(mask, case_id="oaizib_405", classifier=clf, dataset_kl_grade=0)

out_path = REPO / "artifacts" / "cases" / "oaizib_405" / "oa.json"
out_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
print(f"Written: {out_path}")
print(f"OA: {result.oa_detected}, prob={result.probability:.3f}, KL={result.predicted_kl_grade:.1f}")
