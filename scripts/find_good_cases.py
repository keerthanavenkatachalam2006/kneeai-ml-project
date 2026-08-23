"""Find cases that segment well — for demo purposes.

Runs segmentation + quick quality check on test-set cases.
Reports which ones produce clean, unfragmented predictions.
"""
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "backend"))

from app.segmentation.inference import KneeSegmentationModel

MODEL_PATH = REPO / "models" / "segresnet_knee" / "best_model.pth"
DATA_DIR = REPO / "data" / "processed" / "extracted"

LABEL_NAMES = {1: "femur", 2: "fem_cart", 3: "tibia", 4: "med_tib_cart", 5: "lat_tib_cart"}


def check_case(model, image_path: Path) -> dict:
    """Segment and check quality."""
    nii = nib.load(str(image_path))
    image = np.asarray(nii.dataobj, dtype=np.float32)
    voxel_size = tuple(float(x) for x in nii.header.get_zooms()[:3])

    t0 = time.perf_counter()
    labels, confidence = model.predict_with_confidence(image, voxel_size)
    seg_time = time.perf_counter() - t0

    result = {"time_s": round(seg_time, 1), "structures": {}, "good": True, "issues": []}

    for label_idx, name in LABEL_NAMES.items():
        mask = labels == label_idx
        n_voxels = int(mask.sum())

        if n_voxels == 0:
            result["structures"][name] = {"voxels": 0, "components": 0, "fragmented": True}
            result["good"] = False
            result["issues"].append(f"{name}: missing")
            continue

        # Count connected components
        from scipy.ndimage import label as nd_label
        labeled, n_components = nd_label(mask)

        # Check fragmentation: if largest component is < 60% of total, it's bad
        if n_components > 1:
            component_sizes = []
            for c in range(1, n_components + 1):
                component_sizes.append(int((labeled == c).sum()))
            largest = max(component_sizes)
            largest_frac = largest / n_voxels
        else:
            largest_frac = 1.0

        fragmented = largest_frac < 0.6 or n_components > 50
        
        result["structures"][name] = {
            "voxels": n_voxels,
            "components": n_components,
            "largest_frac": round(largest_frac, 3),
            "fragmented": fragmented,
        }

        if fragmented:
            result["good"] = False
            result["issues"].append(f"{name}: fragmented ({n_components} components, largest={largest_frac:.0%})")

    # Check confidence
    if confidence is not None:
        for label_idx, name in LABEL_NAMES.items():
            mask = labels == label_idx
            if mask.any():
                mean_conf = float(confidence[mask].mean())
                result["structures"][name]["confidence"] = round(mean_conf, 3)
                if mean_conf < 0.5 and name in ("med_tib_cart", "lat_tib_cart"):
                    result["issues"].append(f"{name}: low confidence ({mean_conf:.2f})")

    return result


def main():
    model = KneeSegmentationModel(model_path=MODEL_PATH)
    if not model.load():
        print("ERROR: Could not load model")
        return

    # Check both train and test cases
    good_cases = []
    all_results = []

    for subdir in ("imagesTs", "imagesTr"):
        folder = DATA_DIR / subdir
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*_0000.nii.gz"))
        split = "test" if "Ts" in subdir else "train"

        for i, f in enumerate(files):
            case_id = f.name.replace("_0000.nii.gz", "")
            print(f"  [{i+1}/{len(files)}] {case_id} ({split})...", end=" ", flush=True)

            try:
                result = check_case(model, f)
                status = "GOOD" if result["good"] else "FRAGMENTED"
                print(f"{status} ({result['time_s']}s) {'; '.join(result['issues'][:2])}")

                all_results.append({"case_id": case_id, "split": split, **result})
                if result["good"]:
                    good_cases.append({"case_id": case_id, "split": split, "time_s": result["time_s"]})
            except Exception as exc:
                print(f"ERROR: {exc}")
                all_results.append({"case_id": case_id, "split": split, "good": False, "issues": [str(exc)]})

        # Only check test set for now (faster)
        if subdir == "imagesTs":
            break

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(good_cases)} good cases out of {len(all_results)} checked")
    print(f"\nGOOD CASES (will produce clean 3D + OA):")
    for c in good_cases:
        print(f"  {c['case_id']} ({c['split']}, {c['time_s']}s)")

    # Save results
    import json
    out = REPO / "artifacts" / "good_cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"good": good_cases, "all": all_results}, indent=2))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
