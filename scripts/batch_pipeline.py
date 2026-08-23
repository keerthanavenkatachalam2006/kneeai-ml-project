"""Batch run pipeline on test cases to find which ones complete successfully."""
import sys
import time
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "backend"))

from kneeai.bridge import BridgeConfig, run_bridge_pipeline


def try_case(case_id: str) -> dict:
    """Run the pipeline and return success/failure status."""
    config = BridgeConfig(
        data_root=REPO / "data" / "processed" / "extracted",
        artifact_root=REPO / "artifacts",
        strategy="restricted_kinematic",
        side="right",
    )
    t0 = time.perf_counter()
    try:
        result = run_bridge_pipeline(case_id=case_id, config=config)
        elapsed = time.perf_counter() - t0
        # Check if meshes exist
        case_dir = REPO / "artifacts" / "cases" / case_id
        n_meshes = len(list((case_dir / "meshes").glob("*.glb"))) if (case_dir / "meshes").exists() else 0
        has_oa = (case_dir / "oa.json").exists()
        has_plan = (case_dir / "plan.json").exists()
        return {
            "case_id": case_id,
            "success": True,
            "time_s": round(elapsed, 1),
            "n_meshes": n_meshes,
            "has_oa": has_oa,
            "has_plan": has_plan,
            "planned": result.get("stages", {}).get("pipeline", {}).get("planned", False),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "case_id": case_id,
            "success": False,
            "time_s": round(elapsed, 1),
            "error": f"{type(exc).__name__}: {str(exc)[:100]}",
        }


def main():
    # Test cases to try
    test_cases = [f"oaizib_{i}" for i in range(405, 500)]
    
    # Filter to only existing ones
    data_dir = REPO / "data" / "processed" / "extracted"
    existing = []
    for case_id in test_cases:
        for subdir in ("imagesTs", "imagesTr"):
            if (data_dir / subdir / f"{case_id}_0000.nii.gz").exists():
                existing.append(case_id)
                break
    
    print(f"Found {len(existing)} cases to test")
    
    good = []
    failed = []
    
    for i, case_id in enumerate(existing):
        # Skip already processed ones
        case_dir = REPO / "artifacts" / "cases" / case_id
        if (case_dir / "result.json").exists() and (case_dir / "meshes").exists():
            n_meshes = len(list((case_dir / "meshes").glob("*.glb")))
            has_oa = (case_dir / "oa.json").exists()
            has_plan = (case_dir / "plan.json").exists()
            print(f"  [{i+1}/{len(existing)}] {case_id}: ALREADY DONE ({n_meshes} meshes, oa={has_oa}, plan={has_plan})")
            good.append({"case_id": case_id, "success": True, "cached": True, "n_meshes": n_meshes, "has_oa": has_oa, "has_plan": has_plan})
            continue
            
        print(f"  [{i+1}/{len(existing)}] {case_id}: running...", end=" ", flush=True)
        result = try_case(case_id)
        
        if result["success"]:
            print(f"SUCCESS ({result['time_s']}s, {result['n_meshes']} meshes, plan={result['has_plan']})")
            good.append(result)
        else:
            print(f"FAILED ({result['time_s']}s): {result['error'][:80]}")
            failed.append(result)
        
        # Stop after we have enough good ones
        if len(good) >= 10:
            print(f"\nGot 10 good cases, stopping early.")
            break

    print(f"\n{'='*60}")
    print(f"GOOD: {len(good)} | FAILED: {len(failed)}")
    print(f"\nBEST CASES FOR DEMO (all have 3D + OA + plan):")
    for c in good:
        extras = []
        if c.get("has_plan"): extras.append("plan")
        if c.get("has_oa"): extras.append("OA")
        if c.get("n_meshes", 0) >= 5: extras.append(f"{c['n_meshes']} meshes")
        print(f"  {c['case_id']} — {', '.join(extras)}")

    out = REPO / "artifacts" / "demo_cases.json"
    out.write_text(json.dumps({"good": good, "failed": failed}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
