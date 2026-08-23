"""Inspect implant placement vs actual bone/mesh bounds to find the misalignment."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
case_id = sys.argv[1] if len(sys.argv) > 1 else "oaizib_496"
case_dir = REPO / "artifacts" / "cases" / case_id

plan = json.loads((case_dir / "plan.json").read_text())
manifest = json.loads((case_dir / "meshes" / "manifest.json").read_text())

print(f"=== {case_id} ===\n")

print("MESH BOUNDS (mm, image frame):")
for m in manifest["meshes"]:
    lo, hi = m["bounds_mm"]
    centre = [round((lo[i] + hi[i]) / 2, 1) for i in range(3)]
    size = [round(hi[i] - lo[i], 1) for i in range(3)]
    print(f"  {m['structure']:26s} {m['category']:10s} centre={centre} size={size}")

print("\nPLACEMENTS (transform matrices):")
for pl in plan.get("placements", []):
    print(f"  {pl['structure']} <- {pl['component_id']} (size {pl.get('size_label')})")
    mat = pl["transform"]["matrix"]
    for row in mat:
        print("     ", [round(v, 3) for v in row])
    # translation column
    t = [round(mat[i][3], 2) for i in range(3)]
    print(f"      translation = {t}")
    print(f"      {pl['transform']['source']} -> {pl['transform']['target']}")

print("\nCUTS:")
for c in plan.get("cuts", []):
    pt = [round(v, 1) for v in c["plane"]["point"]]
    nr = [round(v, 3) for v in c["plane"]["normal"]]
    print(f"  {c['name']:22s} depth={c['depth_mm']} point={pt} normal={nr}")
