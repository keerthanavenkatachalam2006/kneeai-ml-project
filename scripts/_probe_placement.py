"""Check whether implant component meshes are exported already positioned.

The components look dislocated in the viewer. Two possible causes:

1. The GLB is exported in implant-local coordinates and the placement transform
   from ``plan.json`` still has to be applied by the renderer.
2. The GLB is already baked into the image frame and the placement is simply what
   the planner produced.

Comparing the component's bounds against the resected bone's bounds distinguishes
them: a baked component sits inside the bone's envelope, a local one sits near the
origin.
"""

import json
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[1] / "artifacts/demo/cases/demo-phantom-right"
manifest = json.loads((CASE / "meshes/manifest.json").read_text())
plan = json.loads((CASE / "plan.json").read_text())

print("=" * 78)
print("MESH BOUNDS (image frame, mm)")
print("=" * 78)
by_key = {}
for m in manifest["meshes"]:
    lo, hi = np.array(m["bounds_mm"][0]), np.array(m["bounds_mm"][1])
    centre = (lo + hi) / 2
    by_key[(m["category"], m["structure"])] = (lo, hi, centre)
    print(
        f"{m['category']:10} {m['structure']:24} "
        f"min[{lo[0]:7.1f} {lo[1]:7.1f} {lo[2]:7.1f}]  "
        f"max[{hi[0]:7.1f} {hi[1]:7.1f} {hi[2]:7.1f}]  "
        f"centre[{centre[0]:6.1f} {centre[1]:6.1f} {centre[2]:6.1f}]"
    )

print()
print("=" * 78)
print("PLACEMENT TRANSFORMS from plan.json")
print("=" * 78)
for p in plan.get("placements", []):
    matrix = np.array(p["transform"]["matrix"])
    translation = matrix[:3, 3]
    print(f"{p['component_id']}")
    print(f"  structure   {p['structure']}  size {p['size_label']}")
    print(f"  {p['transform']['source']} -> {p['transform']['target']}")
    print(f"  translation [{translation[0]:7.2f} {translation[1]:7.2f} {translation[2]:7.2f}] mm")

print()
print("=" * 78)
print("DIAGNOSIS")
print("=" * 78)
for structure in ("femur", "tibia"):
    comp = by_key.get(("component", structure))
    bone = by_key.get(("resected", structure)) or by_key.get(("anatomy", structure))
    if not comp or not bone:
        continue
    offset = comp[2] - bone[2]
    inside = bool(np.all(comp[0] >= bone[0] - 25) and np.all(comp[1] <= bone[1] + 25))
    print(f"{structure}:")
    print(f"  component centre - bone centre = [{offset[0]:6.1f} {offset[1]:6.1f} {offset[2]:6.1f}] mm")
    print(f"  component within bone envelope (+/-25 mm): {inside}")
    print(
        "  -> looks BAKED into the image frame"
        if inside
        else "  -> looks UNPLACED; the transform is probably not applied"
    )
