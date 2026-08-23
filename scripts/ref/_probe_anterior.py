"""Is the femoral flange really standing proud of the anterior cortex?

The plan warns that it stands 7.1 mm proud, but that figure is measured against an
AI-detected ANTERIOR_FEMORAL_CORTEX landmark. This measures the same thing against
the mesh instead: for each slab along the patient-frame superoinferior axis that the
component occupies, compare the anterior-most point of the intact femur with the
anterior-most point of the component.

Everything is projected onto the exported patient frame, not onto raw image axes —
for this case the frame is tilted about 19 degrees from the voxel grid, so raw axes
give the wrong answer.
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
CASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts/cases/oaizib-417"

frame = json.loads((CASE / "frame.json").read_text())
ANT = np.array(frame["anterior"], dtype=float)
SUP = np.array(frame["superior"], dtype=float)
RIGHT = np.array(frame["right"], dtype=float)
print(f"case {CASE.name}  side {frame['side']}  frame method {frame['method']}")

MANIFEST = json.loads((CASE / "meshes/manifest.json").read_text())


def mesh_for(category: str, structure: str):
    entry = next(
        m
        for m in MANIFEST["meshes"]
        if m["category"] == category and m["structure"] == structure
    )
    loaded = trimesh.load(CASE / "meshes" / Path(entry["path"]).name, process=False)
    return loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded


fv = np.asarray(mesh_for("anatomy", "femur").vertices)
cv = np.asarray(mesh_for("component", "femur").vertices)
print(f"intact femur {len(fv)} verts   component {len(cv)} verts")

fa, fs = fv @ ANT, fv @ SUP
ca, cs = cv @ ANT, cv @ SUP

print(f"femur     anterior extent {fa.min():8.1f} .. {fa.max():8.1f} mm")
print(f"component anterior extent {ca.min():8.1f} .. {ca.max():8.1f} mm")
print(f"component superior span   {cs.min():8.1f} .. {cs.max():8.1f} mm")

print()
print(f"{'superior slab':>18}  {'femur ant':>10}  {'component ant':>14}  {'proud by':>9}")
edges = np.linspace(cs.min(), cs.max(), 11)
worst, worst_at = -1e9, 0.0
for a, b in zip(edges[:-1], edges[1:]):
    fm = (fs >= a) & (fs < b)
    cm = (cs >= a) & (cs < b)
    if not fm.any() or not cm.any():
        continue
    f_ant, c_ant = fa[fm].max(), ca[cm].max()
    if c_ant - f_ant > worst:
        worst, worst_at = c_ant - f_ant, (a + b) / 2
    print(f"  [{a:7.1f},{b:7.1f}]  {f_ant:10.1f}  {c_ant:14.1f}  {c_ant - f_ant:+9.2f}")

print()
print(f"worst anterior prominence vs the intact femur surface: {worst:+.2f} mm at {worst_at:.1f} mm superior")

lm = json.loads((CASE / "landmarks.json").read_text())
entries = lm["landmarks"] if isinstance(lm, dict) and "landmarks" in lm else lm
print()
for e in entries:
    if "femoral_cortex" in str(e.get("name", "")) or "condyle" in str(e.get("name", "")):
        p = np.array(e.get("position_mm") or e.get("position"), dtype=float)
        print(
            f"  {e['name']:28} anterior {p @ ANT:8.1f}  superior {p @ SUP:8.1f}  "
            f"right {p @ RIGHT:8.1f}  conf {e.get('confidence')}"
        )

plan = json.loads((CASE / "plan.json").read_text())
print()
for c in plan["cuts"]:
    pt = np.array(c["plane"]["point"], dtype=float)
    n = np.array(c["plane"]["normal"], dtype=float)
    print(
        f"  {c['plane']['label']:30} point anterior {pt @ ANT:8.1f} superior {pt @ SUP:8.1f}"
        f"   normal.anterior {n @ ANT:+.3f} normal.superior {n @ SUP:+.3f}"
    )
