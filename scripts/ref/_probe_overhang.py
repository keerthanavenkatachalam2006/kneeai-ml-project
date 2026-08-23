"""Locate the 16.9 mm tibial overhang reported for the AI case.

Coverage is 99.3% and the mediolateral mismatch is only +1.2 mm, so a 16.9 mm
maximum overhang has to come from a local excursion rather than a globally
oversized baseplate. This dumps the overhang region piece by piece so we can see
whether it is a real rim protrusion or an artefact of the footprint polygon.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts/cases/oaizib-417"

fit = json.loads((CASE / "implants.json").read_text())["selected"]["tibial_fit"]
print("reported tibial fit")
for key in (
    "component_ml_mm",
    "component_ap_mm",
    "coverage_pct",
    "overhang_max_mm",
    "overhang_mean_mm",
    "overhang_area_mm2",
    "overhang_perimeter_pct",
    "undercoverage_max_mm",
    "translation_mm",
    "rotation_deg",
):
    print(f"  {key:26} {fit.get(key)}")

# Rebuild the footprint and the placed outline from the plan inputs.
from kneeai.implants import catalog as cat  # noqa: E402
from kneeai.implants import matching as mt  # noqa: E402

meshes = json.loads((CASE / "meshes/manifest.json").read_text())
print()
print("mesh manifest entries")
for m in meshes["meshes"]:
    print(f"  {m['category']:10} {m['structure']:24} {m['path']}")

# The footprint is recomputed by the planner; reproduce it from the resected mesh.
plan = json.loads((CASE / "plan.json").read_text())
cuts = {c["plane"]["label"]: c for c in plan["cuts"]}
print()
print("cut planes")
for name, c in cuts.items():
    print(f"  {name:32} depth {c['depth_mm']} mm")

import trimesh  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

resected = next(
    m for m in meshes["meshes"] if m["category"] == "resected" and m["structure"] == "tibia"
)
loaded = trimesh.load(CASE / "meshes" / Path(resected["path"]).name, process=False)
mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
print()
print(f"resected tibia mesh: {len(mesh.vertices)} verts  bounds {mesh.bounds.tolist()}")

cut = cuts["tibial_resection"]["plane"]
normal = np.array(cut["normal"], dtype=float)
point = np.array(cut["point"], dtype=float)

section = mesh.section(plane_origin=point + normal * 0.35, plane_normal=normal)
if section is None:
    print("no section at the tibial cut -- cannot analyse")
    raise SystemExit(1)
planar, _to3d = section.to_planar()
polys = [Polygon(e.discrete[0][:, :2]) if hasattr(e, "discrete") else None for e in []]
polys = [Polygon(p) for p in planar.polygons_full] if False else list(planar.polygons_full)
print(f"section polygons: {len(polys)}")
for i, p in enumerate(sorted(polys, key=lambda q: -q.area)):
    minx, miny, maxx, maxy = p.bounds
    print(
        f"  poly {i}: area {p.area:8.1f} mm2  extent {maxx - minx:6.1f} x {maxy - miny:6.1f} mm"
        f"  interiors {len(p.interiors)}"
    )
    if p.interiors:
        for j, ring in enumerate(p.interiors):
            ip = Polygon(ring)
            print(f"      hole {j}: area {ip.area:8.1f} mm2")


# ---------------------------------------------------------------------------
# Locate the overhang: place the reference baseplate outline on this section and
# report where implant material falls outside the bone.
# ---------------------------------------------------------------------------
from shapely.affinity import rotate as srotate  # noqa: E402
from shapely.affinity import translate as stranslate  # noqa: E402

from kneeai.core.anatomy import Side  # noqa: E402
from kneeai.implants.reference import tibial_footprint  # noqa: E402

bone = max(polys, key=lambda p: p.area)
outline, pcl = tibial_footprint(74.0, 49.0, Side.RIGHT)
from shapely.geometry import Polygon as SPoly  # noqa: E402

implant = SPoly(outline)
pcl_poly = SPoly(pcl)
print()
print(f"bone     : area {bone.area:8.1f} mm2  bbox {bone.bounds}")
print(f"implant  : area {implant.area:8.1f} mm2  bbox {implant.bounds}")
print(f"pcl notch: area {pcl_poly.area:8.1f} mm2  bbox {pcl_poly.bounds}")

bc = np.asarray(bone.centroid.coords[0])
ic = np.asarray(implant.centroid.coords[0])
placed = stranslate(implant, *(bc - ic))

over = placed.difference(bone)
print()
print(f"overhang area {over.area:.1f} mm2  parts {getattr(over, 'geom_type', '?')}")
parts = list(getattr(over, "geoms", [over]))
parts.sort(key=lambda g: -g.area)
for i, g in enumerate(parts[:8]):
    minx, miny, maxx, maxy = g.bounds
    # depth of this piece = max distance from its boundary to the bone boundary
    pts = np.asarray(g.exterior.coords)
    from shapely.geometry import Point

    d = max(bone.exterior.distance(Point(p)) for p in pts)
    cx, cy = g.centroid.x, g.centroid.y
    quad = ("medial" if cx > 0 else "lateral") + "/" + ("anterior" if cy > 0 else "posterior")
    print(
        f"  part {i}: area {g.area:7.1f} mm2  depth {d:5.2f} mm  "
        f"centre ({cx:6.1f},{cy:6.1f}) {quad:18} bbox "
        f"[{minx:6.1f} {miny:6.1f} {maxx:6.1f} {maxy:6.1f}]"
    )
