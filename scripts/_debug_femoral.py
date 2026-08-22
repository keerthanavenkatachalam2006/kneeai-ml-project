"""Find which intermediate solid in femoral_solid is not a closed volume."""

import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kneeai.core.anatomy import Side  # noqa: E402
from kneeai.implants.geometry import _superellipse, _rounded_box  # noqa: E402
from kneeai.implants.reference import build_reference_catalog  # noqa: E402

catalog = build_reference_catalog()
component = next(
    c
    for f in catalog.families.values()
    for c in f.components.values()
    if c.component_id == "REF-PRIMARY-CR-FEM-6-R"
)

envelope = component.envelope_ap_mm or component.ap_mm
ml = float(component.ml_mm)
distal_t = float(component.distal_thickness_mm) or 9.0
posterior_t = float(component.posterior_offset_mm) or distal_t
flange_t = float(component.anterior_flange_thickness_mm) or 3.0
chamfer = float(component.metadata.get("chamfer_offset_mm", 7.0))
box_ap = component.internal_ap_mm
half = box_ap / 2.0
step = chamfer * np.sqrt(2.0)
flange_h = 0.44 * envelope
posterior_h = 0.30 * envelope

print(f"envelope {envelope}  ml {ml}  box_ap {box_ap:.2f}  half {half:.2f}  step {step:.2f}")
print(f"flange_h {flange_h:.2f}  posterior_h {posterior_h:.2f}")
print(f"thickness distal {distal_t} posterior {posterior_t} flange {flange_t}")

inner = [
    (-half, posterior_h),
    (-half, step),
    (-half + step, 0.0),
    (half - step, 0.0),
    (half, step),
    (half, flange_h),
]
thickness = [posterior_t, posterior_t, distal_t, 0.5 * (distal_t + flange_t), flange_t]
bands = [
    LineString([inner[i], inner[i + 1]]).buffer(thickness[i], cap_style=1, join_style=1, quad_segs=8)
    for i in range(len(inner) - 1)
]
band = unary_union(bands)
print(f"band: {band.geom_type} area {band.area:.1f} interiors "
      f"{len(band.interiors) if band.geom_type == 'Polygon' else 'n/a'}")

reach = max(flange_h, posterior_h) + max(thickness) + 10.0
bone = Polygon([*inner, (half, reach), (-half, reach)])
print(f"bone: valid {bone.is_valid} area {bone.area:.1f}")
shell_2d = band.difference(bone)
print(f"shell_2d: {shell_2d.geom_type} area {shell_2d.area:.1f}")
if shell_2d.geom_type == "MultiPolygon":
    for i, g in enumerate(sorted(shell_2d.geoms, key=lambda g: -g.area)):
        print(f"   part {i}: area {g.area:8.1f}  bounds {tuple(round(v, 1) for v in g.bounds)}")
    shell_2d = max(shell_2d.geoms, key=lambda g: g.area)
print(f"kept: area {shell_2d.area:.1f} interiors {len(shell_2d.interiors)} "
      f"bounds {tuple(round(v, 1) for v in shell_2d.bounds)}")

import shapely  # noqa: E402

shell_2d = shapely.set_precision(shell_2d, 1e-4)
if shell_2d.geom_type == "MultiPolygon":
    shell_2d = max(shell_2d.geoms, key=lambda g: g.area)
print(f"after set_precision: verts {len(shell_2d.exterior.coords)} interiors {len(shell_2d.interiors)}")

swept = trimesh.creation.extrude_polygon(shell_2d, ml)
print(f"swept raw : watertight {swept.is_watertight} volume {swept.volume:.1f} faces {len(swept.faces)}")
swept.merge_vertices()
swept.remove_degenerate_faces() if hasattr(swept, "remove_degenerate_faces") else None
swept.fill_holes()
print(f"swept fix : watertight {swept.is_watertight} volume {swept.volume:.1f} faces {len(swept.faces)}")
verts = np.asarray(swept.vertices, dtype=np.float64)
swept2 = trimesh.Trimesh(
    vertices=np.column_stack([verts[:, 2] - ml / 2.0, verts[:, 0], verts[:, 1]]),
    faces=np.asarray(swept.faces),
    process=False,
)
print(f"swept perm: watertight {swept2.is_watertight} volume {swept2.volume:.1f}")

outline = _superellipse(ml / 2.0, half + flange_t + 1.0, 3.0)
z_lo = -(distal_t + posterior_t) - 5.0
z_hi = flange_h + flange_t + 5.0
env = trimesh.creation.extrude_polygon(Polygon(outline), z_hi - z_lo)
env.apply_translation([0.0, 0.0, z_lo])
print(f"envelope  : watertight {env.is_watertight} volume {env.volume:.1f}")

try:
    step1 = trimesh.boolean.intersection([swept2, env])
    print(f"intersect : watertight {step1.is_watertight} volume {step1.volume:.1f} faces {len(step1.faces)}")
except Exception as exc:
    print("intersect FAILED:", exc)
    raise SystemExit(1)

notch_half = 0.30 * ml / 2.0
notch = _rounded_box(
    x=(-notch_half, notch_half),
    y=(-half - posterior_t - 5.0, -0.06 * box_ap),
    z=(z_lo - 5.0, 0.0),
    radius=min(notch_half * 0.6, 6.0),
)
print(f"notch     : watertight {notch.is_watertight} volume {notch.volume:.1f}")
try:
    step2 = trimesh.boolean.difference([step1, notch])
    print(f"minus notch: watertight {step2.is_watertight} volume {step2.volume:.1f} faces {len(step2.faces)}")
except Exception as exc:
    print("notch difference FAILED:", exc)
    raise SystemExit(1)

groove_r = 0.24 * ml
groove_d = min(0.030 * ml, flange_t * 0.5)
gh = flange_h + flange_t + 20.0
groove = trimesh.creation.cylinder(radius=groove_r, height=gh, sections=48)
groove.apply_translation([0.0, half + flange_t + groove_r - groove_d, gh / 2.0 - 12.0])
print(f"groove    : watertight {groove.is_watertight} volume {groove.volume:.1f}")
try:
    final = trimesh.boolean.difference([step2, groove])
    print(f"final     : watertight {final.is_watertight} volume {final.volume:.1f} faces {len(final.faces)}")
    print(f"bounds    : {np.round(final.bounds, 2).tolist()}")
except Exception as exc:
    print("groove difference FAILED:", exc)
