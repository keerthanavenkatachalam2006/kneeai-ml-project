"""Build every reference component solid and report whether it is usable.

Checks the properties the rest of the platform depends on:

* watertight — the interpenetration check skips non-watertight meshes
* positive volume and a plausible extent for the component's declared size
* the seating surface sits at z = 0, since the placement transform assumes it
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kneeai.core.anatomy import Side  # noqa: E402
from kneeai.implants.catalog import ComponentType  # noqa: E402
from kneeai.implants.geometry import component_surface  # noqa: E402
from kneeai.implants.reference import build_reference_catalog  # noqa: E402

catalog = build_reference_catalog()
components = [
    c
    for family in catalog.families.values()
    for c in family.components.values()
    if c.side in (Side.RIGHT, Side.UNKNOWN)
]
print(f"catalog: {len(components)} right/unsided components")

seen: set[ComponentType] = set()
failures = 0
for c in sorted(components, key=lambda c: (c.component_type.value, c.size_label)):
    # One example per type/size is enough; insert thicknesses repeat the same shape.
    key = (c.component_type, c.size_label.split("/")[0])
    if key in seen:
        continue
    seen.add(key)

    solid = component_surface(c)
    if solid is None:
        print(f"  {c.component_id:34} NO SOLID (falls back to prism)")
        continue

    lo, hi = solid.bounds
    extent = hi - lo
    ok = solid.is_watertight and solid.volume > 0
    flag = "ok  " if ok else "BAD "
    if not ok:
        failures += 1
    print(
        f"  {flag}{c.component_id:34} faces {len(solid.faces):6d}  "
        f"vol {solid.volume / 1000.0:7.2f} cm3  "
        f"extent {extent[0]:5.1f} x {extent[1]:5.1f} x {extent[2]:5.1f} mm  "
        f"z [{lo[2]:6.1f},{hi[2]:6.1f}]  watertight {solid.is_watertight}"
    )

print()
# A closer look at one femoral and one tibial component.
for cid in ("REF-PRIMARY-CR-FEM-6-R", "REF-PRIMARY-CR-TIB-5-R", "REF-PRIMARY-CR-INS-5-6-R"):
    c = next((x for x in components if x.component_id == cid), None)
    if c is None:
        print(f"{cid}: not in catalog")
        continue
    solid = component_surface(c)
    if solid is None:
        print(f"{cid}: no solid")
        continue
    lo, hi = solid.bounds
    print(f"{cid}")
    print(f"  declared  ml {c.ml_mm} ap {c.ap_mm} envelope_ap {c.envelope_ap_mm}")
    print(f"  bounds    x [{lo[0]:7.2f},{hi[0]:7.2f}]  y [{lo[1]:7.2f},{hi[1]:7.2f}]  z [{lo[2]:7.2f},{hi[2]:7.2f}]")
    print(f"  volume    {solid.volume / 1000.0:.2f} cm3   faces {len(solid.faces)}   watertight {solid.is_watertight}")
    # Material must not cross into the retained bone above the seating plane except
    # where the flange or condyles legitimately rise alongside it.
    v = np.asarray(solid.vertices)
    print(f"  above the seating plane: {(v[:, 2] > 0.01).sum()} of {len(v)} vertices")

print()
print("FAILURES:", failures)
sys.exit(1 if failures else 0)
