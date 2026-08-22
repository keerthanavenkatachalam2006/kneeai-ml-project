"""One-line-per-fact summary of an exported case: fit, plan, simulation, placement.

Used to check the effect of a change without re-reading five JSON files by hand.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts/cases/oaizib-417"


def load(name: str) -> dict:
    path = CASE / name
    return json.loads(path.read_text()) if path.exists() else {}


print(f"case {CASE.name}")

plan = load("plan.json")
if plan:
    print("\ncuts")
    for c in plan.get("cuts", []):
        print(f"  {c['plane']['label']:30} {c['depth_mm']:6.2f} mm   {c['reference'][:80]}")

imp = load("implants.json")
sel = imp.get("selected")
if sel:
    print(f"\nimplant  {sel['configuration']['label']}")
    print(f"  combined score {sel['score']:.2f}   acceptable {sel['acceptable']}")
    for key in ("femoral_fit", "tibial_fit"):
        f = sel.get(key) or {}
        print(
            f"  {key:12} score {f.get('score', float('nan')):6.2f}  "
            f"acceptable {f.get('acceptable')}  violations {f.get('violations')}"
        )
    t = sel.get("tibial_fit") or {}
    if t.get("coverage_pct") is not None:
        print(
            f"    footprint: coverage {t['coverage_pct']:.1f}%  "
            f"overhang max {t['overhang_max_mm']:.2f} mean {t['overhang_mean_mm']:.2f} mm  "
            f"uncovered max {t['undercoverage_max_mm']:.2f} mm"
        )
    fem = sel.get("femoral_fit") or {}
    print(
        f"    femoral: ml mismatch {fem.get('ml_mismatch_mm')} mm  "
        f"ap mismatch {fem.get('ap_mismatch_mm')} mm"
    )
    others = [c for c in imp.get("candidates", []) if c is not sel]
    print(f"  candidates considered: {len(imp.get('candidates', []))}")
    for c in imp.get("candidates", [])[:6]:
        print(
            f"    {c['configuration']['label']:34} score {c['score']:6.2f}  "
            f"acceptable {c['acceptable']}"
        )
else:
    print("\nimplant  NONE SELECTED")

sim = load("simulation.json")
if sim:
    print(
        f"\nsimulation passed {sim.get('passed')}  "
        f"failures {sim.get('n_failures')}  warnings {sim.get('n_warnings')}"
    )
    for f in sim.get("findings", []):
        print(f"  [{f.get('severity', '?')}] {f.get('code')}: {str(f.get('message'))[:140]}")

meshes = load("meshes/manifest.json")
if meshes:
    by = {}
    for m in meshes.get("meshes", []):
        lo, hi = np.array(m["bounds_mm"][0]), np.array(m["bounds_mm"][1])
        by[(m["category"], m["structure"])] = (lo + hi) / 2
    print("\ncomponent-to-bone mediolateral offset (should be ~0)")
    for s in ("femur", "tibia"):
        comp, bone = by.get(("component", s)), by.get(("resected", s))
        if comp is not None and bone is not None:
            print(f"  {s:6} {comp[0] - bone[0]:+6.2f} mm")

meas = load("measurements.json")
if meas:
    print(
        f"\nmeasurements {meas.get('n_measurements')}  "
        f"unavailable {len(meas.get('unavailable') or {})}"
    )
