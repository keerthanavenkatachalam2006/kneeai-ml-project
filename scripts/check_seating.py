"""Inspect what the tibial component's Z extent actually represents."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
case_id = sys.argv[1] if len(sys.argv) > 1 else "oaizib_496"

manifest = json.loads(
    (REPO / "artifacts" / "cases" / case_id / "meshes" / "manifest.json").read_text()
)
by = {(m["structure"], m["category"]): m for m in manifest["meshes"]}

print(f"=== {case_id} : Z extents (superior axis) ===\n")
for key in [
    ("tibia", "anatomy"),
    ("tibia", "resected"),
    ("tibia", "component"),
    ("femur", "anatomy"),
    ("femur", "resected"),
    ("femur", "component"),
]:
    m = by.get(key)
    if not m:
        continue
    lo, hi = m["bounds_mm"]
    print(f"{key[0]:6s} {key[1]:10s} Z from {lo[2]:7.2f} to {hi[2]:7.2f}   (height {hi[2]-lo[2]:6.2f} mm)")

t_res = by.get(("tibia", "resected"))
t_comp = by.get(("tibia", "component"))
if t_res and t_comp:
    cut_z = t_res["bounds_mm"][1][2]
    comp_lo = t_comp["bounds_mm"][0][2]
    comp_hi = t_comp["bounds_mm"][1][2]
    print(f"\ntibial cut surface (top of resected tibia) : Z = {cut_z:.2f}")
    print(f"component spans                           : Z = {comp_lo:.2f} .. {comp_hi:.2f}")
    print(f"  above the cut : {comp_hi - cut_z:+.2f} mm   <- baseplate + insert thickness")
    print(f"  below the cut : {cut_z - comp_lo:+.2f} mm   <- keel / stem going into bone")
    print(
        "\nSo comp_lo is the KEEL TIP, not the baseplate underside."
        "\nSnapping comp_lo to the cut lifts the whole component by ~"
        f"{cut_z - comp_lo:.0f} mm and puts the plate up in the femur."
        "\n=> the planner's original Z was already correct; no correction needed."
    )
