"""Smoke check: build the phantom knee and reconstruct its surfaces."""

from __future__ import annotations

import json
import time

from kneeai.core import Side, Structure
from kneeai.imaging import validate_volume
from kneeai.ml import postprocess_mask
from kneeai.phantom import build_phantom
from kneeai.recon import reconstruct_all

t0 = time.time()
knee = build_phantom(side=Side.RIGHT, spacing_mm=0.8)
print(f"phantom built in {time.time() - t0:.1f}s")
print(json.dumps(knee.summary(), indent=2))

print("\n== image validation ==")
report = validate_volume(knee.image)
print(f"  passed={report.passed}")
for issue in report.issues:
    print(f"  {issue}")

print("\n== mask validation ==")
mask_report = validate_volume(knee.mask)
print(f"  passed={mask_report.passed}")
for issue in mask_report.issues:
    print(f"  {issue}")

print("\n== post-processing ==")
cleaned, pp = postprocess_mask(knee.mask)
for action in pp.actions:
    print(f"  {action}")
for warning in pp.warnings:
    print(f"  WARN {warning}")

print("\n== reconstruction ==")
t1 = time.time()
meshes, failures = reconstruct_all(cleaned)
print(f"  reconstructed in {time.time() - t1:.1f}s")
for structure, mesh in sorted(meshes.items(), key=lambda kv: kv[0].value):
    q = mesh.quality()
    print(
        f"  {structure.value:<26} faces={q.n_faces:>7}  vol={q.volume_mm3:>9.1f} mm3  "
        f"watertight={q.is_watertight}  components={q.n_components}"
    )
    if q.issues:
        for issue in q.issues:
            print(f"      ! {issue}")
if failures:
    print("  failures:", failures)

print("\n== femur mesh vs truth ==")
femur = meshes.require(Structure.FEMUR)
import numpy as np

ml = femur.extent_along([1, 0, 0])
ap = femur.extent_along([0, 1, 0])
print(f"  mesh femoral ML {ml:.2f} mm   truth {knee.truth['femoral_ml_mm']:.2f} mm")
print(f"  mesh femoral AP {ap:.2f} mm   truth {knee.truth['femoral_ap_mm']:.2f} mm")
print(f"  mesh bounds {np.round(femur.bounds, 1).tolist()}")
