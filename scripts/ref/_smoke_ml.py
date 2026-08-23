"""Ad-hoc smoke check for the ML metric / post-processing layer.

Kept in scripts/ rather than tests/ because it prints a human-readable summary;
the assertions live in tests/test_ml_metrics.py.
"""

from __future__ import annotations

import json

import numpy as np

from kneeai.core import Structure
from kneeai.imaging import Volume
from kneeai.ml import dice, evaluate_binary, iou, postprocess_mask, surface_distances
from kneeai.ml.postprocess import component_summary

shape = (60, 60, 60)
zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]


def sphere(cz: int, cy: int, cx: int, r: int) -> np.ndarray:
    return ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) < r**2


gt = sphere(30, 30, 30, 12)
pred = sphere(31, 30, 30, 12)  # displaced by exactly one voxel

print("== overlap metrics (1-voxel displaced sphere, r=12) ==")
print(f"  dice {dice(pred, gt):.4f}   iou {iou(pred, gt):.4f}")

sd = surface_distances(pred, gt, (0.5, 0.5, 0.5))
print("== surface metrics (0.5 mm isotropic) ==")
print(f"  hausdorff {sd['hausdorff_mm']:.3f} mm   hd95 {sd['hausdorff95_mm']:.3f} mm")
print(f"  assd {sd['assd_mm']:.3f} mm   surface voxels {sd['n_surface_gt']}")

metrics = evaluate_binary(pred, gt, (0.5, 0.5, 0.5), structure=Structure.FEMUR)
print(f"  volume error {metrics.volume_error_pct:+.2f}%")

print("\n== post-processing ==")
labels = np.zeros(shape, np.uint8)
labels[gt] = 1
labels[30, 30, 30] = 0  # interior hole
labels[2:5, 2:5, 2:5] = 1  # satellite island
labels[sphere(48, 30, 30, 7)] = 3  # separate tibia

mask = Volume.from_labels(
    labels,
    {Structure.BACKGROUND: 0, Structure.FEMUR: 1, Structure.TIBIA: 3},
    spacing=[0.5, 0.5, 0.5],
)
print("  before:", json.dumps(component_summary(mask), indent=None))
cleaned, report = postprocess_mask(mask)
for action in report.actions:
    print(f"  action: {action}")
for warning in report.warnings:
    print(f"  warning: {warning}")
print("  after :", json.dumps(component_summary(cleaned), indent=None))
