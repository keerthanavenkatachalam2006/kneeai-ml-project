"""Smoke check: detect landmarks on the phantom and compare against ground truth."""

from __future__ import annotations

import numpy as np

from kneeai.core import LandmarkName, Side
from kneeai.landmarks import detect_landmarks_and_frame
from kneeai.ml import postprocess_mask
from kneeai.phantom import build_phantom
from kneeai.recon import reconstruct_all

for side in (Side.RIGHT, Side.LEFT):
    print(f"\n{'=' * 70}\n{side.value.upper()} KNEE\n{'=' * 70}")
    knee = build_phantom(side=side, spacing_mm=0.8)
    cleaned, _ = postprocess_mask(knee.mask)
    meshes, _ = reconstruct_all(cleaned)

    landmarks, frame = detect_landmarks_and_frame(
        meshes,
        external={
            LandmarkName.FEMORAL_HEAD_CENTER: knee.landmark(
                LandmarkName.FEMORAL_HEAD_CENTER
            ),
            LandmarkName.ANKLE_CENTER: knee.landmark(LandmarkName.ANKLE_CENTER),
        },
    )

    print(f"\npatient frame: method={frame.method}  axis_angle={frame.axis_angle_deg:.1f}deg")
    print(f"  origin    {np.round(frame.origin, 2).tolist()}")
    print(f"  right     {np.round(frame.right, 4).tolist()}")
    print(f"  anterior  {np.round(frame.anterior, 4).tolist()}")
    print(f"  superior  {np.round(frame.superior, 4).tolist()}")
    print(f"  lateral   {np.round(frame.lateral, 4).tolist()}")

    print("\ndetected vs truth (mm error):")
    errors = []
    for name in sorted(landmarks.landmarks, key=lambda n: n.value):
        lm = landmarks[name]
        try:
            truth = knee.landmark(name)
        except KeyError:
            position = str(np.round(lm.position, 1).tolist())
            print(
                f"  {name.value:<28} {position:<30} conf={lm.confidence:.2f}  "
                "(no truth defined)"
            )
            continue
        err = float(np.linalg.norm(lm.position - truth))
        if lm.origin.value != "imported":
            errors.append((name.value, err))
        print(
            f"  {name.value:<28} err={err:>6.2f} mm  conf={lm.confidence:.2f}  "
            f"unc={lm.uncertainty_mm if lm.uncertainty_mm is None else round(lm.uncertainty_mm, 2)}"
        )

    if errors:
        values = [e for _, e in errors]
        print(
            f"\n  mean error {np.mean(values):.2f} mm   median {np.median(values):.2f} mm   "
            f"max {np.max(values):.2f} mm ({max(errors, key=lambda kv: kv[1])[0]})"
        )
