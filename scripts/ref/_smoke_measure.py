"""Smoke check: measure the phantom knee and compare against analytic truth."""

from __future__ import annotations

from kneeai.core import LandmarkName, Side
from kneeai.landmarks import detect_landmarks_and_frame
from kneeai.measure import MeasurementName as M
from kneeai.measure import measure_all
from kneeai.ml import postprocess_mask
from kneeai.phantom import build_phantom
from kneeai.recon import reconstruct_all

# Map measurement -> phantom truth key, and the tolerance we expect to hold.
CHECKS: list[tuple[M, str, float]] = [
    (M.FEMORAL_ML_WIDTH, "femoral_ml_mm", 2.0),
    (M.FEMORAL_AP_SIZE, "femoral_ap_mm", 2.0),
    (M.TIBIAL_ML_WIDTH, "tibial_ml_mm", 2.5),
    (M.TIBIAL_AP_DEPTH, "tibial_ap_mm", 2.5),
    (M.POSTERIOR_TIBIAL_SLOPE, "posterior_tibial_slope_deg", 1.5),
    (M.CONDYLAR_TWIST_ANGLE, "condylar_twist_deg", 2.0),
    (M.HKA_ANGLE, "hka_varus_deg", 1.5),
    (M.FEMORAL_ASPECT_RATIO, "aspect_ratio_femur", 0.06),
    (M.FEMORAL_CARTILAGE_MEAN_THICKNESS, "femoral_cartilage_thickness_mm", 0.6),
]


def run(side: Side, cartilage_loss: float = 0.0) -> None:
    label = f"{side.value.upper()} knee, medial cartilage loss={cartilage_loss:.0%}"
    print(f"\n{'=' * 76}\n{label}\n{'=' * 76}")

    knee = build_phantom(side=side, spacing_mm=0.8, medial_cartilage_loss=cartilage_loss)
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
    measurements = measure_all(cleaned, meshes, landmarks, frame)
    print(measurements.summary())

    print("\n-- against analytic truth --")
    failures = 0
    for name, truth_key, tolerance in CHECKS:
        measurement = measurements.get(name)
        if measurement is None:
            print(f"  {name.value:<42} NOT COMPUTED")
            failures += 1
            continue
        truth = knee.truth[truth_key]
        error = abs(measurement.value - truth)
        ok = error <= tolerance
        failures += 0 if ok else 1
        print(
            f"  {name.value:<42} got {measurement.value:>9.3f}  truth {truth:>9.3f}  "
            f"err {error:>6.3f}  tol {tolerance:>5.2f}  {'ok' if ok else 'FAIL'}"
        )
    print(f"\n  {failures} check(s) outside tolerance")


run(Side.RIGHT)
run(Side.LEFT)
run(Side.RIGHT, cartilage_loss=1.0)
