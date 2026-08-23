"""Smoke check: resection footprints and implant matching on the phantom."""

from __future__ import annotations

import json
import time

from kneeai.core import LandmarkName, Side, Structure
from kneeai.implants import (
    FemoralDimensions,
    load_catalog,
    match_configurations,
    provisional_tibial_plane,
    rank_components,
    rank_femoral_components,
    resection_footprint,
)
from kneeai.implants.catalog import ComponentType
from kneeai.landmarks import detect_landmarks_and_frame
from kneeai.measure import measure_all
from kneeai.ml import postprocess_mask
from kneeai.phantom import build_phantom
from kneeai.recon import reconstruct_all

catalog = load_catalog("builtin:reference-v0")
print("== catalog ==")
print(json.dumps(catalog.summary(), indent=2))
try:
    catalog.require_clinical_use()
    print("  UNEXPECTED: reference catalog claimed clinical use")
except Exception as exc:  # noqa: BLE001
    print(f"  clinical-use gate works: {type(exc).__name__}")

for side in (Side.RIGHT, Side.LEFT):
    print(f"\n{'=' * 78}\n{side.value.upper()} KNEE\n{'=' * 78}")
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
    measurements = measure_all(cleaned, meshes, landmarks, frame)

    femoral_dims = FemoralDimensions.from_measurements(measurements, side)
    tibial_plane = provisional_tibial_plane(meshes, landmarks, frame)
    tibial_fp = resection_footprint(
        meshes.require(Structure.TIBIA), tibial_plane, frame, provisional=True
    )

    print("\n-- patient geometry --")
    print(f"  femoral dimensions: {json.dumps(femoral_dims.to_dict())}")
    print(
        f"  tibial cut: ML {tibial_fp.ml_extent_mm:6.2f} mm  "
        f"AP {tibial_fp.ap_extent_mm:6.2f} mm  area {tibial_fp.area_mm2:8.1f} mm2  "
        f"aspect {tibial_fp.aspect_ratio:.3f}"
    )

    family = catalog.family("REF-PRIMARY-CR")

    print("\n-- femoral ranking (dimensional) --")
    for fit in rank_femoral_components(
        femoral_dims, family.by_type(ComponentType.FEMORAL, side=side), limit=5
    ):
        flag = "" if fit.acceptable else "  [VIOLATION]"
        print(
            f"  size {fit.size_label:<3} score {fit.score:6.2f}  "
            f"ML {fit.ml_mismatch_mm:+6.2f} mm ({fit.ml_overhang_per_side_mm:4.2f}/side)  "
            f"AP env {fit.ap_mismatch_mm:+6.2f} mm{flag}"
        )

    print("\n-- tibial ranking (footprint polygon) --")
    for fit in rank_components(
        tibial_fp, family.by_type(ComponentType.TIBIAL_BASEPLATE, side=side), limit=5
    ):
        flag = "" if fit.acceptable else "  [VIOLATION]"
        print(
            f"  size {fit.size_label:<3} score {fit.score:6.2f}  "
            f"cov {fit.coverage_pct:6.2f}%  over {fit.overhang_max_mm:5.2f} mm  "
            f"uncov {fit.undercoverage_max_mm:5.2f} mm  "
            f"relief {fit.relief_area_mm2:6.1f} mm2{flag}"
        )

    print("\n-- full configuration matching --")
    t0 = time.time()
    fits, provenance = match_configurations(
        femoral_dims, tibial_fp, catalog, side=side, limit=6
    )
    print(f"  ({time.time() - t0:.1f}s)")
    for fit in fits:
        flag = "" if fit.acceptable else "  [VIOLATION]"
        print(f"  {fit.configuration.label:<38} score {fit.score:6.2f}{flag}")

    print("\n-- explanation for the top candidate --")
    print("\n".join(f"  {line}" for line in fits[0].rationale()))
