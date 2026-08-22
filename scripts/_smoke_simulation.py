"""Smoke check: geometric simulation and flexion sweep on a planned phantom knee."""

from __future__ import annotations

from kneeai.core import LandmarkName, Side, Structure
from kneeai.implants import (
    FemoralDimensions,
    load_catalog,
    match_configurations,
    provisional_tibial_plane,
    resection_footprint,
)
from kneeai.landmarks import detect_landmarks_and_frame
from kneeai.measure import measure_all
from kneeai.ml import postprocess_mask
from kneeai.phantom import build_phantom
from kneeai.planning import (
    AlignmentStrategy,
    CutName,
    PlanningInputs,
    optimize_plan,
    plan_resections,
    targets_for,
)
from kneeai.recon import reconstruct_all
from kneeai.simulation import KinematicConfig, simulate_flexion, simulate_plan

side = Side.RIGHT
knee = build_phantom(side=side, spacing_mm=0.8)
cleaned, _ = postprocess_mask(knee.mask)
meshes, _ = reconstruct_all(cleaned)
landmarks, frame = detect_landmarks_and_frame(
    meshes,
    external={
        LandmarkName.FEMORAL_HEAD_CENTER: knee.landmark(LandmarkName.FEMORAL_HEAD_CENTER),
        LandmarkName.ANKLE_CENTER: knee.landmark(LandmarkName.ANKLE_CENTER),
    },
)
measurements = measure_all(cleaned, meshes, landmarks, frame)

catalog = load_catalog("builtin:reference-v0")
femoral_dims = FemoralDimensions.from_measurements(measurements, side)
tibial_plane = provisional_tibial_plane(meshes, landmarks, frame)
tibial_fp = resection_footprint(
    meshes.require(Structure.TIBIA), tibial_plane, frame, provisional=True
)
fits, _ = match_configurations(femoral_dims, tibial_fp, catalog, side=side, limit=3)
best = fits[0]

inputs = PlanningInputs(
    meshes=meshes,
    landmarks=landmarks,
    frame=frame,
    measurements=measurements,
    configuration=best.configuration,
    targets=targets_for(AlignmentStrategy.RESTRICTED_KINEMATIC),
    fit=best,
    case_id="phantom-right",
    label="Plan B",
)
plan = optimize_plan(inputs).plan
print(plan.summary())

print(f"\n{'=' * 78}\n== geometric simulation ==")
geometric = simulate_plan(plan, meshes, best.configuration, frame)
print(geometric.summary())

print(f"\n{'=' * 78}\n== flexion sweep ==")
kinematic = simulate_flexion(
    geometric,
    landmarks,
    frame,
    config=KinematicConfig(angles_deg=tuple(float(a) for a in range(0, 141, 20))),
)
print(kinematic.summary())

print(f"\n{'=' * 78}\n== deliberately broken plan (cut plane inverted) ==")
from dataclasses import replace as _replace

broken = plan.revise(actor="test", reason="inject a fault to check fail-safe behaviour")
bad_cut = broken.cut(CutName.TIBIAL)
broken.cuts[CutName.TIBIAL] = _replace(bad_cut, plane=bad_cut.plane.flipped())
broken_report = simulate_plan(broken, meshes, best.configuration, frame)
print(f"  passed: {broken_report.passed}")
for finding in broken_report.findings:
    if finding.severity.value != "info":
        print(f"  {finding}")
