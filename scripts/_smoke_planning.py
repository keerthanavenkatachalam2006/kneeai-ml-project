"""Smoke check: plan generation, optimisation, comparison and approval."""

from __future__ import annotations

import numpy as np

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
    compare_many,
    compare_plans,
    cut_plane_normal,
    optimize_plan,
    plan_resections,
    projected_angles,
    targets_for,
)
from kneeai.recon import reconstruct_all

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

# --- angle round-trip: the number planned must be the number measured ---
print("== projected-angle round trip ==")
for coronal, sagittal in ((0.0, 3.0), (3.0, 7.0), (-2.5, 5.0)):
    normal = cut_plane_normal(
        frame, coronal_deg=coronal, sagittal_deg=sagittal, coronal_towards_medial=True
    )
    back_coronal, back_sagittal = projected_angles(
        normal, frame, coronal_towards_medial=True
    )
    ok = abs(back_coronal - coronal) < 1e-9 and abs(back_sagittal - sagittal) < 1e-9
    print(
        f"  planned ({coronal:+.2f}, {sagittal:+.2f}) -> measured "
        f"({back_coronal:+.6f}, {back_sagittal:+.6f})  {'ok' if ok else 'FAIL'}"
    )

# --- implant selection ---
catalog = load_catalog("builtin:reference-v0")
femoral_dims = FemoralDimensions.from_measurements(measurements, side)
tibial_plane = provisional_tibial_plane(meshes, landmarks, frame)
tibial_fp = resection_footprint(
    meshes.require(Structure.TIBIA), tibial_plane, frame, provisional=True
)
fits, _ = match_configurations(femoral_dims, tibial_fp, catalog, side=side, limit=3)
best_fit = fits[0]
print(f"\nselected implant: {best_fit.configuration.label} (score {best_fit.score:.1f})")

# --- one plan per strategy ---
plans = []
for strategy, label in (
    (AlignmentStrategy.MECHANICAL, "Plan A (mechanical)"),
    (AlignmentStrategy.RESTRICTED_KINEMATIC, "Plan B (restricted kinematic)"),
    (AlignmentStrategy.KINEMATIC, "Plan C (kinematic)"),
):
    inputs = PlanningInputs(
        meshes=meshes,
        landmarks=landmarks,
        frame=frame,
        measurements=measurements,
        configuration=best_fit.configuration,
        targets=targets_for(strategy),
        fit=best_fit,
        case_id="phantom-right",
        label=label,
    )
    plan = plan_resections(inputs)
    plans.append((inputs, plan))
    print(f"\n{'=' * 78}")
    print(plan.summary())

# --- cut geometry sanity: planned angles must read back off the planes ---
print(f"\n{'=' * 78}\n== cut plane verification (Plan B) ==")
_, plan_b = plans[1]
tibial_cut = plan_b.cut(CutName.TIBIAL)
coronal, sagittal = projected_angles(
    -tibial_cut.plane.normal, frame, coronal_towards_medial=True
)
print(
    f"  tibial cut: planned coronal {tibial_cut.angles_deg['coronal_varus']:+.2f}° "
    f"slope {tibial_cut.angles_deg['sagittal_slope']:+.2f}°  ->  "
    f"read back {coronal:+.2f}° / {sagittal:+.2f}°"
)
distal_cut = plan_b.cut(CutName.DISTAL_FEMORAL)
fem_coronal, fem_sagittal = projected_angles(
    distal_cut.plane.normal, frame, coronal_towards_medial=False
)
print(
    f"  distal femoral: planned valgus "
    f"{distal_cut.angles_deg['coronal_valgus']:+.2f}° flexion "
    f"{distal_cut.angles_deg['sagittal_flexion']:+.2f}°  ->  "
    f"read back {fem_coronal:+.2f}° / {fem_sagittal:+.2f}°"
)
anterior = plan_b.cut(CutName.ANTERIOR_FEMORAL)
posterior = plan_b.cut(CutName.POSTERIOR_FEMORAL)
separation = abs(
    float(np.dot(anterior.plane.point - posterior.plane.point, posterior.plane.normal))
)
print(
    f"  anterior-posterior cut separation {separation:.2f} mm "
    f"(component envelope {best_fit.configuration.femoral.envelope_ap_mm:.1f} mm)"
)
for name in (CutName.ANTERIOR_CHAMFER, CutName.POSTERIOR_CHAMFER):
    chamfer = plan_b.cut(name)
    angle = chamfer.plane.angle_to(distal_cut.plane)
    print(f"  {name.display_name} is {angle:.1f}° from the distal cut")

# --- optimisation ---
print(f"\n{'=' * 78}\n== optimisation (from Plan B inputs) ==")
inputs_b = plans[1][0]
result = optimize_plan(inputs_b)
print("\n".join(f"  {line}" for line in result.explanation()))
print()
print(result.plan.summary())

# --- comparison ---
print(f"\n{'=' * 78}\n== pairwise comparison ==")
print(compare_plans(plans[0][1], plans[1][1]).summary())
print()
print(compare_plans(plans[1][1], result.plan).summary())

print(f"\n{'=' * 78}\n== multi-plan table ==")
table = compare_many([p for _, p in plans] + [result.plan])
print(f"  {'field':<32}" + "".join(f"{label[:14]:>16}" for label in table["labels"]))
for row in table["rows"]:
    if not row["significant"]:
        continue
    values = "".join(
        f"{v:>16.2f}" if v is not None else f"{'-':>16}" for v in row["values"]
    )
    print(f"  {row['field']:<32}{values}")

# --- approval workflow ---
print(f"\n{'=' * 78}\n== approval workflow ==")
chosen = result.plan
print(f"  approvable: {chosen.approvable}  violations: {len(chosen.violations)}")

chosen.adjust_cut(CutName.TIBIAL, 1.0, actor="dr-smith", reason="surgeon preference")
print(f"  after surgeon edit: origin={chosen.origin.value}, edits={len(chosen.edits)}")

if chosen.approvable:
    chosen.approve(surgeon="dr-smith", note="reviewed on 3D model")
    print(f"  approved by {chosen.approved_by}, locked={chosen.locked}")
    try:
        chosen.adjust_cut(CutName.TIBIAL, 1.0, actor="dr-smith")
        print("  UNEXPECTED: edited a locked plan")
    except Exception as exc:  # noqa: BLE001
        print(f"  lock enforced: {type(exc).__name__}")
    revised = chosen.revise(actor="dr-smith", reason="intraoperative finding")
    print(
        f"  revised to v{revised.version}: approved={revised.approved}, "
        f"locked={revised.locked}"
    )
else:
    print("  plan not approvable; violations:")
    for violation in chosen.violations:
        print(f"    - {violation}")
