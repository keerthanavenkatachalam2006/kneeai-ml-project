"""
End-to-end pipeline test:
Load case → Measure → Reconstruct 3D → Match implants → OA assessment.
"""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

from app.data_loader import OAIZIBDataLoader
from app.measurements import KneeMeasurementEngine
from app.reconstruction import MeshReconstructor
from app.implant_database import ImplantMatcher
from app.oa_research import OAResearchEngine

print("=" * 60)
print("KneeAI Pipeline Test")
print("=" * 60)

# Step 1: Load data
print("\n[1/5] Loading dataset metadata...")
loader = OAIZIBDataLoader()
summary = loader.get_dataset_summary()
print(f"  Total cases: {summary['total_cases']}")
print(f"  Train: {summary['train_cases']}, Test: {summary['test_cases']}")
print(f"  KL grades: {summary['kl_grade_distribution']}")
print(f"  Gender: {summary['gender_distribution']}")
print(f"  Age range: {summary['age_stats']['min']}-{summary['age_stats']['max']}")

# Step 2: Load a specific case
print("\n[2/5] Loading case 405...")
case = loader.load_case(405)
print(f"  Image shape: {case['shape']}")
print(f"  Voxel size (mm): {case['voxel_size']}")
meta = case['metadata']
print(f"  Patient: Age={meta['Age']}, Gender={meta['Gender_Label']}, BMI={meta['BMI']}")
print(f"  KL Grade: {meta['KLGrade']}")

# Step 3: Measurements
print("\n[3/5] Computing anatomical measurements...")
engine = KneeMeasurementEngine()
measurements = engine.measure_case(case['label'], case['voxel_size'])

if 'femur' in measurements:
    f = measurements['femur']
    print(f"  Femur volume: {f['volume_cm3']:.1f} cm3")
    if 'condyle_ml_width_mm' in f:
        print(f"  Femur condyle ML width: {f['condyle_ml_width_mm']:.1f} mm")
        print(f"  Femur condyle AP depth: {f['condyle_ap_depth_mm']:.1f} mm")

if 'tibia' in measurements:
    t = measurements['tibia']
    print(f"  Tibia volume: {t['volume_cm3']:.1f} cm3")
    if 'plateau_ml_width_mm' in t:
        print(f"  Tibia plateau ML width: {t['plateau_ml_width_mm']:.1f} mm")
        print(f"  Tibia plateau AP depth: {t['plateau_ap_depth_mm']:.1f} mm")

if 'femoral_cartilage' in measurements:
    fc = measurements['femoral_cartilage']
    print(f"  Femoral cartilage thickness: {fc['thickness_mean_mm']:.2f} mm (mean)")
    print(f"  Femoral cartilage volume: {fc['volume_cm3']:.3f} cm3")

if 'medial_tibial_cartilage' in measurements:
    mtc = measurements['medial_tibial_cartilage']
    print(f"  Medial tibial cartilage thickness: {mtc['thickness_mean_mm']:.2f} mm (mean)")

if 'lateral_tibial_cartilage' in measurements:
    ltc = measurements['lateral_tibial_cartilage']
    print(f"  Lateral tibial cartilage thickness: {ltc['thickness_mean_mm']:.2f} mm (mean)")

if 'derived' in measurements:
    d = measurements['derived']
    if 'medial_lateral_thickness_ratio' in d:
        print(f"  M/L thickness ratio: {d['medial_lateral_thickness_ratio']:.2f}")
    print(f"  Total cartilage volume: {d['total_cartilage_volume_cm3']:.3f} cm3")

# Step 4: Implant matching
print("\n[4/5] Matching to implant database...")
matcher = ImplantMatcher()
matches = matcher.match_all(measurements, top_n=3)

for component, match_list in matches.items():
    print(f"\n  {component.upper()} implant matches:")
    for i, m in enumerate(match_list, 1):
        imp = m.implant
        print(f"    #{i}: {imp.manufacturer} {imp.system} Size {imp.size} "
              f"(Score: {m.score:.0f}%, {m.fit_category})")
        print(f"        ML: {m.ml_mismatch_mm:+.1f}mm, AP: {m.ap_mismatch_mm:+.1f}mm")

# Step 5: 3D Reconstruction
print("\n[5/5] Generating 3D mesh...")
reconstructor = MeshReconstructor(smoothing_sigma=1.0, step_size=2)
result = reconstructor.export_case(405, case['label'], case['voxel_size'])
print(f"  GLB file: {result['glb_path']}")
print(f"  GLB size: {result['glb_size_bytes'] / 1024:.0f} KB")
print(f"  Structures reconstructed: {result['num_structures']}")
for name, stats in result['structures'].items():
    print(f"    {name}: {stats['vertices']} vertices, {stats['faces']} faces")
    if stats['volume_mm3']:
        print(f"      Volume: {stats['volume_mm3']/1000:.2f} cm3, Area: {stats['surface_area_mm2']:.0f} mm2")

# Step 6: OA Assessment
print("\n[BONUS] OA Assessment...")
oa_engine = OAResearchEngine()
assessment = oa_engine.estimate_kl_grade(measurements)
print(f"  Estimated KL: {assessment['estimated_kl_grade']} - {assessment['description']}")
for ind in assessment['indicators']:
    print(f"    - {ind}")

# Reference comparison
ref = oa_engine.get_reference_comparison(
    measurements,
    patient_age=meta['Age'],
    patient_gender=meta['Gender_Label'],
    patient_kl=meta['KLGrade'],
)
if 'comparisons' in ref:
    print("\n  Reference comparison:")
    for key, comp in ref['comparisons'].items():
        print(f"    {comp.get('label', key)}: {comp.get('patient_mm', 'N/A')} mm")
        if 'interpretation' in comp:
            print(f"      -> {comp['interpretation']}")

print("\n" + "=" * 60)
print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
print("=" * 60)
