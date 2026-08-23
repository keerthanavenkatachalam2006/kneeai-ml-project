"""
Dataset validation script.
Validates OAIZIB-CM cases for data integrity, excluding known corrupted case 454.
"""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

from app.data_loader import OAIZIBDataLoader

loader = OAIZIBDataLoader()
metadata = loader.load_metadata()

print("OAIZIB-CM Dataset Validation")
print("=" * 50)
print(f"Total cases (excluding corrupted): {len(metadata)}")
print(f"Train: {len(metadata[metadata['split'] == 'train'])}")
print(f"Test: {len(metadata[metadata['split'] == 'test'])}")
print(f"Excluded cases: {loader.info.corrupted_cases}")
print()

# Validate a sample of test cases
test_cases = metadata[metadata['split'] == 'test']['CMT-ID'].tolist()
sample = test_cases[:5]

print(f"Validating {len(sample)} sample cases...")
for case_id in sample:
    result = loader.validate_case(case_id)
    status = "PASS" if result.get("valid") else "FAIL"
    print(f"  Case {case_id}: {status}")
    if result.get("errors"):
        for err in result["errors"]:
            print(f"    ERROR: {err}")
    elif result.get("valid"):
        print(f"    Shape: {result['image_shape']}, Labels: {result['label_values']}")

print("\nValidation complete.")
