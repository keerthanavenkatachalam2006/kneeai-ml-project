#!/usr/bin/env python
"""Standalone OA prediction script.

Usage:
    python scripts/predict_oa.py <case_id>
    python scripts/predict_oa.py oaizib_405
    python scripts/predict_oa.py oaizib_405 --measurements path/to/measurements.json
    python scripts/predict_oa.py oaizib_405 --threshold 0.4
    python scripts/predict_oa.py oaizib_405 --run-pipeline  (runs segmentation + measurements first)

Reads the measurements.json for a given case from the artifacts directory and
produces an OA prediction using the deterministic morphometric classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def find_measurements(case_id: str) -> Path | None:
    """Locate the measurements.json for a case in standard artifact directories."""
    candidates = [
        PROJECT_ROOT / "artifacts" / "cases" / case_id / "measurements.json",
        PROJECT_ROOT / "artifacts" / case_id / "measurements.json",
        PROJECT_ROOT / "results" / case_id / "measurements.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_pipeline(case_id: str) -> Path:
    """Run the full pipeline for a case and return the measurements path.

    This requires the kneeai package and the segmentation model to be available.
    """
    try:
        from kneeai.pipeline import run as run_full_pipeline
    except ImportError:
        # Fallback: try the unified pipeline script
        import subprocess

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_unified_pipeline.py"), case_id],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"Pipeline failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(result.stdout)

    measurements_path = find_measurements(case_id)
    if measurements_path is None:
        print(f"ERROR: Pipeline completed but measurements.json not found for {case_id}", file=sys.stderr)
        sys.exit(1)
    return measurements_path


def print_prediction(prediction: dict, case_id: str, verbose: bool = False) -> None:
    """Pretty-print the OA prediction to the console."""
    print("\n" + "=" * 60)
    print(f"  OA MORPHOMETRIC ASSESSMENT — {case_id}")
    print("=" * 60)

    # Main result
    detected = prediction["oa_detected"]
    prob = prediction["probability"]
    severity = prediction["severity"]
    kl_range = prediction["approximate_kl_range"]

    status = "✓ OA DETECTED" if detected else "✗ No OA detected"
    print(f"\n  Result:      {status}")
    print(f"  Probability: {prob:.1%}")
    print(f"  Severity:    {severity}")
    print(f"  KL range:    {kl_range}")
    print(f"  Confidence:  {prediction['confidence_score']:.0%} of features available")

    # Compartment assessment
    compartments = prediction.get("compartment_assessment", {})
    if compartments:
        print(f"\n  Compartment status:")
        for comp, status in compartments.items():
            print(f"    {comp.replace('_', ' ').title():20s} → {status}")

    # Top indicators
    indicators = prediction.get("indicators", [])
    if indicators:
        print(f"\n  Key indicators (top {min(5, len(indicators))}):")
        for ind in indicators[:5]:
            print(
                f"    • {ind['finding']}"
                f"  [{ind.get('value', '?')} {ind.get('unit', '')}]"
                f"  (contribution: {ind.get('contribution', 0):.1f})"
            )

    # Features used
    if verbose:
        features = prediction.get("features_used", {})
        if features:
            print(f"\n  Features used:")
            for name, value in features.items():
                print(f"    {name}: {value}")

        missing = prediction.get("features_missing", [])
        if missing:
            print(f"\n  Features missing:")
            for name in missing:
                print(f"    • {name}")

    # Warnings
    warnings = prediction.get("warnings", [])
    if warnings:
        print(f"\n  ⚠ Warnings:")
        for w in warnings:
            print(f"    • {w}")

    # Dataset KL grade comparison
    kl_gt = prediction.get("dataset_kl_grade")
    if kl_gt is not None:
        print(f"\n  Dataset KL grade: {kl_gt} (for comparison only)")

    print(f"\n  {prediction.get('disclaimer', '')[:80]}...")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict OA from morphometric measurements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("case_id", help="Case identifier (e.g. oaizib_405)")
    parser.add_argument(
        "--measurements", "-m",
        type=Path,
        help="Path to measurements.json (auto-detected if not specified)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.35,
        help="OA detection threshold (default: 0.35)",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the full pipeline first if measurements don't exist",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Write prediction JSON to this path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed feature information",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    parser.add_argument(
        "--kl-grade",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="Ground-truth KL grade (for comparison, not used in prediction)",
    )
    args = parser.parse_args()

    # Find or generate measurements
    measurements_path = args.measurements
    if measurements_path is None:
        measurements_path = find_measurements(args.case_id)

    if measurements_path is None:
        if args.run_pipeline:
            print(f"Measurements not found for {args.case_id}. Running pipeline...")
            measurements_path = run_pipeline(args.case_id)
        else:
            print(
                f"ERROR: No measurements.json found for case '{args.case_id}'.\n"
                f"Searched in: artifacts/cases/{args.case_id}/\n\n"
                f"Options:\n"
                f"  1. Run with --run-pipeline to generate measurements first\n"
                f"  2. Specify a path with --measurements <path>\n"
                f"  3. Run the pipeline manually: python scripts/run_unified_pipeline.py {args.case_id}",
                file=sys.stderr,
            )
            sys.exit(1)

    if not measurements_path.exists():
        print(f"ERROR: File not found: {measurements_path}", file=sys.stderr)
        sys.exit(1)

    # Load measurements
    with open(measurements_path) as f:
        measurements = json.load(f)

    # Run prediction
    from kneeai.oa.morphometric_classifier import predict_oa_from_measurements

    prediction = predict_oa_from_measurements(measurements, threshold=args.threshold)
    result = prediction.to_dict()

    # Add case metadata
    result["case_id"] = args.case_id
    result["measurements_path"] = str(measurements_path)
    if args.kl_grade is not None:
        result["dataset_kl_grade"] = args.kl_grade

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_prediction(result, args.case_id, verbose=args.verbose)

    # Write to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Prediction written to: {args.output}")

    # Also write to the case artifact directory
    artifact_dir = PROJECT_ROOT / "artifacts" / "cases" / args.case_id
    if artifact_dir.exists():
        oa_path = artifact_dir / "oa_morphometric.json"
        with open(oa_path, "w") as f:
            json.dump(result, f, indent=2)
        if not args.json:
            print(f"Assessment saved to: {oa_path}")

    sys.exit(0 if not prediction.oa_detected else 1)


if __name__ == "__main__":
    main()
