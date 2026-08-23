#!/usr/bin/env python3
"""Unified pipeline script: segmentation → landmarks → planning → simulation.

Runs a single case through the entire pipeline:
    1. Load NIfTI image from the OAIZIB-CM dataset
    2. Run the existing SegResNet segmentation model (backend/)
    3. Bridge into the new kneeai pipeline
    4. Detect landmarks from reconstructed surfaces
    5. Measure anatomy
    6. Match implant configurations
    7. Assess osteoarthritis
    8. Plan resections
    9. Simulate cuts and flexion
    10. Write all artifacts to the case directory

Usage:
    python scripts/run_unified_pipeline.py oaizib_001
    python scripts/run_unified_pipeline.py oaizib_001 --compare-legacy
    python scripts/run_unified_pipeline.py oaizib_001 --strategy mechanical_alignment
    python scripts/run_unified_pipeline.py --list-cases

Output:
    artifacts/cases/<case_id>/
        result.json
        segmentation.nii.gz
        landmarks.json
        measurements.json
        implants.json
        plan.json
        simulation.json
        kinematics.json
        oa.json
        robotic_readiness.json
        meshes/
            manifest.json
            *.glb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


def list_available_cases() -> list[str]:
    """List all case IDs available in the dataset."""
    data_root = PROJECT_ROOT / "data" / "processed" / "extracted"
    cases = []

    for subdir in ("imagesTr", "imagesTs"):
        folder = data_root / subdir
        if folder.exists():
            for f in sorted(folder.glob("*_0000.nii.gz")):
                case_id = f.name.replace("_0000.nii.gz", "")
                cases.append(case_id)

    return cases


def print_stage(stage: str) -> None:
    """Pretty-print pipeline stage progress."""
    print(f"  ▸ {stage}")


def run_case(
    case_id: str,
    *,
    strategy: str = "restricted_kinematic",
    optimize: bool = True,
    compare_legacy: bool = False,
    side: str = "right",
    output_root: Path | None = None,
) -> dict:
    """Run the full unified pipeline for a single case."""
    from kneeai.bridge import BridgeConfig, run_bridge_pipeline

    config = BridgeConfig(
        model_path=PROJECT_ROOT / "models" / "segresnet_knee" / "best_model.pth",
        data_root=PROJECT_ROOT / "data" / "processed" / "extracted",
        artifact_root=output_root or (PROJECT_ROOT / "artifacts"),
        strategy=strategy,
        optimize=optimize,
        side=side,
        run_legacy_comparison=compare_legacy,
    )

    print(f"\n{'='*70}")
    print(f"  KneeAI Unified Pipeline — Case: {case_id}")
    print(f"{'='*70}")
    print(f"  Strategy: {strategy}")
    print(f"  Side: {side}")
    print(f"  Model: {config.model_path}")
    print(f"  Output: {config.artifact_root / 'cases' / case_id}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()

    result = run_bridge_pipeline(
        case_id=case_id,
        config=config,
        on_stage=print_stage,
    )

    elapsed = time.perf_counter() - t0

    # Print summary
    print(f"\n{'─'*70}")
    print(f"  ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"{'─'*70}")
    print(f"  Case directory: {result['case_dir']}")
    print(f"  Planned: {result['stages']['pipeline']['planned']}")
    print(f"  Warnings: {result['stages']['pipeline']['warnings_count']}")
    print()

    # Print timing breakdown
    print("  Timing breakdown:")
    if "timings" in result["stages"]["pipeline"]:
        for stage, seconds in result["stages"]["pipeline"]["timings"].items():
            print(f"    {stage:<20} {seconds:>6.2f}s")
    print(f"    {'TOTAL':<20} {elapsed:>6.2f}s")
    print()

    # Print artifact list
    print("  Artifacts written:")
    for name, path in sorted(result.get("artifact_paths", {}).items()):
        size = Path(path).stat().st_size if Path(path).exists() else 0
        size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
        print(f"    {name:<25} {size_str}")
    print()

    # Print warnings (first 5)
    warnings = result.get("warnings", [])
    if warnings:
        print(f"  Warnings ({len(warnings)} total):")
        for w in warnings[:5]:
            # Truncate long warnings for display
            display = w[:100] + "..." if len(w) > 100 else w
            print(f"    ⚠ {display}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more (see result.json)")
    print()

    # Save full result as JSON
    summary_path = Path(result["case_dir"]) / "pipeline_summary.json"
    # Remove non-serializable items
    serializable = {
        k: v for k, v in result.items()
        if k not in ("artifacts",)
    }
    summary_path.write_text(
        json.dumps(serializable, indent=2, default=str) + "\n"
    )
    print(f"  Full summary: {summary_path}")
    print()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KneeAI Unified Pipeline: segmentation → planning → simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_unified_pipeline.py oaizib_001
  python scripts/run_unified_pipeline.py oaizib_001 --compare-legacy
  python scripts/run_unified_pipeline.py oaizib_001 --strategy mechanical_alignment
  python scripts/run_unified_pipeline.py --list-cases
  python scripts/run_unified_pipeline.py oaizib_001 --side left
""",
    )

    parser.add_argument(
        "case_id",
        nargs="?",
        help="Case identifier (e.g. oaizib_001). Omit with --list-cases.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List all available case IDs and exit.",
    )
    parser.add_argument(
        "--strategy",
        default="restricted_kinematic",
        choices=[
            "restricted_kinematic",
            "mechanical_alignment",
            "kinematic",
            "anatomic",
        ],
        help="Alignment strategy for planning (default: restricted_kinematic).",
    )
    parser.add_argument(
        "--side",
        default="right",
        choices=["right", "left"],
        help="Operative side (default: right).",
    )
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Also run the legacy measurement/implant pipeline for comparison.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip plan optimization.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root directory (default: ./artifacts).",
    )

    args = parser.parse_args()

    if args.list_cases:
        cases = list_available_cases()
        print(f"\nAvailable cases ({len(cases)} total):\n")
        for i, case in enumerate(cases):
            print(f"  {case}", end="")
            if (i + 1) % 5 == 0:
                print()
        print()
        return

    if not args.case_id:
        parser.error("case_id is required (or use --list-cases)")

    try:
        result = run_case(
            args.case_id,
            strategy=args.strategy,
            optimize=not args.no_optimize,
            compare_legacy=args.compare_legacy,
            side=args.side,
            output_root=args.output,
        )
    except FileNotFoundError as e:
        print(f"\n  ✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n  ✗ Pipeline failed: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
