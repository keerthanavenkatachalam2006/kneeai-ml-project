"""Build a case library from every scan in a dataset manifest.

Populates the dashboard's case selector so it spans the disease range instead of
showing two or three cases. OAIZIB-CM records a Kellgren-Lawrence grade per knee, so
building every case gives a natural mix of arthritic and healthy knees to compare.

**Default source is ground truth, and that is a deliberate trade.** Two options
exist per case:

``--source ground_truth`` (default)
    Runs the deterministic pipeline on the dataset's expert annotation. ~1 minute
    per case, and the OA verdict is trustworthy because the cartilage geometry is
    correct. Badged ``GROUND TRUTH`` in every view, so nothing here is passed off
    as model output.

``--source ai_prediction``
    Runs the registered segmentation model first. ~5 minutes per case on CPU, and
    with the currently active checkpoint the cartilage is over-segmented badly
    enough that arthritic knees read as healthy — see the model card's limitations.

So ground truth is how the library gets built, and the upload flow is how the model
gets exercised. Mixing them in one library is fine and intended: the badge
distinguishes them, which is exactly what contract rule 2 is for.

Usage::

    python scripts/build_all_cases.py                     # all, ground truth
    python scripts/build_all_cases.py --limit 5
    python scripts/build_all_cases.py --source ai_prediction --limit 3
    python scripts/build_all_cases.py --skip-existing      # resume a part-done run
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from kneeai.core.anatomy import Side
from kneeai.ml.contract import ArtifactSource

REPO = Path(__file__).resolve().parents[1]


def load_records(manifest: Path, *, split: str | None) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text())
    records = payload.get("records", []) or []
    if split:
        records = [r for r in records if r.get("split") == split]
    return records


def build_one(
    record: dict[str, Any],
    *,
    source: ArtifactSource,
    out_root: Path,
    optimize: bool,
) -> dict[str, Any]:
    from kneeai.api.adapter import AdapterConfig, MLInferenceAdapter
    from kneeai.api.pipeline import PipelineOptions, run_case_pipeline

    case_id = str(record["case_id"])
    side = Side(str(record.get("side", "right")).lower())

    config = AdapterConfig(
        backend=source,
        side=side,
        mask_path=REPO / str(record["label_path"]),
        image_path=REPO / str(record["image_path"]),
        model_reference="seg_oaizib",
        allow_degradation=False,
    )
    outcome = MLInferenceAdapter(config).segment()
    artifacts = run_case_pipeline(
        outcome,
        case_id=case_id,
        out_root=out_root,
        options=PipelineOptions(optimize=optimize),
    )

    oa = artifacts.oa_assessment or {}
    readiness = artifacts.readiness or {}
    return {
        "case_id": case_id,
        "kl_grade": record.get("kl_grade"),
        "split": record.get("split"),
        "seconds": artifacts.total_seconds,
        "planned": artifacts.planned,
        "oa_detected": oa.get("oa_detected"),
        "oa_probability": oa.get("probability"),
        "oa_severity": oa.get("severity"),
        "readiness_pct": readiness.get("percentage"),
        "readiness_band": readiness.get("band"),
        "n_warnings": len(artifacts.warnings),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/oaizib_60.json")
    ap.add_argument("--split", default=None, help="only this split (train/val)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--source",
        default="ground_truth",
        choices=["ground_truth", "ai_prediction"],
    )
    ap.add_argument("--out", default="artifacts", help="root; cases land in <out>/cases/")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--no-optimize",
        action="store_true",
        help="skip the plan optimiser; roughly halves the per-case time",
    )
    ap.add_argument("--report", default="artifacts/case_library.json")
    args = ap.parse_args()

    source = ArtifactSource(args.source)
    out_root = REPO / args.out
    records = load_records(REPO / args.manifest, split=args.split)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no records to build")

    print(f"building {len(records)} case(s) from {args.source}")
    print(f"output   : {out_root / 'cases'}\n", flush=True)

    built: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    started = time.perf_counter()

    for i, record in enumerate(records, 1):
        case_id = str(record["case_id"])
        target = out_root / "cases" / case_id
        if args.skip_existing and (target / "robotic_readiness.json").exists():
            print(f"[{i}/{len(records)}] {case_id}: exists, skipped", flush=True)
            continue

        try:
            summary = build_one(
                record,
                source=source,
                out_root=out_root,
                optimize=not args.no_optimize,
            )
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the batch
            failed.append(
                {"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"[{i}/{len(records)}] {case_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            continue

        built.append(summary)
        kl = summary["kl_grade"]
        truth = "OA" if (kl is not None and kl >= 2) else "no OA"
        pred = "OA" if summary["oa_detected"] else "no OA"
        agree = "OK" if truth == pred else "MISS"
        print(
            f"[{i}/{len(records)}] {case_id}: KL {kl} ({truth}) -> {pred} "
            f"p={summary['oa_probability']} | readiness {summary['readiness_pct']}% "
            f"{summary['readiness_band']} | {summary['seconds']:.0f}s {agree}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    report = {
        "source": args.source,
        "manifest": args.manifest,
        "n_built": len(built),
        "n_failed": len(failed),
        "total_seconds": round(elapsed, 1),
        "cases": built,
        "failures": failed,
    }
    out = REPO / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n=== built {len(built)}, failed {len(failed)} in {elapsed / 60:.1f} min ===")

    graded = [c for c in built if c["kl_grade"] is not None]
    if graded:
        oa_true = [c for c in graded if c["kl_grade"] >= 2]
        no_oa = [c for c in graded if c["kl_grade"] < 2]
        correct = sum(
            1 for c in graded if (c["kl_grade"] >= 2) == bool(c["oa_detected"])
        )
        print(f"library spans KL 0-4: {len(oa_true)} with OA, {len(no_oa)} without")
        print(f"OA classifier agreement with the dataset grade: {correct}/{len(graded)}")
        print("\nby KL grade:")
        for grade in range(5):
            group = [c for c in graded if c["kl_grade"] == grade]
            if not group:
                continue
            detected = sum(1 for c in group if c["oa_detected"])
            print(f"  KL {grade}: {len(group):>2} case(s), {detected} called OA")

    print(f"\nreport: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
