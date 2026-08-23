"""Add ``oa.json`` and ``robotic_readiness.json`` to cases built before they existed.

Both stages are pure functions of artefacts that are already on disk — the OA
assessment needs only ``segmentation.nii.gz`` plus the confidence block from
``result.json``, and the readiness score reads the measurements, implants, plan and
simulation. Neither needs the network. So a case produced by an earlier version of
the pipeline can be brought up to date in seconds instead of re-running a 4-minute
segmentation.

This is a backfill, not a substitute for the pipeline: it will not invent a plan for
a case that has none, and it does not touch ``result.json``. A case missing the
upstream artefacts gets an assessment that honestly reports what was unavailable.

Usage::

    python scripts/backfill_case.py                    # every case it can find
    python scripts/backfill_case.py --case oaizib-417   # just one
    python scripts/backfill_case.py --force             # overwrite existing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kneeai.config import get_settings
from kneeai.core.anatomy import Structure

REPO = Path(__file__).resolve().parents[1]

LABEL_MAP = {
    Structure.BACKGROUND: 0,
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
}


def case_roots() -> list[Path]:
    root = Path(get_settings().artifact_root)
    return [root / "cases", root / "demo" / "cases", root / "live" / "cases"]


def discover(case: str | None) -> list[Path]:
    """Case directories, first root winning on a duplicate id.

    Deduplicated by name to match how ``api.main._case_dir`` resolves a case: the
    roots are a search path, not a union, so backfilling a shadowed directory would
    write artefacts the API will never serve.
    """
    found: dict[str, Path] = {}
    for base in case_roots():
        if not base.is_dir():
            continue
        for directory in sorted(base.iterdir()):
            if not (directory / "result.json").exists():
                continue
            if case and directory.name != case:
                continue
            found.setdefault(directory.name, directory)
    return list(found.values())


def read(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")


def dataset_kl_grade(case_id: str) -> int | None:
    manifests = Path(get_settings().data_root) / "manifests"
    if not manifests.is_dir():
        return None
    for path in sorted(manifests.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        content = read(path)
        if not content:
            continue
        for record in content.get("records", []) or []:
            if record.get("case_id") == case_id:
                grade = record.get("kl_grade")
                return int(grade) if grade is not None else None
    return None


def backfill(directory: Path, *, force: bool) -> dict[str, Any]:
    from kneeai.imaging.io import read_volume
    from kneeai.oa import assess_oa
    from kneeai.robotics import build_safety_report
    from kneeai.robotics.readiness import assess_readiness

    case_id = directory.name
    result = read(directory / "result.json") or {}
    report: dict[str, Any] = {"case_id": case_id, "oa": "skipped", "readiness": "skipped"}

    # --- OA ---
    oa_path = directory / "oa.json"
    if oa_path.exists() and not force:
        report["oa"] = "already present"
    else:
        segmentation = directory / "segmentation.nii.gz"
        if not segmentation.exists():
            report["oa"] = "no segmentation.nii.gz"
        else:
            mask = read_volume(segmentation, is_label=True, label_map=LABEL_MAP)
            assessment = assess_oa(
                mask,
                case_id=case_id,
                dataset_kl_grade=dataset_kl_grade(case_id),
                segmentation_confidence=(result.get("confidence") or {}).get("per_structure"),
            )
            payload = assessment.to_dict()
            write(oa_path, payload)
            report["oa"] = (
                f"{'OA' if payload['oa_detected'] else 'no OA'} "
                f"p={payload['probability']} severity={payload['severity']}"
                if payload["available"]
                else f"unavailable: {(payload['unavailable_reason'] or '')[:60]}"
            )

    # --- readiness ---
    readiness_path = directory / "robotic_readiness.json"
    if readiness_path.exists() and not force:
        report["readiness"] = "already present"
    else:
        plan = read(directory / "plan.json")
        simulation = read(directory / "simulation.json")
        measurements = read(directory / "measurements.json")
        implants = read(directory / "implants.json")
        landmarks = read(directory / "landmarks.json")
        frame = read(directory / "frame.json")
        mesh_quality = read(directory / "mesh_quality.json")

        safety = None
        if plan:
            safety = build_safety_report(
                plan=plan,
                simulation=simulation,
                measurements=measurements,
                source=str(result.get("source", "unknown")),
            ).to_dict()

        readiness = assess_readiness(
            result=result,
            plan=plan,
            simulation=simulation,
            measurements=measurements,
            implants=implants,
            landmarks=landmarks,
            frame=frame,
            mesh_quality=mesh_quality,
            safety_report=safety,
            oa=read(oa_path),
        )
        payload = readiness.to_dict()
        write(readiness_path, payload)
        report["readiness"] = (
            f"{payload['percentage']}% {payload['band']} "
            f"({payload['n_blockers']} blocker(s))"
        )

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="one case id; default is all")
    ap.add_argument("--force", action="store_true", help="overwrite existing artifacts")
    args = ap.parse_args()

    directories = discover(args.case)
    if not directories:
        raise SystemExit(f"no cases found{f' matching {args.case!r}' if args.case else ''}")

    print(f"{len(directories)} case(s)\n")
    for directory in directories:
        try:
            report = backfill(directory, force=args.force)
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the rest
            print(f"{directory.name:24s} FAILED  {type(exc).__name__}: {exc}")
            continue
        print(f"{report['case_id']:24s} oa: {report['oa']}")
        print(f"{'':24s} readiness: {report['readiness']}")


if __name__ == "__main__":
    main()
