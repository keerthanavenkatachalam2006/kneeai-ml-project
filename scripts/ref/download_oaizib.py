#!/usr/bin/env python
"""Acquire OAIZIB-CM (test split) from HuggingFace, partially, via HTTP range requests.

    python scripts/download_oaizib.py --limit 40

Why partial
-----------
``imagesTs.zip`` is 2.54 GB (measured, not estimated) for 103 cases — roughly 24 MB
each. Every archive in the repository advertises ``Accept-Ranges: bytes``, so
``remotezip`` can read the ZIP central directory over HTTP and then fetch only the
member bytes for the cases we want. Downloading 40 cases costs ~1 GB instead of
2.54 GB, and the whole-archive download is never needed.

What is fetched
---------------
==================  =========  =====================================================
Archive             Action     Reason
==================  =========  =====================================================
``imagesTs.zip``    partial    2.54 GB; ``--limit N`` fetches N cases
``labelsTs.zip``    whole      14.1 MB — partial fetch would add nothing
``info.zip``        whole      51.7 KB — side, SubjectID, KL grade, demographics
``imagesTr.zip``    skipped    10.03 GB
``labelsTr.zip``    skipped    pairs with imagesTr
==================  =========  =====================================================

The upstream train split is deliberately left untouched so it remains a clean
external set. See ``docs/DATASET_ACQUISITION.md`` §1.8.

Caching
-------
Extraction is idempotent and cached on disk. Re-running with the same ``--limit`` is
a no-op. **Training never touches the network** — it reads
``data/raw/oaizib/`` only.

Licence
-------
OAIZIB-CM is CC-BY-NC-4.0: **non-commercial use only**, and citation of the
CartiMorph paper is compulsory. The script prints the citation and writes it beside
the data, so it travels with the download rather than living only in a document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO = "YongchengYAO/OAIZIB-CM"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
API_URL = f"https://huggingface.co/api/datasets/{REPO}"

DEFAULT_ROOT = Path("data/raw/oaizib")

#: Verified byte sizes (HTTP HEAD, 2026-08-22). Used to detect upstream changes.
EXPECTED_SIZES: dict[str, int] = {
    "imagesTs.zip": 2_544_663_978,
    "labelsTs.zip": 14_122_969,
    "info.zip": 51_680,
    "imagesTr.zip": 10_032_464_145,
    "labelsTr.zip": 57_538_129,
}

#: Case 454 was flagged corrupt upstream on 14-Feb-2025 and the file was *updated*
#: on 15-Feb-2025, so it is probably fine in the current revision. Excluded by
#: default anyway: losing 1 of 103 held-out cases is cheap, training on a corrupt
#: volume is not. Override with --include-454.
SUSPECT_CASES: frozenset[int] = frozenset({454})

CITATION = (
    "Yao Y, Zhong J, Zhang L, Khan S, Chen W. CartiMorph: A framework for automated\n"
    "knee articular cartilage morphometrics. Medical Image Analysis 91:103035, 2024.\n"
    "doi:10.1016/j.media.2023.103035\n\n"
    "Underlying segmentations:\n"
    "Ambellan F, Tack A, Ehlke M, Zachow S. Automated Segmentation of Knee Bone and\n"
    "Cartilage combining Statistical Shape Knowledge and Convolutional Neural\n"
    "Networks: Data from the Osteoarthritis Initiative. Medical Image Analysis, 2019.\n"
    "doi:10.1016/j.media.2018.11.009\n\n"
    "Licence: CC-BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)\n"
    "NON-COMMERCIAL USE ONLY. A model trained on this data must not ship in a\n"
    "commercial product. Citation of the CartiMorph paper is compulsory.\n"
)


# ---------------------------------------------------------------------------


@dataclass
class DownloadReport:
    root: Path
    limit: int | None
    images: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    bytes_fetched: int = 0
    checksums: dict[str, str] = field(default_factory=dict)
    kl_histogram: dict[str, int] = field(default_factory=dict)
    revision: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_id": "oaizib_cm",
            "repo": REPO,
            "revision": self.revision,
            "root": str(self.root),
            "limit": self.limit,
            "n_images": len(self.images),
            "n_labels": len(self.labels),
            "images": sorted(self.images),
            "labels": sorted(self.labels),
            "skipped": [{"name": n, "reason": r} for n, r in self.skipped],
            "bytes_fetched": self.bytes_fetched,
            "checksums_sha256": self.checksums,
            "kl_histogram": self.kl_histogram,
            "selection": "stratified by Kellgren-Lawrence grade; the upstream test split is ordered by severity so case-number order is biased towards mild OA",
            "licence": "CC-BY-NC-4.0",
            "commercial_use": False,
            "citation_required": True,
            "warnings": self.warnings,
        }


def _log(message: str) -> None:
    print(message, flush=True)


def _head(name: str) -> tuple[int, bool]:
    """Return ``(content_length, supports_ranges)`` for an archive."""
    request = urllib.request.Request(BASE_URL + name, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = int(response.headers.get("Content-Length", 0))
        ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    return length, ranges


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _fetch_whole(name: str, destination: Path, report: DownloadReport) -> Path:
    """Download a small archive in full, caching on disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = EXPECTED_SIZES.get(name)

    if destination.exists() and (expected is None or destination.stat().st_size == expected):
        _log(f"  {name}: cached ({destination.stat().st_size:,} bytes)")
    else:
        _log(f"  {name}: downloading…")
        with urllib.request.urlopen(BASE_URL + name, timeout=300) as response:
            payload = response.read()
        destination.write_bytes(payload)
        report.bytes_fetched += len(payload)
        _log(f"  {name}: {len(payload):,} bytes")

    actual = destination.stat().st_size
    if expected is not None and actual != expected:
        report.warnings.append(
            f"{name} is {actual:,} bytes but {expected:,} was recorded on "
            "2026-08-22; upstream may have changed. Re-verify the dataset."
        )
    report.checksums[name] = _sha256(destination)
    return destination


def _case_number(member: str) -> int | None:
    """Extract the case number from ``.../oaizib_454_0000.nii.gz``."""
    stem = Path(member).name
    if not stem.startswith("oaizib_"):
        return None
    token = stem[len("oaizib_") :].split("_")[0].split(".")[0]
    return int(token) if token.isdigit() else None


def _kl_grades_by_case(root: Path) -> dict[int, str]:
    """``{cmt_id: kl_key}`` for the test split, from the already-fetched info.zip.

    Returns ``{}`` if metadata is unavailable, in which case the caller falls back to
    case-number order and says so.
    """
    try:
        from kneeai.datasets.oaizib import load_subject_info

        info = load_subject_info(root)
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort here
        _log(f"  (KL metadata unavailable: {type(exc).__name__}: {exc})")
        return {}
    return {
        subject.cmt_id: (
            str(subject.kl_grade) if subject.kl_grade is not None else "ungraded"
        )
        for subject in info.values()
        if subject.upstream_split == "test"
    }


def _stratified_selection(
    available: Sequence[int], grades: Mapping[int, str], limit: int
) -> list[int]:
    """Pick ``limit`` case numbers spread proportionally across KL grades.

    Uses largest-remainder allocation so the subset's KL histogram approximates the
    full split's. Deterministic: buckets and picks are sorted, no RNG.
    """
    buckets: dict[str, list[int]] = {}
    for number in sorted(available):
        buckets.setdefault(grades.get(number, "ungraded"), []).append(number)

    total = sum(len(v) for v in buckets.values())
    if limit >= total:
        return sorted(available)

    keys = sorted(buckets)
    exact = {k: len(buckets[k]) * limit / total for k in keys}
    allocation = {k: int(exact[k]) for k in keys}

    # Distribute the rounding remainder to the largest fractional parts.
    remainder = limit - sum(allocation.values())
    for key in sorted(keys, key=lambda k: (-(exact[k] - allocation[k]), k))[:remainder]:
        allocation[key] += 1

    chosen: list[int] = []
    for key in keys:
        chosen.extend(buckets[key][: allocation[key]])
    return sorted(chosen)


def _select_members(
    names: Sequence[str],
    limit: int | None,
    include_suspect: bool,
    grades: Mapping[int, str] | None = None,
) -> tuple[list[str], list[tuple[str, str]], dict[str, int]]:
    """Choose which archive members to fetch, deterministically.

    **Selection is stratified by KL grade, not by case number.** OAIZIB-CM's test
    split is *ordered* by severity — cases 405-425 are KL0, 460-487 are KL3, 488-502
    are KL4 — so taking the first N by case number yields a subset containing only
    mild osteoarthritis. At N=40 that is KL 0-2 with **zero KL3/KL4 cases**, which is
    precisely the population a knee replacement targets. A model trained and validated
    on that subset would be evaluated on mild disease and deployed on severe.
    """
    by_number: dict[int, str] = {}
    skipped: list[tuple[str, str]] = []

    for name in names:
        if name.endswith("/") or "__MACOSX" in name:
            continue
        if not name.endswith(".nii.gz"):
            continue
        number = _case_number(name)
        if number is None:
            skipped.append((name, "filename does not carry a case number"))
            continue
        if number in SUSPECT_CASES and not include_suspect:
            skipped.append(
                (
                    name,
                    "case 454 was flagged corrupt upstream (14-Feb-2025) and the file "
                    "was updated the next day; excluded by default, use --include-454",
                )
            )
            continue
        by_number[number] = name

    if limit is None:
        selected = sorted(by_number)
    elif grades:
        selected = _stratified_selection(sorted(by_number), grades, limit)
    else:
        selected = sorted(by_number)[:limit]
        skipped.append(
            (
                "<selection>",
                "KL metadata unavailable, so selection fell back to case-number "
                "order. The test split is ordered by severity, so this subset is "
                "biased towards mild osteoarthritis.",
            )
        )

    histogram: dict[str, int] = {}
    for number in selected:
        key = (grades or {}).get(number, "unknown")
        histogram[key] = histogram.get(key, 0) + 1

    return [by_number[n] for n in selected], skipped, dict(sorted(histogram.items()))


def _extract_from_remote(
    archive: str,
    limit: int | None,
    out_dir: Path,
    report: DownloadReport,
    include_suspect: bool,
    grades: Mapping[int, str] | None = None,
) -> list[str]:
    """Fetch selected members of a remote ZIP using HTTP range requests."""
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise SystemExit(
            "remotezip is required for partial download. Install it with:\n"
            "    pip install remotezip\n"
            f"(underlying error: {exc})"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    with RemoteZip(BASE_URL + archive) as zf:
        names = zf.namelist()
        chosen, skipped, histogram = _select_members(
            names, limit, include_suspect, grades
        )
        report.skipped.extend(skipped)
        report.kl_histogram = histogram
        _log(
            f"  {archive}: {len(names)} members, selecting {len(chosen)}"
            + (f" (limit {limit})" if limit else " (all)")
        )
        _log(f"  KL-stratified selection: {histogram}")

        for member in chosen:
            target = out_dir / Path(member).name
            if target.exists() and target.stat().st_size > 0:
                written.append(target.name)
                continue
            with zf.open(member) as source:
                payload = source.read()
            target.write_bytes(payload)
            report.bytes_fetched += len(payload)
            written.append(target.name)
            _log(f"    {target.name}  {len(payload) / 1e6:.1f} MB")

    return written


def _extract_local_zip(
    archive: Path,
    out_dir: Path,
    keep: Iterable[str] | None = None,
    suffixes: tuple[str, ...] | None = (".nii.gz",),
) -> list[str]:
    """Extract members of a local ZIP, flattening the directory layout.

    ``suffixes=None`` extracts everything. ``info.zip`` needs that: it holds
    ``.csv``, ``.xlsx``, ``.md`` and ``license``, none of which match a volume
    suffix, so filtering to ``.nii.gz`` silently produced an empty ``info/``
    directory and the metadata never landed on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    allow = set(keep) if keep is not None else None
    written: list[str] = []

    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.endswith("/") or "__MACOSX" in member:
                continue
            if suffixes is not None and not member.endswith(suffixes):
                continue
            name = Path(member).name
            if allow is not None and name not in allow:
                continue
            target = out_dir / name
            if target.exists() and target.stat().st_size > 0:
                written.append(name)
                continue
            target.write_bytes(zf.read(member))
            written.append(name)
    return written


def _label_name_for(image_name: str) -> str:
    """``oaizib_405_0000.nii.gz`` -> ``oaizib_405.nii.gz``.

    nnU-Net-style image filenames carry a ``_0000`` channel suffix that label files
    do not, so the two cannot be matched by string equality.
    """
    stem = image_name[: -len(".nii.gz")]
    if stem.endswith("_0000"):
        stem = stem[: -len("_0000")]
    return f"{stem}.nii.gz"


def _verify_loadable(paths: Sequence[Path]) -> list[str]:
    """Confirm each volume actually loads. Returns human-readable failures."""
    failures: list[str] = []
    try:
        import SimpleITK as sitk  # noqa: N813
    except ImportError:
        return ["SimpleITK unavailable, so volumes were not load-verified"]

    for path in paths:
        try:
            image = sitk.ReadImage(str(path))
            if any(n <= 0 for n in image.GetSize()):
                failures.append(f"{path.name}: degenerate size {image.GetSize()}")
        except Exception as exc:  # noqa: BLE001 - untrusted archive content
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return failures


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Partially download the OAIZIB-CM test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="number of image cases to fetch (0 or negative = all 103)",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--include-454",
        action="store_true",
        help="include case 454, flagged corrupt upstream then apparently repaired",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="load every downloaded volume with SimpleITK to confirm it is readable",
    )
    args = parser.parse_args(argv)

    limit = args.limit if args.limit and args.limit > 0 else None
    root: Path = args.root
    report = DownloadReport(root=root, limit=limit)

    _log("=" * 74)
    _log("OAIZIB-CM acquisition  —  CC-BY-NC-4.0, NON-COMMERCIAL USE ONLY")
    _log("=" * 74)

    # --- confirm the repo is still ungated before fetching anything ---
    try:
        with urllib.request.urlopen(API_URL, timeout=60) as response:
            meta = json.load(response)
        report.revision = str(meta.get("sha", ""))
        if meta.get("gated"):
            _log("ABORT: upstream now reports gated=true. Do not bypass the gate.")
            return 2
        _log(f"repo reachable, gated={meta.get('gated')}, revision={report.revision[:12]}")
    except urllib.error.URLError as exc:
        _log(f"ABORT: cannot reach HuggingFace: {exc}")
        return 2

    # --- range-request support is what makes partial fetch possible ---
    length, ranges = _head("imagesTs.zip")
    _log(f"imagesTs.zip: {length / 1e9:.3f} GB, Accept-Ranges={ranges}")
    if not ranges and limit is not None:
        _log(
            "ABORT: upstream no longer advertises byte ranges, so a partial fetch "
            "would silently become a 2.54 GB whole-archive download. Re-run with "
            "--limit 0 if that is genuinely intended."
        )
        return 2

    # --- metadata and labels: small, fetched whole ---
    _log("\n[1/3] metadata and labels")
    info_zip = _fetch_whole("info.zip", root / "info.zip", report)
    labels_zip = _fetch_whole("labelsTs.zip", root / "labelsTs.zip", report)
    info_files = _extract_local_zip(info_zip, root / "info", suffixes=None)
    _log(f"  info/: {len(info_files)} file(s) — {', '.join(sorted(info_files))}")

    # --- images: partial, by case number ---
    _log(f"\n[2/3] images (partial, limit={limit or 'all'})")
    grades = _kl_grades_by_case(root)
    images = _extract_from_remote(
        "imagesTs.zip", limit, root / "imagesTs", report, args.include_454, grades
    )
    report.images = images

    # --- labels matched to the images we actually took ---
    _log("\n[3/3] matching labels")
    wanted = {_label_name_for(name) for name in images}
    report.labels = _extract_local_zip(labels_zip, root / "labelsTs", keep=wanted)
    _log(f"  extracted {len(report.labels)} labels for {len(images)} images")

    missing = wanted - set(report.labels)
    if missing:
        report.warnings.append(
            f"{len(missing)} image(s) have no matching label: {sorted(missing)[:5]}"
        )

    # --- optional load verification ---
    if args.verify:
        _log("\nverifying volumes load…")
        failures = _verify_loadable(
            [root / "imagesTs" / n for n in images]
            + [root / "labelsTs" / n for n in report.labels]
        )
        if failures:
            report.warnings.extend(failures)
            _log(f"  {len(failures)} problem(s):")
            for failure in failures[:10]:
                _log(f"    {failure}")
        else:
            _log("  all volumes loaded cleanly")

    # --- citation travels with the data ---
    (root / "CITATION.txt").write_text(CITATION)
    (root / "download_report.json").write_text(json.dumps(report.to_dict(), indent=2))

    _log("\n" + "=" * 74)
    _log(
        f"done: {len(report.images)} images, {len(report.labels)} labels, "
        f"{report.bytes_fetched / 1e6:.0f} MB fetched this run"
    )
    if report.skipped:
        _log(f"skipped {len(report.skipped)} member(s):")
        for name, reason in report.skipped[:5]:
            _log(f"  {name}: {reason}")
    for warning in report.warnings:
        _log(f"WARNING: {warning}")
    _log(f"report: {root / 'download_report.json'}")
    _log("\nCOMPULSORY CITATION:")
    _log(CITATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
