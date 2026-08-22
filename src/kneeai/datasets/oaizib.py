"""OAIZIB-CM dataset bridge (PRD §43, §61, §63, §113).

Turns the files that ``scripts/download_oaizib.py`` put on disk into
:class:`kneeai.ml.datasets.CaseRecord` objects, with patient-level splits and a
:class:`kneeai.ml.registry.DatasetContract`.

Torch-free: it reads NIfTI headers and spreadsheets, nothing more. Manifests can
therefore be built and audited on a laptop, which matters because training happens
on Colab and the manifest is the thing that has to be reviewed before it starts.

Three decisions worth stating
-----------------------------
**The label map is asserted, never remapped.** OAIZIB-CM's map is already identical
to :data:`kneeai.core.anatomy.DEFAULT_LABEL_MAP` labels 1–5 (verified — see
``docs/DATASET_ACQUISITION.md`` §1.4). A remap would be a silent relabelling; an
assertion fails loudly if upstream ever changes.

**Grouping is by ``SubjectID``, not filename.** In this dataset there is exactly one
knee per subject, so case-level and patient-level splitting happen to coincide.
Relying on that coincidence would break silently the day the dataset gained a second
visit per subject (PRD §63, §115), so the correct key is used regardless.

**KL grade is carried as metadata, never as a target.** It reaches
``CaseRecord.kl_grade`` so the application can display real OA severity instead of a
heuristic. It is not fed to the segmentation model: only ``image`` and ``label`` keys
reach the MONAI transforms.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kneeai.core.anatomy import DEFAULT_LABEL_MAP, Modality, Side, Structure
from kneeai.core.errors import DatasetError
from kneeai.logging import get_logger
from kneeai.ml.datasets import CaseRecord, verify_no_leakage, write_manifest
from kneeai.ml.registry import DatasetContract

__all__ = [
    "DATASET_ID",
    "OAIZIB_LABEL_MAP",
    "SubjectInfo",
    "assert_label_map_matches_canonical",
    "build_dataset_contract",
    "build_manifest",
    "load_subject_info",
    "read_knee_sides",
    "scan_cases",
    "split_patient_level",
]

log = get_logger("datasets.oaizib")

DATASET_ID = "oaizib_cm"

#: OAIZIB-CM's label map, quoted from the upstream README. Kept as a literal so the
#: assertion below compares two independently-written maps rather than comparing
#: DEFAULT_LABEL_MAP with itself.
OAIZIB_LABEL_MAP: dict[Structure, int] = {
    Structure.FEMUR: 1,
    Structure.FEMORAL_CARTILAGE: 2,
    Structure.TIBIA: 3,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 4,
    Structure.LATERAL_TIBIAL_CARTILAGE: 5,
}

#: Structures the clinical pipeline wants that this dataset cannot supply.
#: Reported in ``result.json`` so the application can distinguish "the model looked
#: and found nothing" from "no model has ever looked".
UNAVAILABLE_STRUCTURES: tuple[Structure, ...] = (
    Structure.MEDIAL_MENISCUS,
    Structure.LATERAL_MENISCUS,
    Structure.PATELLA,
    Structure.PATELLAR_CARTILAGE,
)

_CASE_RE = re.compile(r"oaizib_(\d+)")


# ---------------------------------------------------------------------------
# Label map contract
# ---------------------------------------------------------------------------


def assert_label_map_matches_canonical() -> dict[Structure, int]:
    """Assert OAIZIB-CM's label map equals ``DEFAULT_LABEL_MAP`` labels 1-5.

    Raises:
        DatasetError: on any divergence, naming the exact disagreement.
    """
    canonical = {
        structure: value
        for structure, value in DEFAULT_LABEL_MAP.items()
        if 1 <= value <= 5
    }
    if OAIZIB_LABEL_MAP != canonical:
        only_dataset = {
            s.value: v for s, v in OAIZIB_LABEL_MAP.items() if canonical.get(s) != v
        }
        only_canonical = {
            s.value: v for s, v in canonical.items() if OAIZIB_LABEL_MAP.get(s) != v
        }
        raise DatasetError(
            "OAIZIB-CM label map no longer matches core.anatomy.DEFAULT_LABEL_MAP.\n"
            f"  dataset says   : {only_dataset}\n"
            f"  canonical says : {only_canonical}\n"
            "These must be identical. Do NOT remap — a remap silently relabels "
            "anatomy. Reconcile the canonical map and the model card instead."
        )
    return dict(OAIZIB_LABEL_MAP)


def verify_mask_labels(
    label_path: str | os.PathLike[str], *, allow_missing: bool = True
) -> list[int]:
    """Read the label values actually present in a mask.

    Verifies the file rather than trusting the README. ``allow_missing=True``
    tolerates a structure being absent from an individual case — legitimate when a
    compartment has worn through — but any value *outside* the declared map is an
    error, because it means the mask and the label map disagree.
    """
    import SimpleITK as sitk  # noqa: N813
    import numpy as np

    array = sitk.GetArrayViewFromImage(sitk.ReadImage(str(label_path)))
    present = sorted(int(v) for v in np.unique(array))
    declared = {0, *OAIZIB_LABEL_MAP.values()}
    unexpected = [v for v in present if v not in declared]
    if unexpected:
        raise DatasetError(
            f"{Path(label_path).name} contains label value(s) {unexpected} that are "
            f"not in the OAIZIB-CM label map {sorted(declared)}"
        )
    if not allow_missing:
        missing = sorted(declared - set(present))
        if missing:
            raise DatasetError(
                f"{Path(label_path).name} is missing label(s) {missing}"
            )
    return present


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectInfo:
    """One row of ``subInfo_train.xlsx`` / ``subInfo_test.xlsx``."""

    subject_id: str
    cmt_id: int
    upstream_split: str
    knee_side: Side
    kl_grade: int | None = None
    sex: str = ""
    age: float | None = None
    bmi: float | None = None
    path: str = ""

    @property
    def case_stem(self) -> str:
        return f"oaizib_{self.cmt_id:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "cmt_id": self.cmt_id,
            "upstream_split": self.upstream_split,
            "knee_side": self.knee_side.value,
            "kl_grade": self.kl_grade,
            "sex": self.sex,
            "age": self.age,
            "bmi": self.bmi,
        }


def _column_index(reference: str) -> int:
    """``"C12"`` -> ``2``."""
    letters = re.match(r"([A-Z]+)", reference)
    if letters is None:
        return 0
    total = 0
    for character in letters.group(1):
        total = total * 26 + (ord(character) - 64)
    return total - 1


def _read_xlsx(payload: bytes) -> list[dict[str, str]]:
    """Minimal xlsx reader that honours cell references.

    An xlsx worksheet omits empty cells entirely. Reading cells in document order
    without consulting each cell's ``r`` attribute shifts every value after a blank
    one column to the left, which made ``Gender`` appear to contain ages and looked
    like upstream data corruption. It was a parsing bug. Cell references are resolved
    here so blanks stay blank.

    Deliberately dependency-free: adding openpyxl to read two 40 KB files would put a
    dependency in the training path for no benefit.
    """
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    book = zipfile.ZipFile(io.BytesIO(payload))

    shared: list[str] = []
    if "xl/sharedStrings.xml" in book.namelist():
        root = ET.fromstring(book.read("xl/sharedStrings.xml"))
        for item in root.findall("m:si", namespace):
            shared.append(
                "".join(
                    node.text or ""
                    for node in item.iter("{%s}t" % namespace["m"])
                )
            )

    sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", namespace):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", namespace):
            reference = cell.get("r") or ""
            value_node = cell.find("m:v", namespace)
            raw = value_node.text if value_node is not None else ""
            if cell.get("t") == "s" and raw:
                raw = shared[int(raw)]
            cells[_column_index(reference)] = raw or ""
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i, "") for i in range(width)])

    if not rows:
        return []
    header = rows[0]
    return [
        dict(zip(header, row + [""] * (len(header) - len(row)))) for row in rows[1:]
    ]


def _to_int(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sex_from_code(code: str) -> str:
    """OAI encodes sex as 1/2. Mapped to a label, or passed through if unrecognised."""
    return {"1": "male", "2": "female"}.get((code or "").strip(), "")


def load_subject_info(root: str | os.PathLike[str]) -> dict[int, SubjectInfo]:
    """Load per-subject metadata, keyed by CMT-ID (the number in the filename).

    Reads the extracted ``info/`` directory, falling back to ``info.zip``.
    """
    base = Path(root)
    sources: list[tuple[str, bytes]] = []

    info_dir = base / "info"
    for split, filename in (("train", "subInfo_train.xlsx"), ("test", "subInfo_test.xlsx")):
        candidate = info_dir / filename
        if candidate.exists():
            sources.append((split, candidate.read_bytes()))

    if not sources:
        archive = base / "info.zip"
        if not archive.exists():
            raise DatasetError(
                f"no OAIZIB-CM metadata found under {base}. Expected info/ or "
                "info.zip — run scripts/download_oaizib.py first."
            )
        with zipfile.ZipFile(archive) as book:
            for split, filename in (
                ("train", "subInfo_train.xlsx"),
                ("test", "subInfo_test.xlsx"),
            ):
                member = next(
                    (n for n in book.namelist() if n.endswith(filename)), None
                )
                if member:
                    sources.append((split, book.read(member)))

    sides = read_knee_sides(base)
    out: dict[int, SubjectInfo] = {}
    for split, payload in sources:
        for row in _read_xlsx(payload):
            cmt = _to_int(row.get("CMT-ID", ""))
            subject = (row.get("SubjectID") or "").strip()
            if cmt is None or not subject:
                continue
            stem = f"oaizib_{cmt:03d}.nii.gz"
            # Prefer the text side from kneeSideInfo.csv; the spreadsheet's numeric
            # KneeSide is a code whose mapping is not documented upstream.
            side = sides.get(stem, Side.UNKNOWN)
            if side is Side.UNKNOWN:
                side = {"1": Side.RIGHT, "2": Side.LEFT}.get(
                    (row.get("KneeSide") or "").strip(), Side.UNKNOWN
                )
            out[cmt] = SubjectInfo(
                subject_id=subject,
                cmt_id=cmt,
                upstream_split=split,
                knee_side=side,
                kl_grade=_to_int(row.get("KLGrade", "")),
                sex=_sex_from_code(row.get("Gender", "")),
                age=_to_float(row.get("Age", "")),
                bmi=_to_float(row.get("BMI", "")),
                path=(row.get("Path") or "").strip(),
            )

    if not out:
        raise DatasetError(f"OAIZIB-CM metadata under {base} contained no usable rows")
    log.info(
        "OAIZIB-CM subject metadata loaded",
        fields={
            "subjects": len(out),
            "with_kl_grade": sum(1 for s in out.values() if s.kl_grade is not None),
            "sides": sorted({s.knee_side.value for s in out.values()}),
        },
    )
    return out


def read_knee_sides(root: str | os.PathLike[str]) -> dict[str, Side]:
    """Parse ``kneeSideInfo.csv`` -> ``{"oaizib_001.nii.gz": Side.RIGHT}``."""
    base = Path(root)
    text: str | None = None

    candidate = base / "info" / "kneeSideInfo.csv"
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8-sig")
    else:
        archive = base / "info.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as book:
                member = next(
                    (n for n in book.namelist() if n.endswith("kneeSideInfo.csv")), None
                )
                if member:
                    text = book.read(member).decode("utf-8-sig")
    if text is None:
        log.warning("kneeSideInfo.csv not found; laterality will fall back to KneeSide")
        return {}

    sides: dict[str, Side] = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        name = row[0].strip()
        value = row[1].strip().lower()
        if value in ("right", "r"):
            sides[name] = Side.RIGHT
        elif value in ("left", "l"):
            sides[name] = Side.LEFT
    return sides


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


def _case_number(name: str) -> int | None:
    match = _CASE_RE.search(Path(name).name)
    return int(match.group(1)) if match else None


def scan_cases(
    root: str | os.PathLike[str] = "data/raw/oaizib",
    *,
    read_geometry: bool = True,
) -> list[CaseRecord]:
    """Walk the downloaded OAIZIB-CM tree and build unsplit :class:`CaseRecord`s.

    Only cases with **both** an image and a matching label are returned; an
    image-only case would train against nothing.
    """
    assert_label_map_matches_canonical()

    base = Path(root)
    images_dir = base / "imagesTs"
    labels_dir = base / "labelsTs"
    if not images_dir.is_dir():
        raise DatasetError(
            f"{images_dir} does not exist. Run scripts/download_oaizib.py first."
        )

    info = load_subject_info(base)
    structures = tuple(s.value for s in OAIZIB_LABEL_MAP)

    records: list[CaseRecord] = []
    skipped: list[tuple[str, str]] = []

    for image_path in sorted(images_dir.glob("*.nii.gz")):
        number = _case_number(image_path.name)
        if number is None:
            skipped.append((image_path.name, "no case number in filename"))
            continue

        label_path = labels_dir / f"oaizib_{number:03d}.nii.gz"
        if not label_path.exists():
            skipped.append((image_path.name, "no matching label volume"))
            continue

        subject = info.get(number)
        if subject is None:
            # Without a SubjectID there is no valid grouping key, and a case that
            # cannot be grouped cannot be split safely.
            skipped.append(
                (image_path.name, f"case {number} absent from subInfo spreadsheets")
            )
            continue

        spacing: tuple[float, float, float] | None = None
        if read_geometry:
            try:
                import SimpleITK as sitk  # noqa: N813

                reader = sitk.ImageFileReader()
                reader.SetFileName(str(image_path))
                reader.ReadImageInformation()
                spacing = tuple(round(float(v), 6) for v in reader.GetSpacing())  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001
                skipped.append((image_path.name, f"unreadable header: {exc}"))
                continue

        records.append(
            CaseRecord(
                case_id=f"oaizib-{number:03d}",
                patient_id=subject.subject_id,
                image_path=str(image_path),
                label_path=str(label_path),
                dataset_id=DATASET_ID,
                modality=Modality.MR,
                side=subject.knee_side,
                split="unassigned",
                kl_grade=subject.kl_grade,
                age=subject.age,
                sex=subject.sex,
                bmi=subject.bmi,
                spacing_mm=spacing,
                structures=structures,
                extra={
                    "cmt_id": subject.cmt_id,
                    "upstream_split": subject.upstream_split,
                    "oai_path": subject.path,
                    # Recorded per case so the application never has to infer why a
                    # meniscus measurement is absent.
                    "structures_unavailable": [s.value for s in UNAVAILABLE_STRUCTURES],
                },
            )
        )

    if not records:
        raise DatasetError(
            f"no usable OAIZIB-CM cases under {base}. Skipped: {skipped[:5]}"
        )

    log.info(
        "OAIZIB-CM cases scanned",
        fields={
            "cases": len(records),
            "patients": len({r.patient_id for r in records}),
            "skipped": len(skipped),
            "sides": sorted({r.side.value for r in records}),
        },
    )
    if skipped:
        for name, reason in skipped:
            log.warning("case skipped", fields={"case": name, "reason": reason})
    return records


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_patient_level(
    records: Sequence[CaseRecord],
    *,
    val_fraction: float = 0.25,
    seed: int = 42,
    stratify_by_kl: bool = True,
) -> dict[str, list[CaseRecord]]:
    """Split into **train/val only**, grouping by ``patient_id``.

    No test split is produced. The available pool is upstream's *test* split, and
    upstream's ``imagesTr`` is held out entirely as a future clean external set
    (``docs/DATASET_ACQUISITION.md`` §1.8). Carving a third split out of ≤40 cases
    would leave a test set too small to mean anything, and the instruction is
    explicit that upstream's test material must never be used for model selection.

    ``stratify_by_kl`` balances KL grade across the split. At this sample size an
    unstratified draw can easily put most of the severe-OA cases on one side, and
    cartilage is exactly what degrades with KL grade — so the validation score would
    reflect the draw as much as the model.
    """
    import numpy as np

    if not 0.0 < val_fraction < 1.0:
        raise DatasetError(f"val_fraction must be in (0, 1), got {val_fraction}")

    by_patient: dict[str, list[CaseRecord]] = {}
    for record in records:
        by_patient.setdefault(record.patient_id, []).append(record)

    patients = sorted(by_patient)
    rng = np.random.default_rng(seed)

    if stratify_by_kl:
        # Bucket each patient by the highest KL grade among their cases, then take
        # the validation fraction from within each bucket.
        buckets: dict[str, list[str]] = {}
        for patient in patients:
            grades = [
                r.kl_grade for r in by_patient[patient] if r.kl_grade is not None
            ]
            key = str(max(grades)) if grades else "unknown"
            buckets.setdefault(key, []).append(patient)
        val_patients: list[str] = []
        for key in sorted(buckets):
            group = sorted(buckets[key])
            rng.shuffle(group)
            take = max(1, int(round(len(group) * val_fraction))) if group else 0
            # Never consume an entire bucket: that would leave a KL stratum absent
            # from training altogether.
            take = min(take, max(0, len(group) - 1))
            val_patients.extend(group[:take])
        strategy = "patient_level_stratified_by_kl_grade"
    else:
        shuffled = list(patients)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val_patients = shuffled[:n_val]
        strategy = "patient_level_random"

    val_set = set(val_patients)
    splits: dict[str, list[CaseRecord]] = {"train": [], "val": []}
    for patient in patients:
        target = "val" if patient in val_set else "train"
        for record in by_patient[patient]:
            splits[target].append(_with_split(record, target))

    if not splits["train"] or not splits["val"]:
        raise DatasetError(
            f"split produced an empty side: train={len(splits['train'])}, "
            f"val={len(splits['val'])}. Too few patients for val_fraction="
            f"{val_fraction}."
        )

    # Fails loudly on any patient appearing in both sides (PRD §63, §115).
    summary = verify_no_leakage(splits)
    log.info(
        "OAIZIB-CM patient-level split",
        fields={
            "strategy": strategy,
            "train_cases": len(splits["train"]),
            "val_cases": len(splits["val"]),
            "train_patients": summary["splits"]["train"]["patients"],
            "val_patients": summary["splits"]["val"]["patients"],
            "kl_train": _kl_histogram(splits["train"]),
            "kl_val": _kl_histogram(splits["val"]),
        },
    )
    return splits


def _with_split(record: CaseRecord, split: str) -> CaseRecord:
    from dataclasses import replace

    return replace(record, split=split)


def _kl_histogram(records: Iterable[CaseRecord]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for record in records:
        key = "unknown" if record.kl_grade is None else str(record.kl_grade)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items()))


# ---------------------------------------------------------------------------
# Manifest + contract
# ---------------------------------------------------------------------------


def build_dataset_contract(
    splits: Mapping[str, Sequence[CaseRecord]],
    *,
    split_strategy: str = "patient_level_stratified_by_kl_grade",
) -> DatasetContract:
    """Build the :class:`DatasetContract` PRD §113 requires before training.

    Everything here is recorded because it is unrecoverable later: nobody
    reconstructs "who annotated this" six months after the fact.
    """
    all_records = [r for records in splits.values() for r in records]
    patients = {r.patient_id for r in all_records}
    sides = sorted({r.side.value for r in all_records})

    return DatasetContract(
        input_description=(
            "3D sagittal DESS knee MR from the Osteoarthritis Initiative, single "
            "channel. Native voxel spacing 0.7 x 0.3646 x 0.3646 mm, 384 x 384 x 160, "
            "measured from the downloaded headers rather than assumed."
        ),
        output_description=(
            "6-class voxel label map: background plus femur, femoral cartilage, "
            "tibia, medial tibial cartilage, lateral tibial cartilage "
            "(core.anatomy.DEFAULT_LABEL_MAP labels 1-5)."
        ),
        ground_truth_description=(
            "Expert manual segmentations from the OAIZIB dataset (Ambellan et al., "
            "Medical Image Analysis 2019), with tibial cartilage subsequently split "
            "into medial and lateral compartments by the CartiMorph authors "
            "(Yao et al., Medical Image Analysis 91:103035, 2024). No labels were "
            "produced or modified by this project."
        ),
        annotator=(
            "OAIZIB expert manual segmentation (Zuse Institute Berlin); "
            "medial/lateral tibial cartilage division by the CartiMorph authors"
        ),
        dataset_ids=(DATASET_ID,),
        n_patients=len(patients),
        n_studies=len(all_records),
        split_patients={
            name: len({r.patient_id for r in records})
            for name, records in splits.items()
        },
        split_strategy=split_strategy,
        license="CC-BY-NC-4.0",
        commercial_use_permitted=False,
        external_test_dataset_ids=(),
        population_notes=(
            f"Laterality: {sides} only — this dataset contains 507 right knees and "
            "ZERO left knees, so the model has never seen a native left knee. "
            "Training applies sagittal mirror augmentation and inference "
            "canonicalises left knees to right, but performance on native left knees "
            "is UNMEASURED. "
            "OA severity spans Kellgren-Lawrence 0-4 across the full 507-case "
            "dataset (KL0:103, KL1:58, KL2:108, KL3:139, KL4:73, 26 ungraded), "
            "weighted towards moderate-to-severe disease. Age 45-79 (mean 61.9), "
            "BMI 19.8-44.6 (mean 29.3). Not a population-representative sample. "
            "Only upstream's 103-case test split is used here; upstream's 404-case "
            "train split is deliberately held out as a future clean external set."
        ),
    )


def build_manifest(
    root: str | os.PathLike[str] = "data/raw/oaizib",
    *,
    manifest_dir: str | os.PathLike[str] = "data/manifests",
    splits_dir: str | os.PathLike[str] = "data/splits",
    val_fraction: float = 0.25,
    seed: int = 42,
    stratify_by_kl: bool = True,
    verify_masks: bool = False,
) -> dict[str, Any]:
    """Scan, split, verify and write the manifest plus split files.

    Returns a summary dict. Writes:

    - ``data/manifests/oaizib_<n>.json`` — every record, with the split assignment
    - ``data/splits/oaizib_<n>_{train,val}.json`` — patient ids per split
    """
    records = scan_cases(root)

    if verify_masks:
        for record in records:
            if record.label_path:
                verify_mask_labels(record.label_path)
        log.info("all mask label values verified against the declared map")

    splits = split_patient_level(
        records, val_fraction=val_fraction, seed=seed, stratify_by_kl=stratify_by_kl
    )
    ordered = splits["train"] + splits["val"]
    count = len(ordered)

    manifest_path = Path(manifest_dir) / f"oaizib_{count}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "n_cases": count,
                "n_patients": len({r.patient_id for r in ordered}),
                "label_map": {s.value: v for s, v in OAIZIB_LABEL_MAP.items()},
                "structures_unavailable": [s.value for s in UNAVAILABLE_STRUCTURES],
                "split_strategy": (
                    "patient_level_stratified_by_kl_grade"
                    if stratify_by_kl
                    else "patient_level_random"
                ),
                "seed": seed,
                "val_fraction": val_fraction,
                "licence": "CC-BY-NC-4.0",
                "commercial_use": False,
                "records": [r.to_dict() for r in ordered],
            },
            indent=2,
            default=str,
        )
    )

    # JSONL form as well, because ml.datasets.load_manifest streams JSONL.
    jsonl_path = Path(manifest_dir) / f"oaizib_{count}.jsonl"
    write_manifest(ordered, jsonl_path)

    splits_root = Path(splits_dir)
    splits_root.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        (splits_root / f"oaizib_{count}_{name}.json").write_text(
            json.dumps(
                {
                    "dataset_id": DATASET_ID,
                    "split": name,
                    "n_cases": len(subset),
                    "patients": sorted({r.patient_id for r in subset}),
                    "case_ids": sorted(r.case_id for r in subset),
                    "kl_histogram": _kl_histogram(subset),
                },
                indent=2,
            )
        )

    contract = build_dataset_contract(splits)
    summary = {
        "manifest": str(manifest_path),
        "manifest_jsonl": str(jsonl_path),
        "splits": {name: len(subset) for name, subset in splits.items()},
        "patients": {
            name: len({r.patient_id for r in subset})
            for name, subset in splits.items()
        },
        "kl_histogram": {
            name: _kl_histogram(subset) for name, subset in splits.items()
        },
        "sides": sorted({r.side.value for r in ordered}),
        "spacing_observed": sorted(
            {r.spacing_mm for r in ordered if r.spacing_mm is not None}
        ),
        "label_map": {s.value: v for s, v in OAIZIB_LABEL_MAP.items()},
        "dataset_contract": contract.to_dict(),
    }
    log.info("OAIZIB-CM manifest written", fields={k: summary[k] for k in ("manifest", "splits", "patients")})
    return summary
