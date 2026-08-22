"""nnU-Net v2 integration (PRD §35, §95: nnU-Net is the segmentation baseline).

nnU-Net is a self-configuring pipeline, not an architecture to instantiate, so
this module handles the boundary: exporting the platform's manifests into
nnU-Net's expected raw-data layout and generating the commands to run.

The important detail here is :func:`write_patient_level_splits`.

By default nnU-Net generates its own random 5-fold cross-validation over *cases*.
For OAI-derived data that is a leak: the same patient's baseline and 24-month
visits land in different folds, and the reported cross-validation Dice is
optimistic in a way nothing in the training log reveals (PRD §63, §115). This
module writes ``splits_final.json`` explicitly, grouped by patient, so nnU-Net
uses our folds instead of inventing its own.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from kneeai.core.anatomy import Structure
from kneeai.core.errors import DatasetError, ValidationError
from kneeai.logging import get_logger
from kneeai.ml.datasets import CaseRecord, verify_no_leakage

__all__ = [
    "NNUNetExport",
    "export_to_nnunet",
    "write_patient_level_splits",
    "build_dataset_json",
    "nnunet_commands",
    "nnunet_environment",
]

log = get_logger("ml.nnunet")


@dataclass
class NNUNetExport:
    """Result of exporting a manifest into nnU-Net raw format."""

    dataset_id: int
    dataset_name: str
    raw_dir: Path
    n_training: int
    n_test: int
    label_map: Mapping[Structure, int]
    splits_file: Path | None = None
    case_identifiers: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def folder_name(self) -> str:
        return f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "folder": self.folder_name,
            "raw_dir": str(self.raw_dir),
            "n_training": self.n_training,
            "n_test": self.n_test,
            "labels": {s.value: int(v) for s, v in self.label_map.items()},
            "splits_file": str(self.splits_file) if self.splits_file else None,
            "skipped": self.skipped,
        }


def build_dataset_json(
    label_map: Mapping[Structure, int],
    n_training: int,
    *,
    channel_names: Mapping[str, str] | None = None,
    file_ending: str = ".nii.gz",
    description: str = "",
    reference: str = "",
    licence: str = "see dataset registry",
) -> dict[str, Any]:
    """Build nnU-Net v2 ``dataset.json``.

    nnU-Net's ``labels`` map is name -> integer, the inverse of how it reads, and
    background must be 0.
    """
    if label_map.get(Structure.BACKGROUND, 0) != 0:
        raise ValidationError("nnU-Net requires the background label to be 0")
    labels = {"background": 0}
    for structure, value in sorted(label_map.items(), key=lambda kv: kv[1]):
        if structure is Structure.BACKGROUND:
            continue
        labels[structure.value] = int(value)
    return {
        "channel_names": dict(channel_names or {"0": "MR"}),
        "labels": labels,
        "numTraining": int(n_training),
        "file_ending": file_ending,
        "description": description
        or "Knee MR segmentation: bone, cartilage and meniscus.",
        "reference": reference,
        "licence": licence,
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def export_to_nnunet(
    train_records: Sequence[CaseRecord],
    *,
    nnunet_raw: str | os.PathLike[str],
    dataset_id: int,
    dataset_name: str = "KneeMR",
    label_map: Mapping[Structure, int] | None = None,
    test_records: Sequence[CaseRecord] = (),
    channel_names: Mapping[str, str] | None = None,
    link: bool = True,
    overwrite: bool = False,
) -> NNUNetExport:
    """Export manifest records into ``nnUNet_raw/DatasetXXX_Name/``.

    Args:
        link: hard-link rather than copy. Knee MR volumes are large and the
            training set is written once; linking avoids doubling disk use.
            Falls back to copying across filesystem boundaries.
    """
    from kneeai.core.anatomy import DEFAULT_LABEL_MAP

    if not 1 <= dataset_id <= 999:
        raise ValidationError("nnU-Net dataset ids must be between 1 and 999")
    if not train_records:
        raise DatasetError("no training records to export")

    resolved_map = dict(label_map or DEFAULT_LABEL_MAP)
    root = Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{dataset_name}"
    if root.exists():
        if not overwrite:
            raise DatasetError(
                f"{root} already exists. Pass overwrite=True to replace it, so "
                "clobbering a prepared dataset is always deliberate."
            )
        shutil.rmtree(root)

    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    images_ts = root / "imagesTs"
    for directory in (images_tr, labels_tr):
        directory.mkdir(parents=True, exist_ok=True)
    if test_records:
        images_ts.mkdir(parents=True, exist_ok=True)

    skipped: list[tuple[str, str]] = []
    identifiers: list[str] = []
    n_training = 0

    def place(source: Path, destination: Path) -> None:
        if link:
            try:
                os.link(source, destination)
                return
            except OSError:
                pass  # different filesystem, or links unsupported
        shutil.copy2(source, destination)

    for record in train_records:
        if not record.label_path:
            skipped.append((record.case_id, "no label volume"))
            continue
        image = Path(record.image_path)
        label = Path(record.label_path)
        if not image.exists() or not label.exists():
            skipped.append((record.case_id, "referenced file missing"))
            continue
        identifier = _safe_identifier(record.case_id)
        if identifier in identifiers:
            skipped.append((record.case_id, "duplicate case identifier"))
            continue
        place(image, images_tr / f"{identifier}_0000.nii.gz")
        place(label, labels_tr / f"{identifier}.nii.gz")
        identifiers.append(identifier)
        n_training += 1

    n_test = 0
    for record in test_records:
        image = Path(record.image_path)
        if not image.exists():
            skipped.append((record.case_id, "test image missing"))
            continue
        place(image, images_ts / f"{_safe_identifier(record.case_id)}_0000.nii.gz")
        n_test += 1

    if n_training == 0:
        raise DatasetError(
            f"nothing was exported; all {len(train_records)} records were skipped. "
            f"First reasons: {skipped[:5]}"
        )

    dataset_json = build_dataset_json(
        resolved_map, n_training, channel_names=channel_names
    )
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    export = NNUNetExport(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        raw_dir=root,
        n_training=n_training,
        n_test=n_test,
        label_map=resolved_map,
        case_identifiers=identifiers,
        skipped=skipped,
    )
    log.info("nnU-Net export complete", fields=export.to_dict())
    if skipped:
        log.warning(
            "records skipped during nnU-Net export",
            fields={"count": len(skipped), "examples": skipped[:5]},
        )
    return export


def write_patient_level_splits(
    train_records: Sequence[CaseRecord],
    val_records: Sequence[CaseRecord],
    *,
    nnunet_preprocessed: str | os.PathLike[str],
    dataset_id: int,
    dataset_name: str = "KneeMR",
    n_folds: int = 5,
    seed: int = 42,
) -> Path:
    """Write ``splits_final.json`` with patient-level folds.

    Fold 0 is the platform's own train/validation split, so ``nnUNetv2_train
    ... 0`` trains on exactly the split the rest of the platform reports against.
    Folds 1..n-1 are generated by grouping *patients* — never cases — for
    cross-validation.

    Without this file nnU-Net creates random case-level folds and the resulting
    cross-validation metrics are inflated by same-patient leakage.
    """
    import numpy as np

    verify_no_leakage({"train": train_records, "val": val_records})

    all_records = list(train_records) + list(val_records)
    by_patient: dict[str, list[str]] = {}
    for record in all_records:
        by_patient.setdefault(record.patient_id, []).append(
            _safe_identifier(record.case_id)
        )

    splits: list[dict[str, list[str]]] = [
        {
            "train": sorted(_safe_identifier(r.case_id) for r in train_records),
            "val": sorted(_safe_identifier(r.case_id) for r in val_records),
        }
    ]

    patients = sorted(by_patient)
    if n_folds > 1 and len(patients) >= n_folds:
        rng = np.random.default_rng(seed)
        shuffled = list(patients)
        rng.shuffle(shuffled)
        chunks = [shuffled[i::n_folds] for i in range(n_folds)]
        for fold in range(1, n_folds):
            val_patients = set(chunks[fold])
            train_cases: list[str] = []
            val_cases: list[str] = []
            for patient, cases in by_patient.items():
                (val_cases if patient in val_patients else train_cases).extend(cases)
            splits.append({"train": sorted(train_cases), "val": sorted(val_cases)})
    elif n_folds > 1:
        log.warning(
            "too few patients for the requested number of folds; only fold 0 written",
            fields={"patients": len(patients), "requested_folds": n_folds},
        )

    target_dir = Path(nnunet_preprocessed) / f"Dataset{dataset_id:03d}_{dataset_name}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "splits_final.json"
    target.write_text(json.dumps(splits, indent=2))

    log.info(
        "patient-level nnU-Net splits written",
        fields={
            "path": str(target),
            "folds": len(splits),
            "patients": len(patients),
            "fold0_train_cases": len(splits[0]["train"]),
            "fold0_val_cases": len(splits[0]["val"]),
        },
    )
    return target


def nnunet_environment(
    root: str | os.PathLike[str],
) -> dict[str, str]:
    """The three environment variables nnU-Net v2 requires."""
    base = Path(root)
    env = {
        "nnUNet_raw": str(base / "nnUNet_raw"),
        "nnUNet_preprocessed": str(base / "nnUNet_preprocessed"),
        "nnUNet_results": str(base / "nnUNet_results"),
    }
    for path in env.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return env


def nnunet_commands(
    export: NNUNetExport,
    *,
    configuration: str = "3d_fullres",
    folds: Sequence[int] = (0,),
    trainer: str = "nnUNetTrainer",
) -> list[str]:
    """The commands to run, in order.

    Returned as strings rather than executed: nnU-Net training is a multi-hour
    job that belongs in a Colab cell or a scheduler, where its output is visible
    and it can be resumed, not buried inside a library call.
    """
    dataset = f"{export.dataset_id:03d}"
    commands = [
        f"nnUNetv2_plan_and_preprocess -d {dataset} --verify_dataset_integrity",
    ]
    commands.extend(
        f"nnUNetv2_train {dataset} {configuration} {fold} -tr {trainer}" for fold in folds
    )
    fold_args = " ".join(str(f) for f in folds)
    commands.append(
        f"nnUNetv2_find_best_configuration {dataset} -c {configuration} -f {fold_args}"
    )
    if export.n_test:
        commands.append(
            f"nnUNetv2_predict -i {export.raw_dir / 'imagesTs'} "
            f"-o {export.raw_dir.parent.parent / 'predictions' / export.folder_name} "
            f"-d {dataset} -c {configuration} -f {fold_args}"
        )
    return commands


def _safe_identifier(case_id: str) -> str:
    """nnU-Net case identifiers must not contain separators or underscores.

    Underscores are stripped because nnU-Net parses ``_0000`` as the channel
    suffix; an identifier containing an underscore silently becomes a different
    case with a different channel index.
    """
    cleaned = "".join(c if c.isalnum() else "-" for c in str(case_id).strip())
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise ValidationError(f"case id {case_id!r} produces an empty identifier")
    return cleaned
