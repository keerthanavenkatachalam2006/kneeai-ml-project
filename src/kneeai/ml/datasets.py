"""Training data plumbing: case manifests, MONAI transforms, dataloaders.

Requires the ``ml`` extra for the transform/loader helpers. The manifest layer
(:class:`CaseRecord`, :func:`load_manifest`) is torch-free so manifests can be
built and audited locally.

Split integrity
---------------
:func:`build_dataloaders` re-verifies that no patient appears in more than one
split, every time. The check is cheap and the failure mode it prevents —
optimistic metrics from a patient whose other visit was in training (PRD §63,
§115) — is invisible in the loss curves and only surfaces when the model is
deployed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kneeai.core.anatomy import Modality, Side, Structure
from kneeai.core.errors import DatasetError, LeakageError, ValidationError
from kneeai.logging import get_logger

__all__ = [
    "CaseRecord",
    "load_manifest",
    "write_manifest",
    "verify_no_leakage",
    "build_train_transforms",
    "build_val_transforms",
    "build_dataloaders",
    "class_weights_from_manifest",
]

log = get_logger("ml.datasets")


@dataclass
class CaseRecord:
    """One training example: an image, its mask, and the identity it belongs to.

    ``patient_id`` is separate from ``case_id`` deliberately. OAI-derived datasets
    contain multiple visits per patient, and only the patient id is a valid
    grouping key for splitting.
    """

    case_id: str
    patient_id: str
    image_path: str
    label_path: str | None = None
    dataset_id: str = ""
    modality: Modality = Modality.MR
    side: Side = Side.UNKNOWN
    split: str = "train"
    visit: str = ""
    kl_grade: int | None = None
    age: float | None = None
    sex: str = ""
    bmi: float | None = None
    spacing_mm: tuple[float, float, float] | None = None
    structures: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modality"] = self.modality.value
        payload["side"] = self.side.value
        payload["spacing_mm"] = list(self.spacing_mm) if self.spacing_mm else None
        payload["structures"] = list(self.structures)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaseRecord:
        data = dict(payload)
        data["modality"] = Modality(data.get("modality", "MR"))
        data["side"] = Side(data.get("side", "unknown"))
        if data.get("spacing_mm"):
            data["spacing_mm"] = tuple(data["spacing_mm"])
        data["structures"] = tuple(data.get("structures", ()))
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in data.items() if k not in known}
        data = {k: v for k, v in data.items() if k in known}
        if extra:
            data.setdefault("extra", {}).update(extra)
        return cls(**data)

    def as_monai_item(self) -> dict[str, str]:
        item = {"image": self.image_path}
        if self.label_path:
            item["label"] = self.label_path
        return item

    def verify_paths(self) -> list[str]:
        missing = []
        if not Path(self.image_path).exists():
            missing.append(self.image_path)
        if self.label_path and not Path(self.label_path).exists():
            missing.append(self.label_path)
        return missing


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------


def write_manifest(records: Sequence[CaseRecord], path: str | os.PathLike[str]) -> Path:
    """Write a JSONL manifest. One record per line, so it streams and diffs well."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), default=str) + "\n")
    log.info("manifest written", fields={"path": str(target), "records": len(records)})
    return target


def load_manifest(
    path: str | os.PathLike[str],
    *,
    split: str | None = None,
    dataset_ids: Iterable[str] | None = None,
    require_labels: bool = True,
    verify_paths: bool = False,
) -> list[CaseRecord]:
    """Load a JSONL manifest, optionally filtered by split or dataset."""
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"manifest not found: {source}")

    wanted_datasets = set(dataset_ids) if dataset_ids else None
    records: list[CaseRecord] = []
    for line_no, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = CaseRecord.from_dict(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise DatasetError(f"{source}:{line_no} is not a valid record: {exc}") from exc
        if split is not None and record.split != split:
            continue
        if wanted_datasets is not None and record.dataset_id not in wanted_datasets:
            continue
        if require_labels and not record.label_path:
            raise DatasetError(
                f"{source}:{line_no} (case {record.case_id}) has no label_path but "
                "labels are required for this task"
            )
        records.append(record)

    if verify_paths:
        missing: list[str] = []
        for record in records:
            missing.extend(record.verify_paths())
        if missing:
            preview = "\n  ".join(missing[:10])
            raise DatasetError(
                f"{len(missing)} referenced file(s) are missing, e.g.:\n  {preview}"
            )

    if not records:
        raise DatasetError(
            f"no records matched in {source} (split={split!r}, datasets={dataset_ids})"
        )
    log.info(
        "manifest loaded",
        fields={
            "path": str(source),
            "records": len(records),
            "split": split or "all",
            "patients": len({r.patient_id for r in records}),
        },
    )
    return records


def verify_no_leakage(splits: Mapping[str, Sequence[CaseRecord]]) -> dict[str, Any]:
    """Assert that no patient id appears in more than one split.

    Raises:
        LeakageError: naming the offending patients and splits.
    """
    patients_by_split = {
        name: {r.patient_id for r in records} for name, records in splits.items()
    }
    names = sorted(patients_by_split)
    overlaps: dict[str, list[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = patients_by_split[a] & patients_by_split[b]
            if shared:
                overlaps[f"{a}|{b}"] = sorted(shared)

    if overlaps:
        lines = [
            f"  {pair}: {len(ids)} shared patient(s) e.g. {ids[:5]}"
            for pair, ids in overlaps.items()
        ]
        raise LeakageError(
            "patient-level leakage detected between splits:\n"
            + "\n".join(lines)
            + "\nEvery study, visit and derived image for a patient must stay in "
            "one split (PRD §63, §115)."
        )

    summary = {
        "splits": {
            name: {
                "cases": len(splits[name]),
                "patients": len(patients_by_split[name]),
            }
            for name in names
        },
        "total_patients": len(set().union(*patients_by_split.values()))
        if patients_by_split
        else 0,
    }
    log.info("split integrity verified", fields=summary)
    return summary


def class_weights_from_manifest(
    records: Sequence[CaseRecord],
    label_map: Mapping[Structure, int],
    *,
    scheme: str = "inverse_sqrt_frequency",
    sample_limit: int = 40,
) -> tuple[float, ...]:
    """Estimate per-class loss weights from label volumes.

    Femur and tibia occupy roughly two orders of magnitude more voxels than
    cartilage or meniscus. Left unweighted, the loss is effectively optimising the
    bones and treating the structures that actually drive cartilage and meniscus
    measurements as rounding error.

    Inverse *square-root* frequency rather than inverse frequency: full inverse
    weighting over-corrects at this class imbalance and destabilises early
    training.
    """
    import numpy as np

    from kneeai.imaging.io import read_mask

    labelled = [r for r in records if r.label_path]
    if not labelled:
        raise DatasetError("cannot estimate class weights without label volumes")
    sample = labelled[: max(1, sample_limit)]

    n_classes = max(int(v) for v in label_map.values()) + 1
    counts = np.zeros(n_classes, dtype=np.float64)
    for record in sample:
        mask = read_mask(record.label_path, label_map=dict(label_map))  # type: ignore[arg-type]
        values, occurrences = np.unique(mask.array, return_counts=True)
        for value, occurrence in zip(values, occurrences):
            if 0 <= int(value) < n_classes:
                counts[int(value)] += float(occurrence)

    total = counts.sum()
    if total <= 0:
        raise DatasetError("label volumes contained no voxels")
    frequency = np.maximum(counts / total, 1e-8)

    if scheme == "inverse_frequency":
        weights = 1.0 / frequency
    elif scheme == "inverse_sqrt_frequency":
        weights = 1.0 / np.sqrt(frequency)
    elif scheme == "uniform":
        weights = np.ones(n_classes)
    else:
        raise ValidationError(f"unknown class weighting scheme {scheme!r}")

    # Normalise so the mean weight is 1: keeps the loss magnitude (and therefore
    # a sensible learning rate) comparable across schemes.
    weights = weights / weights.mean()
    result = tuple(round(float(w), 4) for w in weights)
    log.info(
        "class weights estimated",
        fields={
            "scheme": scheme,
            "sampled_cases": len(sample),
            "weights": result,
            "voxel_fraction": [round(float(f), 6) for f in frequency],
        },
    )
    return result


# ---------------------------------------------------------------------------
# MONAI transforms
# ---------------------------------------------------------------------------


def build_train_transforms(
    *,
    patch_size: Sequence[int] = (96, 96, 96),
    target_spacing: Sequence[float] | None = (0.5, 0.5, 0.5),
    orientation: str = "LPS",
    n_samples_per_volume: int = 4,
    intensity: str = "zscore",
    ct_window: tuple[float, float] = (400.0, 1800.0),
    augment: bool = True,
) -> Any:
    """Training transform pipeline.

    Augmentation choices are constrained by anatomy:

    - **No left/right flipping.** A flipped right knee is a left knee. The model
      is applied per side and flip augmentation would teach it to ignore the one
      cue that distinguishes them.
    - Rotations are kept small (±10°) because volumes are already reoriented to a
      canonical code; large rotations only model registration failure.
    - Scaling is kept to ±10%: the network's job includes learning absolute
      anatomical size, which is what the measurement engine depends on.
    """
    require_monai()
    from monai import transforms as T

    keys = ["image", "label"]
    steps: list[Any] = [
        T.LoadImaged(keys=keys, image_only=False, ensure_channel_first=True),
        T.Orientationd(keys=keys, axcodes=_monai_axcodes(orientation)),
    ]
    if target_spacing is not None:
        steps.append(
            T.Spacingd(
                keys=keys,
                pixdim=tuple(float(s) for s in target_spacing),
                mode=("bilinear", "nearest"),
            )
        )

    if intensity == "ct_window":
        centre, width = ct_window
        steps.append(
            T.ScaleIntensityRanged(
                keys="image",
                a_min=centre - width / 2,
                a_max=centre + width / 2,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        steps.append(
            T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
        )

    steps.extend(
        [
            T.CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
            T.SpatialPadd(keys=keys, spatial_size=tuple(int(p) for p in patch_size)),
            # Positive/negative sampling: patches centred on a labelled voxel 80%
            # of the time. Uniform random patches in a knee volume are mostly
            # background and mostly bone interior.
            T.RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=tuple(int(p) for p in patch_size),
                pos=4,
                neg=1,
                num_samples=n_samples_per_volume,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )

    if augment:
        steps.extend(
            [
                T.RandAffined(
                    keys=keys,
                    prob=0.3,
                    rotate_range=(0.175, 0.175, 0.175),  # ~10 degrees
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                ),
                T.RandGaussianNoised(keys="image", prob=0.15, std=0.05),
                T.RandGaussianSmoothd(
                    keys="image",
                    prob=0.15,
                    sigma_x=(0.5, 1.0),
                    sigma_y=(0.5, 1.0),
                    sigma_z=(0.5, 1.0),
                ),
                T.RandScaleIntensityd(keys="image", prob=0.15, factors=0.25),
                T.RandShiftIntensityd(keys="image", prob=0.15, offsets=0.1),
                # Simulates the low through-plane resolution of 2D-acquired MR,
                # which is the main domain shift between datasets (PRD §85).
                T.RandSimulateLowResolutiond(keys="image", prob=0.15, zoom_range=(0.5, 1.0)),
                T.RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),
            ]
        )

    steps.append(T.EnsureTyped(keys=keys, track_meta=False))
    return T.Compose(steps)


def build_val_transforms(
    *,
    target_spacing: Sequence[float] | None = (0.5, 0.5, 0.5),
    orientation: str = "LPS",
    intensity: str = "zscore",
    ct_window: tuple[float, float] = (400.0, 1800.0),
    with_label: bool = True,
) -> Any:
    """Validation transforms: identical preprocessing, no augmentation, no cropping.

    Whole volumes are kept so validation uses the same sliding-window inference
    path as production. Validating on random patches reports a number that does
    not correspond to how the model is actually used.
    """
    require_monai()
    from monai import transforms as T

    keys = ["image", "label"] if with_label else ["image"]
    modes = ("bilinear", "nearest") if with_label else ("bilinear",)
    steps: list[Any] = [
        T.LoadImaged(keys=keys, image_only=False, ensure_channel_first=True),
        T.Orientationd(keys=keys, axcodes=_monai_axcodes(orientation)),
    ]
    if target_spacing is not None:
        steps.append(
            T.Spacingd(
                keys=keys, pixdim=tuple(float(s) for s in target_spacing), mode=modes
            )
        )
    if intensity == "ct_window":
        centre, width = ct_window
        steps.append(
            T.ScaleIntensityRanged(
                keys="image",
                a_min=centre - width / 2,
                a_max=centre + width / 2,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        steps.append(T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    steps.append(T.EnsureTyped(keys=keys, track_meta=True))
    return T.Compose(steps)


def _monai_axcodes(orientation: str) -> str:
    """Translate the platform's orientation code to MONAI/nibabel axcodes.

    Both use the same R/L, A/P, I/S letters, so this is a validation step rather
    than a conversion — but it is the boundary where a typo would otherwise
    become a silent axis permutation.
    """
    code = orientation.upper()
    if len(code) != 3 or set(code) - set("RLAPIS"):
        raise ValidationError(f"invalid orientation code {orientation!r}")
    axes = {frozenset("RL"), frozenset("AP"), frozenset("IS")}
    seen = {next(a for a in axes if c in a) for c in code}
    if len(seen) != 3:
        raise ValidationError(
            f"orientation code {orientation!r} does not name three distinct axes"
        )
    return code


def require_monai() -> None:
    from kneeai.ml.models import require_torch

    require_torch()


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def build_dataloaders(
    train_records: Sequence[CaseRecord],
    val_records: Sequence[CaseRecord],
    *,
    patch_size: Sequence[int] = (96, 96, 96),
    target_spacing: Sequence[float] | None = (0.5, 0.5, 0.5),
    batch_size: int = 2,
    n_samples_per_volume: int = 4,
    num_workers: int = 2,
    cache_rate: float = 0.0,
    intensity: str = "zscore",
    augment: bool = True,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Build train/validation dataloaders, verifying split integrity first."""
    require_monai()
    import torch
    from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
    from monai.utils import set_determinism

    verify_no_leakage({"train": train_records, "val": val_records})
    set_determinism(seed=seed)

    train_items = [r.as_monai_item() for r in train_records]
    val_items = [r.as_monai_item() for r in val_records]

    train_tf = build_train_transforms(
        patch_size=patch_size,
        target_spacing=target_spacing,
        n_samples_per_volume=n_samples_per_volume,
        intensity=intensity,
        augment=augment,
    )
    val_tf = build_val_transforms(target_spacing=target_spacing, intensity=intensity)

    if cache_rate > 0:
        train_ds: Any = CacheDataset(
            data=train_items,
            transform=train_tf,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        train_ds = Dataset(data=train_items, transform=train_tf)
    val_ds = Dataset(data=val_items, transform=val_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_items) > batch_size,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    # Validation batch size is 1: volumes have different shapes and are processed
    # with sliding-window inference.
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )
    log.info(
        "dataloaders built",
        fields={
            "train_cases": len(train_items),
            "val_cases": len(val_items),
            "patch_size": list(patch_size),
            "batch_size": batch_size,
            "effective_patches_per_step": batch_size * n_samples_per_volume,
        },
    )
    return train_loader, val_loader
