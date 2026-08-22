"""Volume and mask IO (NIfTI / NRRD / MetaImage) plus DICOM->NIfTI conversion.

The platform's internal interchange format is NIfTI (``.nii.gz``): it is compact,
carries an affine, and is what nnU-Net and MONAI consume directly (PRD §10).

Sidecar metadata
----------------
NIfTI cannot carry the platform's structure/label map, modality, laterality or
provenance. Rather than encode them into filenames, every volume written here
gets a ``<name>.json`` sidecar. A mask without its sidecar is loadable but its
label map must then be supplied explicitly, because guessing label ordering is
exactly the mistake PRD §113 warns about.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kneeai.core.anatomy import Modality, Side, Structure
from kneeai.core.errors import DicomError, ValidationError
from kneeai.core.provenance import ArtifactRef, Provenance
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger

__all__ = [
    "write_volume",
    "read_volume",
    "write_mask",
    "read_mask",
    "dicom_to_nifti",
    "sidecar_path",
]

log = get_logger("imaging.io")

_SUPPORTED_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".nhdr")


def _check_suffix(path: Path) -> None:
    name = path.name.lower()
    if not any(name.endswith(s) for s in _SUPPORTED_SUFFIXES):
        raise ValidationError(
            f"unsupported image extension for {path.name!r}; expected one of "
            f"{_SUPPORTED_SUFFIXES}"
        )


def sidecar_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    name = p.name
    for suffix in _SUPPORTED_SUFFIXES:
        if name.lower().endswith(suffix):
            return p.with_name(name[: -len(suffix)] + ".json")
    return p.with_suffix(".json")


def _sidecar_payload(volume: Volume) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "modality": volume.modality.value,
        "side": volume.side.value,
        "is_label": volume.is_label,
        "spacing_mm": volume.spacing.tolist(),
        "origin_mm": volume.origin.tolist(),
        "direction": volume.direction.tolist(),
        "shape": list(volume.shape),
        "orientation": volume.orientation,
        "metadata": dict(volume.metadata),
    }
    if volume.is_label and volume.label_map is not None:
        payload["label_map"] = {s.value: int(v) for s, v in volume.label_map.items()}
    if volume.provenance is not None:
        payload["provenance"] = volume.provenance.to_dict()
    return payload


def write_volume(
    volume: Volume,
    path: str | os.PathLike[str],
    *,
    compress: bool = True,
    write_sidecar: bool = True,
) -> ArtifactRef:
    """Write a volume plus its JSON sidecar. Returns a content-addressed ref."""
    import SimpleITK as sitk  # noqa: N813

    target = Path(path)
    _check_suffix(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    image = volume.to_sitk()
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(target))
    writer.SetUseCompression(compress)
    writer.Execute(image)

    if write_sidecar:
        side = sidecar_path(target)
        side.write_text(json.dumps(_sidecar_payload(volume), indent=2, default=str))

    ref = ArtifactRef.from_path(
        "mask" if volume.is_label else "volume",
        target,
        note=f"{volume.orientation} {list(volume.shape)}",
    )
    log.info(
        "volume written",
        fields={"path": str(target), "shape": list(volume.shape), "hash": ref.content_hash},
    )
    return ref


def read_volume(
    path: str | os.PathLike[str],
    *,
    modality: Modality | None = None,
    side: Side | None = None,
    label_map: Mapping[Structure, int] | None = None,
    is_label: bool | None = None,
) -> Volume:
    """Read a volume, using its sidecar when present.

    Explicit arguments override the sidecar, which in turn overrides defaults.
    """
    import SimpleITK as sitk  # noqa: N813

    source = Path(path)
    if not source.exists():
        raise ValidationError(f"image file not found: {source}")
    _check_suffix(source)

    meta: dict[str, Any] = {}
    side_file = sidecar_path(source)
    if side_file.exists():
        try:
            meta = json.loads(side_file.read_text())
        except json.JSONDecodeError as exc:
            log.warning(
                "sidecar is not valid JSON and was ignored",
                fields={"path": str(side_file), "error": str(exc)},
            )

    resolved_label = is_label if is_label is not None else bool(meta.get("is_label", False))
    resolved_map: dict[Structure, int] | None = None
    if label_map is not None:
        resolved_map = dict(label_map)
    elif meta.get("label_map"):
        resolved_map = {Structure(k): int(v) for k, v in meta["label_map"].items()}

    if resolved_label and resolved_map is None:
        raise ValidationError(
            f"{source.name} is a label volume but no label map is available "
            "(no sidecar, none supplied). Supply label_map= explicitly rather "
            "than assuming a label ordering."
        )

    image = sitk.ReadImage(str(source))
    resolved_modality = modality or Modality(meta.get("modality", "UNKNOWN"))
    resolved_side = side or Side(meta.get("side", "unknown"))

    if resolved_label:
        arr_check = sitk.GetArrayViewFromImage(image)
        if not np.issubdtype(arr_check.dtype, np.integer):
            image = sitk.Cast(image, sitk.sitkUInt8)

    provenance = None
    if meta.get("provenance"):
        try:
            provenance = Provenance.from_dict(meta["provenance"])
        except Exception as exc:  # noqa: BLE001
            log.warning("sidecar provenance could not be parsed", fields={"error": str(exc)})

    return Volume.from_sitk(
        image,
        modality=resolved_modality,
        side=resolved_side,
        is_label=resolved_label,
        label_map=resolved_map,
        metadata=meta.get("metadata", {}),
        provenance=provenance,
    )


def write_mask(
    mask: Volume, path: str | os.PathLike[str], **kw: Any
) -> ArtifactRef:
    if not mask.is_label:
        raise ValidationError("write_mask() requires a label volume")
    return write_volume(mask, path, **kw)


def read_mask(
    path: str | os.PathLike[str], label_map: Mapping[Structure, int] | None = None
) -> Volume:
    return read_volume(path, label_map=label_map, is_label=True)


# ---------------------------------------------------------------------------
# DICOM -> NIfTI
# ---------------------------------------------------------------------------


def dicom_to_nifti(
    dicom_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    series_uid: str | None = None,
    reorient_to: str | None = "LPS",
) -> tuple[Volume, ArtifactRef]:
    """Convert one DICOM series to NIfTI.

    When the directory holds several series, ``series_uid`` selects one;
    otherwise the best volumetric candidate is chosen and the choice is logged.
    Silently picking a series is acceptable for a research script but not for a
    clinical pipeline, so the selection is always recorded in the sidecar.
    """
    from kneeai.imaging.dicom import load_series, scan_directory

    studies = scan_directory(dicom_dir)
    if not studies:
        raise DicomError(f"no DICOM studies found under {dicom_dir}")
    if len(studies) > 1 and series_uid is None:
        raise DicomError(
            f"{len(studies)} studies found under {dicom_dir}; specify series_uid "
            "so the conversion is unambiguous"
        )

    study = studies[0]
    if series_uid is not None:
        matches = [s for s in study.series if s.series_instance_uid == series_uid]
        if not matches:
            for candidate in studies[1:]:
                matches = [s for s in candidate.series if s.series_instance_uid == series_uid]
                if matches:
                    study = candidate
                    break
        if not matches:
            raise DicomError(f"series {series_uid} not found under {dicom_dir}")
        series = matches[0]
    else:
        series = study.primary_series()
        log.info(
            "series auto-selected for conversion",
            fields={
                "series_uid": series.series_instance_uid,
                "description": series.description,
                "n_instances": series.n_instances,
                "alternatives": len(study.series) - 1,
            },
        )

    volume = load_series(series)
    if reorient_to:
        volume = volume.reorient(reorient_to)
    volume = volume.with_metadata(
        source_series_uid=series.series_instance_uid,
        source_study_uid=series.study_instance_uid,
        source_n_instances=series.n_instances,
        auto_selected=series_uid is None,
    )
    ref = write_volume(volume, output_path)
    return volume, ref
