"""Medical imaging layer: DICOM ingest, validation, de-identification, preprocessing.

:class:`~kneeai.imaging.volume.Volume` is importable without the ``imaging``
extra. DICOM parsing, de-identification and file IO require pydicom + SimpleITK
and are imported lazily so the deterministic core stays usable without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kneeai.imaging.volume import Volume, orientation_code, stack_structure_volumes

if TYPE_CHECKING:  # pragma: no cover
    from kneeai.imaging.anonymize import AnonymizationResult, Anonymizer
    from kneeai.imaging.dicom import DicomSeries, DicomStudy, load_series, scan_directory
    from kneeai.imaging.io import dicom_to_nifti, read_volume, write_volume
    from kneeai.imaging.preprocess import (
        PreprocessedVolume,
        PreprocessingSpec,
        preprocess,
        resample_like,
    )
    from kneeai.imaging.validation import (
        Issue,
        Severity,
        ValidationReport,
        validate_study,
        validate_volume,
    )

__all__ = [
    "Volume",
    "orientation_code",
    "stack_structure_volumes",
    # lazy
    "Anonymizer",
    "AnonymizationResult",
    "DicomSeries",
    "DicomStudy",
    "scan_directory",
    "load_series",
    "PreprocessingSpec",
    "PreprocessedVolume",
    "preprocess",
    "resample_like",
    "ValidationReport",
    "Severity",
    "Issue",
    "validate_study",
    "validate_volume",
    "read_volume",
    "write_volume",
    "dicom_to_nifti",
]

_LAZY: dict[str, str] = {
    "Anonymizer": "kneeai.imaging.anonymize",
    "AnonymizationResult": "kneeai.imaging.anonymize",
    "DicomSeries": "kneeai.imaging.dicom",
    "DicomStudy": "kneeai.imaging.dicom",
    "scan_directory": "kneeai.imaging.dicom",
    "load_series": "kneeai.imaging.dicom",
    "PreprocessingSpec": "kneeai.imaging.preprocess",
    "PreprocessedVolume": "kneeai.imaging.preprocess",
    "preprocess": "kneeai.imaging.preprocess",
    "resample_like": "kneeai.imaging.preprocess",
    "ValidationReport": "kneeai.imaging.validation",
    "Severity": "kneeai.imaging.validation",
    "Issue": "kneeai.imaging.validation",
    "validate_study": "kneeai.imaging.validation",
    "validate_volume": "kneeai.imaging.validation",
    "read_volume": "kneeai.imaging.io",
    "write_volume": "kneeai.imaging.io",
    "dicom_to_nifti": "kneeai.imaging.io",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - missing optional extra
        raise ImportError(
            f"{name} requires the 'imaging' extra. Install it with:\n"
            "    pip install -e '.[imaging]'\n"
            f"(underlying error: {exc})"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value
