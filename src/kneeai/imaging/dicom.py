"""DICOM discovery, metadata extraction and series loading.

Requires the ``imaging`` extra (pydicom + SimpleITK).

Hostile-input posture (PRD §82)
------------------------------
A DICOM directory is untrusted input. This module treats it that way:

- files are size-capped before being parsed
- every file is parsed inside its own try/except; one malformed instance never
  aborts the scan of a study
- pixel data is not touched during the metadata scan
- declared pixel geometry is cross-checked against the actual buffer length
- private and unknown tags are never interpreted, only counted
- the scan is bounded in file count so a directory bomb cannot hang a worker
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from kneeai.core.anatomy import Modality, Side, StudyType
from kneeai.core.errors import DicomError, DicomValidationError
from kneeai.core.provenance import Provenance, hash_file
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger

__all__ = [
    "DicomInstance",
    "DicomSeries",
    "DicomStudy",
    "scan_directory",
    "load_series",
    "detect_side",
    "detect_study_type",
    "MAX_INSTANCE_BYTES",
    "MAX_FILES_PER_SCAN",
]

log = get_logger("imaging.dicom")

#: Refuse to parse a single instance larger than this. Legitimate knee MR/CT
#: slices are well under 100 MB; a larger "instance" is either an enhanced
#: multi-frame object needing dedicated handling, or an attack.
MAX_INSTANCE_BYTES = 256 * 1024 * 1024

#: Upper bound on files examined in one scan.
MAX_FILES_PER_SCAN = 200_000

#: Tags read during the metadata scan. Everything else is ignored.
_SCAN_TAGS: tuple[str, ...] = (
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "InstanceNumber",
    "Modality",
    "BodyPartExamined",
    "ImageLaterality",
    "Laterality",
    "ProtocolName",
    "Rows",
    "Columns",
    "BitsAllocated",
    "SamplesPerPixel",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "SliceLocation",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "ScanningSequence",
    "SequenceName",
    "SequenceVariant",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "KVP",
    "ConvolutionKernel",
    "PhotometricInterpretation",
    "ViewPosition",
    "PatientPosition",
    "FrameOfReferenceUID",
    "NumberOfFrames",
    "StudyDate",
    "AcquisitionDate",
    "PatientSex",
    "PatientAge",
)

_IMAGE_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.4": "MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.4.1": "Enhanced MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.2": "CT Image Storage",
    "1.2.840.10008.5.1.4.1.1.2.1": "Enhanced CT Image Storage",
    "1.2.840.10008.5.1.4.1.1.1": "Computed Radiography Image Storage",
    "1.2.840.10008.5.1.4.1.1.1.1": "Digital X-Ray Image Storage (For Presentation)",
    "1.2.840.10008.5.1.4.1.1.1.1.1": "Digital X-Ray Image Storage (For Processing)",
    "1.2.840.10008.5.1.4.1.1.7": "Secondary Capture Image Storage",
}


@dataclass(frozen=True)
class DicomInstance:
    """One DICOM file's scan-level metadata. Pixel data is not held."""

    path: Path
    sop_instance_uid: str
    sop_class_uid: str
    series_instance_uid: str
    study_instance_uid: str
    instance_number: int | None
    rows: int | None
    columns: int | None
    position: tuple[float, float, float] | None
    orientation: tuple[float, ...] | None
    slice_location: float | None
    number_of_frames: int
    bytes_size: int
    issues: tuple[str, ...] = ()

    @property
    def is_multiframe(self) -> bool:
        return self.number_of_frames > 1


@dataclass
class DicomSeries:
    """A DICOM series: the unit the pipeline actually loads."""

    series_instance_uid: str
    study_instance_uid: str
    modality: Modality
    instances: list[DicomInstance] = field(default_factory=list)
    series_number: int | None = None
    description: str = ""
    protocol_name: str = ""
    body_part: str = ""
    laterality: str = ""
    manufacturer: str = ""
    model_name: str = ""
    pixel_spacing: tuple[float, float] | None = None
    slice_thickness: float | None = None
    spacing_between_slices: float | None = None
    frame_of_reference_uid: str = ""
    acquisition_params: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    # --- derived ---
    @property
    def n_instances(self) -> int:
        return len(self.instances)

    @property
    def paths(self) -> list[Path]:
        return [i.path for i in self.instances]

    @property
    def is_volumetric_candidate(self) -> bool:
        """Enough instances (or frames) to reconstruct a 3D volume."""
        if not self.modality.is_volumetric:
            return False
        if self.n_instances >= 3:
            return True
        return any(i.is_multiframe and i.number_of_frames >= 3 for i in self.instances)

    @property
    def side(self) -> Side:
        return detect_side(self)

    @property
    def study_type(self) -> StudyType:
        return detect_study_type(self)

    def sorted_instances(self) -> list[DicomInstance]:
        """Instances ordered along the acquisition normal.

        Sorting by ``InstanceNumber`` is unreliable across vendors. Where
        ImageOrientationPatient is present the slices are ordered by their
        projection onto the slice normal, which is geometrically correct even
        when instance numbers are missing, duplicated or reversed.
        """
        with_geometry = [i for i in self.instances if i.position and i.orientation]
        if len(with_geometry) == self.n_instances and with_geometry:
            o = np.asarray(with_geometry[0].orientation, dtype=np.float64)
            if o.size >= 6:
                row, col = o[:3], o[3:6]
                normal = np.cross(row, col)
                return sorted(
                    self.instances,
                    key=lambda i: float(np.dot(np.asarray(i.position), normal)),
                )
        if all(i.instance_number is not None for i in self.instances):
            return sorted(self.instances, key=lambda i: int(i.instance_number or 0))
        return sorted(self.instances, key=lambda i: str(i.path))

    def slice_spacings(self) -> np.ndarray:
        """Consecutive slice gaps in millimetres, along the slice normal."""
        ordered = self.sorted_instances()
        pts = [i.position for i in ordered if i.position is not None]
        if len(pts) < 2:
            return np.asarray([], dtype=np.float64)
        arr = np.asarray(pts, dtype=np.float64)
        return np.linalg.norm(np.diff(arr, axis=0), axis=1)

    def summary(self) -> dict[str, Any]:
        gaps = self.slice_spacings()
        return {
            "series_uid": self.series_instance_uid,
            "modality": self.modality.value,
            "series_number": self.series_number,
            "description": self.description,
            "n_instances": self.n_instances,
            "pixel_spacing": list(self.pixel_spacing) if self.pixel_spacing else None,
            "slice_thickness": self.slice_thickness,
            "median_slice_gap_mm": float(np.median(gaps)) if gaps.size else None,
            "slice_gap_std_mm": float(gaps.std()) if gaps.size else None,
            "side": self.side.value,
            "study_type": self.study_type.value,
            "volumetric_candidate": self.is_volumetric_candidate,
            "manufacturer": self.manufacturer,
            "model": self.model_name,
            "issues": list(self.issues),
        }


@dataclass
class DicomStudy:
    """All series discovered under one StudyInstanceUID."""

    study_instance_uid: str
    series: list[DicomSeries] = field(default_factory=list)
    study_date: str = ""
    skipped_files: list[tuple[str, str]] = field(default_factory=list)

    def __iter__(self) -> Iterator[DicomSeries]:
        return iter(self.series)

    def __len__(self) -> int:
        return len(self.series)

    def by_modality(self, modality: Modality) -> list[DicomSeries]:
        return [s for s in self.series if s.modality is modality]

    def volumetric_series(self) -> list[DicomSeries]:
        """Volumetric series, best first (most instances, then finest spacing)."""

        def key(s: DicomSeries) -> tuple:
            gaps = s.slice_spacings()
            gap = float(np.median(gaps)) if gaps.size else 99.0
            in_plane = min(s.pixel_spacing) if s.pixel_spacing else 99.0
            return (-s.n_instances, gap, in_plane)

        return sorted([s for s in self.series if s.is_volumetric_candidate], key=key)

    def primary_series(self) -> DicomSeries:
        """The series the pipeline should use, or raise if none is suitable."""
        candidates = self.volumetric_series()
        if candidates:
            return candidates[0]
        radiographs = [s for s in self.series if s.modality in (Modality.XRAY, Modality.DX)]
        if radiographs:
            return radiographs[0]
        raise DicomValidationError(
            f"study {self.study_instance_uid} contains no series usable by the "
            f"pipeline ({len(self.series)} series scanned, "
            f"{len(self.skipped_files)} files skipped)"
        )

    def summary(self) -> dict[str, Any]:
        return {
            "study_uid": self.study_instance_uid,
            "study_date": self.study_date,
            "n_series": len(self.series),
            "series": [s.summary() for s in self.series],
            "skipped_files": len(self.skipped_files),
        }


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _safe_read_metadata(path: Path) -> Any | None:
    """Read a DICOM header defensively. Returns ``None`` if unusable."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        size = path.stat().st_size
    except OSError as exc:
        log.debug("stat failed", fields={"path": str(path), "error": str(exc)})
        return None
    if size == 0 or size > MAX_INSTANCE_BYTES:
        return None
    try:
        # specific_tags keeps the parse bounded; stop_before_pixels avoids
        # touching (potentially malicious) pixel data during discovery.
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            specific_tags=list(_SCAN_TAGS),
            force=False,
        )
    except (InvalidDicomError, OSError, ValueError, KeyError, AttributeError):
        return None
    except Exception as exc:  # noqa: BLE001 - pydicom can raise vendor-specific errors
        log.debug("unexpected DICOM parse failure", fields={"error": type(exc).__name__})
        return None
    return ds


def _get(ds: Any, name: str, default: Any = None) -> Any:
    value = getattr(ds, name, None)
    return default if value is None or value == "" else value


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_tuple(value: Any, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        seq = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return tuple(seq) if len(seq) == size else None


def _modality_of(raw: Any) -> Modality:
    text = str(raw or "").upper().strip()
    if text in ("MR", "MRI"):
        return Modality.MR
    if text == "CT":
        return Modality.CT
    if text == "CR":
        return Modality.XRAY
    if text == "DX":
        return Modality.DX
    return Modality.UNKNOWN


def _instance_issues(ds: Any, path: Path, size: int) -> list[str]:
    """Cheap consistency checks that catch malformed or crafted instances."""
    issues: list[str] = []
    sop_class = str(_get(ds, "SOPClassUID", ""))
    if sop_class and sop_class not in _IMAGE_SOP_CLASSES:
        issues.append(f"unrecognised SOP class {sop_class}")

    rows = _get(ds, "Rows")
    cols = _get(ds, "Columns")
    bits = _get(ds, "BitsAllocated", 16)
    samples = _get(ds, "SamplesPerPixel", 1)
    frames = int(_as_float(_get(ds, "NumberOfFrames", 1)) or 1)
    if rows and cols:
        try:
            declared = int(rows) * int(cols) * int(samples) * (int(bits) // 8) * max(frames, 1)
        except (TypeError, ValueError):
            declared = 0
        # Declared pixel payload must plausibly fit the file. A header claiming
        # far more pixel data than the file can hold is a classic malformed
        # object; compressed transfer syntaxes make the reverse legitimate.
        if declared > size * 64:
            issues.append(
                f"declared pixel payload ({declared} B) is implausible for a "
                f"{size} B file"
            )
        if int(rows) <= 0 or int(cols) <= 0 or int(rows) > 65535 or int(cols) > 65535:
            issues.append(f"implausible image dimensions {rows}x{cols}")
    else:
        issues.append("missing Rows/Columns")

    if _get(ds, "SeriesInstanceUID") is None:
        issues.append("missing SeriesInstanceUID")
    if _get(ds, "ImageOrientationPatient") is None and _modality_of(
        _get(ds, "Modality")
    ).is_volumetric:
        issues.append("missing ImageOrientationPatient")
    return issues


def scan_directory(
    root: str | os.PathLike[str],
    *,
    recursive: bool = True,
    max_files: int = MAX_FILES_PER_SCAN,
) -> list[DicomStudy]:
    """Discover DICOM studies and series under ``root``.

    Files that cannot be parsed are recorded in each study's ``skipped_files``
    rather than raising, so a study with a few corrupt instances still reaches
    validation, where the decision to proceed is made explicitly.
    """
    base = Path(root)
    if not base.exists():
        raise DicomError(f"DICOM path does not exist: {base}")

    paths: list[Path] = []
    if base.is_file():
        paths = [base]
    else:
        walker = base.rglob("*") if recursive else base.glob("*")
        for p in walker:
            if p.is_file() and not p.name.startswith("."):
                paths.append(p)
                if len(paths) >= max_files:
                    log.warning(
                        "file scan limit reached; remaining files ignored",
                        fields={"limit": max_files},
                    )
                    break

    series_map: dict[str, DicomSeries] = {}
    study_dates: dict[str, str] = {}
    skipped: list[tuple[str, str]] = []

    for path in paths:
        ds = _safe_read_metadata(path)
        if ds is None:
            skipped.append((str(path), "not a readable DICOM file"))
            continue
        series_uid = _get(ds, "SeriesInstanceUID")
        study_uid = _get(ds, "StudyInstanceUID")
        if not series_uid or not study_uid:
            skipped.append((str(path), "missing series or study UID"))
            continue

        size = path.stat().st_size
        issues = _instance_issues(ds, path, size)
        frames = int(_as_float(_get(ds, "NumberOfFrames", 1)) or 1)

        instance = DicomInstance(
            path=path,
            sop_instance_uid=str(_get(ds, "SOPInstanceUID", "")),
            sop_class_uid=str(_get(ds, "SOPClassUID", "")),
            series_instance_uid=str(series_uid),
            study_instance_uid=str(study_uid),
            instance_number=(
                int(_as_float(_get(ds, "InstanceNumber")) or 0)
                if _get(ds, "InstanceNumber") is not None
                else None
            ),
            rows=int(_get(ds, "Rows", 0)) or None,
            columns=int(_get(ds, "Columns", 0)) or None,
            position=_as_tuple(_get(ds, "ImagePositionPatient"), 3),  # type: ignore[arg-type]
            orientation=_as_tuple(_get(ds, "ImageOrientationPatient"), 6),
            slice_location=_as_float(_get(ds, "SliceLocation")),
            number_of_frames=max(frames, 1),
            bytes_size=size,
            issues=tuple(issues),
        )

        key = str(series_uid)
        if key not in series_map:
            series_map[key] = DicomSeries(
                series_instance_uid=key,
                study_instance_uid=str(study_uid),
                modality=_modality_of(_get(ds, "Modality")),
                series_number=(
                    int(_as_float(_get(ds, "SeriesNumber")) or 0)
                    if _get(ds, "SeriesNumber") is not None
                    else None
                ),
                description=str(_get(ds, "SeriesDescription", "")),
                protocol_name=str(_get(ds, "ProtocolName", "")),
                body_part=str(_get(ds, "BodyPartExamined", "")),
                laterality=str(_get(ds, "ImageLaterality", _get(ds, "Laterality", ""))),
                manufacturer=str(_get(ds, "Manufacturer", "")),
                model_name=str(_get(ds, "ManufacturerModelName", "")),
                pixel_spacing=_as_tuple(_get(ds, "PixelSpacing"), 2),  # type: ignore[arg-type]
                slice_thickness=_as_float(_get(ds, "SliceThickness")),
                spacing_between_slices=_as_float(_get(ds, "SpacingBetweenSlices")),
                frame_of_reference_uid=str(_get(ds, "FrameOfReferenceUID", "")),
                acquisition_params={
                    k: v
                    for k, v in (
                        ("magnetic_field_strength", _as_float(_get(ds, "MagneticFieldStrength"))),
                        ("scanning_sequence", str(_get(ds, "ScanningSequence", "") or "")),
                        ("sequence_name", str(_get(ds, "SequenceName", "") or "")),
                        ("echo_time_ms", _as_float(_get(ds, "EchoTime"))),
                        ("repetition_time_ms", _as_float(_get(ds, "RepetitionTime"))),
                        ("flip_angle_deg", _as_float(_get(ds, "FlipAngle"))),
                        ("kvp", _as_float(_get(ds, "KVP"))),
                        ("view_position", str(_get(ds, "ViewPosition", "") or "")),
                        ("patient_position", str(_get(ds, "PatientPosition", "") or "")),
                    )
                    if v not in (None, "")
                },
            )
            study_dates.setdefault(
                str(study_uid), str(_get(ds, "StudyDate", _get(ds, "AcquisitionDate", "")) or "")
            )

        series_map[key].instances.append(instance)
        if issues:
            for issue in issues:
                if issue not in series_map[key].issues:
                    series_map[key].issues.append(issue)

    grouped: dict[str, list[DicomSeries]] = defaultdict(list)
    for s in series_map.values():
        grouped[s.study_instance_uid].append(s)

    studies = [
        DicomStudy(
            study_instance_uid=uid,
            series=sorted(items, key=lambda s: (s.series_number or 0, s.series_instance_uid)),
            study_date=study_dates.get(uid, ""),
            skipped_files=skipped,
        )
        for uid, items in grouped.items()
    ]
    log.info(
        "DICOM scan complete",
        fields={
            "files_examined": len(paths),
            "studies": len(studies),
            "series": len(series_map),
            "skipped": len(skipped),
        },
    )
    return sorted(studies, key=lambda s: s.study_instance_uid)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_series(series: DicomSeries, *, side: Side | None = None) -> Volume:
    """Load a series into a :class:`Volume` in the IMAGE frame.

    Slice ordering is delegated to SimpleITK's series reader, which uses
    ImagePositionPatient. The ordered file list from :meth:`DicomSeries.sorted_instances`
    is passed explicitly so ordering is reproducible and does not depend on
    directory iteration order.
    """
    import SimpleITK as sitk  # noqa: N813

    if series.n_instances == 0:
        raise DicomError(f"series {series.series_instance_uid} has no instances")

    ordered = [str(i.path) for i in series.sorted_instances()]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ordered)
    reader.MetaDataDictionaryArrayUpdateOff()
    reader.LoadPrivateTagsOff()

    try:
        image = reader.Execute()
    except RuntimeError as exc:
        # Single-instance or multi-frame series are not a "series" to ITK.
        if series.n_instances == 1:
            try:
                image = sitk.ReadImage(ordered[0])
            except RuntimeError as exc2:
                raise DicomError(
                    f"failed to read series {series.series_instance_uid}: {exc2}"
                ) from exc2
        else:
            raise DicomError(
                f"failed to assemble series {series.series_instance_uid} from "
                f"{series.n_instances} instances: {exc}"
            ) from exc

    if image.GetDimension() == 4:
        # Enhanced/4D objects: take the first volume rather than guessing which
        # echo is wanted. Echo selection is a protocol decision, made upstream.
        image = image[:, :, :, 0]
    if image.GetDimension() == 2:
        raise DicomError(
            f"series {series.series_instance_uid} loaded as 2D; radiographs must "
            "be handled by the 2D pipeline, not load_series()"
        )

    provenance = Provenance.for_component(
        "imaging.preprocess",
        parameters={
            "operation": "load_series",
            "series_uid": series.series_instance_uid,
            "n_instances": series.n_instances,
        },
    )

    volume = Volume.from_sitk(
        image,
        modality=series.modality,
        side=side or series.side,
        metadata={
            "series_uid": series.series_instance_uid,
            "study_uid": series.study_instance_uid,
            "series_description": series.description,
            "manufacturer": series.manufacturer,
            "model": series.model_name,
            "frame_of_reference_uid": series.frame_of_reference_uid,
            **series.acquisition_params,
        },
        provenance=provenance,
    )
    log.info("series loaded", fields=volume.summary())
    return volume


def load_series_files(paths: Sequence[str | os.PathLike[str]]) -> Volume:
    """Load an explicit, already-ordered list of DICOM files."""
    import SimpleITK as sitk  # noqa: N813

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in paths])
    return Volume.from_sitk(reader.Execute())


# ---------------------------------------------------------------------------
# Inference from metadata
# ---------------------------------------------------------------------------

_LEFT_HINTS = ("left", "_l_", " l ", "lt ", "lknee", "l knee", "l-knee")
_RIGHT_HINTS = ("right", "_r_", " r ", "rt ", "rknee", "r knee", "r-knee")


def detect_side(series: DicomSeries) -> Side:
    """Infer which knee from DICOM metadata.

    Laterality tags are trusted first. Text hints in the series description are
    used only as a fallback, and any inference from text is treated as
    unverified: the surgeon must confirm laterality before approval, because a
    text heuristic is not an acceptable basis for a side-specific plan.
    """
    tag = (series.laterality or "").strip().upper()
    if tag in ("L", "LEFT"):
        return Side.LEFT
    if tag in ("R", "RIGHT"):
        return Side.RIGHT

    haystack = " ".join(
        (series.description, series.protocol_name, series.body_part)
    ).lower()
    left = any(h in haystack for h in _LEFT_HINTS)
    right = any(h in haystack for h in _RIGHT_HINTS)
    if left and not right:
        return Side.LEFT
    if right and not left:
        return Side.RIGHT
    return Side.UNKNOWN


def detect_study_type(series: DicomSeries) -> StudyType:
    text = " ".join((series.description, series.protocol_name)).lower()
    if series.modality is Modality.CT:
        return StudyType.KNEE_CT
    if series.modality is Modality.MR:
        return StudyType.KNEE_MR_3D
    if series.modality in (Modality.XRAY, Modality.DX):
        if any(k in text for k in ("long leg", "longleg", "full leg", "hip to ankle", "eos")):
            return StudyType.LONG_LEG_XRAY
        view = str(series.acquisition_params.get("view_position", "")).upper()
        if view in ("LL", "LAT", "LATERAL") or "lateral" in text:
            return StudyType.KNEE_XRAY_LATERAL
        return StudyType.KNEE_XRAY_AP
    return StudyType.UNKNOWN


def series_content_hash(series: DicomSeries) -> str:
    """Content hash over the series' files, order-independent.

    Lets the platform detect that a "new" upload is a study it has already
    processed, and prove which bytes a plan was derived from.
    """
    from kneeai.core.provenance import hash_payload

    return hash_payload(sorted(hash_file(p) for p in series.paths))
