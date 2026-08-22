"""DICOM de-identification (PRD §10, §81).

Design decision: **allow-list, not deny-list.**

A deny-list ("remove PatientName, PatientID, ...") fails the moment a vendor puts
an identifier in a tag nobody enumerated, which they routinely do. This module
instead keeps only tags the pipeline demonstrably needs and drops everything
else, including all private tags.

What this module does NOT do
---------------------------
- It does not detect burned-in pixel annotation. It flags the DICOM tag that
  declares it and refuses the instance when the tag says annotations are
  present, but it cannot see text rendered into pixels when the tag lies. Pixel
  redaction is a separate, reviewed step.
- It does not make data safe to publish. It produces pseudonymised data suitable
  for internal processing under an existing data agreement.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from kneeai.core.errors import DicomError, ValidationError
from kneeai.core.provenance import Provenance
from kneeai.logging import get_logger

__all__ = [
    "Anonymizer",
    "AnonymizationResult",
    "KEEP_TAGS",
    "DEIDENTIFICATION_PROFILE",
]

log = get_logger("imaging.anonymize")

#: Our pseudonymous UID root. 2.25 is the standard OID arc for UUID-derived UIDs,
#: which avoids colliding with any real organisation's root.
UID_ROOT = "2.25"

DEIDENTIFICATION_PROFILE = "KNEEAI-ALLOWLIST-1"

#: Tag keywords retained verbatim. Everything here is either image geometry the
#: measurements depend on, or acquisition physics needed to interpret intensities.
#: Nothing here identifies a person.
KEEP_TAGS: frozenset[str] = frozenset(
    {
        # --- required DICOM structure ---
        "SOPClassUID",
        "SpecificCharacterSet",
        "ImageType",
        "Modality",
        "InstanceNumber",
        "SeriesNumber",
        "AcquisitionNumber",
        # --- geometry: everything measurement depends on ---
        "Rows",
        "Columns",
        "PixelSpacing",
        "ImagerPixelSpacing",
        "SliceThickness",
        "SpacingBetweenSlices",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "SliceLocation",
        "PatientPosition",
        "NumberOfFrames",
        "PixelRepresentation",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "PlanarConfiguration",
        "RescaleSlope",
        "RescaleIntercept",
        "RescaleType",
        "WindowCenter",
        "WindowWidth",
        "SmallestImagePixelValue",
        "LargestImagePixelValue",
        # --- clinically necessary, non-identifying ---
        "BodyPartExamined",
        "ImageLaterality",
        "Laterality",
        "ViewPosition",
        "PatientOrientation",
        # --- acquisition physics (needed to interpret the image) ---
        "MagneticFieldStrength",
        "ScanningSequence",
        "SequenceVariant",
        "SequenceName",
        "ScanOptions",
        "MRAcquisitionType",
        "EchoTime",
        "RepetitionTime",
        "InversionTime",
        "FlipAngle",
        "EchoTrainLength",
        "NumberOfAverages",
        "PixelBandwidth",
        "ReceiveCoilName",
        "KVP",
        "XRayTubeCurrent",
        "ExposureTime",
        "Exposure",
        "ConvolutionKernel",
        "FilterType",
        "SpiralPitchFactor",
        "DistanceSourceToDetector",
        "DistanceSourceToPatient",
        # --- equipment model (needed for domain-shift analysis) ---
        "Manufacturer",
        "ManufacturerModelName",
        "SoftwareVersions",
        # --- retained but rewritten by this module ---
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "FrameOfReferenceUID",
        "PatientID",
        "PatientSex",
        "PatientAge",
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "ContentDate",
        "StudyTime",
        "SeriesTime",
        "AcquisitionTime",
        "ContentTime",
        "PatientIdentityRemoved",
        "DeidentificationMethod",
        "BurnedInAnnotation",
        # --- pixel data itself ---
        "PixelData",
    }
)

#: Tags rewritten rather than copied. Kept explicit so the transformation applied
#: to each is auditable.
_REWRITTEN = {
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
    "PatientID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    "PatientAge",
}


@dataclass
class AnonymizationResult:
    """Outcome of de-identifying one study."""

    pseudonym: str
    files_written: list[Path] = field(default_factory=list)
    uid_map: dict[str, str] = field(default_factory=dict)
    removed_tag_counts: dict[str, int] = field(default_factory=dict)
    date_shift_days: int = 0
    refused: list[tuple[Path, str]] = field(default_factory=list)
    provenance: Provenance | None = None

    @property
    def n_written(self) -> int:
        return len(self.files_written)

    def summary(self) -> dict[str, Any]:
        return {
            "pseudonym": self.pseudonym,
            "files_written": self.n_written,
            "uids_remapped": len(self.uid_map),
            "distinct_tags_removed": len(self.removed_tag_counts),
            "date_shift_days": self.date_shift_days,
            "refused": [(str(p), why) for p, why in self.refused],
            "profile": DEIDENTIFICATION_PROFILE,
        }


class Anonymizer:
    """Deterministic, keyed pseudonymisation of DICOM studies.

    A single secret key drives every mapping, so:

    - the same patient always maps to the same pseudonym, which is what makes
      patient-level train/test splitting possible (PRD §63, §115)
    - the mapping cannot be reversed or brute-forced without the key
    - re-processing a study reproduces identical output

    The key is a secret. Store it separately from the de-identified data; holding
    both together defeats the purpose.
    """

    def __init__(
        self,
        secret: str | bytes,
        *,
        date_shift_days: int | None = None,
        keep_age: bool = True,
        keep_sex: bool = True,
        refuse_burned_in_annotation: bool = True,
    ) -> None:
        if isinstance(secret, str):
            secret = secret.encode("utf-8")
        if len(secret) < 16:
            raise ValidationError(
                "anonymisation secret must be at least 16 bytes; a short key makes "
                "the pseudonym mapping guessable"
            )
        self._secret = secret
        self._explicit_shift = date_shift_days
        self.keep_age = keep_age
        self.keep_sex = keep_sex
        self.refuse_burned_in_annotation = refuse_burned_in_annotation
        self._uid_cache: dict[str, str] = {}

    # --- keyed derivations -------------------------------------------------
    def _digest(self, namespace: str, value: str) -> bytes:
        return hmac.new(
            self._secret, f"{namespace}|{value}".encode("utf-8"), hashlib.sha256
        ).digest()

    def pseudonym(self, patient_id: str, *, prefix: str = "KA") -> str:
        """Stable pseudonymous patient identifier."""
        if not patient_id:
            raise ValidationError("cannot derive a pseudonym from an empty patient id")
        digest = self._digest("patient", patient_id).hex()[:12].upper()
        return f"{prefix}-{digest}"

    def map_uid(self, uid: str) -> str:
        """Deterministically replace a UID, preserving referential integrity."""
        if not uid:
            return uid
        if uid in self._uid_cache:
            return self._uid_cache[uid]
        digest = self._digest("uid", uid)
        # A DICOM UID is limited to 64 characters and must be dotted numeric.
        number = int.from_bytes(digest[:16], "big")
        new_uid = f"{UID_ROOT}.{number}"
        if len(new_uid) > 64:  # pragma: no cover - defensive
            new_uid = new_uid[:64].rstrip(".")
        self._uid_cache[uid] = new_uid
        return new_uid

    def date_shift(self, patient_id: str) -> int:
        """Per-patient date offset in days.

        Shifting all of a patient's dates by the same amount preserves the
        intervals that longitudinal analysis needs (OAI is longitudinal — PRD
        §47) while destroying the absolute dates.
        """
        if self._explicit_shift is not None:
            return int(self._explicit_shift)
        digest = self._digest("date", patient_id or "unknown")
        # +/- ~5 years, deterministic per patient.
        return int.from_bytes(digest[:4], "big") % 3651 - 1825

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _shift_date_string(value: str, days: int) -> str:
        try:
            parsed = datetime.strptime(value[:8], "%Y%m%d")
        except (ValueError, TypeError):
            return ""
        return (parsed + timedelta(days=days)).strftime("%Y%m%d")

    @staticmethod
    def _coarsen_age(value: str) -> str:
        """Keep age but blunt its precision.

        Ages above 89 are aggregated because an exact very-high age is
        identifying in a small cohort (the HIPAA safe-harbour convention).
        """
        text = str(value or "").strip().upper()
        if len(text) == 4 and text[3] in "YMWD" and text[:3].isdigit():
            n = int(text[:3])
            if text[3] == "Y" and n > 89:
                return "090Y"
            return text
        return ""

    # --- main operation ----------------------------------------------------
    def anonymize_dataset(self, dataset: Any, *, patient_id_override: str | None = None) -> Any:
        """Return a de-identified copy of a pydicom ``Dataset``.

        The output dataset is built from scratch and populated only with
        allow-listed elements, so nothing can survive by accident.
        """
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian

        original_patient_id = str(getattr(dataset, "PatientID", "") or "")
        patient_key = patient_id_override or original_patient_id
        if not patient_key:
            # No patient id at all: fall back to the study UID so the study still
            # gets a stable pseudonym, and record that we did so.
            patient_key = str(getattr(dataset, "StudyInstanceUID", "") or "unknown")

        if self.refuse_burned_in_annotation:
            burned = str(getattr(dataset, "BurnedInAnnotation", "") or "").upper()
            if burned == "YES":
                raise DicomError(
                    "instance declares BurnedInAnnotation=YES; pixel data may "
                    "contain identifying text. Route it through pixel redaction "
                    "and manual review instead of automated de-identification."
                )

        out = Dataset()
        removed: dict[str, int] = {}
        shift = self.date_shift(patient_key)

        for element in dataset:
            keyword = pydicom.datadict.keyword_for_tag(element.tag)
            if element.tag.is_private or not keyword or keyword not in KEEP_TAGS:
                label = keyword or f"private_{element.tag}"
                removed[label] = removed.get(label, 0) + 1
                continue
            if keyword in _REWRITTEN:
                continue  # handled explicitly below
            # Sequences can nest identifiers arbitrarily deep. None of the
            # allow-listed tags are sequences, so a sequence here means a vendor
            # variant we have not vetted: drop it.
            if element.VR == "SQ":
                removed[f"{keyword}(SQ)"] = removed.get(f"{keyword}(SQ)", 0) + 1
                continue
            out[element.tag] = element

        # --- rewritten identifiers ---
        out.PatientID = self.pseudonym(patient_key)
        out.PatientName = out.PatientID
        out.StudyInstanceUID = self.map_uid(str(getattr(dataset, "StudyInstanceUID", "") or ""))
        out.SeriesInstanceUID = self.map_uid(str(getattr(dataset, "SeriesInstanceUID", "") or ""))
        out.SOPInstanceUID = self.map_uid(str(getattr(dataset, "SOPInstanceUID", "") or ""))
        for_uid = str(getattr(dataset, "FrameOfReferenceUID", "") or "")
        if for_uid:
            out.FrameOfReferenceUID = self.map_uid(for_uid)

        for date_tag in ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"):
            value = str(getattr(dataset, date_tag, "") or "")
            if value:
                setattr(out, date_tag, self._shift_date_string(value, shift))
        # Times are dropped: they add no analytic value and narrow re-identification.
        for time_tag in ("StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime"):
            if hasattr(dataset, time_tag):
                removed[time_tag] = removed.get(time_tag, 0) + 1

        if self.keep_sex:
            sex = str(getattr(dataset, "PatientSex", "") or "").upper()
            if sex in ("M", "F", "O"):
                out.PatientSex = sex
        if self.keep_age:
            age = self._coarsen_age(getattr(dataset, "PatientAge", ""))
            if age:
                out.PatientAge = age

        out.PatientIdentityRemoved = "YES"
        out.DeidentificationMethod = DEIDENTIFICATION_PROFILE

        # --- file meta ---
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = getattr(
            dataset, "SOPClassUID", getattr(dataset.file_meta, "MediaStorageSOPClassUID", "")
        )
        meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
        source_ts = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
        meta.TransferSyntaxUID = source_ts or ExplicitVRLittleEndian
        meta.ImplementationClassUID = f"{UID_ROOT}.1"
        out.file_meta = meta
        out.is_little_endian = True
        out.is_implicit_VR = False

        setattr(out, "_kneeai_removed", removed)
        return out

    def anonymize_files(
        self,
        paths: Iterable[str | os.PathLike[str]],
        output_dir: str | os.PathLike[str],
        *,
        patient_id_override: str | None = None,
    ) -> AnonymizationResult:
        """De-identify DICOM files into ``output_dir``.

        Output is laid out as ``<pseudonym>/<series_uid>/<sop_uid>.dcm`` so the
        de-identified tree is navigable without consulting the original.
        """
        import pydicom

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        result = AnonymizationResult(pseudonym="", date_shift_days=0)
        removed_totals: dict[str, int] = {}

        for path in paths:
            src = Path(path)
            try:
                dataset = pydicom.dcmread(str(src), force=False)
            except Exception as exc:  # noqa: BLE001 - untrusted input
                result.refused.append((src, f"unreadable: {type(exc).__name__}"))
                continue
            try:
                clean = self.anonymize_dataset(
                    dataset, patient_id_override=patient_id_override
                )
            except DicomError as exc:
                result.refused.append((src, str(exc)))
                continue

            if not result.pseudonym:
                result.pseudonym = str(clean.PatientID)
                result.date_shift_days = self.date_shift(
                    patient_id_override or str(getattr(dataset, "PatientID", "") or "")
                )

            for key, count in getattr(clean, "_kneeai_removed", {}).items():
                removed_totals[key] = removed_totals.get(key, 0) + count
            delattr(clean, "_kneeai_removed")

            target_dir = out_root / str(clean.PatientID) / str(clean.SeriesInstanceUID)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{clean.SOPInstanceUID}.dcm"
            clean.save_as(str(target), enforce_file_format=True)
            result.files_written.append(target)

        result.removed_tag_counts = dict(sorted(removed_totals.items()))
        result.uid_map = dict(self._uid_cache)
        result.provenance = Provenance.for_component(
            "imaging.preprocess",
            parameters={
                "operation": "anonymize",
                "profile": DEIDENTIFICATION_PROFILE,
                "files_in": len(result.files_written) + len(result.refused),
                "files_out": len(result.files_written),
                "keep_age": self.keep_age,
                "keep_sex": self.keep_sex,
            },
            notes=(
                "Keyed allow-list de-identification. The pseudonym mapping is "
                "recoverable only with the secret key, which is not stored with "
                "the output."
            ),
        )
        log.info("de-identification complete", fields=result.summary())
        if result.refused:
            log.warning(
                "some instances were refused during de-identification",
                fields={"count": len(result.refused)},
            )
        return result
