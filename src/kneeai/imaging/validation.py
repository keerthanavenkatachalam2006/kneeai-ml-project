"""Study validation (PRD §10, §62, §84).

Validation is a *gate*, not a formality. A study that fails validation must not
reach the AI pipeline, because a segmentation model will happily produce a
confident, entirely wrong mask on a mis-spaced or partially-missing volume, and
the resulting plan looks as credible as a correct one.

The report separates severities so a case with only warnings can proceed with
those warnings surfaced in the UI, while errors block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np

from kneeai.core.anatomy import Modality, Side, StudyType
from kneeai.core.errors import DicomValidationError
from kneeai.imaging.dicom import DicomSeries, DicomStudy
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger

__all__ = [
    "Severity",
    "Issue",
    "ValidationReport",
    "ValidationLimits",
    "validate_series",
    "validate_volume",
    "validate_study",
]

log = get_logger("imaging.validation")


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def blocks(self) -> bool:
        return self is Severity.ERROR


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    #: What the operator should do about it. Present on every blocking issue.
    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
            "remedy": self.remedy,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    subject: str
    issues: list[Issue] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        *,
        remedy: str = "",
        **detail: Any,
    ) -> None:
        self.issues.append(
            Issue(code=code, severity=severity, message=message, detail=detail, remedy=remedy)
        )

    def info(self, code: str, message: str, **detail: Any) -> None:
        self.add(code, Severity.INFO, message, **detail)

    def warn(self, code: str, message: str, *, remedy: str = "", **detail: Any) -> None:
        self.add(code, Severity.WARNING, message, remedy=remedy, **detail)

    def error(self, code: str, message: str, *, remedy: str = "", **detail: Any) -> None:
        self.add(code, Severity.ERROR, message, remedy=remedy, **detail)

    def extend(self, other: ValidationReport) -> None:
        self.issues.extend(other.issues)
        for key, value in other.facts.items():
            self.facts.setdefault(f"{other.subject}.{key}", value)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> ValidationReport:
        if self.errors:
            lines = [f"  - {i.code}: {i.message}" for i in self.errors]
            remedies = [f"  * {i.remedy}" for i in self.errors if i.remedy]
            msg = f"{self.subject} failed validation:\n" + "\n".join(lines)
            if remedies:
                msg += "\nSuggested action:\n" + "\n".join(remedies)
            raise DicomValidationError(msg)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
            "facts": self.facts,
        }


@dataclass(frozen=True)
class ValidationLimits:
    """Acceptance thresholds.

    Defaults reflect what the segmentation and measurement stack can actually
    handle rather than what a scanner can produce. They are configurable per
    deployment because protocols differ between sites; they are *not* silently
    relaxed at runtime.
    """

    min_instances_3d: int = 20
    max_in_plane_spacing_mm: float = 1.5
    max_slice_gap_mm: float = 3.0
    #: Coefficient of variation of slice gaps above which spacing is "irregular".
    max_slice_gap_cv: float = 0.05
    #: Ratio of largest gap to median gap above which a slice is likely missing.
    missing_slice_ratio: float = 1.5
    #: Anisotropy beyond this degrades 3D segmentation noticeably.
    max_anisotropy: float = 4.0
    min_fov_mm: float = 80.0
    max_fov_mm: float = 600.0
    #: MR studies for cartilage/meniscus work need reasonably fine slices.
    max_slice_thickness_mr_mm: float = 2.0
    require_laterality: bool = True


DEFAULT_LIMITS = ValidationLimits()


# ---------------------------------------------------------------------------
# Series-level
# ---------------------------------------------------------------------------


def validate_series(
    series: DicomSeries, *, limits: ValidationLimits = DEFAULT_LIMITS
) -> ValidationReport:
    """Validate series metadata before any pixel data is loaded."""
    report = ValidationReport(subject=f"series {series.series_instance_uid}")
    report.facts.update(series.summary())

    # --- modality ---
    if series.modality is Modality.UNKNOWN:
        report.error(
            "modality_unknown",
            "series modality could not be determined",
            remedy="Confirm the source export includes the Modality tag (0008,0060).",
        )
    elif series.modality not in (Modality.MR, Modality.CT, Modality.XRAY, Modality.DX):
        report.error(
            "modality_unsupported",
            f"modality {series.modality.value} is not supported",
            remedy="Upload an MR, CT or radiographic series.",
            modality=series.modality.value,
        )

    # --- carry forward instance-level issues ---
    for issue in series.issues:
        report.warn("instance_metadata", issue)

    duplicate_uids = _duplicates(i.sop_instance_uid for i in series.instances if i.sop_instance_uid)
    if duplicate_uids:
        report.error(
            "duplicate_sop_instances",
            f"{len(duplicate_uids)} SOPInstanceUID(s) appear more than once",
            remedy="Re-export the study; duplicated instances corrupt slice ordering.",
            count=len(duplicate_uids),
        )

    frames_of_reference = {
        i.orientation for i in series.instances if i.orientation is not None
    }
    if len(frames_of_reference) > 1:
        # Differing orientation within one series means it is not a single
        # geometric stack, whatever the SeriesInstanceUID says.
        report.error(
            "mixed_orientation",
            f"series contains {len(frames_of_reference)} distinct image orientations",
            remedy="Split the export into one series per acquisition plane.",
            n_orientations=len(frames_of_reference),
        )

    if not series.modality.is_volumetric:
        _validate_radiograph(series, report, limits)
        _validate_laterality(series, report, limits)
        return report

    # --- volumetric geometry ---
    if series.n_instances < limits.min_instances_3d:
        report.error(
            "too_few_slices",
            f"series has {series.n_instances} instances, minimum is "
            f"{limits.min_instances_3d} for 3D analysis",
            remedy="Upload the full 3D acquisition rather than a subset of slices.",
            n_instances=series.n_instances,
        )

    if series.pixel_spacing is None:
        report.error(
            "missing_pixel_spacing",
            "PixelSpacing is absent; physical measurements are impossible",
            remedy="Re-export with PixelSpacing (0028,0030) preserved.",
        )
    else:
        in_plane = max(series.pixel_spacing)
        if in_plane > limits.max_in_plane_spacing_mm:
            report.warn(
                "coarse_in_plane_spacing",
                f"in-plane spacing {in_plane:.2f} mm exceeds "
                f"{limits.max_in_plane_spacing_mm} mm; cartilage and meniscus "
                "measurements will carry higher error",
                remedy="Prefer a higher-resolution 3D sequence where available.",
                in_plane_spacing_mm=in_plane,
            )

    gaps = series.slice_spacings()
    if gaps.size:
        median_gap = float(np.median(gaps))
        report.facts["median_slice_gap_mm"] = median_gap
        if median_gap > limits.max_slice_gap_mm:
            report.error(
                "slice_gap_too_large",
                f"median slice gap {median_gap:.2f} mm exceeds "
                f"{limits.max_slice_gap_mm} mm",
                remedy="Upload a thinner-slice acquisition.",
                median_gap_mm=median_gap,
            )
        if median_gap > 1e-6:
            cv = float(gaps.std() / median_gap)
            report.facts["slice_gap_cv"] = cv
            if cv > limits.max_slice_gap_cv:
                report.error(
                    "irregular_slice_spacing",
                    f"slice spacing is irregular (CV {cv:.3f}); the volume cannot "
                    "be treated as uniformly sampled",
                    remedy=(
                        "Check for a mixed or interleaved export. Resampling an "
                        "irregular stack as if uniform distorts every measurement."
                    ),
                    slice_gap_cv=cv,
                )
            largest = float(gaps.max())
            if largest > limits.missing_slice_ratio * median_gap:
                n_missing = int(round(largest / median_gap)) - 1
                report.error(
                    "missing_slices",
                    f"a {largest:.2f} mm gap against a {median_gap:.2f} mm median "
                    f"suggests ~{n_missing} missing slice(s)",
                    remedy="Re-export the complete series.",
                    largest_gap_mm=largest,
                    estimated_missing=n_missing,
                )

        if series.slice_thickness and series.modality is Modality.MR:
            if series.slice_thickness > limits.max_slice_thickness_mr_mm:
                report.warn(
                    "thick_mr_slices",
                    f"MR slice thickness {series.slice_thickness:.2f} mm exceeds "
                    f"{limits.max_slice_thickness_mr_mm} mm",
                    remedy="Use a 3D sequence (e.g. DESS/CUBE) for cartilage work.",
                    slice_thickness_mm=series.slice_thickness,
                )
    else:
        report.warn(
            "no_slice_geometry",
            "slice positions unavailable; spacing regularity cannot be checked",
        )

    _validate_laterality(series, report, limits)
    return report


def _validate_radiograph(
    series: DicomSeries, report: ValidationReport, limits: ValidationLimits
) -> None:
    if series.pixel_spacing is None:
        report.warn(
            "radiograph_uncalibrated",
            "radiograph has no PixelSpacing; absolute measurements and implant "
            "templating require a calibration marker or ImagerPixelSpacing",
            remedy="Include a calibration object or supply the imager pixel spacing.",
        )
    if series.study_type is StudyType.LONG_LEG_XRAY:
        report.info("long_leg_detected", "treated as a long-leg alignment radiograph")


def _validate_laterality(
    series: DicomSeries, report: ValidationReport, limits: ValidationLimits
) -> None:
    side = series.side
    if side is Side.UNKNOWN:
        if limits.require_laterality:
            report.error(
                "laterality_unknown",
                "which knee this study shows could not be determined",
                remedy=(
                    "Set laterality explicitly on the case. Planning a "
                    "side-specific implant without confirmed laterality is a "
                    "wrong-site risk."
                ),
            )
        else:
            report.warn("laterality_unknown", "laterality unknown")
    elif not (series.laterality or "").strip():
        report.warn(
            "laterality_inferred",
            f"laterality inferred as {side.value} from the series description, "
            "not from a DICOM laterality tag",
            remedy="Have the surgeon confirm laterality before approving the plan.",
            inferred_side=side.value,
        )


# ---------------------------------------------------------------------------
# Volume-level
# ---------------------------------------------------------------------------


def validate_volume(
    volume: Volume, *, limits: ValidationLimits = DEFAULT_LIMITS
) -> ValidationReport:
    """Validate a loaded volume's geometry and intensities."""
    report = ValidationReport(subject="volume")
    report.facts.update(volume.summary())

    extent = volume.physical_extent_mm
    if float(extent.min()) < limits.min_fov_mm:
        report.warn(
            "small_field_of_view",
            f"smallest field-of-view dimension is {float(extent.min()):.0f} mm; "
            "the joint may be incompletely covered",
            remedy="Confirm the acquisition covers the full distal femur and proximal tibia.",
            extent_mm=[round(float(e), 1) for e in extent],
        )
    if float(extent.max()) > limits.max_fov_mm:
        report.info(
            "large_field_of_view",
            f"field of view spans {float(extent.max()):.0f} mm; ROI cropping will "
            "be applied before segmentation",
        )

    anisotropy = float(volume.spacing.max() / volume.spacing.min())
    report.facts["anisotropy"] = anisotropy
    if anisotropy > limits.max_anisotropy:
        report.warn(
            "high_anisotropy",
            f"voxel anisotropy is {anisotropy:.1f}:1; 3D segmentation accuracy "
            "degrades and out-of-plane boundaries will be interpolated",
            remedy="Prefer a near-isotropic 3D sequence.",
            anisotropy=anisotropy,
        )

    if volume.is_label:
        _validate_mask(volume, report)
        return report

    try:
        stats = volume.intensity_stats()
    except Exception as exc:  # noqa: BLE001
        report.error(
            "intensity_unreadable",
            f"voxel intensities could not be summarised: {exc}",
            remedy="Re-export the series; the pixel data may be corrupt.",
        )
        return report

    report.facts["intensity"] = stats
    if not np.all(np.isfinite(volume.array)):
        report.error(
            "non_finite_voxels",
            "volume contains NaN or infinite voxels",
            remedy="Re-export the series or repair the source data.",
        )
    if stats["max"] == stats["min"]:
        report.error(
            "constant_image",
            f"every voxel has the same value ({stats['min']}); the image is blank",
            remedy="Check the export; this series carries no image content.",
        )
    elif stats["std"] < 1e-3:
        report.warn("near_constant_image", "image has almost no intensity variation")

    if volume.modality is Modality.CT:
        # A CT that never reaches cortical-bone HU is either not a CT or has been
        # rescaled, which breaks any HU-based bone thresholding.
        if stats["p99"] < 300.0:
            report.warn(
                "ct_intensity_range_unexpected",
                f"CT 99th percentile is {stats['p99']:.0f} HU; cortical bone is "
                "normally well above 300 HU. Rescale slope/intercept may be missing",
                remedy="Verify RescaleSlope/RescaleIntercept were applied on export.",
                p99=stats["p99"],
            )

    if volume.side is Side.UNKNOWN:
        report.warn(
            "volume_side_unknown",
            "volume carries no laterality; it must be set on the case before planning",
        )
    return report


def _validate_mask(volume: Volume, report: ValidationReport) -> None:
    """Checks specific to label volumes (PRD §62: empty masks, missing labels)."""
    if volume.label_map is None:  # pragma: no cover - guarded by Volume
        report.error("mask_without_label_map", "label volume has no label map")
        return

    present = set(volume.structures_present())
    declared = {s for s in volume.label_map if s.value != "background"}
    missing = sorted(s.value for s in declared - present)
    if missing:
        report.error(
            "empty_structures",
            f"the label map declares {len(missing)} structure(s) with no voxels: "
            f"{', '.join(missing)}",
            remedy=(
                "Do not reconstruct or measure from this mask. Re-run segmentation, "
                "or narrow the label map to the structures actually predicted."
            ),
            missing=missing,
        )

    unexpected = sorted(
        int(v) for v in np.unique(volume.array) if int(v) not in set(volume.label_map.values())
    )
    if unexpected:
        report.error(
            "unmapped_labels",
            f"mask contains label value(s) {unexpected} absent from the label map",
            remedy="The mask and label map disagree; check the model's label ordering.",
            unmapped=unexpected,
        )

    for structure in sorted(present, key=lambda s: s.value):
        vol_mm3 = volume.structure_volume_mm3(structure)
        report.facts[f"volume_mm3.{structure.value}"] = round(vol_mm3, 1)
        if vol_mm3 < 10.0:
            report.warn(
                "tiny_structure",
                f"{structure.display_name} occupies only {vol_mm3:.1f} mm³, which "
                "is almost certainly a spurious component",
                remedy="Apply largest-connected-component post-processing.",
                structure=structure.value,
                volume_mm3=vol_mm3,
            )


# ---------------------------------------------------------------------------
# Study-level
# ---------------------------------------------------------------------------


def validate_study(
    study: DicomStudy,
    *,
    limits: ValidationLimits = DEFAULT_LIMITS,
    expected_side: Side | None = None,
) -> ValidationReport:
    """Validate a whole study and identify whether a usable series exists."""
    report = ValidationReport(subject=f"study {study.study_instance_uid}")
    report.facts["n_series"] = len(study.series)
    report.facts["skipped_files"] = len(study.skipped_files)

    if not study.series:
        report.error(
            "no_series",
            "no DICOM series were found",
            remedy="Check the upload contains DICOM files, not a proprietary export.",
        )
        return report

    if study.skipped_files:
        report.warn(
            "unreadable_files",
            f"{len(study.skipped_files)} file(s) could not be parsed and were ignored",
            count=len(study.skipped_files),
        )

    per_series = [validate_series(s, limits=limits) for s in study.series]
    usable = [
        s for s, rep in zip(study.series, per_series) if rep.passed and s.is_volumetric_candidate
    ]
    radiographs = [
        s
        for s, rep in zip(study.series, per_series)
        if rep.passed and s.modality in (Modality.XRAY, Modality.DX)
    ]

    if not usable and not radiographs:
        report.error(
            "no_usable_series",
            "no series in this study passed validation",
            remedy="Review the per-series issues below and re-export.",
        )
        # Surface why each series failed, so the operator has something to act on.
        for series, rep in zip(study.series, per_series):
            for issue in rep.errors:
                report.add(
                    f"series.{issue.code}",
                    issue.severity,
                    f"series {series.series_number or '?'} "
                    f"({series.description or 'no description'}): {issue.message}",
                    remedy=issue.remedy,
                    series_uid=series.series_instance_uid,
                )
    else:
        report.facts["usable_series"] = [s.series_instance_uid for s in usable]
        report.facts["radiograph_series"] = [s.series_instance_uid for s in radiographs]

    sides = {s.side for s in study.series if s.side is not Side.UNKNOWN}
    if len(sides) > 1:
        report.error(
            "mixed_laterality",
            f"study contains series from both knees: {sorted(s.value for s in sides)}",
            remedy="Split into one case per knee. A case represents one operative side.",
        )
    elif expected_side is not None and sides and expected_side not in sides:
        report.error(
            "laterality_mismatch",
            f"case is registered as {expected_side.value} but the imaging indicates "
            f"{sorted(s.value for s in sides)}",
            remedy="Resolve the discrepancy before proceeding. This is a wrong-site risk.",
        )

    log.info(
        "study validated",
        fields={
            "passed": report.passed,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
        },
    )
    return report


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for v in values:
        if v in seen:
            dupes.add(v)
        seen.add(v)
    return sorted(dupes)
