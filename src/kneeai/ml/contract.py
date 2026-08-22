"""Machine-checkable ML -> application contract (see ``docs/ML_CONTRACT.md``).

The contract exists so the application can consume inference output without
importing model code. A prose document alone is not enough: it drifts, and neither
side notices until something downstream is quietly wrong. So the schema is
validated here, by both sides.

- The producer (:mod:`scripts.infer_case`) validates before writing, so a malformed
  manifest never reaches the application.
- The consumer validates before trusting, so a manifest produced by an older
  version is rejected rather than half-understood.

Torch-free by design. The application must be able to validate a manifest on a
machine with no CUDA toolchain.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kneeai.core.anatomy import DEFAULT_LABEL_MAP, Side, Structure
from kneeai.core.errors import ValidationError

__all__ = [
    "CONTRACT_VERSION",
    "ArtifactSource",
    "ContractReport",
    "FORBIDDEN_METRIC_KEYS",
    "MODEL_REGISTRY_RELATIVE_PATH",
    "RESULT_FILENAME",
    "SEGMENTATION_FILENAME",
    "build_result_payload",
    "validate_result_json",
    "validate_result_payload",
]

#: Bumped only by the process in docs/ML_CONTRACT.md §8.
CONTRACT_VERSION = "1"

SEGMENTATION_FILENAME = "segmentation.nii.gz"
RESULT_FILENAME = "result.json"

#: The registry path Agent 2 depends on. Relative to the artifact root.
MODEL_REGISTRY_RELATIVE_PATH = "models/seg_oaizib"

#: Keys that must never appear anywhere in result.json (contract rule 1).
#:
#: Every one of these requires ground truth, which does not exist at inference time
#: on a patient scan. Their presence means either a ground-truth leak into the
#: prediction path, or a number borrowed from another cohort and presented as this
#: patient's.
FORBIDDEN_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "dice",
        "dice_score",
        "iou",
        "jaccard",
        "hd95",
        "hausdorff",
        "hausdorff95",
        "assd",
        "asd",
        "surface_distance",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "f1_score",
        "auroc",
        "auprc",
        "ground_truth",
        "gt_path",
    }
)

_ORIENTATION_RE = re.compile(r"^[RLAPIS]{3}$")
_AXIS_GROUPS = (frozenset("RL"), frozenset("AP"), frozenset("IS"))


class ArtifactSource(str, Enum):
    """Provenance of a segmentation artifact (contract rule 2).

    Mandatory, and the application must surface it in every view and export.
    Presenting ground-truth or phantom output as an AI result misrepresents model
    capability to whoever is evaluating it.
    """

    AI_PREDICTION = "ai_prediction"
    GROUND_TRUTH = "ground_truth"
    PHANTOM = "phantom"
    MOCK = "mock"

    @property
    def is_model_output(self) -> bool:
        return self is ArtifactSource.AI_PREDICTION

    @property
    def badge(self) -> str:
        """Short uppercase chip text for the viewer.

        These exact strings are part of the contract: the application renders them
        verbatim, so they must not be reworded here without a contract change. Rule 2
        requires the source to be visible in every view and every export, and a badge
        is how that requirement is met in the UI.
        """
        return {
            ArtifactSource.AI_PREDICTION: "AI PREDICTION",
            ArtifactSource.GROUND_TRUTH: "GROUND TRUTH",
            ArtifactSource.PHANTOM: "PHANTOM",
            ArtifactSource.MOCK: "MOCK",
        }[self]

    @property
    def display_label(self) -> str:
        """Longer explanatory label, for report headers and tooltips.

        Distinct from :attr:`badge` on purpose. The badge has to fit in a chip; this
        carries the caveat that matters — that ground truth and phantom output are
        not model output — where there is room to say it.
        """
        return {
            ArtifactSource.AI_PREDICTION: "AI prediction",
            ArtifactSource.GROUND_TRUTH: "Expert ground truth (not model output)",
            ArtifactSource.PHANTOM: "Synthetic phantom (no patient)",
            ArtifactSource.MOCK: "Mock data (development only)",
        }[self]


@dataclass
class ContractReport:
    """Outcome of validating a manifest."""

    subject: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> ContractReport:
        self.errors.append(message)
        return self

    def warn(self, message: str) -> ContractReport:
        self.warnings.append(message)
        return self

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> ContractReport:
        if self.errors:
            raise ValidationError(
                f"{self.subject} does not satisfy ML contract v{CONTRACT_VERSION}:\n  - "
                + "\n  - ".join(self.errors)
            )
        return self

    def summary(self) -> str:
        lines = [
            f"{self.subject}: "
            + ("valid" if self.valid else f"{len(self.errors)} error(s)")
        ]
        lines.extend(f"  ERROR   {e}" for e in self.errors)
        lines.extend(f"  warning {w}" for w in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "contract_version": CONTRACT_VERSION,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _find_forbidden_keys(payload: Any, path: str = "") -> list[str]:
    """Recursively locate forbidden accuracy-metric keys."""
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key).strip().lower() in FORBIDDEN_METRIC_KEYS:
                found.append(here)
            found.extend(_find_forbidden_keys(value, here))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            found.extend(_find_forbidden_keys(item, f"{path}[{index}]"))
    return found


def _require(
    payload: Mapping[str, Any], key: str, report: ContractReport, context: str = ""
) -> Any:
    if key not in payload:
        where = f" in {context}" if context else ""
        report.error(f"missing required field {key!r}{where}")
        return None
    return payload[key]


def validate_result_payload(
    payload: Mapping[str, Any],
    *,
    subject: str = "result.json",
    base_dir: str | os.PathLike[str] | None = None,
) -> ContractReport:
    """Validate a decoded ``result.json`` payload against contract v1."""
    report = ContractReport(subject=subject)

    if not isinstance(payload, Mapping):
        return report.error("result.json must be a JSON object")

    # --- rule 1: no accuracy metrics anywhere ---
    forbidden = _find_forbidden_keys(payload)
    if forbidden:
        report.error(
            "accuracy metrics must not appear in result.json (contract rule 1) — "
            f"found: {forbidden}. These require ground truth, which does not exist "
            "at inference time. Reference metrics belong in the model card via "
            "ml.registry.ModelMetrics."
        )

    # --- version ---
    version = payload.get("contract_version")
    if version is None:
        report.error("missing required field 'contract_version'")
    elif str(version) != CONTRACT_VERSION:
        report.error(
            f"contract_version is {version!r} but this validator implements "
            f"v{CONTRACT_VERSION}. Reconcile both sides before proceeding."
        )

    case_id = _require(payload, "case_id", report)
    if case_id is not None:
        text = str(case_id)
        if not text.strip():
            report.error("case_id is empty")
        elif any(c in text for c in "/\\") or text in (".", ".."):
            report.error(
                f"case_id {text!r} is not filesystem-safe; it reaches a path join"
            )

    # --- rule 2: source ---
    raw_source = _require(payload, "source", report)
    source: ArtifactSource | None = None
    if raw_source is not None:
        try:
            source = ArtifactSource(str(raw_source))
        except ValueError:
            report.error(
                f"source {raw_source!r} is not one of "
                f"{[s.value for s in ArtifactSource]} (contract rule 2)"
            )

    _validate_model(payload, source, report)
    label_map = _validate_segmentation(payload, report, base_dir)
    _validate_geometry(payload, report)
    _validate_confidence(payload, label_map, report)
    _validate_provenance(payload, source, report)

    warnings = payload.get("warnings")
    if warnings is None:
        report.error("missing required field 'warnings' (use [] rather than null)")
    elif not isinstance(warnings, list) or any(
        not isinstance(w, str) for w in warnings
    ):
        report.error("warnings must be a list of strings")

    return report


def _validate_model(
    payload: Mapping[str, Any],
    source: ArtifactSource | None,
    report: ContractReport,
) -> None:
    model = payload.get("model")
    if not isinstance(model, Mapping):
        report.error("missing or invalid 'model' object")
        return
    for key in ("model_id", "version", "architecture", "model_card"):
        if key not in model:
            report.error(f"model.{key} is required (may be null)")

    if source is ArtifactSource.AI_PREDICTION:
        # An AI result that cannot name the model that produced it cannot be
        # audited, which defeats the traceability requirement (PRD §118).
        for key in ("model_id", "version"):
            if not model.get(key):
                report.error(
                    f"model.{key} must be set when source is 'ai_prediction': an "
                    "unattributed AI output cannot be audited"
                )
    elif model.get("model_id"):
        report.warn(
            f"source is {source.value if source else 'unknown'} but model.model_id is "
            f"set to {model['model_id']!r}. Non-AI artifacts should not claim a model."
        )


def _validate_segmentation(
    payload: Mapping[str, Any],
    report: ContractReport,
    base_dir: str | os.PathLike[str] | None,
) -> dict[str, int]:
    segmentation = payload.get("segmentation")
    if not isinstance(segmentation, Mapping):
        report.error("missing or invalid 'segmentation' object")
        return {}

    path = segmentation.get("path")
    if path != SEGMENTATION_FILENAME:
        report.error(
            f"segmentation.path must be {SEGMENTATION_FILENAME!r}, got {path!r}"
        )
    elif base_dir is not None:
        target = Path(base_dir) / str(path)
        if not target.exists():
            report.error(f"segmentation file is missing from disk: {target}")

    # --- rule 4: label_map ---
    raw_map = segmentation.get("label_map")
    label_map: dict[str, int] = {}
    if not isinstance(raw_map, Mapping) or not raw_map:
        report.error(
            "segmentation.label_map is required and must be a non-empty object "
            "(contract rule 4)"
        )
    else:
        canonical = {s.value: v for s, v in DEFAULT_LABEL_MAP.items()}
        for name, value in raw_map.items():
            try:
                label = int(value)
            except (TypeError, ValueError):
                report.error(f"label_map[{name!r}] is not an integer: {value!r}")
                continue
            label_map[str(name)] = label
            if str(name) not in canonical:
                report.error(
                    f"label_map contains {name!r}, which is not a known "
                    "kneeai.core.anatomy.Structure"
                )
            elif canonical[str(name)] != label:
                report.error(
                    f"label_map assigns {name!r} = {label}, but "
                    f"core.anatomy.DEFAULT_LABEL_MAP assigns {canonical[str(name)]}. "
                    "Label maps must match exactly — a different ordering produces "
                    "silently mislabelled anatomy (contract rule 4)."
                )
        if len(set(label_map.values())) != len(label_map):
            report.error(f"label_map is not injective: {label_map}")
        if 0 in label_map.values():
            report.error("label 0 is reserved for background and must not be mapped")

    present = segmentation.get("structures_present")
    if not isinstance(present, list):
        report.error("segmentation.structures_present must be a list")
    else:
        unknown = [s for s in present if s not in label_map]
        if unknown:
            report.error(
                f"structures_present contains {unknown}, which are absent from "
                "label_map"
            )
        if not present:
            report.warn(
                "structures_present is empty: the mask contains no labelled voxels"
            )

    missing = segmentation.get("structures_expected_but_missing")
    if not isinstance(missing, list):
        report.error("segmentation.structures_expected_but_missing must be a list")
    else:
        overlap = sorted(set(missing) & set(present or []))
        if overlap:
            report.error(
                f"{overlap} appear in both structures_present and "
                "structures_expected_but_missing"
            )

    return label_map


def _validate_geometry(payload: Mapping[str, Any], report: ContractReport) -> None:
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping):
        report.error("missing or invalid 'geometry' object")
        return

    spacing = geometry.get("spacing_mm")
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
        report.error("geometry.spacing_mm must be a list of 3 values")
    else:
        for index, value in enumerate(spacing):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                report.error(f"spacing_mm[{index}] is not numeric: {value!r}")
                continue
            if not math.isfinite(numeric) or numeric <= 0:
                report.error(
                    f"spacing_mm[{index}] must be finite and positive, got {numeric}"
                )

    orientation = geometry.get("orientation")
    if not isinstance(orientation, str) or not _ORIENTATION_RE.match(orientation):
        report.error(
            f"geometry.orientation must be three letters from R/L, A/P, I/S; got "
            f"{orientation!r}"
        )
    else:
        seen = {next(g for g in _AXIS_GROUPS if c in g) for c in orientation}
        if len(seen) != 3:
            report.error(
                f"geometry.orientation {orientation!r} does not name three distinct "
                "anatomical axes"
            )

    # --- rule 5: affine ---
    affine = geometry.get("affine")
    if not isinstance(affine, (list, tuple)) or len(affine) != 4:
        report.error(
            "geometry.affine is required and must be a 4x4 row-major matrix "
            "(contract rule 5): voxel indices are not millimetres"
        )
    else:
        for row_index, row in enumerate(affine):
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                report.error(f"affine row {row_index} must have 4 elements")
                continue
            for column_index, value in enumerate(row):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    report.error(
                        f"affine[{row_index}][{column_index}] is not numeric: {value!r}"
                    )
                    continue
                if not math.isfinite(numeric):
                    report.error(
                        f"affine[{row_index}][{column_index}] is not finite"
                    )
        try:
            bottom = [float(v) for v in affine[3]]
            if bottom != [0.0, 0.0, 0.0, 1.0]:
                report.error(
                    f"the bottom row of a homogeneous affine must be [0,0,0,1], got "
                    f"{bottom}"
                )
        except (TypeError, ValueError, IndexError):
            pass

    shape = geometry.get("image_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        report.error("geometry.image_shape must be a list of 3 integers")
    else:
        for index, value in enumerate(shape):
            if not isinstance(value, int) or value <= 0:
                report.error(
                    f"image_shape[{index}] must be a positive integer, got {value!r}"
                )

    # --- rule 3: side ---
    raw_side = geometry.get("side")
    if raw_side is None:
        report.error("geometry.side is required (contract rule 3)")
    else:
        try:
            side = Side(str(raw_side))
        except ValueError:
            report.error(
                f"geometry.side must be one of {[s.value for s in Side]}, got "
                f"{raw_side!r}"
            )
        else:
            if side is Side.UNKNOWN:
                warnings = payload.get("warnings") or []
                mentions_side = any(
                    "side" in str(w).lower() or "lateral" in str(w).lower()
                    for w in warnings
                )
                if not mentions_side:
                    report.error(
                        "side is 'unknown', which blocks every medial/lateral "
                        "resolution downstream, but no warning explains it. An "
                        "unknown side must be surfaced, not silently tolerated "
                        "(contract rule 3)."
                    )
                else:
                    report.warn(
                        "side is 'unknown': planning is blocked until it is "
                        "established. This is a wrong-site risk."
                    )


def _validate_confidence(
    payload: Mapping[str, Any],
    label_map: Mapping[str, int],
    report: ContractReport,
) -> None:
    confidence = payload.get("confidence")
    if not isinstance(confidence, Mapping):
        report.error("missing or invalid 'confidence' object")
        return

    per_structure = confidence.get("per_structure")
    if not isinstance(per_structure, Mapping):
        report.error("confidence.per_structure must be an object")
    else:
        for name, value in per_structure.items():
            if name not in label_map:
                report.error(
                    f"confidence.per_structure has {name!r}, absent from label_map"
                )
            if value is None:
                continue  # permitted: a structure with no voxels has no confidence
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                report.error(f"confidence for {name!r} is not numeric: {value!r}")
                continue
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                report.error(
                    f"confidence for {name!r} must be in [0, 1] or null, got {numeric}"
                )

    method = confidence.get("method")
    if not isinstance(method, str) or not method.strip():
        report.error(
            "confidence.method is required and must describe how the number was "
            "computed: 'confidence' means nothing on its own"
        )

    volumes = confidence.get("volume_mm3")
    if not isinstance(volumes, Mapping):
        report.error("confidence.volume_mm3 must be an object")
    else:
        for name, value in volumes.items():
            if name not in label_map:
                report.error(
                    f"confidence.volume_mm3 has {name!r}, absent from label_map"
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                report.error(f"volume for {name!r} is not numeric: {value!r}")
                continue
            if numeric < 0 or not math.isfinite(numeric):
                report.error(f"volume for {name!r} must be finite and >= 0")


def _validate_provenance(
    payload: Mapping[str, Any],
    source: ArtifactSource | None,
    report: ContractReport,
) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        report.error("missing or invalid 'provenance' object (contract rule 6)")
        return
    for key in ("component", "component_version", "origin", "created_at"):
        if not provenance.get(key):
            report.error(f"provenance.{key} is required")
    if source is ArtifactSource.AI_PREDICTION and provenance.get("origin") != "ai":
        report.error(
            f"source is 'ai_prediction' but provenance.origin is "
            f"{provenance.get('origin')!r}; it must be 'ai'"
        )


def validate_result_json(path: str | os.PathLike[str]) -> ContractReport:
    """Load and validate a ``result.json`` from disk."""
    target = Path(path)
    report = ContractReport(subject=str(target))
    if not target.exists():
        return report.error(f"file does not exist: {target}")
    try:
        payload = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        return report.error(f"not valid JSON: {exc}")
    loaded = validate_result_payload(
        payload, subject=str(target), base_dir=target.parent
    )
    loaded.warnings.extend(report.warnings)
    return loaded


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_result_payload(
    *,
    case_id: str,
    source: ArtifactSource,
    mask: Any,
    confidence: Mapping[Structure, float | None],
    confidence_method: str,
    provenance: Any,
    model_id: str | None = None,
    model_version: str | None = None,
    architecture: str | None = None,
    model_card: str | None = None,
    structures_expected_but_missing: Iterable[Structure] = (),
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble a contract-compliant ``result.json`` payload.

    Args:
        mask: a :class:`kneeai.imaging.Volume` label volume, already on the source
            acquisition grid.

    Deriving ``structures_present`` and the volumes from the mask itself — rather
    than from what the model was asked to predict — is deliberate: a structure the
    model was asked for but did not find must not be reported as present.
    """
    if not mask.is_label or mask.label_map is None:
        raise ValidationError("build_result_payload requires a label Volume")

    label_map = {
        structure.value: int(value)
        for structure, value in mask.label_map.items()
        if structure is not Structure.BACKGROUND
    }
    present = sorted(s.value for s in mask.structures_present())

    volumes = {
        structure.value: round(mask.structure_volume_mm3(structure), 1)
        for structure in mask.label_map
        if structure is not Structure.BACKGROUND
    }

    per_structure: dict[str, float | None] = {}
    for structure in mask.label_map:
        if structure is Structure.BACKGROUND:
            continue
        value = confidence.get(structure)
        if value is None or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            per_structure[structure.value] = None
        else:
            per_structure[structure.value] = round(float(value), 4)

    missing = sorted({s.value for s in structures_expected_but_missing})
    all_warnings = list(warnings)
    if missing:
        readable = ", ".join(
            Structure(name).display_name for name in missing
        )
        note = (
            f"This model does not segment: {readable}. Measurements depending on "
            "those structures are unavailable for this case."
        )
        if note not in all_warnings:
            all_warnings.append(note)
    if mask.side is Side.UNKNOWN:
        all_warnings.append(
            "The operative side could not be established from the data. Medial and "
            "lateral cannot be resolved, so planning is blocked until it is set. "
            "This is a wrong-site risk."
        )

    return {
        "contract_version": CONTRACT_VERSION,
        "case_id": case_id,
        "source": source.value,
        "model": {
            "model_id": model_id,
            "version": model_version,
            "architecture": architecture,
            "model_card": model_card,
        },
        "segmentation": {
            "path": SEGMENTATION_FILENAME,
            "label_map": label_map,
            "structures_present": present,
            "structures_expected_but_missing": missing,
        },
        "geometry": {
            "spacing_mm": [round(float(s), 6) for s in mask.spacing],
            "orientation": mask.orientation,
            "affine": [
                [round(float(v), 8) for v in row] for row in mask.voxel_to_image.matrix
            ],
            "image_shape": [int(n) for n in mask.shape],
            "side": mask.side.value,
        },
        "confidence": {
            "per_structure": per_structure,
            "method": confidence_method,
            "volume_mm3": volumes,
        },
        "provenance": provenance.to_dict() if hasattr(provenance, "to_dict") else dict(provenance),
        "warnings": all_warnings,
    }
