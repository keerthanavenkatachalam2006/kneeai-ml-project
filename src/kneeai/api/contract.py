"""Consumer-side access to the ML -> application contract.

The contract itself — schema, enum, validator, payload builder — is defined by
:mod:`kneeai.ml.contract` and specified in ``docs/ML_CONTRACT.md``. That module
is canonical and owned by the ML side. This one deliberately defines **no**
schema, **no** second source enum and **no** second validator; it re-exports the
canonical names and adds only what a consumer needs and the producer has no
reason to provide:

- :class:`CaseResultView` — a typed, read-only accessor over a *validated*
  payload, so the application works with :class:`~kneeai.core.anatomy.Structure`
  and :class:`~kneeai.core.anatomy.Side` rather than raw strings.
- :func:`write_result_json` / :func:`read_result_json` — file IO that validates
  through :func:`kneeai.ml.contract.validate_result_payload` on the way in and
  on the way out, so a malformed manifest is never written and never trusted.
- :func:`canonical_label_volume` — restricts a mask to the canonical label map
  before it becomes a contract artifact.

Every payload entering the application passes through
:func:`validate_result_payload`. Nothing here re-implements those checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import DEFAULT_LABEL_MAP, Side, Structure
from kneeai.core.errors import ValidationError
from kneeai.core.provenance import Provenance
from kneeai.imaging.volume import Volume

# --- canonical contract surface, re-exported unchanged ----------------------
from kneeai.ml.contract import (
    CONTRACT_VERSION,
    FORBIDDEN_METRIC_KEYS,
    MODEL_REGISTRY_RELATIVE_PATH,
    RESULT_FILENAME,
    SEGMENTATION_FILENAME,
    ArtifactSource,
    ContractReport,
    build_result_payload,
    validate_result_json,
    validate_result_payload,
)

__all__ = [
    # re-exported from kneeai.ml.contract (canonical)
    "CONTRACT_VERSION",
    "FORBIDDEN_METRIC_KEYS",
    "MODEL_REGISTRY_RELATIVE_PATH",
    "RESULT_FILENAME",
    "SEGMENTATION_FILENAME",
    "ArtifactSource",
    "ContractReport",
    "build_result_payload",
    "validate_result_json",
    "validate_result_payload",
    # consumer-side additions
    "MESH_DIRNAME",
    "MESH_MANIFEST_FILENAME",
    "CaseResultView",
    "canonical_label_volume",
    "read_result_json",
    "write_result_json",
]

#: Per-case mesh subdirectory. Not part of the ML contract — the application
#: writes the GLB meshes, so it owns this path.
MESH_DIRNAME = "meshes"
MESH_MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# Mask canonicalisation
# ---------------------------------------------------------------------------


def canonical_label_volume(mask: Volume) -> tuple[Volume, list[str]]:
    """Restrict a label volume to :data:`~kneeai.core.anatomy.DEFAULT_LABEL_MAP`.

    Contract rule 4 requires ``label_map`` to agree exactly with the canonical
    map, so a mask carrying structures the canonical map does not define cannot
    be published as-is. The phantom is the live example: it generates a fibula at
    label 10, which no canonical map defines and which nothing in the landmark,
    measurement, implant, planning or simulation path consumes.

    Rather than silently renumbering (which would mislabel anatomy) or failing
    (which would block the phantom backend entirely), the extra structures are
    dropped to background and named in the returned warnings so the exclusion
    appears in ``result.json``.

    Returns:
        ``(canonical_mask, warnings)``. The mask is returned unchanged when it is
        already canonical.
    """
    if not mask.is_label or mask.label_map is None:
        raise ValidationError("canonical_label_volume requires a label volume")

    incoming = {Structure(s): int(v) for s, v in mask.label_map.items()}
    extra = sorted(
        (s for s in incoming if s is not Structure.BACKGROUND and s not in DEFAULT_LABEL_MAP),
        key=lambda s: s.value,
    )
    misnumbered = sorted(
        (
            s
            for s in incoming
            if s in DEFAULT_LABEL_MAP and int(incoming[s]) != int(DEFAULT_LABEL_MAP[s])
        ),
        key=lambda s: s.value,
    )
    if not extra and not misnumbered:
        return mask, []

    warnings: list[str] = []
    array = np.asarray(mask.array)
    out = np.zeros_like(array)
    kept: dict[Structure, int] = {Structure.BACKGROUND: 0}
    for structure, label in incoming.items():
        if structure is Structure.BACKGROUND:
            continue
        if structure not in DEFAULT_LABEL_MAP:
            continue
        canonical = int(DEFAULT_LABEL_MAP[structure])
        out[array == int(label)] = canonical
        kept[structure] = canonical

    if extra:
        dropped_voxels = {
            s.value: int(np.count_nonzero(array == int(incoming[s]))) for s in extra
        }
        warnings.append(
            "Excluded from the published segmentation because "
            "core.anatomy.DEFAULT_LABEL_MAP does not define them: "
            + ", ".join(f"{name} ({count} voxels)" for name, count in dropped_voxels.items())
            + ". No measurement, implant, planning or simulation step uses these "
            "structures."
        )
    if misnumbered:
        warnings.append(
            "Relabelled to the canonical integers required by contract rule 4: "
            + ", ".join(
                f"{s.value} {incoming[s]}->{DEFAULT_LABEL_MAP[s]}" for s in misnumbered
            )
        )

    canonical_mask = Volume.from_labels(
        out.astype(np.int16),
        kept,
        reference=mask,
        metadata=dict(mask.metadata),
        provenance=mask.provenance,
    )
    return canonical_mask, warnings


# ---------------------------------------------------------------------------
# Typed read-only view
# ---------------------------------------------------------------------------


class CaseResultView:
    """Typed, read-only accessor over a validated ``result.json`` payload.

    Holds the payload itself as the single source of truth and converts on
    access. It intentionally cannot represent a manifest that the canonical
    validator would reject: construction validates, and there is no setter.
    """

    __slots__ = ("_payload", "_report")

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        subject: str = RESULT_FILENAME,
        base_dir: str | Path | None = None,
    ) -> None:
        report = validate_result_payload(payload, subject=subject, base_dir=base_dir)
        report.raise_if_failed()
        self._payload: dict[str, Any] = json.loads(json.dumps(payload))
        self._report = report

    # --- construction -------------------------------------------------
    @classmethod
    def read(cls, case_dir: str | Path) -> CaseResultView:
        """Load and validate ``result.json`` from a case directory."""
        directory = Path(case_dir)
        target = directory if directory.is_file() else directory / RESULT_FILENAME
        if not target.exists():
            raise ValidationError(
                f"{RESULT_FILENAME} not found at {target}. Run `kneeai demo run` or "
                "`kneeai infer` to produce one."
            )
        try:
            payload = json.loads(target.read_text())
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{target} is not valid JSON: {exc}") from exc
        return cls(payload, subject=str(target), base_dir=target.parent)

    # --- raw ----------------------------------------------------------
    @property
    def payload(self) -> dict[str, Any]:
        """The validated payload. Serve this to clients rather than re-deriving."""
        return json.loads(json.dumps(self._payload))

    @property
    def report(self) -> ContractReport:
        return self._report

    def section(self, name: str) -> dict[str, Any]:
        value = self._payload.get(name) or {}
        return dict(value) if isinstance(value, Mapping) else {}

    # --- identity -----------------------------------------------------
    @property
    def case_id(self) -> str:
        return str(self._payload["case_id"])

    @property
    def source(self) -> ArtifactSource:
        return ArtifactSource(str(self._payload["source"]))

    @property
    def badge(self) -> str:
        """Exact chip text for the UI. Part of the contract; rendered verbatim."""
        return self.source.badge

    @property
    def display_label(self) -> str:
        return self.source.display_label

    @property
    def is_model_output(self) -> bool:
        return self.source.is_model_output

    # --- model --------------------------------------------------------
    @property
    def model_id(self) -> str | None:
        return self.section("model").get("model_id")

    @property
    def model_version(self) -> str | None:
        return self.section("model").get("version")

    @property
    def model_card_path(self) -> str | None:
        return self.section("model").get("model_card")

    @property
    def model_key(self) -> str | None:
        if not self.model_id:
            return None
        return f"{self.model_id}:{self.model_version}" if self.model_version else self.model_id

    # --- segmentation -------------------------------------------------
    @property
    def label_map(self) -> dict[Structure, int]:
        raw = self.section("segmentation").get("label_map") or {}
        label_map = {Structure(k): int(v) for k, v in raw.items()}
        label_map.setdefault(Structure.BACKGROUND, 0)
        return label_map

    @property
    def structures_present(self) -> list[Structure]:
        return [
            Structure(s) for s in self.section("segmentation").get("structures_present", [])
        ]

    @property
    def structures_missing(self) -> list[Structure]:
        return [
            Structure(s)
            for s in self.section("segmentation").get("structures_expected_but_missing", [])
        ]

    @property
    def segmentation_relative_path(self) -> str:
        return str(self.section("segmentation").get("path", SEGMENTATION_FILENAME))

    # --- geometry -----------------------------------------------------
    @property
    def side(self) -> Side:
        return Side(str(self.section("geometry")["side"]))

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        spacing = self.section("geometry")["spacing_mm"]
        return (float(spacing[0]), float(spacing[1]), float(spacing[2]))

    @property
    def orientation(self) -> str:
        return str(self.section("geometry").get("orientation", ""))

    @property
    def image_shape(self) -> tuple[int, int, int]:
        shape = self.section("geometry")["image_shape"]
        return (int(shape[0]), int(shape[1]), int(shape[2]))

    @property
    def affine(self) -> NDArray[np.float64]:
        """The 4x4 voxel-index -> millimetre transform.

        The authority for all physical placement. Never multiply an index by
        spacing and call the result a position.
        """
        return np.asarray(self.section("geometry")["affine"], dtype=np.float64).reshape(4, 4)

    def index_to_mm(self, indices: Any) -> NDArray[np.float64]:
        """Map voxel indices to millimetres through the affine."""
        idx = np.asarray(indices, dtype=np.float64)
        flat = np.atleast_2d(idx)
        homogeneous = np.hstack([flat, np.ones((flat.shape[0], 1))])
        out = (self.affine @ homogeneous.T).T[:, :3]
        return out[0] if idx.ndim == 1 else out

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing_mm))

    # --- confidence ---------------------------------------------------
    @property
    def confidence_method(self) -> str:
        return str(self.section("confidence").get("method", ""))

    def confidence_of(self, structure: Structure) -> float | None:
        """Confidence for one structure, or ``None`` when unavailable.

        ``None`` means unavailable and must be rendered as such. It is not zero.
        """
        raw = (self.section("confidence").get("per_structure") or {}).get(structure.value)
        return None if raw is None else float(raw)

    def volume_mm3_of(self, structure: Structure) -> float | None:
        raw = (self.section("confidence").get("volume_mm3") or {}).get(structure.value)
        return None if raw is None else float(raw)

    @property
    def volumes_mm3(self) -> dict[Structure, float]:
        return {
            Structure(k): float(v)
            for k, v in (self.section("confidence").get("volume_mm3") or {}).items()
        }

    # --- provenance / warnings ----------------------------------------
    @property
    def provenance(self) -> Provenance:
        return Provenance.from_dict(self.section("provenance"))

    @property
    def warnings(self) -> list[str]:
        return [str(w) for w in self._payload.get("warnings", [])]

    # --- artifacts ----------------------------------------------------
    def segmentation_path(self, case_dir: str | Path) -> Path:
        """Resolve the mask path, refusing to escape the case directory."""
        directory = Path(case_dir).resolve()
        candidate = (directory / self.segmentation_relative_path).resolve()
        if not candidate.is_relative_to(directory):
            raise ValidationError(
                f"segmentation path {self.segmentation_relative_path!r} escapes the "
                "case directory"
            )
        return candidate

    def load_mask(self, case_dir: str | Path) -> Volume:
        """Read the label volume, checking it against the manifest's geometry.

        A ``result.json`` that describes a different volume than the one on disk
        is worse than no manifest: every downstream number would be computed on
        geometry nobody declared.
        """
        from kneeai.imaging.io import read_volume

        path = self.segmentation_path(case_dir)
        mask = read_volume(path, label_map=self.label_map, is_label=True)
        if mask.shape != self.image_shape:
            raise ValidationError(
                f"{path.name} has shape {mask.shape} but {RESULT_FILENAME} declares "
                f"{self.image_shape}"
            )
        on_disk = np.asarray(mask.voxel_to_image.matrix, dtype=np.float64)
        if not np.allclose(on_disk, self.affine, atol=1e-5):
            raise ValidationError(
                f"{path.name} affine does not match {RESULT_FILENAME}; the manifest "
                "describes a different volume"
            )
        if mask.side is not self.side:
            raise ValidationError(
                f"{path.name} declares side {mask.side.value!r} but "
                f"{RESULT_FILENAME} declares {self.side.value!r}"
            )
        return mask

    def mesh_dir(self, case_dir: str | Path) -> Path:
        return Path(case_dir) / MESH_DIRNAME

    # --- display ------------------------------------------------------
    def summary(self) -> str:
        present = ", ".join(s.value for s in self.structures_present) or "<none>"
        lines = [
            f"Case {self.case_id}  [{self.badge}]",
            f"  source        : {self.source.value} — {self.display_label}",
            f"  side          : {self.side.value}",
            f"  model         : {self.model_key or '<none>'}",
            f"  spacing       : {[round(s, 3) for s in self.spacing_mm]} mm "
            f"({self.orientation})",
            f"  shape         : {list(self.image_shape)}",
            f"  structures    : {present}",
            f"  confidence    : {self.confidence_method}",
        ]
        if self.structures_missing:
            lines.append(
                "  not segmented : " + ", ".join(s.value for s in self.structures_missing)
            )
        for warning in self.warnings:
            lines.append(f"  warning       : {warning}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CaseResultView({self.case_id}, source={self.source.value}, "
            f"side={self.side.value})"
        )


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------


def write_result_json(
    payload: Mapping[str, Any], case_dir: str | Path, *, indent: int = 2
) -> Path:
    """Validate then write ``result.json``.

    Validation happens before the write so a manifest that breaches the contract
    never lands on disk for something else to pick up.
    """
    directory = Path(case_dir)
    directory.mkdir(parents=True, exist_ok=True)
    validate_result_payload(payload, subject=RESULT_FILENAME).raise_if_failed()
    target = directory / RESULT_FILENAME
    # allow_nan=False: NaN and Infinity are not valid JSON, and a consumer that
    # accepted them would parse a non-number as a number.
    target.write_text(json.dumps(payload, indent=indent, allow_nan=False, default=str) + "\n")
    return target


def read_result_json(case_dir: str | Path) -> CaseResultView:
    """Load, validate and wrap a case's ``result.json``."""
    return CaseResultView.read(case_dir)
