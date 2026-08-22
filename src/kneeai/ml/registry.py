"""Model registry and model cards (PRD §69, §113, §118).

PRD §113 (Rule 1) forbids training without a dataset contract. This module makes
that structural: a :class:`ModelCard` cannot be constructed without a
:class:`DatasetContract`, and the registry refuses to register a model without a
card. A checkpoint with no card is not loadable through this API.

The card also carries the label map and preprocessing spec the model was trained
with. :func:`ModelCard.check_compatibility` is called before inference, which is
what stops a checkpoint trained with ``{femur:1, tibia:2}`` from being used
against a pipeline expecting ``{femur:1, femoral_cartilage:2}`` — a mismatch that
produces confident, anatomically scrambled output and no error.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kneeai.core.anatomy import Modality, Structure
from kneeai.core.errors import ConfigurationError, ValidationError
from kneeai.core.provenance import ArtifactRef, hash_file, now_utc
from kneeai.logging import get_logger
from kneeai.version import PLATFORM_VERSION

__all__ = [
    "ModelTask",
    "Architecture",
    "DatasetContract",
    "ModelMetrics",
    "ModelCard",
    "ModelRegistry",
    "get_registry",
]

log = get_logger("ml.registry")

CARD_FILENAME = "model_card.json"


class ModelTask(str, Enum):
    """What the model does. Determines how its output is consumed."""

    SEGMENTATION = "segmentation"
    LANDMARK = "landmark"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SHAPE = "shape"
    IMPLANT_RANKING = "implant_ranking"


class Architecture(str, Enum):
    NNUNET = "nnunet"
    UNET = "unet"
    SEGRESNET = "segresnet"
    UNETR = "unetr"
    SWIN_UNETR = "swin_unetr"
    ATTENTION_UNET = "attention_unet"
    HEATMAP_3D = "heatmap_3d"
    RESNET3D = "resnet3d"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC = "logistic"


@dataclass(frozen=True)
class DatasetContract:
    """The dataset contract required by PRD §113 before any training run.

    Every field answers one of the questions that rule demands. They are
    mandatory because the answers are unrecoverable after the fact: nobody
    reconstructs "who annotated this" six months later.
    """

    #: What the model consumes, e.g. "3D DESS knee MR, 0.5 mm isotropic, 1 channel".
    input_description: str

    #: What it emits, e.g. "10-class voxel label map".
    output_description: str

    #: How ground truth was established.
    ground_truth_description: str

    #: Who or what annotated it (e.g. "ZIB expert manual", "propagated from OAI-ZIB").
    annotator: str

    #: Dataset ids as registered in the dataset registry.
    dataset_ids: tuple[str, ...]

    #: Distinct patients — not studies, not slices (PRD §63).
    n_patients: int

    n_studies: int

    #: Split sizes in patients: {"train": n, "val": n, "test": n}.
    split_patients: Mapping[str, int]

    #: Split strategy, which must be patient-level.
    split_strategy: str = "patient_level_random"

    #: Licence(s) of the training data, and whether commercial use is permitted.
    license: str = "unverified"
    commercial_use_permitted: bool | None = None

    #: Datasets held out entirely for external validation (PRD §64).
    external_test_dataset_ids: tuple[str, ...] = ()

    #: Known population characteristics, for bias assessment.
    population_notes: str = ""

    def __post_init__(self) -> None:
        for name in (
            "input_description",
            "output_description",
            "ground_truth_description",
            "annotator",
        ):
            if not str(getattr(self, name)).strip():
                raise ValidationError(
                    f"dataset contract field {name!r} is required (PRD §113). "
                    "Training without it is not permitted."
                )
        if not self.dataset_ids:
            raise ValidationError("dataset contract must name at least one dataset id")
        if self.n_patients <= 0:
            raise ValidationError("dataset contract must state a positive patient count")
        if "patient" not in self.split_strategy:
            raise ValidationError(
                f"split strategy {self.split_strategy!r} is not patient-level. "
                "Slice- or study-level splits leak the same patient across splits "
                "(PRD §63, §115)."
            )
        total_split = sum(int(v) for v in self.split_patients.values())
        if total_split > self.n_patients:
            raise ValidationError(
                f"split patient counts sum to {total_split} but the contract "
                f"declares {self.n_patients} patients"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "dataset_ids": list(self.dataset_ids),
            "external_test_dataset_ids": list(self.external_test_dataset_ids),
            "split_patients": dict(self.split_patients),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetContract:
        data = dict(payload)
        data["dataset_ids"] = tuple(data.get("dataset_ids", ()))
        data["external_test_dataset_ids"] = tuple(
            data.get("external_test_dataset_ids", ())
        )
        data["split_patients"] = dict(data.get("split_patients", {}))
        return cls(**data)


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation results, separated by which cohort produced them.

    Internal validation and external test are kept apart because a single blended
    "accuracy" number hides the generalisation gap that matters most (PRD §64).
    """

    internal_validation: Mapping[str, Any] = field(default_factory=dict)
    internal_test: Mapping[str, Any] = field(default_factory=dict)
    external_test: Mapping[str, Any] = field(default_factory=dict)

    #: Agreement between measurements derived from this model's masks and from
    #: reference masks (PRD §114). Absent means it was never assessed.
    clinical_measurement_impact: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "internal_validation": dict(self.internal_validation),
            "internal_test": dict(self.internal_test),
            "external_test": dict(self.external_test),
            "clinical_measurement_impact": dict(self.clinical_measurement_impact),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelMetrics:
        return cls(
            internal_validation=dict(payload.get("internal_validation", {})),
            internal_test=dict(payload.get("internal_test", {})),
            external_test=dict(payload.get("external_test", {})),
            clinical_measurement_impact=dict(
                payload.get("clinical_measurement_impact", {})
            ),
        )


@dataclass
class ModelCard:
    """Everything needed to use a model correctly and audit its output."""

    model_id: str
    version: str
    task: ModelTask
    architecture: Architecture
    dataset_contract: DatasetContract

    #: Structure -> integer label. Required for segmentation models.
    label_map: Mapping[Structure, int] | None = None

    #: Serialised PreprocessingSpec the model was trained with.
    preprocessing: Mapping[str, Any] = field(default_factory=dict)

    modality: Modality = Modality.MR
    input_channels: int = 1
    patch_size: tuple[int, int, int] | None = None

    metrics: ModelMetrics = field(default_factory=ModelMetrics)

    #: Free-text but required: what the model is for, and what it is not for.
    intended_use: str = ""
    limitations: str = ""

    #: Situations in which the model's output must not be used.
    contraindications: tuple[str, ...] = ()

    framework: str = "pytorch"
    framework_version: str = ""
    platform_version: str = PLATFORM_VERSION
    created_at: datetime = field(default_factory=now_utc)
    created_by: str = ""
    training_run_id: str = ""
    git_commit: str = ""

    checkpoint: ArtifactRef | None = None
    notes: str = ""

    #: Whether this model may be used on real patient cases. Defaults to False:
    #: a newly trained checkpoint is a research artefact until someone decides
    #: otherwise, and that decision should be explicit.
    approved_for_clinical_use: bool = False

    def __post_init__(self) -> None:
        if not self.model_id or not self.version:
            raise ValidationError("model cards require both model_id and version")
        if self.task is ModelTask.SEGMENTATION and not self.label_map:
            raise ValidationError(
                "segmentation model cards must declare a label_map; without it the "
                "output channels cannot be mapped to anatomy"
            )
        if not self.intended_use.strip():
            raise ValidationError(
                "model card must state intended_use. A model whose intended use is "
                "undocumented cannot be reviewed."
            )
        if self.label_map is not None:
            values = list(self.label_map.values())
            if len(set(values)) != len(values):
                raise ValidationError(f"label map is not injective: {dict(self.label_map)}")

    # --- identity ---
    @property
    def key(self) -> str:
        return f"{self.model_id}:{self.version}"

    @property
    def structures(self) -> list[Structure]:
        if not self.label_map:
            return []
        return [s for s in self.label_map if s is not Structure.BACKGROUND]

    @property
    def n_classes(self) -> int:
        return len(self.label_map) if self.label_map else 0

    # --- compatibility gate ---
    def check_compatibility(
        self,
        *,
        expected_label_map: Mapping[Structure, int] | None = None,
        expected_modality: Modality | None = None,
        require_clinical_approval: bool = False,
    ) -> None:
        """Verify this model may be applied in the caller's context.

        Raises:
            ValidationError: on any mismatch. Inference must not proceed.
        """
        if require_clinical_approval and not self.approved_for_clinical_use:
            raise ValidationError(
                f"model {self.key} is not approved for clinical use. It may be run "
                "in research mode only."
            )
        if expected_modality is not None and self.modality is not expected_modality:
            raise ValidationError(
                f"model {self.key} was trained on {self.modality.value} but is being "
                f"applied to {expected_modality.value}. Cross-modality use requires "
                "its own validation."
            )
        if expected_label_map is not None:
            if self.label_map is None:
                raise ValidationError(f"model {self.key} declares no label map")
            mine = {s: int(v) for s, v in self.label_map.items()}
            theirs = {s: int(v) for s, v in expected_label_map.items()}
            if mine != theirs:
                only_mine = sorted(
                    f"{s.value}={v}" for s, v in mine.items() if theirs.get(s) != v
                )
                only_theirs = sorted(
                    f"{s.value}={v}" for s, v in theirs.items() if mine.get(s) != v
                )
                raise ValidationError(
                    f"label map mismatch for model {self.key}.\n"
                    f"  model expects:    {only_mine}\n"
                    f"  pipeline expects: {only_theirs}\n"
                    "Using a checkpoint with a different label ordering produces "
                    "silently mislabelled anatomy."
                )

    # --- serialisation ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "task": self.task.value,
            "architecture": self.architecture.value,
            "dataset_contract": self.dataset_contract.to_dict(),
            "label_map": (
                {s.value: int(v) for s, v in self.label_map.items()}
                if self.label_map
                else None
            ),
            "preprocessing": dict(self.preprocessing),
            "modality": self.modality.value,
            "input_channels": self.input_channels,
            "patch_size": list(self.patch_size) if self.patch_size else None,
            "metrics": self.metrics.to_dict(),
            "intended_use": self.intended_use,
            "limitations": self.limitations,
            "contraindications": list(self.contraindications),
            "framework": self.framework,
            "framework_version": self.framework_version,
            "platform_version": self.platform_version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "training_run_id": self.training_run_id,
            "git_commit": self.git_commit,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "notes": self.notes,
            "approved_for_clinical_use": self.approved_for_clinical_use,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCard:
        label_map = None
        if payload.get("label_map"):
            label_map = {Structure(k): int(v) for k, v in payload["label_map"].items()}
        created = payload.get("created_at")
        return cls(
            model_id=str(payload["model_id"]),
            version=str(payload["version"]),
            task=ModelTask(payload["task"]),
            architecture=Architecture(payload["architecture"]),
            dataset_contract=DatasetContract.from_dict(payload["dataset_contract"]),
            label_map=label_map,
            preprocessing=dict(payload.get("preprocessing", {})),
            modality=Modality(payload.get("modality", "MR")),
            input_channels=int(payload.get("input_channels", 1)),
            patch_size=(
                tuple(payload["patch_size"]) if payload.get("patch_size") else None
            ),
            metrics=ModelMetrics.from_dict(payload.get("metrics", {})),
            intended_use=str(payload.get("intended_use", "")),
            limitations=str(payload.get("limitations", "")),
            contraindications=tuple(payload.get("contraindications", ())),
            framework=str(payload.get("framework", "pytorch")),
            framework_version=str(payload.get("framework_version", "")),
            platform_version=str(payload.get("platform_version", PLATFORM_VERSION)),
            created_at=(
                datetime.fromisoformat(created) if created else now_utc()
            ),
            created_by=str(payload.get("created_by", "")),
            training_run_id=str(payload.get("training_run_id", "")),
            git_commit=str(payload.get("git_commit", "")),
            checkpoint=(
                ArtifactRef.from_dict(payload["checkpoint"])
                if payload.get("checkpoint")
                else None
            ),
            notes=str(payload.get("notes", "")),
            approved_for_clinical_use=bool(payload.get("approved_for_clinical_use", False)),
        )

    def render_summary(self) -> str:
        """Human-readable summary for the UI's model-info panel."""
        lines = [
            f"{self.model_id} v{self.version}",
            f"  task         : {self.task.value} ({self.architecture.value})",
            f"  modality     : {self.modality.value}",
            f"  structures   : {', '.join(s.display_name for s in self.structures) or '-'}",
            f"  patients     : {self.dataset_contract.n_patients} "
            f"({self.dataset_contract.split_strategy})",
            f"  datasets     : {', '.join(self.dataset_contract.dataset_ids)}",
            f"  licence      : {self.dataset_contract.license}",
            f"  clinical use : {'approved' if self.approved_for_clinical_use else 'RESEARCH ONLY'}",
        ]
        if self.metrics.internal_validation:
            lines.append(f"  val metrics  : {self.metrics.internal_validation}")
        if self.metrics.external_test:
            lines.append(f"  external test: {self.metrics.external_test}")
        if self.contraindications:
            lines.append("  do not use when:")
            lines.extend(f"    - {c}" for c in self.contraindications)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Filesystem-backed registry: ``<root>/<model_id>/<version>/model_card.json``.

    Deliberately a directory of JSON plus checkpoints rather than a database.
    A model registry that can be inspected with ``ls`` and copied to an
    air-gapped hospital server with ``rsync`` is worth more here than one behind
    a service.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        if root is None:
            from kneeai.config import get_settings

            root = get_settings().model_registry
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- paths ---
    def _version_dir(self, model_id: str, version: str) -> Path:
        for part in (model_id, version):
            if not part or any(c in part for c in "/\\") or part in (".", ".."):
                raise ConfigurationError(f"unsafe registry path component: {part!r}")
        return self.root / model_id / version

    def card_path(self, model_id: str, version: str) -> Path:
        return self._version_dir(model_id, version) / CARD_FILENAME

    # --- write ---
    def register(
        self,
        card: ModelCard,
        checkpoint_path: str | os.PathLike[str] | None = None,
        *,
        overwrite: bool = False,
    ) -> ModelCard:
        """Store a model card and (optionally) copy its checkpoint in."""
        target_dir = self._version_dir(card.model_id, card.version)
        card_file = target_dir / CARD_FILENAME
        if card_file.exists() and not overwrite:
            raise ConfigurationError(
                f"model {card.key} is already registered. Registry entries are "
                "immutable by default; bump the version or pass overwrite=True."
            )
        target_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_path is not None:
            import shutil

            source = Path(checkpoint_path)
            if not source.exists():
                raise ConfigurationError(f"checkpoint not found: {source}")
            destination = target_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            card.checkpoint = ArtifactRef(
                kind="checkpoint",
                path=str(destination),
                content_hash=hash_file(destination),
                bytes_size=destination.stat().st_size,
                note=f"{card.framework} {card.architecture.value}",
            )

        card_file.write_text(json.dumps(card.to_dict(), indent=2, default=str))
        log.info(
            "model registered",
            fields={
                "model": card.key,
                "task": card.task.value,
                "clinical": card.approved_for_clinical_use,
            },
        )
        return card

    def approve_for_clinical_use(
        self, model_id: str, version: str, *, approver: str, justification: str
    ) -> ModelCard:
        """Mark a model as clinically usable. Requires an approver and rationale."""
        if not approver or not justification:
            raise ValidationError(
                "clinical approval requires both an approver and a written "
                "justification; it is an accountable decision"
            )
        card = self.get(model_id, version)
        card.approved_for_clinical_use = True
        card.notes = (
            f"{card.notes}\n[clinical approval] by={approver} at={now_utc().isoformat()} "
            f"reason={justification}"
        ).strip()
        self.register(card, overwrite=True)
        log.audit(
            "model_clinical_approval",
            model=card.key,
            approver=approver,
            justification=justification,
        )
        return card

    # --- read ---
    def get(self, model_id: str, version: str | None = None) -> ModelCard:
        if version is None:
            return self.latest(model_id)
        path = self.card_path(model_id, version)
        if not path.exists():
            available = self.versions(model_id)
            raise ConfigurationError(
                f"model {model_id}:{version} is not registered. "
                f"Available versions: {available or ['<none>']}"
            )
        return ModelCard.from_dict(json.loads(path.read_text()))

    def resolve(self, reference: str) -> ModelCard:
        """Resolve a ``model_id`` or ``model_id:version`` reference."""
        if ":" in reference:
            model_id, version = reference.split(":", 1)
            return self.get(model_id, version)
        return self.latest(reference)

    def versions(self, model_id: str) -> list[str]:
        model_dir = self.root / model_id
        if not model_dir.is_dir():
            return []
        return sorted(
            p.name for p in model_dir.iterdir() if (p / CARD_FILENAME).exists()
        )

    def latest(self, model_id: str) -> ModelCard:
        """Most recently created registered version of a model."""
        versions = self.versions(model_id)
        if not versions:
            raise ConfigurationError(
                f"no registered versions for model {model_id!r}. Registered models: "
                f"{self.model_ids() or ['<none>']}"
            )
        cards = [self.get(model_id, v) for v in versions]
        return max(cards, key=lambda c: c.created_at)

    def model_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and any(child.is_dir() for child in p.iterdir())
        )

    def list_cards(
        self,
        *,
        task: ModelTask | None = None,
        clinical_only: bool = False,
    ) -> list[ModelCard]:
        cards: list[ModelCard] = []
        for model_id in self.model_ids():
            for version in self.versions(model_id):
                card = self.get(model_id, version)
                if task is not None and card.task is not task:
                    continue
                if clinical_only and not card.approved_for_clinical_use:
                    continue
                cards.append(card)
        return sorted(cards, key=lambda c: (c.model_id, c.created_at))

    def checkpoint_for(self, model_id: str, version: str | None = None) -> Path:
        """Path to a model's checkpoint, verifying its content hash first.

        A checkpoint whose bytes no longer match the hash recorded at
        registration has been modified outside the registry. That invalidates
        every metric on the card, so it is refused rather than loaded.
        """
        card = self.get(model_id, version)
        if card.checkpoint is None or not card.checkpoint.path:
            raise ConfigurationError(f"model {card.key} has no checkpoint attached")
        path = Path(card.checkpoint.path)
        if not path.exists():
            raise ConfigurationError(
                f"checkpoint for {card.key} is missing from disk: {path}"
            )
        if card.checkpoint.content_hash:
            actual = hash_file(path)
            if actual != card.checkpoint.content_hash:
                raise ConfigurationError(
                    f"checkpoint for {card.key} does not match its registered hash.\n"
                    f"  registered: {card.checkpoint.content_hash}\n"
                    f"  on disk   : {actual}\n"
                    "The file has been modified since registration; its recorded "
                    "metrics no longer apply."
                )
        return path

    def summary(self) -> str:
        cards = self.list_cards()
        if not cards:
            return f"No models registered under {self.root}"
        return "\n\n".join(c.render_summary() for c in cards)


_REGISTRY: ModelRegistry | None = None


def get_registry(root: str | os.PathLike[str] | None = None) -> ModelRegistry:
    global _REGISTRY
    if root is not None:
        return ModelRegistry(root)
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY
