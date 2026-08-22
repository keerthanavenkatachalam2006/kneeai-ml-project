"""Machine-learning layer.

Split by dependency, deliberately:

**Torch-free** (importable anywhere, including CI):
    - :mod:`kneeai.ml.metrics` — Dice, IoU, Hausdorff-95, ASSD, clinical impact
    - :mod:`kneeai.ml.postprocess` — component cleanup, hole filling, overlap resolution
    - :mod:`kneeai.ml.registry` — model cards, dataset contracts, model registry
    - :mod:`kneeai.ml.datasets` — manifests, split-integrity checks
    - :mod:`kneeai.ml.nnunet` — nnU-Net export and patient-level splits

**Requires the ``ml`` extra** (torch + MONAI, installed on Colab):
    - :mod:`kneeai.ml.models` — architecture/loss/optimiser builders
    - :mod:`kneeai.ml.train` — training loop
    - :mod:`kneeai.ml.infer` — sliding-window inference

The modular decomposition is intentional (PRD §34): one model per task, so a
segmentation failure is localised and individually replaceable rather than
requiring the whole system to be debugged as a unit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kneeai.ml.metrics import (
    SegmentationEvaluation,
    StructureMetrics,
    aggregate_evaluations,
    assess_clinical_impact,
    dice,
    evaluate_binary,
    evaluate_masks,
    iou,
    surface_distances,
)
from kneeai.ml.postprocess import (
    PostProcessConfig,
    PostProcessReport,
    largest_component,
    postprocess_mask,
)
from kneeai.ml.registry import (
    Architecture,
    DatasetContract,
    ModelCard,
    ModelMetrics,
    ModelRegistry,
    ModelTask,
    get_registry,
)

if TYPE_CHECKING:  # pragma: no cover
    from kneeai.ml.infer import InferenceConfig, SegmentationPredictor, SegmentationResult
    from kneeai.ml.models import ModelConfig, build_model, describe_device
    from kneeai.ml.train import SegmentationTrainer, TrainConfig, TrainingResult

__all__ = [
    # metrics
    "dice",
    "iou",
    "surface_distances",
    "evaluate_binary",
    "evaluate_masks",
    "aggregate_evaluations",
    "assess_clinical_impact",
    "StructureMetrics",
    "SegmentationEvaluation",
    # postprocess
    "PostProcessConfig",
    "PostProcessReport",
    "postprocess_mask",
    "largest_component",
    # registry
    "ModelTask",
    "Architecture",
    "DatasetContract",
    "ModelMetrics",
    "ModelCard",
    "ModelRegistry",
    "get_registry",
    # torch-dependent (lazy)
    "ModelConfig",
    "build_model",
    "describe_device",
    "TrainConfig",
    "TrainingResult",
    "SegmentationTrainer",
    "InferenceConfig",
    "SegmentationResult",
    "SegmentationPredictor",
]

_LAZY: dict[str, str] = {
    "ModelConfig": "kneeai.ml.models",
    "build_model": "kneeai.ml.models",
    "describe_device": "kneeai.ml.models",
    "TrainConfig": "kneeai.ml.train",
    "TrainingResult": "kneeai.ml.train",
    "SegmentationTrainer": "kneeai.ml.train",
    "InferenceConfig": "kneeai.ml.infer",
    "SegmentationResult": "kneeai.ml.infer",
    "SegmentationPredictor": "kneeai.ml.infer",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
