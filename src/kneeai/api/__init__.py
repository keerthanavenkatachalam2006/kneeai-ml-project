"""Application layer: contract access, the inference adapter, and the HTTP API.

The contract is defined by :mod:`kneeai.ml.contract` (canonical, producer-side).
:mod:`kneeai.api.contract` adds consumer-side access on top of it.

The FastAPI application (:mod:`kneeai.api.main`) is imported lazily so the
contract and adapter — which the CLI, worker and tests all need — stay importable
without the ``api`` extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kneeai.api.contract import (
    CONTRACT_VERSION,
    MESH_DIRNAME,
    RESULT_FILENAME,
    SEGMENTATION_FILENAME,
    ArtifactSource,
    CaseResultView,
    ContractReport,
    build_result_payload,
    canonical_label_volume,
    read_result_json,
    validate_result_json,
    validate_result_payload,
    write_result_json,
)

if TYPE_CHECKING:  # pragma: no cover
    from kneeai.api.adapter import AdapterConfig, InferenceBackend, MLInferenceAdapter
    from kneeai.api.main import app
    from kneeai.api.pipeline import CaseArtifacts, PipelineOptions, run_case_pipeline

__all__ = [
    # contract (canonical definitions live in kneeai.ml.contract)
    "CONTRACT_VERSION",
    "ArtifactSource",
    "ContractReport",
    "CaseResultView",
    "RESULT_FILENAME",
    "SEGMENTATION_FILENAME",
    "MESH_DIRNAME",
    "build_result_payload",
    "validate_result_payload",
    "validate_result_json",
    "read_result_json",
    "write_result_json",
    "canonical_label_volume",
    # lazy
    "MLInferenceAdapter",
    "AdapterConfig",
    "InferenceBackend",
    "run_case_pipeline",
    "PipelineOptions",
    "CaseArtifacts",
    "app",
]

_LAZY: dict[str, str] = {
    "MLInferenceAdapter": "kneeai.api.adapter",
    "AdapterConfig": "kneeai.api.adapter",
    "InferenceBackend": "kneeai.api.adapter",
    "run_case_pipeline": "kneeai.api.pipeline",
    "PipelineOptions": "kneeai.api.pipeline",
    "CaseArtifacts": "kneeai.api.pipeline",
    "app": "kneeai.api.main",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value
