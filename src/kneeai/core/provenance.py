"""Provenance and traceability.

PRD §118 (Rule 6): for every output, ``input -> model -> version -> measurement
-> logic -> recommendation`` must be reconstructable.

Every artefact-producing function in the platform attaches a :class:`Provenance`
record. A clinical output with no provenance is treated as invalid — the report
generator refuses to render it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from kneeai.core.errors import ValidationError
from kneeai.version import PLATFORM_VERSION, SCHEMA_VERSION

__all__ = [
    "Provenance",
    "ArtifactRef",
    "Origin",
    "now_utc",
    "new_id",
    "hash_bytes",
    "hash_file",
    "hash_payload",
    "hash_array",
]

_CHUNK = 1 << 20


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str = "") -> str:
    token = uuid.uuid4().hex[:16]
    return f"{prefix}_{token}" if prefix else token


def hash_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def hash_payload(payload: Any) -> str:
    """Stable hash of a JSON-serialisable payload (sorted keys, no whitespace)."""
    try:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ValidationError(f"payload is not hashable as JSON: {exc}") from exc
    return hash_bytes(blob.encode("utf-8"))


def hash_array(array: Any) -> str:
    """Hash a numpy array including its shape and dtype.

    Shape and dtype are part of the digest so an identical byte buffer
    reinterpreted at a different shape does not collide.
    """
    import numpy as np

    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


class Origin(str, Enum):
    """Who or what produced a value. Reports must separate these (PRD §32)."""

    AI = "ai"
    DETERMINISTIC = "deterministic"
    SURGEON = "surgeon"
    SURGEON_EDITED_AI = "surgeon_edited_ai"
    IMPORTED = "imported"
    DEFAULT = "default"

    @property
    def is_human_confirmed(self) -> bool:
        return self in (Origin.SURGEON, Origin.SURGEON_EDITED_AI)


@dataclass(frozen=True)
class ArtifactRef:
    """A reference to a stored artefact, identified by content hash.

    Storing the hash rather than only the path means a plan can prove it was
    computed from a specific mask, even if the file is later moved or
    regenerated.
    """

    kind: str
    path: str | None = None
    content_hash: str | None = None
    bytes_size: int | None = None
    note: str = ""

    @classmethod
    def from_path(cls, kind: str, path: str | os.PathLike[str], note: str = "") -> ArtifactRef:
        p = Path(path)
        if not p.exists():
            raise ValidationError(f"artifact does not exist: {p}")
        return cls(
            kind=kind,
            path=str(p),
            content_hash=hash_file(p),
            bytes_size=p.stat().st_size,
            note=note,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "content_hash": self.content_hash,
            "bytes_size": self.bytes_size,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactRef:
        return cls(
            kind=str(payload["kind"]),
            path=payload.get("path"),
            content_hash=payload.get("content_hash"),
            bytes_size=payload.get("bytes_size"),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class Provenance:
    """Traceability record attached to every clinical artefact.

    Attributes:
        component: dotted engine name, e.g. ``"planning.engine"``.
        component_version: version of that deterministic engine.
        origin: AI, deterministic, surgeon, or surgeon-edited.
        model_id: model registry id when an ML model was involved.
        model_version: model registry version.
        inputs: content-addressed inputs this artefact was derived from.
        parameters: the exact parameters used, enough to replay the computation.
        created_at: UTC timestamp.
        environment: platform/python/host fingerprint for reproducibility.
        notes: free text for human context. Never parsed.
    """

    component: str
    component_version: str
    origin: Origin = Origin.DETERMINISTIC
    model_id: str | None = None
    model_version: str | None = None
    inputs: tuple[ArtifactRef, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)
    platform_version: str = PLATFORM_VERSION
    schema_version: str = SCHEMA_VERSION
    environment: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""
    provenance_id: str = field(default_factory=lambda: new_id("prov"))

    def __post_init__(self) -> None:
        if not self.component:
            raise ValidationError("provenance requires a component name")
        if not self.component_version:
            raise ValidationError("provenance requires a component version")
        if self.origin is Origin.AI and not self.model_id:
            raise ValidationError(
                "AI-origin provenance must name the model that produced it; "
                "an unattributed AI output cannot be audited"
            )
        if not self.environment:
            object.__setattr__(self, "environment", default_environment())

    @classmethod
    def for_component(
        cls,
        component: str,
        *,
        origin: Origin = Origin.DETERMINISTIC,
        inputs: Iterable[ArtifactRef] = (),
        parameters: Mapping[str, Any] | None = None,
        model_id: str | None = None,
        model_version: str | None = None,
        notes: str = "",
    ) -> Provenance:
        from kneeai.version import component_version

        return cls(
            component=component,
            component_version=component_version(component),
            origin=origin,
            inputs=tuple(inputs),
            parameters=dict(parameters or {}),
            model_id=model_id,
            model_version=model_version,
            notes=notes,
        )

    def derive(
        self,
        component: str,
        *,
        origin: Origin | None = None,
        parameters: Mapping[str, Any] | None = None,
        extra_inputs: Iterable[ArtifactRef] = (),
    ) -> Provenance:
        """Create a child provenance that keeps this record's inputs in the chain."""
        from kneeai.version import component_version

        return Provenance(
            component=component,
            component_version=component_version(component),
            origin=origin or self.origin,
            model_id=self.model_id,
            model_version=self.model_version,
            inputs=tuple(self.inputs) + tuple(extra_inputs),
            parameters=dict(parameters or {}),
            notes=self.notes,
        )

    def signature(self) -> str:
        """Deterministic digest of everything that affects the output.

        Two artefacts with the same signature were produced by the same logic
        from the same inputs with the same parameters, so a cached result can be
        reused safely.
        """
        return hash_payload(
            {
                "component": self.component,
                "component_version": self.component_version,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "inputs": sorted(i.content_hash or i.path or "" for i in self.inputs),
                "parameters": self.parameters,
                "schema_version": self.schema_version,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "provenance_id": self.provenance_id,
            "component": self.component,
            "component_version": self.component_version,
            "origin": self.origin.value,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "inputs": [i.to_dict() for i in self.inputs],
            "parameters": dict(self.parameters),
            "created_at": self.created_at.isoformat(),
            "platform_version": self.platform_version,
            "schema_version": self.schema_version,
            "environment": dict(self.environment),
            "notes": self.notes,
            "signature": self.signature(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Provenance:
        created = payload.get("created_at")
        return cls(
            component=str(payload["component"]),
            component_version=str(payload["component_version"]),
            origin=Origin(payload.get("origin", "deterministic")),
            model_id=payload.get("model_id"),
            model_version=payload.get("model_version"),
            inputs=tuple(ArtifactRef.from_dict(i) for i in payload.get("inputs", [])),
            parameters=dict(payload.get("parameters", {})),
            created_at=datetime.fromisoformat(created) if created else now_utc(),
            platform_version=str(payload.get("platform_version", PLATFORM_VERSION)),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            environment=dict(payload.get("environment", {})),
            notes=str(payload.get("notes", "")),
            provenance_id=str(payload.get("provenance_id", new_id("prov"))),
        )


_ENV_CACHE: dict[str, str] | None = None


def default_environment() -> dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "platform_version": PLATFORM_VERSION,
        }
    return dict(_ENV_CACHE)
