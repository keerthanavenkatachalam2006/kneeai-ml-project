"""Single source of truth for platform version identifiers.

Every artefact the platform emits (mask, mesh, measurement set, plan, report)
carries the platform version plus the version of the component that produced it.
PRD §118 Rule 6 requires every recommendation to be reconstructable from
input -> model -> version -> measurement -> logic -> recommendation.
"""

from __future__ import annotations

__all__ = [
    "PLATFORM_VERSION",
    "SCHEMA_VERSION",
    "COMPONENT_VERSIONS",
    "component_version",
]

#: Version of the platform as a whole.
PLATFORM_VERSION = "0.1.0"

#: Version of the persisted artefact/DB schema. Bump on any breaking change to
#: stored plan / measurement / landmark payloads.
SCHEMA_VERSION = "1"

#: Versions of individually replaceable engines. These are *not* ML model
#: versions (those live in the model registry) — these version the
#: deterministic logic, so a plan can be replayed exactly.
COMPONENT_VERSIONS: dict[str, str] = {
    "imaging.preprocess": "0.1.0",
    "recon.surface": "0.1.0",
    "landmarks.geometric": "0.1.0",
    "measure.engine": "0.1.0",
    "implants.matcher": "0.1.0",
    "planning.engine": "0.1.0",
    "planning.optimizer": "0.1.0",
    "simulation.geometric": "0.1.0",
    "simulation.kinematic": "0.1.0",
    "robotics.safety": "0.1.0",
    "reporting.engine": "0.1.0",
}


def component_version(component: str) -> str:
    """Return the version of a deterministic engine.

    Raises:
        KeyError: if the component is not registered. Unregistered components
            must not silently produce unversioned clinical output.
    """
    try:
        return COMPONENT_VERSIONS[component]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown component {component!r}. Register it in "
            "kneeai.version.COMPONENT_VERSIONS so its output stays traceable."
        ) from exc
