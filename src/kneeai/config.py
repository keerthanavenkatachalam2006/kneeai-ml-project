"""Runtime configuration.

Loaded from environment variables (``KNEEAI_*``) with sane local-development
defaults. Settings that have safety or privacy consequences are validated here
rather than at the point of use, so a misconfigured deployment fails at startup
instead of mid-case.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kneeai.core.errors import ConfigurationError

__all__ = ["Environment", "Settings", "get_settings", "reset_settings"]


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION


_INSECURE_JWT_SECRETS = {
    "",
    "change-me-before-any-shared-deployment",
    "secret",
    "changeme",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KNEEAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- environment -------------------------------------------------------
    env: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_json: bool = False

    # --- storage -----------------------------------------------------------
    data_root: Path = Path("E:/KneeAI/data")
    artifact_root: Path = Path("E:/KneeAI/artifacts")
    model_registry: Path = Path("E:/KneeAI/artifacts/models")

    # --- database / queue --------------------------------------------------
    database_url: str = "sqlite:///./artifacts/kneeai.db"
    redis_url: str = "redis://localhost:6379/0"

    # --- api ---------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    jwt_secret: str = "change-me-before-any-shared-deployment"
    jwt_ttl_minutes: int = 60
    cors_origins: str = "http://localhost:3000"

    # --- DICOM store -------------------------------------------------------
    orthanc_url: str = "http://localhost:8042"
    orthanc_user: str = ""
    orthanc_password: str = ""

    # --- experiment tracking ----------------------------------------------
    mlflow_tracking_uri: str = "./mlruns"

    # --- implant catalog ---------------------------------------------------
    implant_catalog: str = "builtin:reference-v0"

    #: The bundled catalog is synthetic reference geometry for development. It
    #: must never be used to plan a real operation, so production refuses it.
    allow_nonclinical_implants: bool = True

    # --- LLM ---------------------------------------------------------------
    llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_api_key: str = Field(default="", repr=False)

    # --- safety ------------------------------------------------------------
    #: Maximum acceptable patient-registration RMS before navigation is allowed.
    max_registration_rms_mm: float = 1.0
    #: Maximum age of a tracking sample before it is considered stale.
    max_tracking_age_s: float = 0.1
    #: Hard ceiling on any planned bone resection depth. A plan exceeding this is
    #: rejected outright rather than warned about.
    max_resection_depth_mm: float = 25.0

    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        up = v.upper()
        if up not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return up

    @field_validator("data_root", "artifact_root", "model_registry")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(os.path.expanduser(str(v))).resolve()

    @model_validator(mode="after")
    def _validate_production_posture(self) -> Settings:
        if not self.env.is_production:
            return self

        problems: list[str] = []
        if self.jwt_secret in _INSECURE_JWT_SECRETS or len(self.jwt_secret) < 32:
            problems.append(
                "KNEEAI_JWT_SECRET must be set to a unique value of at least 32 characters"
            )
        if self.database_url.startswith("sqlite"):
            problems.append("SQLite is not a supported production database; use PostgreSQL")
        if self.allow_nonclinical_implants:
            problems.append(
                "KNEEAI_ALLOW_NONCLINICAL_IMPLANTS must be false in production: the "
                "bundled catalog is synthetic reference geometry, not authorized "
                "manufacturer geometry (PRD §21)"
            )
        if "*" in self.cors_origins:
            problems.append("wildcard CORS origins are not permitted in production")
        if self.llm_enabled and not self.llm_model:
            problems.append("llm_enabled is true but no llm_model is configured")
        if problems:
            raise ConfigurationError(
                "unsafe production configuration:\n  - " + "\n  - ".join(problems)
            )
        return self

    # ------------------------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def cases_root(self) -> Path:
        return self.artifact_root / "cases"

    @property
    def registry_root(self) -> Path:
        return self.data_root / "registry"

    @property
    def splits_root(self) -> Path:
        return self.data_root / "splits"

    @property
    def manifests_root(self) -> Path:
        return self.data_root / "manifests"

    def case_dir(self, case_id: str) -> Path:
        """Per-case artefact directory.

        The identifier is checked because it reaches the filesystem: a case id
        containing path separators would otherwise let a crafted upload write
        outside the artefact root.
        """
        safe = case_id.strip()
        if not safe or any(c in safe for c in "/\\") or safe in (".", ".."):
            raise ConfigurationError(f"unsafe case id for filesystem use: {case_id!r}")
        return self.cases_root / safe

    def ensure_directories(self) -> None:
        for path in (
            self.data_root,
            self.artifact_root,
            self.model_registry,
            self.cases_root,
            self.registry_root,
            self.splits_root,
            self.manifests_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Clear the settings cache. Used by tests that patch the environment."""
    get_settings.cache_clear()
