"""Structured logging with patient-data redaction.

Two requirements shape this module:

1. PRD §81 requires audit logs and controlled handling of patient data. Logs are
   a classic leak path, so identifiers are redacted by default.
2. Clinical debugging needs the case id present in every line, which is why the
   case id is bound into a context variable rather than passed to each call.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping

__all__ = ["get_logger", "configure_logging", "case_context", "redact"]

_case_id: ContextVar[str | None] = ContextVar("kneeai_case_id", default=None)
_actor: ContextVar[str | None] = ContextVar("kneeai_actor", default=None)

#: Keys whose values are never written to a log, at any level.
_SENSITIVE_KEYS = frozenset(
    {
        "patient_name",
        "patientname",
        "patient_id",
        "patientid",
        "patient_birth_date",
        "patientbirthdate",
        "birth_date",
        "dob",
        "accession_number",
        "accessionnumber",
        "address",
        "phone",
        "email",
        "mrn",
        "aadhaar",
        "password",
        "token",
        "api_key",
        "jwt_secret",
        "authorization",
        "referring_physician_name",
        "institution_address",
    }
)

_REDACTED = "<redacted>"

# DICOM UIDs are pseudonymous but still study-identifying; truncate rather than
# drop them so a support engineer can correlate without holding the full UID.
_UID_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+){4,})\b")


def _shorten_uid(match: re.Match[str]) -> str:
    uid = match.group(1)
    return f"{uid[:12]}…{uid[-6:]}" if len(uid) > 24 else uid


def redact(value: Any) -> Any:
    """Recursively redact sensitive keys and shorten long DICOM UIDs."""
    if isinstance(value, Mapping):
        return {
            k: (_REDACTED if str(k).lower() in _SENSITIVE_KEYS else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v) for v in value)
    if isinstance(value, str):
        return _UID_RE.sub(_shorten_uid, value)
    return value


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.case_id = _case_id.get() or "-"
        record.actor = _actor.get() or "-"
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "case_id": getattr(record, "case_id", "-"),
            "actor": getattr(record, "actor", "-"),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload["fields"] = redact(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s [case=%(case_id)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = getattr(record, "extra_fields", None)
        if extra:
            rendered = " ".join(f"{k}={v}" for k, v in redact(extra).items())
            base = f"{base} | {rendered}"
        return base


class KneeAILogger(logging.LoggerAdapter):
    """Adapter that routes keyword fields through redaction."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        fields = kwargs.pop("fields", None)
        extra = kwargs.setdefault("extra", {})
        if fields:
            extra["extra_fields"] = fields
        return msg, kwargs

    def audit(self, action: str, **fields: Any) -> None:
        """Log an auditable action.

        Audit entries are emitted at INFO on the ``kneeai.audit`` logger so a
        deployment can route them to append-only storage independently of
        application logs.
        """
        logging.getLogger("kneeai.audit").info(
            action, extra={"extra_fields": redact(fields)}
        )


_configured = False


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    global _configured
    root = logging.getLogger("kneeai")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if json_output else _TextFormatter())
    handler.addFilter(_ContextFilter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    _configured = True


def get_logger(name: str) -> KneeAILogger:
    """Return a namespaced logger. Auto-configures on first use."""
    global _configured
    if not _configured:
        try:
            from kneeai.config import get_settings

            settings = get_settings()
            configure_logging(settings.log_level, json_output=settings.log_json)
        except Exception:  # pragma: no cover - config may be unavailable early
            configure_logging("INFO")
    full = name if name.startswith("kneeai") else f"kneeai.{name}"
    return KneeAILogger(logging.getLogger(full), {})


@contextmanager
def case_context(case_id: str, actor: str | None = None) -> Iterator[None]:
    """Bind a case id (and optionally an actor) to all logs in this scope."""
    case_token = _case_id.set(case_id)
    actor_token = _actor.set(actor) if actor is not None else None
    try:
        yield
    finally:
        _case_id.reset(case_token)
        if actor_token is not None:
            _actor.reset(actor_token)
