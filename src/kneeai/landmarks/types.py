"""Landmark value objects (PRD §14).

A landmark carries its position, the frame it is expressed in, a confidence, and
how it was obtained. The last two matter for the workflow: the surgeon dashboard
shows which landmarks the platform is unsure about, and once a surgeon moves a
landmark its origin becomes ``SURGEON`` so the report can separate AI-derived
from surgeon-confirmed geometry (PRD §32).

Axes and planes are deliberately *not* stored. They are derived from landmarks on
demand, so correcting one landmark updates every dependent axis, angle and
resection plane consistently instead of leaving a stale cached axis behind.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import LandmarkName, Side, Structure
from kneeai.core.errors import FrameMismatchError, LandmarkError, ValidationError
from kneeai.core.frames import Frame, FramedPoints, Transform
from kneeai.core.geometry import Axis
from kneeai.core.provenance import Origin, Provenance

__all__ = ["Landmark", "LandmarkSet"]


@dataclass(frozen=True)
class Landmark:
    """One anatomical point."""

    name: LandmarkName
    position: NDArray[np.float64]
    frame: Frame
    confidence: float = 1.0
    origin: Origin = Origin.DETERMINISTIC

    #: How this point was established, e.g. "extreme_point(+lateral, p99.5)".
    #: Recorded so a surgeon can see *why* the platform put a landmark somewhere.
    method: str = ""

    #: Estimated localisation error, when the method supports estimating one.
    uncertainty_mm: float | None = None

    def __post_init__(self) -> None:
        p = np.asarray(self.position, dtype=np.float64).reshape(-1)
        if p.size != 3:
            raise ValidationError(f"landmark {self.name.value} must be a 3D point")
        if not np.all(np.isfinite(p)):
            raise LandmarkError(f"landmark {self.name.value} has non-finite coordinates")
        object.__setattr__(self, "position", p)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(
                f"landmark confidence must be in [0, 1], got {self.confidence}"
            )

    @property
    def bone(self) -> Structure:
        return self.name.bone

    def transform(self, t: Transform) -> Landmark:
        if t.source is not self.frame:
            raise FrameMismatchError(
                f"landmark {self.name.value} is in '{self.frame.value}' but the "
                f"transform maps from '{t.source.value}'"
            )
        return replace(self, position=t.apply(self.position), frame=t.target)

    def moved_to(
        self, position: ArrayLike, *, actor: str, reason: str = ""
    ) -> Landmark:
        """Surgeon override. Marks the landmark as surgeon-edited (PRD §2.3)."""
        if not actor:
            raise ValidationError("a landmark override must record who made it")
        return replace(
            self,
            position=np.asarray(position, dtype=np.float64).reshape(3),
            origin=Origin.SURGEON,
            confidence=1.0,
            method=f"surgeon override by {actor}" + (f": {reason}" if reason else ""),
            uncertainty_mm=None,
        )

    def distance_to(self, other: Landmark | ArrayLike) -> float:
        if isinstance(other, Landmark):
            if other.frame is not self.frame:
                raise FrameMismatchError(
                    f"cannot measure between landmarks in '{self.frame.value}' and "
                    f"'{other.frame.value}'"
                )
            target = other.position
        else:
            target = np.asarray(other, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(self.position - target))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "position": [round(float(v), 4) for v in self.position],
            "frame": self.frame.value,
            "confidence": round(float(self.confidence), 4),
            "origin": self.origin.value,
            "method": self.method,
            "uncertainty_mm": self.uncertainty_mm,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Landmark:
        return cls(
            name=LandmarkName(payload["name"]),
            position=np.asarray(payload["position"], dtype=np.float64),
            frame=Frame(payload["frame"]),
            confidence=float(payload.get("confidence", 1.0)),
            origin=Origin(payload.get("origin", "deterministic")),
            method=str(payload.get("method", "")),
            uncertainty_mm=payload.get("uncertainty_mm"),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        pos = np.round(self.position, 1).tolist()
        return f"Landmark({self.name.value}, {pos}, conf={self.confidence:.2f})"


@dataclass
class LandmarkSet:
    """A case's landmarks, all in one frame."""

    landmarks: dict[LandmarkName, Landmark] = field(default_factory=dict)
    side: Side = Side.UNKNOWN
    frame: Frame = Frame.IMAGE
    provenance: Provenance | None = None

    # --- construction ---
    @classmethod
    def from_iterable(
        cls,
        landmarks: Iterable[Landmark],
        *,
        side: Side = Side.UNKNOWN,
        provenance: Provenance | None = None,
    ) -> LandmarkSet:
        items = list(landmarks)
        frames = {lm.frame for lm in items}
        if len(frames) > 1:
            raise FrameMismatchError(
                f"landmarks span multiple frames: {sorted(f.value for f in frames)}"
            )
        return cls(
            landmarks={lm.name: lm for lm in items},
            side=side,
            frame=frames.pop() if frames else Frame.IMAGE,
            provenance=provenance,
        )

    def add(self, landmark: Landmark) -> LandmarkSet:
        if landmark.frame is not self.frame:
            raise FrameMismatchError(
                f"cannot add a landmark in '{landmark.frame.value}' to a set in "
                f"'{self.frame.value}'"
            )
        self.landmarks[landmark.name] = landmark
        return self

    # --- access ---
    def __contains__(self, name: object) -> bool:
        return name in self.landmarks

    def __len__(self) -> int:
        return len(self.landmarks)

    def __iter__(self) -> Iterator[Landmark]:
        return iter(self.landmarks.values())

    def __getitem__(self, name: LandmarkName) -> Landmark:
        return self.require(name)

    def get(self, name: LandmarkName) -> Landmark | None:
        return self.landmarks.get(name)

    def require(self, *names: LandmarkName) -> Landmark | tuple[Landmark, ...]:
        """Fetch landmarks, raising a specific error naming what is missing."""
        missing = [n.value for n in names if n not in self.landmarks]
        if missing:
            available = sorted(k.value for k in self.landmarks)
            raise LandmarkError(
                f"required landmark(s) not available: {', '.join(missing)}.\n"
                f"Present: {available or ['<none>']}"
            )
        found = tuple(self.landmarks[n] for n in names)
        return found[0] if len(found) == 1 else found

    def position(self, name: LandmarkName) -> NDArray[np.float64]:
        landmark = self.require(name)
        assert isinstance(landmark, Landmark)
        return landmark.position

    def positions(self, *names: LandmarkName) -> NDArray[np.float64]:
        return np.vstack([self.position(n) for n in names])

    def has(self, *names: LandmarkName) -> bool:
        return all(n in self.landmarks for n in names)

    def framed(self, *names: LandmarkName) -> FramedPoints:
        return FramedPoints(self.positions(*names), frame=self.frame)

    # --- derived geometry ---
    def axis_between(
        self, start: LandmarkName, end: LandmarkName, *, label: str = ""
    ) -> Axis:
        """Axis from ``start`` to ``end``. Derived, never cached."""
        return Axis.through(
            self.position(start),
            self.position(end),
            frame=self.frame,
            label=label or f"{start.value}->{end.value}",
        )

    def midpoint(self, a: LandmarkName, b: LandmarkName) -> NDArray[np.float64]:
        return (self.position(a) + self.position(b)) / 2.0

    # --- transformation ---
    def transform(self, t: Transform) -> LandmarkSet:
        return LandmarkSet(
            landmarks={
                name: lm.transform(t) for name, lm in self.landmarks.items()
            },
            side=self.side,
            frame=t.target,
            provenance=self.provenance,
        )

    # --- editing ---
    def override(
        self,
        name: LandmarkName,
        position: ArrayLike,
        *,
        actor: str,
        reason: str = "",
    ) -> LandmarkSet:
        """Apply a surgeon correction, returning a new set."""
        existing = self.require(name)
        assert isinstance(existing, Landmark)
        updated = dict(self.landmarks)
        updated[name] = existing.moved_to(position, actor=actor, reason=reason)
        return LandmarkSet(
            landmarks=updated,
            side=self.side,
            frame=self.frame,
            provenance=self.provenance,
        )

    # --- quality ---
    @property
    def mean_confidence(self) -> float:
        if not self.landmarks:
            return float("nan")
        return float(np.mean([lm.confidence for lm in self.landmarks.values()]))

    def low_confidence(self, threshold: float = 0.6) -> list[Landmark]:
        """Landmarks the surgeon should check first."""
        return sorted(
            (lm for lm in self.landmarks.values() if lm.confidence < threshold),
            key=lambda lm: lm.confidence,
        )

    def surgeon_edited(self) -> list[Landmark]:
        return [lm for lm in self.landmarks.values() if lm.origin is Origin.SURGEON]

    # --- serialisation ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "frame": self.frame.value,
            "mean_confidence": round(self.mean_confidence, 4),
            "n_landmarks": len(self.landmarks),
            "n_surgeon_edited": len(self.surgeon_edited()),
            "landmarks": [
                lm.to_dict()
                for lm in sorted(self.landmarks.values(), key=lambda x: x.name.value)
            ],
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LandmarkSet:
        items = [Landmark.from_dict(item) for item in payload.get("landmarks", [])]
        return cls(
            landmarks={lm.name: lm for lm in items},
            side=Side(payload.get("side", "unknown")),
            frame=Frame(payload.get("frame", "image")),
            provenance=(
                Provenance.from_dict(payload["provenance"])
                if payload.get("provenance")
                else None
            ),
        )

    def summary(self) -> str:
        lines = [
            f"LandmarkSet ({self.side.value} knee, {self.frame.value} frame, "
            f"{len(self.landmarks)} landmarks)"
        ]
        for lm in sorted(self.landmarks.values(), key=lambda x: x.name.value):
            pos = np.round(lm.position, 1).tolist()
            flag = " [surgeon]" if lm.origin is Origin.SURGEON else ""
            lines.append(
                f"  {lm.name.value:<28} {str(pos):<28} conf={lm.confidence:.2f}{flag}"
            )
        return "\n".join(lines)
