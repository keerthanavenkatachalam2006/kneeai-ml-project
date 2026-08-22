"""Frame-aware geometric primitives: planes, axes, fits and projections.

Everything clinical downstream (alignment angles, resection depths, implant
footprints) reduces to operations on these primitives. They carry their
:class:`~kneeai.core.frames.Frame` so a resection plane defined in the planning
frame cannot be silently intersected with a mesh still in image coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.errors import FrameMismatchError, ValidationError
from kneeai.core.frames import Frame, FramedPoints, Transform

__all__ = [
    "Plane",
    "Axis",
    "Sphere",
    "OrientedBox",
    "unit",
    "angle_between_deg",
    "signed_angle_in_plane_deg",
    "project_onto_plane",
    "fit_plane",
    "fit_line",
    "fit_sphere",
    "principal_axes",
    "extreme_point",
    "bounding_extent",
]

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------


def unit(v: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        raise ValidationError("cannot normalize a zero-length vector")
    return a / n


def angle_between_deg(a: ArrayLike, b: ArrayLike) -> float:
    """Unsigned angle in [0, 180]."""
    ua, ub = unit(a), unit(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ua, ub))))))


def signed_angle_in_plane_deg(a: ArrayLike, b: ArrayLike, plane_normal: ArrayLike) -> float:
    """Signed angle from ``a`` to ``b`` measured in the plane with ``plane_normal``.

    Positive is counter-clockwise when looking down the normal. Clinical angles
    (varus/valgus, slope, rotation) are signed, and the sign convention is only
    meaningful once the viewing plane is stated — hence the required normal.
    """
    n = unit(plane_normal)
    pa = project_onto_plane(a, n)
    pb = project_onto_plane(b, n)
    if float(np.linalg.norm(pa)) < 1e-7 or float(np.linalg.norm(pb)) < 1e-7:
        raise ValidationError(
            "a vector is (near) parallel to the plane normal; the projected "
            "angle is undefined"
        )
    pa, pb = unit(pa), unit(pb)
    cos = max(-1.0, min(1.0, float(np.dot(pa, pb))))
    sin = float(np.dot(n, np.cross(pa, pb)))
    return math.degrees(math.atan2(sin, cos))


def project_onto_plane(v: ArrayLike, plane_normal: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    n = unit(plane_normal)
    return a - float(np.dot(a, n)) * n


# ---------------------------------------------------------------------------
# Plane
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plane:
    """An oriented plane: all points ``p`` with ``dot(normal, p - point) == 0``.

    The normal direction is meaningful. For resection planes the convention used
    throughout the planner is: **the normal points away from the resected
    fragment**, i.e. towards the bone that is kept.
    """

    point: NDArray[np.float64]
    normal: NDArray[np.float64]
    frame: Frame
    label: str = ""

    def __post_init__(self) -> None:
        p = np.asarray(self.point, dtype=np.float64).reshape(-1)
        if p.size != 3:
            raise ValidationError(f"plane point must be 3D, got {p.size} components")
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "normal", unit(self.normal))

    # --- constructors ---
    @classmethod
    def from_points(
        cls, a: ArrayLike, b: ArrayLike, c: ArrayLike, *, frame: Frame, label: str = ""
    ) -> Plane:
        pa = np.asarray(a, dtype=np.float64).reshape(3)
        pb = np.asarray(b, dtype=np.float64).reshape(3)
        pc = np.asarray(c, dtype=np.float64).reshape(3)
        n = np.cross(pb - pa, pc - pa)
        if float(np.linalg.norm(n)) < 1e-7:
            raise ValidationError("the three points are collinear; no plane is defined")
        return cls(pa, n, frame=frame, label=label)

    @classmethod
    def from_point_normal(
        cls, point: ArrayLike, normal: ArrayLike, *, frame: Frame, label: str = ""
    ) -> Plane:
        return cls(np.asarray(point, dtype=np.float64), normal, frame=frame, label=label)

    # --- queries ---
    @property
    def offset(self) -> float:
        """``d`` in ``dot(normal, x) = d``."""
        return float(np.dot(self.normal, self.point))

    def signed_distance(self, points: ArrayLike) -> NDArray[np.float64] | float:
        p = np.asarray(points, dtype=np.float64)
        squeeze = p.ndim == 1
        p = np.atleast_2d(p)
        d = p @ self.normal - self.offset
        return float(d[0]) if squeeze else d

    def signed_distance_framed(self, points: FramedPoints) -> NDArray[np.float64]:
        points.require_frame(self.frame, f"Plane[{self.label}].signed_distance")
        return np.atleast_1d(self.signed_distance(points.array))  # type: ignore[arg-type]

    def project(self, points: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(points, dtype=np.float64)
        squeeze = p.ndim == 1
        p = np.atleast_2d(p)
        d = (p @ self.normal - self.offset)[:, None]
        out = p - d * self.normal
        return out[0] if squeeze else out

    def flipped(self) -> Plane:
        return Plane(self.point, -self.normal, frame=self.frame, label=self.label)

    def offset_by(self, distance_mm: float) -> Plane:
        """Translate the plane ``distance_mm`` along its normal.

        Positive moves the plane in the direction of the normal, i.e. *into* the
        retained bone — which for a resection plane means resecting more.
        """
        return Plane(
            self.point + float(distance_mm) * self.normal,
            self.normal,
            frame=self.frame,
            label=self.label,
        )

    def basis(self, hint: ArrayLike | None = None) -> tuple[NDArray[np.float64], ...]:
        """Orthonormal in-plane basis ``(u, v)`` plus the normal.

        ``hint`` seeds ``u``; supply an anatomical direction (e.g. the AP axis)
        so 2D footprint coordinates are anatomically interpretable rather than
        arbitrary.
        """
        if hint is None:
            seed = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(seed, self.normal))) > 0.9:
                seed = np.array([0.0, 1.0, 0.0])
        else:
            seed = np.asarray(hint, dtype=np.float64).reshape(3)
        u = project_onto_plane(seed, self.normal)
        if float(np.linalg.norm(u)) < 1e-7:
            raise ValidationError("basis hint is parallel to the plane normal")
        u = unit(u)
        v = np.cross(self.normal, u)
        return u, v, self.normal

    def to_2d(self, points: ArrayLike, hint: ArrayLike | None = None) -> NDArray[np.float64]:
        """Project points into in-plane ``(u, v)`` coordinates in millimetres."""
        u, v, _ = self.basis(hint)
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        rel = p - self.point
        return np.column_stack([rel @ u, rel @ v])

    def from_2d(self, uv: ArrayLike, hint: ArrayLike | None = None) -> NDArray[np.float64]:
        u, v, _ = self.basis(hint)
        q = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        return self.point + q[:, 0:1] * u + q[:, 1:2] * v

    def transform(self, t: Transform) -> Plane:
        if t.source is not self.frame:
            raise FrameMismatchError(
                f"plane is in '{self.frame.value}' but transform expects "
                f"'{t.source.value}'"
            )
        return Plane(
            t.apply(self.point),
            t.apply_normal(self.normal),
            frame=t.target,
            label=self.label,
        )

    def angle_to(self, other: Plane | ArrayLike) -> float:
        n = other.normal if isinstance(other, Plane) else other
        a = angle_between_deg(self.normal, n)
        return min(a, 180.0 - a)

    def intersect_line(self, origin: ArrayLike, direction: ArrayLike) -> NDArray[np.float64]:
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        d = unit(direction)
        denom = float(np.dot(self.normal, d))
        if abs(denom) < 1e-9:
            raise ValidationError("line is parallel to the plane; no unique intersection")
        t = (self.offset - float(np.dot(self.normal, o))) / denom
        return o + t * d

    def to_dict(self) -> dict[str, object]:
        return {
            "point": self.point.tolist(),
            "normal": self.normal.tolist(),
            "frame": self.frame.value,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Plane:
        return cls(
            np.asarray(payload["point"], dtype=np.float64),
            np.asarray(payload["normal"], dtype=np.float64),
            frame=Frame(payload["frame"]),
            label=payload.get("label", ""),
        )


# ---------------------------------------------------------------------------
# Axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """A directed line: ``origin + t * direction``.

    Clinical axes (mechanical, anatomical, transepicondylar, posterior
    condylar) are all instances of this.
    """

    origin: NDArray[np.float64]
    direction: NDArray[np.float64]
    frame: Frame
    label: str = ""

    def __post_init__(self) -> None:
        o = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        if o.size != 3:
            raise ValidationError("axis origin must be 3D")
        object.__setattr__(self, "origin", o)
        object.__setattr__(self, "direction", unit(self.direction))

    @classmethod
    def through(
        cls, a: ArrayLike, b: ArrayLike, *, frame: Frame, label: str = ""
    ) -> Axis:
        pa = np.asarray(a, dtype=np.float64).reshape(3)
        pb = np.asarray(b, dtype=np.float64).reshape(3)
        d = pb - pa
        if float(np.linalg.norm(d)) < 1e-7:
            raise ValidationError(f"axis endpoints coincide (label={label!r})")
        return cls(pa, d, frame=frame, label=label)

    def point_at(self, t: float) -> NDArray[np.float64]:
        return self.origin + float(t) * self.direction

    def closest_point(self, p: ArrayLike) -> NDArray[np.float64]:
        q = np.asarray(p, dtype=np.float64).reshape(3)
        return self.origin + float(np.dot(q - self.origin, self.direction)) * self.direction

    def distance_to_point(self, p: ArrayLike) -> float:
        q = np.asarray(p, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(q - self.closest_point(q)))

    def angle_to(self, other: Axis | ArrayLike, *, signed_about: ArrayLike | None = None) -> float:
        d = other.direction if isinstance(other, Axis) else other
        if signed_about is not None:
            return signed_angle_in_plane_deg(self.direction, d, signed_about)
        a = angle_between_deg(self.direction, d)
        return min(a, 180.0 - a)

    def perpendicular_plane(self, at: ArrayLike | None = None, label: str = "") -> Plane:
        pt = self.origin if at is None else np.asarray(at, dtype=np.float64).reshape(3)
        return Plane(pt, self.direction, frame=self.frame, label=label or f"{self.label}_perp")

    def reversed(self) -> Axis:
        return Axis(self.origin, -self.direction, frame=self.frame, label=self.label)

    def transform(self, t: Transform) -> Axis:
        if t.source is not self.frame:
            raise FrameMismatchError(
                f"axis is in '{self.frame.value}' but transform expects '{t.source.value}'"
            )
        return Axis(
            t.apply(self.origin),
            t.apply_vector(self.direction),
            frame=t.target,
            label=self.label,
        )

    def with_label(self, label: str) -> Axis:
        return replace(self, label=label)

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin.tolist(),
            "direction": self.direction.tolist(),
            "frame": self.frame.value,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Axis:
        return cls(
            np.asarray(payload["origin"], dtype=np.float64),
            np.asarray(payload["direction"], dtype=np.float64),
            frame=Frame(payload["frame"]),
            label=payload.get("label", ""),
        )


@dataclass(frozen=True)
class Sphere:
    center: NDArray[np.float64]
    radius_mm: float
    frame: Frame
    fit_rms_mm: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        c = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if c.size != 3:
            raise ValidationError("sphere center must be 3D")
        if self.radius_mm <= 0:
            raise ValidationError("sphere radius must be positive")
        object.__setattr__(self, "center", c)


@dataclass(frozen=True)
class OrientedBox:
    """Oriented bounding box, used for ROI extraction and workspace limits."""

    center: NDArray[np.float64]
    rotation: NDArray[np.float64]
    extents_mm: NDArray[np.float64]
    frame: Frame
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float64).reshape(3))
        R = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        object.__setattr__(self, "rotation", R)
        e = np.asarray(self.extents_mm, dtype=np.float64).reshape(3)
        if np.any(e <= 0):
            raise ValidationError("box extents must be positive")
        object.__setattr__(self, "extents_mm", e)

    def contains(self, points: ArrayLike) -> NDArray[np.bool_]:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        local = (p - self.center) @ self.rotation
        half = self.extents_mm / 2.0
        return np.all(np.abs(local) <= half + 1e-9, axis=1)

    @property
    def volume_mm3(self) -> float:
        return float(np.prod(self.extents_mm))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_plane(points: ArrayLike, *, frame: Frame, label: str = "") -> tuple[Plane, float]:
    """Total-least-squares plane fit. Returns ``(plane, rms_residual_mm)``."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if p.shape[0] < 3:
        raise ValidationError("plane fit needs at least 3 points")
    centroid = p.mean(axis=0)
    _u, _s, vt = np.linalg.svd(p - centroid, full_matrices=False)
    normal = vt[-1]
    plane = Plane(centroid, normal, frame=frame, label=label)
    resid = np.asarray(plane.signed_distance(p))
    return plane, float(np.sqrt(np.mean(resid**2)))


def fit_line(points: ArrayLike, *, frame: Frame, label: str = "") -> tuple[Axis, float]:
    """Total-least-squares line fit. Returns ``(axis, rms_residual_mm)``."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if p.shape[0] < 2:
        raise ValidationError("line fit needs at least 2 points")
    centroid = p.mean(axis=0)
    _u, _s, vt = np.linalg.svd(p - centroid, full_matrices=False)
    axis = Axis(centroid, vt[0], frame=frame, label=label)
    rel = p - centroid
    proj = np.outer(rel @ axis.direction, axis.direction)
    resid = np.linalg.norm(rel - proj, axis=1)
    return axis, float(np.sqrt(np.mean(resid**2)))


def fit_sphere(points: ArrayLike, *, frame: Frame, label: str = "") -> Sphere:
    """Algebraic sphere fit (Coope's linear formulation).

    Used to derive the femoral head centre from a partial articular surface,
    which is what defines the femoral mechanical axis. When the imaging field of
    view does not include the hip, this cannot be used and the mechanical axis
    must come from a long-leg radiograph instead.
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if p.shape[0] < 4:
        raise ValidationError("sphere fit needs at least 4 points")
    A = np.hstack([2.0 * p, np.ones((p.shape[0], 1))])
    b = np.sum(p**2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    r2 = sol[3] + float(np.dot(center, center))
    if r2 <= 0:
        raise ValidationError("sphere fit produced a non-positive radius")
    radius = math.sqrt(r2)
    resid = np.linalg.norm(p - center, axis=1) - radius
    return Sphere(
        center,
        radius,
        frame=frame,
        fit_rms_mm=float(np.sqrt(np.mean(resid**2))),
        label=label,
    )


def principal_axes(
    points: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """PCA of a point cloud: ``(centroid, axes_as_columns, singular_values)``.

    Axes are ordered by decreasing variance. For a long bone the first axis is
    the shaft direction, which gives a cheap and robust anatomical-axis estimate.
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if p.shape[0] < 3:
        raise ValidationError("PCA needs at least 3 points")
    centroid = p.mean(axis=0)
    _u, s, vt = np.linalg.svd(p - centroid, full_matrices=False)
    axes = vt.T
    if float(np.linalg.det(axes)) < 0:
        axes[:, 2] *= -1.0
    return centroid, axes, s


def extreme_point(
    points: ArrayLike,
    direction: ArrayLike,
    *,
    percentile: float | None = None,
) -> NDArray[np.float64]:
    """Point furthest along ``direction``.

    ``percentile`` (0-100) selects a robust extreme instead of the single most
    extreme vertex — segmentation noise routinely produces one stray voxel that
    would otherwise define a clinical landmark. When given, the returned point
    is the centroid of vertices beyond that percentile.
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    d = unit(direction)
    proj = p @ d
    if percentile is None:
        return p[int(np.argmax(proj))]
    if not 0.0 < percentile < 100.0:
        raise ValidationError("percentile must be in (0, 100)")
    threshold = float(np.percentile(proj, percentile))
    sel = p[proj >= threshold]
    if sel.size == 0:  # pragma: no cover - defensive
        return p[int(np.argmax(proj))]
    return sel.mean(axis=0)


def bounding_extent(points: ArrayLike, direction: ArrayLike) -> float:
    """Extent of a point cloud along ``direction``, in millimetres."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    proj = p @ unit(direction)
    return float(proj.max() - proj.min())
