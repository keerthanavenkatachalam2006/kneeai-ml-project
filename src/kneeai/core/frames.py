"""Frame-tagged coordinate systems and transforms.

PRD §71 states: never mix image, world, mesh and robot coordinates without an
explicit transformation. A comment cannot enforce that. This module makes the
frame part of the *type*, so mixing frames raises instead of silently producing
a plausible-looking wrong number.

Convention
----------
``Transform(source=A, target=B)`` maps a point expressed in frame ``A`` into
frame ``B``. Composition follows matrix order::

    T_a_to_c = T_b_to_c @ T_a_to_b

Points are row-major ``(N, 3)`` arrays in millimetres.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.errors import FrameMismatchError, TransformError

__all__ = [
    "Frame",
    "Transform",
    "FramedPoints",
    "TransformGraph",
    "NAVIGATION_CHAIN",
    "as_points",
    "rotation_matrix",
    "rotation_from_axis_angle",
    "rotation_between_vectors",
    "rigid_from_correspondences",
]

_EPS = 1e-9


class Frame(str, Enum):
    """Named coordinate frames used across the platform.

    The order of the navigation chain is fixed by PRD §72:
    image -> patient -> planning -> tracking -> robot.
    """

    #: Discrete voxel indices (i, j, k) of an image volume. Unitless.
    VOXEL = "voxel"

    #: DICOM patient/world space in millimetres (LPS after normalization).
    IMAGE = "image"

    #: Anatomical patient frame derived from landmarks: origin at the knee
    #: centre, axes aligned to the mechanical/transepicondylar convention.
    PATIENT = "patient"

    #: Local frame a mesh file was authored in, before being placed.
    MESH = "mesh"

    #: Implant design frame, as defined by the manufacturer's reference system.
    IMPLANT = "implant"

    #: Frame the surgical plan is expressed in. Equal to PATIENT by convention
    #: but kept distinct so a plan can be re-based without touching anatomy.
    PLANNING = "planning"

    #: Optical/EM tracker camera frame.
    TRACKING = "tracking"

    #: Rigid body attached to the patient's bone, seen by the tracker.
    PATIENT_TRACKER = "patient_tracker"

    #: Robot kinematic base.
    ROBOT_BASE = "robot_base"

    #: Robot end-effector flange.
    ROBOT_FLANGE = "robot_flange"

    #: Cutting tool tip.
    TOOL = "tool"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Frame.{self.name}"


#: The transform chain that must be fully resolved and validated before any
#: robotic execution (PRD §72).
NAVIGATION_CHAIN: tuple[Frame, ...] = (
    Frame.IMAGE,
    Frame.PATIENT,
    Frame.PLANNING,
    Frame.TRACKING,
    Frame.ROBOT_BASE,
)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _unit(v: ArrayLike, *, name: str = "vector") -> NDArray[np.float64]:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise TransformError(f"{name} must have 3 components, got shape {np.shape(v)}")
    norm = float(np.linalg.norm(arr))
    if norm < _EPS:
        raise TransformError(f"{name} has zero length and cannot be normalized")
    return arr / norm


def rotation_from_axis_angle(axis: ArrayLike, angle_deg: float) -> NDArray[np.float64]:
    """Rodrigues rotation about ``axis`` by ``angle_deg`` degrees (right-handed)."""
    k = _unit(axis, name="axis")
    theta = math.radians(float(angle_deg))
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rotation_matrix(
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    z: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Build an orthonormal rotation matrix from two specified axes.

    Supply exactly two of ``x``, ``y``, ``z``. The first named axis is treated
    as primary (kept exactly), the second is orthogonalized against it, and the
    third is the cross product. Columns of the result are the new basis vectors
    expressed in the old frame.
    """
    given = {k: v for k, v in (("x", x), ("y", y), ("z", z)) if v is not None}
    if len(given) != 2:
        raise TransformError(
            f"rotation_matrix requires exactly two axes, got {sorted(given)}"
        )
    (n1, v1), (n2, v2) = given.items()
    a1 = _unit(v1, name=n1)
    a2 = _unit(v2, name=n2)
    # Orthogonalize the secondary axis against the primary.
    a2 = a2 - np.dot(a2, a1) * a1
    if float(np.linalg.norm(a2)) < 1e-6:
        raise TransformError(f"axes {n1} and {n2} are parallel; cannot build a frame")
    a2 = a2 / np.linalg.norm(a2)

    axes: dict[str, NDArray[np.float64]] = {n1: a1, n2: a2}
    missing = ({"x", "y", "z"} - set(axes)).pop()

    # Keep the basis right-handed. (x,y,z) cycles as 0->1->2->0, so if the two
    # supplied axes are in cyclic order the third is a1 x a2, otherwise a2 x a1.
    order = ("x", "y", "z")
    idx1, idx2 = order.index(n1), order.index(n2)
    cyclic = (idx2 - idx1) % 3 == 1
    axes[missing] = np.cross(a1, a2) if cyclic else np.cross(a2, a1)

    R = np.column_stack([axes["x"], axes["y"], axes["z"]])
    if float(np.linalg.det(R)) < 0:  # pragma: no cover - defensive
        raise TransformError("failed to construct a right-handed basis")
    return R


def rotation_between_vectors(source: ArrayLike, target: ArrayLike) -> NDArray[np.float64]:
    """Shortest-arc rotation taking ``source`` onto ``target``."""
    a = _unit(source, name="source")
    b = _unit(target, name="target")
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # Anti-parallel: rotate 180 deg about any axis orthogonal to a.
        seed = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(seed, a)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, seed)
        return rotation_from_axis_angle(axis, 180.0)
    axis = np.cross(a, b)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return rotation_from_axis_angle(axis, angle)


def rigid_from_correspondences(
    source_points: ArrayLike,
    target_points: ArrayLike,
    *,
    source_frame: Frame,
    target_frame: Frame,
    weights: ArrayLike | None = None,
    label: str | None = None,
) -> Transform:
    """Least-squares rigid registration (Kabsch/Horn) between point sets.

    This is the workhorse for patient registration: paired anatomical points
    picked on the 3D model and touched with a tracked probe intraoperatively.

    The residual RMS is stored on the returned transform. Registration error is
    not a diagnostic curiosity — it bounds the accuracy of everything
    downstream, so it travels with the transform rather than being logged and
    forgotten.
    """
    P = np.asarray(source_points, dtype=np.float64)
    Q = np.asarray(target_points, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise TransformError(
            f"correspondence arrays must both be (N, 3); got {P.shape} and {Q.shape}"
        )
    if P.shape[0] < 3:
        raise TransformError("rigid registration needs at least 3 correspondences")

    if weights is None:
        w = np.ones(P.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != P.shape[0]:
            raise TransformError("weights length must match number of correspondences")
        if np.any(w < 0):
            raise TransformError("weights must be non-negative")
    wsum = float(w.sum())
    if wsum < _EPS:
        raise TransformError("weights sum to zero")

    p_bar = (w[:, None] * P).sum(axis=0) / wsum
    q_bar = (w[:, None] * Q).sum(axis=0) / wsum
    Pc = P - p_bar
    Qc = Q - q_bar

    H = (w[:, None] * Pc).T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    t = q_bar - R @ p_bar

    residuals = (R @ P.T).T + t - Q
    rms = float(np.sqrt((w * np.einsum("ij,ij->i", residuals, residuals)).sum() / wsum))

    return Transform.from_rotation_translation(
        R,
        t,
        source=source_frame,
        target=target_frame,
        label=label or "rigid_registration",
        rms_error_mm=rms,
        n_correspondences=int(P.shape[0]),
    )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """An immutable, frame-tagged homogeneous transform.

    Attributes:
        matrix: 4x4 homogeneous matrix mapping ``source`` -> ``target``.
        source: frame the input points are expressed in.
        target: frame the output points are expressed in.
        label: human-readable provenance ("landmark_alignment", "icp_refine"...).
        timestamp: acquisition time for time-varying transforms (tracking).
        rms_error_mm: residual error if this transform came from a fit.
        n_correspondences: number of points used, if fitted.
    """

    matrix: NDArray[np.float64]
    source: Frame
    target: Frame
    label: str = ""
    timestamp: float | None = None
    rms_error_mm: float | None = None
    n_correspondences: int | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise TransformError(f"transform matrix must be 4x4, got {m.shape}")
        if not np.all(np.isfinite(m)):
            raise TransformError("transform matrix contains non-finite values")
        if not np.allclose(m[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise TransformError(
                f"bottom row of a homogeneous transform must be [0,0,0,1], got {m[3, :]}"
            )
        if self.source is self.target and self.label == "":
            # Allowed, but keep it explicit so identity self-loops in the graph
            # are traceable.
            object.__setattr__(self, "label", "identity")
        object.__setattr__(self, "matrix", m.copy())
        self.matrix.setflags(write=False)

    # --- constructors ------------------------------------------------------

    @classmethod
    def identity(cls, frame: Frame, *, target: Frame | None = None) -> Transform:
        return cls(np.eye(4), source=frame, target=target or frame, label="identity")

    @classmethod
    def from_rotation_translation(
        cls,
        rotation: ArrayLike,
        translation: ArrayLike,
        *,
        source: Frame,
        target: Frame,
        label: str = "",
        **meta: object,
    ) -> Transform:
        R = np.asarray(rotation, dtype=np.float64)
        if R.shape != (3, 3):
            raise TransformError(f"rotation must be 3x3, got {R.shape}")
        t = np.asarray(translation, dtype=np.float64).reshape(-1)
        if t.size != 3:
            raise TransformError(f"translation must have 3 components, got {t.size}")
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = t
        return cls(m, source=source, target=target, label=label, **meta)  # type: ignore[arg-type]

    @classmethod
    def from_translation(
        cls, translation: ArrayLike, *, source: Frame, target: Frame, label: str = ""
    ) -> Transform:
        return cls.from_rotation_translation(
            np.eye(3), translation, source=source, target=target, label=label
        )

    @classmethod
    def from_basis(
        cls,
        origin: ArrayLike,
        rotation: ArrayLike,
        *,
        source: Frame,
        target: Frame,
        label: str = "",
    ) -> Transform:
        """Transform *into* a frame defined by ``origin`` and basis ``rotation``.

        ``rotation`` columns are the new frame's axes expressed in the source
        frame. The result maps source points into the new frame, i.e. it applies
        ``R^T (p - origin)``.
        """
        R = np.asarray(rotation, dtype=np.float64)
        if R.shape != (3, 3):
            raise TransformError(f"rotation must be 3x3, got {R.shape}")
        o = np.asarray(origin, dtype=np.float64).reshape(-1)
        if o.size != 3:
            raise TransformError("origin must have 3 components")
        Rt = R.T
        return cls.from_rotation_translation(
            Rt, -Rt @ o, source=source, target=target, label=label or "basis"
        )

    @classmethod
    def from_image_geometry(
        cls,
        origin: ArrayLike,
        spacing: ArrayLike,
        direction: ArrayLike,
        *,
        label: str = "voxel_to_image",
    ) -> Transform:
        """Build the VOXEL -> IMAGE transform from DICOM/ITK geometry.

        ``direction`` is the 3x3 direction cosine matrix whose columns are the
        image axes in patient space (ITK convention).
        """
        o = np.asarray(origin, dtype=np.float64).reshape(-1)
        s = np.asarray(spacing, dtype=np.float64).reshape(-1)
        D = np.asarray(direction, dtype=np.float64)
        if D.size == 9:
            D = D.reshape(3, 3)
        else:
            raise TransformError(f"direction must have 9 elements, got {D.size}")
        if o.size != 3 or s.size != 3:
            raise TransformError("origin and spacing must each have 3 components")
        if np.any(s <= 0):
            raise TransformError(f"voxel spacing must be positive, got {s}")
        return cls.from_rotation_translation(
            D @ np.diag(s), o, source=Frame.VOXEL, target=Frame.IMAGE, label=label
        )

    # --- properties --------------------------------------------------------

    @property
    def rotation(self) -> NDArray[np.float64]:
        return np.array(self.matrix[:3, :3])

    @property
    def translation(self) -> NDArray[np.float64]:
        return np.array(self.matrix[:3, 3])

    @property
    def is_rigid(self) -> bool:
        """True when the linear part is a proper rotation (no scale/shear/flip)."""
        R = self.matrix[:3, :3]
        return bool(
            np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            and abs(float(np.linalg.det(R)) - 1.0) < 1e-6
        )

    @property
    def scale_factors(self) -> NDArray[np.float64]:
        """Per-axis scale of the linear part (1,1,1 for a rigid transform)."""
        return np.linalg.norm(self.matrix[:3, :3], axis=0)

    def require_rigid(self, context: str = "") -> Transform:
        """Assert rigidity. Anatomical registration must never scale a patient."""
        if not self.is_rigid:
            where = f" in {context}" if context else ""
            raise TransformError(
                f"transform {self.describe()} must be rigid{where}; "
                f"scale factors are {np.round(self.scale_factors, 6).tolist()}"
            )
        return self

    def describe(self) -> str:
        tag = f" [{self.label}]" if self.label else ""
        return f"{self.source.value}->{self.target.value}{tag}"

    # --- algebra -----------------------------------------------------------

    def inverse(self) -> Transform:
        try:
            inv = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as exc:
            raise TransformError(
                f"transform {self.describe()} is singular and cannot be inverted"
            ) from exc
        return Transform(
            inv,
            source=self.target,
            target=self.source,
            label=f"inv({self.label})" if self.label else "",
            timestamp=self.timestamp,
            rms_error_mm=self.rms_error_mm,
            n_correspondences=self.n_correspondences,
        )

    def __matmul__(self, other: Transform) -> Transform:
        """Compose: ``self @ other`` yields ``other.source -> self.target``."""
        if not isinstance(other, Transform):
            return NotImplemented
        if other.target is not self.source:
            raise FrameMismatchError(
                f"cannot compose {self.describe()} with {other.describe()}: "
                f"expected inner frame '{self.source.value}' but got "
                f"'{other.target.value}'. Register the missing transform "
                "explicitly instead of assuming the frames coincide."
            )
        stamps = [t for t in (self.timestamp, other.timestamp) if t is not None]
        errs = [e for e in (self.rms_error_mm, other.rms_error_mm) if e is not None]
        return Transform(
            self.matrix @ other.matrix,
            source=other.source,
            target=self.target,
            label=f"{other.label}>{self.label}".strip(">"),
            timestamp=max(stamps) if stamps else None,
            # Errors add in quadrature along a chain.
            rms_error_mm=float(np.sqrt(sum(e * e for e in errs))) if errs else None,
        )

    def then(self, other: Transform) -> Transform:
        """Readable composition: ``a.then(b)`` == ``b @ a``."""
        return other @ self

    # --- application -------------------------------------------------------

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform raw ``(N, 3)`` or ``(3,)`` coordinates. No frame checking.

        Prefer :meth:`apply_points` with :class:`FramedPoints` so the frame is
        validated.
        """
        p = np.asarray(points, dtype=np.float64)
        squeeze = p.ndim == 1
        p = np.atleast_2d(p)
        if p.shape[-1] != 3:
            raise TransformError(f"points must be (N, 3); got {p.shape}")
        out = (self.matrix[:3, :3] @ p.T).T + self.matrix[:3, 3]
        return out[0] if squeeze else out

    def apply_points(self, points: FramedPoints) -> FramedPoints:
        if points.frame is not self.source:
            raise FrameMismatchError(
                f"transform {self.describe()} expects points in "
                f"'{self.source.value}' but received '{points.frame.value}'"
            )
        return FramedPoints(self.apply(points.array), frame=self.target)

    def apply_vector(self, vectors: ArrayLike) -> NDArray[np.float64]:
        """Rotate a direction (no translation). Use for normals and axes."""
        v = np.asarray(vectors, dtype=np.float64)
        squeeze = v.ndim == 1
        v = np.atleast_2d(v)
        if v.shape[-1] != 3:
            raise TransformError(f"vectors must be (N, 3); got {v.shape}")
        out = (self.matrix[:3, :3] @ v.T).T
        return out[0] if squeeze else out

    def apply_normal(self, normals: ArrayLike) -> NDArray[np.float64]:
        """Transform a surface normal (inverse-transpose; matters if scaled)."""
        n = np.asarray(normals, dtype=np.float64)
        squeeze = n.ndim == 1
        n = np.atleast_2d(n)
        M = np.linalg.inv(self.matrix[:3, :3]).T
        out = (M @ n.T).T
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.divide(out, norms, out=np.zeros_like(out), where=norms > _EPS)
        return out[0] if squeeze else out

    # --- introspection -----------------------------------------------------

    def angle_deg(self) -> float:
        """Rotation magnitude in degrees."""
        R = self.matrix[:3, :3]
        cos = (float(np.trace(R)) - 1.0) / 2.0
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    def translation_norm_mm(self) -> float:
        return float(np.linalg.norm(self.matrix[:3, 3]))

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.value,
            "target": self.target.value,
            "matrix": self.matrix.tolist(),
            "label": self.label,
            "timestamp": self.timestamp,
            "rms_error_mm": self.rms_error_mm,
            "n_correspondences": self.n_correspondences,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Transform:
        return cls(
            np.asarray(payload["matrix"], dtype=np.float64),
            source=Frame(payload["source"]),
            target=Frame(payload["target"]),
            label=payload.get("label", ""),
            timestamp=payload.get("timestamp"),
            rms_error_mm=payload.get("rms_error_mm"),
            n_correspondences=payload.get("n_correspondences"),
        )

    def with_label(self, label: str) -> Transform:
        return replace(self, label=label)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Transform({self.describe()}, rot={self.angle_deg():.2f}deg, "
            f"trans={self.translation_norm_mm():.2f}mm)"
        )


# ---------------------------------------------------------------------------
# FramedPoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FramedPoints:
    """An ``(N, 3)`` point set that knows which frame it lives in."""

    array: NDArray[np.float64]
    frame: Frame

    def __post_init__(self) -> None:
        a = np.asarray(self.array, dtype=np.float64)
        a = np.atleast_2d(a)
        if a.ndim != 2 or a.shape[1] != 3:
            raise TransformError(f"FramedPoints requires (N, 3); got {a.shape}")
        object.__setattr__(self, "array", a)

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def __iter__(self) -> Iterator[NDArray[np.float64]]:
        return iter(self.array)

    def __getitem__(self, idx: int | slice) -> FramedPoints:
        sub = np.atleast_2d(self.array[idx])
        return FramedPoints(sub, frame=self.frame)

    def require_frame(self, frame: Frame, context: str = "") -> FramedPoints:
        if self.frame is not frame:
            where = f" ({context})" if context else ""
            raise FrameMismatchError(
                f"expected points in '{frame.value}' but got '{self.frame.value}'{where}"
            )
        return self

    @property
    def centroid(self) -> NDArray[np.float64]:
        return self.array.mean(axis=0)

    def to(self, transform: Transform) -> FramedPoints:
        return transform.apply_points(self)

    def concat(self, other: FramedPoints) -> FramedPoints:
        other.require_frame(self.frame, "concat")
        return FramedPoints(np.vstack([self.array, other.array]), frame=self.frame)


def as_points(values: ArrayLike, frame: Frame) -> FramedPoints:
    return FramedPoints(np.asarray(values, dtype=np.float64), frame=frame)


# ---------------------------------------------------------------------------
# TransformGraph
# ---------------------------------------------------------------------------


@dataclass
class TransformGraph:
    """A registry of transforms that can resolve composite paths between frames.

    Used to hold a case's full transform chain (PRD §72). Resolution is an
    explicit, auditable operation: every resolved path records which registered
    transforms it traversed, and the accumulated registration error.
    """

    _edges: dict[Frame, dict[Frame, Transform]] = field(default_factory=dict)

    def add(self, transform: Transform, *, allow_replace: bool = False) -> TransformGraph:
        src, dst = transform.source, transform.target
        if src is dst:
            raise TransformError(
                f"refusing to register a self-loop on frame '{src.value}'"
            )
        existing = self._edges.get(src, {}).get(dst)
        if existing is not None and not allow_replace:
            raise TransformError(
                f"transform {src.value}->{dst.value} already registered "
                f"({existing.label!r}). Pass allow_replace=True to update it, so "
                "overwriting a registration is always deliberate."
            )
        self._edges.setdefault(src, {})[dst] = transform
        # Register the inverse so resolution can walk either direction, but keep
        # the derived flag visible in the label.
        inv = transform.inverse()
        self._edges.setdefault(dst, {})[src] = inv
        return self

    def add_all(self, transforms: Iterable[Transform], **kw: bool) -> TransformGraph:
        for t in transforms:
            self.add(t, **kw)
        return self

    def has(self, source: Frame, target: Frame) -> bool:
        try:
            self.path(source, target)
        except FrameMismatchError:
            return False
        return True

    def frames(self) -> set[Frame]:
        out: set[Frame] = set(self._edges)
        for nbrs in self._edges.values():
            out.update(nbrs)
        return out

    def path(self, source: Frame, target: Frame) -> list[Frame]:
        """Shortest frame path, breadth-first. Fewer hops = less accumulated error."""
        if source is target:
            return [source]
        if source not in self._edges:
            raise FrameMismatchError(
                f"frame '{source.value}' has no registered transforms; "
                f"known frames: {sorted(f.value for f in self.frames())}"
            )
        prev: dict[Frame, Frame] = {source: source}
        q: deque[Frame] = deque([source])
        while q:
            cur = q.popleft()
            for nxt in self._edges.get(cur, {}):
                if nxt in prev:
                    continue
                prev[nxt] = cur
                if nxt is target:
                    chain = [target]
                    while chain[-1] is not source:
                        chain.append(prev[chain[-1]])
                    return list(reversed(chain))
                q.append(nxt)
        raise FrameMismatchError(
            f"no registered transform path from '{source.value}' to '{target.value}'. "
            f"Registered frames: {sorted(f.value for f in self.frames())}. "
            "Every frame conversion must be explicit (PRD §71)."
        )

    def resolve(self, source: Frame, target: Frame) -> Transform:
        """Compose the registered transforms taking ``source`` to ``target``."""
        chain = self.path(source, target)
        if len(chain) == 1:
            return Transform.identity(source)
        composite = self._edges[chain[0]][chain[1]]
        for a, b in zip(chain[1:], chain[2:]):
            composite = self._edges[a][b] @ composite
        return composite.with_label("+".join(f.value for f in chain))

    def validate_chain(self, chain: Sequence[Frame] = NAVIGATION_CHAIN) -> list[Transform]:
        """Verify every consecutive hop of a required chain is registered.

        Called before navigation or robotic execution is permitted. Returns the
        per-hop transforms so the caller can inspect residual errors.
        """
        hops: list[Transform] = []
        missing: list[str] = []
        for a, b in zip(chain, chain[1:]):
            try:
                hops.append(self.resolve(a, b))
            except FrameMismatchError:
                missing.append(f"{a.value}->{b.value}")
        if missing:
            raise FrameMismatchError(
                "incomplete transform chain; missing links: " + ", ".join(missing)
            )
        return hops

    def total_error_mm(self, source: Frame, target: Frame) -> float | None:
        return self.resolve(source, target).rms_error_mm

    def to_dict(self) -> dict[str, object]:
        seen: list[dict[str, object]] = []
        emitted: set[tuple[Frame, Frame]] = set()
        for src, nbrs in self._edges.items():
            for dst, t in nbrs.items():
                if (dst, src) in emitted:
                    continue
                emitted.add((src, dst))
                seen.append(t.to_dict())
        return {"transforms": seen}

    @classmethod
    def from_dict(cls, payload: dict) -> TransformGraph:
        graph = cls()
        for item in payload.get("transforms", []):
            graph.add(Transform.from_dict(item), allow_replace=True)
        return graph
