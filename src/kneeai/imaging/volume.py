"""The platform's 3D image representation.

:class:`Volume` is a numpy array plus the geometry needed to place it in patient
space. It is deliberately independent of SimpleITK so the geometry, preprocessing
and reconstruction logic can be tested without the imaging extra installed.

Index convention
----------------
``array[i, j, k]`` corresponds to voxel index ``(i, j, k)``, and
``voxel_to_image`` maps that index triple to millimetres in the IMAGE frame.

This is the *opposite* of what ``SimpleITK.GetArrayFromImage`` returns (which is
``(k, j, i)``). The conversion happens exactly once, in :meth:`Volume.from_sitk`
and :meth:`Volume.to_sitk`, rather than being re-derived at every call site —
axis-order bugs of this kind produce anatomically mirrored output that still
looks plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kneeai.core.anatomy import Modality, Side, Structure
from kneeai.core.errors import PreprocessingError, ValidationError
from kneeai.core.frames import Frame, FramedPoints, Transform
from kneeai.core.geometry import OrientedBox
from kneeai.core.provenance import Provenance, hash_array

__all__ = ["Volume", "orientation_code", "AXIS_LETTERS"]

#: DICOM patient axes are LPS: +x -> Left, +y -> Posterior, +z -> Superior.
AXIS_LETTERS: dict[int, tuple[str, str]] = {
    0: ("R", "L"),  # negative, positive
    1: ("A", "P"),
    2: ("I", "S"),
}

_LETTER_TO_AXIS: dict[str, tuple[int, int]] = {}
for _axis, (_neg, _pos) in AXIS_LETTERS.items():
    _LETTER_TO_AXIS[_neg] = (_axis, -1)
    _LETTER_TO_AXIS[_pos] = (_axis, +1)


def orientation_code(direction: ArrayLike) -> str:
    """Three-letter orientation code for a 3x3 direction cosine matrix.

    Each letter names the patient direction the corresponding *image* axis points
    towards. An identity direction matrix yields ``"LPS"``.
    """
    D = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    letters: list[str] = []
    used: set[int] = set()
    for col in range(3):
        v = D[:, col]
        order = np.argsort(-np.abs(v))
        axis = next((int(a) for a in order if int(a) not in used), int(order[0]))
        used.add(axis)
        neg, pos = AXIS_LETTERS[axis]
        letters.append(pos if v[axis] >= 0 else neg)
    return "".join(letters)


@dataclass(frozen=True)
class Volume:
    """A 3D image or label volume with explicit patient-space geometry.

    Attributes:
        array: ``(ni, nj, nk)`` voxel data.
        spacing: voxel size in millimetres along each image axis.
        origin: patient-space position of voxel ``(0, 0, 0)``, millimetres.
        direction: 3x3 matrix whose columns are the image axes in patient space.
        modality: acquisition modality.
        side: which knee, when known.
        is_label: True for segmentation masks (changes interpolation rules).
        label_map: structure -> integer label, required when ``is_label``.
        metadata: de-identified acquisition metadata.
        provenance: how this volume was produced.
    """

    array: NDArray[Any]
    spacing: NDArray[np.float64] = field(default_factory=lambda: np.ones(3))
    origin: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    direction: NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    modality: Modality = Modality.UNKNOWN
    side: Side = Side.UNKNOWN
    is_label: bool = False
    label_map: Mapping[Structure, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        arr = np.asarray(self.array)
        if arr.ndim != 3:
            raise ValidationError(
                f"Volume requires a 3D array, got shape {arr.shape}. Multi-channel "
                "data should be held as separate Volumes."
            )
        object.__setattr__(self, "array", arr)

        sp = np.asarray(self.spacing, dtype=np.float64).reshape(-1)
        if sp.size != 3:
            raise ValidationError(f"spacing must have 3 components, got {sp.size}")
        if np.any(sp <= 0) or not np.all(np.isfinite(sp)):
            raise ValidationError(f"voxel spacing must be finite and positive, got {sp}")
        object.__setattr__(self, "spacing", sp)

        org = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        if org.size != 3:
            raise ValidationError(f"origin must have 3 components, got {org.size}")
        object.__setattr__(self, "origin", org)

        D = np.asarray(self.direction, dtype=np.float64)
        D = D.reshape(3, 3) if D.size == 9 else D
        if D.shape != (3, 3):
            raise ValidationError(f"direction must be 3x3, got {D.shape}")
        if abs(abs(float(np.linalg.det(D))) - 1.0) > 1e-3:
            raise ValidationError(
                f"direction matrix must be orthonormal (|det| == 1), got "
                f"det={float(np.linalg.det(D)):.6f}"
            )
        object.__setattr__(self, "direction", D)

        if self.is_label:
            if not np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(
                arr.dtype, np.bool_
            ):
                raise ValidationError(
                    f"label volumes must have an integer dtype, got {arr.dtype}"
                )
            if self.label_map is None:
                raise ValidationError(
                    "label volumes must declare a label_map; an unlabelled mask "
                    "cannot be interpreted downstream (PRD §113)"
                )

    # --- geometry -----------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.array.shape)  # type: ignore[return-value]

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.array.shape))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def voxel_to_image(self) -> Transform:
        return Transform.from_image_geometry(self.origin, self.spacing, self.direction)

    @property
    def image_to_voxel(self) -> Transform:
        return self.voxel_to_image.inverse()

    @property
    def orientation(self) -> str:
        return orientation_code(self.direction)

    @property
    def physical_extent_mm(self) -> NDArray[np.float64]:
        """Size of the field of view along each image axis."""
        return np.asarray(self.shape, dtype=np.float64) * self.spacing

    @property
    def center_image(self) -> NDArray[np.float64]:
        centre_idx = (np.asarray(self.shape, dtype=np.float64) - 1.0) / 2.0
        return self.voxel_to_image.apply(centre_idx)

    def index_to_physical(self, indices: ArrayLike) -> NDArray[np.float64]:
        return self.voxel_to_image.apply(indices)

    def physical_to_index(self, points: ArrayLike, *, rounded: bool = False) -> NDArray[Any]:
        idx = self.image_to_voxel.apply(points)
        return np.rint(idx).astype(np.int64) if rounded else idx

    def framed_points(self, indices: ArrayLike) -> FramedPoints:
        """Convert voxel indices to :class:`FramedPoints` in the IMAGE frame."""
        return FramedPoints(self.index_to_physical(indices), frame=Frame.IMAGE)

    def contains_index(self, indices: ArrayLike) -> NDArray[np.bool_]:
        idx = np.atleast_2d(np.asarray(indices, dtype=np.float64))
        upper = np.asarray(self.shape, dtype=np.float64) - 1.0
        return np.all((idx >= 0) & (idx <= upper), axis=1)

    def bounding_box(self) -> OrientedBox:
        return OrientedBox(
            center=self.center_image,
            rotation=self.direction,
            extents_mm=self.physical_extent_mm,
            frame=Frame.IMAGE,
            label="field_of_view",
        )

    # --- derivation ---------------------------------------------------
    def with_array(
        self,
        array: NDArray[Any],
        *,
        is_label: bool | None = None,
        label_map: Mapping[Structure, int] | None = None,
        provenance: Provenance | None = None,
    ) -> Volume:
        """Return a copy carrying a new array but identical geometry."""
        a = np.asarray(array)
        if a.shape != self.array.shape:
            raise ValidationError(
                f"with_array expects the same shape {self.array.shape}, got {a.shape}. "
                "Use resample() or crop() to change geometry."
            )
        return replace(
            self,
            array=a,
            is_label=self.is_label if is_label is None else is_label,
            label_map=self.label_map if label_map is None else label_map,
            provenance=provenance or self.provenance,
        )

    def with_metadata(self, **extra: Any) -> Volume:
        merged = dict(self.metadata)
        merged.update(extra)
        return replace(self, metadata=merged)

    # --- label helpers ------------------------------------------------
    def label_of(self, structure: Structure) -> int:
        if not self.is_label or self.label_map is None:
            raise ValidationError("label_of() is only valid on a label volume")
        try:
            return int(self.label_map[structure])
        except KeyError as exc:
            available = sorted(s.value for s in self.label_map)
            raise ValidationError(
                f"structure {structure.value!r} is not in this mask's label map; "
                f"available: {available}"
            ) from exc

    def binary_mask(self, structure: Structure) -> NDArray[np.bool_]:
        return self.array == self.label_of(structure)

    def structures_present(self, *, min_voxels: int = 1) -> list[Structure]:
        """Structures with at least ``min_voxels`` voxels.

        An empty mask for a structure the pipeline expected is a failure mode
        worth surfacing explicitly rather than propagating as a zero-volume mesh.
        """
        if not self.is_label or self.label_map is None:
            raise ValidationError("structures_present() is only valid on a label volume")
        present: list[Structure] = []
        labels, counts = np.unique(self.array, return_counts=True)
        found = {int(v): int(c) for v, c in zip(labels, counts)}
        for structure, label in self.label_map.items():
            if structure is Structure.BACKGROUND:
                continue
            if found.get(int(label), 0) >= min_voxels:
                present.append(structure)
        return present

    def structure_volume_mm3(self, structure: Structure) -> float:
        return float(self.binary_mask(structure).sum()) * self.voxel_volume_mm3

    def extract_structure(self, structure: Structure) -> Volume:
        """A binary label volume containing only ``structure`` (label 1)."""
        mask = self.binary_mask(structure).astype(np.uint8)
        return replace(
            self,
            array=mask,
            is_label=True,
            label_map={Structure.BACKGROUND: 0, structure: 1},
        )

    # --- intensity ----------------------------------------------------
    def intensity_stats(self) -> dict[str, float]:
        a = self.array.astype(np.float64, copy=False)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            raise PreprocessingError("volume contains no finite voxels")
        return {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "p01": float(np.percentile(finite, 1)),
            "p99": float(np.percentile(finite, 99)),
        }

    def content_hash(self) -> str:
        """Hash of voxel data *and* geometry.

        Geometry is included because two volumes with identical voxels but
        different spacing are clinically different images.
        """
        from kneeai.core.provenance import hash_payload

        return hash_payload(
            {
                "array": hash_array(self.array),
                "spacing": self.spacing.tolist(),
                "origin": self.origin.tolist(),
                "direction": self.direction.tolist(),
                "modality": self.modality.value,
            }
        )

    # --- summary ------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "shape": list(self.shape),
            "spacing_mm": [round(float(s), 4) for s in self.spacing],
            "origin_mm": [round(float(o), 3) for o in self.origin],
            "orientation": self.orientation,
            "modality": self.modality.value,
            "side": self.side.value,
            "extent_mm": [round(float(e), 1) for e in self.physical_extent_mm],
            "is_label": self.is_label,
            "dtype": str(self.array.dtype),
        }
        if self.is_label and self.label_map is not None:
            info["structures"] = [s.value for s in self.structures_present()]
        return info

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "mask" if self.is_label else self.modality.value
        return (
            f"Volume({kind}, shape={self.shape}, "
            f"spacing={np.round(self.spacing, 3).tolist()}, "
            f"orient={self.orientation})"
        )

    # ------------------------------------------------------------------
    # Reorientation
    # ------------------------------------------------------------------
    def reorient(self, target: str = "LPS") -> Volume:
        """Permute and flip axes so the orientation code becomes ``target``.

        Every volume entering the AI pipeline is reoriented to a single canonical
        code (PRD §10). Models trained on one orientation silently degrade on
        another, and left/right flips are indistinguishable from a genuinely
        contralateral knee.
        """
        code = target.upper()
        if len(code) != 3 or set(code) - set(_LETTER_TO_AXIS):
            raise ValidationError(
                f"invalid orientation code {target!r}; letters must come from "
                "R/L, A/P, I/S"
            )
        wanted = [_LETTER_TO_AXIS[c] for c in code]
        if len({axis for axis, _ in wanted}) != 3:
            raise ValidationError(
                f"orientation code {target!r} does not name three distinct axes"
            )

        # Which image axis currently maps to which patient axis, and with what sign.
        current: dict[int, tuple[int, int]] = {}
        used: set[int] = set()
        for img_axis in range(3):
            v = self.direction[:, img_axis]
            order = np.argsort(-np.abs(v))
            pat_axis = next((int(a) for a in order if int(a) not in used), int(order[0]))
            used.add(pat_axis)
            current[pat_axis] = (img_axis, 1 if v[pat_axis] >= 0 else -1)

        perm: list[int] = []
        flips: list[bool] = []
        for pat_axis, want_sign in wanted:
            img_axis, have_sign = current[pat_axis]
            perm.append(img_axis)
            flips.append(have_sign != want_sign)

        if perm == [0, 1, 2] and not any(flips):
            return self

        arr = np.transpose(self.array, perm)
        flip_axes = tuple(i for i, f in enumerate(flips) if f)
        if flip_axes:
            arr = np.flip(arr, axis=flip_axes)
        arr = np.ascontiguousarray(arr)

        # The new voxel (0,0,0) sits at the far end of every flipped axis.
        old_index = np.zeros(3, dtype=np.float64)
        for new_axis, (old_axis, flipped) in enumerate(zip(perm, flips)):
            old_index[old_axis] = (self.shape[old_axis] - 1) if flipped else 0
        new_origin = self.voxel_to_image.apply(old_index)

        signs = np.array([-1.0 if f else 1.0 for f in flips])
        new_direction = self.direction[:, perm] * signs
        new_spacing = self.spacing[perm]

        out = replace(
            self,
            array=arr,
            spacing=new_spacing,
            origin=new_origin,
            direction=new_direction,
        )
        # Guard against a sign/permutation slip: the reoriented volume must
        # report exactly the code we asked for.
        if out.orientation != code:  # pragma: no cover - defensive
            raise PreprocessingError(
                f"reorientation produced {out.orientation!r}, expected {code!r}"
            )
        return out

    # ------------------------------------------------------------------
    # Resampling / cropping
    # ------------------------------------------------------------------
    def resample(
        self,
        spacing: Sequence[float] | float,
        *,
        order: int | None = None,
        mode: str = "constant",
        cval: float = 0.0,
    ) -> Volume:
        """Resample to isotropic or per-axis ``spacing`` in millimetres.

        Label volumes are resampled with nearest-neighbour regardless of the
        requested order: interpolating integer labels invents anatomy that never
        existed (e.g. a voxel of "cartilage" between femur and tibia).
        """
        from scipy import ndimage

        target = (
            np.full(3, float(spacing), dtype=np.float64)
            if np.isscalar(spacing)
            else np.asarray(spacing, dtype=np.float64).reshape(-1)
        )
        if target.size != 3 or np.any(target <= 0):
            raise ValidationError(f"target spacing must be 3 positive values, got {spacing}")

        if np.allclose(target, self.spacing, atol=1e-9):
            return self

        interp_order = 0 if self.is_label else (3 if order is None else order)
        if self.is_label and order not in (None, 0):
            interp_order = 0

        ratio = self.spacing / target
        new_shape = np.maximum(1, np.rint(np.asarray(self.shape) * ratio).astype(int))

        # Map output index -> input index. Using the shape ratio (rather than the
        # spacing ratio) keeps the physical extent consistent after rounding.
        scale = np.asarray(self.shape, dtype=np.float64) / new_shape.astype(np.float64)
        matrix = np.diag(scale)
        offset = 0.5 * (scale - 1.0)

        src = self.array.astype(np.float32 if not self.is_label else self.array.dtype)
        out = ndimage.affine_transform(
            src,
            matrix,
            offset=offset,
            output_shape=tuple(int(n) for n in new_shape),
            order=interp_order,
            mode=mode,
            cval=cval,
            prefilter=interp_order > 1,
        )
        if self.is_label:
            out = out.astype(self.array.dtype)

        actual_spacing = self.spacing * np.asarray(self.shape) / new_shape
        # Origin refers to a voxel *centre*, which moves by half the change in
        # voxel size along each axis.
        shift_index = 0.5 * (actual_spacing / self.spacing) - 0.5
        new_origin = self.voxel_to_image.apply(shift_index)

        return replace(self, array=out, spacing=actual_spacing, origin=new_origin)

    def crop(self, lower: Sequence[int], upper: Sequence[int]) -> Volume:
        """Crop to the half-open index box ``[lower, upper)``."""
        lo = np.asarray(lower, dtype=int)
        hi = np.asarray(upper, dtype=int)
        if lo.size != 3 or hi.size != 3:
            raise ValidationError("crop bounds must each have 3 components")
        lo = np.clip(lo, 0, np.asarray(self.shape))
        hi = np.clip(hi, 0, np.asarray(self.shape))
        if np.any(hi <= lo):
            raise ValidationError(f"empty crop region: lower={lo.tolist()} upper={hi.tolist()}")
        arr = self.array[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        return replace(
            self,
            array=np.ascontiguousarray(arr),
            origin=self.voxel_to_image.apply(lo.astype(np.float64)),
        )

    def crop_to_mask(
        self, mask: NDArray[np.bool_] | None = None, *, margin_mm: float = 10.0
    ) -> Volume:
        """Crop to the bounding box of ``mask`` (or of all non-zero voxels).

        Used for ROI detection before segmentation (PRD §12): a knee occupies a
        small fraction of a long-leg field of view, and cropping first is what
        makes patch-based 3D training tractable.
        """
        m = (self.array != 0) if mask is None else np.asarray(mask, dtype=bool)
        if m.shape != self.array.shape:
            raise ValidationError(
                f"mask shape {m.shape} does not match volume shape {self.array.shape}"
            )
        if not m.any():
            raise PreprocessingError("cannot crop to an empty mask")
        idx = np.array(np.nonzero(m))
        lo = idx.min(axis=1)
        hi = idx.max(axis=1) + 1
        margin_vox = np.ceil(float(margin_mm) / self.spacing).astype(int)
        return self.crop(lo - margin_vox, hi + margin_vox)

    def pad_to(self, shape: Sequence[int], *, cval: float = 0.0) -> Volume:
        """Symmetrically pad up to at least ``shape`` (needed for patch inference)."""
        target = np.asarray(shape, dtype=int)
        cur = np.asarray(self.shape, dtype=int)
        total = np.maximum(0, target - cur)
        if not total.any():
            return self
        before = total // 2
        after = total - before
        arr = np.pad(
            self.array,
            tuple((int(b), int(a)) for b, a in zip(before, after)),
            mode="constant",
            constant_values=cval,
        )
        new_origin = self.voxel_to_image.apply(-before.astype(np.float64))
        return replace(self, array=arr, origin=new_origin)

    # ------------------------------------------------------------------
    # SimpleITK interop — the single place axis order is converted
    # ------------------------------------------------------------------
    @classmethod
    def from_sitk(
        cls,
        image: Any,
        *,
        modality: Modality = Modality.UNKNOWN,
        side: Side = Side.UNKNOWN,
        is_label: bool = False,
        label_map: Mapping[Structure, int] | None = None,
        metadata: Mapping[str, Any] | None = None,
        provenance: Provenance | None = None,
    ) -> Volume:
        import SimpleITK as sitk  # noqa: N813  (imported lazily: optional extra)

        if image.GetDimension() != 3:
            raise ValidationError(
                f"expected a 3D image, got {image.GetDimension()}D. Split 4D "
                "acquisitions into per-echo volumes first."
            )
        arr = sitk.GetArrayFromImage(image)  # (k, j, i)
        arr = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))  # -> (i, j, k)
        return cls(
            array=arr,
            spacing=np.asarray(image.GetSpacing(), dtype=np.float64),
            origin=np.asarray(image.GetOrigin(), dtype=np.float64),
            direction=np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
            modality=modality,
            side=side,
            is_label=is_label,
            label_map=label_map,
            metadata=dict(metadata or {}),
            provenance=provenance,
        )

    def to_sitk(self) -> Any:
        import SimpleITK as sitk  # noqa: N813

        arr = np.ascontiguousarray(np.transpose(self.array, (2, 1, 0)))  # -> (k, j, i)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(tuple(float(s) for s in self.spacing))
        img.SetOrigin(tuple(float(o) for o in self.origin))
        img.SetDirection(tuple(float(v) for v in self.direction.reshape(-1)))
        return img

    # ------------------------------------------------------------------
    @classmethod
    def zeros_like(cls, other: Volume, *, dtype: Any = np.uint8, **overrides: Any) -> Volume:
        base = dict(
            array=np.zeros(other.shape, dtype=dtype),
            spacing=other.spacing,
            origin=other.origin,
            direction=other.direction,
            modality=other.modality,
            side=other.side,
        )
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

    @classmethod
    def from_labels(
        cls,
        array: NDArray[Any],
        label_map: Mapping[Structure, int],
        *,
        reference: Volume | None = None,
        **kw: Any,
    ) -> Volume:
        geom: dict[str, Any] = {}
        if reference is not None:
            geom = {
                "spacing": reference.spacing,
                "origin": reference.origin,
                "direction": reference.direction,
                "modality": reference.modality,
                "side": reference.side,
            }
        geom.update(kw)
        return cls(array=array, is_label=True, label_map=dict(label_map), **geom)


def stack_structure_volumes(volumes: Iterable[tuple[Structure, Volume]]) -> Volume:
    """Merge per-structure binary masks into one multi-label volume.

    Overlaps are a real occurrence when independent binary models are combined
    (cartilage predicted inside bone, for instance). Rather than silently
    letting the last writer win, overlapping voxels are counted and reported in
    the returned volume's metadata so the caller can decide.
    """
    items = list(volumes)
    if not items:
        raise ValidationError("no volumes to stack")
    ref = items[0][1]
    merged = np.zeros(ref.shape, dtype=np.uint8)
    label_map: dict[Structure, int] = {Structure.BACKGROUND: 0}
    overlaps: dict[str, int] = {}
    for idx, (structure, vol) in enumerate(items, start=1):
        if vol.shape != ref.shape:
            raise ValidationError(
                f"{structure.value} mask shape {vol.shape} != reference {ref.shape}"
            )
        if not np.allclose(vol.spacing, ref.spacing, atol=1e-6):
            raise ValidationError(f"{structure.value} mask spacing differs from reference")
        binary = vol.array != 0
        clash = int(np.count_nonzero(binary & (merged != 0)))
        if clash:
            overlaps[structure.value] = clash
        merged[binary] = idx
        label_map[structure] = idx
    out = Volume.from_labels(merged, label_map, reference=ref)
    return out.with_metadata(overlap_voxels=overlaps) if overlaps else out
