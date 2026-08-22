"""2D orthogonal slice rendering with segmentation overlay.

Renders axial, coronal and sagittal slices as PNG on the server rather than
shipping the NIfTI to the browser and decoding it there. Three reasons:

**Correctness.** Slice geometry depends on the affine. Getting anisotropic spacing
and axis order right once, next to the code that already understands
:class:`~kneeai.imaging.volume.Volume`, is safer than reimplementing it in
TypeScript against a hand-rolled NIfTI parser.

**Aspect ratio.** Native spacing here is 0.7 x 0.365 x 0.365 mm. A slice drawn on a
square pixel grid is stretched by roughly 2:1 in one direction, which silently
distorts every visual judgement a surgeon makes. Slices are resampled to isotropic
millimetres before encoding.

**Size.** A full volume is ~25 MB compressed; a rendered slice is ~40 KB.

Contours rather than filled masks are the default. A filled overlay hides the
underlying image, and the image is the evidence the segmentation is being judged
against.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from kneeai.core.errors import ValidationError
from kneeai.logging import get_logger

log = get_logger(__name__)

__all__ = ["PlaneName", "SliceGeometry", "slice_geometry", "render_slice", "STRUCTURE_RGB"]

PlaneName = Literal["axial", "coronal", "sagittal"]

#: Overlay colours for 2D.
#:
#: Deliberately more saturated than the 3D palette, and every structure distinct in
#: hue rather than only in lightness. Two reasons the 3D colours cannot be reused:
#:
#: - The 3D bone colours are near-white, which is right for a lit surface but
#:   invisible drawn over the bright cortical and fat signal in an MR slice.
#: - In 3D, femur and tibia are separated by position and shading. In a single 2D
#:   plate they touch, so if they differ only in lightness the boundary between them
#:   disappears.
#:
#: Hue families are kept where they carry meaning: cartilage stays in the blues,
#: meniscus in the ambers.
STRUCTURE_RGB: dict[int, tuple[int, int, int]] = {
    1: (240, 163, 94),    # femur — warm tan
    2: (56, 189, 248),    # femoral cartilage — cyan
    3: (125, 211, 160),   # tibia — green, unmistakably not the femur
    4: (99, 102, 241),    # medial tibial cartilage — indigo
    5: (168, 85, 247),    # lateral tibial cartilage — violet
    6: (244, 114, 182),   # patella — pink
    7: (34, 211, 238),    # patellar cartilage — light cyan
    8: (251, 191, 36),    # medial meniscus — amber
    9: (251, 146, 60),    # lateral meniscus — orange
    10: (148, 163, 184),  # fibula — slate
}

# The RAS axis each plane cuts along, after the volume is oriented to RAS.
_PLANE_AXIS: dict[str, int] = {"sagittal": 0, "coronal": 1, "axial": 2}


@dataclass(frozen=True)
class SliceGeometry:
    """How many slices each plane has, and where the anatomy actually is."""

    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    n_slices: dict[str, int]
    #: Slice index at the centroid of the labelled anatomy, per plane. A volume's
    #: midpoint is often empty; the centroid is where a reviewer wants to land.
    default_index: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "spacing_mm": [round(float(s), 4) for s in self.spacing_mm],
            "n_slices": dict(self.n_slices),
            "default_index": dict(self.default_index),
            "planes": list(_PLANE_AXIS),
        }


def _load_ras(path: Path) -> tuple[NDArray[Any], tuple[float, float, float]]:
    """Load a NIfTI and reorient to canonical RAS, returning array and spacing."""
    import nibabel as nib

    if not path.exists():
        raise ValidationError(f"volume not found: {path}")
    img = nib.as_closest_canonical(nib.load(str(path)))
    array = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()[:3]
    return array, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def slice_geometry(label_path: Path) -> SliceGeometry:
    labels, spacing = _load_ras(label_path)
    n_slices = {name: int(labels.shape[axis]) for name, axis in _PLANE_AXIS.items()}

    default: dict[str, int] = {}
    foreground = labels > 0
    if bool(foreground.any()):
        coords = np.argwhere(foreground)
        centroid = coords.mean(axis=0)
        for name, axis in _PLANE_AXIS.items():
            default[name] = int(round(float(centroid[axis])))
    else:
        for name, axis in _PLANE_AXIS.items():
            default[name] = labels.shape[axis] // 2

    return SliceGeometry(
        shape=(int(labels.shape[0]), int(labels.shape[1]), int(labels.shape[2])),
        spacing_mm=spacing,
        n_slices=n_slices,
        default_index=default,
    )


def _take(array: NDArray[Any], plane: str, index: int) -> NDArray[Any]:
    axis = _PLANE_AXIS[plane]
    index = int(np.clip(index, 0, array.shape[axis] - 1))
    plate = np.take(array, index, axis=axis)
    # Display convention: superior up, patient-left to screen-right.
    return np.rot90(plate)


def _in_plane_spacing(spacing: tuple[float, float, float], plane: str) -> tuple[float, float]:
    """Millimetres per pixel of the extracted plate, as (row, col) after rot90."""
    sx, sy, sz = spacing
    if plane == "sagittal":      # plate axes (y, z) -> rot90 -> (z, y)
        return sz, sy
    if plane == "coronal":       # plate axes (x, z) -> rot90 -> (z, x)
        return sz, sx
    return sy, sx                # axial: (x, y) -> rot90 -> (y, x)


def _window(image: NDArray[Any]) -> NDArray[np.uint8]:
    """Percentile window to 8-bit. MR has no absolute intensity scale, so a fixed
    window is meaningless and a min/max stretch is dominated by outliers."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(finite.min()), float(max(finite.max(), finite.min() + 1))
    scaled = (np.clip(image, lo, hi) - lo) / (hi - lo)
    return (scaled * 255).astype(np.uint8)


def _boundary(mask: NDArray[np.bool_], thickness: int = 2) -> NDArray[np.bool_]:
    """Boundary of a binary mask, `thickness` pixels wide.

    Two pixels rather than one: contours are upsampled with nearest-neighbour (any
    interpolation erases a thin line), and a single native pixel becomes a ragged
    staircase at 2-3x zoom. Two survives legibly.
    """
    from scipy.ndimage import binary_erosion

    if not bool(mask.any()):
        return mask
    return mask & ~binary_erosion(mask, iterations=max(1, thickness), border_value=0)


def render_slice(
    *,
    label_path: Path,
    plane: PlaneName,
    index: int,
    image_path: Path | None = None,
    mode: str = "contour",
    zoom: float = 2.0,
    max_side: int = 720,
) -> bytes:
    """Render one slice to PNG bytes.

    Args:
        label_path: segmentation NIfTI.
        image_path: source intensity NIfTI. When absent — the phantom has no
            intensity volume — labels are drawn filled on black, because contours
            over nothing convey nothing.
        mode: ``contour`` outlines each structure over the greyscale image;
            ``fill`` tints it; ``image`` draws the scan alone, for comparison.
        zoom: upsampling factor, applied after isotropic correction.
    """
    from PIL import Image

    if plane not in _PLANE_AXIS:
        raise ValidationError(f"unknown plane {plane!r}; expected one of {list(_PLANE_AXIS)}")

    labels, spacing = _load_ras(label_path)
    label_plate = _take(labels, plane, index).astype(np.int16)

    grey: NDArray[np.uint8] | None = None
    if image_path is not None and image_path.exists():
        image, image_spacing = _load_ras(image_path)
        if image.shape == labels.shape:
            grey = _window(_take(image, plane, index).astype(np.float32))
        else:
            log.warning(
                "image and label shapes differ; rendering labels only",
                fields={"image": str(image.shape), "label": str(labels.shape)},
            )

    height, width = label_plate.shape
    if grey is None:
        # No intensity data: fill so the anatomy is visible at all.
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        effective_mode = "fill" if mode != "image" else "fill"
    else:
        rgb = np.repeat(grey[:, :, None], 3, axis=2)
        effective_mode = mode

    if effective_mode != "image":
        for value in np.unique(label_plate):
            if value <= 0:
                continue
            colour = np.array(STRUCTURE_RGB.get(int(value), (148, 163, 184)), dtype=np.float32)
            mask = label_plate == value
            if effective_mode == "contour":
                rgb[_boundary(mask)] = colour.astype(np.uint8)
            elif grey is None:
                # Nothing underneath to preserve, so blending only mutes the colour
                # and makes two structures converge towards the same grey.
                rgb[mask] = colour.astype(np.uint8)
                # A darker outline separates structures that touch, which femur and
                # tibia do at the joint line.
                rgb[_boundary(mask, thickness=1)] = (colour * 0.55).astype(np.uint8)
            else:
                # Alpha blend so the underlying image stays readable.
                rgb[mask] = (0.45 * rgb[mask] + 0.55 * colour).astype(np.uint8)
                rgb[_boundary(mask, thickness=1)] = colour.astype(np.uint8)

    picture = Image.fromarray(rgb, mode="RGB")

    # Correct anisotropy: without this a 0.7 x 0.365 mm slice is stretched ~2:1.
    row_mm, col_mm = _in_plane_spacing(spacing, plane)
    target_w = max(1, int(round(width * col_mm * zoom)))
    target_h = max(1, int(round(height * row_mm * zoom)))
    longest = max(target_w, target_h)
    if longest > max_side:
        scale = max_side / longest
        target_w = max(1, int(target_w * scale))
        target_h = max(1, int(target_h * scale))
    # Nearest for contours: bilinear would blur a one-pixel outline into nothing.
    resample = Image.NEAREST if effective_mode == "contour" else Image.BILINEAR
    picture = picture.resize((target_w, target_h), resample)

    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
