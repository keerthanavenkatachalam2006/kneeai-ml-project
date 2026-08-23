"""Mask post-processing (PRD §12).

Raw network output is not a usable anatomical mask. Three failure modes recur and
are handled here:

1. **Satellite components.** A few stray voxels of "femur" 40 mm from the femur.
   Marching cubes turns each into its own closed surface, and the resulting mesh
   has no single well-defined bone.
2. **Interior holes.** Under-confident voxels inside a bone. These become
   internal cavities in the mesh and break watertightness.
3. **Anatomically impossible overlaps.** Cartilage predicted inside cortical
   bone, or medial and lateral menisci touching across the midline.

Every operation reports what it changed. A post-processing step that silently
removes 30% of a structure is indistinguishable from a good mask unless it says
so, and that difference matters when the surgeon is deciding whether to trust the
segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import (
    BONE_STRUCTURES,
    CARTILAGE_STRUCTURES,
    MENISCUS_STRUCTURES,
    Structure,
)
from kneeai.core.errors import SegmentationError
from kneeai.core.provenance import Provenance
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger

__all__ = [
    "PostProcessConfig",
    "PostProcessReport",
    "largest_component",
    "keep_largest_components",
    "fill_holes",
    "remove_small_components",
    "postprocess_mask",
]

log = get_logger("ml.postprocess")


@dataclass(frozen=True)
class PostProcessConfig:
    """What to clean up, and how aggressively.

    Bone and meniscus keep a single component; cartilage may legitimately be
    discontinuous once it is worn through, so its component count is not forced
    to one — collapsing a bipartite worn cartilage plate into its larger half
    would erase exactly the pathology being measured.
    """

    #: Structures reduced to their single largest connected component.
    single_component: tuple[Structure, ...] = BONE_STRUCTURES + MENISCUS_STRUCTURES

    #: Structures where small islands are dropped but multiple components survive.
    multi_component: tuple[Structure, ...] = CARTILAGE_STRUCTURES

    #: Islands below this physical volume are removed.
    min_component_mm3: float = 30.0

    #: Fill interior holes in these structures.
    fill_holes_for: tuple[Structure, ...] = BONE_STRUCTURES

    #: Warn if a step removes more than this fraction of a structure.
    large_removal_fraction: float = 0.05

    #: Resolve voxels assigned to two structures at once.
    resolve_overlaps: bool = True

    #: Structures that must be present, else the mask is rejected outright.
    required: tuple[Structure, ...] = (Structure.FEMUR, Structure.TIBIA)


@dataclass
class PostProcessReport:
    """Record of every change made, per structure."""

    actions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def record(self, structure: Structure, operation: str, **detail: Any) -> None:
        self.actions.append(
            {"structure": structure.value, "operation": operation, **detail}
        )

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        log.warning("post-processing concern", fields={"detail": message})

    def to_dict(self) -> dict[str, Any]:
        return {"actions": self.actions, "warnings": self.warnings}

    @property
    def changed(self) -> bool:
        return bool(self.actions)


# ---------------------------------------------------------------------------
# Component operations
# ---------------------------------------------------------------------------


def _label_components(
    mask: NDArray[np.bool_], connectivity: int = 1
) -> tuple[NDArray[np.int32], int]:
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labelled, n = ndimage.label(mask, structure=structure)
    return np.asarray(labelled, dtype=np.int32), int(n)


def largest_component(
    mask: NDArray[Any], *, connectivity: int = 1
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Keep only the largest connected component."""
    m = np.asarray(mask) != 0
    if not m.any():
        return m, {"n_components": 0, "kept_fraction": 0.0}

    labelled, n = _label_components(m, connectivity)
    if n <= 1:
        return m, {"n_components": n, "kept_fraction": 1.0}

    counts = np.bincount(labelled.reshape(-1))
    counts[0] = 0
    keep = int(np.argmax(counts))
    result = labelled == keep
    total = int(m.sum())
    return result, {
        "n_components": n,
        "kept_voxels": int(result.sum()),
        "removed_voxels": total - int(result.sum()),
        "kept_fraction": float(result.sum()) / float(total),
        "component_sizes": sorted(
            (int(c) for c in counts[1:] if c > 0), reverse=True
        )[:10],
    }


def remove_small_components(
    mask: NDArray[Any],
    *,
    min_voxels: int,
    connectivity: int = 1,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Drop connected components smaller than ``min_voxels``."""
    m = np.asarray(mask) != 0
    if not m.any() or min_voxels <= 1:
        return m, {"n_components": 0, "removed_components": 0}

    labelled, n = _label_components(m, connectivity)
    if n == 0:
        return m, {"n_components": 0, "removed_components": 0}

    counts = np.bincount(labelled.reshape(-1))
    counts[0] = 0
    small = np.where((counts > 0) & (counts < min_voxels))[0]
    if small.size == 0:
        return m, {"n_components": n, "removed_components": 0}

    result = m.copy()
    result[np.isin(labelled, small)] = False
    return result, {
        "n_components": n,
        "removed_components": int(small.size),
        "removed_voxels": int(counts[small].sum()),
        "kept_fraction": (
            float(result.sum()) / float(m.sum()) if m.sum() else 0.0
        ),
    }


def fill_holes(mask: NDArray[Any]) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Fill fully enclosed interior holes.

    Only cavities with no path to the volume boundary are filled, so a genuine
    open notch or intercondylar space is preserved.
    """
    from scipy import ndimage

    m = np.asarray(mask) != 0
    if not m.any():
        return m, {"filled_voxels": 0}
    filled = ndimage.binary_fill_holes(m)
    filled = np.asarray(filled, dtype=bool)
    return filled, {"filled_voxels": int(filled.sum() - m.sum())}


def keep_largest_components(
    mask: NDArray[Any], *, n_keep: int, connectivity: int = 1
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Keep the ``n_keep`` largest components."""
    m = np.asarray(mask) != 0
    if not m.any() or n_keep < 1:
        return m, {"n_components": 0}
    labelled, n = _label_components(m, connectivity)
    if n <= n_keep:
        return m, {"n_components": n, "kept": n}
    counts = np.bincount(labelled.reshape(-1))
    counts[0] = 0
    keep_labels = np.argsort(-counts)[:n_keep]
    result = np.isin(labelled, keep_labels)
    return result, {
        "n_components": n,
        "kept": n_keep,
        "kept_fraction": float(result.sum()) / float(m.sum()),
    }


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

#: Priority when two structures claim the same voxel. Bone wins over cartilage
#: because a cortical bone boundary is high-contrast and reliably segmented,
#: whereas the bone-cartilage interface is where cartilage models are least
#: certain. Ties inside a group are broken by the label map order.
_OVERLAP_PRIORITY: dict[Structure, int] = {
    Structure.FEMUR: 100,
    Structure.TIBIA: 100,
    Structure.PATELLA: 100,
    Structure.FIBULA: 95,
    Structure.FEMORAL_CARTILAGE: 60,
    Structure.MEDIAL_TIBIAL_CARTILAGE: 60,
    Structure.LATERAL_TIBIAL_CARTILAGE: 60,
    Structure.PATELLAR_CARTILAGE: 60,
    Structure.MEDIAL_MENISCUS: 50,
    Structure.LATERAL_MENISCUS: 50,
}


def _resolve_overlaps(
    binaries: dict[Structure, NDArray[np.bool_]], report: PostProcessReport
) -> dict[Structure, NDArray[np.bool_]]:
    ordered = sorted(binaries, key=lambda s: -_OVERLAP_PRIORITY.get(s, 0))
    claimed = np.zeros(next(iter(binaries.values())).shape, dtype=bool)
    out: dict[Structure, NDArray[np.bool_]] = {}
    for structure in ordered:
        mask = binaries[structure]
        conflict = np.logical_and(mask, claimed)
        n_conflict = int(conflict.sum())
        if n_conflict:
            mask = np.logical_and(mask, np.logical_not(claimed))
            report.record(
                structure,
                "resolve_overlap",
                voxels_yielded=n_conflict,
                reason="voxels already claimed by a higher-priority structure",
            )
        out[structure] = mask
        claimed = np.logical_or(claimed, mask)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def postprocess_mask(
    mask: Volume,
    config: PostProcessConfig = PostProcessConfig(),
) -> tuple[Volume, PostProcessReport]:
    """Clean a predicted multi-label mask and report every change.

    Raises:
        SegmentationError: if a required structure is missing or is destroyed by
            cleanup. Reconstructing a plan from a mask with no tibia is worse
            than failing the case.
    """
    if not mask.is_label or mask.label_map is None:
        raise SegmentationError("postprocess_mask() requires a label volume")

    report = PostProcessReport()
    voxel_mm3 = mask.voxel_volume_mm3
    min_voxels = max(1, int(round(config.min_component_mm3 / max(voxel_mm3, 1e-9))))

    structures = [s for s in mask.label_map if s is not Structure.BACKGROUND]
    binaries: dict[Structure, NDArray[np.bool_]] = {
        s: mask.binary_mask(s) for s in structures
    }

    for structure in structures:
        current = binaries[structure]
        original_voxels = int(current.sum())
        if original_voxels == 0:
            report.record(structure, "skip", reason="no voxels predicted")
            continue

        if structure in config.fill_holes_for:
            current, detail = fill_holes(current)
            if detail["filled_voxels"] > 0:
                report.record(structure, "fill_holes", **detail)

        if structure in config.single_component:
            current, detail = largest_component(current)
            if detail.get("n_components", 0) > 1:
                report.record(structure, "largest_component", **detail)
                removed_fraction = 1.0 - detail.get("kept_fraction", 1.0)
                if removed_fraction > config.large_removal_fraction:
                    report.warn(
                        f"{structure.display_name}: keeping the largest component "
                        f"discarded {removed_fraction * 100:.1f}% of predicted voxels "
                        f"across {detail['n_components']} components. The "
                        "segmentation is fragmented; review before relying on it."
                    )
        elif structure in config.multi_component:
            current, detail = remove_small_components(current, min_voxels=min_voxels)
            if detail.get("removed_components", 0) > 0:
                report.record(
                    structure,
                    "remove_small_components",
                    min_component_mm3=config.min_component_mm3,
                    **detail,
                )
        else:
            current, detail = remove_small_components(current, min_voxels=min_voxels)
            if detail.get("removed_components", 0) > 0:
                report.record(structure, "remove_small_components", **detail)

        if not current.any() and original_voxels > 0:
            report.warn(
                f"{structure.display_name}: post-processing removed every voxel "
                f"(started with {original_voxels}). All predicted components were "
                f"below {config.min_component_mm3} mm³."
            )
        binaries[structure] = current

    if config.resolve_overlaps and len(binaries) > 1:
        binaries = _resolve_overlaps(binaries, report)

    cleaned = np.zeros(mask.shape, dtype=mask.array.dtype)
    for structure, binary in binaries.items():
        cleaned[binary] = mask.label_of(structure)

    result = mask.with_array(
        cleaned,
        provenance=Provenance.for_component(
            "recon.surface",
            parameters={
                "operation": "postprocess_mask",
                "min_component_mm3": config.min_component_mm3,
                "min_component_voxels": min_voxels,
                "actions": report.actions,
            },
            notes="mask cleanup prior to surface reconstruction",
        ),
    )

    missing = [
        s.display_name
        for s in config.required
        if s in mask.label_map and not binaries.get(s, np.zeros(1, bool)).any()
    ]
    if missing:
        raise SegmentationError(
            f"required structure(s) absent after post-processing: {', '.join(missing)}. "
            "The case cannot proceed to 3D reconstruction or planning."
        )

    log.info(
        "mask post-processed",
        fields={
            "actions": len(report.actions),
            "warnings": len(report.warnings),
            "structures": len(binaries),
        },
    )
    return result, report


def component_summary(mask: Volume, structures: Iterable[Structure] | None = None) -> dict[str, Any]:
    """Connected-component census, useful for QC dashboards."""
    if not mask.is_label or mask.label_map is None:
        raise SegmentationError("component_summary() requires a label volume")
    targets: Sequence[Structure] = (
        list(structures)
        if structures is not None
        else [s for s in mask.label_map if s is not Structure.BACKGROUND]
    )
    out: dict[str, Any] = {}
    for structure in targets:
        binary = mask.binary_mask(structure)
        if not binary.any():
            out[structure.value] = {"n_components": 0, "volume_mm3": 0.0}
            continue
        labelled, n = _label_components(binary)
        counts = np.bincount(labelled.reshape(-1))
        counts[0] = 0
        sizes = sorted((int(c) for c in counts if c > 0), reverse=True)
        out[structure.value] = {
            "n_components": n,
            "volume_mm3": round(float(binary.sum()) * mask.voxel_volume_mm3, 1),
            "largest_component_fraction": round(sizes[0] / float(binary.sum()), 4),
            "component_voxels": sizes[:10],
        }
    return out
