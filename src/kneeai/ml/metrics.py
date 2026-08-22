"""Segmentation evaluation metrics (PRD §13).

Implemented with numpy/scipy only, so evaluation runs anywhere — including in
CI without a GPU or a torch install.

PRD §114 (Rule 2) is the reason this module reports surface metrics alongside
overlap metrics: a mask can reach an excellent Dice score while its boundary is
displaced enough to move a clinical measurement by millimetres. Dice is
dominated by a structure's interior, and the interior is not what an implant
sits on. :func:`evaluate_masks` therefore always returns Hausdorff-95 and
average surface distance next to Dice, and :func:`assess_clinical_impact`
compares the measurements themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from kneeai.core.anatomy import Structure
from kneeai.core.errors import ValidationError
from kneeai.imaging.volume import Volume

__all__ = [
    "StructureMetrics",
    "SegmentationEvaluation",
    "dice",
    "iou",
    "sensitivity",
    "precision",
    "volume_similarity",
    "surface_distances",
    "evaluate_binary",
    "evaluate_masks",
    "aggregate_evaluations",
    "assess_clinical_impact",
]


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------


def _as_bool(mask: NDArray[Any], name: str) -> NDArray[np.bool_]:
    arr = np.asarray(mask)
    if arr.dtype != bool:
        arr = arr != 0
    return arr


def _check_shapes(pred: NDArray[Any], gt: NDArray[Any]) -> None:
    if pred.shape != gt.shape:
        raise ValidationError(
            f"prediction shape {pred.shape} != ground-truth shape {gt.shape}. "
            "Resample one onto the other's grid with imaging.preprocess.resample_like "
            "before evaluating."
        )


def dice(pred: NDArray[Any], gt: NDArray[Any]) -> float:
    """Dice similarity coefficient.

    Returns 1.0 when both masks are empty: a model correctly predicting the
    absence of a structure should not be scored as a total failure. Returns 0.0
    when exactly one is empty.
    """
    p, g = _as_bool(pred, "pred"), _as_bool(gt, "gt")
    _check_shapes(p, g)
    p_sum, g_sum = int(p.sum()), int(g.sum())
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0
    return 2.0 * float(np.logical_and(p, g).sum()) / float(p_sum + g_sum)


def iou(pred: NDArray[Any], gt: NDArray[Any]) -> float:
    p, g = _as_bool(pred, "pred"), _as_bool(gt, "gt")
    _check_shapes(p, g)
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum()) / float(union)


def sensitivity(pred: NDArray[Any], gt: NDArray[Any]) -> float:
    """Recall: fraction of the true structure that was found."""
    p, g = _as_bool(pred, "pred"), _as_bool(gt, "gt")
    _check_shapes(p, g)
    denom = int(g.sum())
    if denom == 0:
        return float("nan")
    return float(np.logical_and(p, g).sum()) / float(denom)


def precision(pred: NDArray[Any], gt: NDArray[Any]) -> float:
    p, g = _as_bool(pred, "pred"), _as_bool(gt, "gt")
    _check_shapes(p, g)
    denom = int(p.sum())
    if denom == 0:
        return float("nan")
    return float(np.logical_and(p, g).sum()) / float(denom)


def volume_similarity(pred: NDArray[Any], gt: NDArray[Any]) -> float:
    """1 - |Vp - Vg| / (Vp + Vg).

    Reported separately from Dice because a systematic volume bias (a model that
    consistently under-segments cartilage by 15%) can coexist with a respectable
    Dice, and volume bias directly biases every derived volume measurement.
    """
    p, g = _as_bool(pred, "pred"), _as_bool(gt, "gt")
    _check_shapes(p, g)
    total = int(p.sum()) + int(g.sum())
    if total == 0:
        return 1.0
    return 1.0 - abs(int(p.sum()) - int(g.sum())) / float(total)


# ---------------------------------------------------------------------------
# Surface metrics
# ---------------------------------------------------------------------------


def _surface_voxels(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Boundary voxels: mask voxels with at least one background 6-neighbour."""
    from scipy import ndimage

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def surface_distances(
    pred: NDArray[Any],
    gt: NDArray[Any],
    spacing_mm: Sequence[float],
) -> dict[str, float]:
    """Symmetric surface distance statistics in millimetres.

    Returns Hausdorff, Hausdorff-95, average symmetric surface distance (ASSD),
    the two directed mean distances, and the median.

    ``spacing_mm`` must be the *physical* voxel spacing; passing voxel units
    silently reports distances in voxels, which is the kind of unit error that
    makes a 3 mm boundary error look like 3 "units" of nothing in particular.
    """
    from scipy import ndimage

    p, g = _as_bool(pred, "pred"), _as_bool(gt, "gt")
    _check_shapes(p, g)
    sp = np.asarray(spacing_mm, dtype=np.float64).reshape(-1)
    if sp.size != p.ndim or np.any(sp <= 0):
        raise ValidationError(
            f"spacing_mm must have {p.ndim} positive components, got {spacing_mm}"
        )

    nan_result = {
        "hausdorff_mm": float("nan"),
        "hausdorff95_mm": float("nan"),
        "assd_mm": float("nan"),
        "mean_pred_to_gt_mm": float("nan"),
        "mean_gt_to_pred_mm": float("nan"),
        "median_mm": float("nan"),
        "n_surface_pred": 0,
        "n_surface_gt": 0,
    }
    if not p.any() or not g.any():
        return nan_result

    surf_p = _surface_voxels(p)
    surf_g = _surface_voxels(g)
    if not surf_p.any() or not surf_g.any():  # pragma: no cover - degenerate
        return nan_result

    # Distance from every voxel to the nearest ground-truth surface voxel, and
    # vice versa. Sampling the EDT with the physical spacing makes the result
    # anisotropy-correct.
    dt_to_gt = ndimage.distance_transform_edt(np.logical_not(surf_g), sampling=sp)
    dt_to_pred = ndimage.distance_transform_edt(np.logical_not(surf_p), sampling=sp)

    d_pg = np.asarray(dt_to_gt)[surf_p]
    d_gp = np.asarray(dt_to_pred)[surf_g]
    both = np.concatenate([d_pg, d_gp])

    return {
        "hausdorff_mm": float(max(d_pg.max(), d_gp.max())),
        "hausdorff95_mm": float(
            max(np.percentile(d_pg, 95.0), np.percentile(d_gp, 95.0))
        ),
        "assd_mm": float(both.sum() / both.size),
        "mean_pred_to_gt_mm": float(d_pg.mean()),
        "mean_gt_to_pred_mm": float(d_gp.mean()),
        "median_mm": float(np.median(both)),
        "n_surface_pred": int(surf_p.sum()),
        "n_surface_gt": int(surf_g.sum()),
    }


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureMetrics:
    """Metrics for one structure in one case."""

    structure: Structure
    dice: float
    iou: float
    sensitivity: float
    precision: float
    volume_similarity: float
    hausdorff_mm: float
    hausdorff95_mm: float
    assd_mm: float
    mean_pred_to_gt_mm: float
    mean_gt_to_pred_mm: float
    pred_volume_mm3: float
    gt_volume_mm3: float
    present_in_gt: bool
    present_in_pred: bool

    @property
    def volume_error_mm3(self) -> float:
        return self.pred_volume_mm3 - self.gt_volume_mm3

    @property
    def volume_error_pct(self) -> float:
        if self.gt_volume_mm3 <= 0:
            return float("nan")
        return 100.0 * self.volume_error_mm3 / self.gt_volume_mm3

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure.value,
            "dice": round(self.dice, 4),
            "iou": round(self.iou, 4),
            "sensitivity": round(self.sensitivity, 4),
            "precision": round(self.precision, 4),
            "volume_similarity": round(self.volume_similarity, 4),
            "hausdorff_mm": round(self.hausdorff_mm, 3),
            "hausdorff95_mm": round(self.hausdorff95_mm, 3),
            "assd_mm": round(self.assd_mm, 3),
            "pred_volume_mm3": round(self.pred_volume_mm3, 1),
            "gt_volume_mm3": round(self.gt_volume_mm3, 1),
            "volume_error_pct": round(self.volume_error_pct, 2),
            "present_in_gt": self.present_in_gt,
            "present_in_pred": self.present_in_pred,
        }


@dataclass
class SegmentationEvaluation:
    """Per-structure metrics for one case, plus case-level context."""

    case_id: str
    per_structure: dict[Structure, StructureMetrics] = field(default_factory=dict)
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    model_id: str | None = None
    model_version: str | None = None
    notes: list[str] = field(default_factory=list)

    def mean_dice(self, structures: Iterable[Structure] | None = None) -> float:
        selected = self._select(structures)
        values = [m.dice for m in selected if m.present_in_gt]
        return float(np.mean(values)) if values else float("nan")

    def mean_hausdorff95(self, structures: Iterable[Structure] | None = None) -> float:
        selected = self._select(structures)
        values = [m.hausdorff95_mm for m in selected if np.isfinite(m.hausdorff95_mm)]
        return float(np.mean(values)) if values else float("nan")

    def worst_structure(self) -> StructureMetrics | None:
        candidates = [m for m in self.per_structure.values() if m.present_in_gt]
        return min(candidates, key=lambda m: m.dice) if candidates else None

    def _select(self, structures: Iterable[Structure] | None) -> list[StructureMetrics]:
        if structures is None:
            return list(self.per_structure.values())
        wanted = set(structures)
        return [m for s, m in self.per_structure.items() if s in wanted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "spacing_mm": list(self.spacing_mm),
            "mean_dice": round(self.mean_dice(), 4),
            "mean_hausdorff95_mm": round(self.mean_hausdorff95(), 3),
            "structures": [m.to_dict() for m in self.per_structure.values()],
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Evaluation drivers
# ---------------------------------------------------------------------------


def evaluate_binary(
    pred: NDArray[Any],
    gt: NDArray[Any],
    spacing_mm: Sequence[float],
    *,
    structure: Structure = Structure.FEMUR,
) -> StructureMetrics:
    """Full metric set for a single binary mask pair."""
    p, g = _as_bool(pred, "pred"), _as_bool(gt, "gt")
    _check_shapes(p, g)
    voxel_mm3 = float(np.prod(np.asarray(spacing_mm, dtype=np.float64)))
    sd = surface_distances(p, g, spacing_mm)
    return StructureMetrics(
        structure=structure,
        dice=dice(p, g),
        iou=iou(p, g),
        sensitivity=sensitivity(p, g),
        precision=precision(p, g),
        volume_similarity=volume_similarity(p, g),
        hausdorff_mm=sd["hausdorff_mm"],
        hausdorff95_mm=sd["hausdorff95_mm"],
        assd_mm=sd["assd_mm"],
        mean_pred_to_gt_mm=sd["mean_pred_to_gt_mm"],
        mean_gt_to_pred_mm=sd["mean_gt_to_pred_mm"],
        pred_volume_mm3=float(p.sum()) * voxel_mm3,
        gt_volume_mm3=float(g.sum()) * voxel_mm3,
        present_in_gt=bool(g.any()),
        present_in_pred=bool(p.any()),
    )


def evaluate_masks(
    prediction: Volume,
    ground_truth: Volume,
    *,
    case_id: str = "unknown",
    structures: Iterable[Structure] | None = None,
    model_id: str | None = None,
    model_version: str | None = None,
) -> SegmentationEvaluation:
    """Evaluate a predicted label volume against ground truth.

    Both volumes must already share a grid. Rather than resampling silently —
    which would let a geometry mismatch pass unnoticed and quietly change the
    metrics — a mismatch raises.
    """
    if not prediction.is_label or not ground_truth.is_label:
        raise ValidationError("evaluate_masks() requires two label volumes")
    if prediction.shape != ground_truth.shape:
        raise ValidationError(
            f"prediction shape {prediction.shape} != ground truth {ground_truth.shape}; "
            "resample explicitly with imaging.preprocess.resample_like"
        )
    if not np.allclose(prediction.spacing, ground_truth.spacing, atol=1e-4):
        raise ValidationError(
            f"spacing mismatch: prediction {prediction.spacing.tolist()} vs "
            f"ground truth {ground_truth.spacing.tolist()}"
        )
    if prediction.label_map is None or ground_truth.label_map is None:
        raise ValidationError("both volumes must carry a label map")

    if structures is None:
        candidates = sorted(
            (set(prediction.label_map) | set(ground_truth.label_map))
            - {Structure.BACKGROUND},
            key=lambda s: s.value,
        )
    else:
        candidates = sorted(set(structures), key=lambda s: s.value)

    evaluation = SegmentationEvaluation(
        case_id=case_id,
        spacing_mm=tuple(float(s) for s in ground_truth.spacing),  # type: ignore[arg-type]
        model_id=model_id,
        model_version=model_version,
    )

    for structure in candidates:
        in_pred = structure in prediction.label_map
        in_gt = structure in ground_truth.label_map
        if not in_gt:
            evaluation.notes.append(
                f"{structure.value}: predicted but absent from the ground-truth "
                "label map; not scored"
            )
            continue
        if not in_pred:
            evaluation.notes.append(
                f"{structure.value}: present in ground truth but the model does "
                "not predict it; scored as a complete miss"
            )
            pred_mask = np.zeros(prediction.shape, dtype=bool)
        else:
            pred_mask = prediction.binary_mask(structure)
        gt_mask = ground_truth.binary_mask(structure)
        evaluation.per_structure[structure] = evaluate_binary(
            pred_mask, gt_mask, ground_truth.spacing, structure=structure
        )

    return evaluation


def aggregate_evaluations(
    evaluations: Sequence[SegmentationEvaluation],
) -> dict[str, Any]:
    """Cohort-level summary.

    Reports median and IQR alongside the mean. A cohort mean Dice hides the
    handful of catastrophic failures that matter most clinically, so the worst
    cases are listed explicitly.
    """
    if not evaluations:
        raise ValidationError("no evaluations to aggregate")

    by_structure: dict[Structure, list[StructureMetrics]] = {}
    for ev in evaluations:
        for structure, metrics in ev.per_structure.items():
            if metrics.present_in_gt:
                by_structure.setdefault(structure, []).append(metrics)

    summary: dict[str, Any] = {
        "n_cases": len(evaluations),
        "structures": {},
    }
    for structure, items in sorted(by_structure.items(), key=lambda kv: kv[0].value):
        dices = np.asarray([m.dice for m in items], dtype=np.float64)
        hd95 = np.asarray(
            [m.hausdorff95_mm for m in items if np.isfinite(m.hausdorff95_mm)],
            dtype=np.float64,
        )
        assd = np.asarray(
            [m.assd_mm for m in items if np.isfinite(m.assd_mm)], dtype=np.float64
        )
        vol_err = np.asarray(
            [m.volume_error_pct for m in items if np.isfinite(m.volume_error_pct)],
            dtype=np.float64,
        )
        summary["structures"][structure.value] = {
            "n": len(items),
            "dice_mean": round(float(dices.mean()), 4),
            "dice_std": round(float(dices.std()), 4),
            "dice_median": round(float(np.median(dices)), 4),
            "dice_p25": round(float(np.percentile(dices, 25)), 4),
            "dice_p05": round(float(np.percentile(dices, 5)), 4),
            "dice_min": round(float(dices.min()), 4),
            "hausdorff95_mean_mm": round(float(hd95.mean()), 3) if hd95.size else None,
            "hausdorff95_p95_mm": (
                round(float(np.percentile(hd95, 95)), 3) if hd95.size else None
            ),
            "assd_mean_mm": round(float(assd.mean()), 3) if assd.size else None,
            "volume_bias_pct": round(float(vol_err.mean()), 2) if vol_err.size else None,
        }

    # Surface the worst cases: these are the ones a reviewer must actually look at.
    ranked = sorted(evaluations, key=lambda e: e.mean_dice())
    summary["worst_cases"] = [
        {
            "case_id": ev.case_id,
            "mean_dice": round(ev.mean_dice(), 4),
            "worst_structure": (
                ev.worst_structure().structure.value if ev.worst_structure() else None
            ),
            "worst_structure_dice": (
                round(ev.worst_structure().dice, 4) if ev.worst_structure() else None
            ),
        }
        for ev in ranked[: min(10, len(ranked))]
    ]
    return summary


# ---------------------------------------------------------------------------
# Clinical impact (PRD §13, §86, §114)
# ---------------------------------------------------------------------------


@dataclass
class ClinicalImpact:
    """Per-measurement agreement between two measurement sets."""

    measurement: str
    unit: str
    n: int
    mae: float
    rmse: float
    bias: float
    p95_abs_error: float
    max_abs_error: float
    tolerance: float | None = None

    @property
    def within_tolerance(self) -> bool | None:
        """Whether the 95th-percentile error meets the stated tolerance.

        Judged on the 95th percentile rather than the mean: a measurement engine
        whose average error is fine but which is badly wrong on 1 in 20 cases is
        not clinically acceptable.
        """
        if self.tolerance is None:
            return None
        return self.p95_abs_error <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement": self.measurement,
            "unit": self.unit,
            "n": self.n,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "bias": round(self.bias, 4),
            "p95_abs_error": round(self.p95_abs_error, 4),
            "max_abs_error": round(self.max_abs_error, 4),
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


def assess_clinical_impact(
    predicted: Sequence[Mapping[str, float]],
    reference: Sequence[Mapping[str, float]],
    *,
    units: Mapping[str, str] | None = None,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, ClinicalImpact]:
    """Compare paired measurement sets derived from predicted vs reference masks.

    This is the test PRD §13 asks for — "does segmentation error materially
    affect clinical measurements?" — expressed directly. Each entry in the input
    sequences is one case's measurement dictionary; keys are measurement names.
    """
    if len(predicted) != len(reference):
        raise ValidationError(
            f"paired comparison needs equal lengths, got {len(predicted)} predicted "
            f"and {len(reference)} reference measurement sets"
        )
    if not predicted:
        raise ValidationError("no measurement pairs supplied")

    units = units or {}
    tolerances = tolerances or {}
    keys = sorted({k for m in reference for k in m})
    out: dict[str, ClinicalImpact] = {}

    for key in keys:
        pairs = [
            (float(p[key]), float(r[key]))
            for p, r in zip(predicted, reference)
            if key in p and key in r and np.isfinite(p[key]) and np.isfinite(r[key])
        ]
        if not pairs:
            continue
        arr = np.asarray(pairs, dtype=np.float64)
        errors = arr[:, 0] - arr[:, 1]
        abs_errors = np.abs(errors)
        out[key] = ClinicalImpact(
            measurement=key,
            unit=units.get(key, ""),
            n=len(pairs),
            mae=float(abs_errors.mean()),
            rmse=float(np.sqrt((errors**2).mean())),
            bias=float(errors.mean()),
            p95_abs_error=float(np.percentile(abs_errors, 95)),
            max_abs_error=float(abs_errors.max()),
            tolerance=tolerances.get(key),
        )
    return out
