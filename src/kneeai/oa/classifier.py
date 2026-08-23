"""A small, honest osteoarthritis classifier.

Regularised logistic regression on seven geometric features, fitted with
full-batch gradient descent in NumPy. Two deliberate choices:

**Logistic regression, not a network.** The training set is 48 knees. A model with
more capacity than this would fit the noise and report a validation number that
means nothing. Seven features on 48 samples is already generous; L2 is not
optional at that ratio.

**Trained on labels that exist.** The target is the dataset's own
Kellgren-Lawrence grade, dichotomised at KL >= 2 — the standard radiographic
definition of definite osteoarthritis. No new annotation was invented.

The thing this model must never be confused with is a radiologist. It reads
cartilage geometry from an MR segmentation; KL is graded from a radiograph on
osteophytes and joint space narrowing. They correlate, which is why this works at
all, but they are not the same measurement, and the grade this predicts is a
*proxy* trained against a proxy. :meth:`OaClassifier.assess` carries that caveat
into every payload it produces rather than leaving it to the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kneeai.oa.features import FEATURE_NAMES

__all__ = [
    "MODEL_FILENAME",
    "BinaryMetrics",
    "LogisticModel",
    "OaClassifier",
    "fit_logistic",
    "evaluate_binary",
]

MODEL_FILENAME = "oa_classifier.json"

#: KL >= 2 is "definite osteoarthritis" and is the conventional dichotomy for
#: radiographic OA prevalence. Recorded here so the threshold is stated once.
OA_KL_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class LogisticModel:
    """Standardise, then apply a linear model through a logistic link."""

    coefficients: list[float]
    intercept: float
    feature_names: list[str]
    mean: list[float]
    scale: list[float]

    #: Training-set medians, used to fill features that could not be computed.
    impute: list[float]

    def design(self, x: np.ndarray) -> np.ndarray:
        """Impute missing features, then standardise.

        Public because the assessment layer needs the standardised values to
        decompose a prediction into per-feature contributions, and recomputing the
        standardisation there would risk the two drifting apart.
        """
        x = np.atleast_2d(np.asarray(x, dtype=np.float64)).copy()
        impute = np.asarray(self.impute, dtype=np.float64)
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.take(impute, np.where(bad)[1])
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        return (x - mean) / scale

    # Keep backward compat alias
    _design = design

    def decision(self, x: np.ndarray) -> np.ndarray:
        z = self.design(x) @ np.asarray(self.coefficients, dtype=np.float64)
        return z + self.intercept

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.decision(x)
        # Numerically stable logistic.
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": [float(c) for c in self.coefficients],
            "intercept": float(self.intercept),
            "feature_names": list(self.feature_names),
            "mean": [float(v) for v in self.mean],
            "scale": [float(v) for v in self.scale],
            "impute": [float(v) for v in self.impute],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LogisticModel:
        return cls(
            coefficients=[float(c) for c in payload["coefficients"]],
            intercept=float(payload["intercept"]),
            feature_names=list(payload["feature_names"]),
            mean=[float(v) for v in payload["mean"]],
            scale=[float(v) for v in payload["scale"]],
            impute=[float(v) for v in payload["impute"]],
        )


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
    l2: float = 1.0,
    iterations: int = 20_000,
    learning_rate: float = 0.05,
    class_weight: bool = True,
) -> LogisticModel:
    """Fit L2-regularised logistic regression by gradient descent.

    Args:
        class_weight: reweight the two classes to equal total mass. The cohort is
            not balanced, and without this the model buys accuracy by predicting
            the majority class, which is exactly the failure that looks fine in a
            single accuracy number.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64)).copy()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x has {x.shape[0]} rows but y has {y.shape[0]}")

    impute = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    impute = np.where(np.isfinite(impute), impute, 0.0)
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(impute, np.where(bad)[1])

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale

    n, d = z.shape
    weights = np.zeros(d, dtype=np.float64)
    bias = 0.0

    if class_weight:
        n_pos = float(max(y.sum(), 1.0))
        n_neg = float(max(n - y.sum(), 1.0))
        sample_w = np.where(y > 0.5, n / (2.0 * n_pos), n / (2.0 * n_neg))
    else:
        sample_w = np.ones(n, dtype=np.float64)
    sample_w = sample_w / sample_w.sum()

    for _ in range(iterations):
        logit = z @ weights + bias
        pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
        residual = (pred - y) * sample_w
        grad_w = z.T @ residual + l2 * weights / n
        grad_b = float(residual.sum())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return LogisticModel(
        coefficients=[float(w) for w in weights],
        intercept=float(bias),
        feature_names=list(feature_names),
        mean=[float(v) for v in mean],
        scale=[float(v) for v in scale],
        impute=[float(v) for v in impute],
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class BinaryMetrics:
    """Held-out performance. Every field is measured, none is asserted."""

    n: int
    n_positive: int
    threshold: float
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    balanced_accuracy: float
    auroc: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n,
            "n_positive": self.n_positive,
            "threshold": round(self.threshold, 4),
            "accuracy": round(self.accuracy, 4),
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "precision": round(self.precision, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "auroc": round(self.auroc, 4),
            "confusion": {
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "true_negative": self.true_negative,
                "false_negative": self.false_negative,
            },
        }


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    """AUROC via the rank identity, with ties averaged."""
    pos = score[y > 0.5]
    neg = score[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    # Average ranks within tied groups.
    values = np.concatenate([pos, neg])
    for value in np.unique(values):
        tie = values == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    rank_sum = ranks[: pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def evaluate_binary(y: np.ndarray, proba: np.ndarray, *, threshold: float = 0.5) -> BinaryMetrics:
    y = np.asarray(y, dtype=np.float64).ravel()
    proba = np.asarray(proba, dtype=np.float64).ravel()
    pred = proba >= threshold
    truth = y > 0.5

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth))
    fn = int(np.sum(~pred & truth))

    def safe(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    sensitivity = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    balanced = (
        float(np.nanmean([sensitivity, specificity]))
        if np.isfinite(sensitivity) or np.isfinite(specificity)
        else float("nan")
    )
    return BinaryMetrics(
        n=int(y.size),
        n_positive=int(truth.sum()),
        threshold=threshold,
        accuracy=safe(tp + tn, int(y.size)),
        sensitivity=sensitivity,
        specificity=specificity,
        precision=safe(tp, tp + fp),
        balanced_accuracy=balanced,
        auroc=_auroc(y, proba),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


# ---------------------------------------------------------------------------
# The packaged classifier
# ---------------------------------------------------------------------------

#: Severity bands over the predicted KL grade. Named to match the
#: Kellgren-Lawrence vocabulary because that is what the model was trained
#: against, with the grade itself always reported alongside.
SEVERITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.5, "none"),
    (1.5, "doubtful"),
    (2.5, "mild"),
    (3.5, "moderate"),
    (float("inf"), "severe"),
)


def severity_for(grade: float) -> str:
    for upper, label in SEVERITY_BANDS:
        if grade < upper:
            return label
    return "severe"


@dataclass
class OaClassifier:
    """The binary OA model, a KL-grade regressor, and their measured metrics."""

    binary: LogisticModel

    #: Least-squares linear predictor of KL grade, used only for the severity
    #: band. Deliberately separate from the binary model: the yes/no answer is
    #: the one with measured operating characteristics, and the grade is
    #: supporting detail.
    grade_coefficients: list[float] = field(default_factory=list)
    grade_intercept: float = 0.0

    version: str = "0.1.0"
    threshold: float = 0.5
    metrics: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    # --- persistence ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "threshold": self.threshold,
            "binary": self.binary.to_dict(),
            "grade_coefficients": [float(c) for c in self.grade_coefficients],
            "grade_intercept": float(self.grade_intercept),
            "metrics": self.metrics,
            "training": self.training,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OaClassifier:
        return cls(
            binary=LogisticModel.from_dict(payload["binary"]),
            grade_coefficients=[float(c) for c in payload.get("grade_coefficients", [])],
            grade_intercept=float(payload.get("grade_intercept", 0.0)),
            version=str(payload.get("version", "0.1.0")),
            threshold=float(payload.get("threshold", 0.5)),
            metrics=dict(payload.get("metrics", {})),
            training=dict(payload.get("training", {})),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return target

    @classmethod
    def load(cls, path: str | Path) -> OaClassifier:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # --- prediction ---
    def probability(self, vector: np.ndarray) -> float:
        return float(self.binary.predict_proba(np.atleast_2d(vector))[0])

    def grade(self, vector: np.ndarray) -> float:
        if not self.grade_coefficients:
            return float("nan")
        design = self.binary.design(np.atleast_2d(vector))
        # design is (1, n_features), so the product is (1,) rather than a scalar.
        raw = (design @ np.asarray(self.grade_coefficients, dtype=np.float64))[0]
        return float(np.clip(raw + self.grade_intercept, 0.0, 4.0))
