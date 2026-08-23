"""Turn a segmentation into an osteoarthritis assessment payload.

This is the module the upload flow calls to answer the first question a user asks:
*is this osteoarthritis or not?*

It answers with four things, and the fourth is the one that keeps the other three
honest:

1. a yes/no, from the trained classifier, at its calibrated threshold;
2. a probability, and a severity band on the KL scale it was trained against;
3. the geometric indicators behind the answer, each with its own value, so the
   answer can be checked rather than believed;
4. the classifier's *measured* held-out performance, attached to every single
   assessment.

Point 4 exists because this model is fitted to 48 knees. Quoting a probability of
0.87 without also saying the validation set was 12 cases invites a reader to trust
a number that has not earned it. The metrics travel with the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kneeai.core.anatomy import Structure
from kneeai.imaging.volume import Volume
from kneeai.logging import get_logger
from kneeai.oa.classifier import OA_KL_THRESHOLD, MODEL_FILENAME, OaClassifier, severity_for
from kneeai.oa.features import FEATURE_NAMES, OaFeatures, extract_features
from kneeai.oa.morphometric_classifier import (
    MorphometricOaClassifier,
    MorphometricOaPrediction,
    predict_oa_from_measurements,
)

__all__ = [
    "OA_DISCLAIMER",
    "OaAssessment",
    "assess_oa",
    "assess_oa_from_measurements",
    "classifier_path",
    "load_classifier",
]

log = get_logger("oa.assess")

OA_DISCLAIMER = (
    "Not a diagnosis. This is a research prototype: a geometric classifier fitted "
    "to 48 knees and validated on 12, predicting a radiographic Kellgren-Lawrence "
    "grade from MR cartilage geometry. It is not a radiologist's read, it has no "
    "regulatory clearance, and it must not inform the care of any patient."
)

#: Human-readable descriptions of each feature, for the interface.
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "femoral_cartilage_thickness_mm": ("Femoral cartilage thickness", "mm"),
    "medial_tibial_cartilage_thickness_mm": ("Medial tibial cartilage thickness", "mm"),
    "lateral_tibial_cartilage_thickness_mm": ("Lateral tibial cartilage thickness", "mm"),
    "femoral_cartilage_volume_ratio": ("Femoral cartilage : femur volume", "per mille"),
    "medial_tibial_cartilage_volume_ratio": ("Medial tibial cartilage : tibia volume", "per mille"),
    "lateral_tibial_cartilage_volume_ratio": (
        "Lateral tibial cartilage : tibia volume",
        "per mille",
    ),
    "medial_lateral_thickness_ratio": ("Medial : lateral thickness ratio", "ratio"),
}


def classifier_path(model_registry: str | Path | None = None) -> Path:
    """Where the OA classifier lives: ``<registry>/oa_kl/oa_classifier.json``."""
    if model_registry is None:
        from kneeai.config import get_settings

        model_registry = get_settings().model_registry
    return Path(model_registry) / "oa_kl" / MODEL_FILENAME


def load_classifier(path: str | Path | None = None) -> OaClassifier | None:
    """Load the trained classifier, or ``None`` when it has not been fitted.

    Returns ``None`` rather than raising: an absent classifier means the OA step
    reports itself unavailable, which is a legitimate state and must not take the
    rest of the pipeline down with it.
    """
    target = Path(path) if path is not None else classifier_path()
    if not target.exists():
        return None
    try:
        return OaClassifier.load(target)
    except Exception as exc:  # noqa: BLE001 - a corrupt model must degrade, not crash
        log.warning("OA classifier could not be loaded", fields={"error": str(exc)})
        return None


@dataclass
class OaAssessment:
    """The OA verdict for one case."""

    available: bool
    case_id: str

    oa_detected: bool | None = None
    probability: float | None = None
    threshold: float = 0.5
    predicted_kl_grade: float | None = None
    severity: str | None = None

    #: Per-feature contribution to the decision, strongest first.
    indicators: list[dict[str, Any]] = field(default_factory=list)

    features: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    reference_metrics: dict[str, Any] = field(default_factory=dict)

    #: Dataset-recorded KL grade, when the case comes from a cohort that has one.
    #: Shown for comparison; never an input to the prediction.
    dataset_kl_grade: int | None = None

    unavailable_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = OA_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "case_id": self.case_id,
            "oa_detected": self.oa_detected,
            "probability": (
                round(float(self.probability), 4) if self.probability is not None else None
            ),
            "threshold": round(float(self.threshold), 4),
            "predicted_kl_grade": (
                round(float(self.predicted_kl_grade), 2)
                if self.predicted_kl_grade is not None
                else None
            ),
            "severity": self.severity,
            "kl_threshold_for_oa": OA_KL_THRESHOLD,
            "indicators": self.indicators,
            "features": self.features,
            "model": self.model,
            "reference_metrics": self.reference_metrics,
            "dataset_kl_grade": self.dataset_kl_grade,
            "unavailable_reason": self.unavailable_reason,
            "warnings": list(self.warnings),
            "disclaimer": self.disclaimer,
            "method": (
                "L2-regularised logistic regression on seven cartilage-geometry "
                "features derived from the segmentation, trained against the "
                f"dataset's Kellgren-Lawrence grade dichotomised at KL >= "
                f"{OA_KL_THRESHOLD}. Features are computed by kneeai.oa.features "
                "using the same thickness estimator as the measurement stage."
            ),
        }


def _indicators(
    features: OaFeatures, classifier: OaClassifier
) -> list[dict[str, Any]]:
    """Per-feature contribution to the log-odds, strongest first.

    For a linear model the contribution of each feature is exactly
    ``coefficient * standardised_value``, so this is the actual decomposition of
    the decision rather than an approximation of it.
    """
    vector = features.vector()
    standardised = classifier.binary._design(np.atleast_2d(vector))[0]
    coefficients = np.asarray(classifier.binary.coefficients, dtype=np.float64)
    contributions = coefficients * standardised

    out: list[dict[str, Any]] = []
    for i, name in enumerate(FEATURE_NAMES):
        label, unit = FEATURE_LABELS.get(name, (name.replace("_", " "), ""))
        raw = float(vector[i])
        out.append(
            {
                "name": name,
                "display_name": label,
                "unit": unit,
                "value": None if not np.isfinite(raw) else round(raw, 4),
                "imputed": not np.isfinite(raw),
                "z_score": round(float(standardised[i]), 3),
                "coefficient": round(float(coefficients[i]), 4),
                "log_odds_contribution": round(float(contributions[i]), 4),
                # Positive contribution pushes towards OA.
                "direction": "towards_oa" if contributions[i] > 0 else "away_from_oa",
            }
        )
    out.sort(key=lambda d: abs(d["log_odds_contribution"]), reverse=True)
    return out


def assess_oa(
    mask: Volume,
    *,
    case_id: str,
    classifier: OaClassifier | None = None,
    dataset_kl_grade: int | None = None,
    segmentation_confidence: Mapping[str, float | None] | None = None,
) -> OaAssessment:
    """Assess osteoarthritis from a segmentation.

    Args:
        segmentation_confidence: per-structure confidence from ``result.json``.
            Low cartilage confidence is surfaced as a warning on the assessment,
            because a confident-looking OA probability computed from cartilage the
            model was unsure about is the most misleading output this feature can
            produce.
    """
    model = classifier or load_classifier()
    if model is None:
        return OaAssessment(
            available=False,
            case_id=case_id,
            dataset_kl_grade=dataset_kl_grade,
            unavailable_reason=(
                "No OA classifier is registered. Fit one with "
                "`python scripts/train_oa_classifier.py`, which requires the "
                "OAIZIB-CM manifest and its Kellgren-Lawrence grades."
            ),
        )

    features = extract_features(mask)

    if not features.segmentation_usable:
        return OaAssessment(
            available=False,
            case_id=case_id,
            features=features.to_dict(),
            dataset_kl_grade=dataset_kl_grade,
            unavailable_reason=(
                "The segmentation does not support an OA assessment. "
                + " ".join(features.notes)
            ),
            warnings=list(features.notes),
        )

    vector = features.vector()
    probability = model.probability(vector)
    grade = model.grade(vector)
    detected = bool(probability >= model.threshold)

    warnings = list(features.notes)

    # Cartilage confidence gates how much the answer is worth. The tibial
    # cartilages are this model's weakest structures, and they carry most of the
    # OA signal, so a low-confidence prediction has to say so.
    if segmentation_confidence:
        weak = [
            name
            for name in (
                Structure.FEMORAL_CARTILAGE.value,
                Structure.MEDIAL_TIBIAL_CARTILAGE.value,
                Structure.LATERAL_TIBIAL_CARTILAGE.value,
            )
            if isinstance(segmentation_confidence.get(name), (int, float))
            and float(segmentation_confidence[name] or 0.0) < 0.7
        ]
        if weak:
            warnings.append(
                "Low segmentation confidence on "
                + ", ".join(Structure(w).display_name for w in weak)
                + ". These structures supply most of the OA signal, so treat this "
                "assessment as weaker than the probability alone suggests."
            )

    if 0.4 <= probability <= 0.6:
        warnings.append(
            f"The probability ({probability:.2f}) sits close to the decision "
            f"threshold ({model.threshold:.2f}). The yes/no on a borderline case "
            "should not be read as confident either way."
        )

    metrics = dict(model.metrics or {})
    return OaAssessment(
        available=True,
        case_id=case_id,
        oa_detected=detected,
        probability=probability,
        threshold=model.threshold,
        predicted_kl_grade=grade if np.isfinite(grade) else None,
        severity=severity_for(grade) if np.isfinite(grade) else None,
        indicators=_indicators(features, model),
        features=features.to_dict(),
        model={
            "model_id": "oa_kl",
            "version": model.version,
            "architecture": "logistic",
            "task": "classification",
            **{k: v for k, v in (model.training or {}).items() if k != "per_case"},
        },
        reference_metrics=metrics,
        dataset_kl_grade=dataset_kl_grade,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Morphometric-only assessment (no trained model required)
# ---------------------------------------------------------------------------


def assess_oa_from_measurements(
    measurements: dict,
    *,
    case_id: str = "unknown",
    dataset_kl_grade: int | None = None,
    threshold: float = 0.35,
) -> dict:
    """Assess OA using the rule-based morphometric classifier.

    This is the fallback path when no trained logistic model is available.
    It reads directly from the measurements.json payload and uses
    literature-derived thresholds to produce a prediction.

    Args:
        measurements: The measurements.json content (full payload or measurements list).
        case_id: Identifier for the case.
        dataset_kl_grade: Ground-truth KL grade from the dataset, if known.
        threshold: OA detection threshold (probability above which OA is declared).

    Returns:
        A dict matching the OaAssessment shape, suitable for writing to oa.json.
    """
    classifier = MorphometricOaClassifier(oa_threshold=threshold)
    prediction = classifier.predict(measurements)

    result = {
        "available": True,
        "case_id": case_id,
        "oa_detected": prediction.oa_detected,
        "probability": round(prediction.probability, 4),
        "threshold": round(threshold, 4),
        "predicted_kl_grade": None,  # rule-based cannot predict a single grade
        "severity": prediction.severity,
        "approximate_kl_range": prediction.approximate_kl_range,
        "kl_threshold_for_oa": OA_KL_THRESHOLD,
        "indicators": prediction.indicators,
        "features": prediction.features_used,
        "features_missing": prediction.features_missing,
        "compartment_assessment": prediction.compartment_assessment,
        "confidence_score": round(prediction.confidence_score, 3),
        "raw_score": round(prediction.raw_score, 2),
        "max_possible_score": round(prediction.max_possible_score, 2),
        "model": {
            "model_id": "morphometric_rules",
            "version": "1.0.0",
            "architecture": "deterministic_rule_based",
            "task": "classification",
            "method": "Weighted scoring on literature-derived morphometric thresholds",
        },
        "reference_metrics": {},
        "dataset_kl_grade": dataset_kl_grade,
        "unavailable_reason": None,
        "warnings": prediction.warnings,
        "disclaimer": prediction.disclaimer,
        "method": (
            "Deterministic rule-based classifier using weighted scoring across "
            "cartilage thickness, JLCA, asymmetry, and volume features. Thresholds "
            "derived from Eckstein et al. (2006), Wirth et al. (2013), and "
            "Sharma et al. (2001). No model training required."
        ),
    }
    return result
