"""Osteoarthritis assessment from a knee segmentation.

The one learned component here is a logistic regression over cartilage geometry
(:mod:`kneeai.oa.classifier`). Everything it consumes is computed deterministically
from the segmentation by :mod:`kneeai.oa.features`, and everything it emits is
wrapped in its own measured validation performance by :mod:`kneeai.oa.assess`.

Kept as its own package rather than folded into ``measure/`` because the
distinction matters: ``measure/`` reports quantities, this package reports a
*judgement*, and judgements need model cards, thresholds and held-out metrics that
measurements do not.
"""

from __future__ import annotations

from kneeai.oa.assess import (
    OA_DISCLAIMER,
    OaAssessment,
    assess_oa,
    assess_oa_from_measurements,
    classifier_path,
    load_classifier,
)
from kneeai.oa.classifier import (
    OA_KL_THRESHOLD,
    BinaryMetrics,
    LogisticModel,
    OaClassifier,
    evaluate_binary,
    fit_logistic,
    severity_for,
)
from kneeai.oa.features import FEATURE_NAMES, OaFeatures, extract_features
from kneeai.oa.morphometric_classifier import (
    MorphometricOaClassifier,
    MorphometricOaPrediction,
    predict_oa_from_measurements,
)

__all__ = [
    "FEATURE_NAMES",
    "MorphometricOaClassifier",
    "MorphometricOaPrediction",
    "OA_DISCLAIMER",
    "OA_KL_THRESHOLD",
    "BinaryMetrics",
    "LogisticModel",
    "OaAssessment",
    "OaClassifier",
    "OaFeatures",
    "assess_oa",
    "assess_oa_from_measurements",
    "classifier_path",
    "evaluate_binary",
    "extract_features",
    "fit_logistic",
    "load_classifier",
    "predict_oa_from_measurements",
    "severity_for",
]
