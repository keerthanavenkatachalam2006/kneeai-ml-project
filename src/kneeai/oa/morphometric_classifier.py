"""Deterministic rule-based OA classifier using morphometric features.

This classifier uses evidence-based thresholds from the orthopaedic literature to
assess osteoarthritis from MR-derived cartilage and bone geometry. It requires no
trained model — only the measurements.json output by the KneeAI pipeline.

The approach uses weighted scoring across multiple morphometric indicators:

    - Cartilage thickness (femoral, medial tibial, lateral tibial)
    - Cartilage thickness asymmetry (medial vs lateral)
    - Joint line convergence angle (JLCA)
    - Cartilage volume ratios
    - Posterior tibial slope (as secondary indicator)
    - Femoral-to-tibial width ratio

References:
    - Eckstein et al. "Quantitative MRI measures of cartilage" (2006)
    - Wirth et al. "Longitudinal analysis of cartilage" (2013)
    - Kellgren & Lawrence. "Radiological assessment of osteo-arthrosis" (1957)
    - Hunter et al. "The reliability of a new scoring system for knee OA" (2008)
    - Sharma et al. "The role of knee alignment in disease progression" (2001)
    - Cicuttini et al. "Association of cartilage defects with loss of knee
      cartilage in healthy, middle-age adults" (2005)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MorphometricOaClassifier",
    "MorphometricOaPrediction",
    "predict_oa_from_measurements",
]


# ---------------------------------------------------------------------------
# Reference thresholds from the literature
# ---------------------------------------------------------------------------

#: Normal cartilage thickness ranges (mm) from Eckstein et al. 2006, Wirth 2013.
#: These are volume/surface-area-derived means from 3T MR (DESS sequence).
NORMAL_THICKNESS = {
    "femoral_cartilage": {"mean": 2.5, "sd": 0.4, "thin": 1.8, "very_thin": 1.2},
    "medial_tibial_cartilage": {"mean": 2.0, "sd": 0.4, "thin": 1.4, "very_thin": 0.9},
    "lateral_tibial_cartilage": {"mean": 2.2, "sd": 0.4, "thin": 1.5, "very_thin": 1.0},
}

#: JLCA threshold: >3° suggests medial compartment disease (Sharma et al. 2001)
JLCA_THRESHOLD_MILD = 3.0  # degrees
JLCA_THRESHOLD_MODERATE = 5.0
JLCA_THRESHOLD_SEVERE = 8.0

#: Cartilage asymmetry: >20% suggests compartmental disease
ASYMMETRY_THRESHOLD_MILD = 15.0  # percent
ASYMMETRY_THRESHOLD_MODERATE = 25.0
ASYMMETRY_THRESHOLD_SEVERE = 40.0

#: Medial:lateral thickness ratio (< 0.8 suggests medial disease)
ML_RATIO_MILD = 0.85
ML_RATIO_MODERATE = 0.7
ML_RATIO_SEVERE = 0.5


# ---------------------------------------------------------------------------
# Scoring weights — how much each indicator contributes
# ---------------------------------------------------------------------------

#: Maximum contribution from each feature domain to the final score.
#: Total maximum = 100 (maps directly to a 0-1 probability-like score).
WEIGHTS = {
    "femoral_thickness": 18,       # primary indicator
    "medial_tibial_thickness": 22, # most affected in medial OA
    "lateral_tibial_thickness": 12,
    "jlca": 18,                    # joint space convergence, strong OA signal
    "asymmetry": 14,               # compartmental imbalance
    "volume_loss": 10,             # overall cartilage loss
    "secondary_signs": 6,         # PTS, bone geometry
}

#: Total possible score
MAX_SCORE = sum(WEIGHTS.values())  # 100


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


@dataclass
class MorphometricOaPrediction:
    """Result of the morphometric OA classification."""

    oa_detected: bool
    probability: float
    severity: str  # "none", "mild", "moderate", "severe"
    approximate_kl_range: str  # e.g. "2-3"
    confidence_score: float  # 0-1, how many features were available
    raw_score: float  # 0-100
    max_possible_score: float  # max given available features

    indicators: list[dict[str, Any]] = field(default_factory=list)
    features_used: dict[str, Any] = field(default_factory=dict)
    features_missing: list[str] = field(default_factory=list)
    compartment_assessment: dict[str, str] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)
    disclaimer: str = (
        "Not a clinical diagnosis. This is a deterministic rule-based assessment "
        "using cartilage morphometry thresholds from the orthopaedic literature. "
        "It approximates radiographic KL grading from MR geometry but is not "
        "equivalent to a radiologist's assessment of a weight-bearing radiograph. "
        "It has no regulatory clearance and must not inform patient care."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "oa_detected": self.oa_detected,
            "probability": round(self.probability, 4),
            "severity": self.severity,
            "approximate_kl_range": self.approximate_kl_range,
            "confidence_score": round(self.confidence_score, 3),
            "raw_score": round(self.raw_score, 2),
            "max_possible_score": round(self.max_possible_score, 2),
            "indicators": self.indicators,
            "features_used": self.features_used,
            "features_missing": self.features_missing,
            "compartment_assessment": self.compartment_assessment,
            "warnings": self.warnings,
            "method": "deterministic_morphometric_rules",
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


class MorphometricOaClassifier:
    """Rule-based OA classifier operating on pipeline measurements.

    Reads the measurements.json produced by the KneeAI measurement engine and
    applies literature-derived thresholds to produce:
        - OA detected (yes/no) at a configurable threshold
        - A probability-like confidence score (0-1)
        - Severity categorisation (none/mild/moderate/severe)
        - Approximate KL grade range

    The classifier is deterministic and requires no training data. It uses a
    weighted scoring system where each morphometric indicator contributes
    proportionally to its diagnostic relevance as established in the literature.
    """

    def __init__(self, oa_threshold: float = 0.35):
        """
        Args:
            oa_threshold: Score threshold (0-1) above which OA is declared.
                Default 0.35 corresponds roughly to KL >= 2.
        """
        self.oa_threshold = oa_threshold

    def predict(self, measurements: dict[str, Any]) -> MorphometricOaPrediction:
        """Predict OA status from a measurements payload.

        Args:
            measurements: The full measurements.json content, or just the
                "measurements" list within it.

        Returns:
            MorphometricOaPrediction with all classification details.
        """
        # Extract measurement values from the pipeline output format
        features = self._extract_features(measurements)
        return self._classify(features)

    def predict_from_file(self, path: str | Path) -> MorphometricOaPrediction:
        """Predict OA from a measurements.json file on disk."""
        with open(path) as f:
            data = json.load(f)
        return self.predict(data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_features(self, measurements: dict[str, Any]) -> dict[str, float | None]:
        """Pull the relevant features from the measurements payload."""
        # The measurements can be either the full file or the inner list
        if "measurements" in measurements and isinstance(measurements["measurements"], list):
            meas_list = measurements["measurements"]
        elif isinstance(measurements, list):
            meas_list = measurements
        else:
            meas_list = []

        # Build a lookup by measurement name
        lookup: dict[str, float | None] = {}
        for m in meas_list:
            name = m.get("name", "")
            quantity = m.get("quantity", {})
            value = quantity.get("value")
            if value is not None:
                lookup[name] = float(value)

        # Extract the features we need
        features: dict[str, float | None] = {
            "femoral_cartilage_mean_thickness": lookup.get("femoral_cartilage_mean_thickness"),
            "medial_tibial_cartilage_mean_thickness": lookup.get(
                "medial_tibial_cartilage_mean_thickness"
            ),
            "lateral_tibial_cartilage_mean_thickness": lookup.get(
                "lateral_tibial_cartilage_mean_thickness"
            ),
            "cartilage_thickness_asymmetry": lookup.get("cartilage_thickness_asymmetry"),
            "jlca": lookup.get("jlca"),
            "posterior_tibial_slope": lookup.get("posterior_tibial_slope"),
            "femoral_ml_width": lookup.get("femoral_ml_width"),
            "tibial_ml_width": lookup.get("tibial_ml_width"),
            "femoral_cartilage_volume": lookup.get("femoral_cartilage_volume"),
            "medial_tibial_cartilage_volume": lookup.get("medial_tibial_cartilage_volume"),
            "lateral_tibial_cartilage_volume": lookup.get("lateral_tibial_cartilage_volume"),
        }
        return features

    def _classify(self, features: dict[str, float | None]) -> MorphometricOaPrediction:
        """Run the weighted scoring classification."""
        score = 0.0
        max_available = 0.0
        indicators: list[dict[str, Any]] = []
        features_used: dict[str, Any] = {}
        features_missing: list[str] = []
        warnings: list[str] = []
        compartment_assessment: dict[str, str] = {}

        # --- Femoral cartilage thickness ---
        fc_thick = features.get("femoral_cartilage_mean_thickness")
        if fc_thick is not None:
            max_available += WEIGHTS["femoral_thickness"]
            features_used["femoral_cartilage_mean_thickness_mm"] = round(fc_thick, 3)
            fc_score, fc_indicators = self._score_thickness(
                fc_thick, "femoral_cartilage", WEIGHTS["femoral_thickness"]
            )
            score += fc_score
            indicators.extend(fc_indicators)
            compartment_assessment["femoral"] = self._compartment_status(fc_thick, "femoral_cartilage")
        else:
            features_missing.append("femoral_cartilage_mean_thickness")

        # --- Medial tibial cartilage thickness ---
        mtc_thick = features.get("medial_tibial_cartilage_mean_thickness")
        if mtc_thick is not None:
            max_available += WEIGHTS["medial_tibial_thickness"]
            features_used["medial_tibial_cartilage_mean_thickness_mm"] = round(mtc_thick, 3)
            mtc_score, mtc_indicators = self._score_thickness(
                mtc_thick, "medial_tibial_cartilage", WEIGHTS["medial_tibial_thickness"]
            )
            score += mtc_score
            indicators.extend(mtc_indicators)
            compartment_assessment["medial_tibial"] = self._compartment_status(
                mtc_thick, "medial_tibial_cartilage"
            )
        else:
            features_missing.append("medial_tibial_cartilage_mean_thickness")

        # --- Lateral tibial cartilage thickness ---
        ltc_thick = features.get("lateral_tibial_cartilage_mean_thickness")
        if ltc_thick is not None:
            max_available += WEIGHTS["lateral_tibial_thickness"]
            features_used["lateral_tibial_cartilage_mean_thickness_mm"] = round(ltc_thick, 3)
            ltc_score, ltc_indicators = self._score_thickness(
                ltc_thick, "lateral_tibial_cartilage", WEIGHTS["lateral_tibial_thickness"]
            )
            score += ltc_score
            indicators.extend(ltc_indicators)
            compartment_assessment["lateral_tibial"] = self._compartment_status(
                ltc_thick, "lateral_tibial_cartilage"
            )
        else:
            features_missing.append("lateral_tibial_cartilage_mean_thickness")

        # --- JLCA ---
        jlca = features.get("jlca")
        if jlca is not None:
            max_available += WEIGHTS["jlca"]
            features_used["jlca_degrees"] = round(jlca, 2)
            jlca_score, jlca_indicators = self._score_jlca(jlca, WEIGHTS["jlca"])
            score += jlca_score
            indicators.extend(jlca_indicators)
        else:
            features_missing.append("jlca")

        # --- Cartilage asymmetry ---
        asymmetry = features.get("cartilage_thickness_asymmetry")
        if asymmetry is not None:
            max_available += WEIGHTS["asymmetry"]
            features_used["cartilage_thickness_asymmetry_pct"] = round(asymmetry, 2)
            asym_score, asym_indicators = self._score_asymmetry(
                asymmetry, WEIGHTS["asymmetry"]
            )
            score += asym_score
            indicators.extend(asym_indicators)
        else:
            features_missing.append("cartilage_thickness_asymmetry")

        # --- Volume loss (compare to expected volumes) ---
        fc_vol = features.get("femoral_cartilage_volume")
        mtc_vol = features.get("medial_tibial_cartilage_volume")
        ltc_vol = features.get("lateral_tibial_cartilage_volume")
        if fc_vol is not None and mtc_vol is not None:
            max_available += WEIGHTS["volume_loss"]
            features_used["femoral_cartilage_volume_mm3"] = round(fc_vol, 1)
            features_used["medial_tibial_cartilage_volume_mm3"] = round(mtc_vol, 1)
            if ltc_vol is not None:
                features_used["lateral_tibial_cartilage_volume_mm3"] = round(ltc_vol, 1)
            vol_score, vol_indicators = self._score_volume(
                fc_vol, mtc_vol, ltc_vol, WEIGHTS["volume_loss"]
            )
            score += vol_score
            indicators.extend(vol_indicators)
        else:
            if fc_vol is None:
                features_missing.append("femoral_cartilage_volume")
            if mtc_vol is None:
                features_missing.append("medial_tibial_cartilage_volume")

        # --- Secondary signs (PTS, bone geometry) ---
        pts = features.get("posterior_tibial_slope")
        fem_width = features.get("femoral_ml_width")
        tib_width = features.get("tibial_ml_width")
        if pts is not None or (fem_width is not None and tib_width is not None):
            max_available += WEIGHTS["secondary_signs"]
            sec_score, sec_indicators = self._score_secondary(
                pts, fem_width, tib_width, WEIGHTS["secondary_signs"]
            )
            score += sec_score
            indicators.extend(sec_indicators)
            if pts is not None:
                features_used["posterior_tibial_slope_deg"] = round(pts, 2)
            if fem_width is not None and tib_width is not None:
                features_used["femoral_tibial_width_ratio"] = round(fem_width / tib_width, 3)
        else:
            if pts is None:
                features_missing.append("posterior_tibial_slope")

        # --- Compute final probability-like score ---
        if max_available > 0:
            # Normalise to 0-1 based on available features
            probability = score / max_available
            confidence = max_available / MAX_SCORE
        else:
            probability = 0.0
            confidence = 0.0
            warnings.append(
                "No morphometric features available for OA assessment. "
                "The measurements.json may be empty or incompatible."
            )

        # Confidence penalty if too few features available
        if confidence < 0.5:
            warnings.append(
                f"Only {confidence * 100:.0f}% of expected morphometric features "
                f"are available. The assessment has reduced reliability."
            )

        # --- Map to severity and KL grade ---
        severity = self._map_severity(probability)
        kl_range = self._map_kl_range(probability)
        oa_detected = probability >= self.oa_threshold

        # Sort indicators by contribution (strongest first)
        indicators.sort(key=lambda x: x.get("contribution", 0), reverse=True)

        return MorphometricOaPrediction(
            oa_detected=oa_detected,
            probability=probability,
            severity=severity,
            approximate_kl_range=kl_range,
            confidence_score=confidence,
            raw_score=score,
            max_possible_score=max_available,
            indicators=indicators,
            features_used=features_used,
            features_missing=features_missing,
            compartment_assessment=compartment_assessment,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Scoring functions
    # ------------------------------------------------------------------

    def _score_thickness(
        self, thickness_mm: float, structure: str, weight: float
    ) -> tuple[float, list[dict[str, Any]]]:
        """Score cartilage thickness against literature norms."""
        ref = NORMAL_THICKNESS[structure]
        indicators: list[dict[str, Any]] = []
        score = 0.0

        if thickness_mm <= 0:
            # Full-thickness loss
            score = weight
            indicators.append({
                "feature": f"{structure}_thickness",
                "value": thickness_mm,
                "unit": "mm",
                "finding": f"Full-thickness cartilage loss ({structure.replace('_', ' ')})",
                "severity": "severe",
                "contribution": weight,
                "reference": f"Normal: {ref['mean']} ± {ref['sd']} mm",
            })
        elif thickness_mm < ref["very_thin"]:
            # Very thin: 80-100% of weight
            fraction = 0.8 + 0.2 * (1.0 - thickness_mm / ref["very_thin"])
            score = weight * min(fraction, 1.0)
            indicators.append({
                "feature": f"{structure}_thickness",
                "value": round(thickness_mm, 2),
                "unit": "mm",
                "finding": f"Very thin {structure.replace('_', ' ')} (< {ref['very_thin']} mm)",
                "severity": "severe",
                "contribution": round(score, 2),
                "reference": f"Normal: {ref['mean']} ± {ref['sd']} mm",
            })
        elif thickness_mm < ref["thin"]:
            # Thin: 40-80% of weight
            t_range = ref["thin"] - ref["very_thin"]
            fraction = 0.4 + 0.4 * (1.0 - (thickness_mm - ref["very_thin"]) / t_range)
            score = weight * fraction
            indicators.append({
                "feature": f"{structure}_thickness",
                "value": round(thickness_mm, 2),
                "unit": "mm",
                "finding": f"Thin {structure.replace('_', ' ')} (< {ref['thin']} mm)",
                "severity": "moderate",
                "contribution": round(score, 2),
                "reference": f"Normal: {ref['mean']} ± {ref['sd']} mm",
            })
        elif thickness_mm < ref["mean"] - ref["sd"]:
            # Below 1 SD: 10-40% of weight
            low_normal = ref["mean"] - ref["sd"]
            t_range = low_normal - ref["thin"]
            fraction = 0.1 + 0.3 * (1.0 - (thickness_mm - ref["thin"]) / t_range)
            score = weight * fraction
            indicators.append({
                "feature": f"{structure}_thickness",
                "value": round(thickness_mm, 2),
                "unit": "mm",
                "finding": f"Below-average {structure.replace('_', ' ')} thickness",
                "severity": "mild",
                "contribution": round(score, 2),
                "reference": f"Normal: {ref['mean']} ± {ref['sd']} mm",
            })
        # else: normal thickness, no score contribution

        return score, indicators

    def _score_jlca(
        self, jlca_deg: float, weight: float
    ) -> tuple[float, list[dict[str, Any]]]:
        """Score JLCA. Positive = medial convergence (varus pattern)."""
        indicators: list[dict[str, Any]] = []
        score = 0.0
        abs_jlca = abs(jlca_deg)

        if abs_jlca >= JLCA_THRESHOLD_SEVERE:
            # Severe: 80-100% of weight
            score = weight * min(0.8 + 0.2 * (abs_jlca - JLCA_THRESHOLD_SEVERE) / 5.0, 1.0)
            indicators.append({
                "feature": "jlca",
                "value": round(jlca_deg, 1),
                "unit": "degrees",
                "finding": f"Severe joint line convergence ({abs_jlca:.1f}° > {JLCA_THRESHOLD_SEVERE}°)",
                "severity": "severe",
                "contribution": round(score, 2),
                "reference": f"Normal: 0-{JLCA_THRESHOLD_MILD}°",
                "direction": "medial" if jlca_deg > 0 else "lateral",
            })
        elif abs_jlca >= JLCA_THRESHOLD_MODERATE:
            fraction = 0.5 + 0.3 * (abs_jlca - JLCA_THRESHOLD_MODERATE) / (
                JLCA_THRESHOLD_SEVERE - JLCA_THRESHOLD_MODERATE
            )
            score = weight * fraction
            indicators.append({
                "feature": "jlca",
                "value": round(jlca_deg, 1),
                "unit": "degrees",
                "finding": f"Moderate joint line convergence ({abs_jlca:.1f}°)",
                "severity": "moderate",
                "contribution": round(score, 2),
                "reference": f"Normal: 0-{JLCA_THRESHOLD_MILD}°",
                "direction": "medial" if jlca_deg > 0 else "lateral",
            })
        elif abs_jlca >= JLCA_THRESHOLD_MILD:
            fraction = 0.2 + 0.3 * (abs_jlca - JLCA_THRESHOLD_MILD) / (
                JLCA_THRESHOLD_MODERATE - JLCA_THRESHOLD_MILD
            )
            score = weight * fraction
            indicators.append({
                "feature": "jlca",
                "value": round(jlca_deg, 1),
                "unit": "degrees",
                "finding": f"Mild joint line convergence ({abs_jlca:.1f}°)",
                "severity": "mild",
                "contribution": round(score, 2),
                "reference": f"Normal: 0-{JLCA_THRESHOLD_MILD}°",
                "direction": "medial" if jlca_deg > 0 else "lateral",
            })

        return score, indicators

    def _score_asymmetry(
        self, asymmetry_pct: float, weight: float
    ) -> tuple[float, list[dict[str, Any]]]:
        """Score cartilage thickness asymmetry.

        Positive asymmetry = lateral thicker than medial = medial disease pattern.
        """
        indicators: list[dict[str, Any]] = []
        score = 0.0
        abs_asym = abs(asymmetry_pct)

        if abs_asym >= ASYMMETRY_THRESHOLD_SEVERE:
            score = weight * min(0.8 + 0.2 * (abs_asym - ASYMMETRY_THRESHOLD_SEVERE) / 20.0, 1.0)
            indicators.append({
                "feature": "cartilage_asymmetry",
                "value": round(asymmetry_pct, 1),
                "unit": "%",
                "finding": f"Severe compartmental cartilage asymmetry ({abs_asym:.1f}%)",
                "severity": "severe",
                "contribution": round(score, 2),
                "reference": f"Normal: < {ASYMMETRY_THRESHOLD_MILD}%",
                "compartment": "medial" if asymmetry_pct > 0 else "lateral",
            })
        elif abs_asym >= ASYMMETRY_THRESHOLD_MODERATE:
            fraction = 0.4 + 0.4 * (abs_asym - ASYMMETRY_THRESHOLD_MODERATE) / (
                ASYMMETRY_THRESHOLD_SEVERE - ASYMMETRY_THRESHOLD_MODERATE
            )
            score = weight * fraction
            indicators.append({
                "feature": "cartilage_asymmetry",
                "value": round(asymmetry_pct, 1),
                "unit": "%",
                "finding": f"Moderate compartmental asymmetry ({abs_asym:.1f}%)",
                "severity": "moderate",
                "contribution": round(score, 2),
                "reference": f"Normal: < {ASYMMETRY_THRESHOLD_MILD}%",
                "compartment": "medial" if asymmetry_pct > 0 else "lateral",
            })
        elif abs_asym >= ASYMMETRY_THRESHOLD_MILD:
            fraction = 0.15 + 0.25 * (abs_asym - ASYMMETRY_THRESHOLD_MILD) / (
                ASYMMETRY_THRESHOLD_MODERATE - ASYMMETRY_THRESHOLD_MILD
            )
            score = weight * fraction
            indicators.append({
                "feature": "cartilage_asymmetry",
                "value": round(asymmetry_pct, 1),
                "unit": "%",
                "finding": f"Mild compartmental asymmetry ({abs_asym:.1f}%)",
                "severity": "mild",
                "contribution": round(score, 2),
                "reference": f"Normal: < {ASYMMETRY_THRESHOLD_MILD}%",
                "compartment": "medial" if asymmetry_pct > 0 else "lateral",
            })

        return score, indicators

    def _score_volume(
        self,
        fc_vol: float,
        mtc_vol: float,
        ltc_vol: float | None,
        weight: float,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Score cartilage volume against expected ranges.

        Expected volumes (3D DESS, 0.3-0.7mm spacing):
            - Femoral cartilage: 12000-20000 mm³ (healthy)
            - Medial tibial cartilage: 2500-4500 mm³ (healthy)
            - Lateral tibial cartilage: 2500-4500 mm³ (healthy)

        These are approximate and highly variable with body size, so volume
        contributes less weight than thickness.
        """
        indicators: list[dict[str, Any]] = []
        score = 0.0

        # Medial tibial volume is the most sensitive to OA
        # Expected range for healthy: ~2500-4500 mm³
        if mtc_vol < 1500:
            score += weight * 0.8
            indicators.append({
                "feature": "medial_tibial_cartilage_volume",
                "value": round(mtc_vol, 0),
                "unit": "mm³",
                "finding": "Low medial tibial cartilage volume (< 1500 mm³)",
                "severity": "moderate",
                "contribution": round(weight * 0.8, 2),
                "reference": "Expected healthy: 2500-4500 mm³",
            })
        elif mtc_vol < 2500:
            fraction = 0.3 + 0.5 * (1.0 - (mtc_vol - 1500) / 1000)
            partial = weight * fraction
            score += partial
            indicators.append({
                "feature": "medial_tibial_cartilage_volume",
                "value": round(mtc_vol, 0),
                "unit": "mm³",
                "finding": "Below-average medial tibial cartilage volume",
                "severity": "mild",
                "contribution": round(partial, 2),
                "reference": "Expected healthy: 2500-4500 mm³",
            })

        # Medial:lateral volume ratio
        if ltc_vol is not None and ltc_vol > 0:
            vol_ratio = mtc_vol / ltc_vol
            if vol_ratio > 3.0:
                # Lateral much smaller than medial — possible lateral disease
                partial = weight * 0.3
                score += partial
                indicators.append({
                    "feature": "volume_ratio_ml",
                    "value": round(vol_ratio, 2),
                    "unit": "ratio",
                    "finding": "Marked medial-to-lateral volume imbalance",
                    "severity": "mild",
                    "contribution": round(partial, 2),
                    "reference": "Expected ratio ~1.0-2.0",
                })

        # Cap at weight
        score = min(score, weight)
        return score, indicators

    def _score_secondary(
        self,
        pts: float | None,
        fem_width: float | None,
        tib_width: float | None,
        weight: float,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Score secondary morphometric signs of OA."""
        indicators: list[dict[str, Any]] = []
        score = 0.0

        # Posterior tibial slope: extreme values may indicate bony remodelling.
        # Normal: 5-12°. Negative or > 15° is abnormal.
        if pts is not None:
            if pts < 0:
                # Reversed slope — suggests significant remodelling
                partial = weight * 0.6
                score += partial
                indicators.append({
                    "feature": "posterior_tibial_slope",
                    "value": round(pts, 1),
                    "unit": "degrees",
                    "finding": "Reversed posterior tibial slope (anterior tilt)",
                    "severity": "moderate",
                    "contribution": round(partial, 2),
                    "reference": "Normal: 5-12°",
                })
            elif pts > 15:
                partial = weight * 0.4
                score += partial
                indicators.append({
                    "feature": "posterior_tibial_slope",
                    "value": round(pts, 1),
                    "unit": "degrees",
                    "finding": "Excessive posterior tibial slope",
                    "severity": "mild",
                    "contribution": round(partial, 2),
                    "reference": "Normal: 5-12°",
                })

        # Femoral-to-tibial width ratio: significant mismatch may indicate
        # periarticular bony changes (osteophytes widen the joint)
        if fem_width is not None and tib_width is not None and tib_width > 0:
            ratio = fem_width / tib_width
            if ratio > 1.25:
                # Femoral flaring — possible osteophyte formation
                partial = weight * 0.4
                score += partial
                indicators.append({
                    "feature": "femoral_tibial_width_ratio",
                    "value": round(ratio, 3),
                    "unit": "ratio",
                    "finding": "Increased femoral-to-tibial width ratio (possible osteophyte widening)",
                    "severity": "mild",
                    "contribution": round(partial, 2),
                    "reference": "Expected: 1.05-1.20",
                })

        score = min(score, weight)
        return score, indicators

    # ------------------------------------------------------------------
    # Mapping functions
    # ------------------------------------------------------------------

    @staticmethod
    def _compartment_status(thickness_mm: float, structure: str) -> str:
        """Map thickness to a compartment status string."""
        ref = NORMAL_THICKNESS[structure]
        if thickness_mm <= 0:
            return "full-thickness loss"
        elif thickness_mm < ref["very_thin"]:
            return "severe thinning"
        elif thickness_mm < ref["thin"]:
            return "moderate thinning"
        elif thickness_mm < ref["mean"] - ref["sd"]:
            return "mild thinning"
        else:
            return "normal"

    @staticmethod
    def _map_severity(probability: float) -> str:
        """Map probability to severity category."""
        if probability >= 0.70:
            return "severe"
        elif probability >= 0.50:
            return "moderate"
        elif probability >= 0.35:
            return "mild"
        else:
            return "none"

    @staticmethod
    def _map_kl_range(probability: float) -> str:
        """Map probability to approximate KL grade range."""
        if probability >= 0.75:
            return "3-4"
        elif probability >= 0.55:
            return "2-3"
        elif probability >= 0.35:
            return "1-2"
        elif probability >= 0.15:
            return "0-1"
        else:
            return "0"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def predict_oa_from_measurements(
    measurements: dict[str, Any] | str | Path,
    *,
    threshold: float = 0.35,
) -> MorphometricOaPrediction:
    """One-call interface to predict OA from measurements.

    Args:
        measurements: Either the measurements dict, a path to measurements.json,
            or a string path.
        threshold: OA detection threshold (default 0.35 ~ KL >= 2).

    Returns:
        MorphometricOaPrediction with all classification details.
    """
    classifier = MorphometricOaClassifier(oa_threshold=threshold)

    if isinstance(measurements, (str, Path)):
        return classifier.predict_from_file(measurements)
    return classifier.predict(measurements)
