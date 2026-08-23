"""
Implant Database and Matching Engine.
Contains knee implant specifications from published surgical technique guides
and matches patient anatomy to optimal implant candidates.
"""
import numpy as np
from typing import Optional
from pydantic import BaseModel


class ImplantSpec(BaseModel):
    """Specification for a knee implant component."""
    manufacturer: str
    system: str
    component: str  # "femoral" or "tibial"
    size: str
    ml_width_mm: float
    ap_depth_mm: float
    thickness_mm: Optional[float] = None
    source: str  # Citation/reference


class ImplantMatch(BaseModel):
    """Result of matching a patient to an implant."""
    implant: ImplantSpec
    score: float  # 0-100, higher is better
    ml_mismatch_mm: float
    ap_mismatch_mm: float
    ml_coverage_pct: float
    ap_coverage_pct: float
    fit_category: str  # "optimal", "acceptable", "overhang_risk", "underhang_risk"
    explanation: str


# Published implant dimensions sourced from surgical technique guides.
#
# AUDIT NOTE (Stage 1.5):
# These values are APPROXIMATE and sourced from publicly available surgical
# technique guides and published research comparing implant systems. They
# represent typical size progressions but may not exactly match current
# production specifications. Exact dimensions should be verified against
# the manufacturer's current product catalog for clinical use.
#
# Sources:
# - Mahoney & Kinsey, "Overhang of the femoral component in TKA", J Arthroplasty (2010)
# - Dai et al., "Tibial component size in TKA", Knee (2014)
# - Hitt et al., "Anthropometric measurements of the human knee", J Arthroplasty (2003)
# - Manufacturer surgical technique guides (public PDFs):
#   * Stryker Triathlon Total Knee System - Surgical Protocol (2018)
#   * Zimmer NexGen Complete Knee Solution - Surgical Technique (2016)
#   * Smith & Nephew LEGION Total Knee System - Surgical Technique (2017)
#
# Size progressions are approximately 3-4mm increments in ML width per size,
# which matches published surgical technique documentation.
#
# DISCLAIMER: These are representative values for research/demonstration purposes.
# NOT for clinical implant selection. Clinical sizing requires manufacturer-specific
# templates, intraoperative measurements, and surgeon judgment.

FEMORAL_IMPLANTS: list[ImplantSpec] = [
    # Stryker Triathlon (representative sizes)
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="femoral",
                size="1", ml_width_mm=55.0, ap_depth_mm=51.5,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="femoral",
                size="2", ml_width_mm=57.0, ap_depth_mm=55.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="femoral",
                size="3", ml_width_mm=60.0, ap_depth_mm=58.5,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="femoral",
                size="4", ml_width_mm=63.0, ap_depth_mm=62.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="femoral",
                size="5", ml_width_mm=66.5, ap_depth_mm=65.5,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="femoral",
                size="6", ml_width_mm=70.0, ap_depth_mm=69.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="femoral",
                size="7", ml_width_mm=74.0, ap_depth_mm=72.5,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="femoral",
                size="8", ml_width_mm=78.0, ap_depth_mm=76.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),

    # Zimmer NexGen (representative sizes)
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="femoral",
                size="C", ml_width_mm=56.0, ap_depth_mm=52.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="femoral",
                size="D", ml_width_mm=59.0, ap_depth_mm=56.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="femoral",
                size="E", ml_width_mm=62.0, ap_depth_mm=60.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="femoral",
                size="F", ml_width_mm=65.0, ap_depth_mm=64.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="femoral",
                size="G", ml_width_mm=69.0, ap_depth_mm=68.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="femoral",
                size="H", ml_width_mm=74.5, ap_depth_mm=73.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),

    # Smith & Nephew LEGION (representative sizes)
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="femoral",
                size="2", ml_width_mm=56.5, ap_depth_mm=53.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="femoral",
                size="3", ml_width_mm=60.0, ap_depth_mm=57.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="femoral",
                size="4", ml_width_mm=63.5, ap_depth_mm=61.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="femoral",
                size="5", ml_width_mm=67.0, ap_depth_mm=65.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="femoral",
                size="6", ml_width_mm=71.0, ap_depth_mm=69.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="femoral",
                size="7", ml_width_mm=76.0, ap_depth_mm=74.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
]

TIBIAL_IMPLANTS: list[ImplantSpec] = [
    # Stryker Triathlon Tibial
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="tibial",
                size="1", ml_width_mm=57.0, ap_depth_mm=43.0, thickness_mm=10.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="tibial",
                size="2", ml_width_mm=61.0, ap_depth_mm=46.0, thickness_mm=10.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="tibial",
                size="3", ml_width_mm=65.0, ap_depth_mm=49.0, thickness_mm=10.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="tibial",
                size="4", ml_width_mm=69.0, ap_depth_mm=52.0, thickness_mm=10.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="tibial",
                size="5", ml_width_mm=73.0, ap_depth_mm=55.0, thickness_mm=10.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="tibial",
                size="6", ml_width_mm=77.0, ap_depth_mm=58.0, thickness_mm=10.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="tibial",
                size="7", ml_width_mm=81.0, ap_depth_mm=61.0, thickness_mm=10.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),
    ImplantSpec(manufacturer="Stryker", system="Triathlon", component="tibial",
                size="8", ml_width_mm=85.0, ap_depth_mm=64.0, thickness_mm=10.0,
                source="Stryker Triathlon Surgical Protocol (2018), approximate values"),

    # Zimmer NexGen Tibial
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="tibial",
                size="C", ml_width_mm=59.0, ap_depth_mm=44.0, thickness_mm=10.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="tibial",
                size="D", ml_width_mm=63.0, ap_depth_mm=47.0, thickness_mm=10.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="tibial",
                size="E", ml_width_mm=67.0, ap_depth_mm=50.0, thickness_mm=10.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="tibial",
                size="F", ml_width_mm=71.0, ap_depth_mm=53.0, thickness_mm=10.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="tibial",
                size="G", ml_width_mm=75.0, ap_depth_mm=57.0, thickness_mm=10.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),
    ImplantSpec(manufacturer="Zimmer Biomet", system="NexGen", component="tibial",
                size="H", ml_width_mm=81.0, ap_depth_mm=61.0, thickness_mm=10.0,
                source="Zimmer NexGen Surgical Technique (2016), approximate values"),

    # Smith & Nephew LEGION Tibial
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="tibial",
                size="2", ml_width_mm=60.0, ap_depth_mm=44.0, thickness_mm=10.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="tibial",
                size="3", ml_width_mm=64.0, ap_depth_mm=47.0, thickness_mm=10.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="tibial",
                size="4", ml_width_mm=68.0, ap_depth_mm=50.0, thickness_mm=10.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="tibial",
                size="5", ml_width_mm=72.0, ap_depth_mm=54.0, thickness_mm=10.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="tibial",
                size="6", ml_width_mm=77.0, ap_depth_mm=57.0, thickness_mm=10.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
    ImplantSpec(manufacturer="Smith & Nephew", system="LEGION", component="tibial",
                size="7", ml_width_mm=82.0, ap_depth_mm=61.0, thickness_mm=10.0,
                source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"),
]


class ImplantMatcher:
    """
    Matches patient knee anatomy measurements to implant candidates.
    Ranks by dimensional fit with clinical reasoning.
    """

    # Tolerance thresholds (mm)
    OPTIMAL_TOLERANCE_MM = 2.0
    ACCEPTABLE_TOLERANCE_MM = 4.0
    OVERHANG_THRESHOLD_MM = 3.0  # Implant wider than bone

    def __init__(self):
        self.femoral_db = FEMORAL_IMPLANTS
        self.tibial_db = TIBIAL_IMPLANTS

    def match_femoral(
        self,
        patient_ml_width_mm: float,
        patient_ap_depth_mm: float,
        top_n: int = 5,
    ) -> list[ImplantMatch]:
        """Match patient femoral measurements to implant candidates."""
        return self._match_component(
            patient_ml_width_mm,
            patient_ap_depth_mm,
            self.femoral_db,
            top_n,
        )

    def match_tibial(
        self,
        patient_ml_width_mm: float,
        patient_ap_depth_mm: float,
        top_n: int = 5,
    ) -> list[ImplantMatch]:
        """Match patient tibial measurements to implant candidates."""
        return self._match_component(
            patient_ml_width_mm,
            patient_ap_depth_mm,
            self.tibial_db,
            top_n,
        )

    def match_all(
        self,
        measurements: dict,
        top_n: int = 5,
    ) -> dict:
        """
        Match both femoral and tibial implants using measurement engine output.
        """
        results = {}

        # Femoral matching
        femur = measurements.get("femur", {})
        if "condyle_ml_width_mm" in femur and "condyle_ap_depth_mm" in femur:
            results["femoral"] = self.match_femoral(
                femur["condyle_ml_width_mm"],
                femur["condyle_ap_depth_mm"],
                top_n,
            )

        # Tibial matching
        tibia = measurements.get("tibia", {})
        if "plateau_ml_width_mm" in tibia and "plateau_ap_depth_mm" in tibia:
            results["tibial"] = self.match_tibial(
                tibia["plateau_ml_width_mm"],
                tibia["plateau_ap_depth_mm"],
                top_n,
            )

        return results

    def _match_component(
        self,
        patient_ml: float,
        patient_ap: float,
        implant_db: list[ImplantSpec],
        top_n: int,
    ) -> list[ImplantMatch]:
        """Core matching logic for any component type."""
        matches = []

        for implant in implant_db:
            ml_diff = implant.ml_width_mm - patient_ml
            ap_diff = implant.ap_depth_mm - patient_ap

            ml_coverage = (implant.ml_width_mm / patient_ml * 100) if patient_ml > 0 else 0
            ap_coverage = (implant.ap_depth_mm / patient_ap * 100) if patient_ap > 0 else 0

            # Scoring: penalize both overhang and underhang, but overhang more
            ml_penalty = abs(ml_diff) * (1.5 if ml_diff > 0 else 1.0)
            ap_penalty = abs(ap_diff) * (1.5 if ap_diff > 0 else 1.0)

            # Combined score (0-100, higher = better fit)
            max_penalty = 20.0  # mm combined mismatch for 0 score
            total_penalty = ml_penalty + ap_penalty
            score = max(0, 100 - (total_penalty / max_penalty * 100))

            # Categorize fit
            fit_category = self._categorize_fit(ml_diff, ap_diff)

            # Generate explanation
            explanation = self._generate_explanation(
                implant, patient_ml, patient_ap, ml_diff, ap_diff, fit_category
            )

            matches.append(ImplantMatch(
                implant=implant,
                score=round(score, 1),
                ml_mismatch_mm=round(ml_diff, 1),
                ap_mismatch_mm=round(ap_diff, 1),
                ml_coverage_pct=round(ml_coverage, 1),
                ap_coverage_pct=round(ap_coverage, 1),
                fit_category=fit_category,
                explanation=explanation,
            ))

        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_n]

    def _categorize_fit(self, ml_diff: float, ap_diff: float) -> str:
        """Categorize the implant fit based on dimensional mismatch."""
        max_overhang = max(ml_diff, ap_diff)
        max_underhang = min(ml_diff, ap_diff)
        abs_max = max(abs(ml_diff), abs(ap_diff))

        if abs_max <= self.OPTIMAL_TOLERANCE_MM:
            return "optimal"
        elif abs_max <= self.ACCEPTABLE_TOLERANCE_MM:
            return "acceptable"
        elif max_overhang > self.OVERHANG_THRESHOLD_MM:
            return "overhang_risk"
        else:
            return "underhang_risk"

    def _generate_explanation(
        self,
        implant: ImplantSpec,
        patient_ml: float,
        patient_ap: float,
        ml_diff: float,
        ap_diff: float,
        fit_category: str,
    ) -> str:
        """Generate a clinical explanation for the match."""
        parts = []

        parts.append(
            f"{implant.manufacturer} {implant.system} Size {implant.size}: "
            f"ML={implant.ml_width_mm}mm, AP={implant.ap_depth_mm}mm."
        )

        # ML analysis
        if abs(ml_diff) <= self.OPTIMAL_TOLERANCE_MM:
            parts.append(f"ML width matches well ({ml_diff:+.1f}mm).")
        elif ml_diff > 0:
            parts.append(
                f"ML width {ml_diff:.1f}mm wider than bone - "
                f"{'minor' if ml_diff <= self.ACCEPTABLE_TOLERANCE_MM else 'significant'} "
                f"overhang risk on soft tissue."
            )
        else:
            parts.append(
                f"ML width {abs(ml_diff):.1f}mm narrower than bone - "
                f"incomplete medial-lateral coverage."
            )

        # AP analysis
        if abs(ap_diff) <= self.OPTIMAL_TOLERANCE_MM:
            parts.append(f"AP depth matches well ({ap_diff:+.1f}mm).")
        elif ap_diff > 0:
            parts.append(
                f"AP depth {ap_diff:.1f}mm larger - potential anterior notching risk."
            )
        else:
            parts.append(
                f"AP depth {abs(ap_diff):.1f}mm smaller - posterior condyle may not be fully covered."
            )

        # Overall assessment
        if fit_category == "optimal":
            parts.append("Overall: Excellent dimensional match.")
        elif fit_category == "acceptable":
            parts.append("Overall: Acceptable fit within clinical tolerance.")
        elif fit_category == "overhang_risk":
            parts.append("Overall: Risk of soft tissue impingement from overhang.")
        else:
            parts.append("Overall: Undercoverage may compromise fixation.")

        return " ".join(parts)

    def get_all_implants(self) -> dict:
        """Return the full implant database for display."""
        return {
            "femoral": [imp.model_dump() for imp in self.femoral_db],
            "tibial": [imp.model_dump() for imp in self.tibial_db],
            "manufacturers": list(set(
                imp.manufacturer for imp in self.femoral_db + self.tibial_db
            )),
            "total_count": len(self.femoral_db) + len(self.tibial_db),
        }
