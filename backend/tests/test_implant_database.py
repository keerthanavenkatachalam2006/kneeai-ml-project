"""
Audit tests for the implant database.
Validates that all 40 implant specifications are internally consistent,
have plausible dimensions, and follow expected size progressions.
"""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

import pytest

from app.implant_database import (
    FEMORAL_IMPLANTS,
    TIBIAL_IMPLANTS,
    ImplantMatcher,
    ImplantSpec,
)


# ─── Database Integrity Tests ────────────────────────────────────────────────

class TestDatabaseCompleteness:
    """Verify the database has expected structure."""

    def test_total_implant_count(self):
        """Should have exactly 40 implants (20 femoral + 20 tibial)."""
        assert len(FEMORAL_IMPLANTS) == 20
        assert len(TIBIAL_IMPLANTS) == 20

    def test_three_manufacturers(self):
        """Should cover 3 manufacturers."""
        fem_mfrs = set(imp.manufacturer for imp in FEMORAL_IMPLANTS)
        tib_mfrs = set(imp.manufacturer for imp in TIBIAL_IMPLANTS)

        assert "Stryker" in fem_mfrs
        assert "Zimmer Biomet" in fem_mfrs
        assert "Smith & Nephew" in fem_mfrs
        assert fem_mfrs == tib_mfrs  # Same manufacturers for both components

    def test_all_have_source(self):
        """Every implant must have a non-empty source citation."""
        for imp in FEMORAL_IMPLANTS + TIBIAL_IMPLANTS:
            assert imp.source, f"{imp.manufacturer} {imp.system} Size {imp.size} missing source"
            assert "approximate" in imp.source.lower(), (
                f"{imp.manufacturer} {imp.system} Size {imp.size} source should "
                f"indicate approximate values"
            )

    def test_all_have_component_type(self):
        """All femoral implants labeled 'femoral', all tibial labeled 'tibial'."""
        for imp in FEMORAL_IMPLANTS:
            assert imp.component == "femoral"
        for imp in TIBIAL_IMPLANTS:
            assert imp.component == "tibial"


class TestDimensionPlausibility:
    """Verify implant dimensions are within physiologically plausible ranges."""

    def test_femoral_ml_width_range(self):
        """Femoral ML width should be 50-85mm (covers small to large adults)."""
        for imp in FEMORAL_IMPLANTS:
            assert 50 <= imp.ml_width_mm <= 85, (
                f"{imp.manufacturer} {imp.system} Size {imp.size}: "
                f"ML={imp.ml_width_mm}mm outside 50-85mm range"
            )

    def test_femoral_ap_depth_range(self):
        """Femoral AP depth should be 45-80mm."""
        for imp in FEMORAL_IMPLANTS:
            assert 45 <= imp.ap_depth_mm <= 80, (
                f"{imp.manufacturer} {imp.system} Size {imp.size}: "
                f"AP={imp.ap_depth_mm}mm outside 45-80mm range"
            )

    def test_tibial_ml_width_range(self):
        """Tibial ML width should be 55-90mm."""
        for imp in TIBIAL_IMPLANTS:
            assert 55 <= imp.ml_width_mm <= 90, (
                f"{imp.manufacturer} {imp.system} Size {imp.size}: "
                f"ML={imp.ml_width_mm}mm outside 55-90mm range"
            )

    def test_tibial_ap_depth_range(self):
        """Tibial AP depth should be 40-70mm."""
        for imp in TIBIAL_IMPLANTS:
            assert 40 <= imp.ap_depth_mm <= 70, (
                f"{imp.manufacturer} {imp.system} Size {imp.size}: "
                f"AP={imp.ap_depth_mm}mm outside 40-70mm range"
            )

    def test_femoral_ml_greater_than_ap(self):
        """
        In most designs, femoral ML width >= AP depth (condyle shape).
        Allow small exceptions for narrow designs.
        """
        exceptions = 0
        for imp in FEMORAL_IMPLANTS:
            if imp.ml_width_mm < imp.ap_depth_mm - 2:
                exceptions += 1
        # At most 25% can violate this
        assert exceptions < len(FEMORAL_IMPLANTS) * 0.25

    def test_tibial_ml_greater_than_ap(self):
        """Tibial tray ML width should always exceed AP depth."""
        for imp in TIBIAL_IMPLANTS:
            assert imp.ml_width_mm > imp.ap_depth_mm, (
                f"{imp.manufacturer} {imp.system} Size {imp.size}: "
                f"ML={imp.ml_width_mm} should exceed AP={imp.ap_depth_mm}"
            )


class TestSizeProgression:
    """Verify that sizes increase monotonically within each system."""

    def _get_system_implants(self, implants: list[ImplantSpec], manufacturer: str, system: str):
        return [imp for imp in implants if imp.manufacturer == manufacturer and imp.system == system]

    def test_stryker_femoral_progression(self):
        """Stryker Triathlon femoral sizes should increase monotonically."""
        implants = self._get_system_implants(FEMORAL_IMPLANTS, "Stryker", "Triathlon")
        for i in range(len(implants) - 1):
            assert implants[i].ml_width_mm < implants[i + 1].ml_width_mm, (
                f"Stryker femoral: Size {implants[i].size} ML >= Size {implants[i+1].size} ML"
            )
            assert implants[i].ap_depth_mm < implants[i + 1].ap_depth_mm, (
                f"Stryker femoral: Size {implants[i].size} AP >= Size {implants[i+1].size} AP"
            )

    def test_zimmer_femoral_progression(self):
        """Zimmer NexGen femoral sizes should increase monotonically."""
        implants = self._get_system_implants(FEMORAL_IMPLANTS, "Zimmer Biomet", "NexGen")
        for i in range(len(implants) - 1):
            assert implants[i].ml_width_mm < implants[i + 1].ml_width_mm
            assert implants[i].ap_depth_mm < implants[i + 1].ap_depth_mm

    def test_sn_femoral_progression(self):
        """Smith & Nephew LEGION femoral sizes should increase monotonically."""
        implants = self._get_system_implants(FEMORAL_IMPLANTS, "Smith & Nephew", "LEGION")
        for i in range(len(implants) - 1):
            assert implants[i].ml_width_mm < implants[i + 1].ml_width_mm
            assert implants[i].ap_depth_mm < implants[i + 1].ap_depth_mm

    def test_stryker_tibial_progression(self):
        """Stryker Triathlon tibial sizes should increase monotonically."""
        implants = self._get_system_implants(TIBIAL_IMPLANTS, "Stryker", "Triathlon")
        for i in range(len(implants) - 1):
            assert implants[i].ml_width_mm < implants[i + 1].ml_width_mm
            assert implants[i].ap_depth_mm < implants[i + 1].ap_depth_mm

    def test_size_increments_reasonable(self):
        """Size increments should be 2-5mm per size step (typical for TKA)."""
        for manufacturer in ["Stryker", "Zimmer Biomet", "Smith & Nephew"]:
            system = {"Stryker": "Triathlon", "Zimmer Biomet": "NexGen", "Smith & Nephew": "LEGION"}[manufacturer]
            implants = self._get_system_implants(FEMORAL_IMPLANTS, manufacturer, system)
            for i in range(len(implants) - 1):
                ml_step = implants[i + 1].ml_width_mm - implants[i].ml_width_mm
                assert 1.5 <= ml_step <= 6.0, (
                    f"{manufacturer} {system}: ML step {ml_step}mm between "
                    f"size {implants[i].size} and {implants[i+1].size}"
                )


# ─── Matching Algorithm Tests ────────────────────────────────────────────────

class TestImplantMatching:
    """Verify the matching algorithm produces correct rankings."""

    @pytest.fixture
    def matcher(self):
        return ImplantMatcher()

    def test_exact_match_scores_100(self, matcher):
        """An implant matching patient dimensions exactly should score ~100."""
        # Stryker Triathlon Size 4: ML=63, AP=62
        matches = matcher.match_femoral(63.0, 62.0, top_n=20)
        top = matches[0]
        assert top.score >= 90, f"Exact match scored only {top.score}"
        assert top.fit_category == "optimal"

    def test_large_mismatch_scores_low(self, matcher):
        """A 20mm mismatch should score near 0."""
        matches = matcher.match_femoral(40.0, 40.0, top_n=20)
        # All implants are much larger than 40mm patient
        for m in matches:
            assert m.score < 30, f"Large mismatch scored {m.score}"

    def test_overhang_penalized_more(self, matcher):
        """Implant wider than bone should score lower than implant narrower by same amount."""
        # Patient ML=65, AP=65
        matches = matcher.match_femoral(65.0, 65.0, top_n=20)

        # Find one with +3mm overhang vs one with -3mm underhang
        overhang_scores = [m.score for m in matches if m.ml_mismatch_mm > 2.5]
        underhang_scores = [m.score for m in matches if m.ml_mismatch_mm < -2.5]

        if overhang_scores and underhang_scores:
            # Overhang should generally score lower than equivalent underhang
            # (due to 1.5x penalty multiplier)
            assert min(underhang_scores) >= min(overhang_scores) - 5

    def test_ranking_order(self, matcher):
        """Results should be sorted by score descending."""
        matches = matcher.match_femoral(65.0, 64.0, top_n=10)
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limit(self, matcher):
        """Should return at most top_n results."""
        matches = matcher.match_femoral(65.0, 64.0, top_n=3)
        assert len(matches) == 3

    def test_fit_categories_correct(self, matcher):
        """Verify fit category assignment."""
        # Exact match → optimal
        matches = matcher.match_femoral(63.0, 62.0, top_n=1)
        assert matches[0].fit_category == "optimal"

    def test_coverage_percentage_correct(self, matcher):
        """Coverage % = implant_dim / patient_dim × 100."""
        matches = matcher.match_femoral(60.0, 60.0, top_n=20)
        for m in matches:
            expected_ml_cov = m.implant.ml_width_mm / 60.0 * 100
            assert abs(m.ml_coverage_pct - expected_ml_cov) < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
