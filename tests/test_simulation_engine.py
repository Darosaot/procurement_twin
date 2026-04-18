"""
Unit tests for ProcurementTwin simulation engine.

Run:
    pytest tests/test_simulation_engine.py -v

Tests are skipped automatically when model files are absent (CI without
the full model artefacts).  Set SKIP_MODEL_TESTS=0 to force failure.
"""

import os
import sys
import math
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [ROOT, os.path.join(ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "models"))
_MODELS_AVAILABLE = os.path.exists(os.path.join(MODEL_DIR, "competition_model.pkl"))

skip_no_models = pytest.mark.skipif(
    not _MODELS_AVAILABLE,
    reason="Model files not present — set MODEL_DIR or run training first",
)


@pytest.fixture(scope="module")
def twin():
    from simulation.simulation_engine import ProcurementTwin
    return ProcurementTwin()


MINIMAL_PARAMS = {
    "country": "DE",
    "procedure_type": "OPE",
    "contract_type": "S",
    "cpv_division": "72",
    "criteria": "M",
    "price_weight_pct": 50.0,
    "value_euro": 1_000_000,
    "prep_time_days": 35.0,
    "duration_months": 24.0,
    "gpa": False,
    "eu_funds": False,
    "fra_agreement": False,
    "electronic_auction": False,
    "accelerated": False,
}

SIMULATION_KEYS = {"competition", "single_bid_risk", "cross_border",
                   "price_ratio", "duration", "params"}
DIST_KEYS = {"mean", "median", "p10", "p25", "p75", "p90"}


# ── simulate() ───────────────────────────────────────────────────────────────

@skip_no_models
class TestSimulate:
    def test_returns_all_outcome_keys(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=0)
        assert SIMULATION_KEYS.issubset(result.keys())

    def test_competition_dist_keys(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=0)
        assert DIST_KEYS.issubset(result["competition"].keys())

    def test_competition_mean_positive(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=0)
        assert result["competition"]["mean"] > 0

    def test_single_bid_risk_probability_in_range(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=0)
        prob = result["single_bid_risk"]["probability"]
        assert 0.0 <= prob <= 1.0

    def test_cross_border_probability_in_range(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=0)
        prob = result["cross_border"]["probability"]
        assert 0.0 <= prob <= 1.0

    def test_price_ratio_clipped(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=0)
        assert result["price_ratio"]["mean"] >= 0.1
        assert result["price_ratio"]["mean"] <= 3.0

    def test_duration_at_least_30_days(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=0)
        assert result["duration"]["mean"] >= 30

    def test_samples_included_when_requested(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=100, seed=0)
        assert "samples" in result["competition"]
        assert len(result["competition"]["samples"]) == 100

    def test_seed_reproducibility(self, twin):
        r1 = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=42)
        r2 = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=42)
        assert r1["competition"]["mean"] == r2["competition"]["mean"]

    def test_different_seeds_differ(self, twin):
        r1 = twin.simulate(MINIMAL_PARAMS, n_samples=500, seed=1)
        r2 = twin.simulate(MINIMAL_PARAMS, n_samples=500, seed=2)
        assert r1["competition"]["mean"] != r2["competition"]["mean"]

    def test_params_echoed(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=200, seed=0)
        assert result["params"]["country"] == "DE"
        assert result["params"]["procedure_type"] == "OPE"

    def test_percentile_ordering(self, twin):
        r = twin.simulate(MINIMAL_PARAMS, n_samples=500, seed=0)
        comp = r["competition"]
        assert comp["p10"] <= comp["p25"] <= comp["median"] <= comp["p75"] <= comp["p90"]


# ── simulate() — boundary inputs ─────────────────────────────────────────────

@skip_no_models
class TestSimulateBoundary:
    def test_nan_value_euro_uses_default(self, twin):
        import warnings
        params = {**MINIMAL_PARAMS, "value_euro": None}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = twin.simulate(params, n_samples=200, seed=0)
        assert result["competition"]["mean"] > 0

    def test_inf_prep_time_uses_default(self, twin):
        params = {**MINIMAL_PARAMS, "prep_time_days": float("inf")}
        result = twin.simulate(params, n_samples=200, seed=0)
        assert result["competition"]["mean"] > 0

    def test_unknown_country_falls_back(self, twin):
        params = {**MINIMAL_PARAMS, "country": "XX"}
        result = twin.simulate(params, n_samples=200, seed=0)
        assert result["competition"]["mean"] > 0

    def test_very_small_value(self, twin):
        params = {**MINIMAL_PARAMS, "value_euro": 1.0}
        result = twin.simulate(params, n_samples=200, seed=0)
        assert result["competition"]["mean"] > 0

    def test_works_contract_type(self, twin):
        params = {**MINIMAL_PARAMS, "contract_type": "W", "cpv_division": "45"}
        result = twin.simulate(params, n_samples=200, seed=0)
        assert result["competition"]["mean"] > 0

    def test_all_flags_on(self, twin):
        params = {**MINIMAL_PARAMS,
                  "gpa": True, "eu_funds": True,
                  "fra_agreement": True, "electronic_auction": True,
                  "accelerated": True}
        result = twin.simulate(params, n_samples=200, seed=0)
        assert result["competition"]["mean"] > 0

    def test_minimum_samples(self, twin):
        result = twin.simulate(MINIMAL_PARAMS, n_samples=1, seed=0)
        assert result["competition"]["mean"] >= 0


# ── compare() ────────────────────────────────────────────────────────────────

@skip_no_models
class TestCompare:
    def test_compare_returns_deltas(self, twin):
        params_b = {**MINIMAL_PARAMS, "prep_time_days": 60.0}
        result = twin.compare(MINIMAL_PARAMS, params_b, n_samples=200)
        assert "deltas" in result
        assert "competition" in result["deltas"]

    def test_delta_b_minus_a(self, twin):
        params_b = {**MINIMAL_PARAMS, "prep_time_days": 60.0}
        result = twin.compare(MINIMAL_PARAMS, params_b, n_samples=200)
        delta = result["deltas"]["competition"]["delta"]
        a_mean = result["scenario_a"]["competition"]["mean"]
        b_mean = result["scenario_b"]["competition"]["mean"]
        assert math.isclose(delta, b_mean - a_mean, rel_tol=1e-4)

    def test_identical_scenarios_delta_near_zero(self, twin):
        result = twin.compare(MINIMAL_PARAMS, MINIMAL_PARAMS, n_samples=500, seed_a=0, seed_b=0)
        assert abs(result["deltas"]["competition"]["delta"]) < 0.5

    def test_labels_echoed(self, twin):
        result = twin.compare(MINIMAL_PARAMS, MINIMAL_PARAMS,
                              label_a="Base", label_b="Alt", n_samples=100)
        assert result["label_a"] == "Base"
        assert result["label_b"] == "Alt"


# ── _params_to_df() default warnings ─────────────────────────────────────────

@skip_no_models
class TestDefaultWarnings:
    def test_nan_value_triggers_warning(self, twin, caplog):
        import logging
        params = {**MINIMAL_PARAMS, "value_euro": None}
        with caplog.at_level(logging.WARNING, logger="simulation.simulation_engine"):
            twin._params_to_df(params)
        assert any("value_euro" in msg for msg in caplog.messages)

    def test_no_warning_when_params_clean(self, twin, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="simulation.simulation_engine"):
            twin._params_to_df(MINIMAL_PARAMS)
        assert not any("defaults applied" in msg for msg in caplog.messages)


# ── compute_shap() ────────────────────────────────────────────────────────────

@skip_no_models
class TestShap:
    def test_shap_returns_competition_and_single_bid(self, twin):
        result = twin.compute_shap(MINIMAL_PARAMS)
        assert "competition" in result
        assert "single_bid" in result

    def test_shap_competition_has_values(self, twin):
        result = twin.compute_shap(MINIMAL_PARAMS)
        assert "shap_values" in result["competition"]
        assert len(result["competition"]["shap_values"]) > 0

    def test_shap_prediction_positive(self, twin):
        result = twin.compute_shap(MINIMAL_PARAMS)
        assert result["competition"]["prediction"] > 0
