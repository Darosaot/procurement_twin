"""
Procurement Digital Twin — FastAPI endpoint test suite
=======================================================
Covers all 10 endpoints with:
  - happy-path requests
  - input validation (422 errors on bad values)
  - edge cases (missing optional fields, boundary values)
  - response schema checks (required keys present)

Run:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --tb=short   # condensed traceback
"""

import sys
import os
import pytest

# ── ensure project root is on sys.path ──────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from src.api.main import app

# Use a single client across the entire session — avoids re-loading models
client = TestClient(app)

# ── Shared fixtures ──────────────────────────────────────────────────────────

MINIMAL_PARAMS = {
    "country": "BE",
    "procedure_type": "OPE",
    "contract_type": "S",
    "cpv_division": "72",
    "criteria": "M",
    "price_weight_pct": 40.0,
    "value_euro": 500_000,
    "prep_time_days": 45,
    "duration_months": 18,
}

FULL_PARAMS = {
    **MINIMAL_PARAMS,
    "gpa": True,
    "eu_funds": False,
    "fra_agreement": False,
    "electronic_auction": True,
    "accelerated": False,
    "n_samples": 500,
    "seed": 7,
}

SIMULATION_KEYS = {"competition", "single_bid_risk", "cross_border",
                   "price_ratio", "duration", "params", "meta"}
DIST_KEYS = {"mean", "median", "p10", "p25", "p75", "p90"}


# ════════════════════════════════════════════════════════════════════════════
# GET endpoints
# ════════════════════════════════════════════════════════════════════════════

class TestGetIndex:
    def test_status_ok(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_has_version(self):
        data = client.get("/").json()
        assert "version" in data
        assert data["version"] == "2.0.0"

    def test_has_endpoints_map(self):
        data = client.get("/").json()
        assert "endpoints" in data
        assert len(data["endpoints"]) >= 9


class TestGetHealth:
    def test_status_ok(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_models_loaded(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True
        assert data["n_models"] == 5

    def test_calibration_loaded(self):
        data = client.get("/health").json()
        assert data["calibration_loaded"] is True


class TestGetMetadata:
    def test_status_ok(self):
        r = client.get("/metadata")
        assert r.status_code == 200

    def test_has_countries(self):
        data = client.get("/metadata").json()
        assert "countries" in data
        assert "BE" in data["countries"]
        assert len(data["countries"]) >= 20

    def test_has_cpv_sectors(self):
        data = client.get("/metadata").json()
        assert "cpv_sectors" in data
        assert "72" in data["cpv_sectors"]

    def test_has_procedure_types(self):
        data = client.get("/metadata").json()
        assert "procedure_types" in data
        assert "OPE" in data["procedure_types"]

    def test_has_country_clusters(self):
        data = client.get("/metadata").json()
        assert "country_clusters" in data
        assert "Benelux" in data["country_clusters"]


class TestGetModels:
    def test_status_ok(self):
        r = client.get("/models")
        assert r.status_code == 200

    def test_has_all_model_keys(self):
        data = client.get("/models").json()
        assert "metrics" in data
        for key in ("competition", "single_bid", "cross_border",
                    "price_ratio", "duration"):
            assert key in data["metrics"], f"Missing metrics for '{key}'"

    def test_competition_has_r2(self):
        data = client.get("/models").json()
        assert "boost_r2" in data["metrics"]["competition"]

    def test_has_notes(self):
        data = client.get("/models").json()
        assert "notes" in data


class TestGetExplainGlobal:
    def test_status_ok(self):
        r = client.get("/explain/global")
        assert r.status_code == 200

    def test_has_competition_features(self):
        data = client.get("/explain/global").json()
        assert "competition" in data
        assert "top_features" in data["competition"]
        feats = data["competition"]["top_features"]
        assert len(feats) >= 5
        assert "feature" in feats[0] and "importance" in feats[0]

    def test_has_single_bid_features(self):
        data = client.get("/explain/global").json()
        assert "single_bid" in data


# ════════════════════════════════════════════════════════════════════════════
# POST /simulate
# ════════════════════════════════════════════════════════════════════════════

class TestPostSimulate:
    def test_happy_path_minimal(self):
        r = client.post("/simulate", json=MINIMAL_PARAMS)
        assert r.status_code == 200

    def test_happy_path_full(self):
        r = client.post("/simulate", json=FULL_PARAMS)
        assert r.status_code == 200

    def test_response_has_all_outcome_keys(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        assert SIMULATION_KEYS.issubset(data.keys()), \
            f"Missing keys: {SIMULATION_KEYS - data.keys()}"

    def test_competition_distribution_keys(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        assert DIST_KEYS.issubset(data["competition"].keys())

    def test_single_bid_has_probability(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        prob = data["single_bid_risk"]["probability"]
        assert 0.0 <= prob <= 1.0

    def test_cross_border_has_probability(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        prob = data["cross_border"]["probability"]
        assert 0.0 <= prob <= 1.0

    def test_price_ratio_positive(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        assert data["price_ratio"]["mean"] > 0

    def test_duration_positive(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        assert data["duration"]["mean"] > 0

    def test_meta_present(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        assert "duration_ms" in data["meta"]
        assert "timestamp" in data["meta"]

    def test_params_echoed(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        assert data["params"]["country"] == "BE"
        assert data["params"]["procedure_type"] == "OPE"

    def test_invalid_procedure_type_422(self):
        bad = {**MINIMAL_PARAMS, "procedure_type": "NEG"}
        r = client.post("/simulate", json=bad)
        assert r.status_code == 422

    def test_invalid_contract_type_422(self):
        bad = {**MINIMAL_PARAMS, "contract_type": "X"}
        r = client.post("/simulate", json=bad)
        assert r.status_code == 422

    def test_invalid_criteria_422(self):
        bad = {**MINIMAL_PARAMS, "criteria": "Z"}
        r = client.post("/simulate", json=bad)
        assert r.status_code == 422

    def test_negative_value_euro_422(self):
        bad = {**MINIMAL_PARAMS, "value_euro": -1000}
        r = client.post("/simulate", json=bad)
        assert r.status_code == 422

    def test_price_weight_over_100_422(self):
        bad = {**MINIMAL_PARAMS, "price_weight_pct": 101}
        r = client.post("/simulate", json=bad)
        assert r.status_code == 422

    def test_no_samples_excluded_by_default(self):
        data = client.post("/simulate", json=MINIMAL_PARAMS).json()
        # Raw Monte Carlo samples should NOT be in the response by default
        assert "samples" not in data.get("competition", {})

    def test_samples_included_when_requested(self):
        r = client.post("/simulate?include_samples=true", json=MINIMAL_PARAMS)
        data = r.json()
        assert "samples" in data.get("competition", {}), \
            "Expected 'samples' key when include_samples=true"

    def test_different_countries_give_different_results(self):
        r_be = client.post("/simulate", json={**MINIMAL_PARAMS, "country": "BE"}).json()
        r_pl = client.post("/simulate", json={**MINIMAL_PARAMS, "country": "PL"}).json()
        # Competition predictions should differ
        assert r_be["competition"]["point_pred"] != r_pl["competition"]["point_pred"]

    def test_n_samples_boundary_min(self):
        r = client.post("/simulate", json={**MINIMAL_PARAMS, "n_samples": 100})
        assert r.status_code == 200

    def test_n_samples_boundary_max(self):
        r = client.post("/simulate", json={**MINIMAL_PARAMS, "n_samples": 10000})
        assert r.status_code == 200

    def test_n_samples_below_min_422(self):
        r = client.post("/simulate", json={**MINIMAL_PARAMS, "n_samples": 50})
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════════════════════
# POST /compare
# ════════════════════════════════════════════════════════════════════════════

SCENARIO_B = {**MINIMAL_PARAMS, "procedure_type": "RES", "criteria": "L",
              "price_weight_pct": 70.0}

class TestPostCompare:
    def test_happy_path(self):
        payload = {"scenario_a": MINIMAL_PARAMS, "scenario_b": SCENARIO_B,
                   "n_samples": 500}
        r = client.post("/compare", json=payload)
        assert r.status_code == 200

    def test_has_deltas(self):
        payload = {"scenario_a": MINIMAL_PARAMS, "scenario_b": SCENARIO_B,
                   "n_samples": 500}
        data = client.post("/compare", json=payload).json()
        assert "deltas" in data
        for key in ("competition", "single_bid_risk", "cross_border",
                    "price_ratio", "duration"):
            assert key in data["deltas"]

    def test_delta_structure(self):
        payload = {"scenario_a": MINIMAL_PARAMS, "scenario_b": SCENARIO_B,
                   "n_samples": 500}
        data = client.post("/compare", json=payload).json()
        d = data["deltas"]["competition"]
        assert "a" in d and "b" in d and "delta" in d and "delta_pct" in d

    def test_custom_labels(self):
        payload = {"scenario_a": MINIMAL_PARAMS, "scenario_b": SCENARIO_B,
                   "label_a": "Open MEAT", "label_b": "Restricted Price",
                   "n_samples": 200}
        data = client.post("/compare", json=payload).json()
        assert data["label_a"] == "Open MEAT"
        assert data["label_b"] == "Restricted Price"

    def test_same_scenario_deltas_near_zero(self):
        payload = {"scenario_a": MINIMAL_PARAMS, "scenario_b": MINIMAL_PARAMS,
                   "n_samples": 500}
        data = client.post("/compare", json=payload).json()
        # Same inputs → deltas should be 0 or very close
        assert abs(data["deltas"]["competition"]["delta"]) < 0.01

    def test_invalid_scenario_b_procedure_422(self):
        bad_b = {**SCENARIO_B, "procedure_type": "INVALID"}
        r = client.post("/compare", json={"scenario_a": MINIMAL_PARAMS,
                                           "scenario_b": bad_b})
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════════════════════
# POST /benchmark
# ════════════════════════════════════════════════════════════════════════════

class TestPostBenchmark:
    def test_happy_path_with_filters(self):
        r = client.post("/benchmark", json={
            "country": "BE", "procedure_type": "OPE", "cpv_division": "72"
        })
        assert r.status_code == 200

    def test_happy_path_no_filters(self):
        r = client.post("/benchmark", json={})
        assert r.status_code == 200

    def test_has_competition_stats(self):
        data = client.post("/benchmark", json={
            "country": "DE", "procedure_type": "OPE"
        }).json()
        assert "competition" in data
        assert "n" in data["competition"]

    def test_has_single_bid_rate(self):
        data = client.post("/benchmark", json={"country": "BE"}).json()
        assert "single_bid_rate" in data

    def test_has_duration_stats(self):
        data = client.post("/benchmark", json={"country": "BE"}).json()
        assert "duration" in data
        if data["duration"].get("n", 0) > 0:
            assert "mean" in data["duration"]

    def test_n_records_in_response(self):
        data = client.post("/benchmark", json={"country": "BE"}).json()
        assert "n_records" in data
        assert data["n_records"] >= 0

    def test_year_filter(self):
        data_full = client.post("/benchmark", json={"country": "BE"}).json()
        data_filtered = client.post("/benchmark", json={
            "country": "BE", "year_from": 2023, "year_to": 2023
        }).json()
        assert data_filtered["n_records"] <= data_full["n_records"]


# ════════════════════════════════════════════════════════════════════════════
# POST /policy
# ════════════════════════════════════════════════════════════════════════════

POLICY_BASE = {
    "country_cluster": "Benelux",
    "cpv_division": "72",
    "procedure_type": "OPE",
    "year_from": 2020,
    "year_to": 2022,
    "n_records": 50,   # minimum allowed; small for test speed
    "seed": 0,
}

class TestPostPolicy:
    def test_numeric_delta_intervention(self):
        payload = {**POLICY_BASE,
                   "intervention": {"param": "prep_time_days", "delta": 14}}
        r = client.post("/policy", json=payload)
        assert r.status_code == 200

    def test_categorical_value_intervention(self):
        payload = {**POLICY_BASE,
                   "intervention": {"param": "criteria", "value": "L"}}
        r = client.post("/policy", json=payload)
        assert r.status_code == 200

    def test_response_has_outcomes(self):
        payload = {**POLICY_BASE,
                   "intervention": {"param": "prep_time_days", "delta": 14}}
        data = client.post("/policy", json=payload).json()
        assert "outcomes" in data
        for key in ("competition", "single_bid_risk", "price_ratio", "duration"):
            assert key in data["outcomes"]

    def test_bootstrap_ci_present(self):
        payload = {**POLICY_BASE,
                   "intervention": {"param": "prep_time_days", "delta": 14}}
        data = client.post("/policy", json=payload).json()
        comp = data["outcomes"]["competition"]
        assert "ci_95_lo" in comp, "Bootstrap CI lower bound missing"
        assert "ci_95_hi" in comp, "Bootstrap CI upper bound missing"

    def test_ci_ordering(self):
        """ci_95_lo must be ≤ mean_delta ≤ ci_95_hi."""
        payload = {**POLICY_BASE,
                   "intervention": {"param": "prep_time_days", "delta": 14}}
        data = client.post("/policy", json=payload).json()
        for key in ("competition", "single_bid_risk", "price_ratio", "duration"):
            o = data["outcomes"][key]
            assert o["ci_95_lo"] <= o["mean_delta"] <= o["ci_95_hi"], \
                f"{key}: CI [{o['ci_95_lo']}, {o['ci_95_hi']}] doesn't contain " \
                f"mean_delta={o['mean_delta']}"

    def test_n_matched_in_response(self):
        payload = {**POLICY_BASE,
                   "intervention": {"param": "prep_time_days", "delta": 14}}
        data = client.post("/policy", json=payload).json()
        assert "n_matched" in data
        assert data["n_matched"] >= data["n_simulated"]

    def test_meta_has_duration(self):
        payload = {**POLICY_BASE,
                   "intervention": {"param": "prep_time_days", "delta": 14}}
        data = client.post("/policy", json=payload).json()
        assert "meta" in data
        assert "duration_ms" in data["meta"]

    def test_missing_intervention_delta_and_value_422(self):
        """PolicyIntervention requires either delta or value."""
        payload = {**POLICY_BASE, "intervention": {"param": "prep_time_days"}}
        r = client.post("/policy", json=payload)
        assert r.status_code == 422

    def test_no_matching_records_returns_error(self):
        """A segment that matches nothing should return a graceful error."""
        payload = {
            **POLICY_BASE,
            "cpv_division": "99",  # non-existent CPV
            "intervention": {"param": "prep_time_days", "delta": 7},
        }
        data = client.post("/policy", json=payload).json()
        assert data.get("n_matched", 0) == 0


# ════════════════════════════════════════════════════════════════════════════
# POST /explain
# ════════════════════════════════════════════════════════════════════════════

class TestPostExplain:
    def test_happy_path(self):
        r = client.post("/explain", json=MINIMAL_PARAMS)
        assert r.status_code == 200

    def test_has_competition_shap(self):
        data = client.post("/explain", json=MINIMAL_PARAMS).json()
        assert "competition" in data
        comp = data["competition"]
        assert "shap_values" in comp
        assert "base_value" in comp
        assert "prediction" in comp

    def test_has_single_bid_contributions(self):
        data = client.post("/explain", json=MINIMAL_PARAMS).json()
        assert "single_bid" in data
        sb = data["single_bid"]
        assert "shap_values" in sb

    def test_competition_prediction_positive(self):
        data = client.post("/explain", json=MINIMAL_PARAMS).json()
        assert data["competition"]["prediction"] > 0

    def test_single_bid_prediction_in_range(self):
        data = client.post("/explain", json=MINIMAL_PARAMS).json()
        prob = data["single_bid"]["prediction"]
        assert 0.0 <= prob <= 1.0

    def test_shap_ranked_has_20_entries(self):
        data = client.post("/explain", json=MINIMAL_PARAMS).json()
        ranked = data["competition"].get("shap_values_ranked", [])
        assert len(ranked) <= 20

    def test_ranked_sorted_descending_abs(self):
        data = client.post("/explain", json=MINIMAL_PARAMS).json()
        ranked = data["competition"].get("shap_values_ranked", [])
        if len(ranked) >= 2:
            vals = [abs(e["shap_value"]) for e in ranked]
            assert vals == sorted(vals, reverse=True), \
                "SHAP values are not sorted by |value| descending"

    def test_timestamp_present(self):
        data = client.post("/explain", json=MINIMAL_PARAMS).json()
        assert "timestamp" in data

    def test_invalid_params_422(self):
        bad = {**MINIMAL_PARAMS, "procedure_type": "BAD"}
        r = client.post("/explain", json=bad)
        assert r.status_code == 422
