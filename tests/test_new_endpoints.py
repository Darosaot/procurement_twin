"""
Tests for Sprint 5+6 — infrastructure modules and new API endpoints.

Infrastructure tests (cache, history, auth) run without model files.
API endpoint tests are skipped when model files are absent.

Run:
    pytest tests/test_new_endpoints.py -v
"""

import os, sys, json
import pytest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [ROOT, os.path.join(ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Infrastructure modules (no models needed) ───────────────────────────────

class TestCache:
    def test_miss_then_hit(self):
        from api.cache import SimulationCache
        c = SimulationCache(maxsize=10, ttl=60)
        params = {"country": "DE", "n": 100}
        assert c.get(params) is None
        c.set(params, {"result": 42})
        assert c.get(params) == {"result": 42}

    def test_eviction_at_maxsize(self):
        from api.cache import SimulationCache
        c = SimulationCache(maxsize=3, ttl=60)
        for i in range(5):
            c.set({"i": i}, {"v": i})
        assert c.stats["size"] == 3

    def test_lru_evicts_oldest(self):
        from api.cache import SimulationCache
        c = SimulationCache(maxsize=2, ttl=60)
        c.set({"k": 1}, {"v": 1})
        c.set({"k": 2}, {"v": 2})
        c.get({"k": 1})            # touch k=1 → k=2 is now LRU
        c.set({"k": 3}, {"v": 3}) # evicts k=2
        assert c.get({"k": 2}) is None
        assert c.get({"k": 1}) is not None

    def test_stats_hit_rate(self):
        from api.cache import SimulationCache
        c = SimulationCache(maxsize=10, ttl=60)
        c.set({"k": 1}, {"v": 1})
        c.get({"k": 1})   # hit
        c.get({"k": 2})   # miss
        assert c.stats["hit_rate"] == 0.5
        assert c.stats["hits"] == 1
        assert c.stats["misses"] == 1

    def test_clear_resets(self):
        from api.cache import SimulationCache
        c = SimulationCache(maxsize=10, ttl=60)
        c.set({"k": 1}, {"v": 1})
        c.clear()
        assert c.stats["size"] == 0
        assert c.stats["hits"] == 0

    def test_invalidate(self):
        from api.cache import SimulationCache
        c = SimulationCache(maxsize=10, ttl=60)
        c.set({"k": 1}, {"v": 1})
        assert c.invalidate({"k": 1}) is True
        assert c.get({"k": 1}) is None

    def test_disabled_never_returns(self):
        from api import cache as cache_mod
        orig = cache_mod._ENABLED
        cache_mod._ENABLED = False
        try:
            c = cache_mod.SimulationCache(maxsize=10, ttl=60)
            c.set({"k": 1}, {"v": 1})
            assert c.get({"k": 1}) is None
        finally:
            cache_mod._ENABLED = orig


class TestHistoryModule:
    def test_record_and_retrieve(self):
        from api.history import SimulationHistory
        h = SimulationHistory(db_path=":memory:")
        h.record("/simulate", {"a": 1}, 123.4, 200)
        rows = h.recent(limit=10)
        assert len(rows) == 1
        assert rows[0]["endpoint"] == "/simulate"
        assert rows[0]["status_code"] == 200

    def test_cached_flag(self):
        from api.history import SimulationHistory
        h = SimulationHistory(db_path=":memory:")
        h.record("/simulate", {}, 50, 200, cached=True)
        rows = h.recent(limit=1)
        assert rows[0]["cached"] == 1

    def test_aggregate_stats(self):
        from api.history import SimulationHistory
        h = SimulationHistory(db_path=":memory:")
        h.record("/simulate", {}, 100, 200)
        h.record("/batch",    {}, 300, 200)
        stats = h.aggregate_stats()
        assert stats["total_runs"] == 2
        endpoints = [r["endpoint"] for r in stats["by_endpoint"]]
        assert "/simulate" in endpoints

    def test_max_rows_pruning(self):
        from api.history import SimulationHistory
        h = SimulationHistory(db_path=":memory:", max_rows=5)
        for i in range(10):
            h.record("/simulate", {"i": i}, 100, 200)
        assert h.aggregate_stats()["total_runs"] <= 5

    def test_endpoint_filter(self):
        from api.history import SimulationHistory
        h = SimulationHistory(db_path=":memory:")
        h.record("/simulate", {}, 100, 200)
        h.record("/batch",    {}, 200, 200)
        rows = h.recent(endpoint="/simulate")
        assert all(r["endpoint"] == "/simulate" for r in rows)

    def test_get_run_by_id(self):
        from api.history import SimulationHistory
        h = SimulationHistory(db_path=":memory:")
        h.record("/explain", {}, 80, 200)
        rows = h.recent(limit=1)
        run = h.get_run(rows[0]["id"])
        assert run is not None
        assert run["endpoint"] == "/explain"

    def test_missing_run_returns_none(self):
        from api.history import SimulationHistory
        h = SimulationHistory(db_path=":memory:")
        assert h.get_run(99999) is None


class TestAuth:
    def test_no_keys_auth_disabled(self):
        from api.auth import AUTH_ENABLED
        if not os.environ.get("API_KEYS"):
            assert not AUTH_ENABLED

    def test_valid_key_passes(self):
        import asyncio
        from api import auth as auth_mod
        orig_keys = auth_mod._API_KEYS
        orig_enabled = auth_mod.AUTH_ENABLED
        try:
            auth_mod._API_KEYS = frozenset({"test-key-abc"})
            auth_mod.AUTH_ENABLED = True
            result = asyncio.get_event_loop().run_until_complete(
                auth_mod.require_api_key("test-key-abc")
            )
            assert result == "test-key-abc"
        finally:
            auth_mod._API_KEYS = orig_keys
            auth_mod.AUTH_ENABLED = orig_enabled

    def test_invalid_key_raises(self):
        import asyncio
        from fastapi import HTTPException
        from api import auth as auth_mod
        orig_keys = auth_mod._API_KEYS
        orig_enabled = auth_mod.AUTH_ENABLED
        try:
            auth_mod._API_KEYS = frozenset({"test-key-abc"})
            auth_mod.AUTH_ENABLED = True
            with pytest.raises(HTTPException) as exc:
                asyncio.get_event_loop().run_until_complete(
                    auth_mod.require_api_key("wrong-key")
                )
            assert exc.value.status_code == 401
        finally:
            auth_mod._API_KEYS = orig_keys
            auth_mod.AUTH_ENABLED = orig_enabled


# ── API endpoint tests (require models) ─────────────────────────────────────

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "models"))
_MODELS_AVAILABLE = os.path.exists(os.path.join(MODEL_DIR, "competition_model.pkl"))
skip_no_models = pytest.mark.skipif(
    not _MODELS_AVAILABLE,
    reason="Model files not present",
)


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


BASE = {
    "country": "DE", "procedure_type": "OPE", "contract_type": "S",
    "cpv_division": "72", "criteria": "M", "price_weight_pct": 50.0,
    "value_euro": 1_000_000, "prep_time_days": 35.0, "duration_months": 24.0,
    "n_samples": 200, "seed": 42,
}


@skip_no_models
class TestStatusEndpoint:
    def test_returns_200(self, api_client):
        assert api_client.get("/status").status_code == 200

    def test_has_all_keys(self, api_client):
        d = api_client.get("/status").json()
        for key in ("version", "auth_enabled", "cache", "history"):
            assert key in d

    def test_version_is_3(self, api_client):
        assert api_client.get("/status").json()["version"] == "3.0.0"


@skip_no_models
class TestHistoryEndpoint:
    def test_returns_200(self, api_client):
        assert api_client.get("/history").status_code == 200

    def test_has_runs(self, api_client):
        assert "runs" in api_client.get("/history").json()

    def test_limit_enforced(self, api_client):
        r = api_client.get("/history?limit=5")
        assert r.json()["limit"] == 5

    def test_limit_too_large_rejected(self, api_client):
        assert api_client.get("/history?limit=9999").status_code == 422


@skip_no_models
class TestBatch:
    def test_single_procedure(self, api_client):
        r = api_client.post("/batch", json={"procedures": [BASE]})
        assert r.status_code == 200
        assert r.json()["n"] == 1

    def test_two_procedures_ordered(self, api_client):
        p1 = {**BASE, "cpv_division": "45", "seed": 1}
        p2 = {**BASE, "cpv_division": "72", "seed": 2}
        r = api_client.post("/batch", json={"procedures": [p1, p2]})
        assert r.status_code == 200
        assert r.json()["n"] == 2

    def test_too_many_rejected(self, api_client):
        assert api_client.post("/batch", json={"procedures": [BASE] * 21}).status_code == 422

    def test_empty_rejected(self, api_client):
        assert api_client.post("/batch", json={"procedures": []}).status_code == 422

    def test_has_duration_ms(self, api_client):
        r = api_client.post("/batch", json={"procedures": [BASE]})
        assert "duration_ms" in r.json()


@skip_no_models
class TestSensitivity:
    def test_explicit_values(self, api_client):
        r = api_client.post("/sensitivity", json={
            "base_params": BASE, "param": "prep_time_days", "values": [14.0, 35.0, 60.0],
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["outcomes"]["competition"]) == 3

    def test_n_steps(self, api_client):
        r = api_client.post("/sensitivity", json={
            "base_params": BASE, "param": "prep_time_days", "n_steps": 4,
        })
        assert r.status_code == 200
        assert len(r.json()["values_tested"]) == 4

    def test_invalid_param_rejected(self, api_client):
        assert api_client.post("/sensitivity", json={
            "base_params": BASE, "param": "country", "values": ["DE"],
        }).status_code == 422

    def test_all_outcomes_present(self, api_client):
        r = api_client.post("/sensitivity", json={
            "base_params": BASE, "param": "price_weight_pct", "n_steps": 3,
        })
        for key in ("competition", "single_bid_risk", "cross_border", "price_ratio", "duration"):
            assert key in r.json()["outcomes"]

    def test_too_many_values_rejected(self, api_client):
        assert api_client.post("/sensitivity", json={
            "base_params": BASE, "param": "prep_time_days",
            "values": [float(i) for i in range(21)],
        }).status_code == 422


@skip_no_models
class TestExport:
    SAMPLE = {"competition": {"mean": 3.2, "p25": 2.0}, "duration": {"mean": 120}}

    def test_json_export(self, api_client):
        r = api_client.post("/export", json={"data": self.SAMPLE, "format": "json", "filename": "t"})
        assert r.status_code == 200
        assert "application/json" in r.headers["content-type"]

    def test_csv_export(self, api_client):
        r = api_client.post("/export", json={"data": self.SAMPLE, "format": "csv", "filename": "t"})
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]

    def test_csv_flattens_nested(self, api_client):
        r = api_client.post("/export", json={"data": self.SAMPLE, "format": "csv", "filename": "t"})
        assert "competition.mean" in r.text

    def test_json_roundtrip(self, api_client):
        r = api_client.post("/export", json={"data": self.SAMPLE, "format": "json", "filename": "t"})
        assert json.loads(r.content)["competition"]["mean"] == 3.2

    def test_invalid_format_rejected(self, api_client):
        assert api_client.post("/export", json={"data": self.SAMPLE, "format": "xlsx", "filename": "f"}).status_code == 422
