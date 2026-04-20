"""
Procurement Digital Twin — FastAPI REST API
============================================
Exposes the simulation engine as a fully documented HTTP API, enabling
programmatic access and integration with eProcurement platforms.

Base URL (local):  http://localhost:8000
Interactive docs:  http://localhost:8000/docs   (Swagger UI)
Alternative docs:  http://localhost:8000/redoc  (ReDoc)

Endpoints
---------
GET  /                   API info and link index
GET  /health             Health check + model status
GET  /metadata           Reference lists (countries, CPVs, procedure types)
GET  /models             Model evaluation metrics from last training run
GET  /explain/global     Pre-computed global feature importances
POST /simulate           Single-procedure Monte Carlo simulation
POST /compare            Side-by-side comparison of two procedure designs
POST /benchmark          Empirical statistics from 1.1M historical records
POST /policy             Aggregate counterfactual policy simulation
POST /explain            Per-prediction SHAP feature contributions
"""

import sys, os, json, time, logging, asyncio, functools, csv, io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
import warnings
warnings.filterwarnings("ignore")

# ── Request timeout (seconds) — override via API_TIMEOUT_SECONDS env var ──
_REQUEST_TIMEOUT = int(os.environ.get("API_TIMEOUT_SECONDS", "120"))

# ── CORS — comma-separated origins via CORS_ORIGINS env var ───────────────
_DEFAULT_CORS = "http://localhost:8050,http://localhost:8888,http://127.0.0.1:8050,http://127.0.0.1:8888"
_CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", _DEFAULT_CORS).split(",") if o.strip()]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Path resolution (works from any working directory) ────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SRC_DIR     = os.path.join(_PROJECT_ROOT, "src")
for _p in [_PROJECT_ROOT, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _SLOWAPI = True
except ImportError:
    _SLOWAPI = False
    logger.warning("slowapi not installed — rate limiting disabled.")

from simulation.simulation_engine import (
    ProcurementTwin, COUNTRY_CLUSTERS, CPV_SECTORS, value_bracket
)
from api.auth    import require_api_key, AUTH_ENABLED
from api.cache   import simulation_cache
from api.metrics import track_simulation, record_cache_hit, record_cache_miss, prometheus_response
from api.history import simulation_history

try:
    from advisor.advisor import ProcurementAdvisor as _ProcurementAdvisor
    _advisor = _ProcurementAdvisor()
    _ADVISOR_AVAILABLE = True
except Exception as _adv_err:
    _advisor = None
    _ADVISOR_AVAILABLE = False
    logger.warning("Advisor module unavailable: %s", _adv_err)

# ── Rate limiter ──────────────────────────────────────────────────
_RATE_LIMIT = os.environ.get("RATE_LIMIT", "60/minute")
if _SLOWAPI:
    _limiter = Limiter(key_func=get_remote_address, default_limits=[_RATE_LIMIT])
else:
    _limiter = None

# ── App initialisation ────────────────────────────────────────────
app = FastAPI(
    title="Procurement Digital Twin API",
    description=(
        "REST API for the EU Procurement Digital Twin. "
        "Simulates competition, cross-border participation, price ratios and "
        "procedure duration from 1.1 million TED contract records (2018–2023). "
        "All simulation endpoints use calibrated Monte Carlo sampling.\n\n"
        "**Auth**: pass `X-API-Key: <key>` header when `API_KEYS` is configured.\n"
        "**Rate limit**: configurable via `RATE_LIMIT` env var (default 60/minute)."
    ),
    version="3.0.0",
    contact={"name": "Procurement Digital Twin"},
    license_info={"name": "Internal use"},
)
if _SLOWAPI and _limiter:
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


class _TimeoutMiddleware(BaseHTTPMiddleware):
    """Abort requests that exceed _REQUEST_TIMEOUT seconds."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=_REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Request timed out after %ds: %s %s",
                           _REQUEST_TIMEOUT, request.method, request.url.path)
            return Response(
                content='{"detail":"Request timed out. Try reducing n_records or n_samples."}',
                status_code=504,
                media_type="application/json",
            )


app.add_middleware(_TimeoutMiddleware)

# ── Load twin once at startup ─────────────────────────────────────
logger.info("Loading Procurement Twin models...")
_twin = ProcurementTwin()
_startup_time = datetime.utcnow().isoformat() + "Z"
logger.info("Models ready.")

# Dedicated executor for long-running blocking calls (/policy, /benchmark).
# Size configurable via POLICY_WORKERS env var; default 2 keeps memory bounded.
_blocking_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("POLICY_WORKERS", "2")),
    thread_name_prefix="twin-blocking",
)

# ─────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────

VALID_PROCEDURE_TYPES = {"OPE", "RES", "NIC", "COD", "INP", "AWP", "NOC"}
VALID_CONTRACT_TYPES  = {"S", "U", "W"}
VALID_CRITERIA        = {"M", "L"}
VALID_COUNTRIES       = frozenset(COUNTRY_CLUSTERS.keys())
VALID_CPV_DIVISIONS   = frozenset(CPV_SECTORS.keys())


class ProcedureParams(BaseModel):
    """Parameters that define a single procurement procedure."""

    country: str = Field(
        "DE",
        description="ISO 3166-1 alpha-2 country code of the contracting authority (e.g. 'DE', 'FR', 'PL').",
        examples=["DE", "FR", "PL", "IT", "ES"],
    )
    procedure_type: str = Field(
        "OPE",
        description="Procedure type code: OPE=Open, RES=Restricted, NIC=Negotiated with prior call, COD=Competitive dialogue, INP=Innovation partnership, AWP=Award without prior publication.",
    )
    contract_type: str = Field(
        "S",
        description="Contract type: S=Services, U=Supplies, W=Works.",
    )
    cpv_division: str = Field(
        "72",
        description="2-digit CPV (Common Procurement Vocabulary) division code (e.g. '45' for Construction, '72' for IT Services).",
    )
    criteria: str = Field(
        "M",
        description="Award criteria: M=MEAT (Most Economically Advantageous Tender), L=Lowest price only.",
    )
    price_weight_pct: float = Field(
        50.0, ge=0, le=100,
        description="Percentage weight given to price in MEAT evaluation (0–100). Ignored when criteria='L'.",
    )
    value_euro: Optional[float] = Field(
        1_000_000, gt=0,
        description="Estimated contract value in EUR.",
    )
    prep_time_days: float = Field(
        35.0, ge=1, le=365,
        description="Days between publication and submission deadline (EU minimum is 35 days for open procedures).",
    )
    duration_months: float = Field(
        24.0, ge=1, le=120,
        description="Planned contract duration in months.",
    )
    gpa: bool = Field(False, description="Whether the contract is covered by the WTO Government Procurement Agreement.")
    eu_funds: bool = Field(False, description="Whether EU structural or cohesion funds are involved.")
    fra_agreement: bool = Field(False, description="Whether this is a framework agreement.")
    electronic_auction: bool = Field(False, description="Whether an electronic auction is used.")
    accelerated: bool = Field(False, description="Whether an accelerated procedure is used.")
    n_samples: int = Field(5000, ge=100, le=10000, description="Monte Carlo sample count.")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility.")

    @field_validator("procedure_type")
    @classmethod
    def validate_procedure_type(cls, v):
        if v not in VALID_PROCEDURE_TYPES:
            raise ValueError(f"procedure_type must be one of {sorted(VALID_PROCEDURE_TYPES)}")
        return v

    @field_validator("contract_type")
    @classmethod
    def validate_contract_type(cls, v):
        if v not in VALID_CONTRACT_TYPES:
            raise ValueError(f"contract_type must be one of {sorted(VALID_CONTRACT_TYPES)}")
        return v

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v):
        if v not in VALID_CRITERIA:
            raise ValueError(f"criteria must be one of {sorted(VALID_CRITERIA)}")
        return v

    @field_validator("country")
    @classmethod
    def validate_country(cls, v):
        if v not in VALID_COUNTRIES:
            raise ValueError(
                f"Unknown country code '{v}'. "
                f"Valid codes: {sorted(VALID_COUNTRIES)}"
            )
        return v

    @field_validator("cpv_division")
    @classmethod
    def validate_cpv_division(cls, v):
        if v not in VALID_CPV_DIVISIONS:
            raise ValueError(
                f"Unknown CPV division '{v}'. "
                f"Valid codes: {sorted(VALID_CPV_DIVISIONS)}"
            )
        return v

    def to_twin_params(self) -> dict:
        return {
            "country":            self.country,
            "procedure_type":     self.procedure_type,
            "contract_type":      self.contract_type,
            "cpv_division":       self.cpv_division,
            "criteria":           self.criteria,
            "price_weight_pct":   self.price_weight_pct,
            "value_euro":         self.value_euro,
            "prep_time_days":     self.prep_time_days,
            "duration_months":    self.duration_months,
            "gpa":                self.gpa,
            "eu_funds":           self.eu_funds,
            "fra_agreement":      self.fra_agreement,
            "electronic_auction": self.electronic_auction,
            "accelerated":        self.accelerated,
        }


class OutcomeStats(BaseModel):
    mean:   float
    median: float
    p10:    float
    p25:    float
    p75:    float
    p90:    float


class SimulationMeta(BaseModel):
    n_samples:   int
    seed:        int
    duration_ms: float
    timestamp:   str


class SimulationResponse(BaseModel):
    competition:     dict  # OutcomeStats + point_pred
    single_bid_risk: dict  # OutcomeStats + probability
    cross_border:    dict  # OutcomeStats + probability
    price_ratio:     dict  # OutcomeStats + point_pred
    duration:        dict  # OutcomeStats + point_pred
    params:          dict
    meta:            SimulationMeta


class CompareRequest(BaseModel):
    scenario_a:  ProcedureParams
    scenario_b:  ProcedureParams
    label_a:     str = Field("Scenario A", max_length=60)
    label_b:     str = Field("Scenario B", max_length=60)
    n_samples:   int = Field(5000, ge=100, le=10000)


class DeltaStats(BaseModel):
    a:         float
    b:         float
    delta:     float
    delta_pct: Optional[float]


class CompareResponse(BaseModel):
    label_a:    str
    label_b:    str
    scenario_a: dict
    scenario_b: dict
    deltas:     Dict[str, DeltaStats]
    meta:       SimulationMeta


class BenchmarkRequest(BaseModel):
    country:        Optional[str] = Field(None, description="Filter by ISO country code.")
    procedure_type: Optional[str] = Field(None, description="Filter by procedure type code.")
    cpv_division:   Optional[str] = Field(None, description="Filter by 2-digit CPV division.")
    year_from:      Optional[int] = Field(None, ge=2018, le=2023)
    year_to:        Optional[int] = Field(None, ge=2018, le=2023)

    @field_validator("country")
    @classmethod
    def validate_country(cls, v):
        if v is not None and v not in VALID_COUNTRIES:
            raise ValueError(f"Unknown country code '{v}'. Valid codes: {sorted(VALID_COUNTRIES)}")
        return v

    @field_validator("procedure_type")
    @classmethod
    def validate_procedure_type(cls, v):
        if v is not None and v not in VALID_PROCEDURE_TYPES:
            raise ValueError(f"procedure_type must be one of {sorted(VALID_PROCEDURE_TYPES)}")
        return v

    @field_validator("cpv_division")
    @classmethod
    def validate_cpv_division(cls, v):
        if v is not None and v not in VALID_CPV_DIVISIONS:
            raise ValueError(f"Unknown CPV division '{v}'. Valid codes: {sorted(VALID_CPV_DIVISIONS)}")
        return v


class PolicyIntervention(BaseModel):
    param: str = Field(
        ...,
        description=(
            "Parameter to change. Numeric params support 'delta'; "
            "categorical params use 'value'. "
            "Valid params: prep_time_days, criteria, procedure_type, eu_funds, electronic_auction, "
            "fra_agreement, gpa, accelerated, price_weight_pct."
        ),
    )
    delta: Optional[float] = Field(
        None,
        description="Numeric change to apply (e.g. +14 for 14 extra prep days). Use for numeric params.",
    )
    value: Optional[Any] = Field(
        None,
        description="New value to set (e.g. 'M' to switch criteria to MEAT). Use for categorical params.",
    )

    @model_validator(mode="after")
    def check_delta_or_value(self):
        if self.delta is None and self.value is None:
            raise ValueError("Provide either 'delta' (numeric change) or 'value' (categorical override).")
        return self


class PolicyRequest(BaseModel):
    country_cluster:  Optional[str] = Field(None, description="Country cluster filter (e.g. 'CEE', 'Nordic').")
    cpv_division:     Optional[str] = Field(None, description="2-digit CPV division filter.")
    procedure_type:   Optional[str] = Field(None, description="Procedure type filter.")
    year_from:        int = Field(2018, ge=2018, le=2023)
    year_to:          int = Field(2023, ge=2018, le=2023)
    intervention:     PolicyIntervention
    n_records:        int = Field(300, ge=50, le=2000, description="Number of historical records to simulate.")
    seed:             int = Field(0, ge=0)


class ExplainRequest(ProcedureParams):
    """
    Same validated fields as ProcedureParams but without Monte Carlo settings.
    Inherits all @field_validators (procedure_type, contract_type, criteria).
    n_samples and seed are ignored if present.
    """

    def to_twin_params(self) -> dict:
        return self.model_dump(exclude={"n_samples", "seed"})


_SENSITIVITY_NUMERIC = {
    "prep_time_days":   {"min": 14.0,        "max": 90.0},
    "value_euro":       {"min": 50_000.0,    "max": 50_000_000.0},
    "price_weight_pct": {"min": 0.0,         "max": 100.0},
    "duration_months":  {"min": 1.0,         "max": 120.0},
}


class SensitivityRequest(BaseModel):
    """Sweep one numeric parameter across a range, holding all others fixed."""
    base_params: ProcedureParams
    param: str = Field(..., description=(
        "Parameter to sweep. Must be one of: "
        + ", ".join(sorted(_SENSITIVITY_NUMERIC))
    ))
    values: Optional[List[float]] = Field(
        None,
        description="Explicit list of values to test (max 20). Provide this OR n_steps.",
        max_length=20,
    )
    n_steps: Optional[int] = Field(
        None, ge=2, le=20,
        description="Auto-generate n_steps evenly-spaced values between the param's min and max.",
    )
    n_samples: int = Field(1000, ge=100, le=5000, description="MC samples per data point.")
    seed: int = Field(42, ge=0)

    @field_validator("param")
    @classmethod
    def validate_param(cls, v):
        if v not in _SENSITIVITY_NUMERIC:
            raise ValueError(
                f"param must be one of {sorted(_SENSITIVITY_NUMERIC)}. "
                "Boolean flags are not supported for sensitivity sweeps."
            )
        return v

    @model_validator(mode="after")
    def check_values_or_steps(self):
        if self.values is None and self.n_steps is None:
            raise ValueError("Provide either 'values' (explicit list) or 'n_steps' (auto-range).")
        if self.values is not None and self.n_steps is not None:
            raise ValueError("Provide 'values' OR 'n_steps', not both.")
        return self

    def resolved_values(self) -> List[float]:
        if self.values is not None:
            return self.values
        spec = _SENSITIVITY_NUMERIC[self.param]
        import numpy as np
        return [round(float(v), 4) for v in np.linspace(spec["min"], spec["max"], self.n_steps)]


class BatchRequest(BaseModel):
    """Submit up to 20 procedures for parallel simulation."""
    procedures: List[ProcedureParams] = Field(..., min_length=1, max_length=20)
    include_samples: bool = Field(False)

    @field_validator("procedures")
    @classmethod
    def check_length(cls, v):
        if len(v) > 20:
            raise ValueError("Batch size is limited to 20 procedures.")
        return v


class ExportFormat(BaseModel):
    """Format selector for the /export endpoint."""
    data: Dict[str, Any] = Field(..., description="Simulation result dict to export.")
    format: str = Field("csv", description="Output format: 'csv' or 'json'.")
    filename: str = Field("export", max_length=60, description="Base filename (no extension).")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        if v not in ("csv", "json"):
            raise ValueError("format must be 'csv' or 'json'.")
        return v


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _drop_samples(result: dict) -> dict:
    """Remove raw sample arrays from simulation output (too large for JSON by default)."""
    out = {}
    for k, v in result.items():
        if isinstance(v, dict):
            out[k] = {sk: sv for sk, sv in v.items() if sk != "samples"}
        else:
            out[k] = v
    return out


def _meta(t0: float, n_samples: int, seed: int) -> SimulationMeta:
    return SimulationMeta(
        n_samples=n_samples,
        seed=seed,
        duration_ms=round((time.time() - t0) * 1000, 1),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ─────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"], summary="API index")
def index():
    """Return API version info and a map of available endpoints."""
    return {
        "name":        "Procurement Digital Twin API",
        "version":     "2.0.0",
        "description": "EU procurement simulator — 1.1M TED contracts 2018–2023",
        "started_at":  _startup_time,
        "docs":        "/docs",
        "redoc":       "/redoc",
        "endpoints": {
            "GET  /health":         "Health check + model status",
            "GET  /status":         "Cache, history, and worker stats",
            "GET  /metrics":        "Prometheus metrics scrape endpoint",
            "GET  /history":        "Recent simulation run log",
            "GET  /metadata":       "Reference lists (countries, CPVs, procedure types)",
            "GET  /models":         "Model evaluation metrics",
            "GET  /explain/global": "Global feature importances for all models",
            "POST /simulate":       "Single-procedure Monte Carlo simulation (cached)",
            "POST /batch":          "Parallel simulation of up to 20 procedures",
            "POST /sensitivity":    "Sweep one parameter across a range",
            "POST /compare":        "Side-by-side comparison of two procedure designs",
            "POST /benchmark":      "Empirical statistics from historical data",
            "POST /policy":         "Aggregate counterfactual policy simulation",
            "POST /explain":        "Per-prediction feature contributions (SHAP)",
            "POST /export":         "Download any result as CSV or JSON",
            # V2 endpoints
            "POST /optimize":       "V2 — Multi-objective procedure optimisation",
            "POST /policy/compare": "V2 — Compare multiple policy interventions",
            "POST /advise":         "V2 — AI Procurement Advisor (rule-based + Claude)",
        },
        "v2_features": {
            "optimisation_engine": "Multi-objective Pareto optimisation over procedure parameters",
            "policy_lab":          "Side-by-side comparison of up to 5 policy interventions",
            "ai_advisor":          "Structured procurement advice; Claude-powered if ANTHROPIC_API_KEY set",
        },
    }


@app.get("/health", tags=["Info"], summary="Health check")
def health():
    """Return API health status and confirm all 5 models are loaded."""
    model_files = ["competition_model", "single_bid_model", "crossborder_model",
                   "price_model", "duration_model"]
    models_ok = all(
        os.path.exists(os.path.join(MODEL_DIR, f"{m}.pkl")) for m in model_files
    )
    cal_ok = os.path.exists(os.path.join(MODEL_DIR, "calibration_offsets.json"))
    return {
        "status":             "ok" if models_ok else "degraded",
        "models_loaded":      models_ok,
        "calibration_loaded": cal_ok,
        "n_models":           5,
        "started_at":         _startup_time,
        "timestamp":          datetime.utcnow().isoformat() + "Z",
    }


@app.get("/metadata", tags=["Reference"], summary="Reference data")
def metadata():
    """Return all valid input values: countries, CPV sectors, procedure types, clusters."""
    return {
        "countries": sorted(list(COUNTRY_CLUSTERS.keys())),
        "country_clusters": {
            cluster: [c for c, cl in COUNTRY_CLUSTERS.items() if cl == cluster]
            for cluster in sorted(set(COUNTRY_CLUSTERS.values()))
        },
        "cpv_sectors": CPV_SECTORS,
        "procedure_types": {
            "OPE": "Open procedure",
            "RES": "Restricted procedure",
            "NIC": "Negotiated with prior competition",
            "COD": "Competitive dialogue",
            "INP": "Innovation partnership",
            "AWP": "Award without prior publication",
        },
        "contract_types": {
            "S": "Services",
            "U": "Supplies",
            "W": "Works",
        },
        "criteria": {
            "M": "MEAT — Most Economically Advantageous Tender",
            "L": "Lowest price only",
        },
        "value_brackets": [
            "Below 135k", "135k-215k", "215k-431k",
            "431k-5M", "5M-50M", ">50M",
        ],
        "policy_intervention_params": [
            "prep_time_days", "criteria", "procedure_type",
            "eu_funds", "electronic_auction", "fra_agreement",
            "gpa", "accelerated", "price_weight_pct",
        ],
    }


@app.get("/models", tags=["Reference"], summary="Model evaluation metrics")
def model_metrics():
    """Return test-set evaluation metrics from the last model training run."""
    eval_path = os.path.join(MODEL_DIR, "model_evaluation.json")
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=503, detail="model_evaluation.json not found. Run training first.")
    with open(eval_path) as f:
        metrics = json.load(f)
    return {
        "metrics":      metrics,
        "training_split": "2018–2021 train / 2022–2023 test",
        "notes": {
            "competition":  "XGBoost regressor. R² reflects residual variance after design variables.",
            "single_bid":   "Logistic regression. AUC on held-out 2022–2023 data.",
            "cross_border": "Random Forest. AUC on held-out 2022–2023 data.",
            "price_ratio":  "2-stage IV Ridge. R² near zero is expected — price is market-driven.",
            "duration":     "Ridge regression on log scale.",
        },
    }


@app.get("/explain/global", tags=["Explainability"], summary="Global feature importances")
def explain_global():
    """
    Return pre-computed feature importances for all models.
    Values represent mean |SHAP| (competition) or coefficient magnitude (others).
    Higher = stronger influence on predictions.
    """
    shap_path = os.path.join(MODEL_DIR, "shap_importances.json")
    if not os.path.exists(shap_path):
        raise HTTPException(
            status_code=503,
            detail="shap_importances.json not found. Re-run model training to generate it.",
        )
    with open(shap_path) as f:
        raw = json.load(f)

    # Sort each model's features by importance descending
    result = {}
    for model_key, importances in raw.items():
        sorted_feats = sorted(importances.items(), key=lambda x: -abs(x[1]))
        result[model_key] = {
            "top_features": [{"feature": k, "importance": v} for k, v in sorted_feats[:20]],
            "n_features":   len(importances),
        }
    return result


@app.post(
    "/simulate",
    tags=["Simulation"],
    summary="Single-procedure simulation",
    response_model=SimulationResponse,
    dependencies=[Depends(require_api_key)],
)
def simulate(
    params: ProcedureParams,
    request: Request,
    include_samples: bool = Query(False),
):
    """
    Run a Monte Carlo simulation for a single procedure design.

    Results are cached by (params + seed) — identical requests return instantly.
    Cache TTL is 1 hour by default (configurable via `CACHE_TTL` env var).
    """
    t0 = time.time()
    cache_key = {**params.to_twin_params(), "_n": params.n_samples, "_seed": params.seed}
    cached = simulation_cache.get(cache_key)
    if cached is not None:
        record_cache_hit()
        result = cached.copy()
        result["meta"] = _meta(t0, params.n_samples, params.seed)
        result["meta"].cached = True  # type: ignore[attr-defined]
        simulation_history.record("/simulate", cache_key, (time.time()-t0)*1000, 200, cached=True)
        return result if include_samples else {k: (v if k != "competition" else _drop_samples({"x": v})["x"]) for k, v in result.items()}

    record_cache_miss()
    with track_simulation("/simulate"):
        try:
            result = _twin.simulate(
                params.to_twin_params(),
                n_samples=params.n_samples,
                seed=params.seed,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        except (AttributeError, IndexError, RuntimeError) as e:
            logger.error("Simulation error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal simulation error.")
        except Exception as e:
            logger.error("Unexpected simulation error: %s", type(e).__name__, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal simulation error.")

    simulation_cache.set(cache_key, result)
    simulation_history.record("/simulate", cache_key, (time.time()-t0)*1000, 200)

    if not include_samples:
        result = _drop_samples(result)
    result["meta"] = _meta(t0, params.n_samples, params.seed)
    return result


@app.post(
    "/compare",
    tags=["Simulation"],
    summary="Compare two procedure designs",
    dependencies=[Depends(require_api_key)],
)
def compare(req: CompareRequest, include_samples: bool = Query(False)):
    """
    Run simulations for two procedure designs and return a side-by-side comparison.

    The **deltas** block shows B − A for each outcome, with direction
    (positive = B is higher), magnitude, and percentage change.

    Typical use cases:
    - Open vs restricted procedure for the same contract
    - MEAT vs lowest-price criteria
    - 35-day vs 52-day preparation time
    - Effect of adding electronic auction
    """
    t0 = time.time()
    try:
        result = _twin.compare(
            req.scenario_a.to_twin_params(),
            req.scenario_b.to_twin_params(),
            label_a=req.label_a,
            label_b=req.label_b,
            n_samples=req.n_samples,
        )
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (AttributeError, IndexError, RuntimeError) as e:
        logger.error("Comparison error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal comparison error.")
    except Exception as e:
        logger.error("Unexpected comparison error: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal comparison error.")

    if not include_samples:
        result["scenario_a"] = _drop_samples(result["scenario_a"])
        result["scenario_b"] = _drop_samples(result["scenario_b"])

    simulation_history.record("/compare", {"a": req.scenario_a.model_dump(), "b": req.scenario_b.model_dump()}, (time.time()-t0)*1000, 200)
    result["meta"] = _meta(t0, req.n_samples, 42)
    return result


@app.post(
    "/benchmark",
    tags=["Empirical Data"],
    summary="Historical empirical benchmark",
    dependencies=[Depends(require_api_key)],
)
async def benchmark(req: BenchmarkRequest):
    """
    Return empirical statistics from matching historical procedures in the feature store.

    Useful for comparing simulation output against the observed distribution.
    All filters are optional — omitting all filters returns EU-wide statistics.

    Returns: n_records matched, and for each outcome: mean, median, P25, P75, coverage.
    """
    loop = asyncio.get_event_loop()
    fn = functools.partial(
        _twin.empirical_benchmark,
        country=req.country,
        procedure_type=req.procedure_type,
        cpv_division=req.cpv_division,
        year_from=req.year_from,
        year_to=req.year_to,
    )
    try:
        result = await loop.run_in_executor(_blocking_executor, fn)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (AttributeError, IndexError, RuntimeError) as e:
        logger.error("Benchmark error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal benchmark error.")
    except Exception as e:
        logger.error("Unexpected benchmark error: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal benchmark error.")

    if result["n_records"] == 0:
        return JSONResponse(
            status_code=200,
            content={"n_records": 0, "message": "No records match these filters."},
        )
    return result


@app.post(
    "/policy",
    tags=["Simulation"],
    summary="Aggregate policy simulation",
    dependencies=[Depends(require_api_key)],
)
async def policy_simulation(req: PolicyRequest):
    """
    Simulate the aggregate effect of a policy intervention across matching historical procedures.

    **How it works:**
    1. Filter historical records by country_cluster, CPV division, procedure type and years.
    2. Sample up to `n_records` matching records.
    3. For each record, run `simulate(actual_params)` (baseline) and `simulate(modified_params)` (counterfactual).
    4. Aggregate the differences to estimate the portfolio-level impact.

    **Example interventions:**
    - `{"param": "prep_time_days", "delta": 14}` — extend prep time by 14 days
    - `{"param": "criteria", "value": "M"}` — switch all to MEAT criteria
    - `{"param": "electronic_auction", "value": true}` — add e-auction

    Returns: per-outcome mean delta, % change, IQR of per-procedure deltas.
    """
    t0 = time.time()

    segment = {}
    if req.country_cluster: segment["country_cluster"] = req.country_cluster
    if req.cpv_division:    segment["cpv_division"]    = req.cpv_division
    if req.procedure_type:  segment["TOP_TYPE"]        = req.procedure_type
    segment["year_from"] = req.year_from
    segment["year_to"]   = req.year_to

    intervention = req.intervention.model_dump(exclude_none=True)

    loop = asyncio.get_event_loop()
    fn = functools.partial(
        _twin.policy_simulation,
        segment_filters=segment,
        intervention=intervention,
        n_records=req.n_records,
        seed=req.seed,
    )
    try:
        result = await loop.run_in_executor(_blocking_executor, fn)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (AttributeError, IndexError, RuntimeError) as e:
        logger.error("Policy simulation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal policy simulation error.")
    except Exception as e:
        logger.error("Unexpected policy simulation error: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal policy simulation error.")

    if "error" in result:
        return JSONResponse(
            status_code=200,
            content={
                "error":     result["error"],
                "n_matched": result.get("n_matched", 0),
                "hint":      "Try broadening the segment filters.",
            },
        )

    result["meta"] = {
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "timestamp":   datetime.utcnow().isoformat() + "Z",
    }
    return result


@app.post(
    "/explain",
    tags=["Explainability"],
    summary="Per-prediction feature contributions",
    dependencies=[Depends(require_api_key)],
)
def explain(req: ExplainRequest):
    """
    Compute SHAP feature contributions for a specific procedure design.

    Returns which features push the prediction above or below the model's
    average for both the competition model and the single-bid risk model.

    Positive SHAP values increase the predicted outcome;
    negative values decrease it.

    Note: SHAP computation requires the models to support TreeExplainer
    (XGBoost or Random Forest). Logistic Regression models fall back to
    coefficient-based explanations.
    """
    try:
        result = _twin.compute_shap(req.to_twin_params())
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (AttributeError, IndexError, RuntimeError) as e:
        logger.error("Explain error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal explain error.")
    except Exception as e:
        logger.error("Unexpected explain error: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal explain error.")

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    # Sort SHAP values by absolute magnitude for easier reading
    for model_key in result:
        if "shap_values" in result[model_key]:
            sv = result[model_key]["shap_values"]
            result[model_key]["shap_values_ranked"] = sorted(
                [{"feature": k, "shap_value": v} for k, v in sv.items()],
                key=lambda x: -abs(x["shap_value"]),
            )[:20]

    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return result


# ─────────────────────────────────────────────────────────────────
# NEW ENDPOINTS — Sprint 5+6
# ─────────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["Observability"], summary="Prometheus metrics")
def metrics_endpoint():
    body, content_type = prometheus_response()
    if body is None:
        raise HTTPException(status_code=503, detail="prometheus-client not installed.")
    return Response(content=body, media_type=content_type)


@app.get("/status", tags=["Info"], summary="System status")
def status():
    """Cache stats, history stats, rate-limit config, and worker info."""
    return {
        "version":        "3.0.0",
        "auth_enabled":   AUTH_ENABLED,
        "rate_limit":     _RATE_LIMIT,
        "cache":          simulation_cache.stats,
        "history":        simulation_history.aggregate_stats(),
        "workers": {
            "blocking_pool": int(os.environ.get("POLICY_WORKERS", "2")),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/history", tags=["Observability"], summary="Recent simulation runs")
def history(
    limit:    int = Query(50, ge=1, le=200),
    offset:   int = Query(0, ge=0),
    endpoint: Optional[str] = Query(None),
):
    """Return the N most recent simulation run records (newest first)."""
    rows = simulation_history.recent(limit=limit, offset=offset, endpoint=endpoint)
    return {"runs": rows, "count": len(rows), "limit": limit, "offset": offset}


@app.post(
    "/batch",
    tags=["Simulation"],
    summary="Batch parallel simulation",
    dependencies=[Depends(require_api_key)],
)
def batch_simulate(req: BatchRequest):
    """
    Simulate up to 20 procedures in parallel.

    Each procedure is simulated independently. Results arrive in the same
    order as the input list. Cache applies per procedure.

    **Typical use:** compare a portfolio of contracts or test many CPV divisions.
    """
    t0 = time.time()
    results = [None] * len(req.procedures)

    def _run_one(idx: int, p: "ProcedureParams"):
        cache_key = {**p.to_twin_params(), "_n": p.n_samples, "_seed": p.seed}
        cached = simulation_cache.get(cache_key)
        if cached is not None:
            record_cache_hit()
            r = cached.copy()
            if not req.include_samples:
                r = _drop_samples(r)
            return idx, r, True
        record_cache_miss()
        r = _twin.simulate(p.to_twin_params(), n_samples=p.n_samples, seed=p.seed)
        simulation_cache.set(cache_key, r)
        if not req.include_samples:
            r = _drop_samples(r)
        return idx, r, False

    from concurrent.futures import as_completed
    futures = {
        _blocking_executor.submit(_run_one, i, p): i
        for i, p in enumerate(req.procedures)
    }
    try:
        for fut in as_completed(futures, timeout=_REQUEST_TIMEOUT - 5):
            idx, r, was_cached = fut.result()
            results[idx] = r
    except Exception as e:
        logger.error("Batch simulation error: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="One or more batch simulations failed.")

    simulation_history.record("/batch", {"n_procedures": len(req.procedures)}, (time.time()-t0)*1000, 200)
    return {
        "results":    results,
        "n":          len(results),
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "timestamp":  datetime.utcnow().isoformat() + "Z",
    }


@app.post(
    "/sensitivity",
    tags=["Simulation"],
    summary="Sensitivity analysis",
    dependencies=[Depends(require_api_key)],
)
def sensitivity(req: SensitivityRequest):
    """
    Sweep one numeric parameter across a range while holding all others fixed.

    Returns how all 5 outcomes change as the parameter varies — the most useful
    tool for procurement officers tuning procedure design.

    **Example**: sweep `prep_time_days` from 14 to 90 in 10 steps and see
    how expected competition, single-bid risk, and duration respond.
    """
    import numpy as np
    t0 = time.time()
    values = req.resolved_values()
    base   = req.base_params.to_twin_params()

    outcome_series: Dict[str, list] = {
        "competition":     [],
        "single_bid_risk": [],
        "cross_border":    [],
        "price_ratio":     [],
        "duration":        [],
    }

    for i, val in enumerate(values):
        params = {**base, req.param: val}
        cache_key = {**params, "_n": req.n_samples, "_seed": req.seed + i}
        cached = simulation_cache.get(cache_key)
        if cached is not None:
            record_cache_hit()
            r = cached
        else:
            record_cache_miss()
            r = _twin.simulate(params, n_samples=req.n_samples, seed=req.seed + i)
            simulation_cache.set(cache_key, r)

        outcome_series["competition"].append({
            "value": val,
            "mean": r["competition"]["mean"],
            "p25":  r["competition"]["p25"],
            "p75":  r["competition"]["p75"],
        })
        outcome_series["single_bid_risk"].append({
            "value":       val,
            "probability": r["single_bid_risk"]["probability"],
        })
        outcome_series["cross_border"].append({
            "value":       val,
            "probability": r["cross_border"]["probability"],
        })
        outcome_series["price_ratio"].append({
            "value": val,
            "mean":  r["price_ratio"]["mean"],
            "p25":   r["price_ratio"]["p25"],
            "p75":   r["price_ratio"]["p75"],
        })
        outcome_series["duration"].append({
            "value": val,
            "mean":  r["duration"]["mean"],
            "p25":   r["duration"]["p25"],
            "p75":   r["duration"]["p75"],
        })

    simulation_history.record("/sensitivity", {"param": req.param, "n_values": len(values)}, (time.time()-t0)*1000, 200)
    return {
        "param":         req.param,
        "values_tested": values,
        "outcomes":      outcome_series,
        "base_params":   base,
        "meta": {
            "n_samples":   req.n_samples,
            "n_values":    len(values),
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "timestamp":   datetime.utcnow().isoformat() + "Z",
        },
    }


# ─────────────────────────────────────────────────────────────────
# V2 SCHEMAS
# ─────────────────────────────────────────────────────────────────

_OBJECTIVE_KEYS = {"competition", "single_bid_risk", "cross_border",
                   "price_ratio", "duration"}


class ObjectiveWeights(BaseModel):
    """
    Signed weights for each procurement outcome.
    Positive = maximise the outcome; negative = minimise it.
    Weights are normalised internally so only relative magnitudes matter.
    """
    competition:     float = Field(0.0, ge=-1.0, le=1.0,
        description="Weight for number of bids (positive = want more competition).")
    single_bid_risk: float = Field(0.0, ge=-1.0, le=1.0,
        description="Weight for single-bid risk (negative = want lower risk).")
    cross_border:    float = Field(0.0, ge=-1.0, le=1.0,
        description="Weight for cross-border win probability (positive = want higher).")
    price_ratio:     float = Field(0.0, ge=-1.0, le=1.0,
        description="Weight for price ratio (negative = want lower ratio / cheaper).")
    duration:        float = Field(0.0, ge=-1.0, le=1.0,
        description="Weight for procedure duration (negative = want shorter).")


class OptimisationConstraints(BaseModel):
    """Optional hard constraints for the optimisation search space."""
    allowed_procedure_types: Optional[List[str]] = Field(
        None, description="Restrict to a subset of procedure types, e.g. ['OPE', 'NIC'].")
    min_prep_time: float = Field(21.0, ge=14.0, le=90.0)
    max_prep_time: float = Field(90.0, ge=14.0, le=90.0)
    must_use_meat: bool  = Field(False, description="Force MEAT award criteria.")

    @field_validator("allowed_procedure_types")
    @classmethod
    def validate_procs(cls, v):
        if v is not None:
            invalid = set(v) - VALID_PROCEDURE_TYPES
            if invalid:
                raise ValueError(f"Unknown procedure types: {invalid}")
        return v


class OptimisationRequest(BaseModel):
    """Request body for /optimize."""
    base_params:       ProcedureParams
    objective_weights: ObjectiveWeights
    constraints:       Optional[OptimisationConstraints] = None
    n_samples:         int = Field(500, ge=100, le=2000,
        description="Monte Carlo samples per candidate. Lower = faster, higher = more precise.")
    seed:              int = Field(42, ge=0)


class PolicyComparePolicyItem(BaseModel):
    """One policy variant for /policy/compare."""
    name:         str = Field(..., max_length=60)
    intervention: Optional[PolicyIntervention] = Field(
        None, description="Use null for 'Status Quo' baseline entry.")


class PolicyCompareRequest(BaseModel):
    """Request body for /policy/compare."""
    country_cluster:  Optional[str] = Field(None)
    cpv_division:     Optional[str] = Field(None)
    procedure_type:   Optional[str] = Field(None)
    year_from:        int = Field(2018, ge=2018, le=2023)
    year_to:          int = Field(2023, ge=2018, le=2023)
    policies:         List[PolicyComparePolicyItem] = Field(
        ..., min_length=1, max_length=5,
        description="Up to 5 policy variants (include a null-intervention entry for Status Quo).")
    n_records:        int = Field(200, ge=50, le=1000)
    seed:             int = Field(0, ge=0)

    @field_validator("cpv_division")
    @classmethod
    def validate_cpv(cls, v):
        if v is not None and v not in VALID_CPV_DIVISIONS:
            raise ValueError(f"Unknown CPV division '{v}'.")
        return v

    @field_validator("procedure_type")
    @classmethod
    def validate_proc(cls, v):
        if v is not None and v not in VALID_PROCEDURE_TYPES:
            raise ValueError(f"Unknown procedure type '{v}'.")
        return v


class AdviseRequest(BaseModel):
    """Request body for /advise."""
    params:           ProcedureParams
    question:         Optional[str] = Field(None, max_length=500,
        description="Specific question for the AI advisor (activates Claude narrative if available).")
    include_shap:     bool = Field(True, description="Compute SHAP contributions for richer advice.")
    n_samples:        int  = Field(2000, ge=200, le=5000)
    seed:             int  = Field(42, ge=0)


# ─────────────────────────────────────────────────────────────────
# V2 ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.post(
    "/optimize",
    tags=["V2 — Optimisation"],
    summary="Multi-objective procedure optimisation",
    dependencies=[Depends(require_api_key)],
)
async def optimise(req: OptimisationRequest):
    """
    **V2 Flagship feature** — Find the optimal procurement procedure configuration
    given user-defined objective weights and optional hard constraints.

    The engine sweeps ~72–200 candidate configurations (varying procedure type,
    award criteria, price weight, preparation time, and e-auction), evaluates each
    via Monte Carlo simulation, and ranks them by a weighted utility score.

    Also returns the **Pareto frontier** for the top-2 weighted objectives, enabling
    trade-off visualisation (e.g. competition vs. duration).

    **Objective weights**: positive = maximise, negative = minimise.
    Example: `{"competition": 0.4, "single_bid_risk": -0.4, "duration": -0.2}`.

    Returns:
    - `best` — single highest-utility configuration
    - `candidates` — top-20 ranked configurations with full outcome predictions
    - `pareto_frontier` — non-dominated solutions for the top-2 objectives
    - `search_space` — number of candidates evaluated
    """
    t0 = time.time()

    weights = req.objective_weights.model_dump()
    if all(w == 0.0 for w in weights.values()):
        raise HTTPException(
            status_code=422,
            detail="At least one objective weight must be non-zero.",
        )

    constraints = None
    if req.constraints:
        constraints = req.constraints.model_dump(exclude_none=True)
        if constraints.get("allowed_procedure_types") is None:
            constraints.pop("allowed_procedure_types", None)

    base = req.base_params.to_twin_params()

    loop = asyncio.get_event_loop()
    fn = functools.partial(
        _twin.optimize,
        base_params=base,
        objective_weights=weights,
        constraints=constraints,
        n_samples=req.n_samples,
        seed=req.seed,
    )
    try:
        result = await loop.run_in_executor(_blocking_executor, fn)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Optimisation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal optimisation error.")

    if "error" in result:
        return JSONResponse(status_code=200, content=result)

    result["meta"] = {
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "n_samples":   req.n_samples,
    }
    simulation_history.record(
        "/optimize",
        {"weights": weights, "n_samples": req.n_samples},
        (time.time() - t0) * 1000, 200,
    )
    return result


@app.post(
    "/policy/compare",
    tags=["V2 — Policy Lab"],
    summary="Compare multiple policy interventions",
    dependencies=[Depends(require_api_key)],
)
async def policy_compare(req: PolicyCompareRequest):
    """
    **V2 Policy Lab** — Simulate and compare up to 5 policy interventions
    against the status-quo baseline on the same population of historical procedures.

    Unlike `/policy` (single intervention), this endpoint runs all interventions
    on the same sampled records so results are directly comparable.

    Include a `{"name": "Status Quo", "intervention": null}` entry to anchor the
    comparison to the unmodified baseline.

    Returns per-policy aggregate outcomes, deltas vs. baseline, and 95% CIs.
    """
    t0 = time.time()

    segment = {}
    if req.country_cluster: segment["country_cluster"] = req.country_cluster
    if req.cpv_division:    segment["cpv_division"]    = req.cpv_division
    if req.procedure_type:  segment["TOP_TYPE"]        = req.procedure_type
    segment["year_from"] = req.year_from
    segment["year_to"]   = req.year_to

    policies = [
        {
            "name":         p.name,
            "intervention": p.intervention.model_dump(exclude_none=True)
                            if p.intervention else None,
        }
        for p in req.policies
    ]

    loop = asyncio.get_event_loop()
    fn = functools.partial(
        _twin.policy_compare,
        segment_filters=segment,
        policies=policies,
        n_records=req.n_records,
        seed=req.seed,
    )
    try:
        result = await loop.run_in_executor(_blocking_executor, fn)
    except Exception as exc:
        logger.error("Policy compare error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal policy comparison error.")

    if "error" in result:
        return JSONResponse(status_code=200, content={
            "error":     result["error"],
            "n_matched": result.get("n_matched", 0),
        })

    result["meta"] = {
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "timestamp":   datetime.utcnow().isoformat() + "Z",
    }
    return result


@app.post(
    "/advise",
    tags=["V2 — AI Advisor"],
    summary="AI Procurement Advisor",
    dependencies=[Depends(require_api_key)],
)
def advise(req: AdviseRequest):
    """
    **V2 AI Advisor** — Generate actionable procurement design recommendations.

    Combines simulation results, SHAP feature contributions, and rule-based
    procurement expertise to produce structured advice with severity ratings.

    If `ANTHROPIC_API_KEY` is configured and the `question` field is provided,
    a Claude-powered narrative is added on top of the rule-based recommendations
    (the tool always works without an API key).

    Returns:
    - `summary` — 2-sentence executive assessment
    - `recommendations` — ranked list with severity (high / medium / low)
    - `key_risks` — bulleted risk signals
    - `strengths` — bulleted positive signals
    - `llm_powered` — bool; `llm_narrative` present when True
    """
    if not _ADVISOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Advisor module unavailable. Check server logs.",
        )

    t0 = time.time()
    params = req.params.to_twin_params()

    try:
        sim_result = _twin.simulate(params, n_samples=req.n_samples, seed=req.seed)
    except Exception as exc:
        logger.error("Advisor — simulation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Simulation failed during advise.")

    shap_result = None
    if req.include_shap:
        try:
            shap_result = _twin.compute_shap(params)
        except Exception as exc:
            logger.warning("Advisor — SHAP error (non-fatal): %s", exc)

    try:
        advice = _advisor.advise(
            params=params,
            simulation_result=sim_result,
            shap_result=shap_result,
            question=req.question,
        )
    except Exception as exc:
        logger.error("Advisor error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Advisor error.")

    advice["simulation_summary"] = {
        "competition":     round(sim_result["competition"]["mean"], 2),
        "single_bid_risk": round(sim_result["single_bid_risk"]["probability"], 3),
        "cross_border":    round(sim_result["cross_border"]["probability"], 3),
        "price_ratio":     round(sim_result["price_ratio"]["mean"], 3),
        "duration":        round(sim_result["duration"]["mean"], 1),
    }
    advice["meta"] = {
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "llm_available": _advisor._claude_available if _advisor else False,
    }
    simulation_history.record("/advise", params, (time.time() - t0) * 1000, 200)
    return advice


@app.post(
    "/export",
    tags=["Utilities"],
    summary="Export simulation result as CSV or JSON",
    dependencies=[Depends(require_api_key)],
)
def export(req: ExportFormat):
    """
    Convert any simulation result dict (from /simulate, /compare, /sensitivity)
    to a downloadable CSV or JSON file.

    The CSV flattens nested dicts with dot-notation keys, making it easy to
    open results in Excel or pandas.
    """
    ext = req.format
    filename = f"{req.filename}.{ext}"

    if ext == "json":
        body = json.dumps(req.data, indent=2, default=str).encode()
        return StreamingResponse(
            io.BytesIO(body),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Flatten nested dict → CSV
    def _flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        out.update(_flatten(item, f"{key}[{i}]"))
                    else:
                        out[f"{key}[{i}]"] = item
            else:
                out[key] = v
        return out

    flat = _flatten(req.data)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(flat.keys()))
    writer.writeheader()
    writer.writerow(flat)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT (when run directly)
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 60)
    logger.info("PROCUREMENT DIGITAL TWIN — REST API v3.0.0")
    logger.info("=" * 60)
    logger.info("Docs:    http://localhost:8000/docs")
    logger.info("ReDoc:   http://localhost:8000/redoc")
    logger.info("Metrics: http://localhost:8000/metrics")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
