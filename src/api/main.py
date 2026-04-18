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

import sys, os, json, time, logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from simulation.simulation_engine import (
    ProcurementTwin, COUNTRY_CLUSTERS, CPV_SECTORS, value_bracket
)

# ── App initialisation ────────────────────────────────────────────
app = FastAPI(
    title="Procurement Digital Twin API",
    description=(
        "REST API for the EU Procurement Digital Twin. "
        "Simulates competition, cross-border participation, price ratios and "
        "procedure duration from 1.1 million TED contract records (2018–2023). "
        "All simulation endpoints use calibrated Monte Carlo sampling."
    ),
    version="2.0.0",
    contact={"name": "Procurement Digital Twin"},
    license_info={"name": "Internal use"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load twin once at startup ─────────────────────────────────────
logger.info("Loading Procurement Twin models...")
_twin = ProcurementTwin()
_startup_time = datetime.utcnow().isoformat() + "Z"
logger.info("Models ready.")

# ─────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────

VALID_PROCEDURE_TYPES = {"OPE", "RES", "NIC", "COD", "INP", "AWP", "NOC"}
VALID_CONTRACT_TYPES  = {"S", "U", "W"}
VALID_CRITERIA        = {"M", "L"}


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
            "GET  /metadata":       "Reference lists (countries, CPVs, procedure types)",
            "GET  /models":         "Model evaluation metrics",
            "GET  /explain/global": "Global feature importances for all models",
            "POST /simulate":       "Single-procedure Monte Carlo simulation",
            "POST /compare":        "Side-by-side comparison of two procedure designs",
            "POST /benchmark":      "Empirical statistics from historical data",
            "POST /policy":         "Aggregate counterfactual policy simulation",
            "POST /explain":        "Per-prediction feature contributions (SHAP)",
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
)
def simulate(params: ProcedureParams, include_samples: bool = Query(False, description="Include raw Monte Carlo sample arrays in the response.")):
    """
    Run a Monte Carlo simulation for a single procedure design.

    Returns distributional predictions (mean, median, percentiles) for:
    - **competition** — expected number of offers received
    - **single_bid_risk** — probability of receiving only one bid
    - **cross_border** — probability of a cross-border winner
    - **price_ratio** — expected award value ÷ estimated value
    - **duration** — expected procedure duration in days

    All predictions use 5 calibrated models trained on 1.1M TED contracts.
    Calibration offsets are applied per CPV sector and country cluster.
    """
    t0 = time.time()
    try:
        result = _twin.simulate(
            params.to_twin_params(),
            n_samples=params.n_samples,
            seed=params.seed,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Simulation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal simulation error.")

    if not include_samples:
        result = _drop_samples(result)

    result["meta"] = _meta(t0, params.n_samples, params.seed)
    return result


@app.post(
    "/compare",
    tags=["Simulation"],
    summary="Compare two procedure designs",
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
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Comparison error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal comparison error.")

    if not include_samples:
        result["scenario_a"] = _drop_samples(result["scenario_a"])
        result["scenario_b"] = _drop_samples(result["scenario_b"])

    result["meta"] = _meta(t0, req.n_samples, 42)
    return result


@app.post(
    "/benchmark",
    tags=["Empirical Data"],
    summary="Historical empirical benchmark",
)
def benchmark(req: BenchmarkRequest):
    """
    Return empirical statistics from matching historical procedures in the feature store.

    Useful for comparing simulation output against the observed distribution.
    All filters are optional — omitting all filters returns EU-wide statistics.

    Returns: n_records matched, and for each outcome: mean, median, P25, P75.
    """
    try:
        result = _twin.empirical_benchmark(
            country=req.country,
            procedure_type=req.procedure_type,
            cpv_division=req.cpv_division,
            year_from=req.year_from,
            year_to=req.year_to,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Benchmark error: %s", e, exc_info=True)
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
)
def policy_simulation(req: PolicyRequest):
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

    try:
        result = _twin.policy_simulation(
            segment_filters=segment,
            intervention=intervention,
            n_records=req.n_records,
            seed=req.seed,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Policy simulation error: %s", e, exc_info=True)
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
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Explain error: %s", e, exc_info=True)
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
# ENTRY POINT (when run directly)
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("PROCUREMENT DIGITAL TWIN — REST API")
    print("=" * 60)
    print("\n  Docs:  http://localhost:8000/docs")
    print("  ReDoc: http://localhost:8000/redoc\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
