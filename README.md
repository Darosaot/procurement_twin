---
title: Procurement Digital Twin
emoji: 🔷
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Procurement Digital Twin

A simulation engine for EU public procurement procedures, built on 1.1 million linked TED contract notices (2018–2023).

---

## How to Run

### ⭐ One-click launch (macOS, recommended)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) — one-time, free
2. **Double-click `Launch Digital Twin.command`**

That's it. The script starts Docker if needed, builds the image on first run (~3 min), and opens the dashboard in your browser automatically.

To stop: double-click **`Stop Digital Twin.command`**, or press `Ctrl+C` in the terminal window.

> **First-time note:** macOS may ask you to confirm running the `.command` file. Right-click → Open the first time to bypass the security prompt.

---

### Option A — Docker (command line)

**Prerequisite:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
docker compose up
```

This starts **three services** simultaneously:

| Service | URL | Description |
|---------|-----|-------------|
| Dashboard | http://localhost:8050 | Plotly Dash interactive UI |
| REST API | http://localhost:8000 | FastAPI + Swagger docs at `/docs` |
| Jupyter | http://localhost:8888 | Analysis notebook (no login required) |

To stop: `Ctrl+C`, then `docker compose down`.

To rebuild after code changes: `docker compose up --build`.

---

### Option B — Plain Python (local)

**Prerequisite:** Python 3.10 or later.

**Mac / Linux:**
```bash
bash setup.sh            # creates a virtual environment and installs dependencies (once)
source .venv/bin/activate
python run.py
```

**Windows:**
```bat
setup.bat                :: creates a virtual environment and installs dependencies (once)
.venv\Scripts\activate
python run.py
```

Open **http://localhost:8050** in your browser.

---

## Dashboard Tabs

### 🎯 Procedure Designer
Set parameters for a planned procedure and receive a real-time simulation:

- Expected number of bids (with confidence intervals)
- Probability of receiving only one bid (zero-competition risk)
- Probability of a cross-border winner
- Expected price ratio (award vs estimate) — 2-stage IV model
- Expected procedure duration (days from publication to award)

Results include benchmark comparison against matching historical procedures, plus
per-CPV and per-country-cluster calibration offsets for improved accuracy.

### ⚖️ Scenario Comparator
Define two procedure designs and compare them side by side:
- Delta table showing direction, magnitude, and direction of impact
- Overlaid distribution charts for each outcome

Typical questions:
- Does MEAT criteria produce more competition than lowest price?
- What is the effect of extending the preparation time from 35 to 52 days?
- How does an open procedure compare to a restricted one for this contract?

### 🔍 Policy Explorer
Filter the full 1.1M record dataset and explore empirical distributions:
- By country, procedure type, CPV sector, year
- Distribution histograms, country comparisons, year trends, CPV breakdown

### 🏛️ Policy Simulation
Aggregate counterfactual analysis — "What if this policy were applied across all matching procedures?":
- Select a target segment (country cluster, CPV sector, procedure type, year range)
- Select a policy intervention (extend prep time, switch criteria, add e-auction, etc.)
- The engine simulates the intervention across 200–1,000 sampled historical procedures
- Shows aggregate change in competition, single-bid risk, price ratio, and duration
- Includes per-procedure impact distributions

### 💡 Explain
Model explainability and SHAP feature importance:
- Global feature importance charts for competition and single-bid risk models
- Per-prediction SHAP waterfall: "why does this procedure have elevated single-bid risk?"
- Natural language summary of the top contributing factors

---

## Project Structure

```
Procurement_Digital_Twin/
│
├── run.py                              ← Launch the dashboard (port 8050)
├── run_api.py                          ← Launch the REST API (port 8000)
├── requirements.txt                    ← Python dependencies
├── Dockerfile                          ← Docker image definition
├── docker-compose.yml                  ← One-command Docker launch (dashboard + API + Jupyter)
├── setup.sh / setup.bat                ← Local install scripts
│
├── src/
│   ├── pipeline/
│   │   ├── 01_linkage.py               ← Resolves CFC–CAN notice linkage
│   │   └── 02_ingest_and_features.py   ← Builds the Parquet feature store
│   ├── models/
│   │   └── 03_train_models.py          ← Trains all predictive models
│   ├── simulation/
│   │   └── simulation_engine.py        ← ProcurementTwin class (Monte Carlo)
│   ├── dashboard/
│   │   └── app.py                      ← Plotly Dash interface
│   └── api/
│       └── main.py                     ← FastAPI REST API
│
├── notebooks/
│   └── analysis_demo.ipynb             ← End-to-end analysis walkthrough
│
├── data/
│   ├── processed/                      ← CFC–CAN linkage table (Parquet)
│   └── features/                       ← Feature store (1.1M records, ~45 MB)
│       ├── procedure_records.parquet   ← Linked CFC + CAN records
│       ├── can_outcomes.parquet
│       └── cfc_unlinked.parquet
│
├── models/                             ← Trained model artefacts
│   ├── competition_model.pkl           ← XGBoost: expected bids
│   ├── single_bid_model.pkl            ← Logistic Regression: single-bid risk
│   ├── crossborder_model.pkl           ← Random Forest: cross-border win
│   ├── price_model.pkl                 ← 2-stage IV Ridge: price ratio
│   ├── duration_model.pkl              ← Ridge: procedure duration
│   ├── model_evaluation.json           ← Validation metrics
│   ├── calibration_offsets.json        ← Per-CPV and per-cluster offsets
│   └── shap_importances.json           ← Pre-computed global feature importances
│
└── Procurement data/                   ← Raw TED exports (not re-distributed)
    ├── export_CAN_2023_2018.csv        (6.2M rows — award notices)
    └── export_CFC_2018_2023.csv        (7.7M rows — contract notices)
```

---

## Rebuild the Data Pipeline (optional)

The processed data and trained models are already included. You only need to re-run the pipeline if you update the raw data files.

```bash
python src/pipeline/01_linkage.py                # ~30s  — CFC–CAN linkage
python src/pipeline/02_ingest_and_features.py    # ~25s  — Feature store
python src/models/03_train_models.py             # ~80s  — Model training
```

---

## REST API

A FastAPI server exposes all simulation capabilities as HTTP endpoints.

### Start the API server

```bash
# Local (Python)
python run_api.py                          # http://localhost:8000
python run_api.py --port 9000              # custom port
python run_api.py --reload                 # hot-reload for development

# Docker (starts alongside the dashboard)
docker compose up
```

Interactive docs auto-generated at **http://localhost:8000/docs** (Swagger UI) and
**http://localhost:8000/redoc** (ReDoc).

---

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info and link index |
| `GET` | `/health` | Model status and uptime |
| `GET` | `/metadata` | Reference lists (countries, CPVs, procedure types) |
| `GET` | `/models` | Validation metrics for all 5 models |
| `GET` | `/explain/global` | Global feature importances |
| `POST` | `/simulate` | Single-procedure Monte Carlo simulation |
| `POST` | `/compare` | Side-by-side comparison of two designs |
| `POST` | `/benchmark` | Empirical statistics from historical data |
| `POST` | `/policy` | Aggregate counterfactual policy simulation |
| `POST` | `/explain` | Per-prediction SHAP feature contributions |

---

### Example requests

**Simulate a procedure**

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "country": "BE",
    "procedure_type": "OPE",
    "contract_type": "S",
    "cpv_division": "72",
    "criteria": "M",
    "price_weight_pct": 40,
    "value_euro": 500000,
    "prep_time_days": 45,
    "duration_months": 18,
    "gpa": true
  }'
```

Response includes `competition`, `single_bid_risk`, `cross_border`, `price_ratio`,
and `duration` — each with mean, median, P10/P25/P75/P90 and a point prediction.

**Compare two designs**

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_a": {"country": "BE", "procedure_type": "OPE", "criteria": "M", ...},
    "scenario_b": {"country": "BE", "procedure_type": "RES", "criteria": "L", ...},
    "label_a": "Open + MEAT",
    "label_b": "Restricted + Price",
    "n_samples": 2000
  }'
```

Response adds a `deltas` block with absolute and percentage differences.

**Benchmark against historical data**

```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{"country": "BE", "procedure_type": "OPE", "cpv_division": "72"}'
```

**Policy simulation**

```bash
curl -X POST http://localhost:8000/policy \
  -H "Content-Type: application/json" \
  -d '{
    "country_cluster": "Benelux",
    "cpv_division": "72",
    "procedure_type": "OPE",
    "year_from": 2020,
    "year_to": 2022,
    "intervention": {"param": "prep_time_days", "delta": 14},
    "n_records": 200
  }'
```

Available intervention params: `prep_time_days`, `price_weight_pct`, `criteria`,
`electronic_auction`, `gpa`, `eu_funds`.

**Per-prediction explanations (SHAP)**

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"country": "BE", "procedure_type": "OPE", "cpv_division": "72", ...}'
```

Returns top-20 feature contributions for competition (XGBoost SHAP) and
single-bid risk (logistic regression linear contributions).

---

### Python client example

```python
import httpx

BASE = "http://localhost:8000"

# Simulate
r = httpx.post(f"{BASE}/simulate", json={
    "country": "ES", "procedure_type": "OPE", "contract_type": "S",
    "cpv_division": "45", "criteria": "M", "price_weight_pct": 60,
    "value_euro": 2_000_000, "prep_time_days": 52, "duration_months": 36,
})
sim = r.json()
print(f"Expected bids:   {sim['competition']['mean']:.1f}")
print(f"Single-bid risk: {sim['single_bid_risk']['probability']:.1%}")

# Get validation metrics
metrics = httpx.get(f"{BASE}/models").json()
print(metrics["metrics"]["competition"])
```

---

## Programmatic Use (direct Python)

```python
from src.simulation.simulation_engine import ProcurementTwin

twin = ProcurementTwin()

# Simulate a single procedure
result = twin.simulate({
    "country":          "DE",
    "procedure_type":   "OPE",
    "contract_type":    "S",
    "cpv_division":     "72",
    "criteria":         "M",
    "price_weight_pct": 60,
    "value_euro":       1_000_000,
    "prep_time_days":   35,
    "duration_months":  24,
})

print(f"Expected bids:    {result['competition']['mean']:.1f}")
print(f"Single-bid risk:  {result['single_bid_risk']['probability']:.1%}")
print(f"Cross-border win: {result['cross_border']['probability']:.1%}")

# Compare two designs
comparison = twin.compare(params_a, params_b,
                           label_a="Lowest price", label_b="MEAT")

# Empirical benchmark from historical data
benchmark = twin.empirical_benchmark(
    country="DE", procedure_type="OPE", cpv_division="72"
)

# Policy simulation
delta = twin.policy_simulation(
    segment_filters={"country_cluster": "Benelux", "cpv_division": "72"},
    intervention={"param": "prep_time_days", "delta": 14},
    n_records=300,
)
```

---

## Model Performance

| Outcome | Algorithm | Test metric |
|---------|-----------|-------------|
| Competition (n_offers) | XGBoost | MAE = 2.4 bids, R² = 0.18 |
| Single-bid risk | Logistic Regression | AUC = 0.68 |
| Cross-border win | Random Forest | AUC = 0.76 |
| Price ratio | 2-stage IV Ridge | MAE = 0.285, R² = −0.06* |
| Procedure duration | Ridge (log scale) | MAE = 43 days, R² = 0.09 |

\* Price ratio is inherently unpredictable from design variables (market conditions dominate).
  The 2-stage IV model captures the competition → price causal pathway.
  Post-hoc calibration offsets are applied per CPV sector and country cluster.

Temporal train/test split: 2018–2021 (train) vs 2022–2023 (test).

---

*Data: TED (Tenders Electronic Daily), European Commission Open Data Portal.*
*Coverage: EU-wide, 2018–2023, above-threshold procedures under Directives 2014/24/EU and 2014/25/EU.*
