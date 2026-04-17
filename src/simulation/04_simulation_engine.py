"""
Phase 3: Simulation Engine
============================
Takes user-specified procedure parameters and returns
Monte Carlo outcome distributions across all four models.

Core class: ProcurementTwin
  .simulate(params, n_samples=5000) → dict of outcome distributions
  .compare(params_a, params_b) → side-by-side comparison
  .empirical_benchmark(filters) → historical baseline from feature store

Procedure parameters (inputs):
  country          : ISO code (e.g. "DE", "FR", "PL")
  procedure_type   : "OPE", "RES", "NIC", "NOC", "AWP", "COD"
  contract_type    : "S" (Services), "U" (Supplies), "W" (Works)
  cpv_division     : 2-digit CPV code (e.g. "45", "72", "85")
  criteria         : "L" (Lowest price) or "M" (MEAT/best value)
  price_weight_pct : 0–100 (only for MEAT, percentage for price criterion)
  value_euro       : estimated contract value in EUR
  prep_time_days   : days between publication and submission deadline
  duration_months  : planned contract duration in months
  eu_funds         : True/False — EU funds involved
  gpa              : True/False — GPA covered
  electronic_auction: True/False
  accelerated      : True/False
  fra_agreement    : True/False — framework agreement
"""

import numpy as np
import pandas as pd
import pickle, json, os
import warnings
warnings.filterwarnings("ignore")

_THIS_DIR2  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_THIS_DIR2, "..", "..", "models")
FEAT_DIR  = os.path.join(_THIS_DIR2, "..", "..", "data", "features")

# ── Country cluster mapping ───────────────────────────────────────
COUNTRY_CLUSTERS = {
    "BE":"Benelux","NL":"Benelux","LU":"Benelux",
    "DE":"Germanic","AT":"Germanic","CH":"Germanic",
    "FR":"Western","IT":"Western",
    "ES":"Iberian","PT":"Iberian",
    "SE":"Nordic","DK":"Nordic","FI":"Nordic","NO":"Nordic","IS":"Nordic",
    "PL":"CEE","CZ":"CEE","SK":"CEE","HU":"CEE","RO":"CEE","BG":"CEE",
    "LT":"Baltic","LV":"Baltic","EE":"Baltic",
    "HR":"Balkan","SI":"Balkan","MK":"Balkan",
    "GR":"Mediterranean","CY":"Mediterranean","MT":"Mediterranean",
    "IE":"Anglophone","UK":"Anglophone",
}

CPV_SECTORS = {
    "03":"Agriculture & Forestry","09":"Petroleum Products","14":"Mining & Quarrying",
    "15":"Food & Beverages","22":"Printed Matter","24":"Chemical Products",
    "30":"IT Equipment","31":"Electrical Equipment","32":"Radio & Comms Equipment",
    "33":"Medical & Pharma","34":"Transport Equipment","38":"Laboratory Equipment",
    "39":"Furniture & Fittings","42":"Industrial Machinery","44":"Construction Materials",
    "45":"Construction Works","48":"Software","50":"Repair & Maintenance",
    "60":"Transport Services","64":"Postal Services","65":"Gas & Electricity",
    "66":"Financial Services","70":"Real Estate","71":"Architecture & Engineering",
    "72":"IT Services","73":"R&D Services","75":"Public Administration",
    "77":"Agricultural Services","79":"Business Services","80":"Education",
    "85":"Health & Social Work","90":"Waste & Environment","98":"Other Services",
}

def value_bracket(v):
    if v is None or v <= 0: return "Unknown"
    if v < 135_000:  return "Below 135k"
    if v < 215_000:  return "135k-215k"
    if v < 431_000:  return "215k-431k"
    if v < 5_000_000: return "431k-5M"
    if v < 50_000_000: return "5M-50M"
    return ">50M"


class ProcurementTwin:
    """
    The Procurement Digital Twin simulation engine.
    Loads trained models and runs Monte Carlo simulations.
    """

    def __init__(self):
        print("Loading models...")
        self.competition_mdl  = self._load("competition_model")
        self.singlebid_mdl    = self._load("single_bid_model")
        self.crossborder_mdl  = self._load("crossborder_model")
        self.price_mdl        = self._load("price_model")
        self.duration_mdl     = self._load("duration_model")
        self.feature_spec     = self._load_json("feature_spec.json")

        # Load empirical stats for simulation noise
        self._comp_meta  = self.competition_mdl["meta"]
        self._price_meta = self.price_mdl["meta"]
        self._dur_meta   = self.duration_mdl["meta"]
        print("  All models loaded.")

    def _load(self, name):
        with open(f"{MODEL_DIR}/{name}.pkl", "rb") as f:
            return pickle.load(f)

    def _load_json(self, name):
        with open(f"{MODEL_DIR}/{name}", "r") as f:
            return json.load(f)

    def _params_to_df(self, params: dict) -> pd.DataFrame:
        """Convert user-supplied params dict to a model-ready DataFrame."""
        v = params.get("value_euro", None)
        log10_v = np.log10(v) if v and v > 0 else np.nan

        row = {
            # Categorical
            "ISO_COUNTRY_CODE":      params.get("country",         "DE"),
            "TOP_TYPE":              params.get("procedure_type",   "OPE"),
            "TYPE_OF_CONTRACT":      params.get("contract_type",    "S"),
            "cpv_division":          str(params.get("cpv_division", "72")),
            "CRIT_CODE":             params.get("criteria",         "M"),
            "value_bracket":         value_bracket(v),
            "country_cluster":       COUNTRY_CLUSTERS.get(params.get("country","DE"), "Other"),
            # Numeric
            "log10_value":           log10_v,
            "prep_time_days":        params.get("prep_time_days",   35.0),
            "contract_duration_months": params.get("duration_months", 24.0),
            "flag_b_gpa":            int(params.get("gpa", False)),
            "flag_b_eu_funds":       int(params.get("eu_funds", False)),
            "flag_b_fra_agreement":  int(params.get("fra_agreement", False)),
            "flag_b_electronic_auction": int(params.get("electronic_auction", False)),
            "flag_b_accelerated":    int(params.get("accelerated", False)),
            "price_weight_pct":      params.get("price_weight_pct", 50.0),
        }
        return pd.DataFrame([row])

    def simulate(self, params: dict, n_samples: int = 5000, seed: int = 42) -> dict:
        """
        Run Monte Carlo simulation for given procedure parameters.

        Returns:
            dict with keys: competition, single_bid_risk, cross_border, price_ratio, duration
            Each contains: mean, median, p10, p25, p75, p90, samples (array)
        """
        rng = np.random.default_rng(seed)
        X = self._params_to_df(params)

        # ── Competition (n_offers) ────────────────────────────────
        # Point prediction from XGBoost
        comp_pred = float(self.competition_mdl["model"].predict(X)[0])
        comp_pred = max(comp_pred, 0.5)

        # Add residual noise using log-normal distribution
        # The model captures explained variance; residuals follow log-normal
        log_pred  = np.log1p(comp_pred)
        log_noise = self._comp_meta["log_std"] * 0.6  # residual std ≈ 60% of total
        log_samples = rng.normal(log_pred, log_noise, n_samples)
        comp_samples = np.expm1(log_samples).clip(0, 100)

        # ── Single-bid risk ───────────────────────────────────────
        sb_prob = float(self.singlebid_mdl["model"].predict_proba(X)[0, 1])
        sb_samples = rng.binomial(1, sb_prob, n_samples)

        # ── Cross-border win probability ──────────────────────────
        cb_prob = float(self.crossborder_mdl["model"].predict_proba(X)[0, 1])
        cb_samples = rng.binomial(1, cb_prob, n_samples)

        # ── Price ratio ───────────────────────────────────────────
        # Price model has near-zero R² → use empirical log-normal distribution
        # modulated by the model's predicted shift from the competition level
        #   lower competition → higher price ratio
        comp_effect = max(0, 1.0 - 0.015 * (comp_pred - 3.0))  # price goes up when competition is low
        price_log_mean  = self._price_meta["log_mean"] + np.log(comp_effect) * 0.3
        price_log_std   = self._price_meta["log_std"]
        price_samples   = np.exp(rng.normal(price_log_mean, price_log_std, n_samples)).clip(0.1, 3.0)

        # ── Procedure duration ────────────────────────────────────
        dur_pred  = float(np.expm1(self.duration_mdl["model"].predict(X)[0]))
        dur_pred  = max(dur_pred, 30)
        log_dur   = np.log1p(dur_pred)
        dur_noise = self._dur_meta["log_std"] * 0.5
        dur_samples = np.expm1(rng.normal(log_dur, dur_noise, n_samples)).clip(0, 1000)

        def summarise(s, is_prob=False):
            return {
                "mean":   float(np.mean(s)),
                "median": float(np.median(s)),
                "p10":    float(np.percentile(s, 10)),
                "p25":    float(np.percentile(s, 25)),
                "p75":    float(np.percentile(s, 75)),
                "p90":    float(np.percentile(s, 90)),
                "samples": s.tolist() if len(s) <= 1000 else s[:1000].tolist(),
            }

        return {
            "competition":     {**summarise(comp_samples),  "point_pred": round(comp_pred, 2)},
            "single_bid_risk": {**summarise(sb_samples),    "probability": round(sb_prob, 3)},
            "cross_border":    {**summarise(cb_samples),    "probability": round(cb_prob, 3)},
            "price_ratio":     {**summarise(price_samples)},
            "duration":        {**summarise(dur_samples),   "point_pred": round(dur_pred, 1)},
            "params":          params,
        }

    def compare(self, params_a: dict, params_b: dict,
                label_a: str = "Scenario A", label_b: str = "Scenario B",
                n_samples: int = 5000) -> dict:
        """
        Compare two procedure designs side by side.
        Returns both simulations plus delta statistics.
        """
        sim_a = self.simulate(params_a, n_samples=n_samples, seed=42)
        sim_b = self.simulate(params_b, n_samples=n_samples, seed=42)

        def delta(key, subkey="mean"):
            a = sim_a[key][subkey]
            b = sim_b[key][subkey]
            return {"a": round(a, 3), "b": round(b, 3),
                    "delta": round(b - a, 3),
                    "delta_pct": round((b - a) / abs(a) * 100, 1) if a != 0 else None}

        comparison = {
            "label_a": label_a, "label_b": label_b,
            "scenario_a": sim_a,
            "scenario_b": sim_b,
            "deltas": {
                "competition":    delta("competition",     "mean"),
                "single_bid_risk":delta("single_bid_risk", "probability"),
                "cross_border":   delta("cross_border",    "probability"),
                "price_ratio":    delta("price_ratio",     "mean"),
                "duration":       delta("duration",        "mean"),
            }
        }
        return comparison

    def empirical_benchmark(self, country=None, procedure_type=None,
                             cpv_division=None, year_from=None, year_to=None) -> dict:
        """
        Return empirical statistics from the feature store for a given filter,
        as a reference benchmark for simulated outcomes.
        """
        import polars as pl
        df = pl.read_parquet(f"{FEAT_DIR}/procedure_records.parquet")

        if country:       df = df.filter(pl.col("ISO_COUNTRY_CODE") == country)
        if procedure_type: df = df.filter(pl.col("TOP_TYPE") == procedure_type)
        if cpv_division:   df = df.filter(pl.col("cpv_division") == str(cpv_division))
        if year_from:      df = df.filter(pl.col("YEAR") >= year_from)
        if year_to:        df = df.filter(pl.col("YEAR") <= year_to)

        def stat_col(col, pct=False):
            s = df[col].drop_nulls()
            if len(s) == 0: return {"n": 0}
            return {
                "n":      len(s),
                "mean":   round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "p25":    round(float(s.quantile(0.25)), 3),
                "p75":    round(float(s.quantile(0.75)), 3),
            }

        return {
            "n_records":       len(df),
            "competition":     stat_col("n_offers"),
            "single_bid_rate": round(float(df["single_bid_flag"].drop_nulls().mean()), 3) if df["single_bid_flag"].drop_nulls().len() > 0 else None,
            "cross_border":    round(float(df["cross_border_win"].drop_nulls().mean()), 3) if df["cross_border_win"].drop_nulls().len() > 0 else None,
            "price_ratio":     stat_col("price_ratio"),
            "duration":        stat_col("proc_duration_days"),
            "prep_time":       stat_col("prep_time_days"),
        }


# ══════════════════════════════════════════════════════════════════
# TEST: Run example simulations
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    twin = ProcurementTwin()

    print("\n" + "="*60)
    print("SIMULATION TEST CASES")
    print("="*60)

    # ── Test 1: Basic open procedure, IT services, Germany ──────
    print("\n[Test 1] Open procedure – IT services – Germany – €1M")
    result = twin.simulate({
        "country":         "DE",
        "procedure_type":  "OPE",
        "contract_type":   "S",
        "cpv_division":    "72",
        "criteria":        "M",
        "price_weight_pct": 60,
        "value_euro":      1_000_000,
        "prep_time_days":  35,
        "duration_months": 24,
        "eu_funds":        False,
        "gpa":             True,
    })
    for k, v in result.items():
        if k == "params": continue
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k:20s}: mean={v['mean']:.2f}  [p25={v['p25']:.2f} – p75={v['p75']:.2f}]")

    # ── Test 2: Scenario comparison – Lowest price vs MEAT ──────
    print("\n[Test 2] Comparison: Lowest price vs MEAT – Construction – Poland – €3M")
    scenario_a = {
        "country": "PL", "procedure_type": "OPE", "contract_type": "W",
        "cpv_division": "45", "criteria": "L",
        "value_euro": 3_000_000, "prep_time_days": 40, "duration_months": 24
    }
    scenario_b = {**scenario_a, "criteria": "M", "price_weight_pct": 60}

    comp = twin.compare(scenario_a, scenario_b,
                        label_a="Lowest Price", label_b="MEAT (best value)")

    print(f"  {'Metric':25s} {'Lowest Price':>15} {'MEAT':>15} {'Delta':>12}")
    print(f"  {'-'*70}")
    for metric, delta in comp["deltas"].items():
        print(f"  {metric:25s} {delta['a']:>15.3f} {delta['b']:>15.3f} {delta['delta']:>+12.3f} ({delta['delta_pct']:+.1f}%)" if delta['delta_pct'] is not None else f"  {metric:25s} {delta['a']:>15.3f} {delta['b']:>15.3f} {delta['delta']:>+12.3f}")

    # ── Test 3: Empirical benchmark ─────────────────────────────
    print("\n[Test 3] Empirical benchmark: Germany – Open procedure – IT Services (2020-2023)")
    bench = twin.empirical_benchmark(country="DE", procedure_type="OPE",
                                      cpv_division="72", year_from=2020)
    print(f"  Records matching filter: {bench['n_records']:,}")
    print(f"  Competition:     median={bench['competition']['median']:.1f}  mean={bench['competition']['mean']:.1f}")
    print(f"  Single-bid rate: {bench['single_bid_rate']*100:.1f}%")
    print(f"  Cross-border:    {bench['cross_border']*100:.1f}%")
    print(f"  Price ratio:     median={bench['price_ratio']['median']:.3f}")
    print(f"  Duration:        median={bench['duration']['median']:.0f} days")

    # ── Test 4: Prep time sensitivity ───────────────────────────
    print("\n[Test 4] Prep time sensitivity – Healthcare – Romania – €500k")
    base_params = {
        "country": "RO", "procedure_type": "OPE", "contract_type": "U",
        "cpv_division": "85", "criteria": "M",
        "value_euro": 500_000, "duration_months": 12
    }
    print(f"  {'Prep time':>12} {'E[offers]':>12} {'P(single bid)':>15} {'P(cross-border)':>17}")
    print(f"  {'-'*60}")
    for prep_days in [22, 30, 35, 45, 52]:
        r = twin.simulate({**base_params, "prep_time_days": prep_days})
        print(f"  {prep_days:>10}d  {r['competition']['mean']:>12.2f}  {r['single_bid_risk']['probability']:>15.3f}  {r['cross_border']['probability']:>17.3f}")

    print("\n✅ Phase 3 complete — Simulation engine ready.\n")
