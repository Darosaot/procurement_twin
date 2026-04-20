"""
Phase 3: Procurement Digital Twin — Simulation Engine  (v2)
=============================================================
Core class: ProcurementTwin

Methods:
  .simulate(params, n_samples=5000)   → Monte Carlo outcome distributions
  .compare(params_a, params_b)        → side-by-side comparison
  .empirical_benchmark(filters)       → historical stats from feature store
  .compute_shap(params)               → SHAP values for a single prediction
  .policy_simulation(segment, intervention, n_records=500)  → counterfactual

Changes in v2:
  - Price model now uses competition_hat (2-stage IV approach)
  - Calibration offsets applied per CPV and country_cluster
  - SHAP values computed on demand via compute_shap()
  - policy_simulation() for aggregate counterfactual analysis
"""

import numpy as np
import pandas as pd
import pickle, json, os, io, logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Block modules that can execute arbitrary code or access the filesystem.
# Our model files come from our own controlled training pipeline, so an
# exhaustive allowlist causes more breakage than it prevents. We block the
# dangerous namespace instead and let everything else through.
_PICKLE_BLOCKED_TOP = frozenset({
    "os", "posix", "nt",                         # filesystem / OS
    "sys",                                        # sys.exit, path manipulation
    "subprocess", "_subprocess",                  # shell execution
    "socket",                                     # network
    "ctypes", "_ctypes",                          # C-level execution
    "importlib", "imp",                           # dynamic imports
    "marshal",                                    # bytecode loading
    "code", "codeop",                             # REPL / code objects
    "pty", "tty", "signal", "mmap",               # low-level system
    "shutil", "pathlib", "tempfile",              # filesystem helpers
    "distutils", "setuptools", "pip",             # package ops
    "atexit",                                     # exit hooks
})

_PICKLE_BLOCKED_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
})


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        top = module.split(".")[0]
        if top in _PICKLE_BLOCKED_TOP:
            raise pickle.UnpicklingError(f"Blocked module in pickle: {module}.{name}")
        if module == "builtins" and name in _PICKLE_BLOCKED_BUILTINS:
            raise pickle.UnpicklingError(f"Blocked builtin in pickle: {name}")
        return super().find_class(module, name)

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
MODEL_DIR  = os.environ.get("MODEL_DIR",  os.path.join(_PROJ_ROOT, "models"))
FEAT_DIR   = os.environ.get("FEAT_DIR",   os.path.join(_PROJ_ROOT, "data", "features"))
DATA_DIR   = os.environ.get("DATA_DIR",   os.path.join(_PROJ_ROOT, "data"))

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
    if v < 135_000:   return "Below 135k"
    if v < 215_000:   return "135k-215k"
    if v < 431_000:   return "215k-431k"
    if v < 5_000_000: return "431k-5M"
    if v < 50_000_000: return "5M-50M"
    return ">50M"


def _bootstrap_ci(deltas: np.ndarray, n_boot: int = 1000,
                   rng: np.random.Generator = None) -> np.ndarray:
    """
    Parametric bootstrap of the mean delta across n_boot resamples.

    Parameters
    ----------
    deltas  : 1-D array of per-record (counterfactual − baseline) differences
    n_boot  : number of bootstrap resamples (default 1 000)
    rng     : numpy Generator for reproducibility

    Returns
    -------
    1-D array of length n_boot with bootstrap mean estimates.
    Caller takes np.percentile(result, [2.5, 97.5]) for the 95 % CI.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(deltas)
    if n == 0:
        return np.zeros(n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))   # (n_boot, n) index matrix
    return deltas[idx].mean(axis=1)               # (n_boot,) bootstrap means


class ProcurementTwin:
    """
    The Procurement Digital Twin simulation engine.
    Loads trained models and runs Monte Carlo simulations.
    """

    def __init__(self):
        logger.info("Loading models...")
        self.competition_mdl = self._load("competition_model")
        self.singlebid_mdl   = self._load("single_bid_model")
        self.crossborder_mdl = self._load("crossborder_model")
        self.price_mdl       = self._load("price_model")
        self.duration_mdl    = self._load("duration_model")
        self.feature_spec    = self._load_json("feature_spec.json")

        # Metadata for Monte Carlo noise
        self._comp_meta  = self.competition_mdl["meta"]
        self._price_meta = self.price_mdl["meta"]
        self._dur_meta   = self.duration_mdl["meta"]

        # Calibration offsets (may not exist in older builds)
        try:
            self._calibration = self._load_json("calibration_offsets.json")
        except FileNotFoundError:
            self._calibration = {}

        # Whether price model uses competition_hat
        self._price_uses_comp_hat = self._price_meta.get("uses_competition_hat", False)

        # SHAP importances (pre-computed, may not exist)
        try:
            self._shap_global = self._load_json("shap_importances.json")
        except FileNotFoundError:
            self._shap_global = {}

        logger.info("All models loaded.")

    def _load(self, name):
        path = f"{MODEL_DIR}/{name}.pkl"
        with open(path, "rb") as f:
            try:
                return _SafeUnpickler(f).load()
            except pickle.UnpicklingError as e:
                logger.error("Blocked unsafe pickle in %s: %s", path, e)
                raise

    def _load_json(self, name):
        with open(f"{MODEL_DIR}/{name}", "r") as f:
            return json.load(f)

    # Default log10 value used when value_euro is missing (≈ median of training set ~€500k)
    _LOG10_VALUE_DEFAULT = 5.70

    def _params_to_df(self, params: dict, include_comp_hat: float = None) -> pd.DataFrame:
        """Convert user-supplied params dict to a model-ready DataFrame."""
        _defaults_applied: list[str] = []

        v = params.get("value_euro", None)
        if not (v and v > 0 and np.isfinite(v)):
            _defaults_applied.append(f"value_euro={v!r} → log10_value={self._LOG10_VALUE_DEFAULT} (~€500k median)")
        log10_v = (np.log10(v) if (v and v > 0 and np.isfinite(v))
                   else self._LOG10_VALUE_DEFAULT)

        def _num(val, default, name=""):
            try:
                f = float(val)
                if np.isnan(f) or np.isinf(f):
                    _defaults_applied.append(f"{name}={val!r} → {default}")
                    return default
                return f
            except (TypeError, ValueError):
                _defaults_applied.append(f"{name}={val!r} → {default}")
                return default

        def _str(val, default, name=""):
            s = str(val) if val is not None else default
            if s in ("nan", "None", ""):
                _defaults_applied.append(f"{name}={val!r} → {default!r}")
                return default
            return s

        row = {
            "ISO_COUNTRY_CODE":      _str(params.get("country"),         "DE",  "country"),
            "TOP_TYPE":              _str(params.get("procedure_type"),   "OPE", "procedure_type"),
            "TYPE_OF_CONTRACT":      _str(params.get("contract_type"),    "S",   "contract_type"),
            "cpv_division":          _str(params.get("cpv_division"),     "72",  "cpv_division"),
            "CRIT_CODE":             _str(params.get("criteria"),         "M",   "criteria"),
            "value_bracket":         value_bracket(v),
            "country_cluster":       COUNTRY_CLUSTERS.get(
                                         _str(params.get("country"), "DE"), "Other"),
            "log10_value":           log10_v,
            "prep_time_days":        _num(params.get("prep_time_days"),   35.0, "prep_time_days"),
            "contract_duration_months": _num(params.get("duration_months"), 24.0, "duration_months"),
            "flag_b_gpa":            int(_num(params.get("gpa", 0), 0)),
            "flag_b_eu_funds":       int(_num(params.get("eu_funds", 0), 0)),
            "flag_b_fra_agreement":  int(_num(params.get("fra_agreement", 0), 0)),
            "flag_b_electronic_auction": int(_num(params.get("electronic_auction", 0), 0)),
            "flag_b_accelerated":    int(_num(params.get("accelerated", 0), 0)),
            "price_weight_pct":      _num(params.get("price_weight_pct"), 50.0, "price_weight_pct"),
        }

        if _defaults_applied:
            logger.warning("Simulation defaults applied: %s", "; ".join(_defaults_applied))
        if include_comp_hat is not None:
            row["competition_hat"] = include_comp_hat

        return pd.DataFrame([row])

    def _get_calibration_offset(self, model_key: str, cpv_div: str, cluster: str) -> float:
        """Return the median calibration offset for this segment."""
        if not self._calibration or model_key not in self._calibration:
            return 0.0
        offsets = self._calibration[model_key]
        cpv_off = offsets.get("by_cpv", {}).get(str(cpv_div), 0.0) or 0.0
        clu_off = offsets.get("by_cluster", {}).get(cluster, 0.0) or 0.0
        # Blend: 60% CPV effect, 40% cluster effect
        return 0.6 * cpv_off + 0.4 * clu_off

    def simulate(self, params: dict, n_samples: int = 5000, seed: int = 42) -> dict:
        """
        Run Monte Carlo simulation for given procedure parameters.

        Returns:
            dict with keys: competition, single_bid_risk, cross_border, price_ratio,
                            duration, params
        """
        rng = np.random.default_rng(seed)
        X   = self._params_to_df(params)
        cpv = str(params.get("cpv_division", "72"))
        cluster = COUNTRY_CLUSTERS.get(params.get("country", "DE"), "Other")

        # ── Stage 1: Competition ──────────────────────────────────
        # Model is trained on log1p(n_bids), so predict() returns log1p-space values.
        # Apply calibration directly in log space, then convert back to bid counts.
        comp_pred_log = max(float(self.competition_mdl["model"].predict(X)[0]), np.log1p(0.5))
        comp_cal  = self._get_calibration_offset("competition", cpv, cluster)
        comp_pred = float(np.expm1(comp_pred_log + comp_cal))

        log_pred  = np.log1p(comp_pred)
        log_noise = self._comp_meta["log_std"] * 0.6
        log_samples = rng.normal(log_pred, log_noise, n_samples)
        comp_samples = np.expm1(log_samples).clip(0, 100)

        # ── Single-bid risk ───────────────────────────────────────
        sb_prob = float(self.singlebid_mdl["model"].predict_proba(X)[0, 1])
        sb_samples = rng.binomial(1, sb_prob, n_samples)

        # ── Cross-border win ──────────────────────────────────────
        cb_prob = float(self.crossborder_mdl["model"].predict_proba(X)[0, 1])
        cb_samples = rng.binomial(1, cb_prob, n_samples)

        # ── Price ratio (2-stage IV) ──────────────────────────────
        price_cal = self._get_calibration_offset("price_ratio", cpv, cluster)

        if self._price_uses_comp_hat:
            X_price = self._params_to_df(params, include_comp_hat=comp_pred)
            log_price_pred = float(self.price_mdl["model"].predict(X_price)[0])
        else:
            # Fallback: old model without competition_hat
            # Use competition-modulated empirical distribution
            comp_effect = max(0, 1.0 - 0.015 * (comp_pred - 3.0))
            log_price_pred = self._price_meta["log_mean"] + np.log(max(comp_effect, 0.1)) * 0.3

        log_price_pred += price_cal
        price_samples = np.exp(
            rng.normal(log_price_pred, self._price_meta["log_std"] * 0.7, n_samples)
        ).clip(0.1, 3.0)

        # ── Procedure duration ────────────────────────────────────
        dur_pred  = float(np.expm1(self.duration_mdl["model"].predict(X)[0]))
        dur_pred  = max(dur_pred, 30)
        log_dur   = np.log1p(dur_pred)
        dur_noise = self._dur_meta["log_std"] * 0.5
        dur_samples = np.expm1(rng.normal(log_dur, dur_noise, n_samples)).clip(0, 1000)

        def summarise(s):
            return {
                "mean":   float(np.mean(s)),
                "median": float(np.median(s)),
                "p10":    float(np.percentile(s, 10)),
                "p25":    float(np.percentile(s, 25)),
                "p75":    float(np.percentile(s, 75)),
                "p90":    float(np.percentile(s, 90)),
                "samples": s[:1000].tolist(),
            }

        return {
            "competition":     {**summarise(comp_samples),  "point_pred": round(comp_pred, 2)},
            "single_bid_risk": {**summarise(sb_samples),    "probability": round(sb_prob, 3)},
            "cross_border":    {**summarise(cb_samples),    "probability": round(cb_prob, 3)},
            "price_ratio":     {**summarise(price_samples), "point_pred": round(float(np.exp(log_price_pred)), 3)},
            "duration":        {**summarise(dur_samples),   "point_pred": round(dur_pred, 1)},
            "params":          params,
        }

    def compare(self, params_a: dict, params_b: dict,
                label_a: str = "Scenario A", label_b: str = "Scenario B",
                n_samples: int = 5000) -> dict:
        """Compare two procedure designs side by side."""
        sim_a = self.simulate(params_a, n_samples=n_samples, seed=42)
        sim_b = self.simulate(params_b, n_samples=n_samples, seed=42)

        def delta(key, subkey="mean"):
            a = sim_a[key][subkey]
            b = sim_b[key][subkey]
            return {"a": round(a, 3), "b": round(b, 3),
                    "delta": round(b - a, 3),
                    "delta_pct": round((b - a) / abs(a) * 100, 1) if a != 0 else (0.0 if b == 0 else None)}

        return {
            "label_a": label_a, "label_b": label_b,
            "scenario_a": sim_a, "scenario_b": sim_b,
            "deltas": {
                "competition":     delta("competition",     "mean"),
                "single_bid_risk": delta("single_bid_risk", "probability"),
                "cross_border":    delta("cross_border",    "probability"),
                "price_ratio":     delta("price_ratio",     "mean"),
                "duration":        delta("duration",        "mean"),
            }
        }

    def empirical_benchmark(self, country=None, procedure_type=None,
                             cpv_division=None, year_from=None, year_to=None) -> dict:
        """Return empirical statistics from the feature store for a given filter."""
        import polars as pl
        df = pl.read_parquet(f"{FEAT_DIR}/procedure_records.parquet")

        if country:        df = df.filter(pl.col("ISO_COUNTRY_CODE") == country)
        if procedure_type: df = df.filter(pl.col("TOP_TYPE") == procedure_type)
        if cpv_division:   df = df.filter(pl.col("cpv_division") == str(cpv_division))
        if year_from:      df = df.filter(pl.col("YEAR") >= year_from)
        if year_to:        df = df.filter(pl.col("YEAR") <= year_to)

        n_total = len(df)

        def stat_col(col):
            s = df[col].drop_nulls()
            n_valid = len(s)
            if n_valid == 0:
                return {"n_total": n_total, "n_valid": 0, "coverage": 0.0}
            return {
                "n_total":  n_total,
                "n_valid":  n_valid,
                "coverage": round(n_valid / n_total, 4) if n_total > 0 else 0.0,
                "mean":     round(float(s.mean()), 3),
                "median":   round(float(s.median()), 3),
                "p25":      round(float(s.quantile(0.25)), 3),
                "p75":      round(float(s.quantile(0.75)), 3),
            }

        def rate_col(col):
            s = df[col].drop_nulls()
            n_valid = len(s)
            if n_valid == 0:
                return {"n_total": n_total, "n_valid": 0, "coverage": 0.0, "rate": None}
            return {
                "n_total":  n_total,
                "n_valid":  n_valid,
                "coverage": round(n_valid / n_total, 4) if n_total > 0 else 0.0,
                "rate":     round(float(s.mean()), 3),
            }

        return {
            "n_records":       n_total,
            "competition":     stat_col("n_offers"),
            "single_bid_rate": rate_col("single_bid_flag"),
            "cross_border":    rate_col("cross_border_win"),
            "price_ratio":     stat_col("price_ratio"),
            "duration":        stat_col("proc_duration_days"),
            "prep_time":       stat_col("prep_time_days"),
        }

    def compute_shap(self, params: dict) -> dict:
        """
        Compute per-prediction feature contributions for the competition and
        single-bid risk models.

        Competition model (XGBoost):
            Uses XGBoost's native pred_contribs=True via the Booster API.
            Each value is the additive SHAP contribution of that feature.

        Single-bid model (LogisticRegression):
            Linear contributions: coef_i × (x_i − mean_i), projected into
            log-odds space.  The intercept is the base log-odds.

        Returns a dict with 'competition' and 'single_bid' keys, each
        containing 'shap_values' (feature→value), 'base_value', and
        'prediction'.
        """
        import xgboost as xgb

        X = self._params_to_df(params)
        results = {}

        # ── Competition model: XGBoost native SHAP ────────────────────
        try:
            mdl   = self.competition_mdl["model"]
            X_enc = mdl["pre"].transform(X)
            feat_names = [
                n.replace("num__", "").replace("cat__", "")
                for n in mdl["pre"].get_feature_names_out()
            ]
            booster   = mdl["mdl"].get_booster()
            dmat      = xgb.DMatrix(X_enc, feature_names=feat_names)
            contribs  = booster.predict(dmat, pred_contribs=True)  # shape (1, n_features+1)
            base_val  = float(contribs[0, -1])                     # last col = bias
            shap_vals = {
                feat_names[i]: round(float(contribs[0, i]), 5)
                for i in range(len(feat_names))
            }
            # Top-20 by absolute magnitude
            top20 = dict(
                sorted(shap_vals.items(), key=lambda kv: -abs(kv[1]))[:20]
            )
            pred_log1p = base_val + sum(shap_vals.values())
            results["competition"] = {
                "base_value":  round(base_val, 4),
                "shap_values": top20,
                "prediction":  round(float(np.expm1(pred_log1p)), 3),
                "method":      "XGBoost pred_contribs",
            }
        except Exception as e:
            results["competition"] = {"error": str(e)}

        # ── Single-bid model: LogisticRegression linear contributions ──
        try:
            mdl   = self.singlebid_mdl["model"]
            X_enc = mdl["pre"].transform(X)
            feat_names = [
                n.replace("num__", "").replace("cat__", "")
                for n in mdl["pre"].get_feature_names_out()
            ]
            lr        = mdl["mdl"]
            coef      = lr.coef_[0]          # shape (n_features,)
            intercept = float(lr.intercept_[0])
            x_vec     = np.asarray(X_enc[0])  # (n_features,)
            contribs  = coef * x_vec          # element-wise linear contribution
            shap_vals = {
                feat_names[i]: round(float(contribs[i]), 5)
                for i in range(len(feat_names))
            }
            top20 = dict(
                sorted(shap_vals.items(), key=lambda kv: -abs(kv[1]))[:20]
            )
            log_odds  = intercept + float(np.dot(coef, x_vec))
            prob      = float(1 / (1 + np.exp(-log_odds)))
            results["single_bid"] = {
                "base_value":  round(intercept, 4),
                "shap_values": top20,
                "prediction":  round(prob, 4),
                "method":      "LogReg linear contributions (coef × feature)",
            }
        except Exception as e:
            results["single_bid"] = {"error": str(e)}

        return results

    def get_global_shap(self) -> dict:
        """Return pre-computed global mean SHAP importances."""
        return self._shap_global

    def optimize(self, base_params: dict, objective_weights: dict,
                 constraints: dict = None, n_samples: int = 500,
                 seed: int = 42) -> dict:
        """
        Multi-objective optimisation engine.

        Sweeps over controllable procedure parameters (procedure type, criteria,
        price weight, prep time, e-auction) and ranks candidate configurations
        by a weighted utility score across all five outcome dimensions.

        Parameters
        ----------
        base_params : dict
            Fixed context (country, contract_type, cpv_division, value_euro, …)
        objective_weights : dict
            Signed weight for each outcome.
            Positive = maximise, Negative = minimise.
            Keys: competition, single_bid_risk, cross_border, price_ratio, duration
        constraints : dict, optional
            allowed_procedure_types : list[str]
            min_prep_time : float
            max_prep_time : float
            must_use_meat : bool
        n_samples : int
            Monte Carlo samples per candidate (lower = faster, ≥300 recommended)
        seed : int

        Returns
        -------
        dict
            candidates        — top-20 configurations with scores and outcomes
            pareto_frontier   — non-dominated solutions for top-2 objectives
            best              — single best configuration
            objective_weights — echo of weights used
            search_space      — number of candidates evaluated
        """
        if constraints is None:
            constraints = {}

        allowed_procs = constraints.get(
            "allowed_procedure_types", ["OPE", "RES", "NIC", "COD"]
        )
        min_prep = float(constraints.get("min_prep_time", 21))
        max_prep = float(constraints.get("max_prep_time", 90))
        must_meat = bool(constraints.get("must_use_meat", False))

        # Prep-time grid (honour constraints)
        all_preps = [21.0, 35.0, 52.0, 65.0, 80.0]
        prep_options = [p for p in all_preps if min_prep <= p <= max_prep]
        if not prep_options:
            prep_options = [max(min_prep, 21.0)]

        # Generate candidate grid
        candidates_params = []
        crit_options = ["M"] if must_meat else ["M", "L"]
        for proc in allowed_procs:
            for crit in crit_options:
                pw_opts = [20.0, 50.0, 80.0] if crit == "M" else [100.0]
                for pw in pw_opts:
                    for prep in prep_options:
                        for ea in [False, True]:
                            candidates_params.append({
                                **base_params,
                                "procedure_type":    proc,
                                "criteria":          crit,
                                "price_weight_pct":  pw,
                                "prep_time_days":    prep,
                                "electronic_auction": ea,
                            })

        if not candidates_params:
            return {"error": "No candidates generated — check constraints.", "candidates": []}

        # Evaluate all candidates
        outcome_keys = ["competition", "single_bid_risk", "cross_border",
                        "price_ratio", "duration"]
        raw_results = []
        for i, cp in enumerate(candidates_params):
            sim = self.simulate(cp, n_samples=n_samples, seed=seed + i)
            raw_results.append({
                "params": cp,
                "outcomes": {
                    "competition":     sim["competition"]["mean"],
                    "single_bid_risk": sim["single_bid_risk"]["probability"],
                    "cross_border":    sim["cross_border"]["probability"],
                    "price_ratio":     sim["price_ratio"]["mean"],
                    "duration":        sim["duration"]["mean"],
                },
            })

        # Normalise each outcome to [0,1] across the candidate pool
        vals = {k: np.array([r["outcomes"][k] for r in raw_results])
                for k in outcome_keys}
        lo = {k: float(vals[k].min()) for k in outcome_keys}
        hi = {k: float(vals[k].max()) for k in outcome_keys}

        def _norm(key, val):
            rng = hi[key] - lo[key]
            return (val - lo[key]) / rng if rng > 1e-12 else 0.5

        weights = {k: float(objective_weights.get(k, 0.0)) for k in outcome_keys}

        for r in raw_results:
            score = sum(weights[k] * _norm(k, r["outcomes"][k]) for k in outcome_keys)
            r["utility_score"] = round(float(score), 5)

        raw_results.sort(key=lambda x: -x["utility_score"])

        def _fmt(r, rank):
            p = r["params"]
            return {
                "rank":               rank,
                "utility_score":      r["utility_score"],
                "procedure_type":     p["procedure_type"],
                "criteria":           p["criteria"],
                "price_weight_pct":   p["price_weight_pct"],
                "prep_time_days":     p["prep_time_days"],
                "electronic_auction": p["electronic_auction"],
                "outcomes":           {k: round(v, 4) for k, v in r["outcomes"].items()},
            }

        top20 = [_fmt(r, i + 1) for i, r in enumerate(raw_results[:20])]

        # Pareto frontier for top-2 objectives by absolute weight
        sorted_w = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
        pareto_objs = [k for k, w in sorted_w[:2] if w != 0.0]
        pareto = []
        if len(pareto_objs) >= 2:
            obj1, obj2 = pareto_objs
            sign1 = 1.0 if weights[obj1] > 0 else -1.0
            sign2 = 1.0 if weights[obj2] > 0 else -1.0

            def _dominated(a, b):
                a1, a2 = sign1 * a["outcomes"][obj1], sign2 * a["outcomes"][obj2]
                b1, b2 = sign1 * b["outcomes"][obj1], sign2 * b["outcomes"][obj2]
                return b1 >= a1 and b2 >= a2 and (b1 > a1 or b2 > a2)

            for r in raw_results:
                if not any(_dominated(r, other)
                           for other in raw_results if other is not r):
                    pareto.append({
                        obj1:              round(r["outcomes"][obj1], 4),
                        obj2:              round(r["outcomes"][obj2], 4),
                        "utility_score":   r["utility_score"],
                        "procedure_type":  r["params"]["procedure_type"],
                        "criteria":        r["params"]["criteria"],
                        "prep_time_days":  r["params"]["prep_time_days"],
                        "electronic_auction": r["params"]["electronic_auction"],
                    })

        return {
            "best":              top20[0] if top20 else None,
            "candidates":        top20,
            "pareto_frontier":   pareto,
            "pareto_objectives": pareto_objs,
            "objective_weights": weights,
            "search_space":      {"n_candidates_evaluated": len(raw_results)},
        }

    def policy_compare(self, segment_filters: dict, policies: list,
                       n_records: int = 300, seed: int = 0) -> dict:
        """
        Compare multiple policy interventions against the status-quo baseline.

        Parameters
        ----------
        segment_filters : dict
            country_cluster, cpv_division, TOP_TYPE, year_from, year_to
        policies : list[dict]
            Each entry: {"name": str, "intervention": dict | None}
            Use intervention=None for "Status Quo" entries.
            intervention format: {"param": str, "delta": float} or {"param": str, "value": any}
        n_records : int
            Historical records to sample

        Returns
        -------
        dict
            baseline   — status-quo aggregate outcomes
            policies   — per-policy outcomes and deltas vs baseline
            n_matched  — records matching the filters
            n_simulated — records actually simulated
        """
        import polars as pl
        df = pl.read_parquet(f"{FEAT_DIR}/procedure_records.parquet").to_pandas()

        if segment_filters.get("country_cluster"):
            df = df[df["country_cluster"] == segment_filters["country_cluster"]]
        if segment_filters.get("cpv_division"):
            df = df[df["cpv_division"] == str(segment_filters["cpv_division"])]
        if segment_filters.get("TOP_TYPE"):
            df = df[df["TOP_TYPE"] == segment_filters["TOP_TYPE"]]
        yf = segment_filters.get("year_from", 2018)
        yt = segment_filters.get("year_to", 2023)
        df = df[df["YEAR"].between(yf, yt)]

        n_matched = len(df)
        if n_matched == 0:
            return {"error": "No records match these filters.", "n_matched": 0}

        rng = np.random.default_rng(seed)
        sample = df.sample(min(n_records, n_matched),
                           random_state=seed).reset_index(drop=True)
        n_sim = len(sample)

        def _s(val, default):
            if val is None:
                return default
            try:
                f = float(val)
                return default if (np.isnan(f) or np.isinf(f)) else f
            except (TypeError, ValueError):
                return val if val else default

        def _ss(val, default):
            s = str(val) if val is not None else default
            return default if s in ("nan", "None", "") else s

        base_records = [
            {
                "country":            _ss(row.get("ISO_COUNTRY_CODE"), "DE"),
                "procedure_type":     _ss(row.get("TOP_TYPE"), "OPE"),
                "contract_type":      _ss(row.get("TYPE_OF_CONTRACT"), "S"),
                "cpv_division":       _ss(row.get("cpv_division"), "72"),
                "criteria":           _ss(row.get("CRIT_CODE"), "M"),
                "value_euro":         _s(row.get("VALUE_EURO"), 1_000_000),
                "prep_time_days":     _s(row.get("prep_time_days"), 35.0),
                "duration_months":    _s(row.get("contract_duration_months"), 24.0),
                "price_weight_pct":   _s(row.get("price_weight_pct"), 50.0),
                "gpa":                bool(int(_s(row.get("flag_b_gpa"), 0))),
                "eu_funds":           bool(int(_s(row.get("flag_b_eu_funds"), 0))),
                "fra_agreement":      bool(int(_s(row.get("flag_b_fra_agreement"), 0))),
                "electronic_auction": bool(int(_s(row.get("flag_b_electronic_auction"), 0))),
                "accelerated":        bool(int(_s(row.get("flag_b_accelerated"), 0))),
            }
            for _, row in sample.iterrows()
        ]

        outcome_keys = ["competition", "single_bid_risk", "price_ratio", "duration"]
        n_seeds = n_sim * (len(policies) + 1)
        seeds = rng.integers(0, 1_000_000, size=n_seeds).tolist()
        sid = 0

        # Simulate status-quo baseline
        baseline = {k: np.zeros(n_sim) for k in outcome_keys}
        for i, bp in enumerate(base_records):
            sim = self.simulate(bp, n_samples=200, seed=seeds[sid]); sid += 1
            baseline["competition"][i]     = sim["competition"]["mean"]
            baseline["single_bid_risk"][i] = sim["single_bid_risk"]["probability"]
            baseline["price_ratio"][i]     = sim["price_ratio"]["mean"]
            baseline["duration"][i]        = sim["duration"]["mean"]

        def _agg(arr_c, arr_b):
            delta = arr_c - arr_b
            return {
                "mean":              round(float(arr_c.mean()), 3),
                "delta_vs_baseline": round(float(delta.mean()), 3),
                "pct_delta":         round(float(
                    delta.mean() / (abs(arr_b.mean()) + 1e-9) * 100), 1),
                "ci_95_lo": round(float(np.percentile(delta, 2.5)), 3),
                "ci_95_hi": round(float(np.percentile(delta, 97.5)), 3),
            }

        baseline_agg = {
            k: {"mean": round(float(baseline[k].mean()), 3),
                "delta_vs_baseline": 0.0, "pct_delta": 0.0,
                "ci_95_lo": 0.0, "ci_95_hi": 0.0}
            for k in outcome_keys
        }

        # Simulate each policy
        policy_results = []
        for policy in policies:
            intervention = policy.get("intervention")
            pol = {k: np.zeros(n_sim) for k in outcome_keys}

            for i, bp in enumerate(base_records):
                if intervention is None:
                    for k in outcome_keys:
                        pol[k][i] = baseline[k][i]
                    sid += 1
                else:
                    cf = bp.copy()
                    pk = intervention.get("param")
                    pdelta = intervention.get("delta")
                    pval   = intervention.get("value")
                    if pdelta is not None and pk in cf:
                        cf[pk] = float(cf.get(pk) or 0) + pdelta
                    elif pval is not None:
                        cf[pk] = pval
                    sim = self.simulate(cf, n_samples=200, seed=seeds[sid]); sid += 1
                    pol["competition"][i]     = sim["competition"]["mean"]
                    pol["single_bid_risk"][i] = sim["single_bid_risk"]["probability"]
                    pol["price_ratio"][i]     = sim["price_ratio"]["mean"]
                    pol["duration"][i]        = sim["duration"]["mean"]

            policy_results.append({
                "name":         policy.get("name", "Policy"),
                "intervention": intervention,
                "outcomes":     {k: _agg(pol[k], baseline[k]) for k in outcome_keys},
            })

        return {
            "n_matched":        n_matched,
            "n_simulated":      n_sim,
            "segment_filters":  segment_filters,
            "baseline":         {"name": "Status Quo", "outcomes": baseline_agg},
            "policies":         policy_results,
        }

    def policy_simulation(self, segment_filters: dict, intervention: dict,
                          n_records: int = 500, seed: int = 0) -> dict:
        """
        Aggregate counterfactual simulation.

        Parameters:
          segment_filters : dict with keys country_cluster, cpv_division, TOP_TYPE, year_from, year_to
          intervention    : dict like {"param": "prep_time_days", "delta": 14}
                            or {"param": "criteria", "value": "M"}
          n_records       : sample size from the matched historical records

        Returns:
          dict with aggregate stats and per-record impact distributions
        """
        import polars as pl
        df = pl.read_parquet(f"{FEAT_DIR}/procedure_records.parquet").to_pandas()

        # Apply segment filters
        if segment_filters.get("country_cluster"):
            df = df[df["country_cluster"] == segment_filters["country_cluster"]]
        if segment_filters.get("cpv_division"):
            df = df[df["cpv_division"] == segment_filters["cpv_division"]]
        if segment_filters.get("TOP_TYPE"):
            df = df[df["TOP_TYPE"] == segment_filters["TOP_TYPE"]]
        yf = segment_filters.get("year_from", 2018)
        yt = segment_filters.get("year_to",   2023)
        df = df[df["YEAR"].between(yf, yt)]

        n_matched = len(df)
        if n_matched == 0:
            return {"error": "No records match these filters", "n_matched": 0}

        # Sample records
        rng = np.random.default_rng(seed)
        sample = df.sample(min(n_records, n_matched), random_state=seed)

        outcomes_baseline     = {"competition": [], "single_bid_risk": [],
                                  "price_ratio": [], "duration": []}
        outcomes_counterfactual = {k: [] for k in outcomes_baseline}

        param_key   = intervention.get("param", "prep_time_days")
        param_delta = intervention.get("delta", None)    # numeric change
        param_value = intervention.get("value", None)    # categorical override

        def _safe(val, default):
            """Return default if val is NaN/None/inf."""
            if val is None: return default
            try:
                f = float(val)
                return default if (np.isnan(f) or np.isinf(f)) else f
            except (TypeError, ValueError):
                return val if val else default

        def _safe_str(val, default):
            s = str(val) if val is not None else default
            return default if s in ("nan","None","") else s

        for _, row in sample.iterrows():
            # Reconstruct params from historical record
            base_params = {
                "country":          _safe_str(row.get("ISO_COUNTRY_CODE"), "DE"),
                "procedure_type":   _safe_str(row.get("TOP_TYPE"), "OPE"),
                "contract_type":    _safe_str(row.get("TYPE_OF_CONTRACT"), "S"),
                "cpv_division":     _safe_str(row.get("cpv_division"), "72"),
                "criteria":         _safe_str(row.get("CRIT_CODE"), "M"),
                "value_euro":       _safe(row.get("VALUE_EURO"), 1_000_000),
                "prep_time_days":   _safe(row.get("prep_time_days"), 35.0),
                "duration_months":  _safe(row.get("contract_duration_months"), 24.0),
                "price_weight_pct": _safe(row.get("price_weight_pct"), 50.0),
                "gpa":              bool(int(_safe(row.get("flag_b_gpa"), 0))),
                "eu_funds":         bool(int(_safe(row.get("flag_b_eu_funds"), 0))),
                "fra_agreement":    bool(int(_safe(row.get("flag_b_fra_agreement"), 0))),
                "electronic_auction": bool(int(_safe(row.get("flag_b_electronic_auction"), 0))),
                "accelerated":      bool(int(_safe(row.get("flag_b_accelerated"), 0))),
            }

            # Apply intervention
            cf_params = base_params.copy()
            if param_delta is not None and param_key in cf_params:
                cf_params[param_key] = float(cf_params.get(param_key, 0) or 0) + param_delta
            elif param_value is not None:
                cf_params[param_key] = param_value

            # Simulate baseline and counterfactual (deterministic, no MC noise)
            for params_dict, store in [(base_params, outcomes_baseline),
                                       (cf_params, outcomes_counterfactual)]:
                sim = self.simulate(params_dict, n_samples=200,
                                    seed=int(rng.integers(0, 100000)))
                store["competition"].append(sim["competition"]["mean"])
                store["single_bid_risk"].append(sim["single_bid_risk"]["probability"])
                store["price_ratio"].append(sim["price_ratio"]["mean"])
                store["duration"].append(sim["duration"]["mean"])

        # Compute aggregate impact with bootstrap CIs
        results = {}
        for key in outcomes_baseline:
            b     = np.array(outcomes_baseline[key])
            c     = np.array(outcomes_counterfactual[key])
            delta = c - b

            # Bootstrap 95 % CI on the mean delta (1 000 resamples)
            boot_means = _bootstrap_ci(delta, n_boot=1000, rng=rng)
            ci_lo = round(float(np.percentile(boot_means, 2.5)),  3)
            ci_hi = round(float(np.percentile(boot_means, 97.5)), 3)

            results[key] = {
                "baseline_mean":       round(float(b.mean()), 3),
                "counterfactual_mean": round(float(c.mean()), 3),
                "mean_delta":          round(float(delta.mean()), 3),
                "ci_95_lo":            ci_lo,
                "ci_95_hi":            ci_hi,
                "pct_delta":           round(float(delta.mean() / (abs(b.mean()) + 1e-9) * 100), 1),
                "delta_p25":           round(float(np.percentile(delta, 25)), 3),
                "delta_p75":           round(float(np.percentile(delta, 75)), 3),
                "delta_samples":       delta[:200].tolist(),
            }

        return {
            "n_matched": n_matched,
            "n_simulated": len(sample),
            "intervention": intervention,
            "segment_filters": segment_filters,
            "outcomes": results,
        }


# ══════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    twin = ProcurementTwin()

    print("\n[Test 1] IT services – Germany – €1M open procedure")
    r = twin.simulate({
        "country": "DE", "procedure_type": "OPE", "contract_type": "S",
        "cpv_division": "72", "criteria": "M", "price_weight_pct": 60,
        "value_euro": 1_000_000, "prep_time_days": 35, "duration_months": 24,
    })
    for k, v in r.items():
        if k == "params": continue
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k:20s}: mean={v['mean']:.2f}  [p25={v['p25']:.2f}–p75={v['p75']:.2f}]")

    print("\n[Test 2] Policy simulation – prep time +14 days, CEE construction")
    ps = twin.policy_simulation(
        segment_filters={"country_cluster": "CEE", "cpv_division": "45",
                         "TOP_TYPE": "OPE", "year_from": 2020, "year_to": 2022},
        intervention={"param": "prep_time_days", "delta": 14},
        n_records=100,
    )
    if "error" not in ps:
        print(f"  Matched {ps['n_matched']:,} records, simulated {ps['n_simulated']}")
        for k, v in ps["outcomes"].items():
            print(f"  {k:20s}: baseline={v['baseline_mean']:.3f}  "
                  f"counterfactual={v['counterfactual_mean']:.3f}  "
                  f"delta={v['mean_delta']:+.3f} ({v['pct_delta']:+.1f}%)")
    else:
        print(f"  {ps['error']}")

    print("\n✅ Simulation engine v2 ready.\n")
