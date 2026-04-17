"""
Phase 2: Predictive Model Training  (v2 — IV price model + calibration)
=========================================================================
Trains five outcome models on the Procedure Record feature store.
Temporal train/test split: 2018-2021 (train) vs 2022-2023 (test).

Models trained:
  1. Competition model      → n_offers (XGBoost regressor)
  2. Single-bid risk model  → single_bid_flag (Random Forest)
  3. Cross-border model     → cross_border_win (Random Forest)
  4. Price ratio model (IV) → price_ratio using competition_hat as instrument
  5. Duration model         → proc_duration_days (Ridge / log scale)

Additional artefacts:
  calibration_offsets.json  ← per-CPV and per-cluster median residuals
  shap_importances.json     ← global mean |SHAP| for each model
  feature_spec.json         ← feature lists for simulation engine
  model_evaluation.json     ← test-set metrics
"""

import polars as pl
import pandas as pd
import numpy as np
import json, os, pickle, time, warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, r2_score
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  XGBoost not available, using GradientBoosting instead")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  SHAP not available, skipping feature importance export")

FEAT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "features")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("PHASE 2: MODEL TRAINING  (v2)")
print("="*60)

# ══════════════════════════════════════════════════════════════════
# Load feature store
# ══════════════════════════════════════════════════════════════════
print("\n[0] Loading feature store...")
df = pl.read_parquet(f"{FEAT_DIR}/procedure_records.parquet").to_pandas()
print(f"  {len(df):,} rows  ×  {df.shape[1]} cols")

# ── Feature specification ─────────────────────────────────────────
CAT_FEATURES = ["ISO_COUNTRY_CODE","TOP_TYPE","TYPE_OF_CONTRACT",
                "cpv_division","CRIT_CODE","value_bracket","country_cluster"]
NUM_FEATURES = ["log10_value","prep_time_days","contract_duration_months",
                "flag_b_gpa","flag_b_eu_funds","flag_b_fra_agreement",
                "flag_b_electronic_auction","flag_b_accelerated",
                "price_weight_pct"]

# Extended feature list for price model (adds competition_hat)
NUM_FEATURES_PRICE = NUM_FEATURES + ["competition_hat"]

ALL_FEATURES       = CAT_FEATURES + NUM_FEATURES
ALL_FEATURES_PRICE = CAT_FEATURES + NUM_FEATURES_PRICE

# ── Train / test split ────────────────────────────────────────────
train_mask = df["YEAR"].isin([2018, 2019, 2020, 2021])
test_mask  = df["YEAR"].isin([2022, 2023])

def prep_X(subset, extra_num=None):
    """Return a clean DataFrame of model features."""
    num_cols = NUM_FEATURES if extra_num is None else NUM_FEATURES_PRICE
    X = subset[CAT_FEATURES + num_cols].copy()
    for c in (NUM_FEATURES_PRICE if extra_num else NUM_FEATURES):
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str).fillna("Unknown").replace("nan","Unknown").replace("None","Unknown")
    return X

# Preprocessing pipelines
def make_preprocessor(num_cols=None):
    if num_cols is None:
        num_cols = NUM_FEATURES
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ], remainder="drop")

evaluation   = {}
shap_importance = {}

def save_model(model, name):
    path = f"{MODEL_DIR}/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {name}.pkl")


# ══════════════════════════════════════════════════════════════════
# MODEL 1: Competition (n_offers)
# ══════════════════════════════════════════════════════════════════
print("\n[1/5] Competition model (n_offers)...")
t0 = time.time()

outcome = "n_offers"
mask = df[outcome].notna() & df[outcome].between(0, 100)
train = df[train_mask & mask]
test  = df[test_mask  & mask]
print(f"  Train: {len(train):,}  Test: {len(test):,}")

X_train = prep_X(train); y_train = train[outcome].astype(float)
X_test  = prep_X(test);  y_test  = test[outcome].astype(float)

# Baseline Ridge
baseline_comp = Pipeline([
    ("pre", make_preprocessor()),
    ("mdl", Ridge(alpha=1.0))
])
baseline_comp.fit(X_train, np.log1p(y_train))
y_pred_base = np.expm1(baseline_comp.predict(X_test)).clip(0)
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base  = r2_score(y_test, y_pred_base)

# Boosted model
if HAS_XGB:
    boost_comp = Pipeline([
        ("pre", make_preprocessor()),
        ("mdl", xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                  objective="count:poisson", random_state=42))
    ])
else:
    boost_comp = Pipeline([
        ("pre", make_preprocessor()),
        ("mdl", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                           subsample=0.8, random_state=42))
    ])
boost_comp.fit(X_train, y_train)
y_pred_boost = boost_comp.predict(X_test).clip(0)
mae_boost = mean_absolute_error(y_test, y_pred_boost)
r2_boost  = r2_score(y_test, y_pred_boost)

print(f"  Baseline   MAE={mae_base:.2f}  R²={r2_base:.3f}")
print(f"  Boost      MAE={mae_boost:.2f}  R²={r2_boost:.3f}")

deploy_comp = boost_comp if r2_boost > r2_base else baseline_comp

# Distribution metadata for Monte Carlo noise
log_offers = np.log1p(y_train.values)
competition_meta = {
    "log_mean": float(log_offers.mean()),
    "log_std":  float(log_offers.std()),
    "p10": float(np.percentile(y_train, 10)),
    "p25": float(np.percentile(y_train, 25)),
    "p50": float(np.percentile(y_train, 50)),
    "p75": float(np.percentile(y_train, 75)),
    "p90": float(np.percentile(y_train, 90)),
}
save_model({"model": deploy_comp, "meta": competition_meta}, "competition_model")
evaluation["competition"] = {
    "baseline_mae": round(mae_base, 3), "baseline_r2": round(r2_base, 3),
    "boost_mae":    round(mae_boost, 3),"boost_r2":    round(r2_boost, 3),
    "n_train": len(train), "n_test": len(test)
}

# Feature importances for competition model (XGBoost gain-based)
if HAS_XGB and r2_boost > r2_base:
    try:
        pre_fitted = deploy_comp["pre"]
        feat_names = pre_fitted.get_feature_names_out()
        fi = deploy_comp["mdl"].feature_importances_
        shap_importance["competition"] = dict(zip(
            [n.replace("num__","").replace("cat__","") for n in feat_names],
            [round(float(v), 6) for v in fi]
        ))
        print(f"  Feature importances computed ({len(fi)} features)")
    except Exception as e:
        print(f"  Feature importance skipped: {e}")
print(f"  Time: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
# MODEL 2: Single-bid risk
# ══════════════════════════════════════════════════════════════════
print("\n[2/5] Single-bid risk model...")
t0 = time.time()

outcome = "single_bid_flag"
mask = df[outcome].notna()
train = df[train_mask & mask]
test  = df[test_mask  & mask]
print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Positive rate: {train[outcome].mean()*100:.1f}%")

X_train = prep_X(train); y_train = train[outcome].astype(int)
X_test  = prep_X(test);  y_test  = test[outcome].astype(int)

lr_sb = Pipeline([
    ("pre", make_preprocessor()),
    ("mdl", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced"))
])
lr_sb.fit(X_train, y_train)
y_prob_lr = lr_sb.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_prob_lr)

rf_sb = Pipeline([
    ("pre", make_preprocessor()),
    ("mdl", RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1,
                                    class_weight="balanced", random_state=42))
])
rf_sb.fit(X_train, y_train)
y_prob_rf = rf_sb.predict_proba(X_test)[:,1]
auc_rf = roc_auc_score(y_test, y_prob_rf)
ap_rf  = average_precision_score(y_test, y_prob_rf)

print(f"  LogReg       AUC={auc_lr:.3f}")
print(f"  RandomForest AUC={auc_rf:.3f}  AP={ap_rf:.3f}")

deploy_sb = rf_sb if auc_rf > auc_lr else lr_sb
save_model({"model": deploy_sb}, "single_bid_model")
evaluation["single_bid"] = {
    "lr_auc": round(auc_lr,3),
    "rf_auc": round(auc_rf,3), "rf_ap": round(ap_rf,3),
    "n_train": len(train), "n_test": len(test),
    "base_rate": round(float(train[outcome].mean()), 3)
}

# Feature importances for single-bid model
try:
    mdl_obj = deploy_sb["mdl"]
    if hasattr(mdl_obj, "feature_importances_"):
        feat_names = deploy_sb["pre"].get_feature_names_out()
        fi = mdl_obj.feature_importances_
    elif hasattr(mdl_obj, "coef_"):
        feat_names = deploy_sb["pre"].get_feature_names_out()
        fi = np.abs(mdl_obj.coef_[0])
    else:
        feat_names, fi = [], []
    if len(fi) > 0:
        shap_importance["single_bid"] = dict(zip(
            [n.replace("num__","").replace("cat__","") for n in feat_names],
            [round(float(v), 6) for v in fi]
        ))
        print(f"  Feature importances computed ({len(fi)} features)")
except Exception as e:
    print(f"  Feature importance skipped: {e}")
print(f"  Time: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
# MODEL 3: Cross-border win
# ══════════════════════════════════════════════════════════════════
print("\n[3/5] Cross-border win model...")
t0 = time.time()

outcome = "cross_border_win"
mask = df[outcome].notna()
train = df[train_mask & mask]
test  = df[test_mask  & mask]
print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Positive rate: {train[outcome].mean()*100:.1f}%")

X_train = prep_X(train); y_train = train[outcome].astype(int)
X_test  = prep_X(test);  y_test  = test[outcome].astype(int)

lr_cb = Pipeline([
    ("pre", make_preprocessor()),
    ("mdl", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced"))
])
lr_cb.fit(X_train, y_train)
auc_lr_cb = roc_auc_score(y_test, lr_cb.predict_proba(X_test)[:,1])

rf_cb = Pipeline([
    ("pre", make_preprocessor()),
    ("mdl", RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1,
                                    class_weight="balanced", random_state=42))
])
rf_cb.fit(X_train, y_train)
y_prob_rf_cb = rf_cb.predict_proba(X_test)[:,1]
auc_rf_cb = roc_auc_score(y_test, y_prob_rf_cb)
ap_rf_cb  = average_precision_score(y_test, y_prob_rf_cb)

print(f"  LogReg       AUC={auc_lr_cb:.3f}")
print(f"  RandomForest AUC={auc_rf_cb:.3f}  AP={ap_rf_cb:.3f}")

deploy_cb = rf_cb if auc_rf_cb > auc_lr_cb else lr_cb
save_model({"model": deploy_cb}, "crossborder_model")
evaluation["cross_border"] = {
    "lr_auc": round(auc_lr_cb,3),
    "rf_auc": round(auc_rf_cb,3), "rf_ap": round(ap_rf_cb,3),
    "n_train": len(train), "n_test": len(test),
    "base_rate": round(float(train[outcome].mean()), 3)
}
print(f"  Time: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
# MODEL 4: Price ratio — 2-stage model (competition as instrument)
#
# Stage 1: predict competition from design variables (already done)
# Stage 2: predict log(price_ratio) from design vars + competition_hat
#
# By using competition_hat (the exogenous component of competition)
# rather than observed competition, we isolate the causal pathway
# from procedure design → competition → price, avoiding the
# endogeneity bias from unobserved market conditions.
# ══════════════════════════════════════════════════════════════════
print("\n[4/5] Price ratio model (2-stage IV)...")
t0 = time.time()

outcome = "price_ratio"
mask = df[outcome].notna() & df[outcome].between(0.1, 3.0)
train = df[train_mask & mask].copy()
test  = df[test_mask  & mask].copy()
print(f"  Train: {len(train):,}  Test: {len(test):,}")

# Stage 1: get competition_hat predictions from the competition model
train["competition_hat"] = deploy_comp.predict(prep_X(train)).clip(0)
test["competition_hat"]  = deploy_comp.predict(prep_X(test)).clip(0)

# Build X with competition_hat added
X_train_p = prep_X(train, extra_num=True); y_train_p = np.log(train[outcome].astype(float))
X_test_p  = prep_X(test,  extra_num=True); y_test_p  = test[outcome].astype(float)

# Stage 2 model: Ridge on log(price_ratio) with competition_hat as regressor
ridge_iv = Pipeline([
    ("pre", make_preprocessor(num_cols=NUM_FEATURES_PRICE)),
    ("mdl", Ridge(alpha=1.0))
])
ridge_iv.fit(X_train_p, y_train_p)
y_pred_iv = np.exp(ridge_iv.predict(X_test_p))
mae_iv  = mean_absolute_error(y_test_p, y_pred_iv)
r2_iv   = r2_score(y_test_p, y_pred_iv)

# Baseline (without competition_hat) for comparison
ridge_base = Pipeline([
    ("pre", make_preprocessor()),
    ("mdl", Ridge(alpha=1.0))
])
ridge_base.fit(prep_X(train), np.log(train[outcome].astype(float)))
y_pred_base = np.exp(ridge_base.predict(prep_X(test)))
mae_base_pr = mean_absolute_error(y_test_p, y_pred_base)
r2_base_pr  = r2_score(y_test_p, y_pred_base)

print(f"  Baseline (no comp) MAE={mae_base_pr:.3f}  R²={r2_base_pr:.3f}")
print(f"  IV (with comp_hat) MAE={mae_iv:.3f}  R²={r2_iv:.3f}")

# Distribution parameters (fit on training data)
log_ratio = np.log(train[outcome].astype(float))
price_meta = {
    "log_mean": float(log_ratio.mean()),
    "log_std":  float(log_ratio.std()),
    "p10": float(np.exp(np.percentile(log_ratio, 10))),
    "p25": float(np.exp(np.percentile(log_ratio, 25))),
    "p50": float(np.exp(np.percentile(log_ratio, 50))),
    "p75": float(np.exp(np.percentile(log_ratio, 75))),
    "p90": float(np.exp(np.percentile(log_ratio, 90))),
    "uses_competition_hat": True,
}

save_model({"model": ridge_iv, "meta": price_meta}, "price_model")
evaluation["price_ratio"] = {
    "baseline_mae": round(mae_base_pr, 3), "baseline_r2": round(r2_base_pr, 3),
    "iv_mae":  round(mae_iv, 3),  "iv_r2":  round(r2_iv, 3),
    "n_train": len(train), "n_test": len(test)
}
print(f"  Time: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
# MODEL 5: Procedure duration
# ══════════════════════════════════════════════════════════════════
print("\n[5/5] Procedure duration model...")
t0 = time.time()

outcome = "proc_duration_days"
mask = df[outcome].notna() & df[outcome].between(0, 730)
train = df[train_mask & mask]
test  = df[test_mask  & mask]
print(f"  Train: {len(train):,}  Test: {len(test):,}")

X_train = prep_X(train); y_train = np.log1p(train[outcome].astype(float))
X_test  = prep_X(test);  y_test  = test[outcome].astype(float)

ridge_dur = Pipeline([
    ("pre", make_preprocessor()),
    ("mdl", Ridge(alpha=1.0))
])
ridge_dur.fit(X_train, y_train)
y_pred = np.expm1(ridge_dur.predict(X_test))
mae_dur = mean_absolute_error(y_test, y_pred)
r2_dur  = r2_score(y_test, y_pred)

log_dur = np.log1p(train[outcome].astype(float))
dur_meta = {
    "log_mean": float(log_dur.mean()),
    "log_std":  float(log_dur.std()),
    "p10": float(np.expm1(np.percentile(log_dur, 10))),
    "p25": float(np.expm1(np.percentile(log_dur, 25))),
    "p50": float(np.expm1(np.percentile(log_dur, 50))),
    "p75": float(np.expm1(np.percentile(log_dur, 75))),
    "p90": float(np.expm1(np.percentile(log_dur, 90))),
}
print(f"  Ridge      MAE={mae_dur:.1f}d  R²={r2_dur:.3f}")

save_model({"model": ridge_dur, "meta": dur_meta}, "duration_model")
evaluation["duration"] = {
    "mae_days": round(mae_dur, 1), "r2": round(r2_dur, 3),
    "n_train": len(train), "n_test": len(test)
}
print(f"  Time: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
# CALIBRATION OFFSETS — per CPV and per country cluster
#
# These capture systematic biases of the global model in specific
# segments (e.g., medical supplies consistently price higher than
# the global model predicts). Applied post-hoc in simulation.
# ══════════════════════════════════════════════════════════════════
print("\n[+] Computing calibration offsets...")

cal_train_mask = df["price_ratio"].notna() & df["price_ratio"].between(0.1, 3.0) & train_mask
cal_df = df[cal_train_mask].copy()
cal_df["competition_hat"] = deploy_comp.predict(prep_X(cal_df)).clip(0)

# Price ratio calibration (log scale)
X_cal = prep_X(cal_df, extra_num=True)
log_hat = ridge_iv.predict(X_cal)
log_actual = np.log(cal_df["price_ratio"].astype(float))
residuals = log_actual.values - log_hat

cal_df["_resid"] = residuals

cpv_offsets = (cal_df.groupby("cpv_division")["_resid"]
               .median().round(4).to_dict())
cluster_offsets = (cal_df.groupby("country_cluster")["_resid"]
                   .median().round(4).to_dict())

# Competition calibration (log scale, per CPV)
comp_train_mask = df["n_offers"].notna() & df["n_offers"].between(0, 100) & train_mask
comp_cal_df = df[comp_train_mask].copy()
X_comp_cal = prep_X(comp_cal_df)
comp_hat = deploy_comp.predict(X_comp_cal)
comp_actual = comp_cal_df["n_offers"].astype(float).values
comp_log_resid = np.log1p(comp_actual) - np.log1p(comp_hat.clip(0))
comp_cal_df["_comp_resid"] = comp_log_resid

comp_cpv_offsets = (comp_cal_df.groupby("cpv_division")["_comp_resid"]
                    .median().round(4).to_dict())
comp_cluster_offsets = (comp_cal_df.groupby("country_cluster")["_comp_resid"]
                        .median().round(4).to_dict())

calibration = {
    "price_ratio": {
        "by_cpv":     cpv_offsets,
        "by_cluster": cluster_offsets,
    },
    "competition": {
        "by_cpv":     comp_cpv_offsets,
        "by_cluster": comp_cluster_offsets,
    },
}

with open(f"{MODEL_DIR}/calibration_offsets.json", "w") as f:
    json.dump(calibration, f, indent=2)
print(f"  Saved calibration offsets: {len(cpv_offsets)} CPV + {len(cluster_offsets)} cluster buckets")


# ══════════════════════════════════════════════════════════════════
# Save artefacts
# ══════════════════════════════════════════════════════════════════
with open(f"{MODEL_DIR}/model_evaluation.json", "w") as f:
    json.dump(evaluation, f, indent=2)

if shap_importance:
    with open(f"{MODEL_DIR}/shap_importances.json", "w") as f:
        json.dump(shap_importance, f, indent=2)
    print("  Saved SHAP importances")

feature_spec = {
    "cat_features":       CAT_FEATURES,
    "num_features":       NUM_FEATURES,
    "num_features_price": NUM_FEATURES_PRICE,
    "all_features":       ALL_FEATURES,
    "all_features_price": ALL_FEATURES_PRICE,
    "price_model_uses_competition_hat": True,
}
with open(f"{MODEL_DIR}/feature_spec.json", "w") as f:
    json.dump(feature_spec, f, indent=2)

print("\n" + "="*60)
print("MODEL EVALUATION SUMMARY")
print("="*60)
print(f"\n  Competition model:    MAE={evaluation['competition']['boost_mae']:.2f} offers  R²={evaluation['competition']['boost_r2']:.3f}")
print(f"  Single-bid model:     AUC={evaluation['single_bid']['rf_auc']:.3f}")
print(f"  Cross-border model:   AUC={evaluation['cross_border']['rf_auc']:.3f}")
print(f"  Price ratio (IV):     MAE={evaluation['price_ratio']['iv_mae']:.3f}  R²={evaluation['price_ratio']['iv_r2']:.3f}  (baseline R²={evaluation['price_ratio']['baseline_r2']:.3f})")
print(f"  Duration model:       MAE={evaluation['duration']['mae_days']:.0f}d  R²={evaluation['duration']['r2']:.3f}")

print("\n✅ Phase 2 complete — all models saved.\n")
