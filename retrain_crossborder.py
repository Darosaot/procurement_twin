"""
Retrain cross-border win model on corrected data + re-upload artifacts.

Run once from your local machine:
    python retrain_crossborder.py

What this does:
  1. Loads the corrected procedure_records.parquet (cross_border_win fixed)
  2. Retrains the Random Forest cross-border model
  3. Saves the new model to models/crossborder_model.pkl
  4. Updates models/model_evaluation.json with new metrics
  5. Re-uploads both files to HF Hub
"""

import os, sys, json, pickle, ssl
import numpy as np
import pandas as pd

# ── SSL fix for Homebrew Python ────────────────────────────────────
_orig_ctx = ssl.create_default_context
def _patched_ctx(*args, **kwargs):
    ctx = _orig_ctx(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    return ctx
ssl.create_default_context        = _patched_ctx
ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

FEAT_DIR  = os.path.join(PROJECT_ROOT, "data", "features")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ── 1. Load data ────────────────────────────────────────────────────
print("Loading corrected procedure_records.parquet ...")
import polars as pl
df = pl.read_parquet(os.path.join(FEAT_DIR, "procedure_records.parquet")).to_pandas()

print(f"  Total records: {len(df):,}")
cb = df['cross_border_win'].dropna()
print(f"  CB rate (corrected): {cb.mean()*100:.2f}%  (n={len(cb):,})")

# ── 2. Load feature spec ─────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "feature_spec.json")) as f:
    feat_spec = json.load(f)

CB_FEATURES = feat_spec.get("crossborder_features", [
    "ISO_COUNTRY_CODE", "TOP_TYPE", "TYPE_OF_CONTRACT", "cpv_division",
    "CRIT_CODE", "value_bracket", "country_cluster",
    "log10_value", "prep_time_days", "contract_duration_months",
    "flag_b_gpa", "flag_b_eu_funds", "flag_b_fra_agreement",
    "flag_b_electronic_auction", "flag_b_accelerated", "price_weight_pct",
])

# ── 3. Prepare features ──────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

available = [c for c in CB_FEATURES if c in df.columns]
print(f"\nUsing {len(available)}/{len(CB_FEATURES)} features: {available}")

target = 'cross_border_win'
mask   = df[target].notna()
X = df.loc[mask, available].copy()
y = df.loc[mask, target].astype(int)

# Fill missing values
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
for c in num_cols:
    X[c] = X[c].fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].fillna("Unknown")

# Temporal split: train on 2018-2021, test on 2022-2023
years = df.loc[mask, 'YEAR'].values
train_mask = years <= 2021
test_mask  = years >= 2022

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\n  Train: {len(X_train):,} records  (CB rate {y_train.mean()*100:.1f}%)")
print(f"  Test:  {len(X_test):,} records   (CB rate {y_test.mean()*100:.1f}%)")

# ── 4. Build & train pipeline ────────────────────────────────────────
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model_pipeline = {"pre": pre, "mdl": rf}

print("\nFitting preprocessing + Random Forest ...")
X_train_enc = pre.fit_transform(X_train)
rf.fit(X_train_enc, y_train)

# ── 5. Evaluate ──────────────────────────────────────────────────────
X_test_enc = pre.transform(X_test)
proba_test = rf.predict_proba(X_test_enc)[:, 1]
auc_rf = roc_auc_score(y_test, proba_test)
ap_rf  = average_precision_score(y_test, proba_test)

print(f"\n  Test AUC:  {auc_rf:.3f}  (old: 0.759)")
print(f"  Test AP:   {ap_rf:.3f}  (old: 0.281)")
print(f"  Base rate: {y_test.mean():.3f}")

# ── 6. Save model ────────────────────────────────────────────────────
model_bundle = {"model": model_pipeline, "meta": {"base_rate": float(y.mean())}}
model_path = os.path.join(MODEL_DIR, "crossborder_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model_bundle, f)
print(f"\n  Saved: {model_path}")

# Update model_evaluation.json
eval_path = os.path.join(MODEL_DIR, "model_evaluation.json")
with open(eval_path) as f:
    eval_data = json.load(f)

eval_data["cross_border"] = {
    "rf_auc":    round(auc_rf, 3),
    "rf_ap":     round(ap_rf, 3),
    "n_train":   int(y_train.sum()),
    "n_test":    int(len(y_test)),
    "base_rate": round(float(y_test.mean()), 3),
    "note":      "Retrained after fixing cross_border_win label bug (multi-lot ---separator)"
}
with open(eval_path, "w") as f:
    json.dump(eval_data, f, indent=2)
print(f"  Updated: {eval_path}")

# ── 7. Upload to HF Hub ──────────────────────────────────────────────
print("\nUploading to HF Hub ...")
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
api   = HfApi(token=TOKEN)

try:
    user = api.whoami()
    print(f"  Logged in as: {user['name']}")
except Exception as e:
    print(f"  ❌ Auth failed: {e}")
    print("  Set HF_TOKEN env var or run: hf auth login")
    sys.exit(1)

HF_REPO   = "Daniarosa/procurement-twin-artifacts"
REPO_TYPE = "dataset"

uploads = [
    (os.path.join(MODEL_DIR, "crossborder_model.pkl"),            "models/crossborder_model.pkl"),
    (os.path.join(MODEL_DIR, "model_evaluation.json"),            "models/model_evaluation.json"),
    (os.path.join(FEAT_DIR,  "procedure_records.parquet"),        "data/features/procedure_records.parquet"),
]

for local, remote in uploads:
    size_mb = os.path.getsize(local) / 1_048_576
    print(f"  ⬆️  {remote}  ({size_mb:.1f} MB) ...", end="", flush=True)
    try:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=HF_REPO, repo_type=REPO_TYPE, token=TOKEN)
        print("  ✅")
    except Exception as e:
        print(f"\n     ❌ {e}")

print("\n🎉  Done. Restart your HF Space to pick up the new model and data.")
