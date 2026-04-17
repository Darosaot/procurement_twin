"""
Procurement Digital Twin — Artifact downloader
===============================================
Called automatically at container startup (from start.py).
Downloads model files and the feature store from HF Hub into the
local project directories — skipping files that are already present.

For public repos no token is needed. If you ever make the repo private,
set the HF_TOKEN environment variable in the HF Space settings.
"""

import os
import sys

HF_REPO      = "Daniarosa/procurement-twin-artifacts"
REPO_TYPE    = "dataset"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# (path inside HF repo, local relative path)
ARTIFACTS = [
    ("models/competition_model.pkl",            "models/competition_model.pkl"),
    ("models/crossborder_model.pkl",            "models/crossborder_model.pkl"),
    ("models/duration_model.pkl",               "models/duration_model.pkl"),
    ("models/price_model.pkl",                  "models/price_model.pkl"),
    ("models/single_bid_model.pkl",             "models/single_bid_model.pkl"),
    ("models/calibration_offsets.json",         "models/calibration_offsets.json"),
    ("models/feature_spec.json",                "models/feature_spec.json"),
    ("models/model_evaluation.json",            "models/model_evaluation.json"),
    ("models/shap_importances.json",            "models/shap_importances.json"),
    ("data/features/procedure_records.parquet", "data/features/procedure_records.parquet"),
]


def download_all(verbose: bool = True) -> bool:
    """
    Download all required artifacts from HF Hub.

    Returns True if all files are present after the run, False if any failed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌  huggingface_hub not installed — cannot download artifacts.")
        return False

    token   = os.environ.get("HF_TOKEN")  # optional; not needed for public repos
    all_ok  = True

    for hf_path, local_rel in ARTIFACTS:
        local_abs = os.path.join(PROJECT_ROOT, local_rel)

        if os.path.exists(local_abs):
            if verbose:
                size_mb = os.path.getsize(local_abs) / 1_048_576
                print(f"✓  Already cached: {local_rel}  ({size_mb:.1f} MB)")
            continue

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(local_abs), exist_ok=True)

        print(f"⬇️   Downloading: {hf_path} ...", end="", flush=True)
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=hf_path,
                repo_type=REPO_TYPE,
                token=token,
                local_dir=PROJECT_ROOT,
            )
            size_mb = os.path.getsize(local_abs) / 1_048_576
            print(f"  ✅  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"\n   ❌  Failed: {e}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("=" * 55)
    print("  Procurement Digital Twin — downloading artifacts")
    print("=" * 55)
    ok = download_all(verbose=True)
    if not ok:
        print("\n⚠️   Some artifacts could not be downloaded.")
        print("    The app may not start correctly.")
        sys.exit(1)
    print("\n✅  All artifacts ready.")
