"""
Procurement Digital Twin — One-time artifact upload to Hugging Face Hub
========================================================================
Run this ONCE from your local machine to push model files and the feature
store to HF Hub. The Space will download them automatically at startup.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # paste your HF write token when prompted

Then:
    python upload_to_hf.py
"""

import os
import sys

# ── Optional SSL workaround — disabled by default ─────────────────
# Activate only when certs cannot be fixed: PROCUREMENT_SKIP_SSL_VERIFY=1
# Preferred fix: pip install certifi && python -m certifi
import ssl
if os.environ.get("PROCUREMENT_SKIP_SSL_VERIFY") == "1":
    import warnings as _w
    _w.warn(
        "SSL certificate verification DISABLED (PROCUREMENT_SKIP_SSL_VERIFY=1). "
        "Only use this on a trusted network.",
        stacklevel=1,
    )
    _orig_ctx = ssl.create_default_context
    def _patched_ctx(*args, **kwargs):
        ctx = _orig_ctx(*args, **kwargs)
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        return ctx
    ssl.create_default_context        = _patched_ctx
    ssl._create_default_https_context = ssl._create_unverified_context

HF_REPO   = os.environ.get("HF_REPO", "Daniarosa/procurement-twin-artifacts")
REPO_TYPE = "dataset"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Files to upload: (local relative path, path inside HF repo)
FILES_TO_UPLOAD = [
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


def main():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌  huggingface_hub is not installed.")
        print("    Run:  pip install huggingface_hub")
        sys.exit(1)

    # Read token from env var or cached HF login (~/.cache/huggingface/token)
    TOKEN = os.environ.get("HF_TOKEN")
    api = HfApi(token=TOKEN)

    # Verify token works
    try:
        user = api.whoami()
        print(f"✅  Logged in as: {user['name']}")
    except Exception as e:
        print(f"❌  Authentication failed: {e}")
        print("    Set HF_TOKEN env var or run:  hf auth login")
        sys.exit(1)

    # Create the dataset repo if it doesn't exist
    print(f"\n📦  Ensuring repo exists: {HF_REPO}")
    api.create_repo(
        repo_id=HF_REPO,
        repo_type=REPO_TYPE,
        exist_ok=True,
        private=False,
    )
    print(f"    https://huggingface.co/datasets/{HF_REPO}\n")

    # Upload each file
    total = len(FILES_TO_UPLOAD)
    for i, (local_rel, hf_path) in enumerate(FILES_TO_UPLOAD, 1):
        local_abs = os.path.join(PROJECT_ROOT, local_rel)

        if not os.path.exists(local_abs):
            print(f"[{i}/{total}] ⚠️   Skipping (not found): {local_rel}")
            continue

        size_mb = os.path.getsize(local_abs) / 1_048_576
        print(f"[{i}/{total}] ⬆️   {local_rel}  ({size_mb:.1f} MB) ...", end="", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=local_abs,
                path_in_repo=hf_path,
                repo_id=HF_REPO,
                repo_type=REPO_TYPE,
                token=TOKEN,
            )
            print("  ✅")
        except Exception as e:
            print(f"\n      ❌ Failed: {e}")

    print("\n🎉  Upload complete.")
    print(f"    View at: https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
