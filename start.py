"""
Procurement Digital Twin — HF Spaces entrypoint
================================================
1. Download models + feature store from HF Hub (skips cached files)
2. Launch the Dash app on PORT (default 7860 for HF Spaces)

This file is the CMD in the Dockerfile for the Space deployment.
For local development, use run.py instead.
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Step 1: Download artifacts ────────────────────────────────────
print()
print("=" * 55)
print("  Procurement Digital Twin  —  starting up")
print("=" * 55)

from download_artifacts import download_all, ARTIFACTS
ok = download_all(verbose=True)

if not ok:
    # Check which critical model files are actually missing
    _missing = [
        local for _, local in ARTIFACTS
        if local.startswith("models/") and local.endswith(".pkl")
        and not os.path.exists(os.path.join(PROJECT_ROOT, local))
    ]
    if _missing:
        print("\n❌  CRITICAL: Required model files are missing:")
        for f in _missing:
            print(f"       • {f}")
        print("    Cannot start. Fix the download errors above and retry.")
        sys.exit(1)
    print("\n⚠️   Some non-critical artifacts failed to download — continuing.")

# ── Step 2: Launch the Dash app ───────────────────────────────────
PORT = int(os.environ.get("PORT", 7860))

print()
print(f"  Starting dashboard on port {PORT} ...")
print()

from src.dashboard.app import app
app.run(debug=False, host="0.0.0.0", port=PORT)
