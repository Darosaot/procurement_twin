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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Step 1: Download artifacts ────────────────────────────────────
print()
print("=" * 55)
print("  Procurement Digital Twin  —  starting up")
print("=" * 55)

from download_artifacts import download_all
ok = download_all(verbose=True)

if not ok:
    # Non-fatal: the app will still start but some tabs may show errors
    print("\n⚠️   Continuing despite download errors — some features may be unavailable.")

# ── Step 2: Launch the Dash app ───────────────────────────────────
PORT = int(os.environ.get("PORT", 7860))

print()
print(f"  Starting dashboard on port {PORT} ...")
print()

from src.dashboard.app import app
app.run(debug=False, host="0.0.0.0", port=PORT)
