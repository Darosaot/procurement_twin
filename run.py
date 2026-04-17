"""
Procurement Digital Twin — Launcher
=====================================
Start the interactive dashboard:

    python run.py

Then open http://localhost:8050 in your browser.

To rebuild the data pipeline and models from the raw CSV files:
    python src/pipeline/01_linkage.py
    python src/pipeline/02_ingest_and_features.py
    python src/models/03_train_models.py
"""

import sys
import os

# ── Make sure the project root is on the Python path ─────────────
# This works whether you launch from the project folder or from elsewhere.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Change working directory to project root ─────────────────────
# All relative paths inside the app (e.g. data/features/*.parquet) are
# resolved relative to this directory.
os.chdir(PROJECT_ROOT)

# ── Import and run ────────────────────────────────────────────────
from src.dashboard.app import app  # noqa: E402

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  PROCUREMENT DIGITAL TWIN")
    print("=" * 60)
    print()
    print("  Dashboard:  http://localhost:8050")
    print("  Press Ctrl+C to stop.")
    print()
    app.run(debug=False, host="0.0.0.0", port=8050)
