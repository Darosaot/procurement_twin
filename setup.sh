#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Procurement Digital Twin — Local Setup (Mac / Linux)
# ─────────────────────────────────────────────────────────────────
#  Run once:  bash setup.sh
#  Then:      python run.py
# ─────────────────────────────────────────────────────────────────

set -e

echo ""
echo "========================================================"
echo "  Procurement Digital Twin — Local Setup"
echo "========================================================"
echo ""

# ── 1. Check Python version ───────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "❌  Python not found. Please install Python 3.10 or later."
    echo "    https://www.python.org/downloads/"
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓  Python $PY_VERSION found at $PYTHON"

# ── 2. Create a virtual environment ──────────────────────────────
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment (.venv)..."
    $PYTHON -m venv .venv
    echo "✓  Virtual environment created."
else
    echo "✓  Virtual environment already exists."
fi

# ── 3. Activate and install dependencies ─────────────────────────
echo ""
echo "Installing dependencies (this takes ~2 minutes on first run)..."
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✓  All dependencies installed."

# ── 4. Done ──────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Setup complete!"
echo "========================================================"
echo ""
echo "  To launch the dashboard:"
echo "    source .venv/bin/activate"
echo "    python run.py"
echo ""
echo "  Then open http://localhost:8050 in your browser."
echo ""
