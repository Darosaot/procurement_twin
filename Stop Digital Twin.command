#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Procurement Digital Twin — Stop
#  Double-click to shut down the dashboard container.
# ─────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

clear
echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║       PROCUREMENT DIGITAL TWIN — STOP           ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

if ! docker info &> /dev/null; then
    echo "  Docker is not running — nothing to stop."
else
    echo "  Stopping dashboard..."
    docker compose down --timeout 5
    echo ""
    echo "  ✓  Dashboard stopped."
fi

echo ""
read -p "  Press Enter to close this window..."
