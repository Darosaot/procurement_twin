#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  Procurement Digital Twin — One-Click Launcher (macOS)
#
#  Double-click this file in Finder to start the dashboard.
#  It will open automatically in your browser at localhost:8050.
#
#  Requirements: Docker Desktop must be installed.
#  Download: https://www.docker.com/products/docker-desktop/
# ─────────────────────────────────────────────────────────────────

# Always run from the folder where this script lives
cd "$(dirname "$0")"

# ── Pretty header ─────────────────────────────────────────────────
clear
echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║       PROCUREMENT DIGITAL TWIN                  ║"
echo "  ║       EU Procurement Simulator · TED 2018–2023  ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Check Docker is installed ─────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "  ❌  Docker is not installed."
    echo ""
    echo "  Please install Docker Desktop for Mac and try again:"
    echo "  https://www.docker.com/products/docker-desktop/"
    echo ""
    read -p "  Press Enter to open the download page..."
    open "https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# ── 2. Check Docker Desktop is running ───────────────────────────
echo "  Checking Docker..."
if ! docker info &> /dev/null; then
    echo "  ⏳  Docker Desktop is not running. Starting it now..."
    open -a Docker
    echo "  Waiting for Docker to start (this can take ~30 seconds)..."
    for i in {1..30}; do
        sleep 2
        if docker info &> /dev/null; then
            echo "  ✓  Docker is ready."
            break
        fi
        if [ $i -eq 30 ]; then
            echo ""
            echo "  ❌  Docker took too long to start."
            echo "  Please open Docker Desktop manually and try again."
            read -p "  Press Enter to exit..."
            exit 1
        fi
        printf "."
    done
else
    echo "  ✓  Docker is running."
fi

# ── 3. Stop any previous instance on port 8050 ───────────────────
if docker ps --format '{{.Names}}' | grep -q "procurement-twin"; then
    echo ""
    echo "  ⏹  Stopping previous instance..."
    docker compose down --timeout 5 2>/dev/null
fi

# ── 4. Build (first run only) and start ──────────────────────────
echo ""
echo "  🔨  Building image (first run takes ~3 minutes)..."
echo "  ─────────────────────────────────────────────────"
docker compose up --build -d 2>&1 | while IFS= read -r line; do
    # Show build progress, filter out noisy lines
    if echo "$line" | grep -qvE "^#[0-9]+ CACHED|^#[0-9]+ DONE"; then
        echo "  $line"
    fi
done

# ── 5. Wait for the dashboard to be ready ────────────────────────
echo ""
echo "  ⏳  Waiting for dashboard to start..."
for i in {1..30}; do
    sleep 2
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8050 | grep -q "200"; then
        echo "  ✓  Dashboard is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo ""
        echo "  ❌  Dashboard didn't start in time."
        echo "  Check logs with:  docker compose logs"
        read -p "  Press Enter to exit..."
        exit 1
    fi
    printf "."
done

# ── 6. Open browser ───────────────────────────────────────────────
echo ""
echo "  🌐  Opening http://localhost:8050 in your browser..."
sleep 1
open "http://localhost:8050"

# ── 7. Show status ────────────────────────────────────────────────
echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  ✅  Dashboard running at http://localhost:8050  ║"
echo "  ║                                                  ║"
echo "  ║  To stop:  press Ctrl+C in this window          ║"
echo "  ║         or run:  docker compose down            ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

# Keep terminal open and stream logs until Ctrl+C
trap 'echo ""; echo "  Stopping..."; docker compose down --timeout 5; echo "  ✓ Stopped."; exit 0' INT
docker compose logs -f
