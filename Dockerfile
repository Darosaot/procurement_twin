# ── Procurement Digital Twin — Docker Image ──────────────────────
#
# LOCAL development:
#   docker build -t procurement-twin .
#   docker run -p 8050:8050 -e PORT=8050 procurement-twin
#   docker compose up         (recommended for local)
#
# HF Spaces deployment:
#   Push this repo to your HF Space — the Space builds and runs
#   this Dockerfile automatically. Port 7860 is the HF default.
#   Models and data are downloaded from HF Hub at container startup.

FROM python:3.11-slim

# ── Non-root user (required by HF Spaces) ─────────────────────────
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────
COPY --chown=user . .

# ── Switch to non-root user ────────────────────────────────────────
USER user

# ── Expose HF Spaces default port ─────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────
# Generous start-period because the container downloads ~50 MB of
# artifacts from HF Hub on the first cold start.
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-7860}')" || exit 1

# ── Entrypoint ────────────────────────────────────────────────────
# start.py:  1) downloads artifacts from HF Hub  2) launches the app
CMD ["python", "start.py"]
