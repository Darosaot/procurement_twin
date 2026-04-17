#!/usr/bin/env python3
"""
Procurement Digital Twin — FastAPI REST server launcher
────────────────────────────────────────────────────────
Usage:
    python run_api.py                  # default: 0.0.0.0:8000, 1 worker
    python run_api.py --port 9000
    python run_api.py --host 127.0.0.1 --workers 4
    python run_api.py --reload         # hot-reload for development

Swagger UI  → http://localhost:8000/docs
ReDoc       → http://localhost:8000/redoc
OpenAPI JSON→ http://localhost:8000/openapi.json
"""

import argparse
import sys
from pathlib import Path

# ── ensure project root is on sys.path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the Procurement Digital Twin REST API"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1). "
             "Set >1 only when not using --reload.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload for development (forces workers=1)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info)",
    )
    args = parser.parse_args()

    # --reload is incompatible with multiple workers
    workers = 1 if args.reload else args.workers

    import uvicorn

    print(
        f"\n🚀  Procurement Digital Twin API\n"
        f"    http://{args.host}:{args.port}\n"
        f"    Swagger UI  → http://localhost:{args.port}/docs\n"
        f"    ReDoc       → http://localhost:{args.port}/redoc\n"
    )

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        workers=workers,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
