"""
Full pipeline orchestrator for the Procurement Digital Twin.

Steps:
  0. download  — fetch TED CSV bulk files from EU Open Data Portal
  1. linkage   — match CFC call notices to CAN award notices
  2. features  — feature engineering → procedure_records.parquet
  3. train     — train all 5 outcome models + emit artifacts
  4. upload    — push model artifacts to HuggingFace Hub

Status is written to pipeline_status.json so the dashboard admin panel
can poll it for live progress display.

Usage:
  python pipeline/run_pipeline.py                     # full pipeline
  python pipeline/run_pipeline.py --skip-download     # skip step 0
  python pipeline/run_pipeline.py --skip-upload       # skip step 4
  python pipeline/run_pipeline.py --steps 2 3         # run only steps 2+3
  python pipeline/run_pipeline.py --years 2022 2023   # download specific years
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).parent.parent
STATUS_FILE = ROOT / "pipeline_status.json"
MAX_LOG_LINES = 500  # keep last N lines in status file


STEPS = [
    {
        "id":    "download",
        "label": "Download TED data",
        "cmd":   [sys.executable, str(ROOT / "pipeline" / "00_download_ted.py")],
    },
    {
        "id":    "linkage",
        "label": "CFC–CAN linkage",
        "cmd":   [sys.executable, str(ROOT / "src" / "pipeline" / "01_linkage.py")],
    },
    {
        "id":    "features",
        "label": "Feature engineering",
        "cmd":   [sys.executable, str(ROOT / "src" / "pipeline" / "02_ingest_and_features.py")],
    },
    {
        "id":    "train",
        "label": "Train models",
        "cmd":   [sys.executable, str(ROOT / "src" / "models" / "03_train_models.py")],
    },
    {
        "id":    "upload",
        "label": "Upload artifacts to HF Hub",
        "cmd":   [sys.executable, str(ROOT / "upload_to_hf.py")],
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(status: dict) -> None:
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(STATUS_FILE)


def read_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {"status": "idle"}


def _append_log(status: dict, line: str) -> None:
    logs = status.setdefault("logs", [])
    logs.append(line)
    if len(logs) > MAX_LOG_LINES:
        status["logs"] = logs[-MAX_LOG_LINES:]


def run_step(step: dict, status: dict, extra_args: list[str] | None = None) -> bool:
    """Run one pipeline step as a subprocess, streaming output into status logs."""
    cmd = step["cmd"] + (extra_args or [])
    _append_log(status, f"\n▶ {step['label']}")
    _append_log(status, f"  $ {' '.join(cmd)}")
    write_status(status)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            line = line.rstrip()
            _append_log(status, line)
            # flush status every 20 lines so dashboard sees live output
            if len(status["logs"]) % 20 == 0:
                write_status(status)

        proc.wait()
        if proc.returncode != 0:
            _append_log(status, f"✗ Step '{step['label']}' exited with code {proc.returncode}")
            return False

        _append_log(status, f"✓ {step['label']} complete")
        return True

    except Exception as exc:
        _append_log(status, f"✗ Exception in '{step['label']}': {exc}")
        return False


def run_pipeline(
    step_ids: list[str] | None = None,
    skip_download: bool = False,
    skip_upload: bool = False,
    download_years: list[int] | None = None,
) -> bool:
    steps = [s for s in STEPS if (step_ids is None or s["id"] in step_ids)]
    if skip_download:
        steps = [s for s in steps if s["id"] != "download"]
    if skip_upload:
        steps = [s for s in steps if s["id"] != "upload"]

    status = {
        "status":      "running",
        "step":        None,
        "step_label":  None,
        "step_idx":    0,
        "total_steps": len(steps),
        "started_at":  _now(),
        "finished_at": None,
        "logs":        [],
        "error":       None,
    }
    write_status(status)
    logger.info("Pipeline started: %d step(s)", len(steps))

    for idx, step in enumerate(steps):
        status["step"]       = step["id"]
        status["step_label"] = step["label"]
        status["step_idx"]   = idx + 1
        write_status(status)

        extra = []
        if step["id"] == "download" and download_years:
            extra = ["--years"] + [str(y) for y in download_years]

        ok = run_step(step, status, extra_args=extra or None)
        if not ok:
            status["status"]      = "error"
            status["error"]       = f"Step '{step['label']}' failed — check logs above"
            status["finished_at"] = _now()
            write_status(status)
            logger.error("Pipeline aborted at step: %s", step["label"])
            return False

    status["status"]      = "done"
    status["step"]        = "done"
    status["step_label"]  = "Complete"
    status["finished_at"] = _now()
    write_status(status)
    logger.info("Pipeline finished successfully.")
    return True


def run_pipeline_async(**kwargs) -> threading.Thread:
    """Start the pipeline in a background thread (for dashboard use)."""
    t = threading.Thread(target=run_pipeline, kwargs=kwargs, daemon=True)
    t.start()
    return t


def main():
    parser = argparse.ArgumentParser(description="Run the full Procurement Twin pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip step 0 (use existing raw data)")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip step 4 (do not push to HF Hub)")
    parser.add_argument("--steps", nargs="+",
                        choices=[s["id"] for s in STEPS],
                        metavar="STEP",
                        help="Run only these steps (e.g. --steps train upload)")
    parser.add_argument("--years", nargs="+", type=int, metavar="YEAR",
                        help="Download specific years only (e.g. --years 2022 2023)")
    args = parser.parse_args()

    ok = run_pipeline(
        step_ids=args.steps,
        skip_download=args.skip_download,
        skip_upload=args.skip_upload,
        download_years=args.years,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
