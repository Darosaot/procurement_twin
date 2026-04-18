"""
Sandboxed Python execution environment for the Analysis tab.

Users write Python snippets against a restricted namespace:
  twin        — ProcurementTwin instance (simulate, compare, empirical_benchmark, ...)
  df          — pandas DataFrame of all 1.1M procedure records
  models      — dict of raw sklearn Pipelines: models["competition"]["model"].predict(row)
  feature_spec  — dict of feature names used by each model
  model_eval  — dict of test-set performance metrics per model
  calibration — dict of CPV/country-cluster calibration offsets
  shap_global — dict of pre-computed mean |SHAP| values per model
  params_to_df(params) — build a model-ready DataFrame from a params dict
  pd, np, pl, go, px — pandas, numpy, polars, plotly
  show(fig)   — capture a plotly Figure for display in the output panel

Dangerous builtins (import, open, exec, eval, compile, etc.) are blocked.
Execution is time-limited to TIMEOUT_SECONDS.
"""

import sys
import io
import time
import builtins
import threading
import traceback

import pandas as pd
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px

TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 20_000

# Safe subset of builtins — nothing that touches the filesystem or executes code
_SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
    "complex", "dict", "dir", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "getattr", "hasattr", "hash", "hex", "id",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "oct", "ord", "pow", "print", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "vars", "zip",
    "True", "False", "None",
    # Exceptions users might want to catch
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "StopIteration", "RuntimeError", "NotImplementedError",
}

_SAFE_BUILTINS = {
    k: getattr(builtins, k)
    for k in _SAFE_BUILTIN_NAMES
    if hasattr(builtins, k)
}


def run_code(code: str, twin, df: pd.DataFrame,
             models: dict, metadata: dict) -> dict:
    """
    Execute `code` in a sandboxed namespace.

    Parameters
    ----------
    code     : Python source string from the user
    twin     : ProcurementTwin instance
    df       : pandas DataFrame of all procedure records
    models   : dict mapping model name → {"model": Pipeline, "meta": dict}
    metadata : dict with keys feature_spec, model_eval, calibration, shap_global

    Returns a dict:
      stdout     — captured print output (truncated to MAX_OUTPUT_CHARS)
      figures    — list of plotly figure dicts (from calls to show())
      error      — traceback string or None
      elapsed_ms — wall-clock execution time in milliseconds
    """
    figures: list = []
    stdout_buf = io.StringIO()
    error_container: list = [None]

    def _show(fig):
        if isinstance(fig, go.Figure):
            figures.append(fig.to_dict())
        else:
            raise TypeError(f"show() expects a plotly Figure, got {type(fig).__name__}")

    sandbox_globals = {
        "__builtins__": _SAFE_BUILTINS,
        # High-level simulation API
        "twin": twin,
        # Raw feature data
        "df": df,
        # Raw model objects — each is {"model": sklearn Pipeline, "meta": dict}
        "models": models,
        # Model metadata dicts
        "feature_spec": metadata.get("feature_spec", {}),
        "model_eval":   metadata.get("model_eval", {}),
        "calibration":  metadata.get("calibration", {}),
        "shap_global":  metadata.get("shap_global", {}),
        # Helper: build a model-ready DataFrame from a params dict
        "params_to_df": lambda params: twin._params_to_df(params),
        # Libraries
        "pd": pd,
        "np": np,
        "pl": pl,
        "go": go,
        "px": px,
        # Output helper
        "show": _show,
    }

    def _exec():
        old_stdout = sys.stdout
        sys.stdout = stdout_buf
        try:
            exec(compile(code, "<analysis>", "exec"), sandbox_globals)  # noqa: S102
        except Exception:
            error_container[0] = traceback.format_exc()
        finally:
            sys.stdout = old_stdout

    t0 = time.time()
    thread = threading.Thread(target=_exec, daemon=True)
    thread.start()
    thread.join(timeout=TIMEOUT_SECONDS)

    elapsed_ms = round((time.time() - t0) * 1000)

    if thread.is_alive():
        return {
            "stdout": "",
            "figures": [],
            "error": f"Execution timed out after {TIMEOUT_SECONDS} seconds.",
            "elapsed_ms": elapsed_ms,
        }

    raw_output = stdout_buf.getvalue()
    truncated = len(raw_output) > MAX_OUTPUT_CHARS
    return {
        "stdout": raw_output[:MAX_OUTPUT_CHARS] + ("\n… [output truncated]" if truncated else ""),
        "figures": figures,
        "error": error_container[0],
        "elapsed_ms": elapsed_ms,
    }
