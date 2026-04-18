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

# Modules users are allowed to import inside the sandbox.
# Everything pre-injected (pd, np, pl, go, px) is available without importing,
# but users naturally write import statements — we support that for safe packages.
_ALLOWED_IMPORT_TOPS = frozenset({
    "math", "statistics", "itertools", "functools", "collections",
    "datetime", "decimal", "fractions", "random", "re", "json",
    "string", "textwrap", "pprint", "copy",
    "pandas", "numpy", "polars", "plotly", "scipy",
    "sklearn", "xgboost", "shap", "statsmodels",
})

def _make_safe_import():
    _real_import = builtins.__import__
    def _safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top not in _ALLOWED_IMPORT_TOPS:
            raise ImportError(
                f"Importing '{name}' is not allowed in the sandbox. "
                f"Pre-loaded: pd, np, pl, go, px, twin, df, models, show."
            )
        return _real_import(name, *args, **kwargs)
    return _safe_import

# Safe subset of builtins — filesystem access and code execution are blocked
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
        "__builtins__": {**_SAFE_BUILTINS, "__import__": _make_safe_import()},
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
