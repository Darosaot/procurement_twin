"""
Tests for the analysis sandbox — safe execution, AST pre-flight, and escape attempts.

Run:
    pytest tests/test_sandbox.py -v
"""

import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [ROOT, os.path.join(ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.analysis_sandbox import run_code, _ast_check


# ── Minimal stubs so we don't need real models ────────────────────────────────

class _StubTwin:
    def simulate(self, params, n_samples=100, seed=0):
        return {"competition": {"mean": 3.0}}


_TWIN = _StubTwin()
_DF = pd.DataFrame({"a": [1, 2, 3]})
_MODELS = {}
_META = {}


def _run(code: str) -> dict:
    return run_code(code, _TWIN, _DF, _MODELS, _META)


# ── _ast_check() ──────────────────────────────────────────────────────────────

class TestAstCheck:
    def test_clean_code_passes(self):
        _ast_check("x = 1 + 1")

    def test_syntax_error_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Syntax error"):
            _ast_check("def (")

    def test_dunder_class_blocked(self):
        import pytest
        with pytest.raises(ValueError, match="__class__"):
            _ast_check("x = ''.__class__")

    def test_dunder_subclasses_blocked(self):
        import pytest
        with pytest.raises(ValueError, match="__subclasses__"):
            _ast_check("x = object.__subclasses__()")

    def test_dunder_globals_blocked(self):
        import pytest
        with pytest.raises(ValueError, match="__globals__"):
            _ast_check("f.__globals__['os']")

    def test_dunder_dict_blocked(self):
        import pytest
        with pytest.raises(ValueError, match="__dict__"):
            _ast_check("x.__dict__")

    def test_import_os_blocked(self):
        import pytest
        with pytest.raises(ValueError, match="os"):
            _ast_check("import os")

    def test_import_subprocess_blocked(self):
        import pytest
        with pytest.raises(ValueError, match="subprocess"):
            _ast_check("import subprocess")

    def test_import_sys_blocked(self):
        import pytest
        with pytest.raises(ValueError, match="sys"):
            _ast_check("import sys")

    def test_from_os_blocked(self):
        import pytest
        with pytest.raises(ValueError, match="os"):
            _ast_check("from os import path")

    def test_allowed_import_passes(self):
        _ast_check("import math")
        _ast_check("import pandas as pd")
        _ast_check("from collections import defaultdict")


# ── run_code() — happy paths ──────────────────────────────────────────────────

class TestRunCodeHappy:
    def test_simple_print(self):
        r = _run("print('hello')")
        assert r["error"] is None
        assert "hello" in r["stdout"]

    def test_df_available(self):
        r = _run("print(len(df))")
        assert r["error"] is None
        assert "3" in r["stdout"]

    def test_numpy_available(self):
        r = _run("print(np.sqrt(4))")
        assert r["error"] is None
        assert "2.0" in r["stdout"]

    def test_pandas_available(self):
        r = _run("print(pd.DataFrame({'x':[1]}).shape)")
        assert r["error"] is None

    def test_elapsed_ms_populated(self):
        r = _run("x = 1")
        assert r["elapsed_ms"] >= 0

    def test_show_captures_figure(self):
        r = _run("import plotly.graph_objects as _go; show(_go.Figure())")
        assert r["error"] is None
        assert len(r["figures"]) == 1

    def test_twin_simulate_callable(self):
        r = _run("res = twin.simulate({'country':'DE'}, n_samples=10); print(res['competition']['mean'])")
        assert r["error"] is None

    def test_math_import_allowed(self):
        r = _run("import math; print(math.pi)")
        assert r["error"] is None
        assert "3.14" in r["stdout"]


# ── run_code() — blocked operations ──────────────────────────────────────────

class TestRunCodeBlocked:
    def test_os_import_blocked_at_ast(self):
        r = _run("import os; os.system('id')")
        assert r["error"] is not None
        assert "Security check" in r["error"]

    def test_subprocess_blocked_at_ast(self):
        r = _run("import subprocess; subprocess.run(['id'])")
        assert r["error"] is not None
        assert "Security check" in r["error"]

    def test_dunder_class_escape_blocked(self):
        r = _run("x = ''.__class__.__mro__[-1].__subclasses__()")
        assert r["error"] is not None

    def test_builtin_import_not_available(self):
        r = _run("__import__('os').system('id')")
        # __import__ is not in safe builtins so NameError expected
        assert r["error"] is not None

    def test_exec_not_available(self):
        r = _run("exec('import os')")
        assert r["error"] is not None

    def test_eval_not_available(self):
        r = _run("eval('1+1')")
        assert r["error"] is not None

    def test_open_not_available(self):
        r = _run("open('/etc/passwd')")
        assert r["error"] is not None

    def test_show_rejects_non_figure(self):
        r = _run("show('not a figure')")
        assert r["error"] is not None

    def test_output_truncated_at_limit(self):
        from dashboard.analysis_sandbox import MAX_OUTPUT_CHARS
        # Write more than the limit
        r = _run(f"print('x' * {MAX_OUTPUT_CHARS + 100})")
        assert r["error"] is None
        assert "truncated" in r["stdout"]


# ── run_code() — timeout ──────────────────────────────────────────────────────

class TestRunCodeTimeout:
    def test_infinite_loop_times_out(self):
        from dashboard.analysis_sandbox import TIMEOUT_SECONDS
        r = _run("while True: pass")
        assert r["error"] is not None
        assert "timed out" in r["error"].lower()
        assert r["elapsed_ms"] >= TIMEOUT_SECONDS * 1000 * 0.9
