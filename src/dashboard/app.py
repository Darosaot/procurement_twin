"""
Procurement Digital Twin — Plotly Dash Application  (v2)
=========================================================
Six tabs:
  1. 🎯 Procedure Designer   — real-time simulation for a single procedure
  2. ⚖️ Scenario Comparator  — side-by-side comparison of two designs
  3. 🔍 Policy Explorer      — empirical distributions from 1.1M records
  4. 🏛️ Policy Simulation    — counterfactual "what-if" aggregate analysis
  5. 💡 Explain              — SHAP feature importance & per-prediction explanations
  6. 📖 Methodology          — how the models work, calibration, Monte Carlo, limitations
"""

import sys, os, json, logging, threading

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")

logger = logging.getLogger(__name__)

for _p in [_PROJECT_ROOT, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

FEAT_DIR          = os.environ.get("FEAT_DIR",  os.path.join(_PROJECT_ROOT, "data", "features"))
MODEL_DIR         = os.environ.get("MODEL_DIR", os.path.join(_PROJECT_ROOT, "models"))
_PIPELINE_STATUS  = os.path.join(_PROJECT_ROOT, "pipeline_status.json")

import dash
from dash import dcc, html, Input, Output, State, ctx, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import polars as pl

from simulation.simulation_engine import (
    ProcurementTwin, COUNTRY_CLUSTERS, CPV_SECTORS, value_bracket
)
from dashboard.analysis_sandbox import run_code as _sandbox_run

try:
    from advisor.advisor import ProcurementAdvisor as _ProcurementAdvisor
    _advisor = _ProcurementAdvisor()
    _ADVISOR_AVAILABLE = True
except Exception as _adv_err:
    _advisor = None
    _ADVISOR_AVAILABLE = False
    logger.warning("Advisor module unavailable: %s", _adv_err)

import subprocess as _subprocess
import threading as _threading

_PIPELINE_SCRIPT = os.path.join(_PROJECT_ROOT, "pipeline", "run_pipeline.py")
_PIPELINE_AVAILABLE = os.path.isfile(_PIPELINE_SCRIPT)
if not _PIPELINE_AVAILABLE:
    logger.warning("Pipeline script not found at %s", _PIPELINE_SCRIPT)


def _run_pipeline_async(step_ids=None, download_years=None,
                        skip_download=False, skip_upload=False):
    """Start run_pipeline.py as a background subprocess — no import needed."""
    cmd = [sys.executable, _PIPELINE_SCRIPT]
    if step_ids:
        cmd += ["--steps"] + list(step_ids)
    if download_years:
        cmd += ["--years"] + [str(y) for y in download_years]
    if skip_download:
        cmd.append("--skip-download")
    if skip_upload:
        cmd.append("--skip-upload")

    def _run():
        _subprocess.run(cmd, cwd=_PROJECT_ROOT, env=os.environ.copy())

    t = _threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ── Initialise twin ───────────────────────────────────────────────
twin = ProcurementTwin()

# ── Lazy-load feature store for the Analysis sandbox ─────────────
# The 1.1M-row parquet is only needed when the user opens the Analysis tab.
# Load on first access so startup memory stays small.
_FEAT_PATH        = os.path.join(FEAT_DIR, "procedure_records.parquet")
_SANDBOX_DF       = None
_SANDBOX_DF_LOCK  = threading.Lock()


def _get_sandbox_df() -> pd.DataFrame:
    global _SANDBOX_DF
    if _SANDBOX_DF is not None:
        return _SANDBOX_DF
    with _SANDBOX_DF_LOCK:
        if _SANDBOX_DF is None:
            try:
                _SANDBOX_DF = pl.read_parquet(_FEAT_PATH).to_pandas()
                logger.info("Feature store lazy-loaded for sandbox: %d rows", len(_SANDBOX_DF))
            except Exception as _e:
                logger.warning("Could not load feature store for sandbox: %s", _e)
                _SANDBOX_DF = pd.DataFrame()
    return _SANDBOX_DF

# ── Build sandbox model + metadata dicts (reuse already-loaded twin state) ──
_SANDBOX_MODELS = {
    "competition":  twin.competition_mdl,
    "single_bid":   twin.singlebid_mdl,
    "cross_border": twin.crossborder_mdl,
    "price":        twin.price_mdl,
    "duration":     twin.duration_mdl,
}
_SANDBOX_META = {
    "feature_spec": twin.feature_spec,
    "model_eval":   twin._load_json("model_evaluation.json"),
    "calibration":  twin._load_json("calibration_offsets.json"),
    "shap_global":  twin._shap_global,
}

# ── Reference lists ───────────────────────────────────────────────
COUNTRIES = sorted(["AT","BE","BG","CY","CZ","DE","DK","EE","ES","FI","FR","GR",
                     "HR","HU","IE","IT","LT","LU","LV","MT","NL","PL","PT","RO",
                     "SE","SI","SK","UK","NO","CH","IS","MK"])

CPV_OPTIONS = [{"label": f"{k} – {v}", "value": k}
               for k, v in sorted(CPV_SECTORS.items())]

PROC_OPTIONS = [
    {"label": "Open procedure",                  "value": "OPE"},
    {"label": "Restricted procedure",            "value": "RES"},
    {"label": "Negotiated with prior call",      "value": "NIC"},
    {"label": "Competitive dialogue",            "value": "COD"},
    {"label": "Innovation partnership",          "value": "INP"},
    {"label": "Award without prior publication", "value": "AWP"},
]

PROC_DESCRIPTIONS = {
    "OPE": "Any supplier may submit a tender. Maximum competition, standard timelines.",
    "RES": "Only pre-selected suppliers are invited to tender.",
    "NIC": "Negotiation allowed after a call for competition. Used for complex contracts.",
    "COD": "Dialogue with selected candidates before tender. For complex requirements.",
    "INP": "Development and purchase of innovative solutions.",
    "AWP": "Direct award without advertising — only for exceptional circumstances.",
}

CONTRACT_OPTIONS = [
    {"label": "Services",   "value": "S"},
    {"label": "Supplies",   "value": "U"},
    {"label": "Works",      "value": "W"},
]
CRITERIA_OPTIONS = [
    {"label": "MEAT — Most Economically Advantageous Tender", "value": "M"},
    {"label": "Lowest price only",                           "value": "L"},
]

CLUSTER_OPTIONS = [
    {"label": c, "value": c}
    for c in ["Benelux","Germanic","Western","Iberian","Nordic",
              "CEE","Baltic","Balkan","Mediterranean","Anglophone"]
]

# ── Colour palette (updated design system) ────────────────────────
COL_NAVY   = "#0F172A"   # dark navy — headings, text
COL_BLUE   = "#2563EB"   # primary blue — simulate actions
COL_LIGHT  = "#E2E8F0"   # borders, dividers
COL_TEAL   = "#0891B2"   # policy/secondary actions
COL_GREEN  = "#059669"   # positive / success
COL_RED    = "#DC2626"   # risk / danger
COL_ORANGE = "#D97706"   # warning / elevated
COL_GREY   = "#64748B"   # secondary text
COL_ACCENT = "#7C3AED"   # scenario B, comparison highlights
COL_BG     = "#F1F5F9"   # page background
COL_CARD   = "#FFFFFF"   # card background

# ── Per-outcome tooltip texts (shown on ℹ️ hover in Designer & Comparator) ──
OUTCOME_TOOLTIPS = {
    "Expected bids": (
        "XGBoost model predicting log(1+bids), trained on 1.1M TED contracts. "
        "Calibrated by CPV sector (60%) and country cluster (40%). "
        "Distribution: 5,000 log-normal draws. Below 3 bids = competition risk."
    ),
    "P(single bid)": (
        "Logistic regression predicting P(only 1 offer received). "
        "EU average: ~25%. Values above 40% suggest the design may be deterring suppliers. "
        "Consider longer prep time, broader advertising, or MEAT criteria."
    ),
    "P(cross-border)": (
        "Logistic regression predicting P(winner from a different EU member state). "
        "Proxy for EU single market integration. Highest in IT services and standardised supplies; "
        "lowest in construction and locally-rooted services."
    ),
    "Price ratio": (
        "Two-stage XGBoost: Stage 1 predicts competition; Stage 2 uses competition as an "
        "instrumental variable for price. Ratio = award value ÷ official estimate. "
        "<1.0 = under-budget; >1.0 = over-budget. Clipped to [0.1, 3.0]."
    ),
    "Duration": (
        "Gradient Boosting predicting days from publication to award notice "
        "(procedure only — not contract execution). Minimum capped at 30 days. "
        "Strong predictors: procedure type, contract value, accelerated flag."
    ),
}


# ══════════════════════════════════════════════════════════════════
# SHARED FORM COMPONENT
# ══════════════════════════════════════════════════════════════════
def build_form(prefix, defaults=None):
    d = defaults or {}

    def _fsection(title, children):
        return html.Div([
            html.Div(title, className="form-section-title"),
            *children,
        ], className="form-section")

    return html.Div([
        _fsection("Contract Details", [
            _form_row("Country", dcc.Dropdown(
                id=f"{prefix}-country",
                options=[{"label": c, "value": c} for c in COUNTRIES],
                value=d.get("country", "DE"), clearable=False)),

            _form_row("Procedure type", html.Div([
                dcc.Dropdown(id=f"{prefix}-proc", options=PROC_OPTIONS,
                             value=d.get("proc", "OPE"), clearable=False),
                html.Div(id=f"{prefix}-proc-desc",
                         style={"fontSize": "11px", "color": COL_GREY, "marginTop": "4px",
                                "fontStyle": "italic", "lineHeight": "1.4"}),
            ])),

            _form_row("Contract type", dcc.RadioItems(
                id=f"{prefix}-ctype", options=CONTRACT_OPTIONS,
                value=d.get("ctype", "S"), inline=True, className="radio-inline")),

            _form_row("CPV sector", dcc.Dropdown(
                id=f"{prefix}-cpv", options=CPV_OPTIONS,
                value=d.get("cpv", "72"), clearable=False)),
        ]),

        _fsection("Evaluation Criteria", [
            _form_row("Award criteria", dcc.RadioItems(
                id=f"{prefix}-crit", options=CRITERIA_OPTIONS,
                value=d.get("crit", "M"), className="radio-block")),

            _form_row("Price weight (%)", html.Div([
                html.Div(id=f"{prefix}-pw-val",
                         style={"fontSize": "11px", "color": COL_GREY,
                                "textAlign": "right", "marginBottom": "2px"}),
                dcc.Slider(id=f"{prefix}-pw", min=0, max=100, step=5,
                           value=d.get("pw", 60),
                           marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ]), id_suffix="-pw-row"),
        ]),

        _fsection("Financial & Timeline", [
            _form_row("Estimated value (€)", dcc.Input(
                id=f"{prefix}-val", type="number", value=d.get("val", 1_000_000),
                min=10_000, step=10_000,
                style={"width": "100%", "padding": "7px 10px",
                       "border": f"1px solid {COL_LIGHT}", "borderRadius": "7px",
                       "fontSize": "13px", "fontFamily": "inherit"})),

            _form_row("Preparation time (days)", dcc.Slider(
                id=f"{prefix}-prep", min=15, max=90, step=1,
                value=d.get("prep", 35),
                marks={15: "15", 35: "35", 52: "52", 90: "90"},
                tooltip={"placement": "bottom", "always_visible": True})),

            _form_row("Contract duration (months)", dcc.Slider(
                id=f"{prefix}-dur", min=3, max=60, step=3,
                value=d.get("dur", 24),
                marks={3: "3m", 12: "1yr", 24: "2yr", 36: "3yr", 60: "5yr"},
                tooltip={"placement": "bottom", "always_visible": True})),
        ]),

        _fsection("Compliance & Options", [
            _form_row("Flags", dcc.Checklist(
                id=f"{prefix}-flags",
                options=[
                    {"label": " GPA covered",         "value": "gpa"},
                    {"label": " EU funds",            "value": "eu_funds"},
                    {"label": " Electronic auction",  "value": "ea"},
                    {"label": " Framework agreement", "value": "fra"},
                    {"label": " Accelerated",         "value": "acc"},
                ],
                value=d.get("flags", ["gpa"]), className="checklist")),
        ]),

        html.Button("▶  Run Simulation", id=f"{prefix}-btn",
                    n_clicks=0, className="btn-primary"),

        html.Div(id=f"{prefix}-status", style={
            "fontSize": "12px", "color": COL_BLUE, "marginTop": "8px",
            "minHeight": "16px", "fontStyle": "italic",
        }),
    ], className="form-panel")


def _form_row(label, component, id_suffix=None):
    return html.Div([
        html.Label(label, className="form-label"),
        component,
    ], className="form-group", **({"id": id_suffix} if id_suffix else {}))


def form_to_params(country, proc, ctype, cpv, crit, val, prep, dur, pw, flags):
    flags = flags or []
    return {
        "country":            country,
        "procedure_type":     proc,
        "contract_type":      ctype,
        "cpv_division":       cpv,
        "criteria":           crit,
        "price_weight_pct":   float(pw) if pw else 50,
        "value_euro":         float(val) if val else 1_000_000,
        "prep_time_days":     float(prep) if prep else 35,
        "duration_months":    float(dur) if dur else 24,
        "gpa":                "gpa" in flags,
        "eu_funds":           "eu_funds" in flags,
        "electronic_auction": "ea" in flags,
        "fra_agreement":      "fra" in flags,
        "accelerated":        "acc" in flags,
    }


# ══════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════
def dist_chart(samples, label, color, vline=None, height=200):
    arr = np.array(samples)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=arr, nbinsx=40, marker_color=color,
                               opacity=0.82, name=label))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color=COL_RED,
                      line_width=1.5,
                      annotation_text=f"  {vline:.2f}",
                      annotation_font=dict(size=11, color=COL_RED))
    fig.update_layout(
        title=dict(text=label, font=dict(size=12, color=COL_GREY, family="Inter, sans-serif")),
        height=height, margin=dict(t=36, b=26, l=36, r=12),
        paper_bgcolor=COL_CARD, plot_bgcolor="#F8FAFC",
        showlegend=False,
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COL_LIGHT, linecolor=COL_LIGHT, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=COL_LIGHT, linecolor=COL_LIGHT, tickfont=dict(size=10)),
    )
    return fig


def kpi_card(label, value_str, badge=None, badge_col=None, note=None, tooltip_text=None,
             accent=None):
    """Redesigned KPI card with accent stripe, badge, and info tooltip."""
    accent_color = accent or COL_BLUE
    badge_class = "kpi-badge kpi-badge-gray"
    if badge_col == COL_GREEN:
        badge_class = "kpi-badge kpi-badge-green"
    elif badge_col == COL_RED:
        badge_class = "kpi-badge kpi-badge-red"

    tooltip_el = []
    if tooltip_text:
        tooltip_el = [html.Span(
            "i",
            className="kpi-info-icon",
            **{"data-kpi-tooltip": tooltip_text},
        )]

    children = [
        html.Div(className="kpi-card-accent", style={"--accent": accent_color}),
        html.Div(value_str, className="kpi-value"),
        html.Div([
            html.Span(label, className="kpi-label"),
            *tooltip_el,
        ], className="kpi-label-row"),
    ]
    if badge:
        children.append(html.Div([badge], className=badge_class))
    if note:
        children.append(html.Div(note, className="kpi-note"))

    return html.Div(children, className="kpi-card")


def _section_header(text):
    return html.H3(text, className="section-header")


def _card(children, style=None):
    s = {"backgroundColor": COL_CARD, "padding": "20px", "borderRadius": "12px",
         "border": f"1px solid {COL_LIGHT}", "boxShadow": "0 1px 4px rgba(15,23,42,0.04)"}
    if style:
        s.update(style)
    return html.Div(children, style=s)


def _phdr(title, subtitle, group=None):
    """Page header: sticky bar with breadcrumb, title, and subtitle."""
    crumb = []
    if group:
        crumb = [
            html.Span(group, className="breadcrumb-sep"),
            html.Span(" / ", className="breadcrumb-sep"),
            html.Span(title, className="breadcrumb-active"),
        ]
    return html.Div([
        *([html.Div(crumb, className="page-breadcrumb")] if crumb else []),
        html.Div(title, className="page-header-title"),
        html.Div(subtitle, className="page-header-sub"),
    ], className="page-header")


# ══════════════════════════════════════════════════════════════════
# APP SHELL  ─  sidebar navigation + content area
# ══════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, title="Procurement Digital Twin",
                suppress_callback_exceptions=True)

# Canonical page IDs — used for pattern-matching callbacks
NAV_PAGES = [
    "home", "designer", "compare", "explorer", "radar",
    "optimise", "advisor", "policy", "explain", "methodology",
    "analysis", "admin",
]


def _nav_item(icon, text, page_id):
    return html.Button(
        [html.Span(icon, className="nav-icon"), html.Span(text)],
        id={"type": "nav-btn", "page": page_id},
        className="nav-item",
        n_clicks=0,
    )


def _nav_group(label, items):
    return html.Div(
        [html.Div(label, className="nav-group-label"),
         *[_nav_item(ic, tx, pg) for ic, tx, pg in items]],
        className="nav-group",
    )


def _sidebar():
    return html.Nav([
        # Brand
        html.Div([
            html.Div("⬡", className="brand-mark"),
            html.Div([
                html.Div("Digital Twin", className="brand-name"),
                html.Div("EU Procurement", className="brand-sub"),
            ]),
        ], className="sidebar-brand"),

        # Home
        html.Div([
            _nav_item("⌂", "Overview", "home"),
        ], className="nav-group", style={"padding": "6px 8px 2px"}),

        html.Div(className="sidebar-divider"),

        _nav_group("SIMULATE", [
            ("◎", "Procedure Designer", "designer"),
            ("⇄", "Scenario Comparator", "compare"),
        ]),
        _nav_group("DISCOVER", [
            ("◉", "Policy Explorer", "explorer"),
            ("▦", "Risk Radar", "radar"),
        ]),
        _nav_group("OPTIMIZE", [
            ("✦", "Optimisation Lab", "optimise"),
            ("◈", "AI Advisor", "advisor"),
        ]),
        _nav_group("IMPACT", [
            ("◤", "Policy Simulation", "policy"),
        ]),

        html.Div(className="sidebar-divider"),

        _nav_group("LEARN", [
            ("⊞", "Explain Models", "explain"),
            ("◯", "Methodology", "methodology"),
        ]),
        _nav_group("DEVELOP", [
            ("⟨⟩", "Analysis Sandbox", "analysis"),
            ("⚙", "Model Admin", "admin"),
        ]),

        # Footer
        html.Div([
            html.Div([
                html.Div(className="sidebar-data-pill-dot"),
                html.Span("Models loaded", className="sidebar-data-pill-text"),
            ], className="sidebar-data-pill"),
            html.Div("1.1M TED contracts · 2018–2023", className="sidebar-footer-label"),
        ], className="sidebar-footer"),
    ], className="sidebar")


app.layout = html.Div([
    dcc.Store(id="nav-store", data="home"),
    _sidebar(),
    html.Div(
        html.Div(id="page-content"),
        className="main-area",
    ),
], className="app-shell")


# ── Page render ───────────────────────────────────────────────────
@app.callback(Output("page-content", "children"), Input("nav-store", "data"))
def render_page(page):
    dispatch = {
        "home":        home_layout,
        "designer":    designer_layout,
        "compare":     comparator_layout,
        "explorer":    explorer_layout,
        "policy":      policy_layout,
        "explain":     explain_layout,
        "methodology": methodology_layout,
        "analysis":    analysis_layout,
        "optimise":    optimise_layout,
        "advisor":     advisor_layout,
        "radar":       radar_layout,
        "admin":       admin_layout,
    }
    return dispatch.get(page, home_layout)()


# ── Sidebar click handler ─────────────────────────────────────────
@app.callback(
    Output("nav-store", "data"),
    Input({"type": "nav-btn", "page": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _handle_nav(n_clicks):
    if not ctx.triggered_id:
        raise PreventUpdate
    tid = ctx.triggered_id
    if isinstance(tid, dict) and "page" in tid:
        return tid["page"]
    raise PreventUpdate


# ── Active nav highlighting ───────────────────────────────────────
@app.callback(
    Output({"type": "nav-btn", "page": ALL}, "className"),
    Input("nav-store", "data"),
)
def _update_nav_classes(page):
    ids = [o["id"]["page"] for o in ctx.outputs_list]
    return ["nav-item active" if p == page else "nav-item" for p in ids]


# ══════════════════════════════════════════════════════════════════
# HOME  ─  overview dashboard
# ══════════════════════════════════════════════════════════════════
def home_layout():
    def _qs(icon, title, desc, page_id, bg):
        return html.Button([
            html.Div(icon, className="qs-icon", style={"background": bg}),
            html.Div(title, className="qs-title"),
            html.Div(desc, className="qs-desc"),
            html.Span("→", className="qs-arrow"),
        ], id={"type": "nav-btn", "page": page_id},
           n_clicks=0, className="quick-start-card")

    return html.Div([
        # Page header
        html.Div([
            html.Div("Procurement Digital Twin", className="page-header-title"),
            html.Div("EU procurement simulator · Monte Carlo engine · 5 ML models",
                     className="page-header-sub"),
        ], className="page-header"),

        html.Div([
            # Hero banner
            html.Div([
                html.Div([
                    html.Div(className="home-hero-eyebrow-dot"),
                    html.Span("Live · Models loaded"),
                ], className="home-hero-eyebrow"),

                html.Div([
                    "Simulate any ",
                    html.Span("EU procurement"),
                    " procedure",
                ], className="home-hero-title"),

                html.P(
                    "Design optimal procedures, compare policy scenarios, and understand "
                    "competition outcomes — powered by machine learning trained on 1.1 million "
                    "TED contract notices across 27+ European countries.",
                    className="home-hero-sub",
                ),

                html.Div([
                    html.Div([
                        html.Div([html.Span("1.1"), html.Span("M", className="home-stat-sup")],
                                 className="home-stat-val"),
                        html.Div("TED Contracts", className="home-stat-label"),
                    ], className="home-stat"),
                    html.Div([
                        html.Div("5", className="home-stat-val"),
                        html.Div("ML Models", className="home-stat-label"),
                    ], className="home-stat"),
                    html.Div([
                        html.Div([html.Span("27"), html.Span("+", className="home-stat-sup")],
                                 className="home-stat-val"),
                        html.Div("EU Countries", className="home-stat-label"),
                    ], className="home-stat"),
                    html.Div([
                        html.Div("5K", className="home-stat-val"),
                        html.Div("Monte Carlo samples", className="home-stat-label"),
                    ], className="home-stat"),
                    html.Div([
                        html.Div("2018–23", className="home-stat-val"),
                        html.Div("Data coverage", className="home-stat-label"),
                    ], className="home-stat"),
                ], className="home-stats"),
            ], className="home-hero"),

            # Quick-start cards (row 1)
            html.Div("WHAT WOULD YOU LIKE TO DO?", className="home-section-title"),
            html.Div([
                _qs("🎯", "Design a Procedure",
                    "Configure any EU procurement procedure and simulate competition, "
                    "single-bid risk, price ratio, cross-border rate, and duration.",
                    "designer", "#EFF6FF"),
                _qs("⚖️", "Compare Two Scenarios",
                    "Run side-by-side Monte Carlo simulations to find which design "
                    "delivers better outcomes before publishing the notice.",
                    "compare", "#F0FDF4"),
                _qs("🔍", "Explore Historical Data",
                    "Browse empirical distributions from 1.1M TED records. "
                    "Filter by country, CPV sector, procedure type, and year range.",
                    "explorer", "#ECFEFF"),
                _qs("📊", "Risk Radar",
                    "Scan a 10×10 heatmap of procurement segments — identify where "
                    "single-bid risk and low competition are most acute.",
                    "radar", "#FFF7ED"),
            ], className="quick-start-grid"),

            # Quick-start cards (row 2)
            html.Div([
                _qs("🏆", "Optimise Design",
                    "Multi-objective search across the procedure design space — "
                    "maximize competition while minimising cost, risk and duration.",
                    "optimise", "#F5F3FF"),
                _qs("🧠", "AI Advisor",
                    "Get evidence-based design recommendations backed by SHAP "
                    "feature importances and 1.1M historical benchmarks.",
                    "advisor", "#FFF1F2"),
                _qs("🏛️", "Policy Simulation",
                    "Model the aggregate counterfactual impact of a policy "
                    "intervention across an entire procurement market segment.",
                    "policy", "#F0FDFA"),
                _qs("💡", "Explain Predictions",
                    "Understand exactly which parameters drive each outcome "
                    "prediction using SHAP waterfall charts and natural-language summaries.",
                    "explain", "#FEFCE8"),
            ], className="quick-start-grid"),

            # Info callout
            html.Div([
                html.Strong("Tip: "),
                "All simulations run 5,000 Monte Carlo draws and report P10–P90 "
                "confidence intervals. Results are calibrated per CPV sector (60%) "
                "and country cluster (40%) against 1.1M historical contracts.",
            ], className="info-callout", style={"marginTop": "24px"}),

        ], className="page-body"),
    ])


# ══════════════════════════════════════════════════════════════════
# TAB 6: METHODOLOGY  ─  how the models and simulation work
# ══════════════════════════════════════════════════════════════════
def methodology_layout():

    # ── Helper: one model description card ────────────────────────
    def _mcard(icon, title, algo_label, algo_color, paras, features, interp):
        return html.Div([
            # Title row
            html.Div([
                html.Span(icon, style={"fontSize":"26px","marginRight":"10px",
                                       "lineHeight":"1"}),
                html.Div([
                    html.Div(title,
                             style={"fontWeight":"700","color":COL_NAVY,
                                    "fontSize":"14px","lineHeight":"1.2"}),
                    html.Span(algo_label,
                              style={"fontSize":"10px","fontWeight":"700",
                                     "color":algo_color,
                                     "backgroundColor":algo_color+"22",
                                     "padding":"1px 7px","borderRadius":"3px",
                                     "marginTop":"4px","display":"inline-block"}),
                ]),
            ], style={"display":"flex","alignItems":"flex-start","marginBottom":"10px"}),
            # Description paragraphs
            *[html.P(p, style={"fontSize":"12px","color":"#444","lineHeight":"1.55",
                                "margin":"0 0 7px 0"}) for p in paras],
            # Feature list
            html.Div([
                html.Div("Key predictors:",
                         style={"fontSize":"10px","fontWeight":"700","color":COL_GREY,
                                "marginBottom":"4px","textTransform":"uppercase",
                                "letterSpacing":"0.5px"}),
                html.Div(" · ".join(features),
                         style={"fontSize":"11px","color":"#555","lineHeight":"1.55"}),
            ], style={"backgroundColor":COL_BG,"padding":"8px 10px",
                       "borderRadius":"6px","marginBottom":"10px"}),
            # How to read
            html.Div([
                html.Span("📖  ", style={"fontSize":"12px"}),
                html.Span(interp,
                          style={"fontSize":"11px","color":"#444","fontStyle":"italic"}),
            ]),
        ], style={
            "backgroundColor":COL_CARD,"padding":"18px","borderRadius":"10px",
            "boxShadow":"0 2px 8px rgba(0,0,0,0.07)",
            "borderTop":f"4px solid {algo_color}",
            "flex":"1","minWidth":"200px",
        })

    # ── Five model cards ───────────────────────────────────────────
    model_row_1 = html.Div([
        _mcard("🏆", "Competition", "XGBoost (gradient boosting)", COL_BLUE,
            [
                "Predicts the number of offers expected for a procedure. "
                "Trained on log(1 + n_bids) to handle the right-skewed distribution of "
                "offer counts, then back-transformed. The calibration system applies "
                "segment-specific corrections on top of the base prediction.",
                "A CPV-sector offset and a country-cluster offset are blended "
                "(60%/40%) and added in log-space to correct for persistent "
                "biases not captured by the cross-sectional model.",
            ],
            ["CPV sector", "procedure type", "contract value (log₁₀)", "country cluster",
             "award criteria", "prep time", "EU funds", "GPA coverage", "electronic auction"],
            "Below 3 bids signals limited competition. Below 2 raises serious VfM concerns."
        ),
        _mcard("⚠️", "Single-bid risk", "Logistic Regression", COL_RED,
            [
                "Predicts the probability that only one supplier submits an offer — "
                "the most direct indicator of failed competition. "
                "EU-wide average is roughly 25%, but rises sharply for complex, "
                "high-value, or restrictive procedures.",
                "A Bernoulli draw from this probability is included in each Monte "
                "Carlo iteration, so the distribution chart reflects both the model "
                "uncertainty and the inherent randomness of the outcome.",
            ],
            ["procedure type", "CPV sector", "country cluster", "contract value",
             "award criteria", "prep time", "accelerated flag"],
            "Values above 35–40% warrant reviewing the design — consider longer prep time "
            "or switching to MEAT criteria."
        ),
        _mcard("🌍", "Cross-border win", "Logistic Regression", COL_TEAL,
            [
                "Predicts the probability that the contract is awarded to a supplier "
                "from a different EU member state — a proxy for EU single market integration. "
                "Rates are highest in IT services and standardised supplies; "
                "lowest in construction and locally-rooted services.",
                "This is modelled as a simple logistic regression because the outcome "
                "is highly determined by sector and country characteristics rather than "
                "fine-grained procedure design choices.",
            ],
            ["CPV sector", "country", "contract value", "procedure type", "criteria"],
            "Low rates are not necessarily problematic; near-zero rates sustained "
            "over time may signal local market barriers worth investigating."
        ),
    ], style={"display":"flex","gap":"14px","marginBottom":"14px","flexWrap":"wrap"})

    model_row_2 = html.Div([
        _mcard("💶", "Price ratio", "XGBoost · 2-stage IV", COL_ACCENT,
            [
                "Predicts the ratio of the awarded contract value to the contracting "
                "authority's official estimate. A ratio below 1.0 means the award was "
                "below estimate; above 1.0 means over-budget.",
                "The model uses a two-stage approach: first, competition is predicted "
                "(Stage 1). That predicted competition value is then fed as an "
                "instrumental variable into the price model (Stage 2). This captures "
                "the causal mechanism — more competition tends to push award prices down — "
                "while avoiding endogeneity bias.",
            ],
            ["competition_hat (predicted bids)", "CPV sector", "country cluster",
             "contract value", "procedure type", "criteria", "EU funds", "GPA"],
            "Typical range 0.85–1.10. Persistent values above 1.0 suggest estimate "
            "quality issues. Clipped to [0.10, 3.00] in simulation."
        ),
        _mcard("⏱️", "Procedure duration", "Gradient Boosting", COL_GREEN,
            [
                "Predicts the number of calendar days from contract notice publication "
                "to the award notice. Covers only the procurement procedure itself — "
                "not the contract execution period.",
                "Duration is modelled on log(1 + days) to handle skewness. "
                "The minimum is capped at 30 days to reflect EU directive minimums. "
                "The accelerated flag has a strong negative coefficient.",
            ],
            ["procedure type", "CPV sector", "contract value (log₁₀)", "prep time",
             "accelerated flag", "country cluster", "contract duration (months)"],
            "EU directive minimums: Open = 35 days, Restricted = 30 days. "
            "Results shorter than these are artefacts of timing in the TED data."
        ),
    ], style={"display":"flex","gap":"14px","marginBottom":"28px","flexWrap":"wrap"})

    # ── Pipeline flow diagram ──────────────────────────────────────
    def _step(icon, title, sub, color):
        return html.Div([
            html.Div(icon, style={"fontSize":"22px","marginBottom":"5px"}),
            html.Div(title, style={"fontSize":"12px","fontWeight":"700",
                                    "color":"white","lineHeight":"1.2"}),
            html.Div(sub,   style={"fontSize":"10px","color":"rgba(255,255,255,0.82)",
                                    "marginTop":"3px","lineHeight":"1.3"}),
        ], className="meth-pipeline-step",
           style={"backgroundColor":color,"flex":"1"})

    def _arr():
        return html.Div("→", style={"fontSize":"20px","color":COL_GREY,
                                     "alignSelf":"center","padding":"0 5px",
                                     "flexShrink":"0"})

    pipeline_diagram = html.Div([
        _step("🖊️", "User inputs",
              "country, CPV, value,\nprocedure, criteria…", COL_NAVY),
        _arr(),
        _step("⚙️", "Feature engineering",
              "encode, scale, cluster\nmapping, log-transforms", COL_BLUE),
        _arr(),
        _step("🤖", "5 trained models",
              "XGBoost + Logistic\nRegression · point preds", COL_ACCENT),
        _arr(),
        _step("🎲", "Monte Carlo sampling",
              "5,000 draws from\ncalibrated noise dist.", COL_TEAL),
        _arr(),
        _step("📊", "Distributions",
              "P10–P90, mean, median,\n95% bootstrap CIs", COL_GREEN),
    ], style={"display":"flex","alignItems":"stretch","gap":"4px",
               "marginBottom":"12px"})

    mc_explainer = html.Div([
        html.P(
            "Each simulation draws 5,000 samples from the uncertainty distribution around each "
            "model's point prediction. For continuous outcomes (competition, price ratio, duration), "
            "noise is log-normal with σ scaled to 60–70% of the training residual standard deviation — "
            "reflecting that some variability is predictable from inputs, but not all. "
            "Binary outcomes (single-bid risk, cross-border win) are drawn from Bernoulli "
            "distributions parameterised by the logistic regression probability.",
            style={"fontSize":"12px","color":"#444","lineHeight":"1.6","margin":"0 0 8px 0"}),
        html.P(
            "Why not use the full residual σ? The training residual captures both true uncertainty "
            "and noise in the training labels (e.g. data entry errors in TED). Using 60–70% avoids "
            "over-inflating uncertainty for users who have well-specified procedures.",
            style={"fontSize":"12px","color":"#666","lineHeight":"1.6","margin":"0",
                   "fontStyle":"italic"}),
    ], style={"backgroundColor":"#F7FAFE","padding":"12px 16px","borderRadius":"6px",
               "borderLeft":f"3px solid {COL_BLUE}"})

    # ── Calibration section ────────────────────────────────────────
    def _cal_box(title, body, color):
        return html.Div([
            html.H4(title, style={"fontSize":"13px","color":color,
                                   "fontWeight":"700","marginBottom":"6px","marginTop":"0"}),
            html.P(body, style={"fontSize":"12px","color":"#444",
                                 "lineHeight":"1.55","margin":"0"}),
        ], style={"backgroundColor":COL_CARD,"padding":"16px","borderRadius":"8px",
                   "boxShadow":"0 1px 5px rgba(0,0,0,0.07)",
                   "borderTop":f"3px solid {color}","flex":"1","minWidth":"0"})

    calibration_section = html.Div([
        html.Div([
            _cal_box("CPV sector offset  (weight: 60%)",
                     "The base model is trained on all procedures together. However, competition "
                     "patterns differ sharply by sector — IT services (CPV 72) typically attract "
                     "far more bids than niche defence or infrastructure contracts. After training, "
                     "the mean residual per CPV division is stored as a correction offset and added "
                     "back in log-space at prediction time.",
                     COL_BLUE),
            html.Div(style={"width":"14px","flexShrink":"0"}),
            _cal_box("Country cluster offset  (weight: 40%)",
                     "The 30 countries are grouped into 10 clusters (Benelux, Germanic, CEE, Nordic, "
                     "Iberian, Baltic, Balkan, Mediterranean, Western, Anglophone) based on "
                     "procurement system similarities. Cluster-level mean residuals are stored "
                     "and blended with the CPV offset: "
                     "final_offset = 0.6 × cpv_offset + 0.4 × cluster_offset.",
                     COL_ACCENT),
            html.Div(style={"width":"14px","flexShrink":"0"}),
            _cal_box("Why calibrate at all?",
                     "Without calibration, a model trained on all EU procedures would systematically "
                     "under-predict competition for IT services in Germany and over-predict it for "
                     "construction in Romania. Calibration offsets anchor predictions to "
                     "segment-specific historical averages while the model still captures "
                     "within-segment variation from fine-grained features.",
                     COL_TEAL),
        ], style={"display":"flex","marginBottom":"0"}),
    ])

    # ── Reading the results guide ──────────────────────────────────
    reading_rows = [
        ("KPI value",
         "The mean (or median for duration) of the 5,000 Monte Carlo draws. "
         "Use as the central estimate — not a hard prediction."),
        ("▲/▼ badge vs benchmark",
         "Compares the simulation mean to the historical median for procedures matching "
         "the same country, procedure type, and CPV sector in the TED 2018–2023 data. "
         "Green = better than benchmark; red = worse."),
        ("Distribution charts",
         "Show 1,000 of the 5,000 draws. Width reflects uncertainty. "
         "A wide, right-skewed distribution means outcomes are highly variable for this segment. "
         "The red dashed line marks the mean."),
        ("P10 – P90 range",
         "80% of simulated outcomes fall within this range. Narrow = high confidence; "
         "wide = the procedure design alone does not determine the outcome."),
        ("Policy Δ charts",
         "Each bar is one sampled historical procedure. Bars right of zero = the intervention "
         "improved that outcome; left = worsened it. The 95% confidence interval on the KPI card "
         "comes from 1,000 bootstrap resamples of the mean delta."),
        ("SHAP contributions",
         "Blue bars (rightward) push the prediction above the model's average baseline. "
         "Red bars (leftward) push it below. The baseline (x=0) is the model intercept "
         "— roughly the EU-wide average log-count or log-odds."),
    ]

    reading_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Element",
                    style={"padding":"9px 14px","backgroundColor":COL_NAVY,
                           "color":"white","textAlign":"left","fontSize":"12px",
                           "fontWeight":"700","whiteSpace":"nowrap"}),
            html.Th("What it means",
                    style={"padding":"9px 14px","backgroundColor":COL_NAVY,
                           "color":"white","textAlign":"left","fontSize":"12px",
                           "fontWeight":"700"}),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(k,
                        style={"padding":"9px 14px","fontWeight":"600",
                               "fontSize":"12px","color":COL_NAVY,
                               "whiteSpace":"nowrap",
                               "backgroundColor":"#F7FAFE" if i%2==0 else COL_CARD,
                               "borderBottom":"1px solid #EEE"}),
                html.Td(v,
                        style={"padding":"9px 14px","fontSize":"12px","color":"#444",
                               "lineHeight":"1.5",
                               "backgroundColor":"#F7FAFE" if i%2==0 else COL_CARD,
                               "borderBottom":"1px solid #EEE"}),
            ])
            for i, (k, v) in enumerate(reading_rows)
        ]),
    ], style={"width":"100%","borderCollapse":"collapse","borderRadius":"8px",
               "overflow":"hidden","boxShadow":"0 1px 4px rgba(0,0,0,0.08)"})

    # ── Training data section ──────────────────────────────────────
    data_section = html.Div([
        html.Div([
            html.Div([
                html.Div("📦", style={"fontSize":"32px","marginBottom":"6px"}),
                html.Div("1,100,000+",
                         style={"fontSize":"26px","fontWeight":"700","color":COL_NAVY,
                                "lineHeight":"1"}),
                html.Div("linked procedures",
                         style={"fontSize":"11px","color":COL_GREY,"textTransform":"uppercase",
                                "letterSpacing":"0.5px","marginTop":"3px"}),
            ], style={"textAlign":"center","padding":"16px","backgroundColor":COL_CARD,
                       "borderRadius":"8px","boxShadow":"0 1px 5px rgba(0,0,0,0.07)",
                       "flex":"0 0 auto","minWidth":"140px"}),
            html.Div([
                html.H4("Data source",
                        style={"marginTop":"0","marginBottom":"6px",
                               "color":COL_NAVY,"fontSize":"13px","fontWeight":"700"}),
                html.P(
                    "Models were trained on CFC–CAN linked notice pairs from the EU Tenders "
                    "Electronic Daily (TED) database, covering calendar years 2018–2023. "
                    "CFC = Contract Future Contract notice (the call for competition, capturing "
                    "design parameters); CAN = Contract Award Notice (the outcome). "
                    "Linking these two notice types enables supervised learning: procedure design "
                    "features at call time → outcome at award time.",
                    style={"fontSize":"12px","color":"#444","lineHeight":"1.6","margin":"0 0 8px 0"}),
                html.P(
                    "Approximately 68% of all CANs in the raw TED export were successfully "
                    "linked to a CFC notice via TED's internal reference fields "
                    "(TITLE_CONTRACT_FRAMEWORK, RELATED_LOTS, and NOTICE_REF). "
                    "Unlinked records (direct awards, missing references) were excluded from training.",
                    style={"fontSize":"12px","color":"#666","lineHeight":"1.6","margin":"0",
                           "fontStyle":"italic"}),
            ], style={"flex":"1","marginLeft":"20px"}),
        ], style={"display":"flex","alignItems":"flex-start","marginBottom":"14px"}),
        html.Div([
            html.Div("🗺️", style={"fontSize":"14px","marginRight":"8px"}),
            html.Span("Coverage: ",
                      style={"fontWeight":"700","fontSize":"12px","color":COL_NAVY}),
            html.Span(
                "Austria, Belgium, Bulgaria, Croatia, Cyprus, Czechia, Denmark, Estonia, "
                "Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania, "
                "Luxembourg, Malta, Netherlands, Poland, Portugal, Romania, Slovakia, Slovenia, "
                "Spain, Sweden  +  UK, Norway, Switzerland, Iceland, North Macedonia.",
                style={"fontSize":"12px","color":"#444"}),
        ], style={"backgroundColor":"#F7FAFE","padding":"10px 14px","borderRadius":"6px",
                   "borderLeft":f"3px solid {COL_BLUE}","display":"flex","alignItems":"flex-start"}),
    ])

    # ── Known limitations ──────────────────────────────────────────
    limitations = [
        ("📅", "Training window",
         "Models were trained on TED data from 2018 to 2023. Predictions may be less reliable for "
         "market conditions or regulatory changes that occurred after 2023, including the revised "
         "EU procurement thresholds and any post-pandemic supply chain shifts."),
        ("📊", "Probabilistic outputs",
         "All predictions are distributional estimates, not guarantees. "
         "The tool is designed for comparative analysis (design A vs B, policy X vs Y) rather than "
         "precise point forecasting. Always cross-reference with local market intelligence."),
        ("🏗️", "Scope",
         "Covers EU/EEA above-threshold public procurement published on TED. Results may not "
         "apply to below-threshold contracts, framework agreement call-offs, joint procurements, "
         "or procurement systems outside the EU scope."),
        ("🔄", "No dynamic feedback",
         "The model does not capture feedback loops (e.g. market saturation after repeated "
         "similar procurements, strategic supplier responses to known criteria, or "
         "incumbent advantage). Each procedure is treated independently."),
        ("📋", "Estimate quality",
         "The price ratio model depends on the accuracy of the contracting authority's own "
         "estimated value. Artificially low or inflated estimates will distort the ratio prediction "
         "regardless of model quality."),
        ("🌐", "Cross-border model",
         "Cross-border win rates are low-variance outcomes (most procedures award to a domestic "
         "supplier). The logistic regression captures sector and country patterns well but has "
         "limited power for predicting individual-procedure cross-border outcomes."),
    ]

    limit_items = html.Div([
        html.Div([
            html.Div([
                html.Span(icon, style={"fontSize":"16px","marginRight":"8px"}),
                html.Span(title,
                          style={"fontWeight":"700","fontSize":"12px","color":COL_NAVY}),
            ], style={"marginBottom":"4px"}),
            html.P(text,
                   style={"fontSize":"12px","color":"#444","lineHeight":"1.55","margin":"0"}),
        ], className="meth-limitation-row")
        for icon, title, text in limitations
    ])

    # ── Section wrapper ────────────────────────────────────────────
    def _section(title, content):
        return html.Div([
            html.H3(title, className="meth-section-title"),
            content,
        ], style={"marginBottom":"28px"})

    # ── Assemble page ──────────────────────────────────────────────
    return html.Div([
        _phdr("Methodology",
              "How the five ML models, Monte Carlo engine, and calibration system work.",
              "Learn"),
        # Intro banner
        html.Div([
            html.H2("How the Procurement Digital Twin works",
                    style={"color":"white","margin":"0 0 8px 0","fontSize":"20px",
                           "fontWeight":"700"}),
            html.P(
                "A machine-learning simulation engine trained on 1.1M EU procurement contracts "
                "(TED, 2018–2023). This page explains what each prediction means, how uncertainty "
                "is modelled, how calibration is applied, and where the estimates come from.",
                style={"color":"#A8C4E0","margin":"0","fontSize":"13px","lineHeight":"1.55"}),
        ], style={"backgroundColor":COL_NAVY,"padding":"22px 28px"}),

        html.Div([
            _section("🤖  The five predictions", html.Div([model_row_1, model_row_2])),
            _section("🔄  Simulation pipeline",
                     html.Div([pipeline_diagram, mc_explainer])),
            _section("🎯  Calibration system", calibration_section),
            _section("📖  Reading the results", reading_table),
            _section("📦  Training data", data_section),
            _section("⚠️  Known limitations & caveats", limit_items),

            # Footer note
            html.Div([
                html.P(
                    "Procurement Digital Twin  ·  EU TED data 2018–2023  ·  "
                    "Models: XGBoost, Logistic Regression, Gradient Boosting  ·  "
                    "Hover ⓘ on any KPI in the Procedure Designer for per-metric explanations.",
                    style={"fontSize":"11px","color":COL_GREY,"margin":"0","textAlign":"center"}),
            ], style={"borderTop":f"1px solid {COL_LIGHT}","paddingTop":"14px",
                       "marginTop":"8px"}),
        ], style={"padding":"24px 28px","maxWidth":"1200px"}),
    ])


# ══════════════════════════════════════════════════════════════════
# TAB 1: PROCEDURE DESIGNER
# ══════════════════════════════════════════════════════════════════
def designer_layout():
    return html.Div([
        _phdr("Procedure Designer",
              "Configure a procurement procedure and run a 5,000-sample Monte Carlo simulation.",
              "Simulate"),

        html.Div([
            html.Div([
                # Left: form panel
                _card([
                    _section_header("Procedure Parameters"),
                    build_form("d"),
                ], style={"width": "310px", "flexShrink": "0"}),

                # Right: results panel
                _card([
                    _section_header("Simulation Results"),
                    dcc.Loading(
                        id="designer-loading",
                        type="circle",
                        color=COL_BLUE,
                        overlay_style={"visibility": "visible", "filter": "blur(1px)"},
                        custom_spinner=html.Div([
                            html.Div("⚙️", style={"fontSize": "32px", "marginBottom": "10px"}),
                            html.Div("Running simulation…",
                                     style={"fontWeight": "700", "color": COL_NAVY, "fontSize": "15px"}),
                            html.Div("Monte Carlo · Calibrating · Benchmarking",
                                     style={"fontSize": "11px", "color": COL_GREY, "marginTop": "4px"}),
                        ], style={"textAlign": "center", "padding": "50px 20px"}),
                        children=html.Div(id="designer-results",
                            children=html.Div([
                                html.Span("◎", className="empty-state-icon"),
                                html.Div("Configure parameters and run simulation",
                                         className="empty-state-title"),
                                html.Div("Set procedure parameters in the left panel "
                                         "and click ▶ Run Simulation to see results.",
                                         className="empty-state-sub"),
                            ], className="empty-state")),
                    ),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
        ], className="page-body"),
    ])


# ── Clientside callbacks: immediate "Computing…" feedback on button click ──
# These run in the browser with zero server round-trip, so feedback is instant.

app.clientside_callback(
    """
    function(n) {
        if (n && n > 0) {
            return "⏳  Computing simulation — please wait…";
        }
        return "";
    }
    """,
    Output("d-status",       "children"),
    Input("d-btn",           "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(n) {
        if (n && n > 0) {
            return "⏳  Running comparison — please wait…";
        }
        return "";
    }
    """,
    Output("compare-status", "children"),
    Input("compare-btn",     "n_clicks"),
    prevent_initial_call=True,
)

# ca-status and cb-status for the two scenario forms
app.clientside_callback(
    """
    function(n) { return ""; }
    """,
    Output("ca-status", "children"),
    Input("ca-btn",     "n_clicks"),
    prevent_initial_call=True,
)
app.clientside_callback(
    """
    function(n) { return ""; }
    """,
    Output("cb-status", "children"),
    Input("cb-btn",     "n_clicks"),
    prevent_initial_call=True,
)

# Procedure type description callback (Tab 1)
@app.callback(Output("d-proc-desc","children"), Input("d-proc","value"))
def update_proc_desc_d(proc):
    return PROC_DESCRIPTIONS.get(proc, "")


@app.callback(
    Output("designer-results","children"),
    Input("d-btn","n_clicks"),
    [State("d-country","value"), State("d-proc","value"),   State("d-ctype","value"),
     State("d-cpv","value"),     State("d-crit","value"),   State("d-val","value"),
     State("d-prep","value"),    State("d-dur","value"),    State("d-pw","value"),
     State("d-flags","value")],
    prevent_initial_call=True
)
def run_designer(n, country, proc, ctype, cpv, crit, val, prep, dur, pw, flags):
    params = form_to_params(country, proc, ctype, cpv, crit, val, prep, dur, pw, flags)
    result = twin.simulate(params)
    bench  = twin.empirical_benchmark(country=country, procedure_type=proc, cpv_division=cpv)

    comp_mean  = result["competition"]["mean"]
    sb_prob    = result["single_bid_risk"]["probability"]
    cb_prob    = result["cross_border"]["probability"]
    pr_mean    = result["price_ratio"]["mean"]
    dur_median = result["duration"]["median"]

    b_comp = bench["competition"]["median"] if bench["competition"].get("n",0) > 0 else None
    b_sb   = bench["single_bid_rate"]
    b_cb   = bench["cross_border"]
    b_pr   = bench["price_ratio"]["median"] if bench["price_ratio"].get("n",0) > 0 else None
    b_dur  = bench["duration"]["median"] if bench["duration"].get("n",0) > 0 else None

    def badge_vs_bench(sim_v, bench_v, low_good=False, unit="", fmt=".1f"):
        if bench_v is None: return None, None
        delta = sim_v - bench_v
        better = delta < 0 if low_good else delta > 0
        arrow = "▲" if delta > 0 else "▼"
        col = COL_GREEN if better else COL_RED
        return f"{arrow} {abs(delta):{fmt}}{unit} vs benchmark", col

    b1, c1 = badge_vs_bench(comp_mean,    b_comp,  False,  " bids")
    b2, c2 = badge_vs_bench(sb_prob*100,  b_sb*100 if b_sb else None, True, "%", ".0f")
    b3, c3 = badge_vs_bench(cb_prob*100,  b_cb*100 if b_cb else None, False, "%", ".0f")
    b4, c4 = badge_vs_bench(pr_mean,      b_pr,    True,  "", ".3f")
    b5, c5 = badge_vs_bench(dur_median,   b_dur,   True,  "d", ".0f")

    # Short notes shown below each KPI value
    outcome_notes = {
        "Expected bids":    "avg offers for similar procedures",
        "P(single bid)":    "risk of only one offer",
        "P(cross-border)":  "P(winner from different EU country)",
        "Price ratio":      "award ÷ estimate  (>1 = over-budget)",
        "Duration":         "days: publication → award",
    }

    kpis = [
        ("Expected bids",    f"{comp_mean:.1f}",    b1, c1),
        ("P(single bid)",    f"{sb_prob*100:.0f}%",  b2, c2),
        ("P(cross-border)",  f"{cb_prob*100:.0f}%",  b3, c3),
        ("Price ratio",      f"{pr_mean:.3f}",       b4, c4),
        ("Duration",         f"{dur_median:.0f}d",   b5, c5),
    ]

    # Accent colour per KPI
    kpi_accents = [COL_BLUE, COL_RED, COL_TEAL, "#7C3AED", COL_GREEN]

    kpi_row = html.Div(
        [kpi_card(label, val, badge, col,
                  note=outcome_notes.get(label),
                  tooltip_text=OUTCOME_TOOLTIPS.get(label),
                  accent=kpi_accents[i])
         for i, (label, val, badge, col) in enumerate(kpis)],
        className="kpi-grid")

    # Distribution charts
    dist_row = html.Div([
        html.Div([dcc.Graph(
            figure=dist_chart(result["competition"]["samples"],
                              "Competition — offers received", COL_BLUE,
                              vline=comp_mean),
            config={"displayModeBar": False})], style={"flex": "1"}),
        html.Div([dcc.Graph(
            figure=dist_chart(result["price_ratio"]["samples"],
                              "Price ratio — award / estimate", "#7C3AED"),
            config={"displayModeBar": False})], style={"flex": "1"}),
        html.Div([dcc.Graph(
            figure=dist_chart(result["duration"]["samples"],
                              "Procedure duration — days", COL_GREEN,
                              vline=dur_median),
            config={"displayModeBar": False})], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "10px", "marginBottom": "14px"})

    # Gauge bars using new CSS classes
    def gauge_bar(prob, label, color):
        pct = round(prob * 100)
        bar_color = color if pct > 25 else COL_GREEN
        return html.Div([
            html.Div([
                html.Span(label, className="gauge-label"),
                html.Span(f"{pct}%", className="gauge-pct", style={"color": bar_color}),
            ], className="gauge-header"),
            html.Div(
                html.Div(className="gauge-fill", style={
                    "width": f"{pct}%", "backgroundColor": bar_color,
                }),
                className="gauge-track"
            ),
        ], className="gauge-wrapper")

    gauge_row = html.Div([
        html.Div([
            gauge_bar(sb_prob, "Single-bid risk", COL_RED),
            gauge_bar(cb_prob, "Cross-border win", COL_GREEN),
        ], style={"flex": "1"}),
        html.Div([
            html.P(
                f"Benchmark: {bench['n_records']:,} historical procedures "
                f"matching country={country}, procedure={proc}, CPV={cpv}",
                style={"fontSize": "11px", "color": COL_GREY, "fontStyle": "italic", "margin": "0"}),
            html.P("Simulation: 5,000 Monte Carlo samples · calibrated per CPV & country cluster",
                   style={"fontSize": "10px", "color": COL_GREY, "margin": "4px 0 0"}),
        ], style={"flex": "1", "display": "flex", "flexDirection": "column",
                  "justifyContent": "center", "paddingLeft": "20px"}),
    ], style={"display": "flex", "gap": "20px", "alignItems": "center"})

    return html.Div([kpi_row, dist_row, gauge_row])


# ══════════════════════════════════════════════════════════════════
# TAB 2: SCENARIO COMPARATOR
# ══════════════════════════════════════════════════════════════════
def comparator_layout():
    return html.Div([
        _phdr("Scenario Comparator",
              "Configure two procedure designs side-by-side and compare outcomes to find the better approach.",
              "Simulate"),

        html.Div([
            # Two scenario forms side by side
            html.Div([
                _card([
                    html.H3("Scenario A",
                            style={"color": COL_BLUE, "marginTop": "0",
                                   "borderBottom": f"3px solid {COL_BLUE}",
                                   "paddingBottom": "8px", "fontSize": "16px",
                                   "letterSpacing": "-0.3px"}),
                    build_form("ca", {"country": "DE", "proc": "OPE", "crit": "L",
                                      "val": 2_000_000, "cpv": "45"}),
                ], style={"flex": "1"}),
                _card([
                    html.H3("Scenario B",
                            style={"color": COL_ACCENT, "marginTop": "0",
                                   "borderBottom": f"3px solid {COL_ACCENT}",
                                   "paddingBottom": "8px", "fontSize": "16px",
                                   "letterSpacing": "-0.3px"}),
                    build_form("cb", {"country": "DE", "proc": "OPE", "crit": "M",
                                      "val": 2_000_000, "cpv": "45"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

            # Compare button
            html.Div([
                html.Button("⚖️  Compare Scenarios", id="compare-btn", n_clicks=0,
                            className="btn-compare"),
                html.Div(id="compare-status", style={
                    "fontSize": "12px", "color": COL_BLUE, "marginTop": "8px",
                    "minHeight": "16px", "fontStyle": "italic", "textAlign": "center",
                }),
            ], style={"textAlign": "center", "marginBottom": "16px"}),

            dcc.Loading(
                id="compare-loading",
                type="circle",
                color=COL_BLUE,
                overlay_style={"visibility": "visible", "filter": "blur(1px)"},
                custom_spinner=html.Div([
                    html.Div("⚙️", style={"fontSize": "32px", "marginBottom": "8px"}),
                    html.Div("Comparing scenarios…",
                             style={"fontWeight": "700", "color": COL_NAVY, "fontSize": "15px"}),
                    html.Div("Simulating A · Simulating B · Computing deltas",
                             style={"fontSize": "11px", "color": COL_GREY, "marginTop": "4px"}),
                ], style={"textAlign": "center", "padding": "50px 20px"}),
                children=html.Div(id="compare-results"),
            ),
        ], className="page-body"),
    ])


@app.callback(
    Output("ca-proc-desc","children"), Input("ca-proc","value"))
def upd_desc_ca(p): return PROC_DESCRIPTIONS.get(p,"")

@app.callback(
    Output("cb-proc-desc","children"), Input("cb-proc","value"))
def upd_desc_cb(p): return PROC_DESCRIPTIONS.get(p,"")


@app.callback(
    Output("compare-results","children"),
    Input("compare-btn","n_clicks"),
    [State("ca-country","value"), State("ca-proc","value"),  State("ca-ctype","value"),
     State("ca-cpv","value"),     State("ca-crit","value"),  State("ca-val","value"),
     State("ca-prep","value"),    State("ca-dur","value"),   State("ca-pw","value"),
     State("ca-flags","value"),
     State("cb-country","value"), State("cb-proc","value"),  State("cb-ctype","value"),
     State("cb-cpv","value"),     State("cb-crit","value"),  State("cb-val","value"),
     State("cb-prep","value"),    State("cb-dur","value"),   State("cb-pw","value"),
     State("cb-flags","value")],
    prevent_initial_call=True
)
def run_compare(n,
                a_co,a_pr,a_ct,a_cpv,a_crit,a_val,a_prep,a_dur,a_pw,a_fl,
                b_co,b_pr,b_ct,b_cpv,b_crit,b_val,b_prep,b_dur,b_pw,b_fl):
    pa = form_to_params(a_co,a_pr,a_ct,a_cpv,a_crit,a_val,a_prep,a_dur,a_pw,a_fl)
    pb = form_to_params(b_co,b_pr,b_ct,b_cpv,b_crit,b_val,b_prep,b_dur,b_pw,b_fl)
    comp = twin.compare(pa, pb, label_a="Scenario A", label_b="Scenario B")

    metrics = [
        ("Expected bids",     "competition",     "mean",        "",  ".2f", False),
        ("Single-bid risk",   "single_bid_risk", "probability", "%", ".1%", True),
        ("Cross-border win",  "cross_border",    "probability", "%", ".1%", False),
        ("Price ratio",       "price_ratio",     "mean",        "",  ".3f", True),
        ("Duration (days)",   "duration",        "mean",        "d", ".0f", True),
    ]

    rows = []
    for label, key, subkey, unit, fmt, low_better in metrics:
        d      = comp["deltas"][key]
        # NOTE: do NOT multiply by 100 for "%" metrics — the ".1%" format
        # specifier already multiplies by 100 and appends the % sign.
        a_v    = d["a"]
        b_v    = d["b"]
        delta  = d["delta"]
        better = delta < 0 if low_better else delta > 0
        arrow  = "▲" if delta > 0 else "▼" if delta < 0 else "–"
        bg     = "#E2EFDA" if better else "#FCE4EC"
        col    = COL_GREEN if better else COL_RED

        rows.append(html.Tr([
            html.Td(label, style={"padding":"10px 14px","fontWeight":"600","fontSize":"13px"}),
            html.Td(f"{a_v:{fmt}}{unit if unit!='%' else ''}",
                    style={"padding":"10px 14px","textAlign":"center","color":COL_BLUE,"fontWeight":"600"}),
            html.Td(f"{b_v:{fmt}}{unit if unit!='%' else ''}",
                    style={"padding":"10px 14px","textAlign":"center","color":COL_ACCENT,"fontWeight":"600"}),
            html.Td(f"{arrow} {abs(delta):{fmt}}{unit if unit!='%' else ''}",
                    style={"padding":"10px 14px","textAlign":"center","fontWeight":"700",
                           "color":col,"backgroundColor":bg}),
        ]))

    def overlay(key, title):
        sa = comp["scenario_a"][key]["samples"]
        sb = comp["scenario_b"][key]["samples"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sa, nbinsx=35, name="A", marker_color=COL_BLUE, opacity=0.6))
        fig.add_trace(go.Histogram(x=sb, nbinsx=35, name="B", marker_color=COL_ACCENT, opacity=0.6))
        fig.update_layout(barmode="overlay", title=dict(text=title, font=dict(size=12)),
                          height=195, margin=dict(t=34,b=24,l=34,r=10),
                          paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD,
                          legend=dict(font=dict(size=10)),
                          xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"))
        return fig

    return _card([
        _section_header("Comparison Results"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Outcome",      style={"padding":"10px 14px","backgroundColor":COL_NAVY,"color":"white","textAlign":"left"}),
                html.Th("Scenario A",   style={"padding":"10px 14px","backgroundColor":COL_BLUE,"color":"white","textAlign":"center"}),
                html.Th("Scenario B",   style={"padding":"10px 14px","backgroundColor":COL_ACCENT,"color":"white","textAlign":"center"}),
                html.Th("Difference (B−A)", style={"padding":"10px 14px","backgroundColor":"#4A4A4A","color":"white","textAlign":"center"}),
            ])),
            html.Tbody(rows),
        ], style={"width":"100%","borderCollapse":"collapse","marginBottom":"20px",
                  "boxShadow":"0 1px 4px rgba(0,0,0,0.08)","borderRadius":"6px",
                  "overflow":"hidden"}),

        html.Div([
            html.Div([dcc.Graph(figure=overlay("competition","Competition (offers)"),
                                config={"displayModeBar":False})], style={"flex":"1"}),
            html.Div([dcc.Graph(figure=overlay("price_ratio","Price ratio"),
                                config={"displayModeBar":False})], style={"flex":"1"}),
            html.Div([dcc.Graph(figure=overlay("duration","Duration (days)"),
                                config={"displayModeBar":False})], style={"flex":"1"}),
        ], style={"display":"flex","gap":"10px"}),
    ])


# ══════════════════════════════════════════════════════════════════
# TAB 3: POLICY EXPLORER
# ══════════════════════════════════════════════════════════════════
def explorer_layout():
    feat_path = os.path.join(FEAT_DIR, "procedure_records.parquet")
    try:
        df = pl.read_parquet(feat_path).to_pandas()
        countries_avail = sorted(df["ISO_COUNTRY_CODE"].dropna().unique().tolist())
        proc_avail      = sorted(df["TOP_TYPE"].dropna().unique().tolist())
    except Exception:
        # Feature store not yet downloaded (e.g. first cold start on HF Spaces)
        return html.Div([
            html.Div([
                html.Div("⏳", style={"fontSize":"48px","marginBottom":"12px"}),
                html.H3("Feature store loading…",
                        style={"color":COL_NAVY,"marginBottom":"8px"}),
                html.P("The historical data file is being downloaded from HF Hub. "
                       "This only happens on the first cold start and takes about 30 seconds. "
                       "Refresh the page in a moment.",
                       style={"color":COL_GREY,"fontSize":"13px","maxWidth":"400px",
                              "margin":"0 auto","lineHeight":"1.6"}),
            ], style={"textAlign":"center","paddingTop":"80px"}),
        ], style={"padding":"40px"})

    return html.Div([
        _phdr("Policy Explorer",
              "Explore empirical distributions from 1.1M TED contract records. Filter by country, sector, and year.",
              "Discover"),
        html.Div([
            _card([
                _section_header("Filter historical data"),
                html.Label("Country", className="form-label"),
                dcc.Dropdown(id="ex-country", multi=True,
                    options=[{"label":c,"value":c} for c in countries_avail],
                    value=["DE","FR","PL"], style={"marginBottom":"12px"}),
                html.Label("Procedure type", className="form-label"),
                dcc.Dropdown(id="ex-proc", multi=True,
                    options=[{"label":p,"value":p} for p in proc_avail],
                    value=["OPE"], style={"marginBottom":"12px"}),
                html.Label("CPV sector", className="form-label"),
                dcc.Dropdown(id="ex-cpv", multi=True,
                    options=CPV_OPTIONS, value=[], style={"marginBottom":"12px"}),
                html.Label("Years", className="form-label"),
                dcc.RangeSlider(id="ex-years", min=2018, max=2023, step=1,
                    value=[2018,2023], marks={y:str(y) for y in range(2018,2024)},
                    tooltip={"placement":"bottom"}),
                html.Div(style={"height":"14px"}),
                html.Label("Outcome", className="form-label"),
                dcc.RadioItems(id="ex-outcome",
                    options=[
                        {"label": "Competition (offers)",    "value": "n_offers"},
                        {"label": "Single-bid rate",         "value": "single_bid_flag"},
                        {"label": "Cross-border win rate",   "value": "cross_border_win"},
                        {"label": "Price ratio",             "value": "price_ratio"},
                        {"label": "Duration (days)",         "value": "proc_duration_days"},
                    ],
                    value="n_offers", className="radio-block",
                    style={"marginTop":"6px"}),
                html.Button("🔍  Explore", id="ex-btn", n_clicks=0,
                            style={"marginTop":"16px","width":"100%",
                                   "padding":"10px","backgroundColor":COL_NAVY,
                                   "color":"white","border":"none",
                                   "borderRadius":"6px","cursor":"pointer",
                                   "fontWeight":"600","fontSize":"14px"}),
            ], style={"width":"280px","flexShrink":"0"}),

            _card([
                html.Div(id="explorer-results",
                         children=html.Div([
                             html.Div("🔍", style={"fontSize":"40px"}),
                             html.P("Apply filters and click Explore.",
                                    style={"color":COL_GREY,"fontSize":"14px"}),
                         ], style={"textAlign":"center","paddingTop":"60px"})),
            ], style={"flex":"1","marginLeft":"16px"}),
        ], style={"display":"flex","padding":"20px","alignItems":"flex-start"}),
    ])


@app.callback(
    Output("explorer-results","children"),
    Input("ex-btn","n_clicks"),
    [State("ex-country","value"), State("ex-proc","value"),
     State("ex-cpv","value"),     State("ex-years","value"),
     State("ex-outcome","value")],
    prevent_initial_call=True
)
def run_explorer(n, countries, procs, cpvs, years, outcome):
    feat_path = os.path.join(FEAT_DIR, "procedure_records.parquet")
    try:
        df = pl.read_parquet(feat_path).to_pandas()
    except Exception:
        return html.Div([
            html.P("⏳  Feature store not yet available — please wait a moment and try again.",
                   style={"color":COL_ORANGE,"fontWeight":"600","padding":"20px"}),
            html.P("The data file downloads automatically on first startup. "
                   "This takes about 30 seconds.",
                   style={"color":COL_GREY,"padding":"0 20px","fontSize":"12px"}),
        ])

    if countries: df = df[df["ISO_COUNTRY_CODE"].isin(countries)]
    if procs:     df = df[df["TOP_TYPE"].isin(procs)]
    if cpvs:      df = df[df["cpv_division"].isin(cpvs)]
    df = df[df["YEAR"].between(years[0], years[1])]
    df = df[df[outcome].notna()]

    n_records = len(df)
    if n_records == 0:
        return html.P("No records match these filters.", style={"color":COL_RED})

    outcome_labels = {
        "n_offers":           "Offers received",
        "single_bid_flag":    "Single-bid flag (0/1)",
        "cross_border_win":   "Cross-border win (0/1)",
        "price_ratio":        "Price ratio (award / estimate)",
        "proc_duration_days": "Procedure duration (days)",
    }
    label = outcome_labels.get(outcome, outcome)
    s = df[outcome]

    # Stats row
    stats_row = html.Div([
        kpi_card("Median", f"{s.median():.2f}"),
        kpi_card("Mean",   f"{s.mean():.2f}"),
        kpi_card("P25",    f"{s.quantile(0.25):.2f}"),
        kpi_card("P75",    f"{s.quantile(0.75):.2f}"),
        kpi_card("Records",f"{n_records:,}"),
    ], style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)","gap":"10px","marginBottom":"16px"})

    # Charts
    fig_hist = px.histogram(df, x=outcome, nbins=50,
                            color_discrete_sequence=[COL_BLUE],
                            title=f"{label} — distribution  (n={n_records:,})")
    fig_hist.update_layout(height=250, margin=dict(t=42,b=28,l=36,r=10),
                           paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD,
                           bargap=0.04)

    by_country = (df.groupby("ISO_COUNTRY_CODE")[outcome].median()
                    .reset_index().rename(columns={outcome:"med"})
                    .sort_values("med", ascending=False).head(20))
    fig_country = px.bar(by_country, x="ISO_COUNTRY_CODE", y="med",
                          color="med", color_continuous_scale="Blues",
                          title=f"Median {label} by country",
                          labels={"ISO_COUNTRY_CODE":"","med":f"Median"})
    fig_country.update_layout(height=260, margin=dict(t=42,b=28,l=36,r=10),
                              paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD,
                              coloraxis_showscale=False)

    by_year = (df.groupby("YEAR")[outcome].median()
                 .reset_index().rename(columns={outcome:"med"}))
    fig_year = px.line(by_year, x="YEAR", y="med", markers=True,
                        color_discrete_sequence=[COL_BLUE],
                        title=f"Trend: {label} by year",
                        labels={"YEAR":"Year","med":"Median"})
    fig_year.update_layout(height=240, margin=dict(t=42,b=28,l=36,r=10),
                           paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD)

    # CPV breakdown
    by_cpv = (df.groupby("cpv_division")[outcome].agg(["median","count"])
                .reset_index().rename(columns={"median":"med","count":"n"})
                .query("n >= 50").sort_values("med", ascending=False).head(15))
    by_cpv["sector"] = by_cpv["cpv_division"].map(CPV_SECTORS).fillna(by_cpv["cpv_division"])
    fig_cpv = px.bar(by_cpv, x="sector", y="med",
                     color="med", color_continuous_scale="Teal",
                     title=f"Median {label} by CPV sector (top 15, min 50 records)",
                     labels={"sector":"CPV sector","med":"Median"})
    fig_cpv.update_layout(height=260, margin=dict(t=42,b=80,l=36,r=10),
                          paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD,
                          coloraxis_showscale=False,
                          xaxis=dict(tickangle=30, tickfont=dict(size=10)))

    return html.Div([
        _section_header(f"Results: {label}"),
        stats_row,
        dcc.Graph(figure=fig_hist, config={"displayModeBar":False},
                  style={"marginBottom":"10px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_country, config={"displayModeBar":False})],
                     style={"flex":"1"}),
            html.Div([dcc.Graph(figure=fig_year,    config={"displayModeBar":False})],
                     style={"flex":"1"}),
        ], style={"display":"flex","gap":"10px","marginBottom":"10px"}),
        dcc.Graph(figure=fig_cpv, config={"displayModeBar":False}),
    ])


# ══════════════════════════════════════════════════════════════════
# TAB 4: POLICY SIMULATION
# Aggregate counterfactual: "What if we changed X across all procedures
# in this segment?"
# ══════════════════════════════════════════════════════════════════
def policy_layout():
    return html.Div([
        _phdr("Policy Simulation",
              "Model the counterfactual impact of a policy intervention across a market segment.",
              "Impact"),
        html.Div([
            _card([
                _section_header("Policy Intervention"),
                html.P("Select a target segment and a policy change, then simulate "
                       "the aggregate impact across matching historical procedures.",
                       style={"fontSize":"12px","color":COL_GREY,"marginTop":"-8px",
                              "marginBottom":"14px","lineHeight":"1.5"}),

                html.Label("Country cluster", className="form-label"),
                dcc.Dropdown(id="pol-cluster", options=[{"label":"All","value":""}]+CLUSTER_OPTIONS,
                    value="CEE", style={"marginBottom":"12px"}),

                html.Label("CPV sector", className="form-label"),
                dcc.Dropdown(id="pol-cpv",
                    options=[{"label":"All sectors","value":""}]+CPV_OPTIONS,
                    value="45", style={"marginBottom":"12px"}),

                html.Label("Procedure type", className="form-label"),
                dcc.Dropdown(id="pol-proc",
                    options=[{"label":"All","value":""}]+PROC_OPTIONS,
                    value="OPE", style={"marginBottom":"12px"}),

                html.Label("Year range", className="form-label"),
                dcc.RangeSlider(id="pol-years", min=2018, max=2023, step=1,
                    value=[2020,2022], marks={y:str(y) for y in range(2018,2024)},
                    tooltip={"placement":"bottom"}),
                html.Div(style={"height":"16px"}),

                html.Label("Policy intervention", className="form-label"),
                dcc.Dropdown(id="pol-type",
                    options=[
                        {"label":"📅  Extend preparation time",          "value":"prep_delta"},
                        {"label":"📋  Switch award criteria to MEAT",     "value":"crit_meat"},
                        {"label":"📋  Switch award criteria to lowest price","value":"crit_low"},
                        {"label":"🔓  Switch to open procedure",          "value":"proc_open"},
                        {"label":"🇪🇺  Add EU funds flag",                 "value":"eu_funds"},
                        {"label":"💻  Add electronic auction",            "value":"ea"},
                    ],
                    value="prep_delta", clearable=False, style={"marginBottom":"12px"}),

                html.Div(id="pol-magnitude-row", children=[
                    html.Label("Additional preparation days", className="form-label"),
                    dcc.Slider(id="pol-magnitude", min=7, max=42, step=7,
                               value=14, marks={7:"7d",14:"14d",21:"21d",28:"28d",42:"42d"},
                               tooltip={"placement":"bottom","always_visible":True}),
                ], style={"marginBottom":"12px"}),

                html.Label("Sample size", className="form-label"),
                dcc.Dropdown(id="pol-sample",
                    options=[{"label":"200 records (fast)","value":200},
                              {"label":"500 records","value":500},
                              {"label":"1,000 records","value":1000}],
                    value=300, clearable=False, style={"marginBottom":"16px"}),

                html.Button("🏛️  Run Policy Simulation", id="pol-btn", n_clicks=0,
                            style={"width":"100%","padding":"11px","fontWeight":"600",
                                   "fontSize":"14px","backgroundColor":COL_TEAL,
                                   "color":"white","border":"none",
                                   "borderRadius":"6px","cursor":"pointer"}),
            ], style={"width":"310px","flexShrink":"0"}),

            _card([
                html.Div(id="policy-results",
                         children=html.Div([
                             html.Div("🏛️", style={"fontSize":"40px"}),
                             html.P("Configure a policy intervention and click Run.",
                                    style={"color":COL_GREY,"fontSize":"14px"}),
                             html.P("The simulation runs Monte Carlo scenarios across hundreds of "
                                    "matching historical procedures to estimate aggregate impact.",
                                    style={"color":COL_GREY,"fontSize":"12px","maxWidth":"380px",
                                           "margin":"0 auto","lineHeight":"1.5"}),
                         ], style={"textAlign":"center","paddingTop":"50px"})),
            ], style={"flex":"1","marginLeft":"16px"}),

        ], style={"display":"flex","padding":"20px","alignItems":"flex-start"}),
    ])


@app.callback(
    Output("pol-magnitude-row","style"),
    Input("pol-type","value")
)
def toggle_magnitude(pol_type):
    if pol_type == "prep_delta":
        return {"marginBottom":"12px"}
    return {"display":"none"}


@app.callback(
    Output("policy-results","children"),
    Input("pol-btn","n_clicks"),
    [State("pol-cluster","value"),   State("pol-cpv","value"),
     State("pol-proc","value"),      State("pol-years","value"),
     State("pol-type","value"),      State("pol-magnitude","value"),
     State("pol-sample","value")],
    prevent_initial_call=True
)
def run_policy(n, cluster, cpv, proc, years, pol_type, magnitude, sample_size):
    segment = {}
    if cluster: segment["country_cluster"] = cluster
    if cpv:     segment["cpv_division"]    = cpv
    if proc:    segment["TOP_TYPE"]        = proc
    segment["year_from"] = years[0]
    segment["year_to"]   = years[1]

    # Build intervention dict
    if pol_type == "prep_delta":
        intervention = {"param": "prep_time_days", "delta": float(magnitude)}
        intervention_label = f"+{magnitude} days preparation time"
    elif pol_type == "crit_meat":
        intervention = {"param": "criteria", "value": "M"}
        intervention_label = "Switch award criteria → MEAT"
    elif pol_type == "crit_low":
        intervention = {"param": "criteria", "value": "L"}
        intervention_label = "Switch award criteria → Lowest price"
    elif pol_type == "proc_open":
        intervention = {"param": "procedure_type", "value": "OPE"}
        intervention_label = "Switch procedure type → Open"
    elif pol_type == "eu_funds":
        intervention = {"param": "eu_funds", "value": True}
        intervention_label = "Add EU funds flag"
    elif pol_type == "ea":
        intervention = {"param": "electronic_auction", "value": True}
        intervention_label = "Add electronic auction"
    else:
        intervention = {"param": "prep_time_days", "delta": 14}
        intervention_label = "+14 days preparation time"

    ps = twin.policy_simulation(segment, intervention, n_records=int(sample_size))

    if "error" in ps:
        return html.Div([
            html.Div("⚠️  " + ps["error"],
                     style={"color":COL_RED,"padding":"20px","fontWeight":"600"}),
            html.P(f"Try broadening the segment filters.",
                   style={"color":COL_GREY,"padding":"0 20px"}),
        ])

    # Build summary cards
    outcomes_cfg = [
        ("competition",    "Expected bids",   False, ".2f", ""),
        ("single_bid_risk","Single-bid risk",  True,  ".1%", "%"),
        ("price_ratio",    "Price ratio",      True,  ".3f", ""),
        ("duration",       "Duration",         True,  ".0f", "d"),
    ]

    summary_cards = []
    for key, label, low_better, fmt, unit in outcomes_cfg:
        v = ps["outcomes"][key]
        delta = v["mean_delta"]
        pct   = v["pct_delta"]
        mul   = 100 if unit == "%" else 1
        better = delta < 0 if low_better else delta > 0
        col    = COL_GREEN if better else COL_RED
        arrow  = "▲" if delta > 0 else "▼"
        badge  = f"{arrow} {abs(delta * mul):{fmt}}{unit} ({pct:+.1f}%)"
        summary_cards.append(
            kpi_card(label,
                     f"{v['baseline_mean'] * mul:{fmt}}{unit} → {v['counterfactual_mean'] * mul:{fmt}}{unit}",
                     badge=badge, badge_col=col,
                     note="baseline → counterfactual"))

    card_row = html.Div(summary_cards,
        style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)",
               "gap":"10px","marginBottom":"20px"})

    # Delta distributions
    def delta_chart(key, label, color):
        deltas = ps["outcomes"][key]["delta_samples"]
        arr = np.array(deltas)
        fig = go.Figure()
        fig.add_vline(x=0, line_color=COL_GREY, line_dash="dot", line_width=1)
        fig.add_trace(go.Histogram(x=arr, nbinsx=30,
                                   marker_color=color, opacity=0.78,
                                   name=label))
        fig.update_layout(
            title=dict(text=f"Δ {label}", font=dict(size=12)),
            height=190, margin=dict(t=34,b=24,l=34,r=10),
            paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD, showlegend=False,
            xaxis=dict(gridcolor="#EEE", zeroline=True, zerolinecolor="#AAA"),
            yaxis=dict(gridcolor="#EEE"),
        )
        return fig

    delta_row = html.Div([
        html.Div([dcc.Graph(figure=delta_chart("competition","Bids",COL_BLUE),
                            config={"displayModeBar":False})], style={"flex":"1"}),
        html.Div([dcc.Graph(figure=delta_chart("single_bid_risk","Single-bid risk",COL_RED),
                            config={"displayModeBar":False})], style={"flex":"1"}),
        html.Div([dcc.Graph(figure=delta_chart("price_ratio","Price ratio",COL_ACCENT),
                            config={"displayModeBar":False})], style={"flex":"1"}),
        html.Div([dcc.Graph(figure=delta_chart("duration","Duration",COL_GREEN),
                            config={"displayModeBar":False})], style={"flex":"1"}),
    ], style={"display":"flex","gap":"10px","marginBottom":"14px"})

    # Summary text
    filters_desc = " | ".join([f"{k}={v}" for k,v in segment.items()])

    return html.Div([
        html.Div([
            html.Div([
                html.Span("Policy: ", style={"fontWeight":"700","color":COL_NAVY}),
                html.Span(intervention_label, style={"color":COL_TEAL,"fontWeight":"600"}),
            ], style={"fontSize":"15px","marginBottom":"4px"}),
            html.Div([
                html.Span(f"Segment: {filters_desc}  ·  "),
                html.Span(f"{ps['n_matched']:,} matching procedures  ·  "
                          f"{ps['n_simulated']} simulated",
                          style={"fontWeight":"600"}),
            ], style={"fontSize":"12px","color":COL_GREY}),
        ], style={"backgroundColor":"#F0F8FF","padding":"12px 16px",
                  "borderRadius":"6px","borderLeft":f"4px solid {COL_TEAL}",
                  "marginBottom":"16px"}),
        card_row,
        html.H4("Distribution of per-procedure impact (Δ counterfactual − baseline)",
                style={"color":"#555","fontSize":"13px","margin":"0 0 8px 0",
                       "fontWeight":"600"}),
        delta_row,
        html.P("Each bar represents the simulated change for one historical procedure. "
               "Bars to the right of zero indicate an increase.",
               style={"fontSize":"11px","color":COL_GREY,"fontStyle":"italic","margin":"0"}),
    ])


# ══════════════════════════════════════════════════════════════════
# TAB 5: EXPLAIN — SHAP Feature Importance
# ══════════════════════════════════════════════════════════════════
def explain_layout():
    global_shap = twin.get_global_shap()

    # Pre-compute global importance charts if available
    global_charts = []
    if global_shap:
        for model_key, label in [("competition","Competition model"),
                                   ("single_bid","Single-bid risk model")]:
            if model_key not in global_shap:
                continue
            imp = global_shap[model_key]
            # Top 15 features
            sorted_imp = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            names = [x[0][:35] for x in sorted_imp]
            vals  = [abs(x[1]) for x in sorted_imp]
            fig = go.Figure(go.Bar(
                x=vals[::-1], y=names[::-1],
                orientation="h",
                marker=dict(color=vals[::-1], colorscale="Blues",
                            showscale=False),
            ))
            fig.update_layout(
                title=dict(text=f"Global Feature Importance — {label}",
                           font=dict(size=13,color=COL_NAVY)),
                height=380, margin=dict(t=45,b=20,l=160,r=20),
                paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD,
                xaxis=dict(title="Mean |SHAP value|",gridcolor="#EEE"),
                yaxis=dict(tickfont=dict(size=10)),
            )
            global_charts.append(
                html.Div([dcc.Graph(figure=fig, config={"displayModeBar":False})],
                         style={"flex":"1"}))

    global_section = html.Div(global_charts,
        style={"display":"flex","gap":"12px","marginBottom":"20px"}) if global_charts else \
        html.Div([
            html.P("Global SHAP importances not found. Re-run model training "
                   "(python src/models/03_train_models.py) with XGBoost installed "
                   "to generate them.",
                   style={"color":COL_GREY,"fontSize":"13px","fontStyle":"italic"}),
        ], style={"marginBottom":"16px"})

    return html.Div([
        _phdr("Explain Models",
              "Understand which procedure parameters drive each prediction using SHAP feature importance.",
              "Learn"),
        html.Div([
            _card([
                _section_header("Model Explainability"),

                # Global importances
                html.H4("Global Feature Importance",
                        style={"fontSize":"14px","color":COL_NAVY,
                               "marginBottom":"8px","fontWeight":"700"}),
                html.P("Mean absolute SHAP values across the training set — "
                       "higher bars indicate stronger influence on predictions.",
                       style={"fontSize":"12px","color":COL_GREY,"marginBottom":"12px"}),
                global_section,

                html.Hr(style={"borderColor":COL_LIGHT,"margin":"20px 0"}),

                # Single-procedure explanation
                html.H4("Per-Prediction Explanation",
                        style={"fontSize":"14px","color":COL_NAVY,
                               "marginBottom":"8px","fontWeight":"700"}),
                html.P("Enter procedure parameters below to see which features push "
                       "single-bid risk and competition above or below average.",
                       style={"fontSize":"12px","color":COL_GREY,"marginBottom":"14px"}),

                html.Div([
                    html.Div([
                        html.Label("Country",       className="form-label"),
                        dcc.Dropdown(id="ex2-country", options=[{"label":c,"value":c} for c in COUNTRIES],
                                     value="RO", clearable=False, style={"marginBottom":"10px"}),
                        html.Label("Procedure type",className="form-label"),
                        dcc.Dropdown(id="ex2-proc",  options=PROC_OPTIONS,
                                     value="OPE", clearable=False, style={"marginBottom":"10px"}),
                        html.Label("CPV sector",    className="form-label"),
                        dcc.Dropdown(id="ex2-cpv",   options=CPV_OPTIONS,
                                     value="45", clearable=False, style={"marginBottom":"10px"}),
                        html.Label("Criteria",      className="form-label"),
                        dcc.RadioItems(id="ex2-crit",options=CRITERIA_OPTIONS,
                                       value="L", className="radio-block",
                                       style={"marginBottom":"10px"}),
                    ], style={"flex":"1"}),
                    html.Div([
                        html.Label("Contract type", className="form-label"),
                        dcc.RadioItems(id="ex2-ctype",options=CONTRACT_OPTIONS,
                                       value="W", inline=True,
                                       className="radio-inline",
                                       style={"marginBottom":"10px"}),
                        html.Label("Value (€)", className="form-label"),
                        dcc.Input(id="ex2-val", type="number", value=500_000, min=10000,
                                  style={"width":"100%","padding":"6px","border":"1px solid #CCC",
                                         "borderRadius":"4px","marginBottom":"10px"}),
                        html.Label("Prep time (days)", className="form-label"),
                        dcc.Slider(id="ex2-prep", min=15, max=90, step=1, value=30,
                                   marks={15:"15",35:"35",52:"52",90:"90"},
                                   tooltip={"always_visible":True,"placement":"bottom"}),
                        html.Label("Duration (months)", className="form-label",
                                   style={"marginTop":"10px"}),
                        dcc.Slider(id="ex2-dur", min=3, max=60, step=3, value=12,
                                   marks={3:"3m",12:"1yr",24:"2yr",36:"3yr",60:"5yr"},
                                   tooltip={"always_visible":True,"placement":"bottom"}),
                    ], style={"flex":"1","marginLeft":"14px"}),
                ], style={"display":"flex","marginBottom":"14px"}),

                html.Button("💡  Explain This Procedure", id="ex2-btn", n_clicks=0,
                            style={"padding":"10px 28px","backgroundColor":COL_NAVY,
                                   "color":"white","border":"none","borderRadius":"6px",
                                   "cursor":"pointer","fontWeight":"600","fontSize":"14px"}),

                html.Div(id="explain-results", style={"marginTop":"16px"}),
            ]),
        ], style={"padding":"20px"}),
    ])


@app.callback(
    Output("explain-results","children"),
    Input("ex2-btn","n_clicks"),
    [State("ex2-country","value"), State("ex2-proc","value"),
     State("ex2-ctype","value"),   State("ex2-cpv","value"),
     State("ex2-crit","value"),    State("ex2-val","value"),
     State("ex2-prep","value"),    State("ex2-dur","value")],
    prevent_initial_call=True
)
def run_explain(n, country, proc, ctype, cpv, crit, val, prep, dur):
    params = {
        "country":          country,
        "procedure_type":   proc,
        "contract_type":    ctype,
        "cpv_division":     cpv,
        "criteria":         crit,
        "value_euro":       float(val) if val else 500_000,
        "prep_time_days":   float(prep) if prep else 35,
        "duration_months":  float(dur) if dur else 24,
        "price_weight_pct": 50,
    }

    shap_result = twin.compute_shap(params)

    if "error" in shap_result:
        return html.P(f"SHAP unavailable: {shap_result['error']}",
                      style={"color":COL_ORANGE})

    charts = []
    model_labels = {"competition": "Competition model  (n_offers)",
                    "single_bid":  "Single-bid risk model  (P(1 bid))"}

    for model_key in ["competition", "single_bid"]:
        if model_key not in shap_result or "error" in shap_result[model_key]:
            continue
        data = shap_result[model_key]
        sv   = data["shap_values"]

        # Top positive and negative contributors
        sorted_sv = sorted(sv.items(), key=lambda x: x[1], reverse=True)
        top_pos = [(k, v) for k, v in sorted_sv if v > 0][:8]
        top_neg = [(k, v) for k, v in sorted_sv if v < 0][-8:]
        top_items = top_pos + top_neg
        top_items.sort(key=lambda x: x[1])

        names = [x[0][:40] for x in top_items]
        vals  = [x[1] for x in top_items]
        colors = [COL_BLUE if v > 0 else COL_RED for v in vals]

        fig = go.Figure(go.Bar(
            x=vals, y=names,
            orientation="h",
            marker=dict(color=colors),
        ))
        fig.add_vline(x=0, line_color=COL_GREY, line_width=1)
        fig.update_layout(
            title=dict(text=f"SHAP contributions — {model_labels.get(model_key, model_key)}",
                       font=dict(size=13,color=COL_NAVY)),
            height=360, margin=dict(t=45,b=20,l=180,r=20),
            paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD,
            xaxis=dict(title="SHAP value (log-odds or log-count shift)",
                       gridcolor="#EEE", zeroline=True, zerolinecolor="#AAA"),
            yaxis=dict(tickfont=dict(size=10)),
            showlegend=False,
        )
        charts.append(
            html.Div([dcc.Graph(figure=fig, config={"displayModeBar":False})],
                     style={"flex":"1","marginBottom":"10px"}))

    if not charts:
        return html.P("No SHAP explanations available for the current models.",
                      style={"color":COL_GREY})

    # Natural language summary
    def nl_summary(model_key, direction="high"):
        if model_key not in shap_result or "error" in shap_result[model_key]:
            return ""
        sv = shap_result[model_key]["shap_values"]
        top_pos = sorted([(k,v) for k,v in sv.items() if v > 0], key=lambda x: -x[1])[:3]
        top_neg = sorted([(k,v) for k,v in sv.items() if v < 0], key=lambda x: x[1])[:3]
        parts = []
        if top_pos:
            parts.append("increases due to " + ", ".join([f"{k}" for k,_ in top_pos]))
        if top_neg:
            parts.append("decreases due to " + ", ".join([f"{k}" for k,_ in top_neg]))
        return "Prediction " + " and ".join(parts) + "." if parts else ""

    sb_summary = nl_summary("single_bid")
    comp_summary = nl_summary("competition")

    return html.Div([
        html.Div([
            html.Div("ℹ️  SHAP values show how each feature pushes the prediction above "
                     "(blue / right) or below (red / left) the model's average.",
                     style={"fontSize":"12px","color":COL_GREY,"fontStyle":"italic",
                            "marginBottom":"12px"}),
            *([html.P(comp_summary, style={"fontSize":"12px","color":"#444","marginBottom":"4px"})]
              if comp_summary else []),
            *([html.P(sb_summary, style={"fontSize":"12px","color":"#444"})]
              if sb_summary else []),
        ], style={"backgroundColor":"#F7FAFE","padding":"10px 14px",
                  "borderRadius":"6px","borderLeft":f"3px solid {COL_BLUE}",
                  "marginBottom":"14px"}),
        html.Div(charts, style={"display":"flex","gap":"12px","flexWrap":"wrap"}),
    ])


# ══════════════════════════════════════════════════════════════════
# TAB 7: ANALYSIS  —  sandboxed Python notebook
# ══════════════════════════════════════════════════════════════════

_ANALYSIS_EXAMPLES = {
    "Basic simulation": """\
result = twin.simulate({
    "country": "DE",
    "procedure_type": "OPE",
    "contract_type": "S",
    "cpv_division": "72",
    "criteria": "M",
    "value_euro": 500_000,
    "prep_time_days": 35,
})
print("Expected bids:", round(result["competition"]["mean"], 2))
print("Single-bid risk:", round(result["single_bid_risk"]["probability"], 3))
print("Price ratio:", round(result["price_ratio"]["mean"], 3))
""",

    "Compare two designs": """\
comp = twin.compare(
    {"country": "PL", "procedure_type": "OPE", "criteria": "L",
     "value_euro": 1_000_000, "prep_time_days": 35},
    {"country": "PL", "procedure_type": "OPE", "criteria": "M",
     "value_euro": 1_000_000, "prep_time_days": 35},
    label_a="Lowest price", label_b="MEAT",
)
for outcome, d in comp["deltas"].items():
    pct = f"{d['delta_pct']:+.1f}%" if d["delta_pct"] is not None else "n/a"
    print(f"{outcome:20s}  A={d['a']:.3f}  B={d['b']:.3f}  Δ={pct}")
""",

    "Sweep prep time": """\
# go, px, pd, np are pre-loaded — no imports needed
prep_days = list(range(15, 91, 5))
bids, single_bid = [], []

for days in prep_days:
    r = twin.simulate({"country": "FR", "procedure_type": "OPE",
                       "value_euro": 500_000, "prep_time_days": days}, n_samples=1000)
    bids.append(r["competition"]["mean"])
    single_bid.append(r["single_bid_risk"]["probability"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=prep_days, y=bids, name="Expected bids",
                         line=dict(color="#2E75B6")))
fig.add_trace(go.Scatter(x=prep_days, y=single_bid, name="P(single bid)",
                         line=dict(color="#C00000"), yaxis="y2"))
fig.update_layout(
    title="Effect of preparation time",
    xaxis_title="Prep time (days)",
    yaxis=dict(title="Expected bids"),
    yaxis2=dict(title="P(single bid)", overlaying="y", side="right"),
    legend=dict(x=0.7, y=0.95),
)
show(fig)
""",

    "Explore historical data": """\
# df is a pandas DataFrame of all procedure records
# px, go, pd, np are pre-loaded — no imports needed
print("Columns:", list(df.columns))
print("Shape:", df.shape)

# Competition by country cluster
agg = (df.groupby("country_cluster")["n_offers"]
         .agg(["mean", "median", "count"])
         .round(2)
         .sort_values("mean", ascending=False))
print(agg.to_string())

fig = px.box(df[df["n_offers"].between(1,20)], x="country_cluster", y="n_offers",
             title="Bid distribution by country cluster",
             color="country_cluster")
fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of bids")
show(fig)
""",

    "Policy simulation": """\
result = twin.policy_simulation(
    segment_filters={"country_cluster": "CEE", "year_from": 2020, "year_to": 2023},
    intervention={"param": "prep_time_days", "delta": 14},
    n_records=200,
    seed=42,
)
print("Segment matched:", result.get("n_simulated", "?"), "procedures")
for outcome, stats in result.get("outcomes", {}).items():
    mean_delta = stats.get("mean_delta", 0)
    print(f"  {outcome:20s}  mean Δ = {mean_delta:+.4f}")
""",

    "Model performance": """\
# model_eval contains test-set metrics for all 5 trained models
for name, metrics in model_eval.items():
    print(f"\\n{name.upper()}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
""",

    "Direct prediction": """\
# params_to_df converts a params dict into a model-ready DataFrame
# models["x"]["model"] is the raw sklearn Pipeline — call .predict() directly
row = params_to_df({
    "country": "DE", "procedure_type": "OPE",
    "contract_type": "S", "cpv_division": "72",
    "criteria": "M", "value_euro": 500_000,
    "prep_time_days": 35,
})
print("Feature columns:", list(row.columns))

# Competition model predicts log10(1 + n_offers)
comp_raw = models["competition"]["model"].predict(row)[0]
print(f"\\nCompetition model raw output (log scale): {comp_raw:.4f}")
print(f"Back-transformed expected bids: {10**comp_raw - 1:.2f}")

# Single-bid model predicts P(only 1 bid)
prob_single = models["single_bid"]["model"].predict_proba(row)[0][1]
print(f"\\nSingle-bid risk: {prob_single:.3f}")

# Cross-border model
prob_cb = models["cross_border"]["model"].predict_proba(row)[0][1]
print(f"Cross-border probability: {prob_cb:.3f}")
""",

    "Feature importances": """\
# shap_global contains pre-computed mean |SHAP| values for each model
# Plot the top 20 features driving competition predictions
shap = shap_global["competition"]
top = sorted(shap.items(), key=lambda x: -x[1])[:20]
features, scores = zip(*top)

fig = go.Figure(go.Bar(
    x=list(scores), y=list(features),
    orientation="h", marker_color="#2E75B6",
))
fig.update_layout(
    title="Top 20 features — competition model (mean |SHAP|)",
    yaxis=dict(autorange="reversed"),
    xaxis_title="Mean |SHAP| value",
    height=520,
)
show(fig)

# Also available: shap_global["single_bid"], ["cross_border"], ["price"], ["duration"]
print("Models with SHAP data:", list(shap_global.keys()))
""",
}

_EDITOR_STYLE = {
    "fontFamily": "monospace",
    "fontSize": "13px",
    "lineHeight": "1.5",
    "width": "100%",
    "height": "340px",
    "padding": "12px",
    "border": "1px solid #C8D4E0",
    "borderRadius": "6px",
    "backgroundColor": "#FAFBFC",
    "resize": "vertical",
    "outline": "none",
    "whiteSpace": "pre",
    "overflowX": "auto",
}


def analysis_layout():

    def _ref_row(code, desc):
        return html.Tr([
            html.Td(html.Code(code, style={"fontSize": "11.5px",
                                           "backgroundColor": "#EEF2F7",
                                           "padding": "2px 5px",
                                           "borderRadius": "3px",
                                           "whiteSpace": "nowrap"}),
                    style={"padding": "5px 12px 5px 0", "verticalAlign": "top",
                           "width": "42%"}),
            html.Td(desc, style={"padding": "5px 0", "fontSize": "12px",
                                 "color": "#444", "verticalAlign": "top"}),
        ])

    reference_panel = html.Details([
        html.Summary("📚  What's available in the sandbox  —  click to expand",
                     style={"fontSize": "13px", "fontWeight": "600",
                            "color": COL_NAVY, "cursor": "pointer",
                            "padding": "10px 0", "userSelect": "none"}),

        html.Div([
            # ── Simulation API ────────────────────────────────────────
            html.P("Simulation API", style={"fontWeight": "700", "color": COL_NAVY,
                                             "margin": "12px 0 4px",
                                             "fontSize": "12px",
                                             "textTransform": "uppercase",
                                             "letterSpacing": "0.5px"}),
            html.Table([
                _ref_row("twin.simulate(params, n_samples=5000)",
                         "Run Monte Carlo simulation → distributions for competition, price, duration, etc."),
                _ref_row("twin.compare(params_a, params_b)",
                         "Side-by-side comparison of two procedure designs → per-outcome deltas"),
                _ref_row("twin.empirical_benchmark(country=, cpv_division=, ...)",
                         "Historical statistics from 1.1M TED records matching your filters"),
                _ref_row("twin.policy_simulation(segment_filters, intervention)",
                         "Aggregate counterfactual: what if a policy change were applied across a segment?"),
                _ref_row("twin.compute_shap(params)",
                         "Per-prediction SHAP feature contributions for a single procedure"),
            ], style={"width": "100%", "borderCollapse": "collapse"}),

            # ── Raw models ────────────────────────────────────────────
            html.P("Raw sklearn Pipelines  (each is a dict: {\"model\": Pipeline, \"meta\": {...}})",
                   style={"fontWeight": "700", "color": COL_NAVY,
                          "margin": "14px 0 4px", "fontSize": "12px",
                          "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Table([
                _ref_row("models[\"competition\"][\"model\"].predict(row)",
                         "XGBoost — predicts log₁₀(1 + n_offers); back-transform: 10^x − 1"),
                _ref_row("models[\"single_bid\"][\"model\"].predict_proba(row)[:,1]",
                         "Random Forest — P(only 1 bid received)"),
                _ref_row("models[\"cross_border\"][\"model\"].predict_proba(row)[:,1]",
                         "Random Forest — P(winner from a different EU country)"),
                _ref_row("models[\"price\"][\"model\"].predict(row)",
                         "Ridge (IV) — price ratio (award value ÷ estimate); uses competition_hat"),
                _ref_row("models[\"duration\"][\"model\"].predict(row)",
                         "Gradient Boosting — procedure duration in days"),
                _ref_row("models[x][\"model\"][\"pre\"].get_feature_names_out()",
                         "Encoded feature names after preprocessing (e.g. 'cat__ISO_COUNTRY_CODE_DE')"),
            ], style={"width": "100%", "borderCollapse": "collapse"}),

            # ── Data ──────────────────────────────────────────────────
            html.P("Data & Metadata",
                   style={"fontWeight": "700", "color": COL_NAVY,
                          "margin": "14px 0 4px", "fontSize": "12px",
                          "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Table([
                _ref_row("df",
                         "pandas DataFrame — 1.1M TED procedures 2018–2023. "
                         "Key columns: ISO_COUNTRY_CODE, TOP_TYPE, TYPE_OF_CONTRACT, "
                         "cpv_division, CRIT_CODE, value_bracket, country_cluster, "
                         "log10_value, prep_time_days, n_offers, price_ratio, "
                         "cross_border_win, single_bid_flag, contract_duration_months"),
                _ref_row("params_to_df(params_dict)",
                         "Convert a params dict to a model-ready DataFrame "
                         "(same keys as twin.simulate: country, procedure_type, value_euro, …)"),
                _ref_row("feature_spec",
                         "dict — cat_features, num_features, num_features_price lists "
                         f"({', '.join(['ISO_COUNTRY_CODE','TOP_TYPE','cpv_division','…'])})"),
                _ref_row("model_eval",
                         "dict — test-set metrics per model (AUC, MAE, R², baseline comparisons)"),
                _ref_row("calibration",
                         "dict — by_cpv and by_cluster calibration offsets "
                         "for competition and price_ratio models"),
                _ref_row("shap_global",
                         "dict — pre-computed mean |SHAP| importances per model "
                         "(keys: competition, single_bid, cross_border, price, duration)"),
            ], style={"width": "100%", "borderCollapse": "collapse"}),

            # ── Libraries ─────────────────────────────────────────────
            html.P("Libraries & Output",
                   style={"fontWeight": "700", "color": COL_NAVY,
                          "margin": "14px 0 4px", "fontSize": "12px",
                          "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Table([
                _ref_row("pd, np, pl", "pandas, numpy, polars"),
                _ref_row("go, px", "plotly.graph_objects, plotly.express"),
                _ref_row("show(fig)", "Render a plotly Figure in the output panel"),
                _ref_row("print(...)", "Print text to the output panel"),
            ], style={"width": "100%", "borderCollapse": "collapse"}),

        ], style={"padding": "0 4px 8px"}),
    ], style={"backgroundColor": COL_CARD, "padding": "0 20px 12px",
              "borderRadius": "8px", "marginBottom": "12px",
              "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"})

    example_buttons = [
        html.Button(
            name,
            id={"type": "analysis-example-btn", "index": i},
            n_clicks=0,
            style={
                "margin": "0 6px 6px 0",
                "padding": "4px 11px",
                "fontSize": "11.5px",
                "border": f"1px solid {COL_BLUE}",
                "borderRadius": "4px",
                "backgroundColor": "white",
                "color": COL_BLUE,
                "cursor": "pointer",
            },
        )
        for i, name in enumerate(_ANALYSIS_EXAMPLES)
    ]

    editor_panel = html.Div([
        # ── Examples row ─────────────────────────────────────────────
        html.Div([
            html.Span("Load example: ",
                      style={"fontSize": "12px", "color": COL_GREY,
                             "marginRight": "4px", "whiteSpace": "nowrap"}),
            html.Div(example_buttons,
                     style={"display": "flex", "flexWrap": "wrap"}),
        ], style={"display": "flex", "alignItems": "flex-start",
                  "marginBottom": "10px", "flexWrap": "wrap", "gap": "4px"}),

        # ── Code editor with IDE-style header bar ─────────────────────
        html.Div([
            html.Div([
                html.Span("Python",
                          style={"color": "#A8C4E0", "fontSize": "12px",
                                 "fontWeight": "700", "letterSpacing": "0.5px"}),
                html.Span("print() for text  ·  show(fig) for charts",
                          style={"color": "#5A7A96", "fontSize": "11px"}),
            ], style={"backgroundColor": "#1F3864",
                      "padding": "7px 14px",
                      "borderRadius": "6px 6px 0 0",
                      "display": "flex",
                      "justifyContent": "space-between",
                      "alignItems": "center"}),
            dcc.Textarea(
                id="analysis-code",
                value=list(_ANALYSIS_EXAMPLES.values())[0],
                placeholder="# Write your Python code here…",
                style={
                    "fontFamily": "'Fira Code', 'Cascadia Code', 'Consolas', monospace",
                    "fontSize": "13px",
                    "lineHeight": "1.55",
                    "width": "100%",
                    "height": "400px",
                    "padding": "14px",
                    "border": "none",
                    "borderRadius": "0 0 6px 6px",
                    "backgroundColor": "#F7F9FC",
                    "resize": "vertical",
                    "outline": "none",
                    "whiteSpace": "pre",
                    "overflowX": "auto",
                    "boxSizing": "border-box",
                },
                spellCheck=False,
            ),
        ], style={"border": f"2px solid {COL_NAVY}",
                  "borderRadius": "6px",
                  "overflow": "hidden",
                  "marginBottom": "10px"}),

        # ── Run button ───────────────────────────────────────────────
        html.Div([
            html.Button(
                "▶  Run",
                id="analysis-run-btn",
                n_clicks=0,
                style={
                    "backgroundColor": COL_NAVY,
                    "color": "white",
                    "border": "none",
                    "borderRadius": "6px",
                    "padding": "9px 28px",
                    "fontSize": "14px",
                    "fontWeight": "700",
                    "cursor": "pointer",
                    "letterSpacing": "0.3px",
                },
            ),
            html.Span(id="analysis-status",
                      style={"marginLeft": "14px", "fontSize": "12px",
                             "color": COL_GREY}),
        ], style={"display": "flex", "alignItems": "center"}),

        dcc.Store(id="analysis-result-store"),
    ])

    output_panel = html.Div([
        html.Div("Output",
                 style={"fontSize": "11px", "fontWeight": "700",
                        "color": COL_GREY, "letterSpacing": "0.8px",
                        "textTransform": "uppercase", "marginBottom": "10px",
                        "paddingBottom": "6px",
                        "borderBottom": f"1px solid {COL_LIGHT}"}),
        dcc.Loading(
            html.Div(id="analysis-output", style={"minHeight": "360px"}),
            type="circle", color=COL_BLUE,
        ),
    ])

    return html.Div([
        _phdr("Analysis Sandbox",
              "Write Python against the procurement twin's models and 1.1M-row dataset. "
              "Results render inline. Execution sandboxed, 30s timeout.",
              "Develop"),

        # ── Reference guide (collapsible) ─────────────────────────────
        reference_panel,

        # ── Editor + Output ───────────────────────────────────────────
        html.Div([
            html.Div(editor_panel,
                     style={"flex": "0 0 47%", "padding": "0 18px 0 0",
                            "minWidth": 0}),
            html.Div(output_panel,
                     style={"flex": "1", "borderLeft": f"2px solid {COL_LIGHT}",
                            "paddingLeft": "18px", "minWidth": 0}),
        ], style={"display": "flex", "backgroundColor": COL_CARD,
                  "padding": "16px 20px", "borderRadius": "8px",
                  "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"}),
    ], style={"padding": "16px 24px"})


@app.callback(
    Output("analysis-code", "value"),
    Input({"type": "analysis-example-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_example(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate
    idx = triggered["index"]
    keys = list(_ANALYSIS_EXAMPLES.keys())
    if idx < 0 or idx >= len(keys):
        raise dash.exceptions.PreventUpdate
    return _ANALYSIS_EXAMPLES[keys[idx]]


@app.callback(
    Output("analysis-result-store", "data"),
    Output("analysis-status", "children"),
    Input("analysis-run-btn", "n_clicks"),
    State("analysis-code", "value"),
    prevent_initial_call=True,
)
def execute_code(n_clicks, code):
    if not code or not code.strip():
        return dash.no_update, "Nothing to run."
    result = _sandbox_run(code, twin, _get_sandbox_df(), _SANDBOX_MODELS, _SANDBOX_META)
    status = f"Ran in {result['elapsed_ms']} ms"
    if result["error"]:
        status += " · error"
    return result, status


@app.callback(
    Output("analysis-output", "children"),
    Input("analysis-result-store", "data"),
    prevent_initial_call=True,
)
def render_output(result):
    if not result:
        return html.Div()

    parts = []

    # Error panel
    if result.get("error"):
        parts.append(html.Div([
            html.Strong("Error", style={"color": COL_RED}),
            html.Pre(result["error"],
                     style={"margin": "6px 0 0", "fontSize": "12px",
                            "whiteSpace": "pre-wrap", "color": "#8B0000"}),
        ], style={"backgroundColor": "#FFF0F0", "border": f"1px solid {COL_RED}",
                  "borderRadius": "6px", "padding": "10px 14px",
                  "marginBottom": "12px"}))

    # Stdout panel
    stdout = result.get("stdout", "")
    if stdout:
        parts.append(html.Div([
            html.Div("Output", style={"fontSize": "11px", "fontWeight": "600",
                                       "color": COL_GREY, "marginBottom": "4px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
            html.Pre(stdout,
                     style={"margin": 0, "fontSize": "12px",
                            "whiteSpace": "pre-wrap", "color": "#2C3E50",
                            "maxHeight": "300px", "overflowY": "auto"}),
        ], style={"backgroundColor": "#F6F8FA", "border": "1px solid #D0DAE6",
                  "borderRadius": "6px", "padding": "10px 14px",
                  "marginBottom": "12px"}))

    # Plotly figures
    for i, fig_dict in enumerate(result.get("figures", [])):
        parts.append(dcc.Graph(
            figure=fig_dict,
            config={"displayModeBar": True, "scrollZoom": False},
            style={"marginBottom": "12px"},
            id=f"analysis-fig-{i}",
        ))

    if not parts:
        parts.append(html.Div(
            "No output. Use print() or show(fig) to display results.",
            style={"color": COL_GREY, "fontSize": "13px",
                   "fontStyle": "italic", "paddingTop": "20px"},
        ))

    return parts


# ══════════════════════════════════════════════════════════════════
# TAB 8: OPTIMISATION LAB
# ══════════════════════════════════════════════════════════════════
def optimise_layout():
    def _weight_slider(id_suffix, label, default, description):
        return html.Div([
            html.Div([
                html.Span(label, style={"fontWeight":"600","fontSize":"13px","color":COL_NAVY}),
                html.Span(description, style={"fontSize":"11px","color":COL_GREY,"marginLeft":"8px"}),
            ], style={"marginBottom":"4px"}),
            dcc.Slider(id=f"opt-w-{id_suffix}", min=-1, max=1, step=0.1,
                       value=default,
                       marks={-1:"−1",0:"0",1:"+1"},
                       tooltip={"placement":"bottom","always_visible":True}),
        ], style={"marginBottom":"16px"})

    left = html.Div([
        html.H3("Context Parameters", style={"color":COL_NAVY,"fontSize":"14px","fontWeight":"700","margin":"0 0 12px 0","borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"6px"}),
        _form_row("Country", dcc.Dropdown(id="opt-country", options=[{"label":c,"value":c} for c in COUNTRIES], value="DE", clearable=False)),
        _form_row("Contract type", dcc.RadioItems(id="opt-ctype", options=CONTRACT_OPTIONS, value="S", inline=True, className="radio-inline")),
        _form_row("CPV sector", dcc.Dropdown(id="opt-cpv", options=CPV_OPTIONS, value="72", clearable=False)),
        _form_row("Estimated value (€)", dcc.Input(id="opt-val", type="number", value=1_000_000, min=10_000, step=10_000, style={"width":"100%","padding":"6px","border":"1px solid #CCC","borderRadius":"4px","fontSize":"13px"})),
        _form_row("Contract duration (months)", dcc.Slider(id="opt-dur", min=3, max=60, step=3, value=24, marks={3:"3m",12:"1yr",24:"2yr",36:"3yr",60:"5yr"}, tooltip={"placement":"bottom","always_visible":True})),

        html.H3("Objective Weights", style={"color":COL_NAVY,"fontSize":"14px","fontWeight":"700","margin":"18px 0 12px 0","borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"6px"}),
        html.P("Set weights: +1 = maximise, −1 = minimise, 0 = ignore.", style={"fontSize":"11px","color":COL_GREY,"margin":"0 0 12px 0"}),
        _weight_slider("competition", "Competition (bids)", 0.4, "want more bids"),
        _weight_slider("singlebid", "Single-bid risk", -0.4, "want lower risk"),
        _weight_slider("crossborder", "Cross-border", 0.1, "want higher participation"),
        _weight_slider("price", "Price ratio", -0.1, "want cheaper awards"),
        _weight_slider("duration", "Duration", -0.1, "want faster procedures"),

        html.H3("Constraints", style={"color":COL_NAVY,"fontSize":"14px","fontWeight":"700","margin":"18px 0 12px 0","borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"6px"}),
        _form_row("Allowed procedure types", dcc.Checklist(id="opt-procs",
            options=[{"label":f" {v}","value":k} for k,v in {"OPE":"Open","RES":"Restricted","NIC":"Negotiated","COD":"Competitive dialogue"}.items()],
            value=["OPE","RES","NIC","COD"], className="checklist")),
        _form_row("Min prep time (days)", dcc.Slider(id="opt-minprep", min=14, max=60, step=1, value=21, marks={14:"14",35:"35",52:"52"}, tooltip={"placement":"bottom","always_visible":True})),
        _form_row("Max prep time (days)", dcc.Slider(id="opt-maxprep", min=35, max=90, step=1, value=90, marks={35:"35",65:"65",90:"90"}, tooltip={"placement":"bottom","always_visible":True})),
        _form_row("MEAT only?", dcc.Checklist(id="opt-meatonly", options=[{"label":" Force MEAT award criteria","value":"meat"}], value=[])),

        html.Button("🏆  Find Optimal Design", id="opt-btn", n_clicks=0, className="btn-primary"),
        html.Div(id="opt-status", style={"fontSize":"12px","color":COL_BLUE,"marginTop":"10px","fontStyle":"italic"}),
    ], style={"width":"320px","flexShrink":"0","backgroundColor":COL_CARD,"borderRadius":"10px","padding":"20px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)","overflowY":"auto","maxHeight":"85vh"})

    right = html.Div([
        html.Div(id="opt-results", children=[
            html.Div([
                html.Div("🏆", style={"fontSize":"48px","marginBottom":"12px"}),
                html.H3("Multi-Objective Optimisation Engine", style={"color":COL_NAVY,"margin":"0 0 8px 0"}),
                html.P("Set your objective weights and click 'Find Optimal Design' to discover the best procedure configuration for your priorities.", style={"color":COL_GREY,"maxWidth":"400px","margin":"0 auto"}),
            ], style={"textAlign":"center","padding":"60px 20px","color":COL_GREY}),
        ]),
    ], style={"flex":"1","overflowY":"auto","maxHeight":"85vh","paddingLeft":"20px"})

    return html.Div([
        _phdr("Optimisation Lab",
              "Multi-objective search across the procedure design space. Maximize competition while minimising cost, risk, and duration.",
              "Optimize"),
        html.Div([left, right],
                 style={"display":"flex","gap":"0","padding":"20px","alignItems":"flex-start","minHeight":"80vh"}),
    ])


@app.callback(
    Output("opt-results","children"),
    Output("opt-status","children"),
    Input("opt-btn","n_clicks"),
    State("opt-country","value"), State("opt-ctype","value"),
    State("opt-cpv","value"), State("opt-val","value"), State("opt-dur","value"),
    State("opt-w-competition","value"), State("opt-w-singlebid","value"),
    State("opt-w-crossborder","value"), State("opt-w-price","value"), State("opt-w-duration","value"),
    State("opt-procs","value"), State("opt-minprep","value"), State("opt-maxprep","value"),
    State("opt-meatonly","value"),
    prevent_initial_call=True,
)
def run_optimise(n_clicks, country, ctype, cpv, val, dur,
                 w_comp, w_sb, w_cb, w_price, w_dur,
                 procs, minprep, maxprep, meatonly):
    base = {"country": country, "contract_type": ctype, "cpv_division": cpv,
            "value_euro": val or 1_000_000, "duration_months": dur or 24,
            "gpa": False, "eu_funds": False, "fra_agreement": False, "accelerated": False}
    weights = {"competition": float(w_comp or 0), "single_bid_risk": float(w_sb or 0),
               "cross_border": float(w_cb or 0), "price_ratio": float(w_price or 0),
               "duration": float(w_dur or 0)}
    constraints = {"allowed_procedure_types": procs or ["OPE","RES","NIC","COD"],
                   "min_prep_time": float(minprep or 21), "max_prep_time": float(maxprep or 90),
                   "must_use_meat": "meat" in (meatonly or [])}

    try:
        result = twin.optimize(base, weights, constraints, n_samples=400, seed=42)
    except Exception as e:
        return html.Div(f"Error: {e}", style={"color":COL_RED,"padding":"20px"}), f"Error: {e}"

    if "error" in result:
        return html.Div(result["error"], style={"color":COL_RED,"padding":"20px"}), result["error"]

    candidates = result.get("candidates", [])
    pareto = result.get("pareto_frontier", [])
    pareto_objs = result.get("pareto_objectives", [])
    best = result.get("best", {})
    n_eval = result.get("search_space", {}).get("n_candidates_evaluated", 0)

    # Best config card
    PROC_LABELS = {"OPE":"Open","RES":"Restricted","NIC":"Negotiated","COD":"Competitive dialogue","AWP":"Direct award","INP":"Innovation partnership"}
    best_card = html.Div([
        html.Div("🥇 Optimal Configuration", style={"fontWeight":"700","color":COL_NAVY,"fontSize":"15px","marginBottom":"10px"}),
        html.Div([
            html.Span(f"Utility score: {best.get('utility_score',0):.3f}", style={"fontWeight":"600","color":COL_BLUE,"fontSize":"13px","marginRight":"16px"}),
            html.Span(f"Procedure: {PROC_LABELS.get(best.get('procedure_type',''),'')}", style={"marginRight":"12px","fontSize":"13px"}),
            html.Span(f"Criteria: {'MEAT' if best.get('criteria')=='M' else 'Lowest price'}", style={"marginRight":"12px","fontSize":"13px"}),
            html.Span(f"Price weight: {best.get('price_weight_pct',0):.0f}%", style={"marginRight":"12px","fontSize":"13px"}),
            html.Span(f"Prep time: {best.get('prep_time_days',0):.0f} days", style={"marginRight":"12px","fontSize":"13px"}),
            html.Span(f"E-auction: {'Yes' if best.get('electronic_auction') else 'No'}", style={"fontSize":"13px"}),
        ], style={"marginBottom":"12px","flexWrap":"wrap","display":"flex","gap":"4px"}),
        html.Div([
            html.Div([html.Div(f"{best['outcomes']['competition']:.1f}", style={"fontSize":"22px","fontWeight":"700","color":COL_BLUE}), html.Div("Exp. Bids", style={"fontSize":"11px","color":COL_GREY})], style={"textAlign":"center","flex":"1"}),
            html.Div([html.Div(f"{best['outcomes']['single_bid_risk']:.0%}", style={"fontSize":"22px","fontWeight":"700","color":COL_RED}), html.Div("Single-bid risk", style={"fontSize":"11px","color":COL_GREY})], style={"textAlign":"center","flex":"1"}),
            html.Div([html.Div(f"{best['outcomes']['cross_border']:.0%}", style={"fontSize":"22px","fontWeight":"700","color":COL_TEAL}), html.Div("Cross-border", style={"fontSize":"11px","color":COL_GREY})], style={"textAlign":"center","flex":"1"}),
            html.Div([html.Div(f"{best['outcomes']['price_ratio']:.3f}", style={"fontSize":"22px","fontWeight":"700","color":COL_ORANGE}), html.Div("Price ratio", style={"fontSize":"11px","color":COL_GREY})], style={"textAlign":"center","flex":"1"}),
            html.Div([html.Div(f"{best['outcomes']['duration']:.0f}d", style={"fontSize":"22px","fontWeight":"700","color":COL_GREY}), html.Div("Duration", style={"fontSize":"11px","color":COL_GREY})], style={"textAlign":"center","flex":"1"}),
        ], style={"display":"flex","gap":"8px","backgroundColor":COL_BG,"borderRadius":"8px","padding":"14px"}),
    ], style={"backgroundColor":COL_CARD,"borderRadius":"10px","padding":"20px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)","marginBottom":"16px","border":f"2px solid {COL_BLUE}"})

    # Pareto frontier chart
    pareto_section = html.Div()
    if pareto and len(pareto_objs) == 2:
        obj1, obj2 = pareto_objs
        x_vals = [p[obj1] for p in pareto]
        y_vals = [p[obj2] for p in pareto]
        labels = [f"{PROC_LABELS.get(p.get('procedure_type',''),'')}, {p.get('prep_time_days',0):.0f}d prep" for p in pareto]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers+lines",
                                 marker=dict(size=10, color=COL_BLUE),
                                 text=labels, hovertemplate=f"{obj1}: %{{x:.3f}}<br>{obj2}: %{{y:.3f}}<br>%{{text}}<extra></extra>"))
        fig.update_layout(title=f"Pareto Frontier: {obj1} vs {obj2}",
                          xaxis_title=obj1.replace("_"," ").title(),
                          yaxis_title=obj2.replace("_"," ").title(),
                          height=280, margin=dict(l=40,r=20,t=40,b=40),
                          paper_bgcolor=COL_CARD, plot_bgcolor=COL_BG)
        pareto_section = html.Div([
            html.H4(f"Pareto Frontier — {obj1.replace('_',' ').title()} vs {obj2.replace('_',' ').title()}", style={"color":COL_NAVY,"fontSize":"13px","margin":"0 0 8px 0"}),
            dcc.Graph(figure=fig, config={"displayModeBar":False}),
        ], style={"backgroundColor":COL_CARD,"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)","marginBottom":"16px"})

    # Candidates table (top 10)
    tbl_rows = []
    for c in candidates[:10]:
        tbl_rows.append(html.Tr([
            html.Td(f"#{c['rank']}", style={"fontWeight":"700","color":COL_BLUE,"padding":"6px 10px","fontSize":"12px"}),
            html.Td(f"{c['utility_score']:.3f}", style={"padding":"6px 8px","fontSize":"12px","fontWeight":"600"}),
            html.Td(PROC_LABELS.get(c['procedure_type'],''), style={"padding":"6px 8px","fontSize":"12px"}),
            html.Td("MEAT" if c['criteria']=="M" else "Lowest price", style={"padding":"6px 8px","fontSize":"12px"}),
            html.Td(f"{c['price_weight_pct']:.0f}%", style={"padding":"6px 8px","fontSize":"12px"}),
            html.Td(f"{c['prep_time_days']:.0f}d", style={"padding":"6px 8px","fontSize":"12px"}),
            html.Td("Yes" if c['electronic_auction'] else "No", style={"padding":"6px 8px","fontSize":"12px"}),
            html.Td(f"{c['outcomes']['competition']:.1f}", style={"padding":"6px 8px","fontSize":"12px","color":COL_BLUE}),
            html.Td(f"{c['outcomes']['single_bid_risk']:.0%}", style={"padding":"6px 8px","fontSize":"12px","color":COL_RED}),
        ]))
    table = html.Div([
        html.H4("Top 10 Candidates", style={"color":COL_NAVY,"fontSize":"13px","margin":"0 0 8px 0"}),
        html.Table([
            html.Thead(html.Tr([html.Th(h, style={"padding":"6px 8px","fontSize":"11px","color":COL_GREY,"textAlign":"left","backgroundColor":COL_BG}) for h in ["Rank","Score","Procedure","Criteria","Price wt","Prep","E-auction","Bids","Single-bid"]])),
            html.Tbody(tbl_rows),
        ], style={"width":"100%","borderCollapse":"collapse","fontSize":"12px"}),
    ], style={"backgroundColor":COL_CARD,"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)","overflowX":"auto"})

    status_msg = f"Evaluated {n_eval} candidates — best utility score {best.get('utility_score',0):.3f}"
    return html.Div([best_card, pareto_section, table]), status_msg


# ══════════════════════════════════════════════════════════════════
# TAB 9: AI ADVISOR
# ══════════════════════════════════════════════════════════════════
def advisor_layout():
    llm_note = ("🤖 Mistral AI active (mistralai/Mistral-7B-Instruct-v0.3)"
                if _ADVISOR_AVAILABLE and _advisor and _advisor._hf_available
                else "⚙️ Rule-based advisor active (set AI_models Space secret for AI narrative)")

    left = html.Div([
        html.H3("Procedure Parameters", style={"color":COL_NAVY,"fontSize":"14px","fontWeight":"700","margin":"0 0 12px 0","borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"6px"}),
        # Form fields manually — avoids duplicate IDs with build_form
        _form_row("Country", dcc.Dropdown(id="adv-country", options=[{"label":c,"value":c} for c in COUNTRIES], value="DE", clearable=False)),
        _form_row("Procedure type", html.Div([
            dcc.Dropdown(id="adv-proc", options=PROC_OPTIONS, value="OPE", clearable=False),
            html.Div(id="adv-proc-desc", style={"fontSize":"11px","color":COL_GREY,"marginTop":"4px","fontStyle":"italic"}),
        ])),
        _form_row("Contract type", dcc.RadioItems(id="adv-ctype", options=CONTRACT_OPTIONS, value="S", inline=True, className="radio-inline")),
        _form_row("CPV sector", dcc.Dropdown(id="adv-cpv", options=CPV_OPTIONS, value="72", clearable=False)),
        _form_row("Award criteria", dcc.RadioItems(id="adv-crit", options=CRITERIA_OPTIONS, value="M", className="radio-block")),
        _form_row("Price weight (%)", html.Div([
            html.Div(id="adv-pw-val", style={"fontSize":"11px","color":COL_GREY,"textAlign":"right","marginBottom":"2px"}),
            dcc.Slider(id="adv-pw", min=0, max=100, step=5, value=60,
                       marks={0:"0%",50:"50%",100:"100%"},
                       tooltip={"placement":"bottom","always_visible":False}),
        ]), id_suffix="adv-pw-row"),
        _form_row("Estimated value (€)", dcc.Input(id="adv-val", type="number", value=1_000_000, min=10_000, step=10_000, style={"width":"100%","padding":"6px","border":"1px solid #CCC","borderRadius":"4px","fontSize":"13px"})),
        _form_row("Preparation time (days)", dcc.Slider(id="adv-prep", min=15, max=90, step=1, value=35, marks={15:"15",35:"35",52:"52",90:"90"}, tooltip={"placement":"bottom","always_visible":True})),
        _form_row("Contract duration (months)", dcc.Slider(id="adv-dur", min=3, max=60, step=3, value=24, marks={3:"3m",12:"1yr",24:"2yr",60:"5yr"}, tooltip={"placement":"bottom","always_visible":True})),
        _form_row("Options", dcc.Checklist(id="adv-flags",
            options=[{"label":" GPA covered","value":"gpa"},{"label":" EU funds","value":"eu_funds"},
                     {"label":" Electronic auction","value":"ea"},{"label":" Framework agreement","value":"fra"},
                     {"label":" Accelerated","value":"acc"}],
            value=["gpa"], className="checklist")),
        html.H3("Ask a Question (optional)", style={"color":COL_NAVY,"fontSize":"14px","fontWeight":"700","margin":"18px 0 8px 0","borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"6px"}),
        dcc.Textarea(id="adv-question", placeholder="e.g. How do I reduce single-bid risk in IT contracts?",
                     style={"width":"100%","height":"80px","padding":"8px","border":"1px solid #CCC","borderRadius":"6px","fontSize":"13px","resize":"vertical"}),
        html.P(llm_note, style={"fontSize":"11px","color":COL_GREY,"marginTop":"6px","fontStyle":"italic"}),
        html.Button("🧠  Get AI Advice", id="adv-btn", n_clicks=0, className="btn-primary"),
        html.Div(id="adv-status", style={"fontSize":"12px","color":COL_BLUE,"marginTop":"10px","fontStyle":"italic"}),
    ], style={"width":"320px","flexShrink":"0","backgroundColor":COL_CARD,"borderRadius":"10px","padding":"20px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)","overflowY":"auto","maxHeight":"85vh"})

    right = html.Div([
        html.Div(id="adv-output", children=[
            html.Div([
                html.Div("🧠", style={"fontSize":"48px","marginBottom":"12px"}),
                html.H3("AI Procurement Advisor", style={"color":COL_NAVY,"margin":"0 0 8px 0"}),
                html.P("Configure your procedure parameters and click 'Get AI Advice' to receive evidence-based recommendations.",
                       style={"color":COL_GREY,"maxWidth":"400px","margin":"0 auto"}),
            ], style={"textAlign":"center","padding":"60px 20px"}),
        ]),
    ], style={"flex":"1","overflowY":"auto","maxHeight":"85vh","paddingLeft":"20px"})

    return html.Div([left, right], style={"display":"flex","gap":"0","padding":"20px","alignItems":"flex-start","minHeight":"80vh"})


@app.callback(Output("adv-proc-desc","children"), Input("adv-proc","value"))
def adv_proc_desc(v): return PROC_DESCRIPTIONS.get(v,"")


@app.callback(Output("adv-pw-val","children"), Input("adv-pw","value"))
def adv_pw_val(v): return f"Price weight: {v}%"


@app.callback(
    Output("adv-pw-row","style"),
    Input("adv-crit","value"),
)
def adv_pw_visibility(crit):
    hidden = {"display":"none"}
    visible = {"marginBottom":"13px"}
    return visible if crit == "M" else hidden


@app.callback(
    Output("adv-output","children"),
    Output("adv-status","children"),
    Input("adv-btn","n_clicks"),
    State("adv-country","value"), State("adv-proc","value"), State("adv-ctype","value"),
    State("adv-cpv","value"), State("adv-crit","value"), State("adv-pw","value"),
    State("adv-val","value"), State("adv-prep","value"), State("adv-dur","value"),
    State("adv-flags","value"), State("adv-question","value"),
    prevent_initial_call=True,
)
def run_advisor(n_clicks, country, proc, ctype, cpv, crit, pw, val, prep, dur, flags, question):
    flags = flags or []
    params = {"country":country,"procedure_type":proc,"contract_type":ctype,
              "cpv_division":cpv,"criteria":crit,"price_weight_pct":float(pw or 50),
              "value_euro":float(val or 1_000_000),"prep_time_days":float(prep or 35),
              "duration_months":float(dur or 24),
              "gpa":"gpa" in flags,"eu_funds":"eu_funds" in flags,
              "fra_agreement":"fra" in flags,"electronic_auction":"ea" in flags,
              "accelerated":"acc" in flags}
    try:
        sim = twin.simulate(params, n_samples=2000, seed=42)
        shap = None
        try:
            shap = twin.compute_shap(params)
        except Exception:
            pass
        if not _ADVISOR_AVAILABLE or not _advisor:
            return html.Div("Advisor module unavailable.", style={"color":COL_RED,"padding":"20px"}), "Error"
        advice = _advisor.advise(params, sim, shap, question or None)
    except Exception as e:
        return html.Div(f"Error: {e}", style={"color":COL_RED,"padding":"20px"}), f"Error: {e}"

    SEV_COLOR = {"high":COL_RED,"medium":COL_ORANGE,"low":COL_GREEN}
    SEV_BG    = {"high":"#FFF0F0","medium":"#FFF8F0","low":"#F0FFF4"}

    # Summary card
    llm_badge = html.Span(" 🤖 AI-powered", style={"fontSize":"10px","backgroundColor":"#E8F0FF","color":COL_BLUE,"padding":"2px 7px","borderRadius":"3px","fontWeight":"700","marginLeft":"8px"}) if advice.get("llm_powered") else html.Span()
    summary_card = html.Div([
        html.Div([html.Span("📋 Executive Summary", style={"fontWeight":"700","color":COL_NAVY,"fontSize":"14px"}), llm_badge]),
        html.P(advice.get("summary",""), style={"fontSize":"13px","color":"#333","margin":"8px 0 0","lineHeight":"1.6"}),
    ], style={"backgroundColor":COL_CARD,"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)","marginBottom":"14px","borderLeft":f"4px solid {COL_BLUE}"})

    # LLM narrative
    llm_block = html.Div()
    if advice.get("llm_narrative"):
        llm_block = html.Div([
            html.Div("🤖 AI Narrative", style={"fontWeight":"700","color":COL_NAVY,"fontSize":"13px","marginBottom":"8px"}),
            html.Pre(advice["llm_narrative"], style={"fontSize":"12px","whiteSpace":"pre-wrap","color":"#333","fontFamily":"'Segoe UI',Arial,sans-serif","lineHeight":"1.65","margin":"0"}),
        ], style={"backgroundColor":"#F0F4FF","borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)","marginBottom":"14px","border":"1px solid #BDD0FF"})

    # Risks & strengths row
    risks = advice.get("key_risks",[])
    strs  = advice.get("strengths",[])
    risk_items  = [html.Div(f"• {r}", style={"fontSize":"12px","color":"#555","marginBottom":"3px"}) for r in risks] or [html.Div("No critical risks identified.", style={"fontSize":"12px","color":COL_GREY})]
    str_items   = [html.Div(f"• {s}", style={"fontSize":"12px","color":"#555","marginBottom":"3px"}) for s in strs] or [html.Div("No specific strengths flagged.", style={"fontSize":"12px","color":COL_GREY})]
    rs_row = html.Div([
        html.Div([html.Div("⚠️ Key Risks", style={"fontWeight":"700","color":COL_RED,"fontSize":"12px","marginBottom":"6px"}), *risk_items],
                 style={"flex":"1","backgroundColor":"#FFF8F8","borderRadius":"8px","padding":"12px","border":"1px solid #FFD0D0"}),
        html.Div([html.Div("✅ Strengths", style={"fontWeight":"700","color":COL_GREEN,"fontSize":"12px","marginBottom":"6px"}), *str_items],
                 style={"flex":"1","backgroundColor":"#F0FFF4","borderRadius":"8px","padding":"12px","border":"1px solid #B0EDB0"}),
    ], style={"display":"flex","gap":"12px","marginBottom":"14px"})

    # Recommendations
    rec_cards = []
    for r in advice.get("recommendations",[]):
        sev = r.get("severity","low")
        rec_cards.append(html.Div([
            html.Div([
                html.Span(r.get("issue",""), style={"fontWeight":"700","color":COL_NAVY,"fontSize":"13px"}),
                html.Span(sev.upper(), style={"fontSize":"9px","fontWeight":"700","color":SEV_COLOR.get(sev,COL_GREY),"backgroundColor":SEV_COLOR.get(sev,COL_GREY)+"22","padding":"2px 7px","borderRadius":"3px","marginLeft":"8px"}),
            ], style={"marginBottom":"6px"}),
            html.P(r.get("recommendation",""), style={"fontSize":"12px","color":"#444","margin":"0 0 6px 0","lineHeight":"1.6"}),
            html.Div(f"💡 {r.get('impact','')}", style={"fontSize":"11px","color":COL_BLUE,"fontStyle":"italic"}),
        ], style={"backgroundColor":SEV_BG.get(sev,"#FFF"),"borderRadius":"8px","padding":"14px","marginBottom":"10px","borderLeft":f"3px solid {SEV_COLOR.get(sev,COL_GREY)}"}))

    rec_children = rec_cards or [html.Div("No specific recommendations — design looks good!", style={"fontSize":"12px","color":COL_GREEN})]
    rec_section = html.Div([
        html.Div(f"📌 Recommendations ({len(rec_cards)})", style={"fontWeight":"700","color":COL_NAVY,"fontSize":"13px","marginBottom":"10px"}),
        *rec_children,
    ], style={"backgroundColor":COL_CARD,"borderRadius":"10px","padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)"})

    status = f"{len(rec_cards)} recommendation(s) · {'AI-powered' if advice.get('llm_powered') else 'Rule-based'}"
    return html.Div([summary_card, llm_block, rs_row, rec_section]), status


# ══════════════════════════════════════════════════════════════════
# TAB 10: RISK RADAR  (interactive v2)
# ══════════════════════════════════════════════════════════════════

# Module-level constants reused by layout + callbacks
_RADAR_CLUSTERS = ["CEE","Germanic","Western","Nordic","Mediterranean",
                   "Iberian","Baltic","Benelux","Anglophone","Balkan"]
_RADAR_CPVS     = ["45","72","33","48","71","60","79","85","39","31"]
_CPV_SHORT      = {"45":"Construction","72":"IT Services","33":"Medical","48":"Software",
                   "71":"Architecture","60":"Transport","79":"Business Svc","85":"Health",
                   "39":"Furniture","31":"Electrical"}
_CLUSTER_COUNTRY= {"CEE":"PL","Germanic":"DE","Western":"FR","Nordic":"SE",
                   "Mediterranean":"GR","Iberian":"ES","Baltic":"LT",
                   "Benelux":"BE","Anglophone":"IE","Balkan":"HR"}
_RADAR_METRICS  = {
    "single_bid_risk": {
        "label":"Single-bid risk","fmt":".0%",
        "cscale":[[0,"#D5E8F0"],[0.35,"#FFD700"],[0.65,"#FF8C00"],[1.0,"#C00000"]],
        "zmin":0,"zmax":0.6,"unit":"%","better":"lower",
        "extract": lambda s: s["single_bid_risk"]["probability"],
    },
    "competition": {
        "label":"Expected bids","fmt":".1f",
        "cscale":[[0,"#C00000"],[0.35,"#FF8C00"],[0.65,"#FFD700"],[1.0,"#217346"]],
        "zmin":1,"zmax":8,"unit":"bids","better":"higher",
        "extract": lambda s: s["competition"]["mean"],
    },
    "cross_border": {
        "label":"Cross-border win","fmt":".0%",
        "cscale":[[0,"#C00000"],[0.4,"#FFD700"],[1.0,"#217346"]],
        "zmin":0,"zmax":0.25,"unit":"%","better":"higher",
        "extract": lambda s: s["cross_border"]["probability"],
    },
    "price_ratio": {
        "label":"Price ratio","fmt":".2f",
        "cscale":[[0,"#217346"],[0.4,"#FFD700"],[0.7,"#FF8C00"],[1.0,"#C00000"]],
        "zmin":0.7,"zmax":1.3,"unit":"×","better":"lower",
        "extract": lambda s: s["price_ratio"]["mean"],
    },
    "duration": {
        "label":"Duration (days)","fmt":".0f",
        "cscale":[[0,"#D5E8F0"],[0.4,"#FFD700"],[0.7,"#FF8C00"],[1.0,"#C00000"]],
        "zmin":60,"zmax":320,"unit":"d","better":"lower",
        "extract": lambda s: s["duration"]["mean"],
    },
}


def radar_layout():
    metric_opts = [{"label": v["label"], "value": k} for k, v in _RADAR_METRICS.items()]

    left = html.Div([
        html.H3("Procedure Context", style={"color":COL_NAVY,"fontSize":"14px","fontWeight":"700","margin":"0 0 12px 0","borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"6px"}),
        _form_row("Procedure type", dcc.RadioItems(id="radar-proc",
            options=[{"label":" Open","value":"OPE"},{"label":" Restricted","value":"RES"},{"label":" Negotiated","value":"NIC"}],
            value="OPE", className="radio-block")),
        _form_row("Contract type", dcc.RadioItems(id="radar-ctype",
            options=CONTRACT_OPTIONS, value="S", inline=True, className="radio-inline")),
        _form_row("Award criteria", dcc.RadioItems(id="radar-crit",
            options=CRITERIA_OPTIONS, value="M", className="radio-block")),
        _form_row("Price weight (%)", dcc.Slider(id="radar-pw", min=0, max=100, step=10,
            value=50, marks={0:"0%",50:"50%",100:"100%"},
            tooltip={"placement":"bottom","always_visible":True})),
        _form_row("Preparation time (days)", dcc.Slider(id="radar-prep", min=15, max=90, step=1,
            value=35, marks={15:"15",35:"35",52:"52",90:"90"},
            tooltip={"placement":"bottom","always_visible":True})),
        _form_row("Contract duration (months)", dcc.Slider(id="radar-dur", min=3, max=60, step=3,
            value=24, marks={3:"3m",12:"1yr",24:"2yr",60:"5yr"},
            tooltip={"placement":"bottom","always_visible":True})),
        _form_row("Estimated value (€)", dcc.Dropdown(id="radar-val",
            options=[
                {"label":"< €135k",       "value":"50000"},
                {"label":"€135k – €215k", "value":"175000"},
                {"label":"€215k – €431k", "value":"323000"},
                {"label":"€431k – €5M",   "value":"500000"},
                {"label":"€5M – €50M",    "value":"10000000"},
                {"label":"> €50M",        "value":"75000000"},
            ], value="500000", clearable=False)),
        _form_row("Flags", dcc.Checklist(id="radar-flags",
            options=[{"label":" EU funds","value":"eu_funds"},
                     {"label":" GPA","value":"gpa"},
                     {"label":" E-auction","value":"ea"},
                     {"label":" Framework","value":"fra"}],
            value=[], className="checklist")),

        html.H3("Display", style={"color":COL_NAVY,"fontSize":"14px","fontWeight":"700","margin":"18px 0 12px 0","borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"6px"}),
        _form_row("Primary metric", dcc.Dropdown(id="radar-metric", options=metric_opts,
            value="single_bid_risk", clearable=False)),

        html.Button("🔄  Update Radar", id="radar-btn", n_clicks=0, className="btn-primary"),
        html.Div(id="radar-status", style={"fontSize":"12px","color":COL_BLUE,"marginTop":"10px","fontStyle":"italic"}),
    ], style={"width":"270px","flexShrink":"0","backgroundColor":COL_CARD,"borderRadius":"10px",
              "padding":"20px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)","overflowY":"auto","maxHeight":"90vh"})

    right = html.Div([
        # KPI row
        html.Div(id="radar-kpis", style={"display":"flex","gap":"12px","marginBottom":"16px"}),
        # Primary heatmap
        html.Div([dcc.Graph(id="radar-heatmap", config={"displayModeBar":False})],
                 style={"backgroundColor":COL_CARD,"borderRadius":"10px","padding":"16px",
                        "boxShadow":"0 2px 8px rgba(0,0,0,0.07)","marginBottom":"14px"}),
        # Scatter + Top-10 side by side
        html.Div([
            html.Div([dcc.Graph(id="radar-scatter", config={"displayModeBar":False})],
                     style={"flex":"1","backgroundColor":COL_CARD,"borderRadius":"10px",
                            "padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)"}),
            html.Div([dcc.Graph(id="radar-bar", config={"displayModeBar":False})],
                     style={"flex":"1","backgroundColor":COL_CARD,"borderRadius":"10px",
                            "padding":"16px","boxShadow":"0 2px 8px rgba(0,0,0,0.07)"}),
        ], style={"display":"flex","gap":"14px","marginBottom":"14px"}),
        # Drill-down panel (hidden until cell clicked)
        html.Div(id="radar-drilldown", style={"display":"none"}),
    ], style={"flex":"1","minWidth":"0","overflowY":"auto","maxHeight":"90vh","paddingLeft":"20px"})

    return html.Div([left, right],
                    style={"display":"flex","gap":"0","padding":"20px","alignItems":"flex-start"})


@app.callback(
    Output("radar-kpis",    "children"),
    Output("radar-heatmap", "figure"),
    Output("radar-scatter", "figure"),
    Output("radar-bar",     "figure"),
    Output("radar-status",  "children"),
    Input("radar-btn",  "n_clicks"),
    State("radar-proc", "value"), State("radar-ctype","value"),
    State("radar-crit", "value"), State("radar-pw",   "value"),
    State("radar-prep", "value"), State("radar-dur",  "value"),
    State("radar-val",  "value"), State("radar-flags","value"),
    State("radar-metric","value"),
)
def update_radar_charts(n_clicks, proc, ctype, crit, pw, prep, dur, val, flags, metric):
    flags  = flags or []
    val_f  = float(val or 500_000)
    mconf  = _RADAR_METRICS[metric]

    # Build base params template (country + CPV will be overridden per cell)
    base = {
        "procedure_type":    proc or "OPE",
        "contract_type":     ctype or "S",
        "criteria":          crit or "M",
        "price_weight_pct":  float(pw or 50),
        "prep_time_days":    float(prep or 35),
        "duration_months":   float(dur or 24),
        "value_euro":        val_f,
        "gpa":               "gpa"    in flags,
        "eu_funds":          "eu_funds" in flags,
        "fra_agreement":     "fra"    in flags,
        "electronic_auction":"ea"     in flags,
        "accelerated":       False,
    }

    # Simulate 10×10 grid
    n_cl  = len(_RADAR_CLUSTERS)
    n_cpv = len(_RADAR_CPVS)
    grids = {k: np.full((n_cl, n_cpv), float("nan")) for k in _RADAR_METRICS}

    for i, cluster in enumerate(_RADAR_CLUSTERS):
        country = _CLUSTER_COUNTRY.get(cluster, "DE")
        for j, cpv in enumerate(_RADAR_CPVS):
            try:
                sim = twin.simulate({**base, "country": country, "cpv_division": cpv},
                                    n_samples=200, seed=42)
                for k, mc in _RADAR_METRICS.items():
                    grids[k][i, j] = mc["extract"](sim)
            except Exception:
                pass

    primary = grids[metric]
    cpv_labels = [_CPV_SHORT.get(c, c) for c in _RADAR_CPVS]

    # ── KPI row ───────────────────────────────────────────────
    avg_sb     = float(np.nanmean(grids["single_bid_risk"]))
    high_risk  = int(np.sum(grids["single_bid_risk"] > 0.40))
    low_comp   = int(np.sum(grids["competition"] < 3.0))
    avg_comp   = float(np.nanmean(grids["competition"]))

    def _kpi(val_str, label, color):
        return html.Div([
            html.Div(val_str, style={"fontSize":"26px","fontWeight":"700","color":color}),
            html.Div(label,   style={"fontSize":"11px","color":COL_GREY}),
        ], style={"flex":"1","textAlign":"center","backgroundColor":COL_CARD,"borderRadius":"8px",
                  "padding":"12px","boxShadow":"0 2px 6px rgba(0,0,0,0.06)"})

    kpis = [
        _kpi(f"{avg_sb:.0%}",    "Avg single-bid risk", COL_RED),
        _kpi(str(high_risk),      "Segments >40% risk",  COL_ORANGE),
        _kpi(str(low_comp),       "Segments <3 bids",    COL_BLUE),
        _kpi(f"{avg_comp:.1f}",   "Avg expected bids",   COL_GREEN),
    ]

    # ── Primary heatmap ───────────────────────────────────────
    fmt = mconf["fmt"]
    if "%" in fmt:
        text_grid = [[f"{v:{fmt[1:]}}" if not np.isnan(v) else "N/A" for v in row]
                     for row in primary.tolist()]
    else:
        text_grid = [[f"{v:{fmt[1:]}}" if not np.isnan(v) else "N/A" for v in row]
                     for row in primary.tolist()]

    fig_heat = go.Figure(go.Heatmap(
        z=primary.tolist(),
        x=cpv_labels,
        y=_RADAR_CLUSTERS,
        colorscale=mconf["cscale"],
        zmin=mconf["zmin"], zmax=mconf["zmax"],
        text=text_grid,
        texttemplate="%{text}",
        textfont={"size":10},
        hovertemplate=f"Cluster: %{{y}}<br>CPV: %{{x}}<br>{mconf['label']}: %{{z:{fmt[1:]}}}<extra></extra>",
    ))
    fig_heat.update_layout(
        title=f"{mconf['label']} by Cluster × CPV Sector  ({proc}, {ctype}, {'MEAT' if crit=='M' else 'L-price'})",
        height=360, margin=dict(l=90,r=20,t=50,b=60),
        paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD,
        xaxis=dict(tickangle=-30),
    )

    # ── Scatter: competition vs single-bid risk ───────────────
    sc_x, sc_y, sc_c, sc_t = [], [], [], []
    for i, cluster in enumerate(_RADAR_CLUSTERS):
        for j, cpv in enumerate(_RADAR_CPVS):
            v_comp = grids["competition"][i, j]
            v_sb   = grids["single_bid_risk"][i, j]
            v_prim = primary[i, j]
            if not (np.isnan(v_comp) or np.isnan(v_sb)):
                sc_x.append(float(v_comp))
                sc_y.append(float(v_sb))
                sc_c.append(float(v_prim) if not np.isnan(v_prim) else 0)
                sc_t.append(f"{cluster}<br>{_CPV_SHORT.get(cpv, cpv)}")

    fig_sc = go.Figure(go.Scatter(
        x=sc_x, y=sc_y, mode="markers",
        marker=dict(size=11, color=sc_c, colorscale=mconf["cscale"],
                    cmin=mconf["zmin"], cmax=mconf["zmax"],
                    showscale=True, colorbar=dict(title=mconf["label"],thickness=12,len=0.8),
                    line=dict(width=0.5, color="white")),
        text=sc_t,
        hovertemplate="<b>%{text}</b><br>Competition: %{x:.1f} bids<br>Single-bid risk: %{y:.1%}<extra></extra>",
    ))
    # Quadrant lines
    fig_sc.add_hline(y=0.30, line_dash="dot", line_color=COL_ORANGE, line_width=1)
    fig_sc.add_vline(x=3.0,  line_dash="dot", line_color=COL_ORANGE, line_width=1)
    fig_sc.update_layout(
        title="All Segments: Competition vs Single-Bid Risk",
        xaxis=dict(title="Expected bids", range=[0, max(sc_x or [10]) + 1]),
        yaxis=dict(title="Single-bid risk", tickformat=".0%", range=[0, max(sc_y or [0.7]) + 0.05]),
        height=310, margin=dict(l=50,r=10,t=45,b=40),
        paper_bgcolor=COL_CARD, plot_bgcolor=COL_BG,
    )

    # ── Top-10 bar chart (ranked by primary metric) ───────────
    segs = []
    for i, cluster in enumerate(_RADAR_CLUSTERS):
        for j, cpv in enumerate(_RADAR_CPVS):
            v = primary[i, j]
            if not np.isnan(v):
                segs.append({"label": f"{cluster} / {_CPV_SHORT.get(cpv,cpv)}", "val": float(v)})
    better = mconf["better"]
    segs.sort(key=lambda s: s["val"] if better == "lower" else -s["val"])
    top10 = segs[:10]
    bar_colors = []
    for s in top10:
        if better == "lower":
            bar_colors.append(COL_RED if s["val"] > mconf["zmax"] * 0.7 else
                              COL_ORANGE if s["val"] > mconf["zmax"] * 0.45 else COL_GREY)
        else:
            bar_colors.append(COL_RED if s["val"] < mconf["zmin"] + (mconf["zmax"] - mconf["zmin"]) * 0.25 else
                              COL_ORANGE if s["val"] < mconf["zmin"] + (mconf["zmax"] - mconf["zmin"]) * 0.45 else COL_GREEN)

    title_dir = "Worst" if better == "lower" else "Best"
    fig_bar = go.Figure(go.Bar(
        x=[s["label"] for s in top10],
        y=[s["val"] for s in top10],
        marker_color=bar_colors,
        text=[f"{s['val']:{fmt[1:]}}" for s in top10],
        textposition="outside",
        hovertemplate=f"%{{x}}<br>{mconf['label']}: %{{y:{fmt[1:]}}}<extra></extra>",
    ))
    y_max = max((s["val"] for s in top10), default=1) * 1.25
    fig_bar.update_layout(
        title=f"Top 10 {title_dir} Segments — {mconf['label']}",
        yaxis=dict(title=mconf["label"], range=[0, y_max]),
        height=310, margin=dict(l=40,r=20,t=45,b=80),
        paper_bgcolor=COL_CARD, plot_bgcolor=COL_BG,
        xaxis=dict(tickangle=-35),
    )

    n_valid = int(np.sum(~np.isnan(primary)))
    status  = (f"Grid: {n_valid}/100 segments · "
               f"{proc}, {ctype}, {'MEAT' if crit=='M' else 'L-price'}, "
               f"€{val_f:,.0f}, {prep}d prep")
    return kpis, fig_heat, fig_sc, fig_bar, status


@app.callback(
    Output("radar-drilldown", "children"),
    Output("radar-drilldown", "style"),
    Input("radar-heatmap", "clickData"),
    State("radar-proc",  "value"), State("radar-ctype","value"),
    State("radar-crit",  "value"), State("radar-pw",   "value"),
    State("radar-prep",  "value"), State("radar-dur",  "value"),
    State("radar-val",   "value"), State("radar-flags","value"),
    prevent_initial_call=True,
)
def radar_drilldown(click, proc, ctype, crit, pw, prep, dur, val, flags):
    if not click:
        raise dash.exceptions.PreventUpdate

    pt      = click["points"][0]
    cpv_lbl = pt.get("x","")
    cluster = pt.get("y","")

    # Reverse-lookup CPV code from short label
    cpv = next((k for k, v in _CPV_SHORT.items() if v == cpv_lbl), "72")
    country = _CLUSTER_COUNTRY.get(cluster, "DE")
    flags   = flags or []

    params = {
        "country":           country,
        "procedure_type":    proc or "OPE",
        "contract_type":     ctype or "S",
        "cpv_division":      cpv,
        "criteria":          crit or "M",
        "price_weight_pct":  float(pw or 50),
        "value_euro":        float(val or 500_000),
        "prep_time_days":    float(prep or 35),
        "duration_months":   float(dur or 24),
        "gpa":               "gpa"     in flags,
        "eu_funds":          "eu_funds" in flags,
        "fra_agreement":     "fra"     in flags,
        "electronic_auction":"ea"      in flags,
        "accelerated":       False,
    }

    try:
        sim = twin.simulate(params, n_samples=3000, seed=42)
    except Exception as e:
        return html.Div(f"Simulation error: {e}", style={"color":COL_RED}), {"display":"block"}

    sb   = sim["single_bid_risk"]["probability"]
    comp = sim["competition"]["mean"]
    pr   = sim["price_ratio"]["mean"]
    dur_v= sim["duration"]["mean"]
    cb   = sim["cross_border"]["probability"]

    def _kpi(val_str, label, color):
        return html.Div([
            html.Div(val_str, style={"fontSize":"22px","fontWeight":"700","color":color}),
            html.Div(label,   style={"fontSize":"11px","color":COL_GREY}),
        ], style={"flex":"1","textAlign":"center","backgroundColor":COL_BG,
                  "borderRadius":"8px","padding":"12px"})

    sb_color  = COL_RED if sb > 0.35 else COL_ORANGE if sb > 0.25 else COL_GREEN
    comp_color= COL_RED if comp < 2.5 else COL_ORANGE if comp < 4 else COL_GREEN

    # Distribution histogram for primary outcome (single-bid risk)
    comp_samples = sim["competition"]["samples"]
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=comp_samples, nbinsx=30,
        marker_color=COL_BLUE, opacity=0.75,
        name="Competition distribution",
    ))
    fig_dist.update_layout(
        title="Competition distribution (bids)",
        height=180, margin=dict(l=30,r=10,t=35,b=30),
        paper_bgcolor=COL_BG, plot_bgcolor=COL_BG,
        showlegend=False,
        xaxis=dict(title="Bids"),
        yaxis=dict(title="Count"),
    )

    panel = html.Div([
        html.Div([
            html.Span(f"📍 Drill-down: {cluster} / {_CPV_SHORT.get(cpv, cpv)}  ({country})",
                      style={"fontWeight":"700","color":COL_NAVY,"fontSize":"14px"}),
            html.Span(" ✕", id="radar-drilldown-close",
                      style={"float":"right","cursor":"pointer","color":COL_GREY,
                             "fontSize":"18px","lineHeight":"1","marginTop":"-2px"}),
        ], style={"marginBottom":"12px"}),
        html.Div([
            _kpi(f"{comp:.1f}", "Expected bids", comp_color),
            _kpi(f"{sb:.0%}",   "Single-bid risk", sb_color),
            _kpi(f"{cb:.1%}",   "Cross-border win", COL_TEAL),
            _kpi(f"{pr:.3f}",   "Price ratio", COL_ORANGE),
            _kpi(f"{dur_v:.0f}d", "Duration", COL_GREY),
        ], style={"display":"flex","gap":"8px","marginBottom":"12px"}),
        html.Div([
            html.Div([
                html.Div("Competition P10–P90",
                         style={"fontSize":"11px","color":COL_GREY,"marginBottom":"4px"}),
                html.Div(f"{sim['competition']['p10']:.1f} – {sim['competition']['p90']:.1f} bids",
                         style={"fontWeight":"600","color":COL_NAVY}),
            ], style={"flex":"1"}),
            html.Div([
                html.Div("Price ratio P10–P90",
                         style={"fontSize":"11px","color":COL_GREY,"marginBottom":"4px"}),
                html.Div(f"{sim['price_ratio']['p10']:.2f} – {sim['price_ratio']['p90']:.2f}",
                         style={"fontWeight":"600","color":COL_NAVY}),
            ], style={"flex":"1"}),
            html.Div([
                html.Div("Duration P10–P90",
                         style={"fontSize":"11px","color":COL_GREY,"marginBottom":"4px"}),
                html.Div(f"{sim['duration']['p10']:.0f} – {sim['duration']['p90']:.0f} days",
                         style={"fontWeight":"600","color":COL_NAVY}),
            ], style={"flex":"1"}),
        ], style={"display":"flex","gap":"16px","backgroundColor":COL_BG,
                  "borderRadius":"8px","padding":"12px","marginBottom":"12px"}),
        dcc.Graph(figure=fig_dist, config={"displayModeBar":False}),
    ], style={"backgroundColor":COL_CARD,"borderRadius":"10px","padding":"18px",
              "boxShadow":"0 2px 8px rgba(0,0,0,0.07)",
              "border":f"2px solid {COL_BLUE}"})

    return panel, {"display":"block","marginBottom":"14px"}


# ══════════════════════════════════════════════════════════════════
# MODEL ADMIN TAB
# ══════════════════════════════════════════════════════════════════

_ADMIN_MODELS = ["competition", "single_bid", "crossborder", "price_ratio", "duration"]
_ADMIN_MODEL_LABELS = {
    "competition":  "Competition (n_bids)",
    "single_bid":   "Single-bid risk",
    "crossborder":  "Cross-border win",
    "price_ratio":  "Price ratio",
    "duration":     "Duration (days)",
}
_CAL_MODELS = ["competition", "price_ratio"]  # only these have calibration offsets


def admin_layout():
    # ── Model evaluation table ────────────────────────────────────
    try:
        with open(os.path.join(MODEL_DIR, "model_evaluation.json")) as f:
            ev = json.load(f)
    except Exception:
        ev = {}

    metric_rows = []
    for mk in _ADMIN_MODELS:
        d = ev.get(mk, {})
        metric_rows.append({
            "Model":        _ADMIN_MODEL_LABELS.get(mk, mk),
            "Type":         "Regression" if mk in ("competition","price_ratio","duration") else "Classifier",
            "Train n":      f"{d.get('n_train', 'N/A'):,}" if isinstance(d.get('n_train'), int) else "N/A",
            "Test n":       f"{d.get('n_test', 'N/A'):,}"  if isinstance(d.get('n_test'), int)  else "N/A",
            "MAE / AUC":    f"{d.get('boost_mae', d.get('boost_auc', 'N/A')):.3f}"
                            if isinstance(d.get('boost_mae', d.get('boost_auc')), float) else "N/A",
            "R² / F1":      f"{d.get('boost_r2', d.get('boost_f1', 'N/A')):.3f}"
                            if isinstance(d.get('boost_r2', d.get('boost_f1')), float) else "N/A",
            "Baseline MAE": f"{d.get('baseline_mae', d.get('baseline_auc', 'N/A')):.3f}"
                            if isinstance(d.get('baseline_mae', d.get('baseline_auc')), float) else "N/A",
        })

    col_style = {"textAlign":"left","padding":"8px 12px","fontSize":"13px"}
    hdr_style = {**col_style,"backgroundColor":COL_NAVY,"color":"white","fontWeight":"700"}

    def _th(label): return html.Th(label, style=hdr_style)
    def _td(val, bold=False):
        s = {**col_style}
        if bold: s["fontWeight"] = "700"
        return html.Td(val, style=s)

    tbl_header = html.Tr([_th(c) for c in ["Model","Type","Train n","Test n","MAE / AUC","R² / F1","Baseline MAE"]])
    tbl_body   = [html.Tr([
        _td(r["Model"], bold=True), _td(r["Type"]), _td(r["Train n"]),
        _td(r["Test n"]), _td(r["MAE / AUC"]), _td(r["R² / F1"]), _td(r["Baseline MAE"]),
    ], style={"backgroundColor": COL_CARD if i%2==0 else "#f7f9fc"})
    for i, r in enumerate(metric_rows)]

    metrics_table = html.Table(
        [html.Thead(tbl_header), html.Tbody(tbl_body)],
        style={"width":"100%","borderCollapse":"collapse","borderRadius":"8px","overflow":"hidden"},
    )

    # ── SHAP section ──────────────────────────────────────────────
    shap_section = html.Div([
        html.H3("Feature Importance (SHAP)", style={"color":COL_NAVY,"fontSize":"15px",
                "fontWeight":"700","margin":"0 0 10px 0"}),
        html.Div([
            html.Label("Select model:", style={"fontWeight":"600","fontSize":"13px"}),
            dcc.Dropdown(
                id="admin-shap-model",
                options=[{"label": _ADMIN_MODEL_LABELS[m], "value": m} for m in _ADMIN_MODELS],
                value="competition", clearable=False,
                style={"width":"260px","display":"inline-block","marginLeft":"10px"},
            ),
        ], style={"marginBottom":"10px"}),
        dcc.Graph(id="admin-shap-chart", style={"height":"320px"}),
    ], style={"backgroundColor":COL_CARD,"borderRadius":"8px","padding":"18px",
              "boxShadow":"0 2px 6px rgba(0,0,0,0.06)","marginBottom":"20px"})

    # ── Calibration editor ────────────────────────────────────────
    cal_section = html.Div([
        html.H3("Calibration Offsets", style={"color":COL_NAVY,"fontSize":"15px",
                "fontWeight":"700","margin":"0 0 4px 0"}),
        html.P("Calibration offsets are applied in log space. Positive values raise predictions; "
               "negative values lower them. Changes take effect immediately after saving — no "
               "restart needed.", style={"fontSize":"12px","color":COL_GREY,"marginBottom":"12px"}),
        html.Div([
            html.Label("Select model:", style={"fontWeight":"600","fontSize":"13px"}),
            dcc.Dropdown(
                id="admin-cal-model",
                options=[{"label": _ADMIN_MODEL_LABELS[m], "value": m} for m in _CAL_MODELS],
                value="competition", clearable=False,
                style={"width":"260px","display":"inline-block","marginLeft":"10px"},
            ),
        ], style={"marginBottom":"14px"}),

        html.Div([
            html.Div([
                html.H4("By Country Cluster", style={"fontSize":"13px","fontWeight":"700",
                        "color":COL_NAVY,"marginBottom":"8px"}),
                html.Div(id="admin-cal-cluster-table"),
            ], style={"flex":"1","marginRight":"20px"}),
            html.Div([
                html.H4("By CPV Division", style={"fontSize":"13px","fontWeight":"700",
                        "color":COL_NAVY,"marginBottom":"8px"}),
                html.Div(id="admin-cal-cpv-table"),
            ], style={"flex":"1"}),
        ], style={"display":"flex","gap":"10px","alignItems":"flex-start"}),

        html.Div([
            html.Button("💾  Save Calibration", id="admin-save-btn",
                style={"backgroundColor":COL_NAVY,"color":"white","border":"none",
                       "borderRadius":"6px","padding":"10px 22px","fontSize":"13px",
                       "fontWeight":"600","cursor":"pointer","marginTop":"16px"}),
            html.Div(id="admin-save-status", style={"display":"inline-block",
                     "marginLeft":"14px","fontSize":"13px"}),
        ]),
        # Hidden stores for edited values
        dcc.Store(id="admin-cluster-edits"),
        dcc.Store(id="admin-cpv-edits"),
    ], style={"backgroundColor":COL_CARD,"borderRadius":"8px","padding":"18px",
              "boxShadow":"0 2px 6px rgba(0,0,0,0.06)","marginBottom":"20px"})

    # ── Pipeline panel ────────────────────────────────────────────
    _code = lambda t: html.Code(t, style={"backgroundColor":"#f0f4f8","padding":"2px 6px",
                                          "borderRadius":"4px","fontFamily":"monospace","fontSize":"12px"})
    _step_labels = ["Download TED data","CFC–CAN linkage","Feature engineering",
                    "Train models","Upload to HF Hub"]

    pipeline_section = html.Div([
        html.H3("🔄 Download & Retrain Pipeline", style={"color":COL_NAVY,"fontSize":"15px",
                "fontWeight":"700","margin":"0 0 6px 0"}),
        html.P([
            "Downloads fresh TED CSV data directly from the ",
            html.A("EU Open Data Portal", href="https://data.europa.eu/data/datasets/ted-csv",
                   target="_blank", style={"color":COL_BLUE}),
            ", rebuilds the feature store, retrains all 5 models, and uploads artifacts to "
            "HuggingFace Hub. Best run locally or via GitHub Actions (see below).",
        ], style={"fontSize":"12px","color":COL_GREY,"marginBottom":"14px"}),

        # Controls row
        html.Div([
            html.Div([
                html.Label("Data years:", style={"fontSize":"12px","fontWeight":"600",
                           "display":"block","marginBottom":"4px"}),
                dcc.RadioItems(
                    id="pipe-years-mode",
                    options=[
                        {"label":" Combined 2018–2023 (recommended)", "value":"combined"},
                        {"label":" Custom range",                      "value":"custom"},
                    ],
                    value="combined", className="radio-block",
                    style={"fontSize":"12px"},
                ),
                html.Div([
                    dcc.RangeSlider(
                        id="pipe-year-range", min=2018, max=2024, step=1,
                        value=[2022, 2023],
                        marks={y: str(y) for y in range(2018, 2025)},
                        tooltip={"always_visible": False},
                    ),
                ], id="pipe-year-range-row", style={"display":"none","marginTop":"8px"}),
            ], style={"flex":"1","marginRight":"24px"}),

            html.Div([
                html.Label("Steps to run:", style={"fontSize":"12px","fontWeight":"600",
                           "display":"block","marginBottom":"4px"}),
                dcc.Checklist(
                    id="pipe-steps",
                    options=[{"label":f" {l}", "value": v}
                             for v, l in zip(["download","linkage","features","train","upload"],
                                             _step_labels)],
                    value=["download","linkage","features","train","upload"],
                    style={"fontSize":"12px","lineHeight":"1.8"},
                ),
            ], style={"flex":"1"}),
        ], style={"display":"flex","marginBottom":"14px"}),

        # Start / cancel buttons
        html.Div([
            html.Button("▶  Start Pipeline", id="pipe-start-btn",
                style={"backgroundColor":COL_NAVY,"color":"white","border":"none",
                       "borderRadius":"6px","padding":"9px 20px","fontSize":"13px",
                       "fontWeight":"600","cursor":"pointer","marginRight":"10px"}),
            html.Span(id="pipe-btn-status", style={"fontSize":"12px","color":COL_GREY}),
        ], style={"marginBottom":"14px"}),

        # Step progress indicators
        html.Div(id="pipe-step-indicators", style={"marginBottom":"10px"}),

        # Live log viewer
        html.Div(
            html.Pre(id="pipe-log", children="No pipeline run yet.",
                style={"margin":"0","fontSize":"11px","lineHeight":"1.5",
                       "fontFamily":"monospace","color":"#cdd3d8"}),
            style={"backgroundColor":"#1e2328","borderRadius":"6px","padding":"12px",
                   "maxHeight":"280px","overflowY":"auto","border":"1px solid #333"},
        ),

        # Polling interval (only active while pipeline running)
        dcc.Interval(id="pipe-poll", interval=3000, disabled=True),
        dcc.Store(id="pipe-running-store", data=False),

    ], style={"backgroundColor":COL_CARD,"borderRadius":"8px","padding":"18px",
              "boxShadow":"0 2px 6px rgba(0,0,0,0.06)","marginBottom":"20px"})

    return html.Div([
        html.H2("⚙️ Model Administration", style={"color":COL_NAVY,"fontSize":"20px",
                "fontWeight":"700","margin":"0 0 20px 0"}),

        html.Div([
            html.H3("Model Health Metrics", style={"color":COL_NAVY,"fontSize":"15px",
                    "fontWeight":"700","margin":"0 0 12px 0"}),
            metrics_table,
        ], style={"backgroundColor":COL_CARD,"borderRadius":"8px","padding":"18px",
                  "boxShadow":"0 2px 6px rgba(0,0,0,0.06)","marginBottom":"20px"}),

        shap_section,
        cal_section,
        pipeline_section,
    ], style={"maxWidth":"1200px","margin":"0 auto","padding":"24px"})


@app.callback(Output("admin-shap-chart","figure"), Input("admin-shap-model","value"))
def admin_shap_chart(model_key):
    try:
        with open(os.path.join(MODEL_DIR, "shap_importances.json")) as f:
            shap_all = json.load(f)
    except Exception:
        return go.Figure()

    shap = shap_all.get(model_key, {})
    if not shap:
        return go.Figure(layout={"title":"No SHAP data available for this model"})

    items = sorted(shap.items(), key=lambda x: x[1], reverse=True)[:20]
    feats, vals = zip(*items) if items else ([], [])
    feats = [f.replace("ISO_COUNTRY_CODE_","Country: ").replace("country_cluster_","Cluster: ")
               .replace("TOP_TYPE_","Proc: ").replace("TYPE_OF_CONTRACT_","Type: ")
               .replace("CRIT_CODE_","Criteria: ").replace("value_bracket_","Value: ")
               .replace("cpv_division_","CPV: ") for f in feats]

    fig = go.Figure(go.Bar(
        x=list(vals), y=list(feats), orientation="h",
        marker_color=COL_BLUE,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top 20 features — {_ADMIN_MODEL_LABELS.get(model_key, model_key)} (mean |SHAP|)",
        yaxis={"autorange":"reversed","tickfont":{"size":11}},
        xaxis={"title":"Mean |SHAP value|"},
        margin={"l":200,"r":20,"t":40,"b":40},
        plot_bgcolor="white", paper_bgcolor="white",
        height=320,
    )
    return fig


def _cal_table_html(data_dict, table_id):
    """Render an editable-looking calibration table as HTML with input fields."""
    if not data_dict:
        return html.P("No calibration data available.", style={"fontSize":"12px","color":COL_GREY})

    hdr = html.Tr([
        html.Th("Segment", style={"padding":"6px 10px","backgroundColor":COL_NAVY,
                "color":"white","fontSize":"12px","fontWeight":"700","textAlign":"left"}),
        html.Th("Offset", style={"padding":"6px 10px","backgroundColor":COL_NAVY,
                "color":"white","fontSize":"12px","fontWeight":"700","textAlign":"right"}),
    ])
    rows = []
    for i, (seg, val) in enumerate(sorted(data_dict.items())):
        bg = COL_CARD if i % 2 == 0 else "#f7f9fc"
        col = COL_GREEN if val > 0 else (COL_RED if val < 0 else COL_GREY)
        rows.append(html.Tr([
            html.Td(seg, style={"padding":"5px 10px","fontSize":"12px","fontWeight":"600"}),
            html.Td(
                dcc.Input(
                    id={"type": table_id, "index": seg},
                    value=round(val, 4), type="number", debounce=True,
                    style={"width":"80px","textAlign":"right","fontSize":"12px",
                           "border":"1px solid #ccc","borderRadius":"4px","padding":"2px 6px",
                           "color": col},
                ),
                style={"padding":"3px 10px","textAlign":"right"},
            ),
        ], style={"backgroundColor": bg}))

    return html.Table(
        [html.Thead(hdr), html.Tbody(rows)],
        style={"width":"100%","borderCollapse":"collapse","fontSize":"12px"},
    )


@app.callback(
    Output("admin-cal-cluster-table","children"),
    Output("admin-cal-cpv-table","children"),
    Input("admin-cal-model","value"),
)
def admin_cal_tables(model_key):
    try:
        with open(os.path.join(MODEL_DIR, "calibration_offsets.json")) as f:
            cal = json.load(f)
    except Exception:
        cal = {}

    m = cal.get(model_key, {})
    cluster_data = m.get("by_cluster", {})
    cpv_data     = m.get("by_cpv", {})

    return (
        _cal_table_html(cluster_data, "admin-cluster-input"),
        _cal_table_html(cpv_data,     "admin-cpv-input"),
    )


@app.callback(
    Output("admin-save-status","children"),
    Input("admin-save-btn","n_clicks"),
    State("admin-cal-model","value"),
    State({"type":"admin-cluster-input","index":dash.ALL},"value"),
    State({"type":"admin-cluster-input","index":dash.ALL},"id"),
    State({"type":"admin-cpv-input","index":dash.ALL},"value"),
    State({"type":"admin-cpv-input","index":dash.ALL},"id"),
    prevent_initial_call=True,
)
def admin_save_calibration(n_clicks, model_key, cluster_vals, cluster_ids,
                           cpv_vals, cpv_ids):
    if not n_clicks:
        return ""
    try:
        cal_path = os.path.join(MODEL_DIR, "calibration_offsets.json")
        with open(cal_path) as f:
            cal = json.load(f)

        if model_key not in cal:
            cal[model_key] = {"by_cluster": {}, "by_cpv": {}}

        for id_obj, val in zip(cluster_ids, cluster_vals):
            seg = id_obj["index"]
            cal[model_key]["by_cluster"][seg] = float(val) if val is not None else 0.0

        for id_obj, val in zip(cpv_ids, cpv_vals):
            seg = id_obj["index"]
            cal[model_key]["by_cpv"][seg] = float(val) if val is not None else 0.0

        with open(cal_path, "w") as f:
            json.dump(cal, f, indent=2)

        # Hot-reload into running twin instance
        twin._calibration = cal

        return html.Span("✅ Saved and applied.", style={"color":COL_GREEN,"fontWeight":"600"})

    except Exception as e:
        return html.Span(f"❌ Error: {e}", style={"color":COL_RED,"fontWeight":"600"})


# ── Pipeline callbacks ─────────────────────────────────────────────────────────

@app.callback(
    Output("pipe-year-range-row","style"),
    Input("pipe-years-mode","value"),
)
def pipe_toggle_year_range(mode):
    return {"display":"block","marginTop":"8px"} if mode == "custom" else {"display":"none"}


_STEP_IDS    = ["download","linkage","features","train","upload"]
_STEP_LABELS = ["Download TED data","CFC–CAN linkage","Feature engineering",
                "Train models","Upload to HF Hub"]


def _make_pills(status, cur_step, step_idx):
    def _pill(label, state):
        colors = {"done":COL_GREEN,"active":COL_ORANGE,"pending":COL_GREY,"error":COL_RED}
        icons  = {"done":"✓ ","active":"⟳ ","pending":"○ ","error":"✗ "}
        bg     = {"done":"#e6f4ea","active":"#fff3cd","pending":"#f0f4f8","error":"#fde8e8"}
        return html.Span(f"{icons[state]}{label}",
            style={"backgroundColor":bg[state],"color":colors[state],"borderRadius":"12px",
                   "padding":"3px 10px","fontSize":"11px","fontWeight":"600",
                   "marginRight":"6px","display":"inline-block","marginBottom":"4px"})
    pills = []
    for i, (sid, slabel) in enumerate(zip(_STEP_IDS, _STEP_LABELS)):
        if status == "done":
            s = "done"
        elif status == "error" and cur_step == sid:
            s = "error"
        elif i < step_idx - 1:
            s = "done"
        elif i == step_idx - 1 and status == "running":
            s = "active"
        else:
            s = "pending"
        pills.append(_pill(slabel, s))
    return pills


@app.callback(
    Output("pipe-log",            "children"),
    Output("pipe-step-indicators","children"),
    Output("pipe-btn-status",     "children"),
    Output("pipe-running-store",  "data"),
    Output("pipe-poll",           "disabled"),
    Input("pipe-start-btn",       "n_clicks"),
    Input("pipe-poll",            "n_intervals"),
    State("pipe-years-mode",      "value"),
    State("pipe-year-range",      "value"),
    State("pipe-steps",           "value"),
    State("pipe-running-store",   "data"),
    prevent_initial_call=True,
)
def pipe_controller(n_clicks, n_intervals, years_mode, year_range, steps, running):
    trigger = ctx.triggered_id

    # ── Start button clicked ─────────────────────────────────────
    if trigger == "pipe-start-btn":
        if not _PIPELINE_AVAILABLE:
            return (
                "❌ Pipeline module not importable — check Space logs for the traceback.",
                [], "❌ Module unavailable", False, True,
            )
        if running:
            return (
                _read_log(), _read_pills(), "⟳ Already running…", True, False,
            )
        try:
            with open(_PIPELINE_STATUS) as f:
                if json.load(f).get("status") == "running":
                    return _read_log(), _read_pills(), "⟳ Already running…", True, False
        except Exception:
            pass

        kwargs = {"step_ids": steps or None}
        if years_mode == "custom" and year_range:
            kwargs["download_years"] = list(range(int(year_range[0]), int(year_range[1]) + 1))

        try:
            _run_pipeline_async(**kwargs)
        except Exception as exc:
            return (f"❌ Failed to start: {exc}", [], f"❌ {exc}", False, True)

        initial_log = (
            "▶ Pipeline started!\n"
            "   Downloading TED data from EU Open Data Portal…\n"
            "   (This first step may take 10–30 min depending on connection speed)\n"
            "   Log will refresh every 3 seconds."
        )
        initial_pills = _make_pills("running", "download", 1)
        return initial_log, initial_pills, "⟳ Starting — Download TED data…", True, False

    # ── Interval poll ────────────────────────────────────────────
    try:
        with open(_PIPELINE_STATUS) as f:
            st = json.load(f)
    except FileNotFoundError:
        return "No pipeline run yet — click ▶ Start Pipeline.", [], "", False, True
    except Exception as exc:
        return f"Error reading status: {exc}", [], "", False, True

    status    = st.get("status", "idle")
    cur_step  = st.get("step", "")
    step_idx  = st.get("step_idx", 0)
    logs      = st.get("logs", [])
    error_msg = st.get("error", "")

    log_text = "\n".join(logs[-200:]) if logs else "Waiting for first output…"
    pills    = _make_pills(status, cur_step, step_idx)
    still_running = (status == "running")

    if status == "done":
        btn = f"✅ Complete — {st.get('finished_at','')}"
    elif status == "error":
        btn = f"❌ {error_msg or 'Pipeline failed — see log'}"
    elif status == "running":
        btn = f"⟳ {st.get('step_label','')}  (step {step_idx}/{st.get('total_steps',5)})"
    else:
        btn = ""

    return log_text, pills, btn, still_running, not still_running


def _read_log():
    try:
        with open(_PIPELINE_STATUS) as f:
            return "\n".join(json.load(f).get("logs", [])[-200:])
    except Exception:
        return ""

def _read_pills():
    try:
        with open(_PIPELINE_STATUS) as f:
            st = json.load(f)
        return _make_pills(st.get("status","idle"), st.get("step",""), st.get("step_idx",0))
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
# HTML TEMPLATE  ─  load Inter font; all styles are in assets/style.css
# ══════════════════════════════════════════════════════════════════
app.index_string = '''<!DOCTYPE html>
<html lang="en">
<head>
{%metas%}
<title>Procurement Digital Twin</title>
{%favicon%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
{%css%}
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>'''

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PROCUREMENT DIGITAL TWIN — DASH APPLICATION  v2")
    print("="*60)
    print("\nStarting server at http://localhost:8050")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
