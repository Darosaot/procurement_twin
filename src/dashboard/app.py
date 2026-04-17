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

import sys, os, json

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")

for _p in [_PROJECT_ROOT, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

FEAT_DIR  = os.path.join(_PROJECT_ROOT, "data", "features")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import polars as pl

from simulation.simulation_engine import (
    ProcurementTwin, COUNTRY_CLUSTERS, CPV_SECTORS, value_bracket
)

# ── Initialise twin ───────────────────────────────────────────────
twin = ProcurementTwin()

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

# ── Colour palette ────────────────────────────────────────────────
COL_NAVY   = "#1F3864"
COL_BLUE   = "#2E75B6"
COL_LIGHT  = "#D5E8F0"
COL_TEAL   = "#00A7A7"
COL_GREEN  = "#217346"
COL_RED    = "#C00000"
COL_ORANGE = "#E67E22"
COL_GREY   = "#8C9099"
COL_ACCENT = "#4472C4"
COL_BG     = "#F0F4F8"
COL_CARD   = "#FFFFFF"

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
    return html.Div([
        _form_row("Country", dcc.Dropdown(
            id=f"{prefix}-country",
            options=[{"label": c, "value": c} for c in COUNTRIES],
            value=d.get("country","DE"), clearable=False)),

        _form_row("Procedure type", html.Div([
            dcc.Dropdown(id=f"{prefix}-proc", options=PROC_OPTIONS,
                         value=d.get("proc","OPE"), clearable=False),
            html.Div(id=f"{prefix}-proc-desc",
                     style={"fontSize":"11px","color":COL_GREY,"marginTop":"4px",
                            "fontStyle":"italic"}),
        ])),

        _form_row("Contract type", dcc.RadioItems(
            id=f"{prefix}-ctype", options=CONTRACT_OPTIONS,
            value=d.get("ctype","S"), inline=True, className="radio-inline")),

        _form_row("CPV sector", dcc.Dropdown(
            id=f"{prefix}-cpv", options=CPV_OPTIONS,
            value=d.get("cpv","72"), clearable=False)),

        _form_row("Award criteria", dcc.RadioItems(
            id=f"{prefix}-crit", options=CRITERIA_OPTIONS,
            value=d.get("crit","M"), className="radio-block")),

        _form_row("Price weight (%)", html.Div([
            html.Div(id=f"{prefix}-pw-val",
                     style={"fontSize":"11px","color":COL_GREY,"textAlign":"right",
                            "marginBottom":"2px"}),
            dcc.Slider(id=f"{prefix}-pw", min=0, max=100, step=5,
                       value=d.get("pw",60),
                       marks={0:"0%",25:"25%",50:"50%",75:"75%",100:"100%"},
                       tooltip={"placement":"bottom","always_visible":False}),
        ]), id_suffix="-pw-row"),

        _form_row("Estimated value (€)", dcc.Input(
            id=f"{prefix}-val", type="number", value=d.get("val",1_000_000),
            min=10_000, step=10_000, style={"width":"100%","padding":"6px",
            "border":"1px solid #CCC","borderRadius":"4px","fontSize":"13px"})),

        _form_row("Preparation time (days)", dcc.Slider(
            id=f"{prefix}-prep", min=15, max=90, step=1,
            value=d.get("prep",35),
            marks={15:"15",35:"35",52:"52",90:"90"},
            tooltip={"placement":"bottom","always_visible":True})),

        _form_row("Contract duration (months)", dcc.Slider(
            id=f"{prefix}-dur", min=3, max=60, step=3,
            value=d.get("dur",24),
            marks={3:"3m",12:"1yr",24:"2yr",36:"3yr",60:"5yr"},
            tooltip={"placement":"bottom","always_visible":True})),

        _form_row("Options", dcc.Checklist(
            id=f"{prefix}-flags",
            options=[
                {"label": " GPA covered",         "value": "gpa"},
                {"label": " EU funds",            "value": "eu_funds"},
                {"label": " Electronic auction",  "value": "ea"},
                {"label": " Framework agreement", "value": "fra"},
                {"label": " Accelerated",         "value": "acc"},
            ],
            value=d.get("flags",["gpa"]), className="checklist")),

        html.Button("▶  Simulate", id=f"{prefix}-btn",
                    n_clicks=0, className="btn-primary"),

        # Immediate status message — updated via clientside callback on click
        html.Div(id=f"{prefix}-status", style={
            "fontSize":"12px","color":COL_BLUE,"marginTop":"10px",
            "minHeight":"18px","fontStyle":"italic",
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
                               opacity=0.78, name=label))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color=COL_RED,
                      line_width=2,
                      annotation_text=f"  {vline:.2f}",
                      annotation_font=dict(size=11, color=COL_RED))
    fig.update_layout(
        title=dict(text=label, font=dict(size=12, color="#333")),
        height=height, margin=dict(t=36, b=26, l=36, r=12),
        paper_bgcolor=COL_CARD, plot_bgcolor=COL_CARD,
        showlegend=False,
        xaxis=dict(gridcolor="#EEE", linecolor="#CCC"),
        yaxis=dict(gridcolor="#EEE", linecolor="#CCC"),
    )
    return fig


def kpi_card(label, value_str, badge=None, badge_col=None, note=None, tooltip_text=None):
    """Polished KPI card with optional badge, benchmark note, and hover tooltip."""
    # Label row: text + optional ℹ️ tooltip icon
    label_content = html.Div(
        [
            html.Span(label,
                      style={"fontSize":"11px","color":COL_GREY,
                             "textTransform":"uppercase","letterSpacing":"0.5px"}),
            *([html.Span(
                "ⓘ",
                className="kpi-info-icon",
                **{"data-kpi-tooltip": tooltip_text},
                style={"marginLeft":"4px","cursor":"help",
                       "color":COL_BLUE,"fontSize":"11px",
                       "fontWeight":"700","verticalAlign":"middle",
                       "userSelect":"none"},
            )] if tooltip_text else []),
        ],
        style={"display":"inline-flex","alignItems":"center",
               "justifyContent":"center","marginTop":"4px",
               "position":"relative"},
    )

    children = [
        html.Div(value_str,
                 style={"fontSize":"26px","fontWeight":"700","color":COL_NAVY,
                        "lineHeight":"1","letterSpacing":"-0.5px"}),
        label_content,
    ]
    if badge:
        children.append(
            html.Div(badge,
                     style={"fontSize":"11px","fontWeight":"600","marginTop":"6px",
                            "color": badge_col or COL_GREY,
                            "backgroundColor": (badge_col or COL_GREY) + "18",
                            "borderRadius":"4px","padding":"2px 6px",
                            "display":"inline-block"}))
    if note:
        children.append(
            html.Div(note, style={"fontSize":"10px","color":COL_GREY,"marginTop":"4px"}))

    return html.Div(children,
        style={"textAlign":"center","padding":"16px 10px",
               "backgroundColor":COL_CARD,
               "borderRadius":"8px","boxShadow":"0 1px 6px rgba(0,0,0,0.08)",
               "border":f"1px solid {COL_LIGHT}"})


def _section_header(text):
    return html.H3(text,
        style={"color":COL_NAVY,"marginTop":"0","marginBottom":"16px",
               "borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"8px",
               "fontSize":"16px","fontWeight":"700"})


def _card(children, style=None):
    s = {"backgroundColor":COL_CARD,"padding":"20px","borderRadius":"10px",
         "boxShadow":"0 2px 10px rgba(0,0,0,0.07)"}
    if style:
        s.update(style)
    return html.Div(children, style=s)


# ══════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, title="Procurement Digital Twin",
                suppress_callback_exceptions=True)

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div("🔷", style={"fontSize":"28px","marginRight":"12px"}),
            html.Div([
                html.H1("Procurement Digital Twin",
                        style={"color":"white","margin":"0","fontSize":"22px",
                               "fontWeight":"700","letterSpacing":"-0.3px"}),
                html.P("EU procurement simulator  ·  1.1M TED contracts 2018–2023",
                       style={"color":"#A8C4E0","margin":"2px 0 0","fontSize":"12px"}),
            ]),
        ], style={"display":"flex","alignItems":"center"}),
    ], style={"backgroundColor":COL_NAVY,"padding":"14px 24px",
              "borderBottom":f"4px solid {COL_BLUE}"}),

    # Tabs
    dcc.Tabs(id="tabs", value="tab-designer", className="tab-bar", children=[
        dcc.Tab(label="🎯  Procedure Designer",   value="tab-designer"),
        dcc.Tab(label="⚖️  Scenario Comparator",  value="tab-compare"),
        dcc.Tab(label="🔍  Policy Explorer",       value="tab-explorer"),
        dcc.Tab(label="🏛️  Policy Simulation",    value="tab-policy"),
        dcc.Tab(label="💡  Explain",               value="tab-explain"),
        dcc.Tab(label="📖  Methodology",           value="tab-methodology"),
    ]),

    html.Div(id="tab-content", style={"minHeight":"600px"}),
], style={"fontFamily":"'Segoe UI', Arial, sans-serif",
          "backgroundColor":COL_BG,"minHeight":"100vh"})


@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    if tab == "tab-designer":    return designer_layout()
    if tab == "tab-compare":     return comparator_layout()
    if tab == "tab-explorer":    return explorer_layout()
    if tab == "tab-policy":      return policy_layout()
    if tab == "tab-explain":     return explain_layout()
    if tab == "tab-methodology": return methodology_layout()
    return html.Div("Unknown tab")


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
        html.Div([
            # Left form
            _card([
                _section_header("Procedure Parameters"),
                build_form("d"),
            ], style={"width":"320px","flexShrink":"0"}),

            # Right results
            _card([
                _section_header("Simulation Results"),
                dcc.Loading(
                    id="designer-loading",
                    type="circle",
                    color=COL_BLUE,
                    overlay_style={"visibility":"visible","filter":"blur(1px)"},
                    custom_spinner=html.Div([
                        html.Div("⚙️", style={"fontSize":"32px","marginBottom":"8px"}),
                        html.Div("Running simulation…",
                                 style={"fontWeight":"700","color":COL_NAVY,"fontSize":"15px"}),
                        html.Div("Monte Carlo sampling · Calibrating · Benchmarking",
                                 style={"fontSize":"11px","color":COL_GREY,"marginTop":"4px"}),
                    ], style={"textAlign":"center","padding":"50px 20px"}),
                    children=html.Div(id="designer-results",
                        children=html.Div([
                            html.Div("👈", style={"fontSize":"40px","marginBottom":"10px"}),
                            html.P("Set procedure parameters and click  ▶ Simulate.",
                                   style={"color":COL_GREY,"fontSize":"14px"}),
                        ], style={"textAlign":"center","paddingTop":"60px"})),
                ),
            ], style={"flex":"1","marginLeft":"16px"}),
        ], style={"display":"flex","padding":"20px","alignItems":"flex-start","gap":"0"}),
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
    [Output("designer-results","children"),
     Output("d-status","children")],
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

    kpi_row = html.Div(
        [kpi_card(label, val, badge, col,
                  note=outcome_notes.get(label),
                  tooltip_text=OUTCOME_TOOLTIPS.get(label))
         for label, val, badge, col in kpis],
        style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)",
               "gap":"10px","marginBottom":"20px"})

    # Distribution charts
    dist_row = html.Div([
        html.Div([dcc.Graph(
            figure=dist_chart(result["competition"]["samples"],
                              "Competition — offers received", COL_BLUE,
                              vline=comp_mean),
            config={"displayModeBar":False})], style={"flex":"1"}),
        html.Div([dcc.Graph(
            figure=dist_chart(result["price_ratio"]["samples"],
                              "Price ratio — award / estimate", COL_ACCENT),
            config={"displayModeBar":False})], style={"flex":"1"}),
        html.Div([dcc.Graph(
            figure=dist_chart(result["duration"]["samples"],
                              "Procedure duration — days", COL_GREEN,
                              vline=dur_median),
            config={"displayModeBar":False})], style={"flex":"1"}),
    ], style={"display":"flex","gap":"10px","marginBottom":"14px"})

    # Single-bid and cross-border gauges
    def gauge_bar(prob, label, color):
        pct = round(prob * 100)
        bar_color = color if pct > 25 else COL_GREEN
        return html.Div([
            html.Div([
                html.Span(label, style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                html.Span(f"{pct}%", style={"fontSize":"20px","fontWeight":"700",
                                             "color":bar_color,"float":"right"}),
            ], style={"overflow":"hidden","marginBottom":"6px"}),
            html.Div(html.Div(style={
                "width":f"{pct}%","height":"8px",
                "backgroundColor":bar_color,"borderRadius":"4px",
                "transition":"width 0.4s ease",
            }), style={"backgroundColor":"#EEF2F7","borderRadius":"4px","overflow":"hidden"}),
        ], style={"padding":"10px 14px","border":f"1px solid {COL_LIGHT}",
                  "borderRadius":"8px","backgroundColor":"#F7FAFE","marginBottom":"8px"})

    gauge_row = html.Div([
        html.Div([
            gauge_bar(sb_prob,   "🔴  Single-bid risk",    COL_RED),
            gauge_bar(cb_prob,   "🟢  Cross-border win",   COL_GREEN),
        ], style={"flex":"1"}),
        html.Div([
            html.P(
                f"Benchmark: {bench['n_records']:,} historical procedures "
                f"matching country={country}, procedure={proc}, CPV={cpv}",
                style={"fontSize":"11px","color":COL_GREY,"fontStyle":"italic","margin":"0"}),
            html.P(f"Simulation: 5,000 Monte Carlo samples · calibrated per CPV & country cluster",
                   style={"fontSize":"10px","color":COL_GREY,"margin":"4px 0 0"}),
        ], style={"flex":"1","display":"flex","flexDirection":"column",
                  "justifyContent":"center","paddingLeft":"20px"}),
    ], style={"display":"flex","gap":"20px","alignItems":"center"})

    return html.Div([kpi_row, dist_row, gauge_row]), ""  # "" clears the status message


# ══════════════════════════════════════════════════════════════════
# TAB 2: SCENARIO COMPARATOR
# ══════════════════════════════════════════════════════════════════
def comparator_layout():
    return html.Div([
        html.Div([
            _card([
                html.H3("Scenario A",
                        style={"color":COL_BLUE,"marginTop":"0",
                               "borderBottom":f"3px solid {COL_BLUE}",
                               "paddingBottom":"8px","fontSize":"16px"}),
                build_form("ca", {"country":"DE","proc":"OPE","crit":"L",
                                   "val":2_000_000,"cpv":"45"}),
            ], style={"flex":"1"}),
            _card([
                html.H3("Scenario B",
                        style={"color":COL_ACCENT,"marginTop":"0",
                               "borderBottom":f"3px solid {COL_ACCENT}",
                               "paddingBottom":"8px","fontSize":"16px"}),
                build_form("cb", {"country":"DE","proc":"OPE","crit":"M",
                                   "val":2_000_000,"cpv":"45"}),
            ], style={"flex":"1","marginLeft":"14px"}),
        ], style={"display":"flex","padding":"20px 20px 0 20px"}),

        html.Div([
            html.Button("⚖️  Compare Scenarios", id="compare-btn", n_clicks=0,
                        style={"padding":"12px 36px","fontSize":"15px","fontWeight":"600",
                               "backgroundColor":COL_NAVY,"color":"white","border":"none",
                               "borderRadius":"6px","cursor":"pointer",
                               "boxShadow":"0 2px 8px rgba(0,0,0,0.18)"}),
            html.Div(id="compare-status", style={
                "fontSize":"12px","color":COL_BLUE,"marginTop":"8px",
                "minHeight":"18px","fontStyle":"italic",
            }),
        ], style={"textAlign":"center","padding":"16px"}),

        dcc.Loading(
            id="compare-loading",
            type="circle",
            color=COL_BLUE,
            overlay_style={"visibility":"visible","filter":"blur(1px)"},
            custom_spinner=html.Div([
                html.Div("⚙️", style={"fontSize":"32px","marginBottom":"8px"}),
                html.Div("Comparing scenarios…",
                         style={"fontWeight":"700","color":COL_NAVY,"fontSize":"15px"}),
                html.Div("Simulating A · Simulating B · Computing deltas",
                         style={"fontSize":"11px","color":COL_GREY,"marginTop":"4px"}),
            ], style={"textAlign":"center","padding":"50px 20px"}),
            children=html.Div(id="compare-results", style={"padding":"0 20px 20px"}),
        ),
    ])


@app.callback(
    Output("ca-proc-desc","children"), Input("ca-proc","value"))
def upd_desc_ca(p): return PROC_DESCRIPTIONS.get(p,"")

@app.callback(
    Output("cb-proc-desc","children"), Input("cb-proc","value"))
def upd_desc_cb(p): return PROC_DESCRIPTIONS.get(p,"")


@app.callback(
    [Output("compare-results","children"),
     Output("compare-status","children")],
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
    ]), ""  # "" clears the status message


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
        html.Div([
            _card([
                _section_header("🏛️  Policy Intervention"),
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
        html.Div([
            _card([
                _section_header("💡  Model Explainability"),

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
# INLINE CSS
# ══════════════════════════════════════════════════════════════════
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Segoe UI", Arial, sans-serif;
         background: #F0F4F8; }

  /* Form */
  .form-label { display: block; font-size: 11px; font-weight: 600;
                color: #555; margin: 0 0 4px 0; text-transform: uppercase;
                letter-spacing: 0.3px; }
  .form-group  { margin-bottom: 13px; }
  .form-panel  { padding: 0; }

  /* Buttons */
  .btn-primary { width: 100%; padding: 11px; background: #1F3864;
                 color: white; border: none; border-radius: 6px;
                 font-size: 14px; font-weight: 600; cursor: pointer;
                 margin-top: 10px; transition: background 0.2s; }
  .btn-primary:hover { background: #2E75B6; }

  /* Radio / checklist */
  .radio-inline label { margin-right: 14px; font-size: 13px; cursor: pointer; }
  .radio-block  label { display: block; margin-bottom: 6px; font-size: 13px;
                        cursor: pointer; }
  .checklist    label { font-size: 12px; cursor: pointer; display: block;
                        margin-bottom: 4px; }

  /* Tabs */
  .tab-bar .tab { font-size: 13px; font-weight: 600;
                  padding: 10px 16px; border: none !important;
                  color: #555 !important; background: #E8EDF2 !important;
                  border-radius: 0 !important; }
  .tab-bar .tab--selected { color: #1F3864 !important;
                             background: #F0F4F8 !important;
                             border-bottom: 3px solid #2E75B6 !important; }
  .tab-bar { border-bottom: 1px solid #D0D8E4 !important; }

  /* Dropdowns */
  .Select-control { font-size: 13px !important; border-radius: 4px !important; }
  .Select-menu-outer { font-size: 13px !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #F0F4F8; }
  ::-webkit-scrollbar-thumb { background: #C0CCDA; border-radius: 3px; }

  /* ── KPI info tooltip ──────────────────────────────────────────── */
  .kpi-info-icon {
    position: relative;
    display: inline-block;
  }
  .kpi-info-icon::after {
    content: attr(data-kpi-tooltip);
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #1F3864;
    color: #fff;
    font-size: 11px;
    font-weight: 400;
    line-height: 1.45;
    padding: 8px 10px;
    border-radius: 6px;
    width: 220px;
    white-space: normal;
    text-align: left;
    text-transform: none;
    letter-spacing: 0;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.18s ease;
    z-index: 9999;
    box-shadow: 0 4px 14px rgba(0,0,0,0.22);
  }
  .kpi-info-icon::before {
    content: "";
    position: absolute;
    bottom: calc(100% + 1px);
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: #1F3864;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.18s ease;
    z-index: 9999;
  }
  .kpi-info-icon:hover::after,
  .kpi-info-icon:hover::before { opacity: 1; }

  /* ── Methodology tab ───────────────────────────────────────────── */
  .meth-model-card {
    background: #fff;
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    flex: 1;
    min-width: 0;
  }
  .meth-section-title {
    color: #1F3864;
    font-size: 15px;
    font-weight: 700;
    margin: 0 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #D5E8F0;
  }
  .meth-pipeline-step {
    flex: 1;
    text-align: center;
    border-radius: 8px;
    padding: 12px 10px;
    color: white;
  }
  .meth-limitation-row {
    padding: 10px 14px;
    background: #FFFBF0;
    border-left: 3px solid #E67E22;
    border-radius: 6px;
    margin-bottom: 8px;
  }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
'''

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PROCUREMENT DIGITAL TWIN — DASH APPLICATION  v2")
    print("="*60)
    print("\nStarting server at http://localhost:8050")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
