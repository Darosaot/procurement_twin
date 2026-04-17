"""
Phase 4: Procurement Digital Twin — Plotly Dash Application
=============================================================
Run with:  python src/dashboard/05_app.py
Then open: http://localhost:8050

Three views:
  1. Procedure Designer   — real-time simulation for a single procedure
  2. Scenario Comparator  — side-by-side comparison of two designs
  3. Policy Explorer      — empirical benchmarks + distribution explorer
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import polars as pl

from simulation.simulation_engine import ProcurementTwin

# ── Initialise twin (loads models once) ──────────────────────────
twin = ProcurementTwin()

# ── Load reference lists ─────────────────────────────────────────
COUNTRIES = sorted(["AT","BE","BG","CY","CZ","DE","DK","EE","ES","FI","FR","GR",
                     "HR","HU","IE","IT","LT","LU","LV","MT","NL","PL","PT","RO",
                     "SE","SI","SK","UK","NO","CH","IS","MK"])

CPV_OPTIONS = [
    {"label": "03 – Agriculture & Forestry",     "value": "03"},
    {"label": "15 – Food & Beverages",           "value": "15"},
    {"label": "22 – Printed Matter",             "value": "22"},
    {"label": "30 – IT Equipment",               "value": "30"},
    {"label": "33 – Medical & Pharma",           "value": "33"},
    {"label": "34 – Transport Equipment",        "value": "34"},
    {"label": "39 – Furniture & Fittings",       "value": "39"},
    {"label": "42 – Industrial Machinery",       "value": "42"},
    {"label": "44 – Construction Materials",     "value": "44"},
    {"label": "45 – Construction Works",         "value": "45"},
    {"label": "48 – Software",                   "value": "48"},
    {"label": "50 – Repair & Maintenance",       "value": "50"},
    {"label": "60 – Transport Services",         "value": "60"},
    {"label": "65 – Gas & Electricity",          "value": "65"},
    {"label": "66 – Financial Services",         "value": "66"},
    {"label": "70 – Real Estate",                "value": "70"},
    {"label": "71 – Architecture & Engineering", "value": "71"},
    {"label": "72 – IT Services",                "value": "72"},
    {"label": "73 – R&D Services",               "value": "73"},
    {"label": "75 – Public Administration",      "value": "75"},
    {"label": "79 – Business Services",          "value": "79"},
    {"label": "80 – Education",                  "value": "80"},
    {"label": "85 – Health & Social Work",       "value": "85"},
    {"label": "90 – Waste & Environment",        "value": "90"},
    {"label": "98 – Other Services",             "value": "98"},
]

PROC_OPTIONS = [
    {"label": "Open procedure",                    "value": "OPE"},
    {"label": "Restricted procedure",              "value": "RES"},
    {"label": "Negotiated with prior call",        "value": "NIC"},
    {"label": "Competitive dialogue",              "value": "COD"},
    {"label": "Innovation partnership",            "value": "INP"},
    {"label": "Award without prior publication",   "value": "AWP"},
]
CONTRACT_OPTIONS = [
    {"label": "Services",   "value": "S"},
    {"label": "Supplies",   "value": "U"},
    {"label": "Works",      "value": "W"},
]
CRITERIA_OPTIONS = [
    {"label": "MEAT (Most Economically Advantageous Tender)", "value": "M"},
    {"label": "Lowest price only",                           "value": "L"},
]

# ── Colour palette ────────────────────────────────────────────────
COL_BLUE   = "#1F3864"
COL_MID    = "#2E75B6"
COL_LIGHT  = "#D5E8F0"
COL_GREEN  = "#375623"
COL_RED    = "#C00000"
COL_GREY   = "#BFBFBF"
COL_ACCENT = "#4472C4"

# ── Shared form component ─────────────────────────────────────────
def build_form(prefix, col_country="DE", col_proc="OPE", col_ctype="S",
               col_cpv="72", col_crit="M", col_val=1_000_000,
               col_prep=35, col_dur=24, col_pw=60):
    return html.Div([
        html.Div([
            html.Label("Country", className="form-label"),
            dcc.Dropdown(
                id=f"{prefix}-country",
                options=[{"label": c, "value": c} for c in COUNTRIES],
                value=col_country, clearable=False, className="dropdown"
            ),
        ], className="form-group"),

        html.Div([
            html.Label("Procedure type", className="form-label"),
            dcc.Dropdown(id=f"{prefix}-proc", options=PROC_OPTIONS, value=col_proc, clearable=False),
        ], className="form-group"),

        html.Div([
            html.Label("Contract type", className="form-label"),
            dcc.RadioItems(id=f"{prefix}-ctype", options=CONTRACT_OPTIONS, value=col_ctype,
                           inline=True, className="radio-inline"),
        ], className="form-group"),

        html.Div([
            html.Label("CPV sector (2-digit)", className="form-label"),
            dcc.Dropdown(id=f"{prefix}-cpv", options=CPV_OPTIONS, value=col_cpv, clearable=False),
        ], className="form-group"),

        html.Div([
            html.Label("Award criteria", className="form-label"),
            dcc.RadioItems(id=f"{prefix}-crit", options=CRITERIA_OPTIONS, value=col_crit,
                           className="radio-block"),
        ], className="form-group"),

        html.Div([
            html.Label(id=f"{prefix}-pw-label", children="Price weight (%)"),
            dcc.Slider(id=f"{prefix}-pw", min=0, max=100, step=5, value=col_pw,
                       marks={0:"0%",25:"25%",50:"50%",75:"75%",100:"100%"},
                       tooltip={"placement":"bottom","always_visible":True}),
        ], className="form-group", id=f"{prefix}-pw-row"),

        html.Div([
            html.Label("Estimated value (€)", className="form-label"),
            dcc.Input(id=f"{prefix}-val", type="number", value=col_val,
                      min=10000, step=10000, className="input-number",
                      style={"width":"100%","padding":"6px"}),
        ], className="form-group"),

        html.Div([
            html.Label("Preparation time (days)", className="form-label"),
            dcc.Slider(id=f"{prefix}-prep", min=15, max=90, step=1, value=col_prep,
                       marks={15:"15",35:"35",52:"52",90:"90"},
                       tooltip={"placement":"bottom","always_visible":True}),
        ], className="form-group"),

        html.Div([
            html.Label("Contract duration (months)", className="form-label"),
            dcc.Slider(id=f"{prefix}-dur", min=3, max=60, step=3, value=col_dur,
                       marks={3:"3m",12:"1yr",24:"2yr",36:"3yr",60:"5yr"},
                       tooltip={"placement":"bottom","always_visible":True}),
        ], className="form-group"),

        html.Div([
            html.Label("Options", className="form-label"),
            dcc.Checklist(
                id=f"{prefix}-flags",
                options=[
                    {"label": "GPA covered",         "value": "gpa"},
                    {"label": "EU funds",            "value": "eu_funds"},
                    {"label": "Electronic auction",  "value": "ea"},
                    {"label": "Framework agreement", "value": "fra"},
                    {"label": "Accelerated",         "value": "acc"},
                ],
                value=["gpa"],
                inline=False,
                className="checklist"
            ),
        ], className="form-group"),

        html.Button("Simulate", id=f"{prefix}-btn",
                    n_clicks=0, className="btn-primary"),
    ], className="form-panel")


def form_to_params(country, proc, ctype, cpv, crit, val, prep, dur, pw, flags):
    flags = flags or []
    return {
        "country":          country,
        "procedure_type":   proc,
        "contract_type":    ctype,
        "cpv_division":     cpv,
        "criteria":         crit,
        "price_weight_pct": float(pw) if pw else 50,
        "value_euro":       float(val) if val else 1_000_000,
        "prep_time_days":   float(prep) if prep else 35,
        "duration_months":  float(dur) if dur else 24,
        "gpa":              "gpa" in flags,
        "eu_funds":         "eu_funds" in flags,
        "electronic_auction": "ea" in flags,
        "fra_agreement":    "fra" in flags,
        "accelerated":      "acc" in flags,
    }


# ── Gauge chart helper ───────────────────────────────────────────
def gauge(value, label, min_val, max_val, color, suffix="", fmt=".1f"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickfont": {"size": 11}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#F5F5F5",
            "steps": [{"range": [min_val, max_val], "color": "#EFEFEF"}],
        },
        title={"text": label, "font": {"size": 13, "color": "#555"}},
    ))
    fig.update_layout(height=180, margin=dict(t=40, b=5, l=10, r=10),
                      paper_bgcolor="white")
    return fig


def dist_chart(samples, label, color, hist=True, vline=None):
    samples_arr = np.array(samples)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples_arr, nbinsx=40,
        marker_color=color, opacity=0.75,
        name=label
    ))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color="#C00000",
                      annotation_text="Prediction", annotation_position="top right")
    fig.update_layout(
        title=dict(text=label, font=dict(size=13, color="#333")),
        height=200, margin=dict(t=40, b=30, l=30, r=10),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, title="Procurement Digital Twin",
                suppress_callback_exceptions=True)

app.layout = html.Div([
    # ── Header ───────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("🔷 Procurement Digital Twin",
                    style={"color": "white", "margin": "0", "fontSize": "24px"}),
            html.P("EU public procurement simulator  ·  2018–2023 TED data  ·  1.1M contracts",
                   style={"color": "#A8C4E0", "margin": "4px 0 0 0", "fontSize": "13px"}),
        ]),
    ], style={"backgroundColor": COL_BLUE, "padding": "16px 24px",
              "borderBottom": f"4px solid {COL_MID}"}),

    # ── Tabs ─────────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="tab-designer", className="tab-bar", children=[
        dcc.Tab(label="🎯  Procedure Designer", value="tab-designer"),
        dcc.Tab(label="⚖️  Scenario Comparator", value="tab-compare"),
        dcc.Tab(label="🔍  Policy Explorer",    value="tab-explorer"),
    ]),

    html.Div(id="tab-content", style={"padding": "0"}),

], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#F7F9FC",
          "minHeight": "100vh"})


# ── Tab routing ───────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-designer":  return designer_layout()
    if tab == "tab-compare":   return comparator_layout()
    if tab == "tab-explorer":  return explorer_layout()
    return html.Div("Unknown tab")


# ══════════════════════════════════════════════════════════════════
# TAB 1: PROCEDURE DESIGNER
# ══════════════════════════════════════════════════════════════════
def designer_layout():
    return html.Div([
        html.Div([
            # Left: form
            html.Div([
                html.H3("Procedure Parameters",
                        style={"color": COL_BLUE, "marginTop": "0", "borderBottom": f"2px solid {COL_LIGHT}", "paddingBottom": "8px"}),
                build_form("d"),
            ], style={"width": "320px", "flexShrink": "0",
                      "backgroundColor": "white", "padding": "20px",
                      "borderRadius": "8px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}),

            # Right: results
            html.Div([
                html.H3("Simulation Results",
                        style={"color": COL_BLUE, "marginTop": "0", "borderBottom": f"2px solid {COL_LIGHT}", "paddingBottom": "8px"}),
                html.Div(id="designer-results",
                         children=html.P("Set procedure parameters and click Simulate.",
                                         style={"color": COL_GREY, "textAlign": "center", "marginTop": "40px"})),
            ], style={"flex": "1", "backgroundColor": "white", "padding": "20px",
                      "borderRadius": "8px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                      "marginLeft": "16px"}),

        ], style={"display": "flex", "gap": "0", "padding": "20px",
                  "alignItems": "flex-start"}),
    ])


@app.callback(
    Output("designer-results", "children"),
    Input("d-btn", "n_clicks"),
    [State("d-country","value"), State("d-proc","value"), State("d-ctype","value"),
     State("d-cpv","value"),    State("d-crit","value"),  State("d-val","value"),
     State("d-prep","value"),   State("d-dur","value"),   State("d-pw","value"),
     State("d-flags","value")],
    prevent_initial_call=True
)
def run_designer(n, country, proc, ctype, cpv, crit, val, prep, dur, pw, flags):
    params = form_to_params(country, proc, ctype, cpv, crit, val, prep, dur, pw, flags)
    result = twin.simulate(params)
    bench  = twin.empirical_benchmark(country=country, procedure_type=proc, cpv_division=cpv)

    def kpi(label, sim_val, bench_val, unit="", fmt=".1f", color=COL_MID, low_good=False):
        delta = sim_val - bench_val if bench_val else None
        arrow = ""
        if delta is not None:
            better = delta < 0 if low_good else delta > 0
            arrow  = "▲" if delta > 0 else "▼"
            col    = COL_GREEN if better else COL_RED
        else:
            col = COL_GREY
        return html.Div([
            html.Div(f"{sim_val:{fmt}}{unit}", style={"fontSize": "28px", "fontWeight": "bold",
                                                       "color": color, "lineHeight": "1"}),
            html.Div(label, style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}),
            html.Div([
                html.Span(f"{arrow} {abs(delta):{fmt}}{unit} vs benchmark",
                          style={"fontSize": "11px", "color": col}) if delta is not None else
                html.Span("No benchmark", style={"fontSize": "11px", "color": COL_GREY})
            ]),
        ], style={"textAlign": "center", "padding": "16px 12px",
                  "backgroundColor": "#F7F9FC", "borderRadius": "8px",
                  "border": f"1px solid {COL_LIGHT}"})

    comp_mean  = result["competition"]["mean"]
    sb_prob    = result["single_bid_risk"]["probability"]
    cb_prob    = result["cross_border"]["probability"]
    pr_median  = result["price_ratio"]["median"]
    dur_median = result["duration"]["median"]

    b_comp = bench["competition"]["median"] if bench["competition"]["n"] > 0 else None
    b_sb   = bench["single_bid_rate"]
    b_cb   = bench["cross_border"]
    b_pr   = bench["price_ratio"]["median"] if bench["price_ratio"]["n"] > 0 else None
    b_dur  = bench["duration"]["median"] if bench["duration"]["n"] > 0 else None

    return html.Div([
        # KPI row
        html.Div([
            kpi("Expected bids",      comp_mean,     b_comp,   fmt=".1f"),
            kpi("P(single bid)",      sb_prob*100,   b_sb*100 if b_sb else None, unit="%", fmt=".0f", color=COL_RED,  low_good=True),
            kpi("P(cross-border win)",cb_prob*100,   b_cb*100 if b_cb else None, unit="%", fmt=".0f", color=COL_ACCENT),
            kpi("Price ratio",        pr_median,     b_pr,     fmt=".3f"),
            kpi("Duration",           dur_median,    b_dur,    unit="d", fmt=".0f", low_good=True),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(5,1fr)",
                  "gap": "12px", "marginBottom": "20px"}),

        # Distribution row
        html.Div([
            html.Div([
                html.H4("Competition distribution", style={"fontSize": "13px", "margin": "0 0 6px 0", "color": "#333"}),
                dcc.Graph(figure=dist_chart(result["competition"]["samples"],
                                            "Offers received", COL_MID,
                                            vline=comp_mean),
                          config={"displayModeBar": False}),
            ], style={"flex": "1"}),
            html.Div([
                html.H4("Price ratio distribution", style={"fontSize": "13px", "margin": "0 0 6px 0", "color": "#333"}),
                dcc.Graph(figure=dist_chart(result["price_ratio"]["samples"],
                                            "Award / Estimate", COL_ACCENT),
                          config={"displayModeBar": False}),
            ], style={"flex": "1"}),
            html.Div([
                html.H4("Procedure duration", style={"fontSize": "13px", "margin": "0 0 6px 0", "color": "#333"}),
                dcc.Graph(figure=dist_chart(result["duration"]["samples"],
                                            "Days", COL_GREEN,
                                            vline=dur_median),
                          config={"displayModeBar": False}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),

        # Benchmark note
        html.Div([
            html.Span(f"Empirical benchmark: {bench['n_records']:,} historical procedures matching country={country}, "
                      f"procedure={proc}, CPV={cpv}",
                      style={"fontSize": "11px", "color": COL_GREY, "fontStyle": "italic"}),
        ], style={"marginTop": "12px"}),
    ])


# ══════════════════════════════════════════════════════════════════
# TAB 2: SCENARIO COMPARATOR
# ══════════════════════════════════════════════════════════════════
def comparator_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.H3("Scenario A", style={"color": COL_BLUE, "marginTop":"0",
                                              "borderBottom": f"3px solid {COL_MID}", "paddingBottom": "8px"}),
                build_form("ca", col_country="DE", col_proc="OPE", col_crit="L",
                           col_val=2_000_000, col_cpv="45"),
            ], style={"flex": "1", "backgroundColor": "white", "padding": "20px",
                      "borderRadius": "8px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}),

            html.Div([
                html.H3("Scenario B", style={"color": COL_ACCENT, "marginTop":"0",
                                              "borderBottom": f"3px solid {COL_ACCENT}", "paddingBottom": "8px"}),
                build_form("cb", col_country="DE", col_proc="OPE", col_crit="M",
                           col_val=2_000_000, col_cpv="45"),
            ], style={"flex": "1", "backgroundColor": "white", "padding": "20px",
                      "borderRadius": "8px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                      "marginLeft": "12px"}),
        ], style={"display": "flex", "padding": "20px 20px 0 20px"}),

        html.Div([
            html.Button("Compare Scenarios", id="compare-btn", n_clicks=0,
                        style={"padding": "12px 32px", "fontSize": "16px",
                               "backgroundColor": COL_BLUE, "color": "white",
                               "border": "none", "borderRadius": "6px", "cursor": "pointer",
                               "boxShadow": "0 2px 6px rgba(0,0,0,0.2)"}),
        ], style={"textAlign": "center", "padding": "16px"}),

        html.Div(id="compare-results", style={"padding": "0 20px 20px 20px"}),
    ])


@app.callback(
    Output("compare-results", "children"),
    Input("compare-btn", "n_clicks"),
    [State("ca-country","value"),State("ca-proc","value"),State("ca-ctype","value"),
     State("ca-cpv","value"),   State("ca-crit","value"), State("ca-val","value"),
     State("ca-prep","value"),  State("ca-dur","value"),  State("ca-pw","value"),
     State("ca-flags","value"),
     State("cb-country","value"),State("cb-proc","value"),State("cb-ctype","value"),
     State("cb-cpv","value"),   State("cb-crit","value"), State("cb-val","value"),
     State("cb-prep","value"),  State("cb-dur","value"),  State("cb-pw","value"),
     State("cb-flags","value")],
    prevent_initial_call=True
)
def run_compare(n,
                a_country,a_proc,a_ctype,a_cpv,a_crit,a_val,a_prep,a_dur,a_pw,a_flags,
                b_country,b_proc,b_ctype,b_cpv,b_crit,b_val,b_prep,b_dur,b_pw,b_flags):
    pa = form_to_params(a_country,a_proc,a_ctype,a_cpv,a_crit,a_val,a_prep,a_dur,a_pw,a_flags)
    pb = form_to_params(b_country,b_proc,b_ctype,b_cpv,b_crit,b_val,b_prep,b_dur,b_pw,b_flags)
    comp = twin.compare(pa, pb, label_a="Scenario A", label_b="Scenario B")

    metrics = [
        ("Expected bids",        "competition",     "mean",        "",    ".2f", False),
        ("Single-bid risk",      "single_bid_risk", "probability", "%",   ".1%", True),
        ("Cross-border win",     "cross_border",    "probability", "%",   ".1%", False),
        ("Price ratio",          "price_ratio",     "mean",        "",    ".3f", True),
        ("Procedure duration",   "duration",        "mean",        "d",   ".0f", True),
    ]

    rows = []
    for label, key, subkey, unit, fmt, low_better in metrics:
        d = comp["deltas"][key]
        a_val_ = d["a"]
        b_val_ = d["b"]
        delta  = d["delta"]
        mul    = 100 if unit == "%" else 1
        better = delta < 0 if low_better else delta > 0
        arrow  = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        bg_delta = "#E2EFDA" if better else "#FCE4EC"
        col_delta = COL_GREEN if better else COL_RED

        rows.append(html.Tr([
            html.Td(label, style={"padding":"10px 14px","fontWeight":"bold","fontSize":"13px"}),
            html.Td(f"{a_val_*mul:{fmt}}{'' if unit!='%' else ''}{unit if unit!='%' else ''}",
                    style={"padding":"10px 14px","textAlign":"center","color": COL_MID, "fontWeight":"600"}),
            html.Td(f"{b_val_*mul:{fmt}}{'' if unit!='%' else ''}{unit if unit!='%' else ''}",
                    style={"padding":"10px 14px","textAlign":"center","color": COL_ACCENT,"fontWeight":"600"}),
            html.Td(f"{arrow} {abs(delta)*mul:{fmt}}{unit if unit!='%' else ''}",
                    style={"padding":"10px 14px","textAlign":"center","fontWeight":"bold",
                           "color": col_delta, "backgroundColor": bg_delta}),
        ]))

    # Overlay distribution charts
    def overlay_dist(key, label, title):
        sa = comp["scenario_a"][key]["samples"]
        sb = comp["scenario_b"][key]["samples"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sa, nbinsx=35, name="Scenario A",
                                    marker_color=COL_MID, opacity=0.6))
        fig.add_trace(go.Histogram(x=sb, nbinsx=35, name="Scenario B",
                                    marker_color=COL_ACCENT, opacity=0.6))
        fig.update_layout(barmode="overlay", title=dict(text=title, font=dict(size=13)),
                          height=200, margin=dict(t=35,b=25,l=30,r=10),
                          paper_bgcolor="white", plot_bgcolor="white",
                          legend=dict(font=dict(size=10)),
                          xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"))
        return fig

    return html.Div([
        html.H3("Comparison Results", style={"color": COL_BLUE,
                "borderBottom": f"2px solid {COL_LIGHT}", "paddingBottom": "8px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Outcome Metric",  style={"padding":"10px 14px","backgroundColor":COL_BLUE,"color":"white","textAlign":"left"}),
                html.Th("Scenario A",      style={"padding":"10px 14px","backgroundColor":COL_MID, "color":"white","textAlign":"center"}),
                html.Th("Scenario B",      style={"padding":"10px 14px","backgroundColor":COL_ACCENT,"color":"white","textAlign":"center"}),
                html.Th("Difference (B−A)",style={"padding":"10px 14px","backgroundColor":"#555","color":"white","textAlign":"center"}),
            ])),
            html.Tbody(rows)
        ], style={"width":"100%","borderCollapse":"collapse","marginBottom":"20px",
                  "boxShadow":"0 1px 4px rgba(0,0,0,0.1)","backgroundColor":"white"}),

        html.Div([
            html.Div([dcc.Graph(figure=overlay_dist("competition","Offers","Competition (offers)"),
                                config={"displayModeBar":False})], style={"flex":"1"}),
            html.Div([dcc.Graph(figure=overlay_dist("price_ratio","Ratio","Price ratio"),
                                config={"displayModeBar":False})], style={"flex":"1"}),
            html.Div([dcc.Graph(figure=overlay_dist("duration","Days","Duration (days)"),
                                config={"displayModeBar":False})], style={"flex":"1"}),
        ], style={"display":"flex","gap":"12px","backgroundColor":"white",
                  "padding":"16px","borderRadius":"8px","boxShadow":"0 1px 4px rgba(0,0,0,0.1)"}),
    ])


# ══════════════════════════════════════════════════════════════════
# TAB 3: POLICY EXPLORER
# ══════════════════════════════════════════════════════════════════
def explorer_layout():
    feat_path = os.path.join(os.path.dirname(__file__), "../../data/features/procedure_records.parquet")
    df = pl.read_parquet(feat_path).to_pandas()
    countries_avail = sorted(df["ISO_COUNTRY_CODE"].dropna().unique().tolist())
    proc_avail = sorted(df["TOP_TYPE"].dropna().unique().tolist())

    return html.Div([
        html.Div([
            # Filter panel
            html.Div([
                html.H3("Filter historical data", style={"color":COL_BLUE,"marginTop":"0"}),
                html.Label("Country"), dcc.Dropdown(id="ex-country", multi=True,
                    options=[{"label":c,"value":c} for c in countries_avail],
                    value=["DE","FR","PL"], style={"marginBottom":"12px"}),
                html.Label("Procedure type"), dcc.Dropdown(id="ex-proc", multi=True,
                    options=[{"label":p,"value":p} for p in proc_avail],
                    value=["OPE"], style={"marginBottom":"12px"}),
                html.Label("CPV sector"), dcc.Dropdown(id="ex-cpv", multi=True,
                    options=CPV_OPTIONS, value=[], style={"marginBottom":"12px"}),
                html.Label("Years"), dcc.RangeSlider(id="ex-years", min=2018, max=2023, step=1,
                    value=[2018,2023], marks={y:str(y) for y in range(2018,2024)},
                    tooltip={"placement":"bottom"}),
                html.Div(style={"height":"16px"}),
                html.Label("Outcome to explore"),
                dcc.RadioItems(id="ex-outcome",
                    options=[
                        {"label":"Competition (offers received)","value":"n_offers"},
                        {"label":"Single-bid rate",              "value":"single_bid_flag"},
                        {"label":"Cross-border win rate",        "value":"cross_border_win"},
                        {"label":"Price ratio",                  "value":"price_ratio"},
                        {"label":"Procedure duration (days)",    "value":"proc_duration_days"},
                    ],
                    value="n_offers",
                    className="radio-block",
                    style={"marginTop":"8px"}
                ),
                html.Button("Explore", id="ex-btn", n_clicks=0,
                            style={"marginTop":"16px","padding":"10px 28px",
                                   "backgroundColor":COL_BLUE,"color":"white",
                                   "border":"none","borderRadius":"6px","cursor":"pointer"}),
            ], style={"width":"300px","flexShrink":"0","backgroundColor":"white",
                      "padding":"20px","borderRadius":"8px","boxShadow":"0 2px 8px rgba(0,0,0,0.08)"}),

            # Charts panel
            html.Div(id="explorer-results",
                     children=html.P("Apply filters and click Explore.",
                                     style={"color":COL_GREY,"textAlign":"center","marginTop":"40px"}),
                     style={"flex":"1","backgroundColor":"white","padding":"20px",
                            "borderRadius":"8px","boxShadow":"0 2px 8px rgba(0,0,0,0.08)",
                            "marginLeft":"16px"}),
        ], style={"display":"flex","gap":"0","padding":"20px","alignItems":"flex-start"}),
    ])


@app.callback(
    Output("explorer-results","children"),
    Input("ex-btn","n_clicks"),
    [State("ex-country","value"), State("ex-proc","value"), State("ex-cpv","value"),
     State("ex-years","value"),   State("ex-outcome","value")],
    prevent_initial_call=True
)
def run_explorer(n, countries, procs, cpvs, years, outcome):
    feat_path = os.path.join(os.path.dirname(__file__), "../../data/features/procedure_records.parquet")
    df = pl.read_parquet(feat_path).to_pandas()

    if countries: df = df[df["ISO_COUNTRY_CODE"].isin(countries)]
    if procs:     df = df[df["TOP_TYPE"].isin(procs)]
    if cpvs:      df = df[df["cpv_division"].isin(cpvs)]
    df = df[df["YEAR"].between(years[0], years[1])]
    df = df[df[outcome].notna()]

    n_records = len(df)
    if n_records == 0:
        return html.P("No records match these filters.", style={"color": COL_RED})

    outcome_labels = {
        "n_offers":          "Offers received",
        "single_bid_flag":   "Single-bid flag (0/1)",
        "cross_border_win":  "Cross-border win (0/1)",
        "price_ratio":       "Price ratio (award/estimate)",
        "proc_duration_days":"Procedure duration (days)",
    }
    label = outcome_labels.get(outcome, outcome)

    # Overall distribution
    fig_hist = px.histogram(df, x=outcome, nbins=40,
                            color_discrete_sequence=[COL_MID],
                            labels={outcome: label},
                            title=f"Distribution of {label}  (n={n_records:,})")
    fig_hist.update_layout(height=260, margin=dict(t=45,b=30,l=35,r=10),
                           paper_bgcolor="white", plot_bgcolor="white",
                           bargap=0.05, xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"))

    # By country (median)
    by_country = (df.groupby("ISO_COUNTRY_CODE")[outcome].median()
                    .reset_index().rename(columns={outcome:"median_val"})
                    .sort_values("median_val", ascending=False).head(20))
    fig_country = px.bar(by_country, x="ISO_COUNTRY_CODE", y="median_val",
                          color="median_val", color_continuous_scale="Blues",
                          labels={"ISO_COUNTRY_CODE":"Country","median_val":f"Median {label}"},
                          title=f"Median {label} by country")
    fig_country.update_layout(height=280, margin=dict(t=45,b=30,l=35,r=10),
                              paper_bgcolor="white", plot_bgcolor="white",
                              coloraxis_showscale=False,
                              xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"))

    # By year (trend)
    by_year = (df.groupby("YEAR")[outcome].median()
                 .reset_index().rename(columns={outcome:"median_val"}))
    fig_year = px.line(by_year, x="YEAR", y="median_val",
                        markers=True, color_discrete_sequence=[COL_MID],
                        labels={"YEAR":"Year","median_val":f"Median {label}"},
                        title=f"Trend: {label} by year")
    fig_year.update_layout(height=240, margin=dict(t=45,b=30,l=35,r=10),
                           paper_bgcolor="white", plot_bgcolor="white",
                           xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"))

    # Summary stats
    s = df[outcome]
    stats_row = html.Div([
        html.Div([html.Div(f"{s.median():.2f}", style={"fontSize":"24px","fontWeight":"bold","color":COL_MID}),
                  html.Div("Median",style={"fontSize":"11px","color":"#666"})],
                 style={"textAlign":"center","padding":"12px","backgroundColor":"#F7F9FC","borderRadius":"6px"}),
        html.Div([html.Div(f"{s.mean():.2f}",   style={"fontSize":"24px","fontWeight":"bold","color":COL_ACCENT}),
                  html.Div("Mean",  style={"fontSize":"11px","color":"#666"})],
                 style={"textAlign":"center","padding":"12px","backgroundColor":"#F7F9FC","borderRadius":"6px"}),
        html.Div([html.Div(f"{s.quantile(0.25):.2f}", style={"fontSize":"24px","fontWeight":"bold","color":"#555"}),
                  html.Div("P25",   style={"fontSize":"11px","color":"#666"})],
                 style={"textAlign":"center","padding":"12px","backgroundColor":"#F7F9FC","borderRadius":"6px"}),
        html.Div([html.Div(f"{s.quantile(0.75):.2f}", style={"fontSize":"24px","fontWeight":"bold","color":"#555"}),
                  html.Div("P75",   style={"fontSize":"11px","color":"#666"})],
                 style={"textAlign":"center","padding":"12px","backgroundColor":"#F7F9FC","borderRadius":"6px"}),
        html.Div([html.Div(f"{n_records:,}",    style={"fontSize":"24px","fontWeight":"bold","color":COL_GREEN}),
                  html.Div("Records",style={"fontSize":"11px","color":"#666"})],
                 style={"textAlign":"center","padding":"12px","backgroundColor":"#F7F9FC","borderRadius":"6px"}),
    ], style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)","gap":"10px","marginBottom":"16px"})

    return html.Div([
        html.H3(f"Results: {label}", style={"color":COL_BLUE,"marginTop":"0",
                 "borderBottom":f"2px solid {COL_LIGHT}","paddingBottom":"8px"}),
        stats_row,
        html.Div([dcc.Graph(figure=fig_hist,   config={"displayModeBar":False})], style={"marginBottom":"12px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_country,config={"displayModeBar":False})], style={"flex":"1"}),
            html.Div([dcc.Graph(figure=fig_year,   config={"displayModeBar":False})], style={"flex":"1"}),
        ], style={"display":"flex","gap":"12px"}),
    ])


# ══════════════════════════════════════════════════════════════════
# CSS (injected inline for standalone operation)
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
  body { margin: 0; font-family: Arial, sans-serif; }
  .form-label { display: block; font-size: 12px; font-weight: 600;
                color: #444; margin: 0 0 4px 0; }
  .form-group { margin-bottom: 14px; }
  .input-number { border: 1px solid #CCC; border-radius: 4px;
                  font-size: 14px; font-family: Arial; }
  .btn-primary { width: 100%; padding: 10px; background: #1F3864;
                 color: white; border: none; border-radius: 6px;
                 font-size: 15px; cursor: pointer; margin-top: 8px; }
  .btn-primary:hover { background: #2E75B6; }
  .radio-inline label { margin-right: 12px; font-size: 13px; }
  .radio-block label { display: block; margin-bottom: 6px; font-size: 13px; }
  .checklist label { font-size: 12px; }
  .tab-bar .tab { font-size: 14px; padding: 10px 20px; }
  .Select-control { font-size: 13px !important; }
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
    print("PROCUREMENT DIGITAL TWIN — DASH APPLICATION")
    print("="*60)
    print("\nStarting server at http://localhost:8050")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
