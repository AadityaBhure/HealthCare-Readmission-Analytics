"""Plotly figure builders for the dashboard (dark navy theme).

Conventions: one y-axis per chart (no dual axes); colour follows the entity (model, measure,
coding system); hairline recessive grids; hover tooltips on every mark; units in axis titles.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config
from .modeling import MODEL_LABELS
from .theme import (CODING_COLORS_DARK, DARK, DIVERGING_DARK, FONT_FAMILY, MEASURE_COLORS_DARK,
                    MODEL_COLORS_DARK, MODEL_SYMBOLS, RATING_COLORS, SEQUENTIAL_DARK, SERIES_DARK)

PCT_HOVER = "%{y:.2f}%"


def base_layout(fig: go.Figure, height: int = 360, title: str | None = None, legend: bool = True,
                margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=15, color=DARK["text"]), x=0, xanchor="left", y=0.98) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=12, color=DARK["text_2"]),
        margin=margin or dict(l=8, r=12, t=44 if title else 16, b=8),
        hoverlabel=dict(bgcolor=DARK["surface_2"], bordercolor=DARK["axis"], font=dict(color=DARK["text"],
                                                                                         family=FONT_FAMILY)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=DARK["text_2"])) if legend else None,
        showlegend=legend,
        colorway=SERIES_DARK,
    )
    fig.update_xaxes(gridcolor=DARK["grid"], linecolor=DARK["axis"], zeroline=False, tickfont=dict(color=DARK["muted"]),
                     title_font=dict(color=DARK["text_2"]), showline=True)
    fig.update_yaxes(gridcolor=DARK["grid"], linecolor=DARK["axis"], zeroline=False, tickfont=dict(color=DARK["muted"]),
                     title_font=dict(color=DARK["text_2"]))
    return fig


@lru_cache(maxsize=1)
def county_geojson() -> dict:
    return json.loads(config.COUNTY_GEOJSON.read_text(encoding="utf-8"))


CA_LON = [-124.6, -113.9]
CA_LAT = [32.3, 42.2]


def _geo_layout(fig: go.Figure) -> None:
    # Explicit California bounds (fitbounds is unreliable when outline and marker traces are combined).
    fig.update_geos(visible=False, bgcolor="rgba(0,0,0,0)", projection_type="mercator",
                    lonaxis_range=CA_LON, lataxis_range=CA_LAT, showframe=False)


# ======================================================================================
# Overview
# ======================================================================================
def statewide_allcause_trend(state: pd.DataFrame, height: int = 340) -> go.Figure:
    fig = go.Figure()
    for code in ("ICD-9", "ICD-10"):
        g = state[state["coding_system"] == code]
        partial = g["coding_transition_year"]
        fig.add_trace(go.Scatter(
            x=g["year"], y=g["rate"], mode="lines+markers", name=f"{code} era",
            line=dict(color=CODING_COLORS_DARK[code], width=2.5),
            marker=dict(size=8, color=[DARK["page"] if p else CODING_COLORS_DARK[code] for p in partial],
                        line=dict(color=CODING_COLORS_DARK[code], width=2)),
            customdata=np.stack([g["admits"], g["readmits"], np.where(partial, "partial year (Jan-Sep)", "full year")], axis=-1),
            hovertemplate="<b>%{x}</b> (" + code + ")<br>Readmission rate %{y:.2f}%<br>"
                          "Admissions %{customdata[0]:,.0f}<br>Readmissions %{customdata[1]:,.0f}<br>%{customdata[2]}<extra></extra>"))
    fig.add_vrect(x0=2015.5, x1=2015.5, line=dict(color=DARK["muted"], width=1, dash="dot"))
    fig.add_annotation(x=2015.5, y=1, yref="paper", text="ICD-9 to ICD-10 coding change", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=11, color=DARK["muted"]), xshift=4)
    base_layout(fig, height)
    fig.update_yaxes(title="30-day readmission rate (%)", ticksuffix="%")
    fig.update_xaxes(title=None, dtick=1)
    return fig


def coverage_timeline(cabg_sw: pd.DataFrame, stroke_sw: pd.DataFrame, state: pd.DataFrame,
                      cabg_coverage: dict[str, int], height: int = 420) -> go.Figure:
    """Each published reporting period as a bar spanning its calendar years."""
    rows = []
    readm = cabg_sw[cabg_sw["measure_code"] == "readmission"]
    for _, r in readm.iterrows():
        n = cabg_coverage.get(r["period"], 0)
        rows.append(("CABG readmission " + r["period"], r["period_start"], r["period_years"], SERIES_DARK[0],
                     f"{n} hospitals", 0.35 if n < 10 else 1.0))
    mort = cabg_sw[cabg_sw["measure_code"] == "iso_mortality"]
    for _, r in mort.iterrows():
        rows.append(("CABG op. mortality (single-year)", r["period_start"], 1, SERIES_DARK[5],
                     f"{int(r['hospitals'])} hospitals", 1.0))
    for _, r in stroke_sw[stroke_sw["measure_code"] == "readmission"].iterrows():
        rows.append(("Stroke outcomes " + r["period"], r["period_start"], 2, SERIES_DARK[1],
                     f"{int(r['cases']):,} statewide cases", 1.0))
    for _, r in state.iterrows():
        label = f"All-cause ({r['coding_system']})"
        col = CODING_COLORS_DARK[r["coding_system"]]
        rows.append((label, r["year"], 1, col, f"{r['admits']:,.0f} admissions"
                     + (" - partial year" if r["coding_transition_year"] else ""), 0.45 if r["coding_transition_year"] else 1.0))
    df = pd.DataFrame(rows, columns=["lane", "start", "years", "color", "info", "opacity"])
    order = list(dict.fromkeys(df["lane"]))
    fig = go.Figure(go.Bar(
        y=df["lane"], x=df["years"] - 0.12, base=df["start"] - 0.44, orientation="h",
        marker=dict(color=df["color"], opacity=df["opacity"], line=dict(width=0), cornerradius=4),
        customdata=np.stack([df["start"], df["start"] + df["years"] - 1, df["info"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}-%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        showlegend=False, width=0.62))
    base_layout(fig, height, legend=False)
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], showgrid=False, tickfont=dict(size=11))
    fig.update_xaxes(dtick=1, range=[2010.4, 2024.6], title="Calendar year covered")
    return fig


def condition_trends(cabg_sw: pd.DataFrame, stroke_sw: pd.DataFrame, height: int = 330) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        "Isolated CABG 30-day readmission (statewide, observed)", "Ischemic stroke 30-day outcomes (statewide, risk-adjusted)"))
    c = cabg_sw[(cabg_sw["measure_code"] == "readmission")]
    sparse = c["hospitals"] < 10
    fig.add_trace(go.Scatter(x=c["period"], y=c["rate"], mode="lines+markers+text", name="CABG readmission",
                             line=dict(color=SERIES_DARK[0], width=2.5),
                             marker=dict(size=9, color=[DARK["page"] if s else SERIES_DARK[0] for s in sparse],
                                         line=dict(color=SERIES_DARK[0], width=2)),
                             text=[f"{v:.2f}%" for v in c["rate"]], textposition="top center",
                             textfont=dict(color=DARK["text_2"], size=11),
                             customdata=np.stack([c["hospitals"], c["cases"]], axis=-1),
                             hovertemplate="<b>%{x}</b><br>%{y:.2f}%<br>%{customdata[0]} hospitals reported<br>"
                                           "%{customdata[1]:,.0f} eligible cases<extra></extra>"), 1, 1)
    for m, color in MEASURE_COLORS_DARK.items():
        s = stroke_sw[stroke_sw["measure"] == m]
        fig.add_trace(go.Scatter(x=s["period"], y=s["risk_adjusted_rate"], mode="lines+markers", name=f"Stroke {m.lower()}",
                                 line=dict(color=color, width=2.5), marker=dict(size=8),
                                 customdata=np.stack([s["events"], s["cases"]], axis=-1),
                                 hovertemplate="<b>%{x}</b> " + m + "<br>%{y:.2f}%<br>%{customdata[0]:,.0f} events / "
                                               "%{customdata[1]:,.0f} cases<extra></extra>"), 1, 2)
    base_layout(fig, height)
    fig.update_yaxes(ticksuffix="%", title="Rate (%)")
    fig.update_annotations(font=dict(size=12, color=DARK["text_2"]))
    fig.update_layout(legend=dict(y=-0.18, yanchor="top"), margin=dict(l=8, r=12, t=36, b=8))
    return fig


# ======================================================================================
# CABG hospital page
# ======================================================================================
def hospital_history(hist: pd.DataFrame, statewide: pd.DataFrame, estimates: pd.DataFrame | None,
                     target_period: str | None, actual_next: float | None, height: int = 380) -> go.Figure:
    fig = go.Figure()
    sw = statewide[statewide["period"].isin(hist["period"].tolist() + ([target_period] if target_period else []))]
    fig.add_trace(go.Scatter(x=sw["period"], y=sw["rate"], mode="lines", name="Statewide observed rate",
                             line=dict(color=DARK["muted"], width=1.5, dash="dot"),
                             hovertemplate="Statewide %{x}: %{y:.2f}%<extra></extra>"))
    h = hist.dropna(subset=["readmit_rate"])
    fig.add_trace(go.Scatter(
        x=h["period"], y=h["readmit_rate"], mode="lines+markers", name="Observed hospital RAAER",
        line=dict(color=DARK["text"], width=2.5), marker=dict(size=9, color=DARK["text"]),
        error_y=dict(type="data", symmetric=False, array=h["readmit_ci_high"] - h["readmit_rate"],
                     arrayminus=h["readmit_rate"] - h["readmit_ci_low"], color=DARK["axis"], thickness=1.2, width=4),
        customdata=np.stack([h["readmit_ci_low"], h["readmit_ci_high"], h["readmit_cases"], h["readmit_rating"].fillna("-")], axis=-1),
        hovertemplate="<b>%{x}</b><br>Risk-adjusted rate %{y:.2f}%<br>95% CI %{customdata[0]:.2f}-%{customdata[1]:.2f}%<br>"
                      "%{customdata[2]:,.0f} eligible cases<br>Rating: %{customdata[3]}<extra></extra>"))
    if estimates is not None and target_period:
        last = h[h["period"] < target_period].tail(1)
        for _, r in estimates.iterrows():
            color = MODEL_COLORS_DARK[r["model"]]
            if r["model"] == "mean_baseline":
                continue
            fig.add_trace(go.Scatter(
                x=[target_period], y=[r["estimate"]], mode="markers", name=f"Estimate - {r['label']}",
                marker=dict(size=13 if r["recommended"] else 11, symbol=MODEL_SYMBOLS[r["model"]], color=color,
                            line=dict(color=DARK["page"], width=2)),
                hovertemplate=f"<b>{r['label']}</b>{' (recommended)' if r['recommended'] else ''}<br>"
                              f"Estimated {target_period} rate: %{{y:.2f}}%<extra></extra>"))
            if r["recommended"] and not last.empty:
                fig.add_trace(go.Scatter(x=[last["period"].iloc[0], target_period],
                                         y=[last["readmit_rate"].iloc[0], r["estimate"]], mode="lines",
                                         line=dict(color=color, width=1.5, dash="dash"), showlegend=False,
                                         hoverinfo="skip"))
        if actual_next is not None and not np.isnan(actual_next):
            fig.add_trace(go.Scatter(x=[target_period], y=[actual_next], mode="markers", name="Actual (published)",
                                     marker=dict(size=14, symbol="star", color=SERIES_DARK[3],
                                                 line=dict(color=DARK["page"], width=1.5)),
                                     hovertemplate="Published actual %{x}: %{y:.2f}%<extra></extra>"))
    order = sorted(set(hist["period"]) | ({target_period} if target_period else set()) | set(sw["period"]))
    base_layout(fig, height)
    fig.update_xaxes(categoryorder="array", categoryarray=order, title="Two-year reporting window")
    fig.update_yaxes(title="Isolated CABG 30-day readmission RAAER (%)", ticksuffix="%", rangemode="tozero")
    fig.update_layout(legend=dict(y=-0.25, yanchor="top"), margin=dict(l=8, r=12, t=16, b=8))
    return fig


def volume_history(hist: pd.DataFrame, height: int = 250) -> go.Figure:
    fig = go.Figure(go.Bar(x=hist["period"], y=hist["cabg_volume"], marker=dict(color=SERIES_DARK[5], cornerradius=4),
                           text=[f"{v:,.0f}" if pd.notna(v) else "" for v in hist["cabg_volume"]],
                           textposition="outside", textfont=dict(color=DARK["text_2"]),
                           hovertemplate="%{x}<br>%{y:,.0f} CABG cases<extra></extra>", name="CABG cases"))
    base_layout(fig, height, legend=False)
    fig.update_yaxes(title="Total CABG cases", rangemode="tozero")
    fig.update_xaxes(title=None)
    return fig


def contribution_chart(contrib: pd.DataFrame, intercept: float, labels: dict[str, str], height: int = 260) -> go.Figure:
    c = contrib.copy()
    c["label"] = c["feature"].map(lambda f: labels.get(f, f))
    c = c.sort_values("contribution")
    colors = [DIVERGING_DARK["pos"] if v > 0 else DIVERGING_DARK["neg"] for v in c["contribution"]]
    fig = go.Figure(go.Bar(y=c["label"], x=c["contribution"], orientation="h",
                           marker=dict(color=colors, cornerradius=4),
                           text=[f"{v:+.2f}" for v in c["contribution"]], textposition="outside", cliponaxis=False,
                           textfont=dict(color=DARK["text_2"]),
                           hovertemplate="%{y}<br>Contribution %{x:+.2f} percentage points<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=DARK["axis"], width=1))
    base_layout(fig, height, legend=False)
    lim = max(0.5, float(np.abs(c["contribution"]).max()) * 1.7)
    fig.update_xaxes(title=f"pp vs average hospital ({intercept:.2f}%)", range=[-lim, lim], zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


def distribution_marker(values: pd.Series, marker_value: float | None, marker_label: str, xlabel: str,
                        statewide: float | None = None, height: int = 260) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=24, marker=dict(color=SERIES_DARK[5], line=dict(color=DARK["page"], width=1)),
                                 hovertemplate="%{x}<br>%{y} hospitals<extra></extra>", name="Hospitals"))
    if statewide is not None:
        fig.add_vline(x=statewide, line=dict(color=DARK["muted"], width=1.5, dash="dot"))
        fig.add_annotation(x=statewide, y=1, yref="paper", text=f"Statewide {statewide:.2f}%", showarrow=False,
                           xanchor="right", yanchor="top", font=dict(size=11, color=DARK["muted"]), xshift=-4)
    if marker_value is not None and not np.isnan(marker_value):
        fig.add_vline(x=marker_value, line=dict(color=SERIES_DARK[3], width=2.5))
        fig.add_annotation(x=marker_value, y=0.68, yref="paper", text=marker_label, showarrow=False, xanchor="left",
                           font=dict(size=11, color=SERIES_DARK[3]), xshift=5)
    base_layout(fig, height, legend=False)
    fig.update_xaxes(title=xlabel, ticksuffix="%")
    fig.update_yaxes(title="Hospitals")
    return fig


def hospital_map(df: pd.DataFrame, value_col: str, value_label: str, highlight_id: str | None = None,
                 height: int = 430) -> go.Figure:
    gj = county_geojson()
    names = [f["properties"]["NAME"] for f in gj["features"]]
    fig = go.Figure(go.Choropleth(geojson=gj, locations=names, z=[0] * len(names), featureidkey="properties.NAME",
                                  colorscale=[[0, DARK["surface_2"]], [1, DARK["surface_2"]]], showscale=False,
                                  marker_line_color=DARK["axis"], marker_line_width=0.6, hoverinfo="skip"))
    d = df.dropna(subset=[value_col, "latitude", "longitude"])
    size = np.clip(np.sqrt(d["readmit_cases"].fillna(50)) * 1.1, 6, 26)
    fig.add_trace(go.Scattergeo(
        lat=d["latitude"], lon=d["longitude"], mode="markers", name="Hospitals",
        marker=dict(size=size, color=d[value_col], colorscale=SEQUENTIAL_DARK, cmin=float(d[value_col].quantile(.02)),
                    cmax=float(d[value_col].quantile(.98)), line=dict(color=DARK["page"], width=1), opacity=0.92,
                    colorbar=dict(title=dict(text=value_label, font=dict(color=DARK["text_2"], size=11)), ticksuffix="%",
                                  tickfont=dict(color=DARK["muted"]), thickness=10, len=0.7)),
        customdata=np.stack([d["hospital_name"], d["county"], d[value_col], d["readmit_cases"].fillna(0)], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} County<br>RAAER %{customdata[2]:.2f}%<br>"
                      "%{customdata[3]:,.0f} eligible cases<extra></extra>"))
    if highlight_id is not None:
        h = d[d["hospital_id"] == highlight_id]
        if not h.empty:
            fig.add_trace(go.Scattergeo(lat=h["latitude"], lon=h["longitude"], mode="markers", name="Selected hospital",
                                        marker=dict(size=22, symbol="circle-open", color=SERIES_DARK[3],
                                                    line=dict(width=3, color=SERIES_DARK[3])), hoverinfo="skip"))
    _geo_layout(fig)
    base_layout(fig, height, legend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig


# ======================================================================================
# Stroke page
# ======================================================================================
def stroke_statewide(sw: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for m, color in MEASURE_COLORS_DARK.items():
        s = sw[sw["measure"] == m]
        fig.add_trace(go.Scatter(x=s["period"], y=s["risk_adjusted_rate"], mode="lines+markers+text", name=m,
                                 line=dict(color=color, width=2.5), marker=dict(size=9),
                                 text=[f"{v:.2f}%" for v in s["risk_adjusted_rate"]], textposition="top center",
                                 textfont=dict(color=DARK["text_2"], size=11),
                                 customdata=np.stack([s["events"], s["cases"]], axis=-1),
                                 hovertemplate="<b>%{x}</b> " + m + "<br>%{y:.2f}%<br>%{customdata[0]:,.0f} events / "
                                               "%{customdata[1]:,.0f} cases<extra></extra>"))
    base_layout(fig, height)
    fig.update_yaxes(title="Statewide risk-adjusted rate (%)", ticksuffix="%", range=[8, 14])
    fig.update_xaxes(title="Two-year reporting window")
    return fig


def ranked_hospitals(df: pd.DataFrame, value_col: str, label_col: str, rating_col: str, statewide: float | None,
                     xlabel: str, height: int = 520) -> go.Figure:
    d = df.sort_values(value_col)
    fig = go.Figure()
    for rating in [r for r in ("Better", "As Expected", "Average", "Worse", "Not Applicable") if r in set(d[rating_col].fillna("Not rated"))] + (
            ["Not rated"] if d[rating_col].isna().any() else []):
        sub = d[d[rating_col].fillna("Not rated") == rating]
        fig.add_trace(go.Bar(y=sub[label_col], x=sub[value_col], orientation="h", name=f"Rated: {rating}",
                             marker=dict(color=RATING_COLORS.get(rating, "#4B5872"), cornerradius=3),
                             customdata=np.stack([sub["cases"].fillna(0), sub.get("county", pd.Series([""] * len(sub))).fillna("")], axis=-1),
                             hovertemplate="<b>%{y}</b><br>%{customdata[1]}<br>%{x:.2f}%<br>%{customdata[0]:,.0f} cases<extra></extra>"))
    if statewide is not None:
        fig.add_vline(x=statewide, line=dict(color=DARK["text_2"], width=1.5, dash="dot"))
        fig.add_annotation(x=statewide, y=1.0, yref="paper", text=f"Statewide {statewide:.2f}%", showarrow=False,
                           yanchor="bottom", font=dict(size=11, color=DARK["text_2"]))
    base_layout(fig, height)
    fig.update_yaxes(categoryorder="array", categoryarray=d[label_col].tolist(), showgrid=False, tickfont=dict(size=11))
    fig.update_xaxes(title=xlabel, ticksuffix="%", rangemode="tozero")
    fig.update_layout(barmode="overlay", legend=dict(y=-(62 / height), yanchor="top"), margin=dict(l=8, r=12, t=24, b=64))
    return fig


def readmission_vs_mortality(panel: pd.DataFrame, period: str, height: int = 380) -> go.Figure:
    d = panel[panel["period"] == period].dropna(subset=["readmit_rate", "mortality_rate"])
    r = d["readmit_rate"].corr(d["mortality_rate"])
    fig = go.Figure(go.Scatter(
        x=d["mortality_rate"], y=d["readmit_rate"], mode="markers", name="Hospitals",
        marker=dict(size=np.clip(np.sqrt(d["readmit_cases"].fillna(10)) * 0.9, 5, 22), color=SERIES_DARK[0], opacity=0.75,
                    line=dict(color=DARK["page"], width=1)),
        customdata=np.stack([d["hospital_name"], d["county"].fillna(""), d["readmit_cases"].fillna(0)], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Mortality %{x:.2f}%<br>Readmission %{y:.2f}%<br>"
                      "%{customdata[2]:,.0f} cases<extra></extra>"))
    fig.add_annotation(x=1, y=1, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                       text=f"Pearson r = {r:+.2f} (n = {len(d)})", font=dict(color=DARK["text"], size=12),
                       bgcolor=DARK["surface_2"], borderpad=6)
    base_layout(fig, height, legend=False)
    fig.update_xaxes(title=f"30-day mortality, risk-adjusted (%) - {period}", ticksuffix="%")
    fig.update_yaxes(title="30-day readmission, risk-adjusted (%)", ticksuffix="%")
    return fig


def hospital_stroke_trend(h: pd.DataFrame, sw: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for m, prefix in (("30-day Readmission", "readmit"), ("30-day Mortality", "mortality")):
        color = MEASURE_COLORS_DARK[m]
        s = sw[sw["measure"] == m]
        fig.add_trace(go.Scatter(x=s["period"], y=s["risk_adjusted_rate"], mode="lines", name=f"Statewide {m.lower()}",
                                 line=dict(color=color, width=1.2, dash="dot"),
                                 hovertemplate="Statewide %{x}: %{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=h["period"], y=h[f"{prefix}_rate"], mode="lines+markers", name=f"Hospital {m.lower()}",
                                 line=dict(color=color, width=2.5), marker=dict(size=9),
                                 customdata=np.stack([h[f"{prefix}_cases"].fillna(0), h[f"{prefix}_rating"].fillna("-")], axis=-1),
                                 hovertemplate="<b>%{x}</b> " + m + "<br>%{y:.2f}%<br>%{customdata[0]:,.0f} cases<br>"
                                               "Rating: %{customdata[1]}<extra></extra>"))
    base_layout(fig, height)
    fig.update_yaxes(title="Risk-adjusted rate (%)", ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(title="Two-year reporting window", categoryorder="category ascending")
    fig.update_layout(legend=dict(y=-0.22, yanchor="top"), margin=dict(l=8, r=12, t=16, b=8))
    return fig


def county_bars(df: pd.DataFrame, value_col: str, label_col: str, xlabel: str, reference: float | None = None,
                ref_label: str = "State", highlight: set | None = None, height: int = 520,
                hover_extra: str = "") -> go.Figure:
    d = df.sort_values(value_col)
    highlight = highlight or set()
    colors = [SERIES_DARK[3] if c in highlight else SERIES_DARK[0] for c in d[label_col]]
    fig = go.Figure(go.Bar(y=d[label_col], x=d[value_col], orientation="h", marker=dict(color=colors, cornerradius=3),
                           customdata=d[["admits"]].fillna(0).to_numpy() if "admits" in d else None,
                           hovertemplate="<b>%{y}</b><br>%{x:.2f}%" + hover_extra + "<extra></extra>"))
    if reference is not None:
        fig.add_vline(x=reference, line=dict(color=DARK["text_2"], width=1.5, dash="dot"))
        fig.add_annotation(x=reference, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"{ref_label} {reference:.2f}%", font=dict(size=11, color=DARK["text_2"]))
    base_layout(fig, height, legend=False)
    fig.update_yaxes(categoryorder="array", categoryarray=d[label_col].tolist(), showgrid=False, tickfont=dict(size=11))
    fig.update_xaxes(title=xlabel, ticksuffix="%", range=[0, float(d[value_col].max()) * 1.08])
    fig.update_layout(margin=dict(l=8, r=12, t=24, b=8))
    return fig


# ======================================================================================
# Population page
# ======================================================================================
def county_choropleth(df: pd.DataFrame, year: int, min_admits: int, height: int = 520) -> go.Figure:
    gj = county_geojson()
    d = df[df["year"] == year]
    ok = d[(d["admits"] >= min_admits) & d["rate"].notna()]
    low = d[~d.index.isin(ok.index)]
    fig = go.Figure()
    if not low.empty:
        fig.add_trace(go.Choropleth(
            geojson=gj, locations=low["county"], z=[0] * len(low), featureidkey="properties.NAME",
            colorscale=[[0, "#2A3346"], [1, "#2A3346"]], showscale=False, marker_line_color=DARK["page"],
            marker_line_width=0.8, name="Below threshold",
            customdata=np.stack([low["admits"].fillna(0), low["rate"].fillna(-1)], axis=-1),
            hovertemplate="<b>%{location}</b><br>Fewer than " + f"{min_admits:,}" +
                          " admissions (%{customdata[0]:,.0f}) - rate not ranked<extra></extra>"))
    fig.add_trace(go.Choropleth(
        geojson=gj, locations=ok["county"], z=ok["rate"], featureidkey="properties.NAME",
        colorscale=SEQUENTIAL_DARK, marker_line_color=DARK["page"], marker_line_width=0.8,
        colorbar=dict(title=dict(text="Readmission<br>rate (%)", font=dict(color=DARK["text_2"], size=11)),
                      ticksuffix="%", tickfont=dict(color=DARK["muted"]), thickness=10, len=0.75),
        customdata=np.stack([ok["admits"], ok["readmits"], ok["coding_system"]], axis=-1),
        hovertemplate="<b>%{location}</b><br>%{z:.2f}% (%{customdata[2]})<br>%{customdata[1]:,.0f} readmissions / "
                      "%{customdata[0]:,.0f} admissions<extra></extra>"))
    _geo_layout(fig)
    base_layout(fig, height, legend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def county_trends(df: pd.DataFrame, state: pd.DataFrame, counties: list[str], color_map: dict[str, str],
                  height: int = 380) -> go.Figure:
    fig = go.Figure()
    for code in ("ICD-9", "ICD-10"):
        s = state[state["coding_system"] == code]
        fig.add_trace(go.Scatter(x=s["year"], y=s["rate"], mode="lines", name="California" if code == "ICD-10" else None,
                                 showlegend=code == "ICD-10", legendgroup="state",
                                 line=dict(color=DARK["text"], width=3),
                                 hovertemplate="California %{x}: %{y:.2f}%<extra></extra>"))
    for c in counties:
        g = df[df["county"] == c]
        for code in ("ICD-9", "ICD-10"):
            s = g[g["coding_system"] == code]
            fig.add_trace(go.Scatter(x=s["year"], y=s["rate"], mode="lines+markers", name=c, legendgroup=c,
                                     showlegend=code == "ICD-10", line=dict(color=color_map[c], width=2), marker=dict(size=6),
                                     customdata=np.stack([s["admits"], s["coding_system"]], axis=-1),
                                     hovertemplate=f"<b>{c}</b> %{{x}}<br>%{{y:.2f}}% (%{{customdata[1]}})<br>"
                                                   "%{customdata[0]:,.0f} admissions<extra></extra>"))
    fig.add_vline(x=2015.5, line=dict(color=DARK["muted"], width=1, dash="dot"))
    base_layout(fig, height)
    fig.update_yaxes(title="30-day readmission rate (%)", ticksuffix="%")
    fig.update_xaxes(dtick=1, title=None)
    return fig


def strata_bars(df: pd.DataFrame, year: int, state_rate: float, height: int = 320) -> go.Figure:
    d = df[df["year"] == year].dropna(subset=["rate"]).sort_values("rate")
    fig = go.Figure(go.Bar(y=d["strata_name"], x=d["rate"], orientation="h", marker=dict(color=SERIES_DARK[0], cornerradius=3),
                           text=[f"{v:.2f}%" for v in d["rate"]], textposition="outside", textfont=dict(color=DARK["text_2"]),
                           customdata=np.stack([d["admits"], d["readmits"]], axis=-1),
                           hovertemplate="<b>%{y}</b><br>%{x:.2f}%<br>%{customdata[1]:,.0f} readmissions / "
                                         "%{customdata[0]:,.0f} admissions<extra></extra>"))
    fig.add_vline(x=state_rate, line=dict(color=DARK["text_2"], width=1.5, dash="dot"))
    fig.add_annotation(x=state_rate, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"All patients {state_rate:.2f}%", font=dict(size=11, color=DARK["text_2"]))
    base_layout(fig, height, legend=False)
    fig.update_xaxes(title=f"30-day readmission rate (%), {year}", ticksuffix="%", range=[0, float(d["rate"].max()) * 1.18])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=8, r=12, t=24, b=8))
    return fig


def strata_small_multiples(df: pd.DataFrame, height: int = 420) -> go.Figure:
    names = sorted(df["strata_name"].unique())
    cols = 3 if len(names) in (5, 6) else min(len(names), 4)
    rows = int(np.ceil(len(names) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, shared_yaxes=True, vertical_spacing=0.16,
                        horizontal_spacing=0.04)
    for i, n in enumerate(names):
        g = df[df["strata_name"] == n]
        for code in ("ICD-9", "ICD-10"):
            s = g[g["coding_system"] == code]
            fig.add_trace(go.Scatter(x=s["year"], y=s["rate"], mode="lines", line=dict(color=CODING_COLORS_DARK[code], width=2),
                                     name=f"{code} era", legendgroup=code, showlegend=i == 0,
                                     hovertemplate=f"<b>{n}</b> %{{x}}<br>%{{y:.2f}}% ({code})<extra></extra>"),
                          row=i // cols + 1, col=i % cols + 1)
    base_layout(fig, height)
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(dtick=4, tickfont=dict(size=10))
    fig.update_annotations(font=dict(size=11, color=DARK["text_2"]))
    fig.update_layout(margin=dict(l=8, r=12, t=28, b=48), legend=dict(y=-(40 / height), yanchor="top"))
    return fig


def readmitted_share_bars(df: pd.DataFrame, year: int, height: int = 320) -> go.Figure:
    d = df[df["year"] == year].dropna(subset=["admits", "readmits"]).sort_values("admits")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d["strata_name"], x=d["readmits"], orientation="h", name="Readmitted within 30 days",
                         marker=dict(color=SERIES_DARK[0], cornerradius=0),
                         hovertemplate="<b>%{y}</b><br>%{x:,.0f} readmissions<extra></extra>"))
    fig.add_trace(go.Bar(y=d["strata_name"], x=d["admits"] - d["readmits"], orientation="h", name="Not readmitted",
                         marker=dict(color="#33415C", cornerradius=3),
                         hovertemplate="<b>%{y}</b><br>%{x:,.0f} index admissions without readmission<extra></extra>"))
    base_layout(fig, height)
    fig.update_layout(barmode="stack", bargap=0.3, legend=dict(y=-0.2, yanchor="top"), margin=dict(l=8, r=12, t=16, b=8))
    fig.update_xaxes(title=f"Index admissions, {year}", tickformat="~s")
    fig.update_yaxes(showgrid=False)
    return fig


# ======================================================================================
# Model evaluation page
# ======================================================================================
def design_comparison(rows: pd.DataFrame, height: int = 360) -> go.Figure:
    """rows: columns design, model, MAE, sd (optional)."""
    fig = go.Figure()
    for m in ("persistence", "linear_regression", "random_forest", "mean_baseline"):
        s = rows[rows["model"] == m]
        if s.empty:
            continue
        fig.add_trace(go.Bar(x=s["design"], y=s["MAE"], name=MODEL_LABELS[m], marker=dict(color=MODEL_COLORS_DARK[m], cornerradius=4),
                             error_y=dict(type="data", array=s["sd"], color=DARK["text_2"], thickness=1.2, width=4) if "sd" in s and s["sd"].notna().any() else None,
                             hovertemplate="<b>" + MODEL_LABELS[m] + "</b><br>%{x}<br>MAE %{y:.3f} pp<extra></extra>"))
    base_layout(fig, height)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title="Mean absolute error (percentage points)", rangemode="tozero")
    fig.update_xaxes(title=None, tickangle=0, tickfont=dict(size=11))
    return fig


def bootstrap_forest(boot: pd.DataFrame, height: int = 260) -> go.Figure:
    d = boot.copy()
    short = {"linear_regression": "Linear Reg.", "random_forest": "Random Forest", "persistence": "Persistence",
             "mean_baseline": "Training mean"}
    d["label"] = d.apply(lambda r: f"{short[r['model']]} - {short[r['reference']]}", axis=1)
    d = d.iloc[::-1]
    colors = [MODEL_COLORS_DARK[m] for m in d["model"]]
    fig = go.Figure(go.Scatter(
        x=d["mae_diff"], y=d["label"], mode="markers", marker=dict(size=12, color=colors, line=dict(color=DARK["page"], width=2)),
        error_x=dict(type="data", symmetric=False, array=d["ci_high"] - d["mae_diff"], arrayminus=d["mae_diff"] - d["ci_low"],
                     color=DARK["text_2"], thickness=1.5, width=6),
        customdata=np.stack([d["ci_low"], d["ci_high"], d["prob_a_better"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>MAE difference %{x:+.3f} pp<br>95% CI [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]"
                      "<br>P(first model better) %{customdata[2]:.0%}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=DARK["muted"], width=1.5))
    fig.add_annotation(x=0, y=1.02, yref="paper", text="left of 0 = first model has lower error", showarrow=False,
                       xanchor="center", yanchor="bottom", font=dict(size=11, color=DARK["muted"]))
    base_layout(fig, height, legend=False)
    fig.update_xaxes(title="Test MAE difference (pp), 95% bootstrap CI")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=8, r=12, t=28, b=8))
    return fig


def actual_vs_predicted(preds: pd.DataFrame, target: str, models: list[str], height: int = 420,
                        unit_label: str = "rate (%)") -> go.Figure:
    lo = float(min(preds[target].min(), *(preds[f"pred_{m}"].min() for m in models))) - 1
    hi = float(max(preds[target].max(), *(preds[f"pred_{m}"].max() for m in models))) + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction (y = x)",
                             line=dict(color=DARK["muted"], width=1.2, dash="dot"), hoverinfo="skip"))
    for m in models:
        fig.add_trace(go.Scatter(
            x=preds[f"pred_{m}"], y=preds[target], mode="markers", name=MODEL_LABELS[m],
            marker=dict(size=8, color=MODEL_COLORS_DARK[m], symbol=MODEL_SYMBOLS.get(m, "circle"), opacity=0.8,
                        line=dict(color=DARK["page"], width=1)),
            customdata=np.stack([preds["hospital_id"].astype(str)], axis=-1),
            hovertemplate="Hospital %{customdata[0]}<br>Predicted %{x:.2f}%<br>Actual %{y:.2f}%<extra>" + MODEL_LABELS[m] + "</extra>"))
    base_layout(fig, height)
    fig.update_xaxes(title=f"Predicted next-window {unit_label}", range=[lo, hi], ticksuffix="%")
    fig.update_yaxes(title=f"Actual next-window {unit_label}", range=[lo, hi], ticksuffix="%", scaleanchor="x", scaleratio=1)
    return fig


def residual_plot(preds: pd.DataFrame, target: str, model: str, x_col: str, x_label: str, log_x: bool = False,
                  height: int = 330) -> go.Figure:
    resid = preds[target] - preds[f"pred_{model}"]
    fig = go.Figure(go.Scatter(x=preds[x_col], y=resid, mode="markers", name="Residual",
                               marker=dict(size=8, color=MODEL_COLORS_DARK[model], opacity=0.8, line=dict(color=DARK["page"], width=1)),
                               customdata=np.stack([preds["hospital_id"].astype(str)], axis=-1),
                               hovertemplate="Hospital %{customdata[0]}<br>" + x_label + " %{x:,.2f}<br>Residual %{y:+.2f} pp<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=DARK["muted"], width=1.2))
    base_layout(fig, height, legend=False)
    fig.update_xaxes(title=x_label, type="log" if log_x else "linear")
    fig.update_yaxes(title="Actual - predicted (percentage points)")
    return fig


def error_by_volume(err: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for m in ("persistence", "linear_regression", "random_forest"):
        s = err[err["model"] == m]
        fig.add_trace(go.Bar(x=s["volume_tercile"].astype(str) + "<br>(" + s["volume_range"] + ")", y=s["MAE"],
                             name=MODEL_LABELS[m], marker=dict(color=MODEL_COLORS_DARK[m], cornerradius=4),
                             hovertemplate="<b>" + MODEL_LABELS[m] + "</b><br>%{x}<br>MAE %{y:.2f} pp<extra></extra>"))
    base_layout(fig, height)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title="Test MAE (percentage points)", rangemode="tozero")
    fig.update_xaxes(title="Target-window cohort size tercile")
    return fig


def permutation_bars(perm: pd.DataFrame, labels: dict[str, str], height: int = 280) -> go.Figure:
    fig = go.Figure()
    for m in ("linear_regression", "random_forest"):
        s = perm[perm["model"] == m]
        fig.add_trace(go.Bar(y=s["feature"].map(lambda f: labels.get(f, f)), x=s["importance_mae_increase"], orientation="h",
                             name=MODEL_LABELS[m], marker=dict(color=MODEL_COLORS_DARK[m], cornerradius=3),
                             error_x=dict(type="data", array=s["importance_sd"], color=DARK["text_2"], thickness=1.2, width=3),
                             hovertemplate="<b>%{y}</b><br>MAE increase when shuffled: %{x:.3f} pp<extra>" + MODEL_LABELS[m] + "</extra>"))
    fig.add_vline(x=0, line=dict(color=DARK["axis"], width=1))
    base_layout(fig, height)
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(title="Increase in test MAE when the feature is shuffled (pp)")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def learned_curve(curve: pd.DataFrame, train: pd.DataFrame, feature: str, target: str, xlabel: str,
                  height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train[feature], y=train[target], mode="markers", name="Training transitions",
                             marker=dict(size=6, color="#4B5872", opacity=0.6),
                             hovertemplate="Previous %{x:.2f}% -> next %{y:.2f}%<extra></extra>"))
    for m in ("persistence", "linear_regression", "random_forest", "mean_baseline"):
        fig.add_trace(go.Scatter(x=curve[feature], y=curve[f"pred_{m}"], mode="lines", name=MODEL_LABELS[m],
                                 line=dict(color=MODEL_COLORS_DARK[m], width=2.5 if m in ("linear_regression", "random_forest") else 1.5,
                                           dash="dot" if m in ("persistence", "mean_baseline") else "solid"),
                                 hovertemplate="Previous %{x:.2f}% -> estimate %{y:.2f}%<extra>" + MODEL_LABELS[m] + "</extra>"))
    base_layout(fig, height)
    fig.update_xaxes(title=xlabel, ticksuffix="%")
    fig.update_yaxes(title="Estimated next-window rate (%)", ticksuffix="%")
    return fig


def synthetic_scatter(preds: pd.DataFrame, model: str, height: int = 360) -> go.Figure:
    lo = float(min(preds["synthetic_label_next_rate"].min(), preds[f"pred_{model}"].min())) - 1
    hi = float(max(preds["synthetic_label_next_rate"].max(), preds[f"pred_{model}"].max())) + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=DARK["muted"], dash="dot", width=1.2),
                             name="y = x", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=preds[f"pred_{model}"], y=preds["synthetic_label_next_rate"], mode="markers",
                             name="SYNTHETIC records", marker=dict(size=8, color=SERIES_DARK[4], symbol="triangle-up", opacity=0.85),
                             customdata=np.stack([preds["Hospital_ID"]], axis=-1),
                             hovertemplate="%{customdata[0]} (SYNTHETIC)<br>Model estimate %{x:.2f}%<br>"
                                           "Synthetic label %{y:.2f}%<extra></extra>"))
    base_layout(fig, height)
    fig.update_xaxes(title=f"{MODEL_LABELS[model]} estimate (%) - synthetic input", range=[lo, hi])
    fig.update_yaxes(title="Synthetic label (%) - not real data", range=[lo, hi])
    return fig
