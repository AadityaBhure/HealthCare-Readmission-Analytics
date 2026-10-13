"""Streamlit presentation helpers: global CSS, page header, KPI cards, callouts, chart cards."""
from __future__ import annotations

import html

import streamlit as st

from .theme import DARK, STATUS

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --page: {DARK['page']}; --surface: {DARK['surface']}; --surface-2: {DARK['surface_2']};
  --border: {DARK['border']}; --text: {DARK['text']}; --text-2: {DARK['text_2']}; --muted: {DARK['muted']};
  --accent: {DARK['accent']};
}}
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: Inter, 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
.stApp {{ background: var(--page); }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }}
section[data-testid="stSidebar"] {{ background: #0A1120; border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] .sidebar-note {{ font-size: 0.78rem; color: var(--muted); line-height: 1.45; }}
h1, h2, h3, h4 {{ color: var(--text); letter-spacing: -0.01em; }}

/* page header */
.page-header {{ margin: 0 0 1.1rem 0; }}
.page-eyebrow {{ color: var(--accent); font-size: 0.74rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 0.25rem; }}
.page-title {{ color: var(--text); font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin: 0; }}
.page-sub {{ color: var(--text-2); font-size: 0.95rem; margin-top: 0.4rem; max-width: 980px; line-height: 1.5; }}
.chip-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.7rem; }}
.chip {{ background: var(--surface-2); border: 1px solid var(--border); color: var(--text-2); border-radius: 999px;
  padding: 0.18rem 0.65rem; font-size: 0.75rem; }}
.chip b {{ color: var(--text); font-weight: 600; }}

/* KPI cards */
.kpi {{ background: linear-gradient(180deg, var(--surface) 0%, #0E182C 100%); border: 1px solid var(--border);
  border-radius: 14px; padding: 1rem 1.1rem 0.9rem; height: 100%; min-height: 138px; position: relative; overflow: hidden; }}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--kpi-accent, var(--accent)); opacity: .9; }}
.kpi-label {{ color: var(--text-2); font-size: 0.8rem; font-weight: 500; line-height: 1.3; }}
.kpi-value {{ color: var(--text); font-size: 1.95rem; font-weight: 700; line-height: 1.15; margin-top: 0.35rem; }}
.kpi-unit {{ color: var(--text-2); font-size: 0.95rem; font-weight: 500; margin-left: 0.15rem; }}
.kpi-delta {{ font-size: 0.8rem; font-weight: 600; margin-top: 0.3rem; }}
.kpi-caption {{ color: var(--muted); font-size: 0.74rem; margin-top: 0.35rem; line-height: 1.35; }}
.delta-up {{ color: #F2A65A; }} .delta-down {{ color: #4FC3A1; }} .delta-flat {{ color: var(--text-2); }}

/* section titles */
.section-title {{ color: var(--text); font-size: 1.08rem; font-weight: 650; margin: 0.4rem 0 0.1rem 0; }}
.section-caption {{ color: var(--muted); font-size: 0.82rem; margin-bottom: 0.4rem; line-height: 1.45; }}

/* cards: keyed st.container(border=True, key="card-...") */
div[class*="st-key-card"] {{ background: var(--surface); border: 1px solid var(--border) !important;
  border-radius: 14px; padding: 0.35rem 0.4rem; box-shadow: 0 1px 0 rgba(255,255,255,0.02), 0 8px 24px rgba(0,0,0,0.18); }}
div[class*="st-key-filters"] {{ background: var(--surface-2); border: 1px solid var(--border) !important;
  border-radius: 12px; padding: 0.15rem 0.5rem; }}

/* callouts */
.callout {{ border-radius: 12px; padding: 0.75rem 0.95rem; font-size: 0.86rem; line-height: 1.5; margin: 0.3rem 0 0.7rem 0;
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text-2); }}
.callout b {{ color: var(--text); }}
.callout-info {{ border-left: 3px solid var(--accent); }}
.callout-warn {{ border-left: 3px solid {STATUS['warning']}; }}
.callout-synth {{ border-left: 3px solid #d55181; background: rgba(213, 81, 129, 0.08); }}
.callout-title {{ font-weight: 650; color: var(--text); margin-bottom: 0.15rem; }}

/* findings list */
.finding {{ display: flex; gap: 0.7rem; padding: 0.55rem 0; border-bottom: 1px solid var(--border); }}
.finding:last-child {{ border-bottom: none; }}
.finding-tag {{ flex: 0 0 auto; font-size: 0.68rem; font-weight: 650; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--accent); background: rgba(34, 211, 238, 0.08); border: 1px solid rgba(34,211,238,.25); border-radius: 6px;
  padding: 0.12rem 0.45rem; height: fit-content; min-width: 96px; text-align: center; }}
.finding-text {{ color: var(--text-2); font-size: 0.87rem; line-height: 1.5; }}

/* status pills */
.pill {{ display: inline-block; padding: 0.08rem 0.5rem; border-radius: 999px; font-size: 0.72rem; font-weight: 650; }}
.pill-pass {{ background: rgba(12,163,12,.15); color: #4ade80; }}
.pill-warn {{ background: rgba(250,178,25,.14); color: #fbbf24; }}
.pill-fail {{ background: rgba(208,59,59,.16); color: #f87171; }}
.pill-label {{ background: var(--surface-2); color: var(--text-2); border: 1px solid var(--border); }}

.footer {{ color: var(--muted); font-size: 0.75rem; margin-top: 2rem; padding-top: 0.8rem; border-top: 1px solid var(--border); }}
div[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 0.6rem 0.8rem; }}
div[data-testid="stDataFrame"] {{ border-radius: 10px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; }}
.stTabs [data-baseweb="tab"] {{ background: var(--surface); border-radius: 10px 10px 0 0; padding: 0.4rem 0.9rem; }}

@media (max-width: 900px) {{
  .page-title {{ font-size: 1.45rem; }}
  .kpi-value {{ font-size: 1.6rem; }}
  .kpi {{ min-height: 0; margin-bottom: 0.5rem; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str, chips: list[tuple[str, str]] | None = None) -> None:
    chip_html = ""
    if chips:
        chip_html = '<div class="chip-row">' + "".join(
            f'<span class="chip">{html.escape(k)} <b>{html.escape(v)}</b></span>' for k, v in chips) + "</div>"
    st.markdown(
        f'<div class="page-header"><div class="page-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1><div class="page-sub">{subtitle}</div>{chip_html}</div>',
        unsafe_allow_html=True)


def kpi(label: str, value: str, unit: str = "", delta: str | None = None, delta_dir: str = "flat",
        caption: str = "", accent: str | None = None) -> None:
    """delta_dir: 'up' / 'down' / 'flat'. For readmission rates 'up' is unfavourable (amber), 'down' favourable."""
    arrow = {"up": "&#9650;", "down": "&#9660;", "flat": "&#9679;"}[delta_dir]
    delta_html = f'<div class="kpi-delta delta-{delta_dir}">{arrow} {html.escape(delta)}</div>' if delta else ""
    style = f' style="--kpi-accent:{accent}"' if accent else ""
    st.markdown(
        f'<div class="kpi"{style}><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}<span class="kpi-unit">{html.escape(unit)}</span></div>'
        f'{delta_html}<div class="kpi-caption">{caption}</div></div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>'
                + (f'<div class="section-caption">{caption}</div>' if caption else ""), unsafe_allow_html=True)


def callout(text: str, kind: str = "info", title: str | None = None) -> None:
    t = f'<div class="callout-title">{html.escape(title)}</div>' if title else ""
    st.markdown(f'<div class="callout callout-{kind}">{t}{text}</div>', unsafe_allow_html=True)


def findings_list(findings: list[dict]) -> None:
    items = "".join(f'<div class="finding"><span class="finding-tag">{html.escape(f["area"])}</span>'
                    f'<span class="finding-text">{html.escape(f["text"])}</span></div>' for f in findings)
    st.markdown(items, unsafe_allow_html=True)


def pill(status: str) -> str:
    return f'<span class="pill pill-{status}">{status.upper()}</span>'


def card(key: str):
    """A styled bordered container. Usage: ``with ui.card("cabg-trend"): ...``"""
    return st.container(border=True, key=f"card-{key}")


def filter_bar(key: str):
    return st.container(border=True, key=f"filters-{key}")


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                            "toImageButtonOptions": {"format": "png", "scale": 2}})


def footer() -> None:
    st.markdown(
        '<div class="footer">Aggregated hospital- and population-level statistics only - no individual patient records. '
        'Estimates are hospital-level rates, not individual patient probabilities, and are not clinically validated. '
        'Not a medical decision-support system. Data: California hospital quality reports and all-cause readmission '
        'statistics as supplied in <code>data/raw/</code>.</div>', unsafe_allow_html=True)


def fmt_pct(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None or v != v else f"{v:.{digits}f}"
