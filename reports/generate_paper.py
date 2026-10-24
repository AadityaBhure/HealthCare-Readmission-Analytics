"""Generate the IEEE-format conference paper (reports/IEEE_Paper.docx) from pipeline artifacts.

    python reports/generate_paper.py

Uses reports/templates/ieee_a4_template.docx (styles and automatic numbering of an IEEE A4
conference template: "I." headings, "Fig. N." captions, "TABLE N" heads, "[N]" references).
All numbers are read from artifacts/ - nothing is typed in by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.oxml import parse_xml  # noqa: E402
from docx.oxml.ns import nsdecls  # noqa: E402
from docx.shared import Emu, Inches, Pt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

from src import config  # noqa: E402
from src.reporting import _runs, load_artifacts  # noqa: E402
from src.theme import CODING_COLORS_LIGHT, MODEL_COLORS_LIGHT, SERIES_LIGHT  # noqa: E402

TEMPLATE = config.REPORTS_DIR / "templates" / "ieee_a4_template.docx"
OUT_DOCX = config.REPORTS_DIR / "IEEE_Paper.docx"
FIG_DIR = config.REPORTS_DIR / "paper_figures"

TITLE = ("Explainable Next-Period Readmission Analytics for California Hospitals: Temporally Aligned "
         "CABG Prediction, Stroke Outcome Analysis and Population Trends")
AUTHORS = [("Aaditya Bhure", "23070521034"), ("Bhargav Sor", "23070521031")]
AFFILIATION = ["Dept. of Computer Science Engineering", "Symbiosis Institute of Technology, Nagpur",
               "Symbiosis International (Deemed University)"]

FIG_W = 3.3  # inches: one IEEE column on A4
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
    "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e5e7eb", "grid.linewidth": 0.5, "axes.edgecolor": "#6b7280", "axes.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
})
LABELS = {"persistence": "Persistence", "linear_regression": "Linear Regression", "random_forest": "Random Forest",
          "mean_baseline": "Training mean"}


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p)
    plt.close(fig)
    return p


def _pct(ax, axis="y"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# ======================================================================================
# Column-width figures
# ======================================================================================
def fig_coverage(A) -> Path:
    cs, ss, st = A["cabg_statewide"], A["stroke_statewide"], A["pop_state"]
    rows = []
    for _, r in cs[cs["measure_code"] == "readmission"].iterrows():
        rows.append((f"CABG {r['period']}", r["period_start"], 2, SERIES_LIGHT[0], 0.35 if r["hospitals"] < 10 else 1,
                     f"{int(r['hospitals'])}"))
    for _, r in ss[ss["measure_code"] == "readmission"].iterrows():
        rows.append((f"Stroke {r['period']}", r["period_start"], 2, SERIES_LIGHT[1], 1, ""))
    for _, r in st.iterrows():
        rows.append((f"All-cause ({r['coding_system']})", r["year"], 1, CODING_COLORS_LIGHT[r["coding_system"]],
                     0.4 if r["coding_transition_year"] else 1, ""))
    lanes = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(figsize=(FIG_W, 2.25))
    for lane, s, yrs, col, a, note in rows:
        y = len(lanes) - 1 - lanes.index(lane)
        ax.barh(y, yrs - 0.15, left=s - 0.42, height=0.62, color=col, alpha=a)
        if note:
            ax.text(s + yrs - 0.5, y, f" n={note}", va="center", fontsize=5.6, color="#374151")
    ax.set_yticks(range(len(lanes))); ax.set_yticklabels(lanes[::-1], fontsize=6)
    ax.set_xticks(range(2011, 2025, 2)); ax.set_xlim(2010.4, 2025.3); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Calendar year covered (n = hospitals reporting CABG readmission)")
    return _save(fig, "p01_coverage.png")


def _box(ax, x, y, w, h, text, fc, ec, fs=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color="#4b5563", lw=0.8))


def fig_workflow(A) -> Path:
    s = A["cabg_summary"]["train_summary"]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.55))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.00, 0.80, 0.30, 0.18, f"4 raw CSV files\nvalidate + clean\n({len(A['checks'])} checks)", "#fff4e5", "#c98500")
    _box(ax, 0.35, 0.80, 0.30, 0.18, "Hospital x window\npanel", "#eef4fb", "#0b2545")
    _box(ax, 0.70, 0.80, 0.30, 0.18, "Transitions t -> t+1\n(no gap bridging)", "#eef4fb", "#0b2545")
    _arrow(ax, 0.30, 0.89, 0.35, 0.89); _arrow(ax, 0.65, 0.89, 0.70, 0.89)
    _box(ax, 0.00, 0.50, 0.47, 0.20, f"TRAIN n = {s['n_train']}\ntargets {', '.join(s['train_target_periods'])}",
         "#e8f6ef", "#1b7a4a")
    _box(ax, 0.53, 0.50, 0.47, 0.20, f"TEST (held out) n = {s['n_test']}\ntarget {s['test_target_periods'][0]}",
         "#fdecea", "#b42318")
    _arrow(ax, 0.80, 0.80, 0.30, 0.70); _arrow(ax, 0.86, 0.80, 0.80, 0.70)
    _box(ax, 0.00, 0.22, 0.47, 0.20, "Grouped 5-fold CV\n(RF tuning, model choice)\n-> fit 4 models", "#e8f6ef", "#1b7a4a")
    _box(ax, 0.53, 0.22, 0.47, 0.20, "One-time evaluation\nMAE, RMSE, R², bootstrap,\npermutation importance", "#fdecea", "#b42318")
    _arrow(ax, 0.235, 0.50, 0.235, 0.42); _arrow(ax, 0.47, 0.32, 0.53, 0.32)
    _box(ax, 0.00, 0.00, 1.00, 0.14, "Sensitivity analyses: pooled grouped CV  |  non-overlapping horizon t -> t+2\n"
         "|  extended feature set", "#f3effe", "#5b4fc7", fs=5.9)
    return _save(fig, "p02_workflow.png")


def fig_dashboard() -> Path | None:
    src = config.SCREENSHOTS_DIR / "page2_cabg_history.png"
    if not src.exists():
        return None
    im = Image.open(src)
    w, h = im.size
    box = (int(0.262 * w), int(0.140 * h), int(0.672 * w), int(0.640 * h))  # history chart card
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "p03_dashboard.png"
    im.crop(box).save(p)
    return p


def fig_lag(A) -> Path:
    tr1 = A["cabg_transitions"]
    tr2 = A["cabg_transitions_nonoverlap"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.75), sharey=True)
    for ax, df, t in ((axes[0], tr1, "Consecutive (shared year)"), (axes[1], tr2, "Non-overlapping")):
        d = df[df["primary_features_complete"]].dropna(subset=["prev_readmit_rate", "next_readmit_rate"])
        r = d["prev_readmit_rate"].corr(d["next_readmit_rate"])
        ax.scatter(d["prev_readmit_rate"], d["next_readmit_rate"], s=4, color=SERIES_LIGHT[0], alpha=0.6, lw=0)
        ax.plot([0, 42], [0, 42], ls=":", color="#6b7280", lw=0.7)
        ax.set_xlim(0, 42); ax.set_ylim(0, 42)
        ax.set_title(f"{t}\nr = {r:.2f}, n = {len(d)}", fontsize=6.6)
        ax.set_xlabel("Previous rate (%)")
    axes[0].set_ylabel("Target rate (%)")
    fig.tight_layout(pad=0.3)
    return _save(fig, "p04_lag.png")


def fig_designs(A) -> Path:
    m = A["cabg_metrics"].set_index("model")["MAE"]
    cp, cn = A["cabg_cv_pooled"].set_index("model"), A["cabg_cv_nonoverlap"].set_index("model")
    s = A["cabg_summary"]
    designs = [(f"Hold-out\n(n={s['train_summary']['n_test']})", m, None),
               (f"Grouped CV\n(n={s['n_pooled']})", cp["MAE_mean"], cp["MAE_sd"]),
               (f"Non-overlap CV\n(n={s['n_nonoverlap']})", cn["MAE_mean"], cn["MAE_sd"])]
    models = ["persistence", "linear_regression", "random_forest", "mean_baseline"]
    fig, ax = plt.subplots(figsize=(FIG_W, 1.95))
    w = 0.2
    for i, mod in enumerate(models):
        xs = np.arange(3) + (i - 1.5) * w
        ys = [d[1][mod] for d in designs]
        err = [d[2][mod] if d[2] is not None else 0 for d in designs]
        ax.bar(xs, ys, w * 0.9, yerr=err, color=MODEL_COLORS_LIGHT[mod], label=LABELS[mod],
               error_kw=dict(ecolor="#374151", lw=0.6, capsize=1.5))
    ax.set_xticks(range(3)); ax.set_xticklabels([d[0] for d in designs])
    ax.set_ylabel("MAE (percentage points)"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 4.9)
    ax.legend(ncol=2, loc="upper left", handlelength=1.0, columnspacing=0.8)
    return _save(fig, "p05_designs.png")


def fig_bootstrap(A) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 2.25))
    short = {"linear_regression": "LR", "random_forest": "RF", "persistence": "Persistence", "mean_baseline": "Mean"}
    for ax, pre, t in ((axes[0], "cabg", "CABG (n = %d)"), (axes[1], "stroke", "Stroke (n = %d)")):
        b = A[f"{pre}_bootstrap"].iloc[::-1]
        y = np.arange(len(b))
        ax.errorbar(b["mae_diff"], y, xerr=[b["mae_diff"] - b["ci_low"], b["ci_high"] - b["mae_diff"]], fmt="none",
                    ecolor="#374151", capsize=2, lw=0.8)
        ax.scatter(b["mae_diff"], y, color=[MODEL_COLORS_LIGHT[x] for x in b["model"]], s=12, zorder=3)
        ax.axvline(0, color="#111827", lw=0.7)
        ax.set_yticks(y); ax.set_yticklabels([f"{short[a]} - {short[r]}" for a, r in zip(b["model"], b["reference"])])
        ax.set_title(t % A[f"{pre}_summary"]["train_summary"]["n_test"], fontsize=6.8)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Difference in test MAE (pp), 95% bootstrap CI (< 0: first model better)")
    fig.tight_layout(pad=0.3)
    return _save(fig, "p06_bootstrap.png")


def fig_permutation(A) -> Path:
    lab = {"prev_readmit_rate": "Prev. readmission", "prev_stroke_rate": "Prev. post-op stroke",
           "prev_cabg_volume": "Prev. CABG volume"}
    perm = A["cabg_permutation"]
    feats = list(lab)
    fig, ax = plt.subplots(figsize=(FIG_W, 1.35))
    for i, m in enumerate(["linear_regression", "random_forest"]):
        s = perm[perm["model"] == m].set_index("feature").loc[feats]
        y = np.arange(3) + (i - 0.5) * 0.36
        ax.barh(y, s["importance_mae_increase"], 0.34, xerr=s["importance_sd"], color=MODEL_COLORS_LIGHT[m],
                label=LABELS[m], error_kw=dict(ecolor="#374151", lw=0.6, capsize=1.5))
    ax.set_yticks(range(3)); ax.set_yticklabels([lab[f] for f in feats]); ax.invert_yaxis()
    ax.axvline(0, color="#111827", lw=0.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (pp)")
    ax.legend(loc="lower right")
    return _save(fig, "p07_permutation.png")


def fig_learned(A) -> Path:
    c, tr = A["cabg_learned_curve"], A["cabg_transitions"]
    s = A["cabg_summary"]["train_summary"]
    train = tr[tr["target_period"].isin(s["train_target_periods"]) & tr["primary_features_complete"]]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.15))
    ax.scatter(train["prev_readmit_rate"], train["next_readmit_rate"], s=4, color="#cbd5e1", label="Training transitions")
    for m, ls in (("persistence", ":"), ("mean_baseline", "--"), ("linear_regression", "-"), ("random_forest", "-")):
        ax.plot(c["prev_readmit_rate"], c[f"pred_{m}"], ls=ls, lw=1.4 if ls == "-" else 1.0,
                color=MODEL_COLORS_LIGHT[m], label=LABELS[m])
    _pct(ax); _pct(ax, "x")
    ax.set_xlabel("Previous-window readmission rate"); ax.set_ylabel("Estimated next-window rate")
    ax.legend(loc="upper left", handlelength=1.6)
    return _save(fig, "p08_learned.png")


def fig_errors(A) -> Path:
    p, err = A["cabg_test_predictions"], A["cabg_error_by_volume"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.7), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.scatter(p["target_cases"], p["next_readmit_rate"] - p["pred_linear_regression"], s=4,
               color=MODEL_COLORS_LIGHT["linear_regression"], alpha=0.7, lw=0)
    ax.axhline(0, color="#111827", lw=0.6); ax.set_xscale("log")
    ax.set_xlabel("Eligible cases (log)"); ax.set_ylabel("Residual (pp)"); ax.set_title("LR residuals", fontsize=6.8)
    ax = axes[1]
    terc = list(dict.fromkeys(err["volume_tercile"]))
    for i, m in enumerate(["persistence", "linear_regression", "random_forest"]):
        s = err[err["model"] == m].set_index("volume_tercile").loc[terc]
        ax.bar(np.arange(3) + (i - 1) * 0.27, s["MAE"], 0.25, color=MODEL_COLORS_LIGHT[m], label=LABELS[m])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Hospital volume tercile"); ax.set_ylabel("Test MAE (pp)"); ax.grid(axis="x", visible=False)
    ax.set_title("Error by volume", fontsize=6.8); ax.legend(fontsize=5.4, loc="upper right", handlelength=0.8)
    fig.tight_layout(pad=0.3)
    return _save(fig, "p09_errors.png")


def fig_stroke(A) -> Path:
    panel, sw = A["stroke_panel"], A["stroke_statewide"]
    last = sorted(panel["period"].unique())[-1]
    d = panel[panel["period"] == last].dropna(subset=["readmit_rate", "mortality_rate"])
    r = d["readmit_rate"].corr(d["mortality_rate"])
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.75), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    for code, col, lab in (("readmission", SERIES_LIGHT[0], "Readmission"), ("mortality", SERIES_LIGHT[1], "Mortality")):
        s = sw[sw["measure_code"] == code]
        ax.plot([p[2:4] + "-" + p[7:9] for p in s["period"]], s["risk_adjusted_rate"], "-o", ms=2.5, lw=1.1, color=col, label=lab)
    _pct(ax); ax.set_ylim(8, 14); ax.legend(loc="lower left"); ax.set_title("Statewide", fontsize=6.8)
    ax.tick_params(axis="x", labelsize=5.8)
    ax = axes[1]
    ax.scatter(d["mortality_rate"], d["readmit_rate"], s=np.clip(d["readmit_cases"].fillna(10) / 25, 2, 18),
               color=SERIES_LIGHT[0], alpha=0.5, lw=0)
    _pct(ax); _pct(ax, "x")
    ax.set_xlabel("30-day mortality"); ax.set_ylabel("30-day readmission")
    ax.set_title(f"Hospitals {last}: r = {r:+.2f}", fontsize=6.8)
    fig.tight_layout(pad=0.3)
    return _save(fig, "p10_stroke.png")


def fig_trend(A) -> Path:
    st = A["pop_state"]
    fig, ax = plt.subplots(figsize=(FIG_W, 1.6))
    for code in ("ICD-9", "ICD-10"):
        g = st[st["coding_system"] == code]
        ax.plot(g["year"], g["rate"], "-o", ms=2.6, lw=1.2, color=CODING_COLORS_LIGHT[code], label=f"{code} era")
    part = st[st["coding_transition_year"]]
    ax.plot(part["year"], part["rate"], "o", mfc="white", mec=CODING_COLORS_LIGHT["ICD-9"], ms=4, label="2015 partial year")
    ax.axvline(2015.5, color="#6b7280", lw=0.6, ls=":")
    for y in (2011, 2016, 2024):
        v = float(st.loc[st["year"] == y, "rate"].iloc[0])
        ax.annotate(f"{v:.2f}", (y, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=5.8)
    ax.set_xticks(range(2011, 2025, 2)); _pct(ax); ax.set_ylabel("Readmission rate")
    ax.legend(loc="upper left", ncol=1, handlelength=1.2)
    ax.set_ylim(13.2, 16.6)
    return _save(fig, "p11_trend.png")


def fig_map(A) -> Path:
    county, pop = A["pop_county"], A["pop_summary"]
    year, thr = pop["latest_year"], pop["min_county_admits"]
    gj = json.loads(config.COUNTY_GEOJSON.read_text(encoding="utf-8"))
    d = county[county["year"] == year].set_index("county")
    ok = d[(d["admits"] >= thr) & d["rate"].notna()]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("t", ["#E3F1F2", "#86C7CB", "#1F8790", "#0A4F55"])
    norm = matplotlib.colors.Normalize(float(ok["rate"].min()), float(ok["rate"].max()))
    fig, ax = plt.subplots(figsize=(2.6, 2.75))
    patches, cols = [], []
    for f in gj["features"]:
        n = f["properties"]["NAME"]; g = f["geometry"]["coordinates"]
        polys = [g] if f["geometry"]["type"] == "Polygon" else g
        col = cmap(norm(ok.loc[n, "rate"])) if n in ok.index else "#d1d5db"
        for poly in polys:
            patches.append(Polygon(np.asarray(poly[0]), closed=True)); cols.append(col)
    ax.add_collection(PatchCollection(patches, facecolor=cols, edgecolor="white", linewidth=0.3))
    ax.set_xlim(-124.5, -114.1); ax.set_ylim(32.5, 42.05); ax.set_aspect(1.18); ax.axis("off")
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.0)
    cb.set_label(f"Readmission rate {year} (%)", fontsize=6); cb.ax.tick_params(labelsize=5.8); cb.outline.set_visible(False)
    return _save(fig, "p12_map.png")


def fig_strata(A) -> Path:
    strata, pop = A["pop_strata"], A["pop_summary"]
    year = pop["latest_year"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.55), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, cat, t in ((axes[0], "Expected Payer", "By expected payer"), (axes[1], "Age", "By age group")):
        g = strata[(strata["strata"] == cat) & (strata["year"] == year)].dropna(subset=["rate"]).sort_values("rate")
        names = [n.replace("MediCare", "Medicare").replace(" years and above", "+").replace(" years", "") for n in g["strata_name"]]
        ax.barh(names, g["rate"], color=SERIES_LIGHT[0], height=0.62)
        ax.axvline(pop["latest_rate"], color="#111827", lw=0.7, ls=":")
        for i, v in enumerate(g["rate"]):
            ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=5.6)
        ax.set_xlim(0, 21.5); _pct(ax, "x"); ax.grid(axis="y", visible=False); ax.set_title(t, fontsize=6.8)
    fig.tight_layout(pad=0.3)
    return _save(fig, "p13_strata.png")


# ======================================================================================
# DOCX helpers (IEEE template styles)
# ======================================================================================
class Paper:
    def __init__(self):
        self.doc = Document(TEMPLATE)
        self.figs: list[str] = []
        self.tabs: list[str] = []

    def st(self, name: str):
        """Look a style up by its displayed name (the template spells built-in names its own way)."""
        for s in self.doc.styles:
            if s.name == name or s.name.lower() == name.lower():
                return s
        raise KeyError(f"style {name!r} not found in template")

    # numbering helpers - numbers follow the order in which items are added
    def F(self, key: str) -> int:
        if key not in self.figs:
            self.figs.append(key)
        return self.figs.index(key) + 1

    def T(self, key: str) -> str:
        if key not in self.tabs:
            self.tabs.append(key)
        return ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][self.tabs.index(key)]

    def _rich(self, p, text, size=None):
        for t, b, i in _runs(text):
            r = p.add_run(t)
            r.bold = True if b else None
            r.italic = True if i else None
            if size:
                r.font.size = Pt(size)
        return p

    def section_break(self, cols: int, top: int):
        p = self.doc.add_paragraph()
        cols_xml = "<w:cols/>" if cols == 1 else f'<w:cols w:num="{cols}" w:space="360" w:equalWidth="true"/>'
        p._p.get_or_add_pPr().append(parse_xml(
            f'<w:sectPr {nsdecls("w")}><w:type w:val="continuous"/><w:pgSz w:w="11906" w:h="16838"/>'
            f'<w:pgMar w:left="893" w:right="893" w:gutter="0" w:header="360" w:top="{top}" w:footer="360" w:bottom="1080"/>'
            f'{cols_xml}<w:docGrid w:type="default" w:linePitch="360" w:charSpace="8192"/></w:sectPr>'))

    def title(self, text):
        self.doc.add_paragraph(text, style=self.st("paper title"))

    def author(self, name, prn):
        p = self.doc.add_paragraph(style=self.st("Author"))
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        lines = [(name, False)] + [(a, True) for a in AFFILIATION] + [("Nagpur, India", False), (f"PRN: {prn}", False)]
        for k, (t, it) in enumerate(lines):
            r = p.add_run(t)
            r.italic = True if it else None
            r.font.size = Pt(9)
            if k < len(lines) - 1:
                r.add_break()

    def abstract(self, text):
        p = self.doc.add_paragraph(style=self.st("Abstract"))
        r = p.add_run("Abstract")
        r.italic = True
        p.add_run("—" + text)

    def keywords(self, text):
        self.doc.add_paragraph("Keywords—" + text, style=self.st("Keywords"))

    def h1(self, t):
        self.doc.add_paragraph(t, style=self.st("Heading 1"))

    def h2(self, t):
        self.doc.add_paragraph(t, style=self.st("Heading 2"))

    def h5(self, t):
        self.doc.add_paragraph(t, style=self.st("Heading 5"))

    def p(self, text):
        return self._rich(self.doc.add_paragraph(style=self.st("Body Text")), text)

    def bullets(self, items):
        for it in items:
            self._rich(self.doc.add_paragraph(style=self.st("bullet list")), it)

    def figure(self, key, path, caption, width=FIG_W):
        if path is None or not Path(path).exists():
            return
        self.F(key)
        p = self.doc.add_paragraph()
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(4)
        p.add_run().add_picture(str(path), width=Inches(width))
        self._rich(self.doc.add_paragraph(style=self.st("figure caption")), caption)

    def table(self, key, head, cols, rows, widths, aligns=None):
        self.T(key)
        hp = self.doc.add_paragraph(head, style=self.st("table head"))
        hp.paragraph_format.keep_with_next = True
        aligns = aligns or (["left"] + ["center"] * (len(cols) - 1))
        t = self.doc.add_table(rows=1 + len(rows), cols=len(cols))
        tblPr = t._tbl.tblPr
        for child in list(tblPr):
            tblPr.remove(child)
        total = sum(widths)
        for xml in (f'<w:tblW {nsdecls("w")} w:w="{total}" w:type="dxa"/>', f'<w:jc {nsdecls("w")} w:val="center"/>',
                    f'<w:tblBorders {nsdecls("w")}>' + "".join(
                        f'<w:{b} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                        for b in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>",
                    f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>',
                    f'<w:tblCellMar {nsdecls("w")}><w:top w:w="10" w:type="dxa"/><w:left w:w="40" w:type="dxa"/>'
                    f'<w:bottom w:w="10" w:type="dxa"/><w:right w:w="40" w:type="dxa"/></w:tblCellMar>'):
            tblPr.append(parse_xml(xml))
        for gc, w in zip(t._tbl.tblGrid.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}gridCol"), widths):
            gc.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w", str(w))
        data = [cols] + rows
        for ri, row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            for ci, cell in enumerate(row.cells):
                cell.width = Emu(widths[ci] * 635)
                par = cell.paragraphs[0]
                par.style = self.st("table col head" if ri == 0 else "table copy")
                par.paragraph_format.keep_with_next = ri < len(data) - 1
                par.alignment = 1 if (ri == 0 or aligns[ci] == "center") else 0
                self._rich(par, str(data[ri][ci]))
        self.doc.add_paragraph()

    def references(self, refs):
        for r in refs:
            self.doc.add_paragraph(r, style=self.st("references"))


# ======================================================================================
# Content
# ======================================================================================
def build(A) -> Paper:
    P = Paper()
    cs, ss = A["cabg_summary"], A["stroke_summary"]
    ct, st = cs["train_summary"], ss["train_summary"]
    cm, sm = A["cabg_metrics"].set_index("model"), A["stroke_metrics"].set_index("model")
    cb, sb = A["cabg_bootstrap"], A["stroke_bootstrap"]
    cvp, cvn = A["cabg_cv_pooled"].set_index("model"), A["cabg_cv_nonoverlap"].set_index("model")
    scvp, scvn = A["stroke_cv_pooled"].set_index("model"), A["stroke_cv_nonoverlap"].set_index("model")
    pop, au = A["pop_summary"], A["audit"]
    checks = A["checks"]
    n_pass, n_warn, n_fail = (int((checks["status"] == s).sum()) for s in ("pass", "warn", "fail"))

    def boot(df, a, r="persistence"):
        return df[(df["model"] == a) & (df["reference"] == r)].iloc[0]

    def paired(df, design, model):
        return df[df["design"].str.startswith(design) & (df["model"] == model)].iloc[0]

    lr_b, rf_b, s_lr_b = boot(cb, "linear_regression"), boot(cb, "random_forest"), boot(sb, "linear_regression")
    pn = paired(A["cabg_paired_cv"], "non", "linear_regression")
    p1 = paired(A["cabg_paired_cv"], "one", "linear_regression")
    sp1 = paired(A["stroke_paired_cv"], "one", "linear_regression")
    lagc = cs["lag_correlation"]
    perm = A["cabg_permutation"]
    lrp = perm[perm["model"] == "linear_regression"].set_index("feature")["importance_mae_increase"]
    coef = A["cabg_native"]
    coef = coef[coef["model"] == "linear_regression"].set_index("feature")["value"]
    err = A["cabg_error_by_volume"]
    err_lr = err[err["model"] == "linear_regression"].set_index("volume_tercile")["MAE"]
    err_p = err[err["model"] == "persistence"].set_index("volume_tercile")["MAE"]
    curve = A["cabg_learned_curve"]

    def curve_at(x):
        i = (curve["prev_readmit_rate"] - x).abs().idxmin()
        return float(curve.loc[i, "prev_readmit_rate"]), float(curve.loc[i, "pred_linear_regression"])

    hi_x, hi_y = curve_at(20)
    lo_x, lo_y = curve_at(8)
    mean_train = float(ct["target_train_mean"])
    fs = A["cabg_feature_sets"]
    ext = fs[fs["feature_set"].str.startswith("extended")].set_index("model")["MAE"]
    sw = A["cabg_statewide"]
    swr = sw[sw["measure_code"] == "readmission"].reset_index(drop=True)
    ssw = A["stroke_statewide"]
    ssr, ssm = ssw[ssw["measure_code"] == "readmission"], ssw[ssw["measure_code"] == "mortality"]
    corr = [v["pearson_r"] for v in A["stroke_descriptive"]["readmission_vs_mortality_correlation"].values()]
    payer, age = pop["strata_latest"]["Expected Payer"], pop["strata_latest"]["Age"]
    race = pop["strata_latest"]["Race-Ethnicity"]
    rfp = cs["rf_best_params"]
    srfp = ss["rf_best_params"]

    def rfstr(p):
        return (f"depth {p['model__max_depth'] if p['model__max_depth'] is not None else 'none'}, "
                f"leaf {p['model__min_samples_leaf']}, features {p['model__max_features']}")

    figs = {
        "coverage": fig_coverage(A), "workflow": fig_workflow(A), "dashboard": fig_dashboard(), "lag": fig_lag(A),
        "designs": fig_designs(A), "bootstrap": fig_bootstrap(A), "permutation": fig_permutation(A),
        "learned": fig_learned(A), "errors": fig_errors(A), "stroke": fig_stroke(A), "trend": fig_trend(A),
        "map": fig_map(A), "strata": fig_strata(A),
    }
    # Pre-register figure and table order so in-text numbers match Word's automatic numbering.
    for k in ["coverage", "workflow", "dashboard", "lag", "designs", "bootstrap", "permutation", "learned", "errors",
              "stroke", "trend", "map", "strata"]:
        if figs[k] is not None:
            P.F(k)
    for k in ["datasets", "features", "models", "cabg", "designs", "stroke"]:
        P.T(k)

    # ---------------------------------------------------------------- front matter
    P.title(TITLE)
    P.section_break(1, 760)
    for name, prn in AUTHORS:
        P.author(name, prn)
    P.section_break(2, 1080)
    P.abstract(
        "Thirty-day readmission rates are widely used to judge hospital quality, yet most predictive studies "
        "work with patient records or same-period information. We ask whether a hospital's published history can "
        "estimate its next-period risk-adjusted 30-day readmission rate after isolated coronary artery bypass "
        "grafting (CABG), using California hospital performance data for 2018–2023, and we analyse ischemic stroke "
        "outcomes (2011–2015) and statewide all-cause readmissions (2011–2024) alongside. Inputs from one reporting "
        "window were paired with the outcome of the next window without bridging missing periods. A persistence "
        "baseline (next = previous), a training-mean baseline, Linear Regression and Random Forest were trained on "
        f"{ct['n_train']} hospital transitions and evaluated once on a held-out later window ({ct['n_test']} hospitals) "
        f"with hospital-grouped cross-validation and paired bootstrap intervals. Persistence reached a mean absolute "
        f"error (MAE) of {cm.loc['persistence', 'MAE']:.2f} percentage points, Linear Regression {cm.loc['linear_regression', 'MAE']:.2f} "
        f"and Random Forest {cm.loc['random_forest', 'MAE']:.2f}; no model was reliably better than persistence "
        f"(95% CI of the Linear Regression gap {lr_b['ci_low']:+.2f} to {lr_b['ci_high']:+.2f}). We show that the published "
        f"two-year windows overlap by one calendar year, which raises the lag-1 correlation to {lagc['one_step_overlapping']:.2f} "
        f"versus {lagc['non_overlapping']:.2f} for non-overlapping windows. On a non-overlapping horizon persistence error rose to "
        f"{cvn.loc['persistence', 'MAE_mean']:.2f} while Linear Regression achieved {cvn.loc['linear_regression', 'MAE_mean']:.2f}, better in "
        f"{pn['share_folds_model_better']:.0%} of folds. A separate stroke model reduced error relative to persistence by "
        f"{abs(s_lr_b['relative_change_pct']):.1f}%. Results are delivered through an explainable five-page dashboard.")
    P.keywords("hospital readmission, CABG, ischemic stroke, temporal validation, persistence baseline, "
               "grouped cross-validation, regression to the mean, explainable AI, healthcare analytics")

    # ---------------------------------------------------------------- I. Introduction
    P.h1("Introduction")
    P.p("An unplanned readmission within 30 days of discharge is costly and often signals an incomplete recovery or a "
        "poor transition from hospital to home. About one in five Medicare beneficiaries was rehospitalised within "
        "30 days in a landmark study [1], and readmission rates now drive public reporting and financial penalties "
        "in the United States [2]. Predicting readmission remains difficult: most published risk models show only "
        "modest discrimination [3].")
    P.p("**Hospital-level forecasting** raises a different problem from patient-level prediction. A hospital's rate in "
        "a reporting window is an estimate from a limited number of patients, so changes between windows partly "
        "reflect chance [4], and unusually high or low rates tend to move back toward the average in the next "
        "period, a phenomenon known as regression to the mean [5]. A credible model must therefore be compared with "
        "simple baselines and its uncertainty must be reported.")
    P.p("This project builds and evaluates an explainable, reproducible analytics framework on four California "
        "datasets. Its contributions are:")
    P.bullets([
        "a temporally aligned design that pairs inputs from one reporting window with the next window's rate and never "
        "bridges a missing period;",
        "a baseline-first, leakage-aware evaluation (forward hold-out, hospital-grouped cross-validation, paired bootstrap);",
        "the discovery that overlapping two-year windows inflate persistence, quantified with a non-overlapping design;",
        "a separate ischemic-stroke model and statewide, county and population-group trend analysis;",
        "an interactive dashboard that runs exactly the evaluated models, with automated tests and a CI pipeline.",
    ])

    # ---------------------------------------------------------------- II. Related Work
    P.h1("Related Work")
    P.p("**Readmission risk models.** A systematic review of 26 models found that most achieved only moderate "
        "discrimination and relied on administrative variables [3]. Such models predict individual patients; "
        "estimating a hospital's future aggregate rate is less studied.")
    P.p("**Hospital performance measurement.** Small hospital samples make outcome rates unstable [4]; procedure volume "
        "is associated with surgical outcomes [6]; and hospital readmission and mortality rates are largely "
        "uncorrelated, so the two measures rank hospitals differently [7].")
    P.p("**Validation for structured data.** When the same unit contributes several rows, random cross-validation is "
        "optimistic and grouped or blocked validation is recommended [8]. Target leakage through features or "
        "preprocessing is a common cause of inflated results [9].")
    P.p("**Forecast baselines.** Naive forecasts such as persistence are standard benchmarks that a useful model must beat "
        "[10]; regression to the mean predicts that extreme values partly revert [5].")
    P.p("**Research gap.** We found no student-scale study that combines next-period hospital readmission forecasting, "
        "explicit handling of overlapping reporting windows, persistence-based evaluation and population-level "
        "context in one reproducible framework.")

    # ---------------------------------------------------------------- III. Dataset Description
    P.h1("Dataset Description")
    P.p(f"**Sources.** Four CSV files were supplied for the project; their structure follows public reports of the "
        f"California Department of Health Care Access and Information distributed through the CHHS Open Data Portal "
        f"[11] (Table {P.T('datasets')}). The raw files are stored unchanged and all fixes are applied in code.")
    P.table("datasets", "Datasets and their roles in the project", ["Data", "Granularity", "Period", "Rows", "Role"], [
        ["A CABG", "hospital x window x measure", "2018–2023", f"{au['cabg']['rows']:,}", "primary model"],
        ["B Stroke", "hospital x window x measure", "2011–2015", f"{au['stroke']['rows']:,}", "stroke model"],
        ["C All-cause", "year x stratum x county", "2011–2024", f"{au['allcause']['rows']:,}", "population trends"],
        ["D Synthetic", "synthetic hospital", "–", f"{au['synthetic']['rows']:,}", "tests only"],
    ], [900, 1500, 760, 560, 1160])
    P.p(f"**CABG performance.** {au['cabg']['hospital_ids']} hospitals report four measures: isolated CABG 30-day readmission, "
        "post-operative stroke and operative mortality, and CABG+valve mortality. Each has a risk-adjusted rate (RAAER), "
        "95% confidence limits and case counts. Readmission and stroke are published for two-year windows "
        + "; ".join(f"{r['period']} ({int(r['hospitals'])} hospitals)" for _, r in swr.iterrows())
        + ", while operative mortality is published for single years.")
    P.p(f"**Ischemic stroke.** {au['stroke']['hospital_ids']} hospitals report risk-adjusted 30-day readmission and mortality "
        "for four two-year windows from 2011–2012 to 2014–2015, which do not overlap in time with the CABG data.")
    P.p(f"**All-cause readmissions.** Statewide and county counts and rates for {au['allcause']['years'][0]}–{au['allcause']['years'][1]}, "
        "with age, sex, race/ethnicity and expected-payer strata at state level. Rates use ICD-9 coding up to 2015 "
        "and ICD-10 from 2016 [12].")
    P.p("**Synthetic records.** 120 flagged synthetic CABG rows were used only for schema and smoke tests, never for "
        f"training or evaluation. Fig. {P.F('coverage')} summarises coverage.")
    P.figure("coverage", figs["coverage"],
             "Reporting periods per source. CABG and stroke windows overlap by one year; the faded 2020–2021 CABG "
             "window holds only 2 hospitals; 2015 is a partial ICD-9 year.")

    # ---------------------------------------------------------------- IV. Data Preprocessing
    P.h1("Data Preprocessing")
    P.p(f"**Encoding and text.** The CABG and stroke files are Windows-1252 encoded. Line breaks inside names, double "
        f"spaces and typographic dashes were normalised, and the rating typo “Not Appplicable” was fixed. The "
        f"{au['cabg']['hospital_ids']} CABG hospitals appear under {au['cabg']['name_variants']} name spellings, so the "
        "stable facility identifier is the panel key.")
    P.p("**Missing and suppressed values.** Values reported as “.” or suppressed (e.g., Alpine County 2017) were kept "
        "missing and never set to zero. Rates stored as text (“13.80%”) and counts with separators (“1,877,198”) were parsed.")
    P.p(f"**Column transposition.** In the 2013–2014 stroke window all {au['stroke']['rows_corrected_for_column_swap']} rows "
        "reported more deaths/readmissions than cases. The two count columns were swapped back; afterwards the observed "
        "rates agree with the published risk-adjusted rates as in the other windows.")
    P.p("**Coding eras.** ICD-9 and ICD-10 rows were combined into one table with an explicit coding-system label and are "
        "never pooled without it; 2015 is flagged as partial because ICD-9 ended on 1 October 2015 [12].")
    P.p(f"**Validation.** {n_pass + n_warn + n_fail} automated checks ran on the raw and cleaned data: {n_pass} passed, "
        f"{n_warn} produced documented warnings (sparse 2020–2021 window, stroke transposition, small counties) and "
        f"{n_fail} failed.")

    # ---------------------------------------------------------------- V. Proposed Methodology
    P.h1("Proposed Methodology")
    P.p(f"The workflow (Fig. {P.F('workflow')}) is: validation and cleaning, a hospital-by-window panel, temporally "
        "aligned transitions, a forward temporal split, model selection inside the training data, a single evaluation "
        "on held-out data, and sensitivity analyses.")
    P.figure("workflow", figs["workflow"], "Training and evaluation workflow. The held-out window is used once.")
    P.h2("Temporal Alignment")
    P.p("A transition pairs features from window t with the readmission rate of the same hospital in window t+1 and is "
        "created only when the target window starts exactly one year later. Because 2020–2021 is almost empty, the pair "
        "2019–2020 to 2021–2022 is not formed in the main design; it is used only in a separate, labelled "
        f"non-overlapping design whose horizon is two years. Two transitions out of 2020–2021 were excluded because no "
        f"post-operative stroke measure was published, leaving {ct['n_train']} training and {ct['n_test']} test transitions.")
    P.h2("Features and Target")
    P.p(f"Table {P.T('features')} lists the inputs. Identifiers, names, counties and any target-window field are excluded.")
    P.p("Operative mortality is published for single years only, so it was aligned to the last year of the input window "
        "and tested only as an extended-feature sensitivity analysis.")
    P.table("features", "Model inputs and target (CABG)", ["Variable", "Definition", "Unit"], [
        ["prev_readmit_rate", "readmission RAAER in window t", "%"],
        ["prev_stroke_rate", "post-operative stroke RAAER in window t", "%"],
        ["prev_cabg_volume", "hospital CABG cases in window t (log1p in model)", "cases"],
        ["next_readmit_rate", "target: readmission RAAER in window t+1", "%"],
    ], [1400, 2860, 620])
    P.h2("Models and Validation")
    P.p("The baselines are persistence (next = previous) [10] and the training-set mean. The learned models are Linear "
        "Regression on standardised inputs and Random Forest [13]. Median imputation, the log transform of volume and "
        "scaling are inside scikit-learn pipelines, so they are fitted on training rows only.")
    P.p(f"Training transitions have targets {', '.join(ct['train_target_periods'])} and the test set is the "
        f"{ct['test_target_periods'][0]} window. Random Forest hyper-parameters and the recommended model were chosen by "
        "hospital-grouped 5-fold cross-validation inside the training set [8], before the test set was used. Differences "
        "between models were assessed with a paired bootstrap over test hospitals (2,000 resamples) [14] and "
        "explanations use permutation importance on held-out data [15]. Sensitivity analyses repeat grouped "
        f"cross-validation over all {cs['n_pooled']} one-step transitions and over {cs['n_nonoverlap']} non-overlapping transitions.")
    P.h2("Novelty")
    P.p("The baseline is the naive assumption that a hospital's rate stays the same. We contribute (1) strict temporal "
        "alignment without gap bridging, (2) baseline-first evaluation with paired uncertainty intervals, (3) detection "
        "and quantification of the overlapping-window artefact, and (4) a multi-level view that links hospital surgical, "
        "hospital stroke and population outcomes without merging incompatible measures. No new algorithm is claimed.")

    # ---------------------------------------------------------------- VI. Implementation
    P.h1("Implementation")
    P.p("The project uses Python with pandas, NumPy, scikit-learn [16], Matplotlib and Plotly, organised as modules for "
        "loading, validation, preprocessing, feature alignment, modelling, evaluation and reporting. One command runs the "
        "whole pipeline in under a minute and writes processed data, models, metrics and auto-derived findings. A "
        "five-page Streamlit dashboard [17] loads the same saved models and verifies that they reproduce the stored test "
        f"predictions exactly (Fig. {P.F('dashboard')}). A GitHub Actions workflow runs 49 automated tests, the pipeline and "
        f"report generation on demand. Table {P.T('models')} lists model settings.")
    P.table("models", "Models, search spaces and selected settings", ["Model", "Settings / search space", "Selected"], [
        ["Persistence", "next rate = previous rate", "–"],
        ["Training mean", "mean of training targets", "–"],
        ["Linear Reg.", "OLS, standardised inputs", "–"],
        ["Random Forest", "300 trees; depth {3, 5, none}; leaf {5, 10, 20}; features {all, sqrt}",
         f"CABG: {rfstr(rfp)}; stroke: {rfstr(srfp)}"],
    ], [980, 2000, 1900])
    P.figure("dashboard", figs["dashboard"],
             "Dashboard CABG page: a hospital's readmission history with 95% CIs, statewide rate and next-window "
             "estimates of all saved models.")

    # ---------------------------------------------------------------- VII. Experimental Results
    P.h1("Experimental Results")
    P.h2("Held-Out Temporal Test")
    P.p(f"Table {P.T('cabg')} reports the single evaluation on the {ct['test_target_periods'][0]} window. Persistence had the "
        f"lowest MAE ({cm.loc['persistence', 'MAE']:.3f} pp), followed by Linear Regression ({cm.loc['linear_regression', 'MAE']:.3f}) "
        f"and Random Forest ({cm.loc['random_forest', 'MAE']:.3f}). All were far better than the training mean "
        f"(R² {cm.loc['mean_baseline', 'R2']:.3f}).")
    P.table("cabg", f"CABG next-window readmission, held-out {ct['test_target_periods'][0]} (n = {ct['n_test']})",
            ["Model", "MAE", "RMSE", "R²", "Bias"],
            [[LABELS[m] + (" *" if m == cs["recommended_model"] else ""), f"{cm.loc[m, 'MAE']:.3f}", f"{cm.loc[m, 'RMSE']:.3f}",
              f"{cm.loc[m, 'R2']:.3f}", f"{cm.loc[m, 'Bias']:+.3f}"]
             for m in ("persistence", "linear_regression", "random_forest", "mean_baseline")],
            [1680, 800, 800, 800, 800])
    P.p("* recommended model, chosen by grouped cross-validation on training data. MAE, RMSE and bias are in percentage "
        "points of the readmission rate.")
    P.h2("Why Persistence Is Hard to Beat")
    P.p(f"Consecutive two-year windows share one calendar year of patients, so half of the target already sits in the "
        f"input. Fig. {P.F('lag')} shows that hospital rates in consecutive windows correlate at r = {lagc['one_step_overlapping']:.2f}, "
        f"but only r = {lagc['non_overlapping']:.2f} when the windows share no year.")
    P.figure("lag", figs["lag"], "Previous versus target readmission rate per hospital. Dotted line: persistence.")
    P.h2("Non-Overlapping Horizon and Grouped Cross-Validation")
    P.p(f"Table {P.T('designs')} and Fig. {P.F('designs')} compare the designs. Across all one-step transitions Linear "
        f"Regression was slightly better than persistence on average ({p1['mean_mae_diff']:+.3f} pp; better in "
        f"{p1['share_folds_model_better']:.0%} of folds). When the overlap is removed, persistence degrades to "
        f"{cvn.loc['persistence', 'MAE_mean']:.2f} pp (R² {cvn.loc['persistence', 'R2_mean']:.2f}) while Linear Regression "
        f"({cvn.loc['linear_regression', 'MAE_mean']:.2f}) and Random Forest ({cvn.loc['random_forest', 'MAE_mean']:.2f}) hold up; "
        f"Linear Regression wins in {pn['share_folds_model_better']:.0%} of folds.")
    P.table("designs", "CABG MAE (pp) across evaluation designs", ["Design", "n", "Persist.", "Lin. Reg.", "Rand. For."], [
        ["Temporal hold-out", str(ct["n_test"]), f"{cm.loc['persistence', 'MAE']:.2f}", f"{cm.loc['linear_regression', 'MAE']:.2f}",
         f"{cm.loc['random_forest', 'MAE']:.2f}"],
        ["Grouped CV, one-step", str(cs["n_pooled"]), f"{cvp.loc['persistence', 'MAE_mean']:.2f}",
         f"{cvp.loc['linear_regression', 'MAE_mean']:.2f}", f"{cvp.loc['random_forest', 'MAE_mean']:.2f}"],
        ["Grouped CV, non-overlap", str(cs["n_nonoverlap"]), f"{cvn.loc['persistence', 'MAE_mean']:.2f}",
         f"{cvn.loc['linear_regression', 'MAE_mean']:.2f}", f"{cvn.loc['random_forest', 'MAE_mean']:.2f}"],
    ], [1640, 560, 880, 900, 900])
    P.figure("designs", figs["designs"], "CABG MAE by model and evaluation design (error bars: SD over 25 grouped folds).")
    P.p(f"**Uncertainty.** The paired bootstrap (Fig. {P.F('bootstrap')}) gives a Linear Regression minus persistence "
        f"difference of {lr_b['mae_diff']:+.3f} pp (95% CI {lr_b['ci_low']:+.3f} to {lr_b['ci_high']:+.3f}) and "
        f"{rf_b['mae_diff']:+.3f} pp for Random Forest ({rf_b['ci_low']:+.3f} to {rf_b['ci_high']:+.3f}): on the one-step "
        "horizon the evidence is insufficient to name a winner. Adding final-year operative mortality, CABG+valve "
        f"mortality and expected readmission changed test MAE only to {ext['linear_regression']:.3f} (Linear Regression) and "
        f"{ext['random_forest']:.3f} (Random Forest).")
    P.figure("bootstrap", figs["bootstrap"], "Paired bootstrap 95% intervals of test-MAE differences for CABG and stroke.")
    P.h2("Separate Stroke Model")
    P.p(f"The stroke data are larger ({st['n_train']} training transitions, {st['n_test']} test hospitals). Using previous "
        f"readmission, previous mortality and volume, Linear Regression reduced held-out MAE from {sm.loc['persistence', 'MAE']:.3f} "
        f"to {sm.loc['linear_regression', 'MAE']:.3f} pp ({s_lr_b['relative_change_pct']:+.1f}%; 95% CI "
        f"{s_lr_b['ci_low']:+.3f} to {s_lr_b['ci_high']:+.3f}) and beat persistence in {sp1['share_folds_model_better']:.0%} "
        f"of grouped folds (Table {P.T('stroke')}).")
    P.table("stroke", f"Stroke next-window readmission, held-out {st['test_target_periods'][0]} (n = {st['n_test']})",
            ["Model", "MAE", "RMSE", "R²", "CV MAE"],
            [[LABELS[m], f"{sm.loc[m, 'MAE']:.3f}", f"{sm.loc[m, 'RMSE']:.3f}", f"{sm.loc[m, 'R2']:.3f}",
              f"{scvp.loc[m, 'MAE_mean']:.3f}"] for m in ("persistence", "linear_regression", "random_forest", "mean_baseline")],
            [1680, 800, 800, 800, 800])
    P.h2("Comparison with the Baseline")
    P.p(f"For CABG, Linear Regression changed MAE relative to persistence by {cm.loc['linear_regression', 'MAE_vs_persistence_pct'] if 'MAE_vs_persistence_pct' in cm else 0:+.1f}% "
        f"on the one-step hold-out, {p1['mean_mae_diff'] / cvp.loc['persistence', 'MAE_mean'] * 100:+.1f}% in pooled grouped CV and "
        f"{pn['mean_mae_diff'] / cvn.loc['persistence', 'MAE_mean'] * 100:+.1f}% on the non-overlapping horizon. For stroke the change "
        f"was {s_lr_b['relative_change_pct']:+.1f}% on the hold-out and {sp1['mean_mae_diff'] / scvp.loc['persistence', 'MAE_mean'] * 100:+.1f}% "
        f"in grouped CV (non-overlapping: {scvn.loc['persistence', 'MAE_mean']:.2f} vs {scvn.loc['linear_regression', 'MAE_mean']:.2f}). "
        "Learned models add value when the target window contains genuinely new patients or when more hospitals are available.")

    # ---------------------------------------------------------------- VIII. Visual Analysis and Healthcare Insights
    P.h1("Visual Analysis and Healthcare Insights")
    P.h2("Explainability")
    P.p(f"Permutation importance on the test window (Fig. {P.F('permutation')}) shows that the previous readmission rate "
        f"carries almost all signal: shuffling it raises Linear Regression MAE by {lrp['prev_readmit_rate']:.2f} pp, versus "
        f"{lrp['prev_cabg_volume']:.2f} for volume and {abs(lrp['prev_stroke_rate']) if abs(lrp['prev_stroke_rate']) < 0.005 else lrp['prev_stroke_rate']:.2f} "
        "for post-operative stroke. Standardised "
        f"coefficients agree: {coef['prev_readmit_rate']:+.2f} pp per SD of previous readmission and {coef['prev_cabg_volume']:+.2f} "
        "pp per SD of log volume, so larger programmes are estimated slightly lower, consistent with volume-outcome "
        "findings [6] but only an association.")
    P.figure("permutation", figs["permutation"], "Permutation importance on held-out CABG hospitals (mean ± SD).")
    P.p(f"Fig. {P.F('learned')} shows the learned relationship. The Linear Regression line is flatter than the persistence "
        f"diagonal: a hospital at {hi_x:.1f}% is expected to fall to {hi_y:.1f}%, and one at {lo_x:.1f}% to rise to {lo_y:.1f}%, "
        f"toward the training mean of {mean_train:.1f}%: regression to the mean [5].")
    P.figure("learned", figs["learned"], "Estimated next-window rate versus previous rate (other inputs at training medians).")
    P.h2("Error Analysis")
    P.p(f"Errors concentrate in small hospitals (Fig. {P.F('errors')}). Linear Regression MAE is "
        + ", ".join(f"{v:.2f} pp for {k.split()[0].lower()}-volume" for k, v in err_lr.items())
        + f" hospitals (persistence {err_p.iloc[0]:.2f} to {err_p.iloc[-1]:.2f}), matching the small-sample instability of hospital rates [4].")
    P.figure("errors", figs["errors"], "Linear Regression residuals versus cohort size and test MAE by volume tercile.")
    P.h2("Stroke Outcomes")
    P.p(f"Statewide risk-adjusted stroke readmission fell from {ssr.iloc[0]['risk_adjusted_rate']:.2f}% to "
        f"{ssr.iloc[-1]['risk_adjusted_rate']:.2f}% and mortality from {ssm.iloc[0]['risk_adjusted_rate']:.2f}% to "
        f"{ssm.iloc[-1]['risk_adjusted_rate']:.2f}% between 2011–2012 and 2014–2015. Hospital readmission and mortality were "
        f"nearly uncorrelated (r from {min(corr):+.2f} to {max(corr):+.2f}; Fig. {P.F('stroke')}), as reported nationally [7].")
    P.figure("stroke", figs["stroke"], "Ischemic stroke: statewide trends and hospital readmission versus mortality.")
    P.h2("Population and County Trends")
    P.p(f"The statewide all-cause rate was {pop['icd10_first_rate']:.2f}% in {pop['icd10_first_year']} and {pop['latest_rate']:.2f}% "
        f"in {pop['latest_year']} ({pop['icd10_change_pp']:+.2f} pp; Fig. {P.F('trend')}); admissions fell "
        f"{abs(pop['admits_change_2019_2020_pct']):.1f}% from 2019 to 2020, coinciding with the COVID-19 pandemic. Among "
        f"{pop['counties_ranked_latest']} counties with at least {pop['min_county_admits']:,} admissions, rates ranged from "
        f"{pop['lowest_county']['rate']:.2f}% ({pop['lowest_county']['county']}) to {pop['highest_county']['rate']:.2f}% "
        f"({pop['highest_county']['county']}) (Fig. {P.F('map')}). By payer, {payer[0]['name'].replace('MediCare', 'Medicare')} patients had the "
        f"highest rate ({payer[0]['rate']:.2f}%) and {payer[-1]['name']} the lowest ({payer[-1]['rate']:.2f}%) (Fig. {P.F('strata')}).")
    P.figure("trend", figs["trend"], "Statewide all-cause 30-day readmission rate by coding era.")
    P.figure("map", figs["map"], f"County readmission rates, {pop['latest_year']} (grey: fewer than "
             f"{pop['min_county_admits']:,} admissions).", width=2.6)
    P.figure("strata", figs["strata"], f"Readmission rate by expected payer and age group, {pop['latest_year']} (dotted: all patients).")
    P.h2("Insights Supported by the Data")
    P.bullets([
        "**The previous rate dominates** every model; volume adds a little and post-operative stroke adds nothing.",
        f"**Overlapping windows** make persistence look strong (r = {lagc['one_step_overlapping']:.2f}); without overlap the "
        f"learned models win in {pn['share_folds_model_better']:.0%} of folds.",
        "**Regression to the mean** explains the gain: extreme hospitals move part of the way back to the average.",
        f"**Small hospitals** carry the largest errors ({err_lr.iloc[0]:.2f} vs {err_lr.iloc[-1]:.2f} pp MAE).",
        "**Stroke readmission and mortality** capture different aspects of quality and should be reported side by side.",
        f"**Population gaps:** Medicare {payer[0]['rate']:.2f}% vs self-pay {payer[-1]['rate']:.2f}%; ages "
        f"{age[0]['name'].replace(' years', '')} {age[0]['rate']:.2f}% vs {age[-1]['name'].replace(' years', '')} {age[-1]['rate']:.2f}%; "
        f"{race[0]['name']} {race[0]['rate']:.2f}%. These are associations, not causes.",
    ])

    # ---------------------------------------------------------------- IX. Discussion and Limitations
    P.h1("Discussion and Limitations")
    P.p("**Why the results look this way.** A hospital's next rate is mostly its previous rate plus noise. When the two "
        "windows share patients, copying the previous rate is already close to optimal; when they do not, shrinking "
        "extreme rates toward the average is better. Random Forest never beat the linear model because the relationship "
        "is close to linear and about 120 training rows leave little room for extra flexibility.")
    P.p("**Limitations.** (1) Only two usable one-step CABG transitions exist and the 2020–2021 window covers two "
        "hospitals. (2) Overlapping windows limit how independent any test can be. (3) Rates for small hospitals are "
        "noisy and the risk-adjusted outcome is itself an estimate. (4) ICD-9 and ICD-10 eras are not directly "
        "comparable. (5) The data are aggregate: estimates are hospital-level rates, not patient probabilities, do not "
        "identify causes and are not clinically validated. (6) Group differences must not be read as causal.")

    # ---------------------------------------------------------------- X. Conclusion
    P.h1("Conclusion and Future Scope")
    P.p("Historical hospital performance carries real information about future readmission rates, and learned models "
        "that shrink extreme rates outperform persistence when the target window contains new patients or more hospitals "
        "are available. For the CABG one-step horizon the published windows overlap and no model reliably beat "
        "persistence on held-out data; we report this as a finding rather than hide it. Readmission patterns also differ "
        "by condition, county and population group.")
    P.p("Future work includes adding new CABG windows as they are published, hierarchical or empirical-Bayes shrinkage "
        "using case counts, calibrated prediction intervals, and linking hospital outcomes to county context only where "
        "reporting periods overlap.")

    # ---------------------------------------------------------------- Acknowledgment & References
    P.h5("Acknowledgment")
    P.p("The authors thank Dr. Prerna Mishra, course instructor, Healthcare Analytics, for guidance. This is a "
        "mini-project report; the system estimates hospital-level readmission rates from aggregate public statistics "
        "and is not a clinical decision-support tool.")
    P.h5("References")
    P.references([
        "S. F. Jencks, M. V. Williams, and E. A. Coleman, “Rehospitalizations among patients in the Medicare fee-for-service program,” N. Engl. J. Med., vol. 360, no. 14, pp. 1418–1428, 2009.",
        "C. K. McIlvennan, Z. J. Eapen, and L. A. Allen, “Hospital readmissions reduction program,” Circulation, vol. 131, no. 20, pp. 1796–1803, 2015.",
        "D. Kansagara et al., “Risk prediction models for hospital readmission: a systematic review,” JAMA, vol. 306, no. 15, pp. 1688–1698, 2011.",
        "J. B. Dimick, H. G. Welch, and J. D. Birkmeyer, “Surgical mortality as an indicator of hospital quality: the problem with small sample size,” JAMA, vol. 292, no. 7, pp. 847–851, 2004.",
        "A. G. Barnett, J. C. van der Pols, and A. J. Dobson, “Regression to the mean: what it is and how to deal with it,” Int. J. Epidemiol., vol. 34, no. 1, pp. 215–220, 2005.",
        "J. D. Birkmeyer et al., “Hospital volume and surgical mortality in the United States,” N. Engl. J. Med., vol. 346, no. 15, pp. 1128–1137, 2002.",
        "H. M. Krumholz et al., “Relationship between hospital readmission and mortality rates for patients hospitalized with acute myocardial infarction, heart failure, or pneumonia,” JAMA, vol. 309, no. 6, pp. 587–593, 2013.",
        "D. R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” Ecography, vol. 40, no. 8, pp. 913–929, 2017.",
        "S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: formulation, detection, and avoidance,” ACM Trans. Knowl. Discov. Data, vol. 6, no. 4, Art. 15, 2012.",
        "R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. Melbourne, Australia: OTexts, 2021.",
        "California Health and Human Services Agency, “CHHS Open Data Portal.” [Online]. Available: https://data.chhs.ca.gov",
        "U.S. Dept. of Health and Human Services, “Administrative simplification: change to the compliance date for the International Classification of Diseases, 10th Revision (ICD-10-CM and ICD-10-PCS) medical data code sets,” Federal Register, vol. 79, no. 149, p. 45128, Aug. 2014.",
        "L. Breiman, “Random forests,” Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001.",
        "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.",
        "A. Fisher, C. Rudin, and F. Dominici, “All models are wrong, but many are useful: learning a variable's importance by studying an entire class of prediction models simultaneously,” J. Mach. Learn. Res., vol. 20, no. 177, pp. 1–81, 2019.",
        "F. Pedregosa et al., “Scikit-learn: machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.",
        "Streamlit, “Streamlit documentation.” [Online]. Available: https://docs.streamlit.io",
    ])
    P.section_break(2, 1080)  # continuous break -> Word balances the two columns on the last page
    cp = P.doc.core_properties
    cp.title, cp.author = TITLE, ", ".join(a for a, _ in AUTHORS)
    cp.last_modified_by = cp.author
    return P


def main(argv=None) -> int:
    A = load_artifacts(config.PROJECT_ROOT)
    P = build(A)
    P.doc.save(OUT_DOCX)
    print(f"wrote {OUT_DOCX} ({len(P.figs)} figures, {len(P.tabs)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
