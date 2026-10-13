"""Matplotlib figures for the written report (light print theme), generated from pipeline artifacts.

Every function writes one PNG and returns its path. Values come from artifacts/ - nothing is typed in.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

from . import config  # noqa: E402
from .modeling import MODEL_LABELS  # noqa: E402
from .theme import CODING_COLORS_LIGHT, LIGHT, MEASURE_COLORS_LIGHT, MODEL_COLORS_LIGHT, SERIES_LIGHT  # noqa: E402

DPI = 200
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "axes.edgecolor": LIGHT["axis"], "axes.labelcolor": LIGHT["text_2"],
    "xtick.color": LIGHT["text_2"], "ytick.color": LIGHT["text_2"], "axes.grid": True, "grid.color": LIGHT["grid"],
    "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.titlelocation": "left",
    "axes.titlecolor": LIGHT["text"], "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})
SEQ = ["#dbe9f6", "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def _pct(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ======================================================================================
# Diagrams
# ======================================================================================
def _box(ax, x, y, w, h, text, fc="#eef4fb", ec=LIGHT["navy"], fs=8.6, bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or LIGHT["text"],
            fontweight="bold" if bold else "normal", wrap=True, linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2, color=LIGHT["text_2"], style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, color=color, lw=1.2,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


def architecture_diagram(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.02, 1.03); ax.set_ylim(-0.02, 1.0); ax.axis("off")
    nv = LIGHT["navy"]
    _box(ax, 0.00, 0.76, 0.17, 0.17, "data/raw/\n4 CSV files\n(never modified)", fc="#fff4e5", ec="#c98500")
    _box(ax, 0.23, 0.47, 0.31, 0.48,
         "run_pipeline.py\n(shared src/ package)\n\n1  load + validate schemas\n2  clean & normalise\n"
         "3  audit + data dictionary\n4  CABG temporal experiment\n5  stroke experiment\n6  population summaries\n"
         "7  synthetic scenario tests\n8  findings + integrity check", fs=8.2)
    _box(ax, 0.60, 0.74, 0.18, 0.21, "models/\nartifacts/\ndata/processed/", fc="#e8f6ef", ec="#1b7a4a")
    _box(ax, 0.83, 0.74, 0.17, 0.21, "generate_report.py\n-> Project_Report\n.docx and .pdf", fc="#f3effe", ec="#5b4fc7", fs=8.2)
    _arrow(ax, 0.17, 0.845, 0.23, 0.845); _arrow(ax, 0.54, 0.845, 0.60, 0.845); _arrow(ax, 0.78, 0.845, 0.83, 0.845)
    _box(ax, 0.60, 0.42, 0.18, 0.22, "app.py + pages/\nStreamlit dashboard\n(loads the SAME\nsaved models)", fc="#e6f6fb", ec="#0e7fa0", fs=8.2)
    _arrow(ax, 0.69, 0.74, 0.69, 0.64)
    _box(ax, 0.83, 0.42, 0.17, 0.22, "Streamlit\nCommunity Cloud\n-> public URL", fc="#e6f6fb", ec="#0e7fa0", bold=True, fs=8.4)
    _arrow(ax, 0.78, 0.53, 0.83, 0.53)
    _box(ax, 0.00, 0.47, 0.17, 0.2, "Local VS Code\nrun_local.py:\npipeline, then\nstreamlit", fc="#f1f5f9", ec=nv, fs=8.2)
    _arrow(ax, 0.17, 0.57, 0.23, 0.60)
    _box(ax, 0.23, 0.22, 0.31, 0.16, "GitHub repository\ncode + data/raw + committed artifacts", fc="#f1f5f9", ec=nv, fs=8.4)
    _arrow(ax, 0.385, 0.47, 0.385, 0.38)
    _arrow(ax, 0.54, 0.30, 0.90, 0.42, ls="--")
    ax.text(0.70, 0.27, "push to main -> Cloud redeploys", fontsize=7.8, color=LIGHT["text_2"], ha="center")
    _box(ax, 0.23, 0.00, 0.31, 0.16, "GitHub Actions: pipeline.yml\n(workflow_dispatch = Run workflow)\ntests -> pipeline -> report -> upload",
         fc="#fdf2f7", ec="#c2416f", fs=8)
    _arrow(ax, 0.385, 0.22, 0.385, 0.16)
    _arrow(ax, 0.54, 0.08, 0.60, 0.08)
    _box(ax, 0.60, 0.00, 0.40, 0.16, "Workflow artifacts (downloadable zip)\nOptional: commit refreshed artifacts to main,\nwhich then triggers a Cloud redeploy",
         fc="#fdf2f7", ec="#c2416f", fs=8)
    return _save(fig, out, "fig01_architecture.png")


def preprocessing_diagram(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    lanes = [
        ("A  CABG", ["cp1252 decode", "'.' -> NaN, strip\nthousand separators", "fix line breaks,\ndashes, rating typo",
                     "stable Hospital ID;\nlatest name", "panel: measures side\nby side per window"], "#0e7fa0"),
        ("B  Stroke", ["cp1252 decode", "statewide rows &\n'AAAA' -> NaN", "swap transposed\n2013-2014 counts",
                       "parse lat/lon", "panel: readmission +\nmortality per window"], "#d9661f"),
        ("C  All-cause", ["strip column\nnames", "parse '13.80%',\n'1,877,198'", "one rate column +\ncoding_system label",
                          "flag 2015 partial;\nkeep suppressed NaN", "state / county /\nstrata tables"], "#5b4fc7"),
        ("D  Synthetic", ["validate flag\n= True", "SYNTH_ prefix,\nno ID collisions", "map to model\nfeature schema",
                          "isolate from all\nreal-data steps", "scenario & smoke\ntests only"], "#c2416f"),
    ]
    for i, (name, steps, col) in enumerate(lanes):
        y = 0.78 - i * 0.245
        ax.text(0.0, y + 0.075, name, fontsize=9.5, fontweight="bold", color=col, va="center")
        for j, s in enumerate(steps):
            x = 0.13 + j * 0.175
            _box(ax, x, y, 0.15, 0.15, s, fc="white", ec=col, fs=7.6)
            if j:
                _arrow(ax, x - 0.025, y + 0.075, x, y + 0.075, color=col)
    return _save(fig, out, "fig03_preprocessing_workflow.png")


def model_workflow_diagram(art: dict, out: Path) -> Path:
    s = art["cabg_summary"]["train_summary"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(-0.02, 1.03); ax.set_ylim(-0.02, 1.0); ax.axis("off")
    _box(ax, 0.00, 0.58, 0.15, 0.32, "Hospital x\nwindow panel\n(readmission,\nstroke, volume)", fs=8.2)
    _box(ax, 0.19, 0.58, 0.17, 0.32, "Transitions\ninputs: window t\ntarget: window t+1\n(start year + 1 exactly;\nno gap bridging)", fs=8)
    _box(ax, 0.41, 0.72, 0.19, 0.2, f"TRAIN  n = {s['n_train']}\ntargets\n{', '.join(s['train_target_periods'])}",
         fc="#e8f6ef", ec="#1b7a4a", fs=8)
    _box(ax, 0.41, 0.42, 0.19, 0.2, f"TEST (held out)\nn = {s['n_test']}\ntarget {s['test_target_periods'][0]}",
         fc="#fdecea", ec="#b42318", fs=8)
    _box(ax, 0.65, 0.72, 0.16, 0.2, "Grouped 5-fold CV\n(by hospital):\nRF tuning +\nmodel selection", fc="#e8f6ef", ec="#1b7a4a", fs=7.8)
    _box(ax, 0.85, 0.72, 0.15, 0.2, "Fit all 4 models\non all training\nrows", fc="#e8f6ef", ec="#1b7a4a", fs=7.8)
    _box(ax, 0.65, 0.42, 0.35, 0.2, "One-time evaluation: MAE, RMSE, R², bias;\npaired bootstrap vs persistence;\n"
                                     "permutation importance; residuals", fc="#fdecea", ec="#b42318", fs=7.8)
    _box(ax, 0.41, 0.03, 0.59, 0.27, "Sensitivity analyses (reported separately):\n- repeated grouped CV over all one-step transitions\n"
                                     "- non-overlapping horizon (window t -> t+2)\n- extended feature set", fc="#f3effe", ec="#5b4fc7", fs=7.8)
    _box(ax, 0.00, 0.03, 0.36, 0.27, "Saved: models/cabg_models.joblib\n+ feature schema (JSON)\n-> dashboard reloads them and\n"
                                     "verifies identical predictions", fc="#e6f6fb", ec="#0e7fa0", fs=7.8)
    _arrow(ax, 0.15, 0.74, 0.19, 0.74); _arrow(ax, 0.36, 0.78, 0.41, 0.82); _arrow(ax, 0.36, 0.68, 0.41, 0.54)
    _arrow(ax, 0.60, 0.82, 0.65, 0.82); _arrow(ax, 0.81, 0.82, 0.85, 0.82); _arrow(ax, 0.925, 0.72, 0.925, 0.62)
    _arrow(ax, 0.60, 0.52, 0.65, 0.52)
    return _save(fig, out, "fig04_model_workflow.png")


# ======================================================================================
# Data figures
# ======================================================================================
def coverage_timeline(art: dict, out: Path) -> Path:
    cabg_sw, stroke_sw, state = art["cabg_statewide"], art["stroke_statewide"], art["pop_state"]
    rows = []
    r = cabg_sw[cabg_sw["measure_code"] == "readmission"]
    for _, x in r.iterrows():
        rows.append((f"CABG readmission/stroke {x['period']}", x["period_start"], 2, SERIES_LIGHT[0],
                     f"{int(x['hospitals'])} hosp.", 0.35 if x["hospitals"] < 10 else 1))
    for _, x in cabg_sw[cabg_sw["measure_code"] == "iso_mortality"].iterrows():
        rows.append(("CABG op. mortality (single years)", x["period_start"], 1, SERIES_LIGHT[5], "", 1))
    for _, x in stroke_sw[stroke_sw["measure_code"] == "readmission"].iterrows():
        rows.append((f"Ischemic stroke {x['period']}", x["period_start"], 2, SERIES_LIGHT[1], "", 1))
    for _, x in state.iterrows():
        rows.append((f"All-cause ({x['coding_system']})", x["year"], 1, CODING_COLORS_LIGHT[x["coding_system"]], "",
                     0.4 if x["coding_transition_year"] else 1))
    lanes = list(dict.fromkeys(r_[0] for r_ in rows))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for lane, start, yrs, col, note, alpha in rows:
        y = len(lanes) - 1 - lanes.index(lane)
        ax.barh(y, yrs - 0.12, left=start - 0.44, height=0.6, color=col, alpha=alpha)
        if note:
            ax.text(start + yrs - 0.5, y, " " + note, va="center", fontsize=7.5, color=LIGHT["text_2"])
    ax.set_yticks(range(len(lanes))); ax.set_yticklabels(lanes[::-1], fontsize=8)
    ax.set_xticks(range(2011, 2025)); ax.set_xlim(2010.4, 2025.2)
    ax.set_xlabel("Calendar year covered"); ax.grid(axis="y", visible=False)
    ax.set_title("Reporting periods available in the four real-data sources")
    return _save(fig, out, "fig02_data_coverage.png")


def allcause_trend(art: dict, out: Path) -> Path:
    state = art["pop_state"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for code in ("ICD-9", "ICD-10"):
        g = state[state["coding_system"] == code]
        ax.plot(g["year"], g["rate"], "-o", color=CODING_COLORS_LIGHT[code], lw=2, ms=5, label=f"{code} era")
        for _, x in g.iterrows():
            ax.annotate(f"{x['rate']:.2f}", (x["year"], x["rate"]), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.2, color=LIGHT["text_2"])
    part = state[state["coding_transition_year"]]
    ax.plot(part["year"], part["rate"], "o", mfc="white", mec=CODING_COLORS_LIGHT["ICD-9"], ms=7, mew=2,
            label="2015: partial year (final ICD-9 year)")
    ax.axvline(2015.5, color=LIGHT["muted"], lw=1, ls=":")
    ax.text(2015.6, ax.get_ylim()[1], "ICD-9 -> ICD-10", va="top", fontsize=8, color=LIGHT["muted"])
    ax.set_xticks(range(2011, 2025)); _pct(ax)
    ax.set_ylabel("30-day readmission rate (%)")
    ax.set_title("California all-cause unplanned 30-day readmission rate, all patients")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out, "fig05_allcause_trend.png")


def county_map(art: dict, out: Path) -> Path:
    county, pop = art["pop_county"], art["pop_summary"]
    year, thr = pop["latest_year"], pop["min_county_admits"]
    gj = json.loads(config.COUNTY_GEOJSON.read_text(encoding="utf-8"))
    d = county[county["year"] == year].set_index("county")
    ok = d[(d["admits"] >= thr) & d["rate"].notna()]
    vmin, vmax = float(ok["rate"].min()), float(ok["rate"].max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    norm = matplotlib.colors.Normalize(vmin, vmax)
    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    patches, colors = [], []
    for f in gj["features"]:
        name = f["properties"]["NAME"]
        polys = f["geometry"]["coordinates"]
        if f["geometry"]["type"] == "Polygon":
            polys = [polys]
        col = cmap(norm(ok.loc[name, "rate"])) if name in ok.index else "#d9dce1"
        for poly in polys:
            patches.append(Polygon(np.asarray(poly[0]), closed=True))
            colors.append(col)
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="white", linewidth=0.5))
    ax.set_xlim(-124.6, -114); ax.set_ylim(32.4, 42.1); ax.set_aspect(1.2); ax.axis("off")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cb.set_label("30-day readmission rate (%)", fontsize=8); cb.outline.set_visible(False)
    ax.set_title(f"County readmission rate, {year}\n(grey: < {thr:,} admissions)", fontsize=10)
    return _save(fig, out, "fig06_county_map.png")


def county_ranking(art: dict, out: Path) -> Path:
    county, pop = art["pop_county"], art["pop_summary"]
    year, thr = pop["latest_year"], pop["min_county_admits"]
    d = county[(county["year"] == year) & (county["admits"] >= thr)].dropna(subset=["rate"]).sort_values("rate")
    sel = pd.concat([d.head(10), d.tail(10)])
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    cols = [SERIES_LIGHT[5]] * 10 + [SERIES_LIGHT[1]] * 10
    ax.barh(sel["county"], sel["rate"], color=cols, height=0.7)
    ax.axvline(pop["latest_rate"], color=LIGHT["text"], lw=1.2, ls=":")
    ax.text(pop["latest_rate"], len(sel) - 0.3, f" California {pop['latest_rate']:.2f}%", fontsize=8, va="bottom")
    for i, v in enumerate(sel["rate"]):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=7.4, color=LIGHT["text_2"])
    _pct(ax, "x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("30-day readmission rate (%)")
    ax.set_title(f"Lowest and highest 10 counties, {year}\n(counties with >= {thr:,} admissions)", fontsize=10)
    return _save(fig, out, "fig07_county_ranking.png")


def strata_panels(art: dict, out: Path) -> Path:
    strata, pop = art["pop_strata"], art["pop_summary"]
    year = pop["latest_year"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for ax, cat in zip(axes.ravel(), ["Expected Payer", "Age", "Race-Ethnicity", "Sex"]):
        g = strata[(strata["strata"] == cat) & (strata["year"] == year)].dropna(subset=["rate"]).sort_values("rate")
        ax.barh(g["strata_name"], g["rate"], color=SERIES_LIGHT[0], height=0.65)
        ax.axvline(pop["latest_rate"], color=LIGHT["text"], lw=1, ls=":")
        for i, (v, n) in enumerate(zip(g["rate"], g["admits"])):
            ax.text(v + 0.2, i, f"{v:.2f}%  (n={n/1000:,.0f}k)", va="center", fontsize=7.2, color=LIGHT["text_2"])
        ax.set_xlim(0, g["rate"].max() * 1.45); _pct(ax, "x"); ax.grid(axis="y", visible=False)
        ax.set_title(cat, fontsize=10)
    fig.suptitle(f"30-day readmission rate by population group, California {year} (dotted line = all patients)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "fig08_population_strata.png")


def cabg_distribution(art: dict, out: Path) -> Path:
    panel = art["cabg_panel"]
    periods = [p for p in sorted(panel["period"].unique()) if panel[panel["period"] == p]["readmit_rate"].notna().sum() >= 10]
    data = [panel.loc[panel["period"] == p, "readmit_rate"].dropna() for p in periods]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bp = ax.boxplot(data, tick_labels=[f"{p}\n(n={len(d)})" for p, d in zip(periods, data)], widths=0.5, patch_artist=True,
                    medianprops=dict(color=LIGHT["navy"], lw=1.5), flierprops=dict(marker="o", ms=3, mfc=LIGHT["muted"], mec="none"))
    for b in bp["boxes"]:
        b.set(facecolor="#dbeef5", edgecolor=SERIES_LIGHT[0])
    sw = art["cabg_statewide"]
    sw = sw[(sw["measure_code"] == "readmission") & sw["period"].isin(periods)]
    ax.plot(range(1, len(periods) + 1), sw["rate"], "D", color=SERIES_LIGHT[1], ms=6, label="Statewide observed rate")
    _pct(ax); ax.set_ylabel("Hospital readmission RAAER (%)"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Distribution of hospital isolated CABG 30-day readmission rates by window")
    return _save(fig, out, "fig09_cabg_distribution.png")


def lag_scatter(art: dict, out: Path) -> Path:
    tr1, tr2 = art["cabg_transitions"], art["cabg_transitions_nonoverlap"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, df, title in ((axes[0], tr1, "Consecutive windows (share 1 calendar year)"),
                          (axes[1], tr2, "Non-overlapping windows (t -> t+2)")):
        d = df.dropna(subset=["prev_readmit_rate", "next_readmit_rate"])
        r = d["prev_readmit_rate"].corr(d["next_readmit_rate"])
        ax.scatter(d["prev_readmit_rate"], d["next_readmit_rate"], s=14, color=SERIES_LIGHT[0], alpha=0.65, edgecolor="none")
        lim = [0, max(d["prev_readmit_rate"].max(), d["next_readmit_rate"].max()) + 1]
        ax.plot(lim, lim, ls=":", color=LIGHT["muted"], lw=1)
        ax.set_title(f"{title}\nPearson r = {r:.2f}, n = {len(d)}", fontsize=9.5)
        ax.set_xlabel("Previous-window readmission RAAER (%)"); _pct(ax); _pct(ax, "x")
    axes[0].set_ylabel("Target-window readmission RAAER (%)")
    return _save(fig, out, "fig10_lag_correlation.png")


# ======================================================================================
# Model figures
# ======================================================================================
def design_comparison(art: dict, prefix: str, out: Path) -> Path:
    m, cvp, cvn = art[f"{prefix}_metrics"], art[f"{prefix}_cv_pooled"], art[f"{prefix}_cv_nonoverlap"]
    s = art[f"{prefix}_summary"]
    designs = [(f"Temporal hold-out\n(n={s['train_summary']['n_test']})", m.set_index("model")["MAE"], None),
               (f"Grouped CV, one-step\n(n={s['n_pooled']})", cvp.set_index("model")["MAE_mean"], cvp.set_index("model")["MAE_sd"]),
               (f"Grouped CV, non-overlapping\n(n={s['n_nonoverlap']})", cvn.set_index("model")["MAE_mean"], cvn.set_index("model")["MAE_sd"])]
    models = ["persistence", "linear_regression", "random_forest", "mean_baseline"]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    w = 0.2
    for i, mod in enumerate(models):
        xs = np.arange(len(designs)) + (i - 1.5) * w
        ys = [d[1][mod] for d in designs]
        err = [d[2][mod] if d[2] is not None else 0 for d in designs]
        ax.bar(xs, ys, w * 0.92, yerr=err, color=MODEL_COLORS_LIGHT[mod], label=MODEL_LABELS[mod],
               error_kw=dict(ecolor=LIGHT["text_2"], lw=0.9, capsize=2.5))
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom", fontsize=6.8, color=LIGHT["text_2"], rotation=90)
    ax.set_xticks(range(len(designs))); ax.set_xticklabels([d[0] for d in designs])
    ax.set_ylabel("MAE (percentage points)"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.set_ylim(0, max(max(d[1].max() for d in designs) * 1.25, 1))
    return _save(fig, out, f"fig11_{prefix}_design_comparison.png")


def bootstrap_forest(art: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharex=False)
    short = {"linear_regression": "Linear Reg.", "random_forest": "Random Forest", "persistence": "Persistence",
             "mean_baseline": "Training mean"}
    for ax, prefix, title in ((axes[0], "cabg", "CABG (test n = %d)"), (axes[1], "stroke", "Stroke (test n = %d)")):
        b = art[f"{prefix}_bootstrap"].iloc[::-1]
        labels = [f"{short[a]} - {short[r]}" for a, r in zip(b["model"], b["reference"])]
        cols = [MODEL_COLORS_LIGHT[m] for m in b["model"]]
        y = np.arange(len(b))
        ax.errorbar(b["mae_diff"], y, xerr=[b["mae_diff"] - b["ci_low"], b["ci_high"] - b["mae_diff"]], fmt="none",
                    ecolor=LIGHT["text_2"], capsize=3, lw=1.2)
        ax.scatter(b["mae_diff"], y, color=cols, s=40, zorder=3)
        ax.axvline(0, color=LIGHT["text"], lw=1)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title % art[f"{prefix}_summary"]["train_summary"]["n_test"], fontsize=10)
        ax.set_xlabel("MAE difference (pp), 95% bootstrap CI\n(< 0: first model more accurate)", fontsize=8.5)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out, "fig12_bootstrap.png")


def actual_vs_predicted(art: dict, prefix: str, out: Path) -> Path:
    p = art[f"{prefix}_test_predictions"]
    models = ["persistence", "linear_regression", "random_forest"]
    lo = float(min(p["next_readmit_rate"].min(), *(p[f"pred_{m}"].min() for m in models))) - 1
    hi = float(max(p["next_readmit_rate"].max(), *(p[f"pred_{m}"].max() for m in models))) + 1
    met = art[f"{prefix}_metrics"].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    for ax, m in zip(axes, models):
        ax.scatter(p[f"pred_{m}"], p["next_readmit_rate"], s=13, color=MODEL_COLORS_LIGHT[m], alpha=0.7, edgecolor="none")
        ax.plot([lo, hi], [lo, hi], ls=":", color=LIGHT["muted"], lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_title(f"{MODEL_LABELS[m]}\nMAE {met.loc[m, 'MAE']:.2f} | R² {met.loc[m, 'R2']:.2f}", fontsize=9.5)
        ax.set_xlabel("Predicted (%)")
    axes[0].set_ylabel("Actual (%)")
    return _save(fig, out, f"fig13_{prefix}_actual_vs_predicted.png")


def residuals(art: dict, out: Path) -> Path:
    p, err = art["cabg_test_predictions"], art["cabg_error_by_volume"]
    rec = "linear_regression"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    res = p["next_readmit_rate"] - p[f"pred_{rec}"]
    ax.scatter(p["target_cases"], res, s=14, color=MODEL_COLORS_LIGHT[rec], alpha=0.7, edgecolor="none")
    ax.axhline(0, color=LIGHT["text"], lw=1); ax.set_xscale("log")
    ax.set_xlabel("Eligible isolated CABG cases in target window (log scale)")
    ax.set_ylabel("Actual - predicted (pp)")
    ax.set_title("Linear Regression residuals vs hospital cohort size", fontsize=10)
    ax = axes[1]
    terc = list(dict.fromkeys(err["volume_tercile"]))
    w = 0.26
    for i, m in enumerate(["persistence", "linear_regression", "random_forest"]):
        s = err[err["model"] == m].set_index("volume_tercile").loc[terc]
        ax.bar(np.arange(len(terc)) + (i - 1) * w, s["MAE"], w * 0.92, color=MODEL_COLORS_LIGHT[m], label=MODEL_LABELS[m])
    rng = err.drop_duplicates("volume_tercile").set_index("volume_tercile").loc[terc, "volume_range"]
    ax.set_xticks(range(len(terc))); ax.set_xticklabels([f"{t}\n({r})" for t, r in zip(terc, rng)], fontsize=8)
    ax.set_ylabel("Test MAE (pp)"); ax.legend(fontsize=7.5); ax.grid(axis="x", visible=False)
    ax.set_title("Test error by cohort-size tercile", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig14_residuals.png")


def permutation(art: dict, out: Path) -> Path:
    labels = {"prev_readmit_rate": "Previous readmission rate", "prev_stroke_rate": "Previous post-op stroke rate",
              "prev_cabg_volume": "Previous CABG volume", "prev_mortality_rate": "Previous mortality rate",
              "prev_stroke_volume": "Previous stroke volume"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.9))
    for ax, prefix, title in ((axes[0], "cabg", "CABG model"), (axes[1], "stroke", "Stroke model")):
        perm = art[f"{prefix}_permutation"]
        feats = list(dict.fromkeys(perm["feature"]))
        for i, m in enumerate(["linear_regression", "random_forest"]):
            s = perm[perm["model"] == m].set_index("feature").loc[feats]
            y = np.arange(len(feats)) + (i - 0.5) * 0.36
            ax.barh(y, s["importance_mae_increase"], 0.33, xerr=s["importance_sd"], color=MODEL_COLORS_LIGHT[m],
                    label=MODEL_LABELS[m], error_kw=dict(ecolor=LIGHT["text_2"], lw=0.8, capsize=2))
        ax.set_yticks(range(len(feats))); ax.set_yticklabels([labels.get(f, f) for f in feats], fontsize=8.5)
        ax.invert_yaxis(); ax.axvline(0, color=LIGHT["text"], lw=0.8); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Increase in test MAE when shuffled (pp)"); ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, out, "fig15_permutation_importance.png")


def learned_curve(art: dict, out: Path) -> Path:
    c, tr = art["cabg_learned_curve"], art["cabg_transitions"]
    s = art["cabg_summary"]["train_summary"]
    train = tr[tr["target_period"].isin(s["train_target_periods"]) & tr["primary_features_complete"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(train["prev_readmit_rate"], train["next_readmit_rate"], s=12, color="#cbd5e1", label="Training transitions")
    for m, ls in (("persistence", ":"), ("mean_baseline", "--"), ("linear_regression", "-"), ("random_forest", "-")):
        ax.plot(c["prev_readmit_rate"], c[f"pred_{m}"], ls=ls, lw=2.2 if ls == "-" else 1.4, color=MODEL_COLORS_LIGHT[m],
                label=MODEL_LABELS[m])
    _pct(ax); _pct(ax, "x")
    ax.set_xlabel("Previous-window readmission RAAER (%)"); ax.set_ylabel("Estimated next-window RAAER (%)")
    ax.set_title("Learned relationship (other inputs at training medians)")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out, "fig16_learned_curve.png")


def hospital_examples(art: dict, out: Path) -> Path:
    hp, sw = art["cabg_hospital_predictions"], art["cabg_statewide"]
    sw = sw[(sw["measure_code"] == "readmission") & (sw["hospitals"] >= 10)]
    test = hp[hp["status"].str.startswith("Test")]
    vol = art["cabg_panel"].groupby("hospital_id")["cabg_volume"].sum()
    picks = vol[vol.index.isin(test["hospital_id"])].sort_values(ascending=False).index[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharey=False)
    for ax, hid in zip(axes.ravel(), picks):
        h = hp[hp["hospital_id"] == hid].sort_values("input_period")
        obs = art["cabg_panel"][art["cabg_panel"]["hospital_id"] == hid].sort_values("period_start")
        t = h[h["status"].str.startswith("Test")].iloc[0]
        order = sorted(set(obs["period"]) | {t["target_period"]})
        xi = {p: i for i, p in enumerate(order)}
        ax.plot([xi[p] for p in sw["period"] if p in xi], [r for p, r in zip(sw["period"], sw["rate"]) if p in xi],
                ls=":", color=LIGHT["muted"], lw=1.2, label="Statewide rate")
        o = obs.dropna(subset=["readmit_rate"])
        ax.errorbar([xi[p] for p in o["period"]], o["readmit_rate"],
                    yerr=[o["readmit_rate"] - o["readmit_ci_low"], o["readmit_ci_high"] - o["readmit_rate"]],
                    fmt="-o", color=LIGHT["text"], ecolor="#cbd5e1", lw=1.8, ms=4.5, capsize=2, label="Observed RAAER (95% CI)")
        x_t = xi[t["target_period"]]
        for m, mk in (("persistence", "o"), ("linear_regression", "D"), ("random_forest", "s")):
            ax.scatter([x_t + {"persistence": -0.12, "linear_regression": 0, "random_forest": 0.12}[m]], [t[f"pred_{m}"]],
                       marker=mk, s=42, color=MODEL_COLORS_LIGHT[m], zorder=4, label=f"Estimate: {MODEL_LABELS[m]}",
                       facecolors="white" if m == "persistence" else None)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=7.5)
        ax.set_title(f"{str(t['hospital_name'])[:46]}", fontsize=9)
        _pct(ax); ax.set_ylim(bottom=0)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Back-test on held-out hospitals: history, estimates for 2022-2023, and the published outcome",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return _save(fig, out, "fig17_cabg_hospital_examples.png")


def stroke_panels(art: dict, out: Path) -> Path:
    sw, panel = art["stroke_statewide"], art["stroke_panel"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    for meas, col in MEASURE_COLORS_LIGHT.items():
        s = sw[sw["measure"] == meas]
        ax.plot(s["period"], s["risk_adjusted_rate"], "-o", color=col, lw=2, ms=5, label=meas)
        for x, y in zip(s["period"], s["risk_adjusted_rate"]):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)
    _pct(ax); ax.set_ylim(8, 14); ax.legend(fontsize=8); ax.set_ylabel("Statewide risk-adjusted rate (%)")
    ax.set_title("Statewide ischemic stroke outcomes", fontsize=10); ax.tick_params(axis="x", labelsize=8)
    ax = axes[1]
    last = sorted(panel["period"].unique())[-1]
    d = panel[panel["period"] == last].dropna(subset=["readmit_rate", "mortality_rate"])
    r = d["readmit_rate"].corr(d["mortality_rate"])
    ax.scatter(d["mortality_rate"], d["readmit_rate"], s=np.clip(d["readmit_cases"].fillna(10) / 6, 6, 60),
               color=SERIES_LIGHT[0], alpha=0.55, edgecolor="none")
    _pct(ax); _pct(ax, "x")
    ax.set_xlabel("30-day mortality (%)"); ax.set_ylabel("30-day readmission (%)")
    ax.set_title(f"Hospitals, {last}: r = {r:+.2f} (n = {len(d)})", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, "fig18_stroke.png")


def synthetic_figure(art: dict, out: Path) -> Path:
    sp = art["synthetic_predictions"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    m = "linear_regression"
    lo = float(min(sp["synthetic_label_next_rate"].min(), sp[f"pred_{m}"].min())) - 1
    hi = float(max(sp["synthetic_label_next_rate"].max(), sp[f"pred_{m}"].max())) + 1
    ax.scatter(sp[f"pred_{m}"], sp["synthetic_label_next_rate"], marker="^", s=18, color=SERIES_LIGHT[4], alpha=0.75)
    ax.plot([lo, hi], [lo, hi], ls=":", color=LIGHT["muted"])
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Linear Regression estimate for synthetic inputs (%)"); ax.set_ylabel("Synthetic label (%)")
    ax.set_title("SYNTHETIC scenario test (not real data)", color="#c2416f")
    ax.text(0.02, 0.97, "Consistency check only -\nnot evidence of accuracy", transform=ax.transAxes, va="top",
            fontsize=8, color="#c2416f")
    return _save(fig, out, "fig19_synthetic.png")


def build_all(art: dict, out: Path) -> dict[str, Path]:
    figs = {
        "architecture": architecture_diagram(out),
        "coverage": coverage_timeline(art, out),
        "preprocessing": preprocessing_diagram(out),
        "model_workflow": model_workflow_diagram(art, out),
        "allcause_trend": allcause_trend(art, out),
        "county_map": county_map(art, out),
        "county_ranking": county_ranking(art, out),
        "strata": strata_panels(art, out),
        "cabg_distribution": cabg_distribution(art, out),
        "lag_scatter": lag_scatter(art, out),
        "cabg_designs": design_comparison(art, "cabg", out),
        "stroke_designs": design_comparison(art, "stroke", out),
        "bootstrap": bootstrap_forest(art, out),
        "cabg_avp": actual_vs_predicted(art, "cabg", out),
        "stroke_avp": actual_vs_predicted(art, "stroke", out),
        "residuals": residuals(art, out),
        "permutation": permutation(art, out),
        "learned_curve": learned_curve(art, out),
        "hospital_examples": hospital_examples(art, out),
        "stroke": stroke_panels(art, out),
        "synthetic": synthetic_figure(art, out),
    }
    return figs
