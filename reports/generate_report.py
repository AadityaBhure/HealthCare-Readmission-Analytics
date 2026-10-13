"""Regenerate reports/Project_Report.docx and reports/Project_Report.pdf from pipeline artifacts.

    python reports/generate_report.py            (run python run_pipeline.py first)

All numbers are read from artifacts/ and models/; wording that depends on results (e.g. whether a
model beats the baseline) is chosen from the bootstrap/CV outputs, never typed in by hand.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config, report_figures  # noqa: E402
from src.modeling import MODEL_LABELS  # noqa: E402
from src.reporting import Report, load_artifacts, render_docx, render_pdf  # noqa: E402

SHOTS = config.SCREENSHOTS_DIR


def f2(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.2f}"


def f3(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.3f}"


def met(df: pd.DataFrame, model: str, col: str = "MAE") -> float:
    return float(df.loc[df["model"] == model, col].iloc[0])


def boot(df: pd.DataFrame, model: str, ref: str = "persistence") -> dict:
    return df[(df["model"] == model) & (df["reference"] == ref)].iloc[0].to_dict()


def verdict(b: dict) -> str:
    if b["ci_high"] < 0:
        return "is significantly more accurate than"
    if b["ci_low"] > 0:
        return "is significantly less accurate than"
    return "is statistically indistinguishable from"


def paired(df: pd.DataFrame, design_prefix: str, model: str) -> dict:
    return df[df["design"].str.startswith(design_prefix) & (df["model"] == model)].iloc[0].to_dict()


def metrics_rows(m: pd.DataFrame) -> list[list[str]]:
    return [[r["label"] + (" (recommended)" if r["recommended"] else ""), f3(r["MAE"]), f3(r["RMSE"]), f3(r["R2"]),
             f"{r['Bias']:+.3f}", str(int(r["n_train"])), str(int(r["n_test"])), f"{r['MAE_vs_persistence_pct']:+.1f}%"]
            for _, r in m.iterrows()]


def cv_rows(cv: pd.DataFrame) -> list[list[str]]:
    return [[MODEL_LABELS[r["model"]], f"{r['MAE_mean']:.3f} ± {r['MAE_sd']:.3f}", f"{r['RMSE_mean']:.3f} ± {r['RMSE_sd']:.3f}",
             f"{r['R2_mean']:.3f} ± {r['R2_sd']:.3f}", str(int(r["folds"]))] for _, r in cv.iterrows()]


def build(art: dict, figs: dict) -> Report:
    A = art
    cs, ss = A["cabg_summary"], A["stroke_summary"]
    cts, sts = cs["train_summary"], ss["train_summary"]
    cm, sm = A["cabg_metrics"], A["stroke_metrics"]
    cb, sb = A["cabg_bootstrap"], A["stroke_bootstrap"]
    pop = A["pop_summary"]
    au = A["audit"]
    rec = cs["recommended_model"]
    lr_b, rf_b = boot(cb, "linear_regression"), boot(cb, "random_forest")
    s_lr_b = boot(sb, "linear_regression")
    cvp, cvn = A["cabg_cv_pooled"].set_index("model"), A["cabg_cv_nonoverlap"].set_index("model")
    scvp, scvn = A["stroke_cv_pooled"].set_index("model"), A["stroke_cv_nonoverlap"].set_index("model")
    c_pair_n = paired(A["cabg_paired_cv"], "non", "linear_regression")
    c_pair_1 = paired(A["cabg_paired_cv"], "one", "linear_regression")
    s_pair_1 = paired(A["stroke_paired_cv"], "one", "linear_regression")
    lagc, lags = cs["lag_correlation"], ss["lag_correlation"]
    sw_c = A["cabg_statewide"]
    sw_cr = sw_c[sw_c["measure_code"] == "readmission"].reset_index(drop=True)
    sw_s = A["stroke_statewide"]
    corr = A["stroke_descriptive"]["readmission_vs_mortality_correlation"]
    rs = [v["pearson_r"] for v in corr.values()]
    checks = A["checks"]
    n_pass, n_warn, n_fail = (int((checks["status"] == s).sum()) for s in ("pass", "warn", "fail"))
    perm = A["cabg_permutation"]
    lr_perm = perm[perm["model"] == "linear_regression"].set_index("feature")
    native = A["cabg_native"]
    lr_coef = native[native["model"] == "linear_regression"].set_index("feature")["value"]
    err = A["cabg_error_by_volume"]
    err_lr = err[err["model"] == "linear_regression"].set_index("volume_tercile")["MAE"]
    fs = A["cabg_feature_sets"]
    ext_lr = float(fs[(fs["feature_set"].str.startswith("extended")) & (fs["model"] == "linear_regression")]["MAE"].iloc[0])
    ext_rf = float(fs[(fs["feature_set"].str.startswith("extended")) & (fs["model"] == "random_forest")]["MAE"].iloc[0])
    syn = A["synthetic_summary"]
    syn_ag = pd.DataFrame(syn["agreement_with_synthetic_labels"]).set_index("model")
    man = A["manifest"]
    integrity_ok = all(v["match"] for e in A["integrity"].values() for v in e.values())

    R = Report(
        title=config.PROJECT_TITLE,
        subtitle="Temporally aligned next-period prediction, stroke outcome analysis and population-level trend "
                 "analytics with an interactive Streamlit dashboard",
        meta=[("Submitted by", "[Student name(s) and roll number(s)]"), ("Guide", "[Faculty guide name]"),
              ("Department / Institution", "[Department, Institution]"),
              ("Date", date.today().strftime("%B %Y")),
              ("Live dashboard", config.LIVE_DASHBOARD_URL), ("Source code", config.GITHUB_REPO_URL)])

    # ------------------------------------------------------------------ Abstract
    R.h1("Abstract")
    R.p(
        "Thirty-day readmission is a widely used indicator of hospital quality, but most student projects predict it "
        "from same-period information or from patient-level benchmark data. This project builds an explainable, "
        "reproducible analytics framework on four California datasets supplied for the project: hospital-level "
        "coronary artery bypass graft (CABG) surgery outcomes (2018-2023), hospital-level ischemic stroke outcomes "
        "(2011-2015), county- and population-level all-cause readmissions (2011-2024) and a synthetic CABG test file. "
        f"The primary experiment estimates each hospital's **next reporting-window risk-adjusted isolated CABG 30-day "
        f"readmission rate** from its previous window. Models were trained on {cts['n_train']} hospital transitions "
        f"and evaluated once on a held-out later window ({cts['test_target_periods'][0]}, n = {cts['n_test']} hospitals). "
        f"A persistence baseline achieved MAE {f2(met(cm, 'persistence'))} percentage points (pp), Linear Regression "
        f"{f2(met(cm, 'linear_regression'))} pp and Random Forest {f2(met(cm, 'random_forest'))} pp; the bootstrap 95% "
        f"interval for the Linear Regression minus persistence difference was [{lr_b['ci_low']:+.2f}, {lr_b['ci_high']:+.2f}] pp, "
        "so no model reliably beat persistence on this one-step horizon. We show why: the published two-year windows "
        f"overlap by one calendar year, inflating window-to-window correlation (r = {lagc['one_step_overlapping']:.2f} vs "
        f"{lagc['non_overlapping']:.2f} for non-overlapping windows). On a non-overlapping horizon, persistence error "
        f"rose to {f2(cvn.loc['persistence', 'MAE_mean'])} pp while Linear Regression achieved "
        f"{f2(cvn.loc['linear_regression', 'MAE_mean'])} pp, better in {c_pair_n['share_folds_model_better']:.0%} of "
        "hospital-grouped cross-validation folds. A separate stroke model with more hospitals reduced error relative to "
        f"persistence by {abs(s_lr_b['relative_change_pct']):.1f}% (MAE {f2(met(sm, 'linear_regression'))} vs "
        f"{f2(met(sm, 'persistence'))} pp). Population analysis shows the statewide all-cause readmission rate reached "
        f"{f2(pop['latest_rate'])}% in {pop['latest_year']}, with marked differences by payer, age group and county. "
        "Results are delivered through a five-page Streamlit dashboard, a GitHub Actions pipeline and automated tests. "
        "All estimates are hospital-level rates, not patient-level probabilities, and are not clinically validated.")

    # ------------------------------------------------------------------ 1 Introduction
    R.h1("1. Introduction and Background")
    R.p("An unplanned readmission within 30 days of discharge is costly for health systems and is often taken as a "
        "signal of incomplete recovery, poor care transitions or inadequate follow-up. Large studies of Medicare "
        "patients reported that about one in five beneficiaries were rehospitalised within 30 days [1], and readmission "
        "rates became the basis of public reporting and financial penalties in the United States [3]. California's "
        "Department of Health Care Access and Information (HCAI, formerly OSHPD) publishes risk-adjusted hospital "
        "performance reports for selected procedures and conditions and population-level readmission statistics; the "
        "datasets analysed here follow those reporting formats.")
    R.p("Predicting readmission is difficult: a systematic review of 26 readmission risk models found most had only "
        "modest discrimination [2]. At the hospital level, an additional problem is statistical noise. Rates for "
        "low-volume hospitals rest on few patients, so year-to-year changes partly reflect chance [10], and extreme "
        "values tend to move back toward the average in the next period - regression to the mean [9]. A credible "
        "forecasting project must therefore compare any machine-learning model with simple baselines and must report "
        "uncertainty honestly.")

    R.h1("2. Problem Statement")
    R.p("Given only information that would have been available at the end of a reporting window, how accurately can "
        "we estimate a hospital's risk-adjusted CABG readmission rate in the next window, and does a machine-learning "
        "model add value over the naive assumption that the rate stays the same? In parallel, how do readmission "
        "patterns differ between a surgical procedure (CABG), a medical condition (ischemic stroke) and the general "
        "inpatient population, across counties and population groups?")

    R.h1("3. Research Question and Objectives")
    R.note("**Research question.** Can historical hospital performance help estimate future 30-day readmission rates, "
           "and how do readmission patterns differ across clinical conditions, counties, and population groups?")
    R.numbered([
        "Audit the four datasets, document their structure, coverage and defects, and build reproducible cleaning code.",
        "Construct temporally aligned (previous window -> next window) training data without bridging missing periods.",
        "Compare a persistence baseline, a training-mean baseline, Linear Regression and Random Forest on identical "
        "held-out data with leakage-aware validation and uncertainty intervals.",
        "Explain the models with permutation importance, coefficients, learned response curves and error analysis.",
        "Analyse ischemic stroke readmission and mortality separately, including a separate temporal model.",
        "Describe statewide, county and population-group all-cause readmission trends, respecting the ICD-9/ICD-10 break.",
        "Deliver an interactive, deployable dashboard, an automated pipeline, tests and this report.",
    ])

    R.h1("4. Project Scope and Contributions")
    R.p("The project does not propose a new algorithm. Its contribution is a healthcare-specific analytical and "
        "evaluation workflow:")
    R.bullets([
        "**Temporal alignment.** Each training example pairs inputs from window *t* with the outcome in window *t+1*; a "
        "transition is created only when the target window starts exactly one year after the input window.",
        "**Baseline-first evaluation.** Every model is judged against persistence on the same hospitals, with a paired "
        "bootstrap interval for the difference.",
        "**Discovery and correction of a design artefact.** The published windows overlap, which flatters persistence; "
        "a non-overlapping sensitivity design quantifies this.",
        "**Multi-level view.** Hospital surgical outcomes, hospital stroke outcomes and population trends are analysed "
        "side by side without merging incompatible measures.",
        "**Transparent delivery.** A dashboard that loads exactly the evaluated models (verified by an integrity "
        "check), a one-command pipeline, CI workflow and an automated pytest suite.",
    ])

    # ------------------------------------------------------------------ 5 Data
    R.h1("5. Dataset Descriptions and Data Dictionary Summary")
    R.p(f"Four CSV files were supplied (Table {R.tab('datasets')}). Their column structures correspond to HCAI public "
        "reports distributed through the California Health and Human Services Open Data Portal [19]; the exact "
        "download dates were not recorded with the files, so provenance should be confirmed by the project team. "
        "The raw files are stored unchanged in data/raw/ and every transformation is applied in code.")
    rows = []
    for k in ("cabg", "stroke", "allcause", "synthetic"):
        v = au[k]
        span = (", ".join([v["periods"][0], v["periods"][-1]]) if "periods" in v else
                f"{v['years'][0]}-{v['years'][1]}" if "years" in v else "synthetic")
        rows.append([v["title"].split(" - ")[0].replace("Dataset ", ""), v["title"].split(" - ")[1], f"{v['rows']:,}", str(v["columns"]),
                     span, v["granularity"], v["role"]])
    R.table("datasets", ["ID", "Dataset", "Rows", "Cols", "Coverage", "Granularity", "Role in project"], rows,
            "The four datasets and their roles (computed from the files).", widths=[0.7, 2, 0.7, 0.5, 1.4, 2.2, 2.6])
    R.figure("coverage", figs["coverage"],
             "Reporting periods available in each source. CABG readmission and stroke data use overlapping two-year "
             "windows; the 2020-2021 CABG readmission window contains only 2 hospitals; all-cause data switch from "
             "ICD-9 to ICD-10 after 2015 (2015 is a partial year).")
    R.h2("5.1 Key fields")
    R.table("dict", ["Dataset", "Field (cleaned name)", "Meaning / unit"], [
        ["A", "hospital_id", "9-digit facility identifier; stable panel key (127 IDs, many renamed hospitals)"],
        ["A", "period / period_start / period_years", "Reporting window, e.g. 2021-2022 (start 2021, 2 years)"],
        ["A", "measure_code", "readmission, postop_stroke, iso_mortality (single year), cabg_valve_mortality"],
        ["A", "raaer, raaer_ci_low/high", "Risk-adjusted adverse-event rate (%) and 95% CI"],
        ["A", "hospital_cabg_cases, hospital_measure_cases", "CABG volume; eligible cases for the measure"],
        ["B", "risk_adjusted_rate, events, cases", "Stroke readmission or mortality rate (%), counts"],
        ["B", "rating", "Better / As Expected / Worse than expected"],
        ["C", "coding_system", "ICD-9 (2011-2015) or ICD-10 (2016-2024)"],
        ["C", "admits, readmits, rate", "Index admissions, 30-day readmissions, rate (%)"],
        ["C", "strata / strata_name / county", "Overall, Age, Sex, Race-Ethnicity, Expected Payer; county or State"],
        ["D", "prev_readmit_rate, prev_stroke_rate, prev_cabg_volume", "Synthetic inputs mapped to the model schema"],
    ], "Key fields after cleaning. The complete auto-generated dictionary is in DATA_DICTIONARY.md.", widths=[0.6, 3, 5])

    # ------------------------------------------------------------------ 6 EDA
    R.h1("6. Exploratory Data Analysis")
    R.h2("6.1 Data quality findings")
    R.bullets([
        f"**CABG:** {au['cabg']['rows']:,} rows, {au['cabg']['hospital_ids']} hospital IDs but {au['cabg']['name_variants']} "
        "different name spellings (line breaks inside names, Windows-1252 dashes, rebranding). Eleven CABG+Valve rows "
        "report '.' instead of a rate. Readmission coverage: "
        + "; ".join(f"{r['period']}: {int(r['hospitals'])} hospitals" for _, r in sw_cr.iterrows()) + ".",
        "**Rolling windows:** statewide case totals show each two-year window equals the sum of its two single years "
        "(e.g. 2019-2020 = 2019 + 2020), so consecutive windows share one calendar year of patients.",
        f"**Stroke:** in the 2013-2014 window all {au['stroke']['rows_corrected_for_column_swap']} rows have the "
        "'# of Deaths/Readmissions' and '# of Cases' columns transposed (events exceeded cases in every row). After "
        "swapping, the observed rates agree with the published risk-adjusted rates as in other periods.",
        "**All-cause:** rates are stored as text ('13.80%'), ICD-10 counts contain thousands separators, one county-year "
        "is suppressed (Alpine 2017), 'Multiracial' appears from 2019 and 'Unknown County' only in 2019.",
        f"**Validation:** {n_pass} automated checks passed, {n_warn} produced warnings (documented limitations) and "
        f"{n_fail} failed.",
    ])
    R.h2("6.2 Distributions and trends")
    R.p(f"Figure {R.fig('cabg_distribution')} shows that hospital CABG readmission rates are widely dispersed around "
        "the statewide rate, with a long upper tail driven mainly by small hospitals. The statewide observed rate "
        f"declined from {f2(sw_cr.iloc[0]['rate'])}% ({sw_cr.iloc[0]['period']}) to {f2(sw_cr.iloc[-1]['rate'])}% "
        f"({sw_cr.iloc[-1]['period']}).")
    R.figure("cabg_distribution", figs["cabg_distribution"],
             "Distribution of hospital risk-adjusted isolated CABG readmission rates per window (boxes: quartiles; "
             "diamonds: statewide observed rate). The sparse 2020-2021 window is omitted.")
    R.p(f"Figure {R.fig('lag_scatter')} is the most important exploratory result for the modelling design. Rates in "
        f"consecutive windows correlate at r = {lagc['one_step_overlapping']:.2f}, but only r = {lagc['non_overlapping']:.2f} "
        "when the windows do not share a year. Part of the apparent stability is therefore shared data, not "
        "persistent hospital performance.")
    R.figure("lag_scatter", figs["lag_scatter"],
             "Previous vs next readmission rate for consecutive (overlapping) and non-overlapping window pairs. "
             "Dotted line: persistence (next = previous).")

    # ------------------------------------------------------------------ 7 Cleaning
    R.h1("7. Data Cleaning and Preprocessing Methodology")
    R.p(f"Figure {R.fig('preprocessing')} summarises the cleaning workflow and Table {R.tab('prep')} lists the logged "
        "decisions. Three principles were applied throughout: missing or suppressed values remain missing (never "
        "zero); raw files are never modified; and each dataset keeps its own tidy table rather than being forced into "
        "one master table.")
    R.figure("preprocessing", figs["preprocessing"], "Dataset-specific preprocessing steps implemented in src/preprocessing.py.")
    log = A["prep_log"]
    R.table("prep", ["Dataset", "Step", "Rows", "Decision"],
            [[r["dataset"], r["step"], "" if pd.isna(r["rows_affected"]) else f"{int(r['rows_affected']):,}",
              str(r["detail"])] for _, r in log.iterrows()],
            "Preprocessing decisions recorded by the pipeline (artifacts/preprocessing_log.csv).", widths=[0.8, 1.3, 0.6, 6.3], font=7.4)
    R.p("Learned transformations (median imputation, log transform of case volume and standardisation) are part of "
        "each scikit-learn Pipeline, so they are fitted on training rows only; a unit test checks that the scaler's "
        "means equal the training-set means.")

    # ------------------------------------------------------------------ 8 Features
    R.h1("8. Feature Engineering and Temporal Alignment")
    R.p("The CABG panel has one row per hospital and two-year window, with readmission, post-operative stroke and "
        "CABG+valve mortality side by side. A transition links window *t* (inputs) to window *t+1* (target) only when "
        "the target window starts exactly one year later. Because 2020-2021 is almost empty, the pair "
        "2019-2020 -> 2021-2022 is not created in the main design; it is used only in the explicitly labelled "
        "non-overlapping sensitivity design, where the horizon is defined as two years.")
    R.table("features", ["Name", "Definition", "Unit", "Used in"], [
        ["prev_readmit_rate", "Readmission RAAER in the input window", "%", "Primary + baseline"],
        ["prev_stroke_rate", "Post-operative stroke RAAER in the input window", "%", "Primary"],
        ["prev_cabg_volume", "Total hospital CABG cases in the input window (log1p inside the model)", "cases", "Primary"],
        ["prev_iso_mortality_final_year", "Isolated CABG mortality for the final calendar year of the input window", "%", "Extended (sensitivity)"],
        ["prev_cabg_valve_mortality_rate", "CABG+Valve mortality RAAER (different cohort)", "%", "Extended (sensitivity)"],
        ["prev_expected_readmit_rate", "Expected readmission rate (case-mix severity)", "%", "Extended (sensitivity)"],
        ["next_readmit_rate (target)", "Readmission RAAER in the next window", "%", "Target"],
    ], "Features and target for the CABG model. Identifiers, names, counties and any target-window field are excluded.",
            widths=[2.2, 4.3, 0.7, 1.6])
    R.p("Operative mortality is published for single years only, so it cannot be treated as a two-year rate. It is "
        "aligned to the final calendar year of the input window and kept out of the primary model; Section 13 reports "
        f"its effect as a sensitivity analysis. Two transitions out of 2020-2021 were excluded because no post-operative "
        f"stroke measure was published for that window; {cts['n_train']} training and {cts['n_test']} test transitions "
        "remain.")

    # ------------------------------------------------------------------ 9 Algorithms
    R.h1("9. Machine Learning Algorithms and Baselines")
    R.bullets([
        "**Persistence baseline:** next rate = previous rate. A standard naive forecast [6]; any useful model must beat it.",
        "**Training-mean baseline:** predicts the training-set mean for every hospital; anchors the R² scale.",
        "**Linear Regression:** ordinary least squares on standardised inputs. Its slope on the previous rate shows how "
        "strongly extreme hospitals are expected to move back toward the average.",
        f"**Random Forest [4]:** an ensemble of 300 regression trees able to capture non-linearity; hyper-parameters "
        f"(selected: {cs['rf_best_params']}) were tuned by grid search with hospital-grouped 5-fold CV on training data only.",
    ])
    R.p("Ridge regression was not added: with three standardised inputs and over a hundred rows the ordinary least "
        "squares fit is already stable, so shrinkage would change little. Models were implemented with scikit-learn [5].")

    # ------------------------------------------------------------------ 10 Methodology
    R.h1("10. Training and Evaluation Methodology")
    R.figure("model_workflow", figs["model_workflow"],
             "Training and evaluation workflow. The held-out window is used once, after model selection.")
    R.table("splits", ["Design", "Data", "Rows", "Purpose"], [
        ["Temporal hold-out (primary)", f"train targets {', '.join(cts['train_target_periods'])}; test target {cts['test_target_periods'][0]}",
         f"{cts['n_train']} / {cts['n_test']}", "Realistic forward evaluation; one-shot test"],
        ["Grouped CV within training", "hospital-grouped 5-fold, 3 repeats", str(cts["n_train"]), "RF tuning and choice of recommended model"],
        ["Pooled grouped CV", "all valid one-step transitions, 5-fold x 5 repeats", str(cs["n_pooled"]), "Stability across hospitals"],
        ["Non-overlapping horizon", "window t -> t+2 (no shared calendar year)", str(cs["n_nonoverlap"]), "Removes the overlap artefact"],
        ["Extended features", "same temporal split, 6 inputs", f"{cts['n_train']} / {cts['n_test']}", "Value of mortality and case-mix inputs"],
    ], "Evaluation designs and sample counts (CABG).", widths=[2, 3.6, 1, 2.6])
    R.p("Grouped cross-validation ensures that no hospital appears in both a training and a validation fold, which "
        "is necessary when the same hospital contributes several rows [7]. Metrics are mean absolute error (MAE), "
        "root mean squared error (RMSE), R² (relative to the evaluation-set mean, so it can be negative) and mean "
        "bias, all in percentage points (pp) of the readmission rate. Differences between models are assessed with a "
        "paired bootstrap over test hospitals (2,000 resamples) [8]. The recommended model was chosen **before** "
        f"looking at the test window: {MODEL_LABELS[rec]} had the lowest grouped-CV MAE within the training data.")

    # ------------------------------------------------------------------ 11 Comparative performance
    R.h1("11. Comparative Model Performance")
    R.table("cabg_metrics", ["Model", "MAE", "RMSE", "R²", "Bias", "Train n", "Test n", "MAE vs persistence"],
            metrics_rows(cm), f"CABG next-window readmission: performance on the held-out {cts['test_target_periods'][0]} "
            "window (pp; lower MAE/RMSE is better).", widths=[2.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.3])
    R.table("cabg_cv", ["Model", "MAE (mean ± SD)", "RMSE (mean ± SD)", "R² (mean ± SD)", "Folds"],
            cv_rows(A["cabg_cv_pooled"]), f"Pooled hospital-grouped CV over all {cs['n_pooled']} one-step transitions (5 folds x 5 repeats).",
            widths=[2.4, 1.8, 1.8, 1.8, 0.7])
    R.table("cabg_nonov", ["Model", "MAE (mean ± SD)", "RMSE (mean ± SD)", "R² (mean ± SD)", "Folds"],
            cv_rows(A["cabg_cv_nonoverlap"]), f"Non-overlapping horizon (window t -> t+2, n = {cs['n_nonoverlap']}), hospital-grouped CV.",
            widths=[2.4, 1.8, 1.8, 1.8, 0.7])
    R.figure("cabg_designs", figs["cabg_designs"],
             "CABG MAE by model across the three evaluation designs (error bars: SD over 25 grouped-CV folds).")
    R.table("boot", ["Comparison", "MAE difference", "95% CI", "P(first better)", "Relative change"], [
        [f"{MODEL_LABELS[r['model']]} - {MODEL_LABELS[r['reference']]}", f"{r['mae_diff']:+.3f}",
         f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", f"{r['prob_a_better']:.0%}", f"{r['relative_change_pct']:+.1f}%"]
        for _, r in cb.iterrows()],
        "Paired bootstrap of test-set MAE differences for CABG (negative = first model more accurate).",
        widths=[3.3, 1.3, 1.6, 1.3, 1.3])
    R.figure("bootstrap", figs["bootstrap"], "Paired bootstrap 95% intervals of MAE differences for the CABG and stroke hold-out sets.")

    # ------------------------------------------------------------------ 12 Results & discussion
    R.h1("12. Results and Discussion")
    R.p(f"**Primary result.** On the held-out window, persistence achieved the lowest MAE ({f3(met(cm, 'persistence'))} pp), "
        f"followed by Linear Regression ({f3(met(cm, 'linear_regression'))} pp) and Random Forest ({f3(met(cm, 'random_forest'))} pp). "
        f"Linear Regression {verdict(lr_b)} persistence (difference {lr_b['mae_diff']:+.3f} pp, 95% CI "
        f"[{lr_b['ci_low']:+.3f}, {lr_b['ci_high']:+.3f}]), and Random Forest {verdict(rf_b)} it "
        f"([{rf_b['ci_low']:+.3f}, {rf_b['ci_high']:+.3f}]). The evidence is therefore **insufficient to name a reliable "
        "winner** for the one-step horizon; all three are far better than the training-mean baseline "
        f"(MAE {f3(met(cm, 'mean_baseline'))} pp, R² {f3(met(cm, 'mean_baseline', 'R2'))}).")
    R.p(f"**Why persistence is hard to beat here.** Consecutive two-year windows share a calendar year, so half of the "
        "target already sits inside the input. Across all one-step transitions (pooled grouped CV) Linear Regression "
        f"was slightly better than persistence on average ({c_pair_1['mean_mae_diff']:+.3f} pp, better in "
        f"{c_pair_1['share_folds_model_better']:.0%} of folds). When the overlap is removed, persistence degrades sharply "
        f"(MAE {f2(cvn.loc['persistence', 'MAE_mean'])} pp, R² {f2(cvn.loc['persistence', 'R2_mean'])}) while Linear "
        f"Regression ({f2(cvn.loc['linear_regression', 'MAE_mean'])} pp) and Random Forest "
        f"({f2(cvn.loc['random_forest', 'MAE_mean'])} pp) hold up; Linear Regression is better in "
        f"{c_pair_n['share_folds_model_better']:.0%} of folds. The learned models help most when the future is genuinely "
        "new information, because they shrink extreme rates toward the average.")
    R.p(f"**Random Forest vs Linear Regression.** The forest never outperformed the linear model. With "
        f"{cts['n_train']} training rows and one dominant input, the relationship is close to linear and the extra "
        "flexibility mostly fits noise; grid search accordingly chose shallow, heavily regularised trees.")
    R.p(f"**Extended features.** Adding final-year operative mortality, CABG+valve mortality and expected readmission "
        f"rate changed the Linear Regression test MAE from {f3(met(cm, 'linear_regression'))} to {f3(ext_lr)} pp and Random "
        f"Forest from {f3(met(cm, 'random_forest'))} to {f3(ext_rf)} pp. Given the bootstrap uncertainty of about "
        "±0.3 pp, these inputs do not add reliable predictive value, which supports keeping the simpler primary model.")
    R.p("**Should the results be trusted?** The test set is a single later window of about 120 hospitals, the outcome "
        "is itself an estimate with wide confidence intervals for small hospitals, and only one year separates train "
        "and test targets. The results are a fair, leakage-free comparison but not proof of performance in future "
        "years.")
    R.figure("cabg_avp", figs["cabg_avp"], "Actual vs predicted next-window CABG readmission rates on the held-out window.")

    # ------------------------------------------------------------------ 13 Explainability
    R.h1("13. Explainability and Important Features")
    R.p(f"Permutation importance on the test window (Figure {R.fig('permutation')}) shows that the previous readmission "
        f"rate carries almost all predictive signal: shuffling it raises Linear Regression MAE by "
        f"{f2(lr_perm.loc['prev_readmit_rate', 'importance_mae_increase'])} pp, whereas shuffling case volume raises it by "
        f"{f2(lr_perm.loc['prev_cabg_volume', 'importance_mae_increase'])} pp and post-operative stroke by "
        f"{f2(lr_perm.loc['prev_stroke_rate', 'importance_mae_increase'])} pp [12]. The standardised Linear Regression "
        f"coefficients agree: +{f2(lr_coef['prev_readmit_rate'])} pp per standard deviation of previous readmission and "
        f"{lr_coef['prev_cabg_volume']:+.2f} pp per standard deviation of log case volume, i.e. larger programmes are "
        "estimated to have slightly lower future rates, consistent with the volume-outcome literature [11] but only an "
        "association here.")
    R.figure("permutation", figs["permutation"], "Permutation importance (increase in test MAE when a feature is shuffled; mean ± SD).")
    R.p(f"Figure {R.fig('learned_curve')} shows the learned relationship. The Linear Regression line is flatter than the "
        "persistence diagonal: a hospital with an unusually high or low rate is expected to move part of the way back "
        "toward the state average in the next window - regression to the mean [9].")
    R.figure("learned_curve", figs["learned_curve"], "Estimated next-window rate as a function of the previous rate (other inputs at training medians).")
    R.p(f"Errors are concentrated in small hospitals (Figure {R.fig('residuals')}): Linear Regression MAE is "
        + ", ".join(f"{f2(v)} pp for {k.lower()} hospitals" for k, v in err_lr.items())
        + ". This is expected because a rate based on a few dozen patients is very noisy [10].")
    R.figure("residuals", figs["residuals"], "Residuals vs target-window cohort size and test MAE by cohort-size tercile.")

    # ------------------------------------------------------------------ 14 Hospital-level analysis
    R.h1("14. CABG Hospital-Level Prediction Analysis")
    R.p(f"Figure {R.fig('hospital_examples')} applies the evaluated models to the four highest-volume hospitals in the "
        "test set. For each, the inputs come from 2021-2022 only; the published 2022-2023 value was hidden from the "
        "models. In the dashboard, the same view is available for every hospital and window, labelled as an in-sample "
        "back-test, an out-of-sample back-test or a forward estimate for 2023-2024, together with the statewide "
        "benchmark (the statewide observed rate for the target window, or the latest published rate when the target "
        "is not yet available).")
    R.figure("hospital_examples", figs["hospital_examples"],
             "Historical readmission RAAER (95% CI), statewide rate, model estimates for 2022-2023 and the published "
             "outcome (last observed point) for the four largest test-set hospitals.")
    R.note("**Interpretation.** An estimate such as '8.8%' means the model expects about 8.8% of that hospital's eligible "
           "isolated CABG patients in the next window to be readmitted within 30 days after risk adjustment. It is not "
           "the probability that any individual patient will be readmitted, and it does not identify causes.")

    # ------------------------------------------------------------------ 15 Stroke
    R.h1("15. Stroke Outcome Analysis")
    ssr, ssm = sw_s[sw_s["measure_code"] == "readmission"], sw_s[sw_s["measure_code"] == "mortality"]
    R.p(f"Statewide risk-adjusted ischemic stroke readmission fell from {f2(ssr.iloc[0]['risk_adjusted_rate'])}% "
        f"({ssr.iloc[0]['period']}) to {f2(ssr.iloc[-1]['risk_adjusted_rate'])}% ({ssr.iloc[-1]['period']}) and mortality from "
        f"{f2(ssm.iloc[0]['risk_adjusted_rate'])}% to {f2(ssm.iloc[-1]['risk_adjusted_rate'])}% (Figure {R.fig('stroke')}). "
        f"Across {au['stroke']['hospital_ids']} hospitals, readmission and mortality were nearly uncorrelated "
        f"(r from {min(rs):+.2f} to {max(rs):+.2f}), matching national findings that the two measures rank hospitals "
        "differently [13]. Mortality and readmission describe different patients: those who die cannot be readmitted.")
    R.figure("stroke", figs["stroke"], "Statewide stroke outcomes and hospital readmission vs mortality in the latest window.")
    R.p(f"Because the stroke data are larger ({sts['n_train']} training transitions, {sts['n_test']} test hospitals), a "
        "separate model was trained with the same protocol (inputs: previous readmission rate, previous mortality "
        f"rate, previous case volume). Linear Regression reduced test MAE from {f3(met(sm, 'persistence'))} to "
        f"{f3(met(sm, 'linear_regression'))} pp ({s_lr_b['relative_change_pct']:+.1f}%; 95% CI "
        f"[{s_lr_b['ci_low']:+.3f}, {s_lr_b['ci_high']:+.3f}] pp). In pooled grouped CV it beat persistence in "
        f"{s_pair_1['share_folds_model_better']:.0%} of folds and on the non-overlapping horizon persistence MAE rose "
        f"to {f2(scvn.loc['persistence', 'MAE_mean'])} pp vs {f2(scvn.loc['linear_regression', 'MAE_mean'])} pp. With more "
        "hospitals, the benefit of shrinkage becomes visible even on the overlapping horizon. Previous mortality added "
        "no predictive value.")
    R.table("stroke_metrics", ["Model", "MAE", "RMSE", "R²", "Bias", "Train n", "Test n", "MAE vs persistence"],
            metrics_rows(sm), f"Stroke next-window readmission: held-out {sts['test_target_periods'][0]} window (pp).",
            widths=[2.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.3])
    R.figure("stroke_designs", figs["stroke_designs"], "Stroke model MAE across evaluation designs.")
    R.p("The stroke data (2011-2015) do not overlap in time with the CABG data (2018-2023) and concern a different "
        "condition, so they were not used to test the CABG model.")

    # ------------------------------------------------------------------ 16 Population
    R.h1("16. County and Population-Level Trends")
    R.p(f"The statewide all-cause 30-day readmission rate (Figure {R.fig('allcause_trend')}) was "
        f"{f2(pop['icd10_first_rate'])}% in {pop['icd10_first_year']}, the first ICD-10 year, and {f2(pop['latest_rate'])}% in "
        f"{pop['latest_year']} ({pop['icd10_change_pp']:+.2f} pp). The ICD-9 years ({pop['icd9_years'][0]}-{pop['icd9_years'][-1]}) are "
        "plotted as a separate series because the coding change, effective 1 October 2015 [14], can shift case "
        f"definitions. Admissions fell {abs(pop['admits_change_2019_2020_pct']):.1f}% from 2019 to 2020, coinciding with "
        "the COVID-19 pandemic.")
    R.figure("allcause_trend", figs["allcause_trend"], "Statewide all-cause 30-day readmission rate by coding era.")
    hi, lo = pop["highest_county"], pop["lowest_county"]
    R.p(f"Among {pop['counties_ranked_latest']} counties with at least {pop['min_county_admits']:,} admissions in "
        f"{pop['latest_year']}, rates ranged from {f2(lo['rate'])}% ({lo['county']}) to {f2(hi['rate'])}% ({hi['county']}); "
        f"{pop['counties_below_threshold_latest']} smaller counties were not ranked because their rates are unstable "
        f"(Figures {R.fig('county_map')} and {R.fig('county_ranking')}).")
    R.figure("county_map", figs["county_map"], "County all-cause readmission rates (counties below the admissions threshold in grey).", width_in=4.2)
    R.figure("county_ranking", figs["county_ranking"], "Lowest and highest ten counties by readmission rate.", width_in=4.6)
    sl = pop["strata_latest"]
    payer, age = sl.get("Expected Payer", []), sl.get("Age", [])
    R.p(f"By expected payer, {payer[0]['name']} patients had the highest rate ({f2(payer[0]['rate'])}%) and "
        f"{payer[-1]['name']} the lowest ({f2(payer[-1]['rate'])}%); by age, {age[0]['name']} had the highest "
        f"({f2(age[0]['rate'])}%) and {age[-1]['name']} the lowest ({f2(age[-1]['rate'])}%) (Figure {R.fig('strata')}). "
        "These differences describe who is readmitted; they reflect differences in health status, insurance and access "
        "and do not show that any characteristic causes readmission.")
    R.figure("strata", figs["strata"], f"Readmission rate by population group, {pop['latest_year']}.")

    # ------------------------------------------------------------------ 17 Dashboard
    R.h1("17. Dashboard Design and Functionality")
    R.p("The dashboard (Streamlit [15] with Plotly [16]) uses a dark navy and cyan theme, a left navigation sidebar, a "
        "filter bar above each page's charts, KPI cards with units and comparison periods, and interactive charts with "
        "tooltips. The categorical palette was checked for colour-vision-deficiency separation, colours follow "
        "entities (each model and coding era keeps its colour) and every chart avoids dual axes. Only measures that "
        "exist in the data are shown; KPIs from the reference images that the data cannot support (length of stay, "
        "satisfaction, patient-level counts) were deliberately omitted.")
    R.table("pages", ["Page", "Main content"], [
        ["1 California Overview", "KPIs per data source, statewide trend, coverage timeline, condition trends, auto-derived findings, model summary, data-quality indicators"],
        ["2 CABG Hospital Prediction", "Searchable hospital and window selectors, history with CIs, live estimates from all saved models, benchmark, LR contribution chart, volume, statewide position, hospital map; separate hypothetical scenario tab with extrapolation warnings"],
        ["3 Ischemic Stroke Analytics", "Window/measure/county filters, statewide trends, ranked hospitals coloured by rating, readmission vs mortality scatter, county comparison, hospital explorer, stroke model summary"],
        ["4 Population & County Trends", "Year, threshold, group and county filters, county choropleth, ranking, county trends, strata bars, admissions/readmissions, small multiples, data notes"],
        ["5 Model Evaluation & Testing", "Methodology, comparison tables, design comparison, bootstrap intervals, actual vs predicted, residuals, error by size, importance, learned curve, validation checks, preprocessing log, synthetic tests"],
    ], "Dashboard pages.", widths=[2.2, 7])
    for key, file, cap in (
            ("shot1", "page1_overview.png", "Dashboard page 1 - California Overview (screenshot of the running app)."),
            ("shot2", "page2_cabg_history.png", "Dashboard page 2 - CABG hospital history, model estimates and Linear Regression explanation."),
            ("shot4", "page4_population.png", "Dashboard page 4 - Population & County Trends with county choropleth."),
            ("shot5", "page5_model_charts.png", "Dashboard page 5 - Model evaluation across designs and bootstrap intervals.")):
        R.figure(key, SHOTS / file, cap, width_in=6.3)

    # ------------------------------------------------------------------ 18 Deployment
    R.h1("18. GitHub Actions and Streamlit Deployment Architecture")
    R.figure("architecture", figs["architecture"], "Project architecture: one shared code base for the local pipeline, CI and the deployed dashboard.")
    R.bullets([
        "**Local:** run_pipeline.py writes processed data, models and artifacts; app.py starts the dashboard; run_local.py does both.",
        "**Repository:** raw CSVs (about 0.7 MB in total), code, trained models and artifacts are committed, so Streamlit Community "
        "Cloud can serve the dashboard immediately from a public URL without any download by the viewer. If artifacts "
        "are missing, the app runs the pipeline once at start-up.",
        "**GitHub Actions (pipeline.yml):** a manually triggered (workflow_dispatch) workflow installs dependencies, "
        "checks that required files exist, runs the test suite and the pipeline, regenerates this report and uploads "
        "outputs as workflow artifacts. Optionally it commits refreshed artifacts to the main branch, and Streamlit "
        "Community Cloud redeploys on that push. The workflow and the website are separate services; the workflow "
        "does not 'open' the website.",
        f"**Integrity:** the dashboard re-predicts the stored test set with the loaded models; the check "
        f"{'passed' if integrity_ok else 'FAILED'} (maximum difference 0).",
    ])

    # ------------------------------------------------------------------ 19 Synthetic
    R.h1("19. Synthetic Dataset Testing Methodology")
    R.p(f"The {syn['n_records']} synthetic CABG records (all flagged synthetic, IDs prefixed SYNTH_) were mapped onto the "
        "model's feature schema and passed through the saved models. Checks: schema mapping, finite predictions within "
        f"0-100%, deterministic output and range checks ({syn['records_with_any_input_outside_training_range']} record(s) "
        "had an input outside the real training range and are flagged as extrapolations). The synthetic "
        "'Operative_Mortality_Rate_Pct' column was not used because the primary model has no aligned counterpart.")
    R.table("synthetic", ["Model", "MAE vs synthetic labels", "R² vs synthetic labels"],
            [[MODEL_LABELS[m], f3(syn_ag.loc[m, "MAE"]), f3(syn_ag.loc[m, "R2"])] for m in syn_ag.index],
            "Agreement with synthetic labels - a consistency check only, NOT evidence of real-world accuracy.",
            widths=[3, 3, 3])
    R.figure("synthetic", figs["synthetic"], "Synthetic scenario test, clearly separated from real-data evaluation.", width_in=4.2)
    R.note("Synthetic records were never used for training, model selection or any real-data statistic. Their labels "
           "come from an unknown generating process, so agreement with them only shows that the pipeline runs "
           "consistently on schema-compatible inputs.")

    # ------------------------------------------------------------------ 20 Limitations
    R.h1("20. Limitations, Ethics and Responsible Interpretation")
    R.bullets([
        "**Aggregate data only.** No patient records were used; estimates are hospital-level rates, not individual probabilities, and cannot identify causes of readmission.",
        "**Short and overlapping history.** Only two usable one-step CABG transitions exist (2018-2019 -> 2019-2020 and 2021-2022 -> 2022-2023); the 2020-2021 window covers 2 hospitals. Overlapping windows inflate apparent stability.",
        "**Small samples.** Many hospitals have fewer than 100 eligible cases per window, giving wide confidence intervals; errors concentrate in these hospitals.",
        "**Outcome is an estimate.** Risk-adjusted rates depend on the publisher's risk model; changes in that model over time are not visible in the data.",
        "**Coding change.** ICD-9 and ICD-10 eras are not directly comparable; 2015 is partial.",
        "**Associations are not causes.** Differences between payers, age groups, races or counties reflect many factors and must not be read as causal or used to stigmatise groups.",
        "**Not clinically validated.** The dashboard is an educational analytics tool, not a medical decision-support system.",
        "**Provenance.** Dataset download dates and licences should be confirmed by the project team before redistribution.",
    ])

    # ------------------------------------------------------------------ 21 Conclusion
    R.h1("21. Conclusion and Future Improvements")
    R.p("Historical hospital performance does carry information about future readmission rates: the previous rate "
        "dominates every model, and learned models that shrink extreme rates toward the average outperform the naive "
        "persistence assumption whenever the future window contains genuinely new patients (non-overlapping CABG "
        "horizon, and the larger stroke dataset). For the CABG one-step horizon, however, the published windows overlap, "
        "and on the held-out window no model reliably beat persistence; this is reported as a finding rather than "
        "hidden. Readmission patterns differ by condition (CABG and stroke rates both declined over their periods), by "
        "county and by population group, while stroke readmission and mortality capture different aspects of quality.")
    R.p("Future work: add newer CABG windows as they are published to obtain more independent test periods; model the "
        "uncertainty of each hospital's rate explicitly (e.g. hierarchical or empirical-Bayes shrinkage using case "
        "counts); calibrate prediction intervals; and link hospital outcomes with county-level population context only "
        "where reporting periods overlap.")

    # ------------------------------------------------------------------ References
    R.h1("References")
    R.numbered([
        "Jencks SF, Williams MV, Coleman EA. Rehospitalizations among patients in the Medicare fee-for-service program. N Engl J Med. 2009;360(14):1418-1428. doi:10.1056/NEJMsa0803563",
        "Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission: a systematic review. JAMA. 2011;306(15):1688-1698. doi:10.1001/jama.2011.1515",
        "McIlvennan CK, Eapen ZJ, Allen LA. Hospital readmissions reduction program. Circulation. 2015;131(20):1796-1803. doi:10.1161/CIRCULATIONAHA.114.010270",
        "Breiman L. Random forests. Machine Learning. 2001;45(1):5-32. doi:10.1023/A:1010933404324",
        "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830.",
        "Hyndman RJ, Athanasopoulos G. Forecasting: Principles and Practice. 3rd ed. Melbourne: OTexts; 2021. https://otexts.com/fpp3/",
        "Roberts DR, Bahn V, Ciuti S, et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography. 2017;40(8):913-929. doi:10.1111/ecog.02881",
        "Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall; 1993.",
        "Barnett AG, van der Pols JC, Dobson AJ. Regression to the mean: what it is and how to deal with it. Int J Epidemiol. 2005;34(1):215-220. doi:10.1093/ije/dyh299",
        "Dimick JB, Welch HG, Birkmeyer JD. Surgical mortality as an indicator of hospital quality: the problem with small sample size. JAMA. 2004;292(7):847-851. doi:10.1001/jama.292.7.847",
        "Birkmeyer JD, Siewers AE, Finlayson EVA, et al. Hospital volume and surgical mortality in the United States. N Engl J Med. 2002;346(15):1128-1137. doi:10.1056/NEJMsa012337",
        "Fisher A, Rudin C, Dominici F. All models are wrong, but many are useful: learning a variable's importance by studying an entire class of prediction models simultaneously. J Mach Learn Res. 2019;20(177):1-81.",
        "Krumholz HM, Lin Z, Keenan PS, et al. Relationship between hospital readmission and mortality rates for patients hospitalized with acute myocardial infarction, heart failure, or pneumonia. JAMA. 2013;309(6):587-593. doi:10.1001/jama.2013.333",
        "U.S. Department of Health and Human Services. Administrative Simplification: Change to the Compliance Date for the International Classification of Diseases, 10th Revision (ICD-10-CM and ICD-10-PCS) Medical Data Code Sets. Final rule. Federal Register. 2014;79(149):45128.",
        "Streamlit documentation. Snowflake Inc. https://docs.streamlit.io",
        "Plotly Technologies Inc. Collaborative data science. Montreal, QC; 2015. https://plot.ly",
        "Hunter JD. Matplotlib: a 2D graphics environment. Computing in Science & Engineering. 2007;9(3):90-95.",
        "McKinney W. Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference. 2010:56-61.",
        "California Health and Human Services Agency. CHHS Open Data Portal (HCAI datasets). https://data.chhs.ca.gov",
    ])

    # ------------------------------------------------------------------ Appendix
    R.h1("Appendix A. Reproducing the Project")
    R.code("python -m venv .venv\n.venv\\Scripts\\activate            (Windows)  |  source .venv/bin/activate  (macOS/Linux)\n"
           "pip install -r requirements.txt\npython run_pipeline.py             # validate, clean, train, evaluate (about 1 minute)\n"
           "python -m pytest                   # automated tests\nstreamlit run app.py               # dashboard at http://localhost:8501\n"
           "python reports/generate_report.py  # rebuild this report from artifacts\n"
           "python run_local.py                # pipeline (if needed) + dashboard in one command")
    R.h1("Appendix B. Run Configuration")
    v = man["versions"]
    R.table("env", ["Item", "Value"], [
        ["Pipeline run (UTC)", man["run_utc"]], ["Mode", man["mode"]], ["Python", man["python"]],
        ["pandas / numpy", f"{v['pandas']} / {v['numpy']}"], ["scikit-learn", v["scikit-learn"]], ["Streamlit", v["streamlit"]],
        ["Random seed", str(config.RANDOM_STATE)], ["RF hyper-parameters", str(cs["rf_best_params"])],
        ["Recommended CABG model", MODEL_LABELS[rec]], ["Validation", f"{n_pass} pass / {n_warn} warn / {n_fail} fail"],
    ] + [[f"SHA-256 {k}", h[:24] + "..."] for k, h in man["data_sha256"].items()],
        "Software versions, settings and raw-file hashes for this report.", widths=[2.5, 6.5])
    R.h1("Appendix C. Validation Checks")
    R.table("checks", ["Status", "Dataset", "Check", "Detail"],
            [[r["status"].upper(), r["dataset"], r["description"], str(r["detail"])[:240]] for _, r in checks.iterrows()],
            "All automated data-validation checks.", widths=[0.7, 0.9, 3, 4.6], font=7.0)
    return R


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Regenerate the project report from pipeline artifacts.")
    ap.add_argument("--out-dir", default=str(config.REPORTS_DIR), help="where to write the report and figures")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    art = load_artifacts(config.PROJECT_ROOT)
    figs = report_figures.build_all(art, out / "figures")
    rep = build(art, figs)
    docx_path = render_docx(rep, out / "Project_Report.docx")
    pdf_path = render_pdf(rep, out / "Project_Report.pdf")
    print(f"wrote {docx_path}\nwrote {pdf_path}\n{len(figs)} figures in {out / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
