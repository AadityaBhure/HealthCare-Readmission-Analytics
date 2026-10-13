"""Page 1 - California Readmission Overview."""
import pandas as pd
import streamlit as st

from src import dashboard_data as dd, ui, visualizations as viz
from src.modeling import MODEL_LABELS
from src.theme import SERIES_DARK, STATUS

pop = dd.artifact_json("population/summary.json")
state = dd.artifact_csv("population/statewide.csv")
cabg_sw = dd.artifact_csv("cabg/statewide.csv")
stroke_sw = dd.artifact_csv("stroke/statewide.csv")
findings = dd.artifact_json("key_findings.json")
cabg_metrics = dd.artifact_csv("cabg/metrics_test.csv")
stroke_metrics = dd.artifact_csv("stroke/metrics_test.csv")
cabg_summary = dd.artifact_json("cabg/summary.json")
stroke_summary = dd.artifact_json("stroke/summary.json")
checks = dd.artifact_csv("validation_checks.csv")
audit = dd.artifact_json("dataset_audit.json")

ui.page_header(
    "Healthcare analytics - California",
    "California Hospital Readmission Overview",
    "Explainable analytics across three separate views of 30-day readmission: statewide all-cause trends by "
    "population group and county, hospital-level isolated CABG surgery outcomes with a next-period prediction "
    "model, and ischemic stroke readmission and mortality. Each dataset keeps its own measure, unit and reporting "
    "period, so rates from different sources are shown side by side, never merged.",
    chips=[("Hospitals (CABG)", f"{audit['cabg']['hospital_ids']}"), ("Hospitals (stroke)", f"{audit['stroke']['hospital_ids']}"),
           ("Counties", f"{pop['n_counties']}"), ("All-cause years", f"{audit['allcause']['years'][0]}-{audit['allcause']['years'][1]}"),
           ("Real-data model test", f"{cabg_summary['train_summary']['n_test']} hospitals")])

# ---------------- KPI row ----------------
readm = cabg_sw[cabg_sw["measure_code"] == "readmission"].reset_index(drop=True)
full = readm[readm["hospitals"] >= 10]
c_cur, c_prev = full.iloc[-1], full.iloc[-2]
s_r = stroke_sw[stroke_sw["measure_code"] == "readmission"].reset_index(drop=True)
s_m = stroke_sw[stroke_sw["measure_code"] == "mortality"].reset_index(drop=True)
rec = cabg_summary["recommended_model"]
cm = cabg_metrics.set_index("model")


def _dir(d):
    return "up" if d > 0.005 else ("down" if d < -0.005 else "flat")


cols = st.columns(5)
with cols[0]:
    d = pop["yoy_change_pp"]
    ui.kpi("All-cause 30-day readmission", f"{pop['latest_rate']:.2f}", "%", f"{d:+.2f} pp vs {pop['latest_year'] - 1}", _dir(d),
           f"Statewide, all patients, {pop['latest_year']} ({pop['latest_coding_system']}) - "
           f"{pop['latest_admits'] / 1e6:.2f}M index admissions", SERIES_DARK[0])
with cols[1]:
    d = c_cur["rate"] - c_prev["rate"]
    ui.kpi("Isolated CABG 30-day readmission", f"{c_cur['rate']:.2f}", "%", f"{d:+.2f} pp vs {c_prev['period']}", _dir(d),
           f"Statewide observed rate, {c_cur['period']} window - {int(c_cur['hospitals'])} hospitals", SERIES_DARK[5])
with cols[2]:
    d = s_r.iloc[-1]["risk_adjusted_rate"] - s_r.iloc[-2]["risk_adjusted_rate"]
    ui.kpi("Ischemic stroke 30-day readmission", f"{s_r.iloc[-1]['risk_adjusted_rate']:.2f}", "%",
           f"{d:+.2f} pp vs {s_r.iloc[-2]['period']}", _dir(d),
           f"Statewide risk-adjusted, {s_r.iloc[-1]['period']} window", SERIES_DARK[0])
with cols[3]:
    d = s_m.iloc[-1]["risk_adjusted_rate"] - s_m.iloc[-2]["risk_adjusted_rate"]
    ui.kpi("Ischemic stroke 30-day mortality", f"{s_m.iloc[-1]['risk_adjusted_rate']:.2f}", "%",
           f"{d:+.2f} pp vs {s_m.iloc[-2]['period']}", _dir(d),
           f"Statewide risk-adjusted, {s_m.iloc[-1]['period']} window", SERIES_DARK[1])
with cols[4]:
    diff = cm.loc[rec, "MAE"] - cm.loc["persistence", "MAE"]
    ui.kpi(f"CABG model error ({MODEL_LABELS[rec]})", f"{cm.loc[rec, 'MAE']:.2f}", " pp MAE",
           f"{diff:+.2f} pp vs persistence ({cm.loc['persistence', 'MAE']:.2f})", "flat",
           f"Held-out {cabg_summary['train_summary']['test_target_periods'][0]} window, "
           f"n = {cabg_summary['train_summary']['n_test']} hospitals", SERIES_DARK[2])

st.write("")
left, right = st.columns([1.35, 1])
with left:
    with ui.card("ov-trend"):
        ui.section("Statewide all-cause 30-day readmission rate",
                   "All payers and ages. Hollow marker: 2015 is a partial year (final ICD-9 year). Rates from the two coding "
                   "eras are shown as separate series because they are not directly comparable.")
        ui.plot(viz.statewide_allcause_trend(state), key="ov_trend")
with right:
    with ui.card("ov-coverage"):
        ui.section("Data coverage by reporting period",
                   "Each bar is one published period. CABG and stroke use overlapping two-year windows; the faded "
                   "2020-2021 CABG bar covers only 2 hospitals.")
        cov = dict(zip(readm["period"], readm["hospitals"]))
        ui.plot(viz.coverage_timeline(cabg_sw, stroke_sw, state, cov, height=360), key="ov_cov")

with ui.card("ov-conditions"):
    ui.section("Condition-specific statewide outcomes",
               "Different measures, cohorts and periods: CABG 2018-2023 (observed statewide rate of isolated CABG patients) "
               "vs ischemic stroke 2011-2015 (risk-adjusted). Compare trends within each panel, not levels across panels.")
    ui.plot(viz.condition_trends(cabg_sw, stroke_sw), key="ov_cond")

left, right = st.columns([1.35, 1])
with left:
    with ui.card("ov-findings"):
        ui.section("Key findings", "Generated automatically from the pipeline outputs (artifacts/key_findings.json).")
        ui.findings_list(findings)
with right:
    with ui.card("ov-models"):
        ui.section("Real-data model evaluation (temporal hold-out)",
                   "Mean absolute error in percentage points on the latest window, never used for training.")
        rows = []
        for name, m, s in (("CABG", cabg_metrics, cabg_summary), ("Stroke", stroke_metrics, stroke_summary)):
            for _, r in m.iterrows():
                rows.append({"Condition": name, "Model": r["label"], "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"],
                             "Test n": int(r["n_test"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                                    "RMSE": st.column_config.NumberColumn(format="%.2f"),
                                    "R2": st.column_config.NumberColumn("R²", format="%.3f")})
        st.caption("Details, uncertainty intervals and sensitivity analyses: Model Evaluation & Testing page.")
    with ui.card("ov-quality"):
        ui.section("Data quality indicators")
        warn = checks[checks["status"] != "pass"]
        n_pass = int((checks["status"] == "pass").sum())
        st.markdown(f"{ui.pill('pass')} {n_pass} checks passed &nbsp; {ui.pill('warn')} {len(warn)} warnings",
                    unsafe_allow_html=True)
        for _, r in warn.iterrows():
            st.markdown(f"<div class='section-caption'>&#9888; <b>{r['description']}</b> - {r['detail'][:220]}</div>",
                        unsafe_allow_html=True)
        st.markdown(f"<div class='section-caption'>&#9432; ICD-9 (2011-{pop['transition_year']}) and ICD-10 "
                    f"({pop['icd10_first_year']}-{pop['latest_year']}) all-cause rates are separate series.</div>",
                    unsafe_allow_html=True)
ui.footer()
