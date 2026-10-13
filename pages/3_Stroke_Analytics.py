"""Page 3 - Ischemic stroke 30-day readmission and mortality."""
import pandas as pd
import streamlit as st

from src import dashboard_data as dd, ui, visualizations as viz
from src.theme import SERIES_DARK, STATUS

panel = dd.processed_csv("stroke_panel")
sw = dd.artifact_csv("stroke/statewide.csv")
county = dd.artifact_csv("stroke/county_summary.csv")
metrics = dd.artifact_csv("stroke/metrics_test.csv")
summary = dd.artifact_json("stroke/summary.json")
boot = dd.artifact_csv("stroke/bootstrap_mae_diff.csv")

ui.page_header(
    "Hospital-level stroke outcomes",
    "Ischemic Stroke Analytics",
    "30-day readmission and 30-day mortality after an ischemic stroke admission, risk-adjusted by the publisher, for "
    "California hospitals in four overlapping two-year windows (2011-2012 to 2014-2015). These years do not overlap "
    "with the CABG data (2018-2023); the stroke analysis is a separate experiment.",
    chips=[("Hospitals", str(panel["hospital_id"].nunique())), ("Counties", str(panel["county"].nunique())),
           ("Windows", f"{panel['period'].min()} to {panel['period'].max()}")])

periods = sorted(panel["period"].unique())
with ui.filter_bar("stroke"):
    c = st.columns([1, 1.1, 1.6, 1, 1])
    period = c[0].selectbox("Reporting window", periods, index=len(periods) - 1)
    measure = c[1].selectbox("Measure", ["30-day Readmission", "30-day Mortality"])
    counties = c[2].multiselect("Counties (optional)", sorted(panel["county"].dropna().unique()), placeholder="All counties")
    min_cases = c[3].number_input("Min. cases per hospital", 0, 500, 50, 10,
                                  help="Hospitals with few cases have very wide uncertainty; they are hidden from rankings.")
    top_n = c[4].slider("Hospitals per ranking", 5, 25, 12)

prefix = "readmit" if measure == "30-day Readmission" else "mortality"
cur = panel[panel["period"] == period].copy()
if counties:
    cur = cur[cur["county"].isin(counties)]
sw_p = sw[sw["period"] == period].set_index("measure")
sw_rate = float(sw_p.loc[measure, "risk_adjusted_rate"])

k = st.columns(5)
with k[0]:
    ui.kpi(f"Statewide readmission, {period}", f"{sw_p.loc['30-day Readmission', 'risk_adjusted_rate']:.2f}", "%", None, "flat",
           f"{sw_p.loc['30-day Readmission', 'events']:,.0f} readmissions / {sw_p.loc['30-day Readmission', 'cases']:,.0f} cases",
           SERIES_DARK[0])
with k[1]:
    ui.kpi(f"Statewide mortality, {period}", f"{sw_p.loc['30-day Mortality', 'risk_adjusted_rate']:.2f}", "%", None, "flat",
           f"{sw_p.loc['30-day Mortality', 'events']:,.0f} deaths / {sw_p.loc['30-day Mortality', 'cases']:,.0f} cases",
           SERIES_DARK[1])
rated = cur[f"{prefix}_rating"].value_counts()
with k[2]:
    ui.kpi("Hospitals reporting", f"{cur[f'{prefix}_rate'].notna().sum()}", "", None, "flat",
           f"{measure}, {period}" + (" - filtered counties" if counties else ""), "#7D8BA3")
with k[3]:
    ui.kpi("Rated 'Worse' than expected", f"{int(rated.get('Worse', 0))}", " hospitals", None, "flat",
           f"Publisher rating for {measure.lower()}", STATUS["critical"])
with k[4]:
    ui.kpi("Rated 'Better' than expected", f"{int(rated.get('Better', 0))}", " hospitals", None, "flat",
           f"Publisher rating for {measure.lower()}", STATUS["good"])

st.write("")
left, right = st.columns([1.4, 1])
with left:
    with ui.card("st-trend"):
        ui.section("Statewide risk-adjusted rates by window",
                   "Consecutive windows share one calendar year, so changes between neighbouring points are smoothed.")
        ui.plot(viz.stroke_statewide(sw), key="st_trend")
with right:
    with ui.card("st-explain"):
        ui.section("Readmission vs mortality - what is the difference?")
        st.markdown(
            "<div class='section-caption' style='font-size:0.88rem'>"
            "<b>30-day mortality</b>: share of ischemic stroke patients who died from any cause within 30 days of "
            "admission.<br><br><b>30-day readmission</b>: share of patients discharged alive who were readmitted to a "
            "hospital within 30 days. Patients who die cannot be readmitted, so the two measures describe different "
            "cohorts and are complementary, not interchangeable.<br><br>Both are <b>risk-adjusted</b> by the "
            "publisher for patient case mix, and ratings compare each hospital with the statewide expectation "
            "(Better / As Expected / Worse).</div>", unsafe_allow_html=True)
        d = summary["lag_correlation"]
        st.markdown(f"<div class='section-caption'>Window-to-window correlation of hospital readmission rates: "
                    f"r = {d['one_step_overlapping']:.2f} for consecutive (overlapping) windows and "
                    f"r = {d['non_overlapping']:.2f} for non-overlapping windows.</div>", unsafe_allow_html=True)

ranked = cur.dropna(subset=[f"{prefix}_rate"])
ranked = ranked[ranked[f"{prefix}_cases"].fillna(0) >= min_cases].copy()
ranked["cases"] = ranked[f"{prefix}_cases"]
ranked["label"] = ranked["hospital_name"].str.slice(0, 42)
xl = f"{measure} risk-adjusted rate (%)"
left, right = st.columns(2)
with left:
    with ui.card("st-high"):
        ui.section(f"Highest {measure.lower()} rates, {period}",
                   f"Hospitals with at least {min_cases} cases. Colour = publisher rating; dotted line = statewide rate.")
        if ranked.empty:
            st.info("No hospitals match the filters.")
        else:
            top = ranked.nlargest(top_n, f"{prefix}_rate")
            ui.plot(viz.ranked_hospitals(top, f"{prefix}_rate", "label", f"{prefix}_rating", sw_rate, xl,
                                         height=110 + 26 * len(top)), key="st_high")
with right:
    with ui.card("st-low"):
        ui.section(f"Lowest {measure.lower()} rates, {period}",
                   "A lower rate is not proof of better care - small cohorts and coding differences add noise.")
        if not ranked.empty:
            low = ranked.nsmallest(top_n, f"{prefix}_rate")
            ui.plot(viz.ranked_hospitals(low, f"{prefix}_rate", "label", f"{prefix}_rating", sw_rate, xl,
                                         height=110 + 26 * len(low)), key="st_low")

left, right = st.columns(2)
with left:
    with ui.card("st-scatter"):
        ui.section(f"Readmission vs mortality by hospital, {period}",
                   "Each dot is a hospital (size = cases). A weak correlation means the two outcomes rank hospitals differently.")
        ui.plot(viz.readmission_vs_mortality(panel if not counties else panel[panel["county"].isin(counties)], period),
                key="st_scatter")
with right:
    with ui.card("st-county"):
        cs = county[(county["period"] == period) & (county["measure"] == measure)]
        if counties:
            cs = cs[cs["county"].isin(counties)]
        cs = cs[cs["hospitals"] >= 2]
        ui.section(f"County comparison (case-weighted), {period}",
                   "Case-weighted mean of hospital risk-adjusted rates for counties with 2 or more reporting hospitals. "
                   "Descriptive only - not a county-level risk-adjusted rate.")
        if cs.empty:
            st.info("No county has two or more reporting hospitals for this selection.")
        else:
            cs = cs.rename(columns={"weighted_rate": "rate"})
            ui.plot(viz.county_bars(cs.assign(admits=cs["cases"]), "rate", "county", xl, sw_rate, "Statewide",
                                    height=120 + 19 * len(cs), hover_extra="<br>%{customdata[0]:,.0f} cases"), key="st_county")

with ui.card("st-hospital"):
    ui.section("Hospital explorer", "Readmission and mortality across all four windows for one hospital, against statewide rates.")
    pool = panel if not counties else panel[panel["county"].isin(counties)]
    names = (pool.groupby("hospital_id").agg(name=("hospital_name", "last"), county=("county", "last"),
                                              cases=("readmit_cases", "sum")).reset_index())
    names["label"] = names["name"] + " - " + names["county"].fillna("") + " (" + names["hospital_id"] + ")"
    names = names.sort_values("name").reset_index(drop=True)
    pick = st.selectbox("Hospital", names["label"], index=int(names["cases"].idxmax()))
    hid = names.loc[names["label"] == pick, "hospital_id"].iloc[0]
    h = panel[panel["hospital_id"] == hid].sort_values("period")
    a, b = st.columns([1.6, 1])
    with a:
        ui.plot(viz.hospital_stroke_trend(h, sw), key="st_htrend")
    with b:
        st.dataframe(h[["period", "readmit_rate", "readmit_cases", "readmit_rating", "mortality_rate", "mortality_cases",
                        "mortality_rating"]].rename(columns={
            "period": "Window", "readmit_rate": "Readmission %", "readmit_cases": "Readm. cases",
            "readmit_rating": "Readm. rating", "mortality_rate": "Mortality %", "mortality_cases": "Mort. cases",
            "mortality_rating": "Mort. rating"}), hide_index=True, width="stretch")
        st.caption("Blank cells were not published (kept missing, not zero).")

with ui.card("st-model"):
    ui.section("Separate stroke model: next-window readmission rate",
               f"Trained on {summary['train_summary']['n_train']} transitions "
               f"({', '.join(summary['train_summary']['train_target_periods'])} targets) and tested on "
               f"{summary['train_summary']['n_test']} hospitals ({summary['train_summary']['test_target_periods'][0]}). "
               "Inputs: previous readmission rate, previous mortality rate, previous case volume.")
    a, b = st.columns([1.2, 1])
    with a:
        st.dataframe(metrics[["label", "MAE", "RMSE", "R2", "n_train", "n_test"]].rename(
            columns={"label": "Model", "R2": "R²", "n_train": "Train n", "n_test": "Test n"}),
            hide_index=True, width="stretch",
            column_config={"MAE": st.column_config.NumberColumn(format="%.3f"),
                           "RMSE": st.column_config.NumberColumn(format="%.3f"),
                           "R²": st.column_config.NumberColumn(format="%.3f")})
    with b:
        r = boot[(boot["model"] == "linear_regression") & (boot["reference"] == "persistence")].iloc[0]
        ui.callout(f"Linear Regression vs persistence: MAE difference <b>{r['mae_diff']:+.3f} pp</b> "
                   f"(95% bootstrap CI {r['ci_low']:+.3f} to {r['ci_high']:+.3f}). "
                   + ("The interval excludes zero." if r["ci_high"] < 0 else
                      "The interval includes zero, so the improvement is suggestive rather than conclusive.")
                   + " See Model Evaluation & Testing for grouped cross-validation and residuals.", "info")
ui.footer()
