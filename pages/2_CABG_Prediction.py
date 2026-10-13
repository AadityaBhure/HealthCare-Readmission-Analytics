"""Page 2 - CABG hospital history and next-period readmission estimate."""
import numpy as np
import pandas as pd
import streamlit as st

from src import cabg_model, dashboard_data as dd, ui, visualizations as viz
from src.modeling import MODEL_LABELS
from src.theme import MODEL_COLORS_DARK, SERIES_DARK

panel = dd.processed_csv("cabg_panel")
preds_table = dd.artifact_csv("cabg/hospital_predictions.csv")
statewide_all = dd.artifact_csv("cabg/statewide.csv")
statewide = statewide_all[statewide_all["measure_code"] == "readmission"].reset_index(drop=True)
models, schema = dd.cabg_bundle()
synthetic = dd.processed_csv("synthetic_clean")
FEATURES = schema["features"]
LABELS = {"prev_readmit_rate": "Prev. readmission", "prev_stroke_rate": "Prev. post-op stroke",
          "prev_cabg_volume": "Prev. CABG volume"}

ui.page_header(
    "Hospital-level surgical outcomes",
    "CABG Hospital Prediction",
    "Historical isolated CABG 30-day readmission performance for one hospital and an <b>estimated next-window "
    "risk-adjusted readmission rate</b>. Estimates are hospital-level rates (percentage of a hospital's eligible "
    "isolated CABG patients), not an individual patient's probability of readmission.",
    chips=[("Target", "next two-year window RAAER (%)"), ("Inputs", "previous window only"),
           ("Recommended model", MODEL_LABELS[schema["recommended_model"]])])

tab_real, tab_scenario = st.tabs(["Real hospital estimate", "Hypothetical scenario (manual inputs)"])

# ======================================================================================
with tab_real:
    hospitals = (panel.groupby("hospital_id")
                 .agg(name=("hospital_name", "last"), county=("county", "last"), volume=("cabg_volume", "sum"))
                 .reset_index())
    hospitals["label"] = hospitals["name"] + " - " + hospitals["county"].fillna("") + " (" + hospitals["hospital_id"] + ")"
    hospitals = hospitals.sort_values("name").reset_index(drop=True)
    default_idx = int(hospitals["volume"].idxmax())

    with ui.filter_bar("cabg"):
        c1, c2, c3 = st.columns([2.2, 1.1, 1.3])
        sel_label = c1.selectbox("Hospital (type to search by name, county or ID)", hospitals["label"], index=default_idx)
        hid = hospitals.loc[hospitals["label"] == sel_label, "hospital_id"].iloc[0]
        rows = preds_table[preds_table["hospital_id"] == hid].sort_values("input_period")
        valid = rows[rows["status"] != "No estimate - input window lacks required measures"]
        if valid.empty:
            st.warning("This hospital has no reporting window with all required inputs.")
            st.stop()
        periods = valid["input_period"].tolist()
        in_period = c2.selectbox("Input window (historical)", periods, index=len(periods) - 1,
                                 help="Only windows where readmission, post-operative stroke and case volume were all "
                                      "published are offered.")
        model_names = [m for m in ("linear_regression", "random_forest", "persistence") if m in models]
        model_choice = c3.selectbox("Model shown as headline", model_names,
                                    index=model_names.index(schema["recommended_model"])
                                    if schema["recommended_model"] in model_names else 0,
                                    format_func=lambda m: MODEL_LABELS[m] + (" (recommended)" if m == schema["recommended_model"] else ""))

    row = valid[valid["input_period"] == in_period].iloc[0]
    target = row["target_period"]
    values = {f: float(row[f]) for f in FEATURES}
    est = cabg_model.predict(models, schema, values)  # live inference with the evaluated model objects
    est_sel = float(est.loc[est["model"] == model_choice, "estimate"].iloc[0])
    pers = float(est.loc[est["model"] == "persistence", "estimate"].iloc[0])
    actual = row["actual_next"]
    hist = panel[panel["hospital_id"] == hid].sort_values("period_start")
    status = row["status"]

    kind = "info"
    if status.startswith("Training"):
        msg = (f"<b>In-sample back-test.</b> The {in_period} -> {target} transition was part of the training data, so "
               "the comparison with the published value is optimistic.")
        kind = "warn"
    elif status.startswith("Test"):
        msg = (f"<b>Out-of-sample back-test.</b> The model never saw the {target} outcome; this hospital is part of "
               "the held-out evaluation set.")
    elif status.startswith("Forward"):
        msg = (f"<b>Forward estimate.</b> {target} has not been published in this dataset, so there is no observed value "
               "to compare against yet.")
    else:
        msg = (f"<b>No published outcome.</b> {target} readmission data were not published for this hospital "
               "(the 2020-2021 window covers only 2 hospitals statewide).")
        kind = "warn"
    ui.callout(msg + f" Inputs come only from the {in_period} window. Model: {MODEL_LABELS[model_choice]}.", kind)

    sw_target = statewide[statewide["period"] == target]
    if not sw_target.empty and sw_target["hospitals"].iloc[0] >= 10:
        bench, bench_label = float(sw_target["rate"].iloc[0]), f"Statewide observed rate, {target}"
    else:
        last = statewide[statewide["hospitals"] >= 10].iloc[-1]
        bench, bench_label = float(last["rate"]), f"Latest published statewide rate ({last['period']}); {target} not yet published"

    k = st.columns(5)
    with k[0]:
        ui.kpi(f"Observed RAAER, {in_period}", f"{row['prev_readmit_rate']:.2f}", "%", None, "flat",
               f"95% CI {row['readmit_ci_low']:.2f}-{row['readmit_ci_high']:.2f}% - rating: {row['readmit_rating']} - "
               f"{row['readmit_cases']:.0f} eligible cases", SERIES_DARK[5])
    with k[1]:
        ui.kpi(f"Estimated RAAER, {target}", f"{est_sel:.2f}", "%", f"{est_sel - row['prev_readmit_rate']:+.2f} pp vs input window",
               "up" if est_sel > row["prev_readmit_rate"] else "down",
               f"{MODEL_LABELS[model_choice]} estimate - hospital-level rate", MODEL_COLORS_DARK.get(model_choice, SERIES_DARK[0]))
    with k[2]:
        ui.kpi("Persistence baseline", f"{pers:.2f}", "%", None, "flat",
               f"Next window = previous window ({in_period})", MODEL_COLORS_DARK["persistence"])
    with k[3]:
        ui.kpi("Benchmark", f"{bench:.2f}", "%", None, "flat", bench_label, "#7D8BA3")
    with k[4]:
        if pd.notna(actual):
            err = est_sel - actual
            ui.kpi(f"Published actual, {target}", f"{actual:.2f}", "%", f"estimate error {err:+.2f} pp", "flat",
                   f"Persistence error {pers - actual:+.2f} pp", SERIES_DARK[3])
        else:
            ui.kpi(f"Published actual, {target}", "n/a", "", None, "flat", "Outcome not available in the dataset", SERIES_DARK[3])

    st.write("")
    left, right = st.columns([1.55, 1])
    with left:
        with ui.card("cabg-history"):
            ui.section("Readmission history and next-window estimate",
                       "White: observed risk-adjusted rate with 95% CI. Markers at the target window: model estimates "
                       "(dashed line = headline model). Star: published actual, when available.")
            est_plot = est.copy()
            est_plot["recommended"] = est_plot["model"].eq(model_choice)
            fig = viz.hospital_history(hist, statewide, est_plot, target, actual if pd.notna(actual) else None)
            ui.plot(fig, key="cabg_hist")
    with right:
        with ui.card("cabg-models"):
            ui.section(f"All models for {target}", "Same inputs, every saved model. MAE = error on the held-out test window.")
            test_m = dd.artifact_csv("cabg/metrics_test.csv").set_index("model")
            tbl = est[est["model"] != "mean_baseline"].copy()
            tbl["Test MAE"] = tbl["model"].map(test_m["MAE"])
            tbl["Model"] = tbl["label"].str.replace(" baseline", "") + np.where(tbl["recommended"], " *", "")
            st.dataframe(tbl[["Model", "estimate", "Test MAE"]].rename(columns={"estimate": "Estimate"}),
                         hide_index=True, width="stretch",
                         column_config={"Estimate": st.column_config.NumberColumn(format="%.2f%%"),
                                        "Test MAE": st.column_config.NumberColumn(format="%.2f pp")})
            st.caption(f"(*) headline model. Test MAE = mean absolute error on the held-out "
                       f"{schema['train_summary']['test_target_periods'][0]} window.")
        with ui.card("cabg-explain"):
            contrib, intercept = cabg_model.explain_linear(models, schema, values)
            lr_est = float(est.loc[est["model"] == "linear_regression", "estimate"].iloc[0])
            ui.section("Why this estimate? (Linear Regression)",
                       f"Exact decomposition: average-hospital estimate {intercept:.2f}% + contributions = {lr_est:.2f}%. "
                       "Orange raises the estimate, blue lowers it.")
            ui.plot(viz.contribution_chart(contrib, intercept, LABELS, height=230), key="cabg_contrib")

    left, right = st.columns([1, 1])
    with left:
        with ui.card("cabg-volume"):
            ui.section("CABG case volume by window", "Total hospital CABG cases (all CABG types) in each two-year window.")
            ui.plot(viz.volume_history(hist), key="cabg_vol")
    with right:
        with ui.card("cabg-dist"):
            ui.section(f"Position among California hospitals, {in_period}",
                       "Distribution of hospital readmission RAAER in the input window; amber line = selected hospital.")
            vals = panel.loc[panel["period"] == in_period, "readmit_rate"].dropna()
            swv = statewide.loc[statewide["period"] == in_period, "rate"]
            ui.plot(viz.distribution_marker(vals, row["prev_readmit_rate"], "Selected", "Readmission RAAER (%)",
                                            float(swv.iloc[0]) if not swv.empty else None), key="cabg_dist")

    with ui.card("cabg-map"):
        ui.section(f"Statewide context: CABG hospitals, {in_period}",
                   "Marker colour = readmission RAAER; size = eligible cases. Ring = selected hospital.")
        ui.plot(viz.hospital_map(panel[panel["period"] == in_period], "readmit_rate", "RAAER", highlight_id=hid), key="cabg_map")

    with st.expander("Hospital data table (all published windows)"):
        show = hist[["period", "readmit_rate", "readmit_ci_low", "readmit_ci_high", "readmit_rating", "readmit_cases",
                     "stroke_rate", "cabg_volume", "iso_mortality_final_year", "cabg_valve_mortality_rate"]].rename(columns={
            "period": "Window", "readmit_rate": "Readmission RAAER %", "readmit_ci_low": "CI low", "readmit_ci_high": "CI high",
            "readmit_rating": "Rating", "readmit_cases": "Eligible cases", "stroke_rate": "Post-op stroke RAAER %",
            "cabg_volume": "CABG cases", "iso_mortality_final_year": "Isolated CABG mortality % (final year)",
            "cabg_valve_mortality_rate": "CABG+Valve mortality %"})
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Missing cells were not published (they are not zero).")

# ======================================================================================
with tab_scenario:
    ui.callout("<b>Hypothetical scenario - not a real hospital.</b> Enter previous-window values to see what the "
               "trained models would estimate. Results are illustrations of model behaviour, not observed or forecast "
               "outcomes for any facility.", "synth")
    ranges = schema["train_summary"]["feature_ranges"]
    syn_opts = ["(manual entry)"] + synthetic["Hospital_ID"].tolist()
    pick = st.selectbox("Optionally pre-fill from a SYNTHETIC test record", syn_opts,
                        help="Records from the synthetic test file. They are clearly artificial and never used for training.")
    if pick != "(manual entry)":
        srow = synthetic[synthetic["Hospital_ID"] == pick].iloc[0]
        defaults = {f: float(srow[f]) for f in FEATURES}
    else:
        defaults = {f: ranges[f]["median"] for f in FEATURES}
    with ui.filter_bar("scenario"):
        c = st.columns(3)
        vals = {
            "prev_readmit_rate": c[0].number_input("Previous readmission RAAER (%)", 0.0, 60.0,
                                                   round(defaults["prev_readmit_rate"], 2), 0.1, key=f"s_r_{pick}"),
            "prev_stroke_rate": c[1].number_input("Previous post-operative stroke RAAER (%)", 0.0, 20.0,
                                                  round(defaults["prev_stroke_rate"], 2), 0.1, key=f"s_s_{pick}"),
            "prev_cabg_volume": c[2].number_input("Previous CABG case volume (cases)", 1.0, 3000.0,
                                                  float(round(defaults["prev_cabg_volume"])), 10.0, key=f"s_v_{pick}"),
        }
    warnings = cabg_model.validate_inputs(schema, vals)
    for w in warnings:
        ui.callout(w, "warn")
    sc = cabg_model.predict(models, schema, vals)
    k = st.columns(3)
    for i, m in enumerate(("linear_regression", "random_forest", "persistence")):
        r = sc[sc["model"] == m].iloc[0]
        with k[i]:
            ui.kpi(f"Scenario estimate - {r['label']}", f"{r['estimate']:.2f}", "%",
                   "recommended model" if r["recommended"] else None, "flat",
                   "Hypothetical next-window hospital RAAER", MODEL_COLORS_DARK[m])
    contrib, intercept = cabg_model.explain_linear(models, schema, vals)
    with ui.card("scenario-explain"):
        ui.section("Linear Regression contributions for this scenario",
                   f"Average-hospital estimate {intercept:.2f}% plus the contributions below.")
        ui.plot(viz.contribution_chart(contrib, intercept, LABELS, height=230), key="scen_contrib")
    st.caption("Training ranges: " + "; ".join(f"{LABELS[f]} {ranges[f]['min']:.1f}-{ranges[f]['max']:.1f}" for f in FEATURES)
               + ". Values outside these ranges are extrapolations.")
ui.footer()
