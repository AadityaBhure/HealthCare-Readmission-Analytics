"""Page 5 - Model evaluation, validation checks and synthetic scenario testing."""
import pandas as pd
import streamlit as st

from src import dashboard_data as dd, ui, visualizations as viz
from src.modeling import MODEL_LABELS
from src.theme import SERIES_DARK, STATUS

FEATURE_LABELS = {
    "prev_readmit_rate": "Previous readmission rate", "prev_stroke_rate": "Previous post-op stroke rate",
    "prev_cabg_volume": "Previous CABG volume", "prev_mortality_rate": "Previous mortality rate",
    "prev_stroke_volume": "Previous stroke case volume",
}
NUM = {c: st.column_config.NumberColumn(format="%.3f") for c in
       ("MAE", "RMSE", "R2", "Bias", "MAE_mean", "MAE_sd", "RMSE_mean", "R2_mean", "R2_sd", "mae_diff", "ci_low", "ci_high")}

ui.page_header(
    "Technical evaluation",
    "Model Evaluation & Testing",
    "How the models were trained and tested on real data, how they compare with simple baselines, which inputs "
    "drive them, where they fail, and which data checks were run. Synthetic scenario tests are kept in their own "
    "tab and never mixed with real-data results.")

tabs = st.tabs(["CABG model (primary)", "Stroke model (secondary)", "Data validation", "Synthetic scenario testing"])


def render_experiment(prefix: str, condition: str, volume_label: str) -> None:
    s = dd.artifact_json(f"{prefix}/summary.json")
    ts = s["train_summary"]
    metrics = dd.artifact_csv(f"{prefix}/metrics_test.csv")
    preds = dd.artifact_csv(f"{prefix}/test_predictions.csv")
    boot = dd.artifact_csv(f"{prefix}/bootstrap_mae_diff.csv")
    cvp = dd.artifact_csv(f"{prefix}/cv_pooled_summary.csv")
    cvn = dd.artifact_csv(f"{prefix}/cv_nonoverlap_summary.csv")
    paired = dd.artifact_csv(f"{prefix}/paired_cv_vs_persistence.csv")
    perm = dd.artifact_csv(f"{prefix}/permutation_importance.csv")
    native = dd.artifact_csv(f"{prefix}/native_importance.csv")
    err = dd.artifact_csv(f"{prefix}/error_by_volume.csv")
    curve = dd.artifact_csv(f"{prefix}/learned_curve.csv")
    trans = dd.artifact_csv(f"{prefix}/transitions.csv")
    sel = dd.artifact_csv(f"{prefix}/train_cv_selection.csv")
    excluded = dd.artifact_csv(f"{prefix}/excluded_transitions.csv")
    target = "next_readmit_rate"

    ui.callout(
        f"<b>Evaluation design.</b> Each row is a hospital transition: inputs from one reporting window, target = the "
        f"same hospital's readmission rate in the next window (start year + 1). "
        f"<b>Training:</b> {ts['n_train']} transitions with targets {', '.join(ts['train_target_periods'])}. "
        f"<b>Test (held out):</b> {ts['n_test']} hospitals, target {ts['test_target_periods'][0]} - never used for fitting, "
        f"tuning or model selection. The recommended model (<b>{MODEL_LABELS[s['recommended_model']]}</b>) and Random "
        f"Forest hyper-parameters were chosen by hospital-grouped 5-fold CV inside the training set. "
        f"<b>Caveat:</b> consecutive two-year windows share one calendar year of patients (lag-1 correlation "
        f"r = {s['lag_correlation']['one_step_overlapping']:.2f} vs {s['lag_correlation']['non_overlapping']:.2f} for "
        "non-overlapping windows), which flatters the persistence baseline; the non-overlapping analysis below removes this.",
        "info", None)

    k = st.columns(5)
    m = metrics.set_index("model")
    rec = s["recommended_model"]
    with k[0]:
        ui.kpi("Training transitions", f"{ts['n_train']}", "", None, "flat", f"{ts['train_hospitals']} hospitals", "#7D8BA3")
    with k[1]:
        ui.kpi("Test hospitals", f"{ts['n_test']}", "", None, "flat", f"Target window {ts['test_target_periods'][0]}", "#7D8BA3")
    with k[2]:
        ui.kpi(f"Test MAE - {MODEL_LABELS[rec]}", f"{m.loc[rec, 'MAE']:.2f}", " pp", None, "flat",
               f"R² {m.loc[rec, 'R2']:.3f} - RMSE {m.loc[rec, 'RMSE']:.2f} pp", SERIES_DARK[0])
    with k[3]:
        ui.kpi("Test MAE - persistence", f"{m.loc['persistence', 'MAE']:.2f}", " pp", None, "flat",
               f"R² {m.loc['persistence', 'R2']:.3f} - next = previous", "#A9B6CC")
    with k[4]:
        b = boot[(boot["model"] == rec) & (boot["reference"] == "persistence")]
        if not b.empty and rec != "persistence":
            b = b.iloc[0]
            verdict = ("better than" if b["ci_high"] < 0 else "worse than" if b["ci_low"] > 0 else "not distinguishable from")
            ui.kpi("Recommended vs persistence", f"{b['mae_diff']:+.2f}", " pp", f"95% CI {b['ci_low']:+.2f} to {b['ci_high']:+.2f}",
                   "flat", f"Paired bootstrap: {verdict} persistence", STATUS["warning"])

    with ui.card(f"{prefix}-table"):
        ui.section("Comparative model performance on the held-out window",
                   "MAE / RMSE in percentage points of the readmission rate. R² below zero means worse than predicting "
                   "the test-set mean. Bias = mean(predicted - actual).")
        show = metrics[["label", "MAE", "RMSE", "R2", "Bias", "n_train", "n_test", "MAE_vs_persistence_pct", "recommended"]]
        st.dataframe(show.rename(columns={"label": "Model", "n_train": "Train n", "n_test": "Test n",
                                          "MAE_vs_persistence_pct": "MAE vs persistence (%)", "recommended": "Recommended"}),
                     hide_index=True, width="stretch",
                     column_config={**NUM, "MAE vs persistence (%)": st.column_config.NumberColumn(format="%+.1f%%")})

    design_rows = []
    for _, r in metrics.iterrows():
        design_rows.append({"design": f"Temporal hold-out<br>(n={ts['n_test']})", "model": r["model"], "MAE": r["MAE"], "sd": None})
    for _, r in cvp.iterrows():
        design_rows.append({"design": f"Grouped CV, one-step<br>(n={s['n_pooled']}, 5x5 folds)", "model": r["model"],
                            "MAE": r["MAE_mean"], "sd": r["MAE_sd"]})
    for _, r in cvn.iterrows():
        design_rows.append({"design": f"Grouped CV, non-overlapping<br>(n={s['n_nonoverlap']}, 5x5 folds)", "model": r["model"],
                            "MAE": r["MAE_mean"], "sd": r["MAE_sd"]})
    left, right = st.columns([1.15, 1])
    with left:
        with ui.card(f"{prefix}-designs"):
            ui.section("Error across evaluation designs",
                       "Lower is better. Error bars = SD across 25 grouped-CV folds. In the non-overlapping design input and "
                       "target windows share no calendar year.")
            ui.plot(viz.design_comparison(pd.DataFrame(design_rows)), key=f"{prefix}_designs")
    with right:
        with ui.card(f"{prefix}-boot"):
            ui.section("Is the difference meaningful?",
                       "Paired bootstrap (2,000 resamples of test hospitals) of the MAE difference.")
            ui.plot(viz.bootstrap_forest(boot), key=f"{prefix}_boot")
            p = paired.copy()
            p["Model"] = p["model"].map({"linear_regression": "LR", "random_forest": "RF"})
            p["Design"] = p["design"].map(lambda d: "Non-overlap CV" if d.startswith("non") else "One-step CV")
            st.dataframe(p[["Design", "Model", "mean_mae_diff", "sd_mae_diff", "share_folds_model_better"]].rename(columns={
                "mean_mae_diff": "dMAE", "sd_mae_diff": "SD", "share_folds_model_better": "Folds better"}),
                hide_index=True, width="stretch",
                column_config={"dMAE": st.column_config.NumberColumn(format="%+.3f"),
                               "SD": st.column_config.NumberColumn(format="%.3f"),
                               "Folds better": st.column_config.NumberColumn(format="percent")})
            st.caption("Fold-wise paired comparison with persistence (dMAE < 0: model more accurate).")

    left, right = st.columns(2)
    with left:
        with ui.card(f"{prefix}-avp"):
            ui.section("Actual vs predicted (test window)", "Points on the dotted line are perfect predictions.")
            chosen = st.multiselect("Models", ["linear_regression", "random_forest", "persistence"],
                                    default=[rec if rec != "persistence" else "linear_regression", "persistence"],
                                    format_func=MODEL_LABELS.get, key=f"{prefix}_avp_models")
            if chosen:
                ui.plot(viz.actual_vs_predicted(preds, target, chosen), key=f"{prefix}_avp")
    with right:
        with ui.card(f"{prefix}-resid"):
            ui.section("Residuals", "Actual minus predicted. A funnel shape against cohort size shows that small hospitals are noisier.")
            rm = st.selectbox("Model", ["linear_regression", "random_forest", "persistence"], format_func=MODEL_LABELS.get,
                              key=f"{prefix}_resid_model")
            xopt = st.radio("Plot residuals against", ["Target-window cohort size", "Predicted value"], horizontal=True,
                            key=f"{prefix}_resid_x")
            if xopt.startswith("Target"):
                ui.plot(viz.residual_plot(preds, target, rm, "target_cases", f"{volume_label} (log scale)", log_x=True),
                        key=f"{prefix}_resid")
            else:
                ui.plot(viz.residual_plot(preds, target, rm, f"pred_{rm}", "Predicted rate (%)"), key=f"{prefix}_resid")

    left, right = st.columns(2)
    with left:
        with ui.card(f"{prefix}-vol"):
            ui.section("Error by hospital size", "Test MAE within thirds of the target-window cohort size.")
            ui.plot(viz.error_by_volume(err), key=f"{prefix}_vol")
    with right:
        with ui.card(f"{prefix}-perm"):
            ui.section("Feature importance (permutation, test window)",
                       "How much test MAE rises when one input is randomly shuffled (mean +/- SD over repeats). "
                       "Values near zero mean the model gains nothing from that input.")
            ui.plot(viz.permutation_bars(perm, FEATURE_LABELS), key=f"{prefix}_perm")
            nat = native.copy()
            nat["feature"] = nat["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
            nat["model"] = nat["model"].map(MODEL_LABELS)
            st.dataframe(nat.rename(columns={"model": "Model", "feature": "Feature", "value": "Value", "kind": "Meaning"}),
                         hide_index=True, width="stretch",
                         column_config={"Value": st.column_config.NumberColumn(format="%.3f")})

    with ui.card(f"{prefix}-curve"):
        ui.section("What did the models learn? Previous rate vs estimated next rate",
                   "Other inputs fixed at their training medians. Persistence is the diagonal; a learned slope below it "
                   "means extreme hospitals are expected to move back toward the average (regression to the mean).")
        train = trans[trans["target_period"].isin(ts["train_target_periods"])]
        ui.plot(viz.learned_curve(curve, train, "prev_readmit_rate", target, "Previous-window readmission rate (%)"),
                key=f"{prefix}_curve")

    with st.expander("Model selection, hyper-parameters and excluded records"):
        st.markdown(f"**Random Forest hyper-parameters** (grid search, hospital-grouped 5-fold CV on training data): "
                    f"`{s['rf_best_params']}`")
        st.markdown("**Training-set grouped CV used for model selection**")
        st.dataframe(sel, hide_index=True, width="stretch", column_config=NUM)
        if prefix == "cabg":
            fs = dd.artifact_csv("cabg/feature_set_comparison.csv")
            st.markdown("**Sensitivity: extended feature set** (adds final-year isolated CABG mortality, CABG+valve "
                        "mortality and expected readmission rate; same temporal split)")
            st.dataframe(fs[["feature_set", "label", "MAE", "RMSE", "R2"]], hide_index=True, width="stretch",
                         column_config=NUM)
        st.markdown(f"**Excluded transitions:** {len(excluded)}")
        if len(excluded):
            st.dataframe(excluded[["hospital_id", "input_period", "target_period", "reason"]], hide_index=True,
                         width="stretch")


with tabs[0]:
    integ = dd.integrity_check()
    ok = all(v["match"] for v in integ.values())
    ui.callout(("<b>Model integrity check passed.</b> The models loaded by this dashboard reproduce the stored test-set "
                "predictions exactly, so the dashboard runs the same models that were evaluated.") if ok else
               "<b>Model integrity check failed</b> - re-run <code>python run_pipeline.py</code>.", "info" if ok else "warn")
    render_experiment("cabg", "CABG", "Eligible isolated CABG cases")

with tabs[1]:
    ui.callout("The stroke data (2011-2015) do not overlap in time with the CABG data, so this model is trained and "
               "tested only on stroke data. It is not an external test of the CABG model.", "info")
    render_experiment("stroke", "Stroke", "Eligible stroke cases")

with tabs[2]:
    checks = dd.artifact_csv("validation_checks.csv")
    log = dd.artifact_csv("preprocessing_log.csv")
    audit = dd.artifact_json("dataset_audit.json")
    counts = checks["status"].value_counts()
    k = st.columns(4)
    with k[0]:
        ui.kpi("Checks passed", str(int(counts.get("pass", 0))), "", None, "flat", "Raw + processed data", STATUS["good"])
    with k[1]:
        ui.kpi("Warnings", str(int(counts.get("warn", 0))), "", None, "flat", "Known limitations, handled explicitly", STATUS["warning"])
    with k[2]:
        ui.kpi("Failures", str(int(counts.get("fail", 0))), "", None, "flat", "Pipeline stops on any failure", STATUS["critical"])
    with k[3]:
        ui.kpi("Raw records audited", f"{sum(v['rows'] for v in audit.values()):,}", "", None, "flat",
               "Across the four CSV files", "#7D8BA3")
    with ui.card("val-checks"):
        ui.section("Validation checks")
        rows = "".join(f"<tr><td>{ui.pill(r['status'])}</td><td><code>{r['dataset']}</code></td><td>{r['description']}</td>"
                       f"<td style='color:#7D8BA3'>{r['detail']}</td></tr>" for _, r in checks.iterrows())
        st.markdown("<table style='width:100%;font-size:0.82rem;border-collapse:collapse'>"
                    "<tr style='color:#A9B6CC;text-align:left'><th>Status</th><th>Dataset</th><th>Check</th><th>Detail</th></tr>"
                    + rows + "</table>", unsafe_allow_html=True)
    with ui.card("val-log"):
        ui.section("Preprocessing decisions", "Every transformation applied in memory (raw files are never modified).")
        st.dataframe(log, hide_index=True, width="stretch")
    with ui.card("val-audit"):
        ui.section("Dataset audit")
        st.dataframe(pd.DataFrame([{"Dataset": v["title"], "File": v["file"], "Rows": v["rows"], "Columns": v["columns"],
                                    "Encoding": v["encoding"], "Duplicates": v["duplicate_rows"], "Role": v["role"]}
                                   for v in audit.values()]), hide_index=True, width="stretch")

with tabs[3]:
    ui.callout("<b>SYNTHETIC DATA - NOT EVIDENCE OF REAL-WORLD PERFORMANCE.</b> The 120 synthetic CABG records are used "
               "only to check that external records map onto the model's feature schema and that inference runs "
               "end to end. Their labels come from an unknown generating process, so agreement with them says nothing "
               "about accuracy on real hospitals. They were never used for training or real-data evaluation.", "synth")
    syn = dd.artifact_json("synthetic/summary.json")
    sp = dd.artifact_csv("synthetic/scenario_predictions.csv")
    checks = [
        ("All records carry Synthetic_Record_Flag = True", syn["all_flagged_synthetic"]),
        ("Synthetic columns map onto the model feature schema (" + ", ".join(syn["schema_features"]) + ")", True),
        ("All predictions are finite", syn["all_predictions_finite"]),
        ("All predictions lie within 0-100 %", syn["predictions_in_0_100"]),
        ("Repeated inference gives identical results (deterministic)", syn["deterministic"]),
    ]
    with ui.card("syn-checks"):
        ui.section("Schema and smoke-test checks")
        for text, passed in checks:
            st.markdown(f"{ui.pill('pass' if passed else 'fail')} &nbsp; {text}", unsafe_allow_html=True)
        st.markdown(f"{ui.pill('warn')} &nbsp; {syn['records_with_any_input_outside_training_range']} record(s) have inputs "
                    f"outside the real training range (extrapolation): {syn['inputs_outside_training_range_by_feature']}",
                    unsafe_allow_html=True)
        st.caption("Unused synthetic column: Operative_Mortality_Rate_Pct (no defensibly aligned real-data feature in the primary model).")
    left, right = st.columns([1, 1])
    with left:
        with ui.card("syn-scatter"):
            ui.section("Model estimates vs synthetic labels (SYNTHETIC)")
            sm = st.selectbox("Model", ["linear_regression", "random_forest", "persistence"], format_func=MODEL_LABELS.get,
                              key="syn_model")
            ui.plot(viz.synthetic_scatter(sp, sm), key="syn_scatter")
    with right:
        with ui.card("syn-agree"):
            ui.section("Agreement with synthetic labels (consistency check only)")
            ag = pd.DataFrame(syn["agreement_with_synthetic_labels"])
            st.dataframe(ag[["label", "MAE", "RMSE", "R2", "n"]].rename(columns={"label": "Model"}), hide_index=True,
                         width="stretch", column_config=NUM)
            st.caption(syn["interpretation"])
    with st.expander("Synthetic scenario predictions (all 120 records)"):
        st.dataframe(sp, hide_index=True, width="stretch")
ui.footer()
