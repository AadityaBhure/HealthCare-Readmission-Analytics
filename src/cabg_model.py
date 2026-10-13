"""Primary experiment: estimate a hospital's next-period risk-adjusted isolated CABG 30-day
readmission rate (RAAER, %) from its previous reporting window.

Key data facts that shape the design (verified in data_validation):
* readmission and post-operative stroke are published for overlapping two-year rolling windows
  (2018-2019, 2019-2020, 2020-2021, 2021-2022, 2022-2023);
* 2020-2021 readmission covers only 2 hospitals and has no stroke measure;
* isolated CABG operative mortality is published for single years (2019, 2020, 2022, 2023).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, evaluation
from .features import build_transitions, next_window_label
from .model_store import load_model_bundle, save_model_bundle
from .modeling import (MODEL_LABELS, ExperimentConfig, linear_contributions, make_models,
                       run_temporal_experiment)

TARGET = "next_readmit_rate"
PRIMARY_FEATURES = ["prev_readmit_rate", "prev_stroke_rate", "prev_cabg_volume"]
EXTENDED_FEATURES = PRIMARY_FEATURES + [
    "prev_iso_mortality_final_year", "prev_cabg_valve_mortality_rate", "prev_expected_readmit_rate"]
LOG_FEATURES = ["prev_cabg_volume"]

FEATURE_INFO = {
    "prev_readmit_rate": ("Previous-window isolated CABG 30-day readmission RAAER", "%"),
    "prev_stroke_rate": ("Previous-window isolated CABG post-operative stroke RAAER", "%"),
    "prev_cabg_volume": ("Previous-window total hospital CABG cases (log-transformed inside the model)", "cases"),
    "prev_iso_mortality_final_year": ("Isolated CABG operative mortality RAAER in the final calendar year "
                                      "of the input window (single-year measure)", "%"),
    "prev_cabg_valve_mortality_rate": ("Previous-window CABG+Valve operative mortality RAAER "
                                       "(different surgical cohort)", "%"),
    "prev_expected_readmit_rate": ("Previous-window expected readmission rate (case-mix severity)", "%"),
}
TARGET_INFO = ("Next rolling window isolated CABG 30-day readmission RAAER "
               "(hospital-level risk-adjusted rate, not an individual patient probability)", "%")

PANEL_FEATURE_MAP = {
    "readmit_rate": "prev_readmit_rate",
    "stroke_rate": "prev_stroke_rate",
    "cabg_volume": "prev_cabg_volume",
    "iso_mortality_final_year": "prev_iso_mortality_final_year",
    "cabg_valve_mortality_rate": "prev_cabg_valve_mortality_rate",
    "readmit_expected_rate": "prev_expected_readmit_rate",
}


# --------------------------------------------------------------------------------------
# Panel & transitions
# --------------------------------------------------------------------------------------
def statewide_series(cabg: pd.DataFrame) -> pd.DataFrame:
    """Statewide observed rate per measure and period (as published in every hospital row)."""
    s = (cabg.groupby(["measure_code", "measure", "period"], as_index=False)
         .agg(period_start=("period_start", "first"), period_years=("period_years", "first"),
              rate=("statewide_rate", "first"), events=("statewide_events", "first"),
              cases=("statewide_measure_cases", "first"), hospitals=("hospital_id", "nunique")))
    return s.sort_values(["measure_code", "period_start", "period_years"]).reset_index(drop=True)


def build_panel(cabg: pd.DataFrame) -> pd.DataFrame:
    """One row per hospital x two-year window that has a readmission record."""
    two = cabg[cabg["period_years"] == 2]
    keys = ["hospital_id", "period", "period_start", "period_end"]

    def pick(code: str, cols: dict[str, str]) -> pd.DataFrame:
        sub = two[two["measure_code"] == code][keys + list(cols)]
        return sub.rename(columns=cols)

    readm = pick("readmission", {
        "raaer": "readmit_rate", "raaer_ci_low": "readmit_ci_low", "raaer_ci_high": "readmit_ci_high",
        "observed_rate": "readmit_observed_rate", "expected_rate": "readmit_expected_rate",
        "hospital_measure_cases": "readmit_cases", "hospital_events": "readmit_events",
        "rating": "readmit_rating", "hospital_cabg_cases": "cabg_volume",
        "statewide_rate": "statewide_readmit_rate",
    })
    stroke = pick("postop_stroke", {"raaer": "stroke_rate", "hospital_measure_cases": "stroke_cases",
                                    "rating": "stroke_rating"})
    valve = pick("cabg_valve_mortality", {"raaer": "cabg_valve_mortality_rate"})
    single = cabg[(cabg["period_years"] == 1) & (cabg["measure_code"] == "iso_mortality")]
    single = single[["hospital_id", "period_end", "raaer", "rating"]].rename(
        columns={"raaer": "iso_mortality_final_year", "rating": "iso_mortality_rating"})

    panel = (readm.merge(stroke, on=keys, how="left", validate="one_to_one")
             .merge(valve, on=keys, how="left", validate="one_to_one")
             .merge(single, on=["hospital_id", "period_end"], how="left", validate="one_to_one"))
    meta = (cabg.sort_values("period_end").groupby("hospital_id")
            .agg(hospital_name=("hospital_name", "last"), county=("county_latest", "last"),
                 latitude=("latitude", "last"), longitude=("longitude", "last")))
    panel = panel.merge(meta, left_on="hospital_id", right_index=True, how="left")
    return panel.sort_values(["hospital_id", "period_start"]).reset_index(drop=True)


def build_cabg_transitions(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    design = ("one-step rolling window (overlapping 1 year)" if horizon == 1
              else "non-overlapping two-year horizon")
    tr = build_transitions(
        panel, id_col="hospital_id", period_col="period", start_col="period_start",
        feature_map=PANEL_FEATURE_MAP, target_source="readmit_rate", target_name=TARGET,
        horizon=horizon, design=design,
        carry_target_cols={"readmit_cases": "target_cases", "statewide_readmit_rate": "target_statewide_rate"})
    tr["primary_features_complete"] = tr[PRIMARY_FEATURES].notna().all(axis=1)
    return tr


def split_temporal(transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train = transitions whose target window ends before the latest one; test = latest target window.
    Returns (train, test, excluded) where excluded lists rows dropped and why."""
    usable = transitions["primary_features_complete"]
    excluded = transitions[~usable].assign(
        reason="missing primary feature(s) in input window (no post-operative stroke measure published)")
    t = transitions[usable]
    test_start = int(t["target_start"].max())
    test = t[t["target_start"] == test_start].reset_index(drop=True)
    train = t[t["target_start"] < test_start].reset_index(drop=True)
    return train, test, excluded.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------
def experiment_config(features: list[str] | None = None, name: str = "cabg") -> ExperimentConfig:
    return ExperimentConfig(name=name, features=features or PRIMARY_FEATURES, target=TARGET,
                            persistence_feature="prev_readmit_rate", log_features=LOG_FEATURES,
                            volume_col="target_cases")


def run(cabg: pd.DataFrame, fast: bool = False) -> dict:
    panel = build_panel(cabg)
    trans1 = build_cabg_transitions(panel, horizon=1)
    train, test, excluded = split_temporal(trans1)
    cfg = experiment_config()
    result = run_temporal_experiment(train, test, cfg, fast=fast)

    # --- Secondary: repeated hospital-grouped CV over all valid one-step transitions -----------
    pooled = trans1[trans1["primary_features_complete"]].reset_index(drop=True)
    rf_params = {k.replace("model__", ""): v for k, v in result.rf_best_params.items()}
    models = make_models(PRIMARY_FEATURES, LOG_FEATURES, "prev_readmit_rate", rf_params)
    if fast:
        models["random_forest"].set_params(model__n_estimators=100)
    reps = 1 if fast else 5
    cv_rows = evaluation.grouped_cv(models, pooled[PRIMARY_FEATURES], pooled[TARGET], pooled["hospital_id"],
                                    n_splits=5, n_repeats=reps)
    cv_summary = evaluation.summarize_cv(cv_rows, list(MODEL_LABELS))

    # --- Sensitivity A: non-overlapping horizon (input and target share no calendar year) -------
    trans2 = build_cabg_transitions(panel, horizon=2)
    nonov = trans2[trans2["primary_features_complete"]].reset_index(drop=True)
    nonov_rows = evaluation.grouped_cv(models, nonov[PRIMARY_FEATURES], nonov[TARGET], nonov["hospital_id"],
                                       n_splits=5, n_repeats=reps)
    nonov_summary = evaluation.summarize_cv(nonov_rows, list(MODEL_LABELS))

    paired = pd.DataFrame(
        [{"design": "one-step (pooled grouped CV)", **evaluation.paired_cv_difference(cv_rows, m, "persistence")}
         for m in ("linear_regression", "random_forest")]
        + [{"design": "non-overlapping horizon (grouped CV)",
            **evaluation.paired_cv_difference(nonov_rows, m, "persistence")}
           for m in ("linear_regression", "random_forest")])

    # --- Sensitivity B: extended feature set on the same temporal split -------------------------
    ext_cfg = experiment_config(EXTENDED_FEATURES, name="cabg_extended")
    ext_cfg.tune_rf = False
    ext = run_temporal_experiment(train, test, ext_cfg, fast=True if fast else False)
    ext_metrics = ext.test_metrics.assign(feature_set="extended (6 features)")
    prim_metrics = result.test_metrics.assign(feature_set="primary (3 features)")
    feature_set_comparison = pd.concat([prim_metrics, ext_metrics], ignore_index=True)

    # --- Correlations that explain the persistence baseline -------------------------------------
    lag_corr = {
        "one_step_overlapping": float(pooled[["prev_readmit_rate", TARGET]].corr().iloc[0, 1]),
        "non_overlapping": float(nonov[["prev_readmit_rate", TARGET]].corr().iloc[0, 1]),
    }

    hospital_preds = hospital_prediction_table(panel, result.models, train, test)
    return {
        "panel": panel, "transitions": trans1, "transitions_nonoverlap": trans2,
        "train": train, "test": test, "excluded": excluded, "result": result,
        "cv_rows": cv_rows, "cv_summary": cv_summary,
        "nonoverlap_rows": nonov_rows, "nonoverlap_summary": nonov_summary, "n_nonoverlap": len(nonov),
        "paired_cv": paired, "n_pooled": len(pooled),
        "feature_set_comparison": feature_set_comparison, "lag_correlation": lag_corr,
        "hospital_predictions": hospital_preds,
    }


# --------------------------------------------------------------------------------------
# Hospital-level prediction table (used by the dashboard and the report)
# --------------------------------------------------------------------------------------
def hospital_prediction_table(panel: pd.DataFrame, models: dict, train: pd.DataFrame,
                              test: pd.DataFrame) -> pd.DataFrame:
    rows = panel.rename(columns=PANEL_FEATURE_MAP).copy()
    rows["input_period"] = rows["period"]
    rows["target_period"] = rows["period"].map(next_window_label)
    rows["inputs_complete"] = rows[PRIMARY_FEATURES].notna().all(axis=1)
    actual = panel.set_index(["hospital_id", "period"])["readmit_rate"]
    rows["actual_next"] = [actual.get((h, t), np.nan) for h, t in zip(rows["hospital_id"], rows["target_period"])]
    train_keys = set(zip(train["hospital_id"], train["input_period"]))
    test_keys = set(zip(test["hospital_id"], test["input_period"]))
    published_targets = set(panel["period"])

    def status(r) -> str:
        key = (r["hospital_id"], r["input_period"])
        if not r["inputs_complete"]:
            return "No estimate - input window lacks required measures"
        if key in train_keys:
            return "Training transition (in-sample)"
        if key in test_keys:
            return "Test transition (out-of-sample back-test)"
        if r["target_period"] not in published_targets:
            return "Forward estimate - target window not yet published"
        return "No published outcome for this hospital in the target window (coverage gap)"

    rows["status"] = rows.apply(status, axis=1)
    ok = rows["inputs_complete"]
    for name, model in models.items():
        rows[f"pred_{name}"] = np.nan
        rows.loc[ok, f"pred_{name}"] = model.predict(rows.loc[ok, PRIMARY_FEATURES])
    keep = ["hospital_id", "hospital_name", "county", "input_period", "target_period", *PRIMARY_FEATURES,
            "readmit_ci_low", "readmit_ci_high", "readmit_rating", "readmit_cases", "statewide_readmit_rate",
            "actual_next", "status", *[f"pred_{m}" for m in models]]
    return rows[keep].sort_values(["hospital_name", "input_period"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Persistence of artifacts & inference
# --------------------------------------------------------------------------------------
def save_bundle(out: dict, models_dir: Path, data_hashes: dict[str, str]) -> dict:
    return save_model_bundle("cabg", out["result"], FEATURE_INFO, TARGET_INFO, models_dir, data_hashes,
                             experiment="cabg_next_period_readmission")


def load_bundle(models_dir: Path | None = None) -> tuple[dict, dict]:
    return load_model_bundle("cabg", models_dir or config.MODELS_DIR)


def validate_inputs(schema: dict, values: dict) -> list[str]:
    """Return human-readable warnings for missing/out-of-range scenario inputs."""
    warnings = []
    for f in schema["features"]:
        v = values.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            warnings.append(f"Missing required input '{f}'.")
            continue
        r = schema["train_summary"]["feature_ranges"][f]
        if v < r["min"] or v > r["max"]:
            warnings.append(f"{f} = {v:g} lies outside the training range "
                            f"[{r['min']:g}, {r['max']:g}]; the estimate is an extrapolation.")
    return warnings


def predict(models: dict, schema: dict, values: dict) -> pd.DataFrame:
    """Predict with every saved model for one feature vector, enforcing the saved schema order."""
    missing = [f for f in schema["features"] if f not in values]
    if missing:
        raise KeyError(f"Missing features for inference: {missing}")
    x = pd.DataFrame([{f: float(values[f]) for f in schema["features"]}], columns=schema["features"])
    rows = []
    for name, model in models.items():
        rows.append({"model": name, "label": MODEL_LABELS.get(name, name),
                     "estimate": float(model.predict(x)[0]),
                     "recommended": name == schema["recommended_model"]})
    return pd.DataFrame(rows)


def explain_linear(models: dict, schema: dict, values: dict) -> tuple[pd.DataFrame, float]:
    x = pd.DataFrame([{f: float(values[f]) for f in schema["features"]}], columns=schema["features"])
    return linear_contributions(models["linear_regression"], x)
