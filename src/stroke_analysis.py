"""Ischemic stroke 30-day outcomes: descriptive analytics and a separate temporal model.

The stroke data cover 2011-2012 ... 2014-2015 (overlapping two-year windows) and do not
overlap in time with the CABG data (2018-2023), so it is analysed on its own and is never
used as an external test set for the CABG model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation
from .features import build_transitions
from .model_store import save_model_bundle
from .modeling import MODEL_LABELS, ExperimentConfig, make_models, run_temporal_experiment

TARGET = "next_readmit_rate"
FEATURES = ["prev_readmit_rate", "prev_mortality_rate", "prev_stroke_volume"]
LOG_FEATURES = ["prev_stroke_volume"]
FEATURE_INFO = {
    "prev_readmit_rate": ("Previous-window ischemic stroke 30-day readmission risk-adjusted rate", "%"),
    "prev_mortality_rate": ("Previous-window ischemic stroke 30-day mortality risk-adjusted rate", "%"),
    "prev_stroke_volume": ("Previous-window ischemic stroke cases eligible for the readmission measure "
                           "(log-transformed inside the model)", "cases"),
}
TARGET_INFO = ("Next rolling window ischemic stroke 30-day readmission risk-adjusted rate "
               "(hospital-level, not an individual patient probability)", "%")
PANEL_FEATURE_MAP = {"readmit_rate": "prev_readmit_rate", "mortality_rate": "prev_mortality_rate",
                     "readmit_cases": "prev_stroke_volume"}


def statewide_series(stroke: pd.DataFrame) -> pd.DataFrame:
    s = stroke[stroke["is_statewide"]][["period", "period_start", "measure", "measure_code",
                                        "risk_adjusted_rate", "events", "cases"]]
    return s.sort_values(["measure_code", "period_start"]).reset_index(drop=True)


def build_panel(stroke: pd.DataFrame) -> pd.DataFrame:
    hosp = stroke[~stroke["is_statewide"]]
    keys = ["hospital_id", "period", "period_start", "period_end"]

    def pick(code, prefix):
        sub = hosp[hosp["measure_code"] == code][keys + ["risk_adjusted_rate", "events", "cases", "rating"]]
        return sub.rename(columns={"risk_adjusted_rate": f"{prefix}_rate", "events": f"{prefix}_events",
                                   "cases": f"{prefix}_cases", "rating": f"{prefix}_rating"})

    panel = pick("readmission", "readmit").merge(pick("mortality", "mortality"), on=keys, how="outer",
                                                 validate="one_to_one")
    meta = (hosp.sort_values("period_end").groupby("hospital_id")
            .agg(hospital_name=("hospital_name", "last"), county=("county_latest", "last"),
                 latitude=("latitude", "last"), longitude=("longitude", "last")))
    panel = panel.merge(meta, left_on="hospital_id", right_index=True, how="left")
    return panel.sort_values(["hospital_id", "period_start"]).reset_index(drop=True)


def build_stroke_transitions(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    design = ("one-step rolling window (overlapping 1 year)" if horizon == 1
              else "non-overlapping two-year horizon")
    tr = build_transitions(panel, id_col="hospital_id", period_col="period", start_col="period_start",
                           feature_map=PANEL_FEATURE_MAP, target_source="readmit_rate", target_name=TARGET,
                           horizon=horizon, design=design, carry_target_cols={"readmit_cases": "target_cases"})
    return tr


def county_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Case-weighted mean of hospital risk-adjusted rates per county/period (descriptive only)."""
    rows = []
    for (county, period), g in panel.dropna(subset=["county"]).groupby(["county", "period"]):
        for m in ("readmit", "mortality"):
            sub = g.dropna(subset=[f"{m}_rate", f"{m}_cases"])
            if sub.empty or sub[f"{m}_cases"].sum() == 0:
                continue
            rows.append({"county": county, "period": period,
                         "measure": "30-day Readmission" if m == "readmit" else "30-day Mortality",
                         "weighted_rate": float(np.average(sub[f"{m}_rate"], weights=sub[f"{m}_cases"])),
                         "hospitals": int(len(sub)), "cases": float(sub[f"{m}_cases"].sum())})
    return pd.DataFrame(rows)


def descriptive_summary(stroke: pd.DataFrame, panel: pd.DataFrame) -> dict:
    out = {"periods": sorted(panel["period"].unique().tolist()),
           "hospitals": int(panel["hospital_id"].nunique()),
           "counties": int(panel["county"].nunique())}
    corr = {}
    for p, g in panel.groupby("period"):
        gg = g.dropna(subset=["readmit_rate", "mortality_rate"])
        corr[p] = {"pearson_r": float(gg["readmit_rate"].corr(gg["mortality_rate"])), "n": int(len(gg))}
    out["readmission_vs_mortality_correlation"] = corr
    ratings = []
    for m in ("readmit", "mortality"):
        rc = panel.groupby(["period", f"{m}_rating"]).size().unstack(fill_value=0)
        for p, row in rc.iterrows():
            ratings.append({"period": p, "measure": m, **{k: int(v) for k, v in row.items()}})
    out["rating_counts"] = ratings
    dist = (panel.groupby("period")[["readmit_rate", "mortality_rate"]]
            .describe(percentiles=[.25, .5, .75]).round(2))
    dist.columns = [f"{a}_{b}" for a, b in dist.columns]
    out["distribution"] = dist.reset_index().to_dict(orient="records")
    return out


def run(stroke: pd.DataFrame, fast: bool = False) -> dict:
    panel = build_panel(stroke)
    trans = build_stroke_transitions(panel, 1)
    usable = trans[trans["features_complete"]]
    excluded = trans[~trans["features_complete"]].assign(reason="missing previous-window rate or volume")
    test_start = int(usable["target_start"].max())
    test = usable[usable["target_start"] == test_start].reset_index(drop=True)
    train = usable[usable["target_start"] < test_start].reset_index(drop=True)
    cfg = ExperimentConfig(name="stroke", features=FEATURES, target=TARGET,
                           persistence_feature="prev_readmit_rate", log_features=LOG_FEATURES,
                           volume_col="target_cases")
    result = run_temporal_experiment(train, test, cfg, fast=fast)

    rf_params = {k.replace("model__", ""): v for k, v in result.rf_best_params.items()}
    models = make_models(FEATURES, LOG_FEATURES, "prev_readmit_rate", rf_params)
    if fast:
        models["random_forest"].set_params(model__n_estimators=100)
    reps = 1 if fast else 5
    pooled = usable.reset_index(drop=True)
    cv_rows = evaluation.grouped_cv(models, pooled[FEATURES], pooled[TARGET], pooled["hospital_id"], 5, reps)
    trans2 = build_stroke_transitions(panel, 2)
    nonov = trans2[trans2["features_complete"]].reset_index(drop=True)
    nonov_rows = evaluation.grouped_cv(models, nonov[FEATURES], nonov[TARGET], nonov["hospital_id"], 5, reps)
    paired = pd.DataFrame(
        [{"design": "one-step (pooled grouped CV)", **evaluation.paired_cv_difference(cv_rows, m, "persistence")}
         for m in ("linear_regression", "random_forest")]
        + [{"design": "non-overlapping horizon (grouped CV)",
            **evaluation.paired_cv_difference(nonov_rows, m, "persistence")}
           for m in ("linear_regression", "random_forest")])
    lag_corr = {"one_step_overlapping": float(pooled[["prev_readmit_rate", TARGET]].corr().iloc[0, 1]),
                "non_overlapping": float(nonov[["prev_readmit_rate", TARGET]].corr().iloc[0, 1])}
    return {
        "panel": panel, "statewide": statewide_series(stroke), "transitions": trans, "train": train,
        "test": test, "excluded": excluded.reset_index(drop=True), "result": result,
        "cv_rows": cv_rows, "cv_summary": evaluation.summarize_cv(cv_rows, list(MODEL_LABELS)),
        "nonoverlap_rows": nonov_rows,
        "nonoverlap_summary": evaluation.summarize_cv(nonov_rows, list(MODEL_LABELS)),
        "n_nonoverlap": len(nonov), "paired_cv": paired, "n_pooled": len(pooled), "lag_correlation": lag_corr,
        "county_summary": county_summary(panel), "descriptive": descriptive_summary(stroke, panel),
    }


def save_bundle(out: dict, models_dir: Path, data_hashes: dict[str, str]) -> dict:
    return save_model_bundle("stroke", out["result"], FEATURE_INFO, TARGET_INFO, models_dir, data_hashes,
                             experiment="stroke_next_period_readmission")
