"""Synthetic CABG scenario testing.

The synthetic file is used ONLY to (1) validate that external records can be mapped onto the
trained model's feature schema, (2) smoke-test inference end to end, and (3) demonstrate
hypothetical scenarios. Agreement with the synthetic labels is NOT evidence of real-world
predictive performance - the labels come from an unknown generating process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluation
from .modeling import MODEL_LABELS


def run_scenarios(synthetic: pd.DataFrame, models: dict, schema: dict) -> tuple[pd.DataFrame, dict]:
    feats = schema["features"]
    missing = [f for f in feats if f not in synthetic.columns]
    if missing:
        raise KeyError(f"Synthetic data cannot be mapped to model schema; missing {missing}")
    X = synthetic[feats].astype(float)
    preds = synthetic[["Hospital_ID", "Hospital_Name", *feats, "synthetic_label_next_rate",
                       "Synthetic_Record_Flag"]].copy()
    for name, model in models.items():
        p1 = model.predict(X)
        p2 = model.predict(X)
        if not np.array_equal(p1, p2):
            raise AssertionError(f"Non-deterministic predictions from {name}")
        preds[f"pred_{name}"] = p1

    ranges = schema["train_summary"]["feature_ranges"]
    outside = pd.DataFrame({f: (X[f] < ranges[f]["min"]) | (X[f] > ranges[f]["max"]) for f in feats})
    preds["n_inputs_outside_training_range"] = outside.sum(axis=1)

    agreement = []
    for name in models:
        m = evaluation.regression_metrics(preds["synthetic_label_next_rate"], preds[f"pred_{name}"])
        agreement.append({"model": name, "label": MODEL_LABELS.get(name, name), **m})
    pred_cols = [f"pred_{m}" for m in models]
    summary = {
        "n_records": int(len(preds)),
        "all_flagged_synthetic": bool(preds["Synthetic_Record_Flag"].eq(True).all()),
        "schema_features": feats,
        "unused_synthetic_columns": ["Operative_Mortality_Rate_Pct"],
        "all_predictions_finite": bool(np.isfinite(preds[pred_cols].to_numpy()).all()),
        "predictions_in_0_100": bool(((preds[pred_cols] >= 0) & (preds[pred_cols] <= 100)).all().all()),
        "deterministic": True,
        "records_with_any_input_outside_training_range": int((preds["n_inputs_outside_training_range"] > 0).sum()),
        "inputs_outside_training_range_by_feature": {f: int(outside[f].sum()) for f in feats},
        "agreement_with_synthetic_labels": agreement,
        "interpretation": "Agreement with synthetic labels only shows the pipeline runs consistently on "
                          "schema-compatible inputs. It is not evidence of real-world accuracy.",
    }
    return preds, summary
