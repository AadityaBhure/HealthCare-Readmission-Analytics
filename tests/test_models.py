"""Model training/prediction, evaluation utilities and train/inference schema consistency."""
import json

import numpy as np
import pandas as pd
import pytest

from src import cabg_model, evaluation, model_store
from src.modeling import PersistenceRegressor, linear_contributions, make_models, run_temporal_experiment


def test_persistence_returns_previous_rate():
    X = pd.DataFrame({"prev_readmit_rate": [10.0, 12.5], "prev_stroke_rate": [1.0, 2.0]})
    p = PersistenceRegressor("prev_readmit_rate").fit(X).predict(X)
    np.testing.assert_array_equal(p, [10.0, 12.5])


def test_metrics_and_bootstrap_basics():
    y = np.array([10.0, 12.0, 14.0])
    m = evaluation.regression_metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1
    b = evaluation.paired_bootstrap_mae_diff(y, y, y + 1, n_boot=200)
    assert b["mae_diff"] == pytest.approx(-1.0) and b["ci_high"] < 0


def test_fast_cabg_experiment_runs_and_reports_all_models(clean):
    panel = cabg_model.build_panel(clean["cabg"])
    train, test, _ = cabg_model.split_temporal(cabg_model.build_cabg_transitions(panel))
    res = run_temporal_experiment(train, test, cabg_model.experiment_config(), fast=True)
    assert set(res.test_metrics["model"]) == {"mean_baseline", "persistence", "linear_regression", "random_forest"}
    assert np.isfinite(res.test_metrics[["MAE", "RMSE", "R2"]].to_numpy()).all()
    assert (res.test_metrics["n_test"] == 118).all()
    assert res.recommended_model in res.models
    # persistence predictions are literally the previous rates
    np.testing.assert_array_equal(res.test_predictions["pred_persistence"], test["prev_readmit_rate"])


def test_linear_contributions_sum_to_prediction(clean):
    panel = cabg_model.build_panel(clean["cabg"])
    train, test, _ = cabg_model.split_temporal(cabg_model.build_cabg_transitions(panel))
    lr = make_models(cabg_model.PRIMARY_FEATURES, cabg_model.LOG_FEATURES, "prev_readmit_rate")["linear_regression"]
    lr.fit(train[cabg_model.PRIMARY_FEATURES], train[cabg_model.TARGET])
    x = test[cabg_model.PRIMARY_FEATURES].iloc[[0]]
    contrib, intercept = linear_contributions(lr, x)
    assert intercept + contrib["contribution"].sum() == pytest.approx(lr.predict(x)[0])


def test_saved_schema_matches_training_and_dashboard_inference(fast_run):
    root = fast_run["root"]
    assert fast_run["rc"] == 0
    models, schema = cabg_model.load_bundle(root / "models")
    assert schema["features"] == cabg_model.PRIMARY_FEATURES
    preds = pd.read_csv(root / "artifacts" / "cabg" / "hospital_predictions.csv", dtype={"hospital_id": str})
    ok = preds.dropna(subset=["pred_linear_regression"]).sample(20, random_state=1)
    for _, r in ok.iterrows():
        live = cabg_model.predict(models, schema, {f: r[f] for f in schema["features"]})
        got = live.set_index("model")["estimate"]
        assert got["linear_regression"] == pytest.approx(r["pred_linear_regression"])
        assert got["random_forest"] == pytest.approx(r["pred_random_forest"])
    test_preds = pd.read_csv(root / "artifacts" / "cabg" / "test_predictions.csv", dtype={"hospital_id": str})
    assert all(v["match"] for v in model_store.verify_bundle(models, schema, test_preds).values())


def test_inference_rejects_missing_features_and_flags_extrapolation(fast_run):
    models, schema = cabg_model.load_bundle(fast_run["root"] / "models")
    with pytest.raises(KeyError):
        cabg_model.predict(models, schema, {"prev_readmit_rate": 10})
    warnings = cabg_model.validate_inputs(schema, {"prev_readmit_rate": 95.0, "prev_stroke_rate": 1.0,
                                                   "prev_cabg_volume": 150})
    assert any("outside the training range" in w for w in warnings)


def test_model_integrity_file(fast_run):
    integ = json.loads((fast_run["root"] / "artifacts" / "model_integrity.json").read_text())
    assert all(v["match"] for exp in integ.values() for v in exp.values())
