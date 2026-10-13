"""Feature/target separation and target-leakage guards."""
import numpy as np
import pytest

from src import cabg_model, stroke_analysis
from src.modeling import make_models


@pytest.mark.parametrize("module", [cabg_model, stroke_analysis])
def test_features_exclude_target_ids_and_future_fields(module):
    feats = module.PRIMARY_FEATURES if module is cabg_model else module.FEATURES
    assert module.TARGET not in feats
    assert all(f.startswith("prev_") for f in feats)
    assert not any(k in f for f in feats for k in ("hospital_id", "name", "county", "next", "target"))


def test_feature_values_come_from_input_window_and_target_from_next(clean):
    panel = cabg_model.build_panel(clean["cabg"])
    tr = cabg_model.build_cabg_transitions(panel, horizon=1)
    idx = panel.set_index(["hospital_id", "period"])
    for _, r in tr.sample(25, random_state=0).iterrows():
        assert r["prev_readmit_rate"] == idx.loc[(r["hospital_id"], r["input_period"]), "readmit_rate"]
        assert r["next_readmit_rate"] == idx.loc[(r["hospital_id"], r["target_period"]), "readmit_rate"]


def test_learned_preprocessing_is_fitted_on_training_rows_only(clean):
    panel = cabg_model.build_panel(clean["cabg"])
    train, test, _ = cabg_model.split_temporal(cabg_model.build_cabg_transitions(panel))
    lr = make_models(cabg_model.PRIMARY_FEATURES, cabg_model.LOG_FEATURES, "prev_readmit_rate")["linear_regression"]
    lr.fit(train[cabg_model.PRIMARY_FEATURES], train[cabg_model.TARGET])
    scaler = lr.named_steps["scale"]
    transformed = lr.named_steps["prep"].transform(train[cabg_model.PRIMARY_FEATURES])
    np.testing.assert_allclose(scaler.mean_, transformed.mean(axis=0))
    both = np.vstack([transformed, lr.named_steps["prep"].transform(test[cabg_model.PRIMARY_FEATURES])])
    assert not np.allclose(scaler.mean_, both.mean(axis=0))


def test_no_hospital_target_window_in_both_train_and_test(clean):
    panel = cabg_model.build_panel(clean["cabg"])
    train, test, _ = cabg_model.split_temporal(cabg_model.build_cabg_transitions(panel))
    tr_keys = set(zip(train["hospital_id"], train["target_period"]))
    te_keys = set(zip(test["hospital_id"], test["target_period"]))
    assert not tr_keys & te_keys
    assert not set(train["target_period"]) & set(test["target_period"])


def test_synthetic_records_never_enter_real_data(clean):
    syn_ids = set(clean["synthetic"]["Hospital_ID"])
    panel = cabg_model.build_panel(clean["cabg"])
    assert not syn_ids & set(panel["hospital_id"])
    assert not syn_ids & set(clean["stroke"]["hospital_id"].dropna())
