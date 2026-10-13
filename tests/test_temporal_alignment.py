"""Reporting-period alignment: lags only between valid consecutive windows, never bridging gaps."""
import pandas as pd
import pytest

from src import cabg_model, stroke_analysis
from src.features import build_transitions, next_window_label


def _toy_panel():
    # Hospital A is missing the window starting 2020 (like the sparse 2020-2021 CABG window).
    return pd.DataFrame({
        "hospital_id": ["A", "A", "A", "B", "B"],
        "period": ["2018-2019", "2019-2020", "2021-2022", "2018-2019", "2019-2020"],
        "period_start": [2018, 2019, 2021, 2018, 2019],
        "rate": [10.0, 12.0, 9.0, 15.0, 14.0],
        "volume": [100, 110, 90, 50, 55],
    })


def test_one_step_pairs_never_bridge_a_missing_window():
    tr = build_transitions(_toy_panel(), id_col="hospital_id", period_col="period", start_col="period_start",
                           feature_map={"rate": "prev_rate", "volume": "prev_volume"}, target_source="rate",
                           target_name="next_rate", horizon=1)
    pairs = set(zip(tr["hospital_id"], tr["input_period"], tr["target_period"]))
    assert pairs == {("A", "2018-2019", "2019-2020"), ("B", "2018-2019", "2019-2020")}
    assert ("A", "2019-2020", "2021-2022") not in pairs


def test_two_step_design_is_explicit():
    tr = build_transitions(_toy_panel(), id_col="hospital_id", period_col="period", start_col="period_start",
                           feature_map={"rate": "prev_rate"}, target_source="rate", target_name="next_rate", horizon=2)
    assert set(zip(tr["hospital_id"], tr["input_period"], tr["target_period"])) == {("A", "2019-2020", "2021-2022")}
    assert (tr["horizon_years"] == 2).all()


def test_invalid_horizon_and_target_as_feature_rejected():
    with pytest.raises(ValueError):
        build_transitions(_toy_panel(), id_col="hospital_id", period_col="period", start_col="period_start",
                          feature_map={"rate": "prev_rate"}, target_source="rate", target_name="next_rate", horizon=0)
    with pytest.raises(ValueError):
        build_transitions(_toy_panel(), id_col="hospital_id", period_col="period", start_col="period_start",
                          feature_map={"rate": "next_rate"}, target_source="rate", target_name="next_rate", horizon=1)


def test_next_window_label():
    assert next_window_label("2022-2023") == "2023-2024"
    assert next_window_label("2019-2020", 2) == "2021-2022"


def test_cabg_transitions_are_consecutive_rolling_windows(clean):
    panel = cabg_model.build_panel(clean["cabg"])
    tr = cabg_model.build_cabg_transitions(panel, horizon=1)
    assert (tr["target_start"] - tr["input_start"] == 1).all()
    bridged = tr[(tr["input_period"] == "2019-2020") & (tr["target_period"] == "2021-2022")]
    assert bridged.empty, "a missing 2020-2021 window must not be bridged in the one-step design"


def test_cabg_split_is_forward_in_time(clean):
    panel = cabg_model.build_panel(clean["cabg"])
    tr = cabg_model.build_cabg_transitions(panel, horizon=1)
    train, test, excluded = cabg_model.split_temporal(tr)
    assert test["target_period"].unique().tolist() == ["2022-2023"]
    assert train["target_start"].max() < test["target_start"].min()
    assert len(test) == 118 and len(train) == 124
    assert (excluded["input_period"] == "2020-2021").all()  # no post-op stroke measure in 2020-2021
    assert train[cabg_model.PRIMARY_FEATURES].notna().all().all()


def test_cabg_operative_mortality_aligned_to_final_year_of_window(clean):
    panel = cabg_model.build_panel(clean["cabg"])
    cabg = clean["cabg"]
    row = panel[(panel["period"] == "2021-2022") & panel["iso_mortality_final_year"].notna()].iloc[0]
    src = cabg[(cabg["hospital_id"] == row["hospital_id"]) & (cabg["measure_code"] == "iso_mortality")
               & (cabg["period"] == "2022")]
    assert src["raaer"].iloc[0] == pytest.approx(row["iso_mortality_final_year"])
    # no single-year value exists for 2018 or 2021, so it is never presented as a two-year rate
    assert not cabg[(cabg["measure_code"] == "iso_mortality") & cabg["period"].isin(["2018", "2021"])].shape[0]


def test_stroke_transitions(clean):
    panel = stroke_analysis.build_panel(clean["stroke"])
    tr = stroke_analysis.build_stroke_transitions(panel, 1)
    assert (tr["target_start"] - tr["input_start"] == 1).all()
    assert set(tr["target_period"]) == {"2012-2013", "2013-2014", "2014-2015"}
