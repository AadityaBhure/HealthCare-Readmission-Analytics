"""Parsing of rates/counts, missing-value handling and dataset-specific fixes."""
import numpy as np
import pandas as pd
import pytest

from src import preprocessing as pp


def test_parse_number_handles_separators_percent_and_missing_tokens():
    s = pd.Series(["1,234", "13.80%", "14%", ".", "", "0", " 7 ", None, "Suppressed"])
    out = pp.parse_number(s)
    assert out.iloc[0] == 1234
    assert out.iloc[1] == pytest.approx(13.8)
    assert out.iloc[2] == 14
    assert np.isnan(out.iloc[3]) and np.isnan(out.iloc[4])
    assert out.iloc[5] == 0  # a genuine zero stays zero
    assert out.iloc[6] == 7
    assert np.isnan(out.iloc[7]) and np.isnan(out.iloc[8])


def test_normalize_id():
    out = pp.normalize_id(pd.Series(["106010937", "106010937.0", ".", "", "abc"]))
    assert out.iloc[0] == "106010937" and out.iloc[1] == "106010937"
    assert out.iloc[2:].isna().all()


def test_parse_period():
    assert pp.parse_period("2018-2019") == (2018, 2019, 2)
    assert pp.parse_period("2019") == (2019, 2019, 1)
    with pytest.raises(ValueError):
        pp.parse_period("FY19")


def test_clean_text_normalises_breaks_spaces_and_dashes():
    out = pp.clean_text(pd.Series(["San\nBernardino", "Contra  Costa", "Cedars – Sinai", "  "]))
    assert out.tolist()[:3] == ["San Bernardino", "Contra Costa", "Cedars - Sinai"]
    assert pd.isna(out.iloc[3])


def test_cabg_missing_values_stay_missing(raw, clean):
    n_dot = (raw["cabg"]["Hospital Risk-Adjusted Adverse Events Rate (%,RAAER)"].str.strip() == ".").sum()
    assert n_dot == 11
    assert clean["cabg"]["raaer"].isna().sum() == n_dot
    assert not (clean["cabg"]["raaer"].isna() & clean["cabg"]["rating"].ne("Not Applicable")).any()


def test_cabg_text_cleaning(clean):
    cabg = clean["cabg"]
    assert "San Bernardino" in set(cabg["county"]) and "San\nBernardino" not in set(cabg["county"])
    assert "Contra Costa" in set(cabg["county"])
    assert "Not Appplicable" not in set(cabg["rating"].dropna())
    # one display name per hospital ID
    assert cabg.groupby("hospital_id")["hospital_name"].nunique().max() == 1


def test_stroke_column_swap_corrected(clean):
    st = clean["stroke"]
    has = st.dropna(subset=["events", "cases"])
    assert (has["events"] <= has["cases"]).all()
    assert set(st.loc[st["counts_corrected"], "period"]) == {"2013-2014"}
    state = st[st["is_statewide"] & (st["period"] == "2013-2014") & (st["measure_code"] == "mortality")].iloc[0]
    assert state["events"] == 7364 and state["cases"] == 73982
    # rates themselves are untouched
    assert state["risk_adjusted_rate"] == pytest.approx(9.95)


def test_allcause_coding_systems_kept_separate(clean):
    ac = clean["allcause"]
    assert ac["coding_system"].notna().all()
    assert ac.loc[ac["coding_system"] == "ICD-9", "year"].max() == 2015
    assert ac.loc[ac["coding_system"] == "ICD-10", "year"].min() == 2016
    assert ac.loc[ac["coding_transition_year"], "year"].unique().tolist() == [2015]


def test_allcause_suppressed_value_not_zero(clean):
    ac = clean["allcause"]
    alpine = ac[(ac["county"] == "Alpine") & (ac["year"] == 2017) & (ac["strata"] == "Overall")].iloc[0]
    assert alpine["admits"] == 9
    assert np.isnan(alpine["readmits"]) and np.isnan(alpine["rate"])


def test_allcause_rates_parsed_in_percentage_points(clean):
    ac = clean["allcause"]
    row = ac[(ac["year"] == 2016) & (ac["county"] == "State") & (ac["strata"] == "Overall")].iloc[0]
    assert row["rate"] == pytest.approx(14.48)
    assert row["admits"] == 1877198
    assert abs(row["rate_reported"] - row["rate_computed"]) < 0.01
