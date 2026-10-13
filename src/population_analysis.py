"""All-cause unplanned 30-day readmissions: statewide trends, county comparisons and
population strata (age, sex, race/ethnicity, expected payer).

Important properties of the source (checked in data_validation):
* ICD-9 era 2011-2015 and ICD-10 era 2016-2024 are kept apart via ``coding_system``;
* 2015 is the final ICD-9 year and appears to be a partial year;
* demographic / payer strata are published at state level only; counties only for 'Overall'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def statewide_overall(ac: pd.DataFrame) -> pd.DataFrame:
    s = ac[(ac["strata"] == "Overall") & (ac["geography_level"] == "State")]
    return s.sort_values("year").reset_index(drop=True)


def county_overall(ac: pd.DataFrame) -> pd.DataFrame:
    s = ac[(ac["strata"] == "Overall") & (ac["geography_level"] == "County")].copy()
    s["meets_min_admits"] = s["admits"] >= config.MIN_COUNTY_ADMITS_DEFAULT
    return s.sort_values(["county", "year"]).reset_index(drop=True)


def strata_statewide(ac: pd.DataFrame) -> pd.DataFrame:
    s = ac[(ac["strata"] != "Overall") & (ac["geography_level"] == "State")]
    return s.sort_values(["strata", "strata_name", "year"]).reset_index(drop=True)


def era_summary(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for era, g in state.groupby("coding_system"):
        full = g[~g["coding_transition_year"]]
        rows.append({"coding_system": era, "years": f"{g['year'].min()}-{g['year'].max()}",
                     "mean_rate_full_years": float(full["rate"].mean()),
                     "min_rate": float(g["rate"].min()), "max_rate": float(g["rate"].max()),
                     "mean_annual_admits": float(full["admits"].mean())})
    return pd.DataFrame(rows)


def summarize(ac: pd.DataFrame) -> dict:
    state = statewide_overall(ac)
    county = county_overall(ac)
    strata = strata_statewide(ac)
    latest = int(state["year"].max())
    cur = state[state["year"] == latest].iloc[0]
    prev = state[state["year"] == latest - 1].iloc[0]
    icd10 = state[state["coding_system"] == "ICD-10"]
    first10 = icd10.iloc[0]
    out = {
        "latest_year": latest,
        "latest_rate": float(cur["rate"]),
        "latest_admits": float(cur["admits"]),
        "latest_readmits": float(cur["readmits"]),
        "latest_coding_system": cur["coding_system"],
        "previous_year_rate": float(prev["rate"]),
        "yoy_change_pp": float(cur["rate"] - prev["rate"]) if cur["coding_system"] == prev["coding_system"] else None,
        "icd10_first_year": int(first10["year"]), "icd10_first_rate": float(first10["rate"]),
        "icd10_change_pp": float(cur["rate"] - first10["rate"]),
        "icd10_min_year": int(icd10.loc[icd10["rate"].idxmin(), "year"]),
        "icd10_min_rate": float(icd10["rate"].min()),
        "icd9_years": [int(y) for y in state.loc[state["coding_system"] == "ICD-9", "year"]],
        "icd10_years": [int(y) for y in icd10["year"]],
        "transition_year": int(state.loc[state["coding_transition_year"], "year"].iloc[0]),
        "n_counties": int(county["county"].nunique()),
        "min_county_admits": config.MIN_COUNTY_ADMITS_DEFAULT,
    }
    lc = county[(county["year"] == latest) & county["meets_min_admits"]].dropna(subset=["rate"])
    out["counties_ranked_latest"] = int(len(lc))
    out["counties_below_threshold_latest"] = int(
        ((county["year"] == latest) & ~county["meets_min_admits"]).sum())
    hi, lo = lc.loc[lc["rate"].idxmax()], lc.loc[lc["rate"].idxmin()]
    out["highest_county"] = {"county": hi["county"], "rate": float(hi["rate"]), "admits": float(hi["admits"])}
    out["lowest_county"] = {"county": lo["county"], "rate": float(lo["rate"]), "admits": float(lo["admits"])}
    out["county_rate_iqr_latest"] = [float(lc["rate"].quantile(.25)), float(lc["rate"].quantile(.75))]
    strata_latest = {}
    for cat, g in strata[strata["year"] == latest].groupby("strata"):
        g = g.dropna(subset=["rate"]).sort_values("rate", ascending=False)
        strata_latest[cat] = [{"name": r["strata_name"], "rate": float(r["rate"]), "admits": float(r["admits"])}
                              for _, r in g.iterrows()]
    out["strata_latest"] = strata_latest
    covid = state[state["year"].isin([2019, 2020])].set_index("year")
    if {2019, 2020} <= set(covid.index):
        out["admits_change_2019_2020_pct"] = float((covid.loc[2020, "admits"] / covid.loc[2019, "admits"] - 1) * 100)
    out["era_summary"] = era_summary(state).to_dict(orient="records")
    return out
