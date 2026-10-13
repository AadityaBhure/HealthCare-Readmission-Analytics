"""Temporal alignment: turn a hospital x period panel into (input period -> target period) pairs.

A *transition* pairs the features observed in an input reporting window with the outcome
observed in a target window whose start year is exactly ``input_start + horizon``.

* ``horizon = 1`` -> the next published rolling window (e.g. 2018-2019 -> 2019-2020).
  For rolling two-year windows these share one calendar year of patients.
* ``horizon = 2`` -> the next *non-overlapping* window (e.g. 2019-2020 -> 2021-2022).
  Used only as an explicitly labelled sensitivity design, never mixed with horizon 1.

Because the match requires an exact start-year difference, a missing period is never
bridged: if 2020-2021 is absent for a hospital, no horizon-1 pair 2019-2020 -> 2021-2022
can be produced.
"""
from __future__ import annotations

import pandas as pd


def build_transitions(panel: pd.DataFrame, *, id_col: str, period_col: str, start_col: str,
                      feature_map: dict[str, str], target_source: str, target_name: str,
                      horizon: int, carry_target_cols: dict[str, str] | None = None,
                      design: str = "") -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1 (a target must lie strictly after its inputs)")
    if target_name in feature_map.values():
        raise ValueError("target column cannot also be a feature")
    carry_target_cols = carry_target_cols or {}

    left = panel[[id_col, period_col, start_col, *feature_map]].rename(
        columns={**feature_map, period_col: "input_period", start_col: "input_start"})
    left["target_start"] = left["input_start"] + horizon
    right = panel[[id_col, period_col, start_col, target_source, *carry_target_cols]].rename(
        columns={period_col: "target_period", start_col: "target_start", target_source: target_name,
                 **carry_target_cols})
    out = left.merge(right, on=[id_col, "target_start"], how="inner", validate="one_to_one")
    out = out.dropna(subset=[target_name])
    out["horizon_years"] = horizon
    out["design"] = design
    feats = list(feature_map.values())
    out["features_complete"] = out[feats].notna().all(axis=1)
    assert (out["target_start"] - out["input_start"] == horizon).all()
    cols = [id_col, "input_period", "target_period", "input_start", "target_start", "horizon_years",
            "design", *feats, target_name, *carry_target_cols.values(), "features_complete"]
    return out[cols].sort_values([id_col, "input_start"]).reset_index(drop=True)


def next_window_label(period_label: str, horizon: int = 1) -> str:
    """'2022-2023' -> '2023-2024' (horizon 1); single years are shifted the same way."""
    parts = [int(p) for p in str(period_label).split("-")]
    return "-".join(str(p + horizon) for p in parts)
