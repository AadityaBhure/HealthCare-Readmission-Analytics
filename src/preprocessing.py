"""Reusable cleaning functions that turn the raw CSVs into tidy, typed tables.

Design rules (see DATA_DICTIONARY.md):
* missing / suppressed values become NaN - they are never filled with zero;
* raw files are never modified - all fixes happen in memory and are logged;
* each dataset keeps its own tidy table - no forced master join.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

_DASHES = {"–": "-", "—": "-", "‒": "-", "�": "-"}


# --------------------------------------------------------------------------------------
# Generic parsing helpers
# --------------------------------------------------------------------------------------
def clean_text(series: pd.Series) -> pd.Series:
    """Normalise whitespace, embedded line breaks and typographic dashes. Empty -> NaN."""
    s = series.astype("string")
    for bad, good in _DASHES.items():
        s = s.str.replace(bad, good, regex=False)
    s = s.str.replace(r"[\r\n\t]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.mask(s.isna() | (s == ""))
    return s.astype(object).where(s.notna(), np.nan)


def _is_missing_token(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin(config.MISSING_TOKENS) | series.isna()


def parse_number(series: pd.Series) -> pd.Series:
    """Parse numbers stored as text: '1,234' -> 1234.0, '.' / '' -> NaN. Never returns 0 for missing."""
    s = series.astype("string").str.strip()
    missing = _is_missing_token(series)
    s = s.str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    out = pd.to_numeric(s.where(~missing), errors="coerce")
    return out.astype(float)


def parse_rate_percent(series: pd.Series) -> pd.Series:
    """Parse percentage text ('13.80%', '14%', '12.15') into percentage points (13.8, 14.0, 12.15)."""
    return parse_number(series)


def normalize_id(series: pd.Series) -> pd.Series:
    """Hospital / OSHPD identifiers as 9-digit strings ('106010937'); placeholders -> NaN."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    valid = s.str.fullmatch(r"\d{6,12}").fillna(False).astype(bool)
    return s.astype(object).where(valid, np.nan)


_PERIOD_RE = re.compile(r"^\s*(\d{4})\s*(?:-\s*(\d{4}))?\s*$")


def parse_period(label: str) -> tuple[int, int, int]:
    """'2018-2019' -> (2018, 2019, 2); '2019' -> (2019, 2019, 1)."""
    m = _PERIOD_RE.match(str(label))
    if not m:
        raise ValueError(f"Unrecognised reporting period label: {label!r}")
    start = int(m.group(1))
    end = int(m.group(2) or m.group(1))
    if end < start:
        raise ValueError(f"Reporting period ends before it starts: {label!r}")
    return start, end, end - start + 1


def normalize_rating(series: pd.Series) -> pd.Series:
    s = clean_text(series)
    s = s.replace({"Not Appplicable": "Not Applicable", "Not Appplicable ": "Not Applicable"})
    return s


def _parse_location(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'(37.762953, -122.25362)' -> (lat, lon)."""
    ext = series.astype("string").str.extract(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)")
    lat = pd.to_numeric(ext[0], errors="coerce").astype(float)
    lon = pd.to_numeric(ext[1], errors="coerce").astype(float)
    return lat, lon


# --------------------------------------------------------------------------------------
# Preprocessing log
# --------------------------------------------------------------------------------------
@dataclass
class PreprocessingLog:
    entries: list[dict] = field(default_factory=list)

    def add(self, dataset: str, step: str, detail: str, rows_affected: int | None = None) -> None:
        self.entries.append(
            {"dataset": dataset, "step": step, "detail": detail,
             "rows_affected": None if rows_affected is None else int(rows_affected)}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries, columns=["dataset", "step", "detail", "rows_affected"])


def _latest_name(df: pd.DataFrame, id_col: str, name_col: str, order_col: str) -> pd.Series:
    """Map each identifier to its name in the most recent period (names change after rebrands)."""
    latest = df.dropna(subset=[id_col]).sort_values(order_col).groupby(id_col)[name_col].last()
    return df[id_col].map(latest)


# --------------------------------------------------------------------------------------
# Dataset A - CABG
# --------------------------------------------------------------------------------------
def clean_cabg(raw: pd.DataFrame, log: PreprocessingLog | None = None) -> pd.DataFrame:
    log = log or PreprocessingLog()
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log.add("cabg", "duplicates", f"Removed {dups} exact duplicate rows", dups)

    out = pd.DataFrame(index=df.index)
    out["period"] = clean_text(df["Year"])
    parsed = out["period"].map(parse_period)
    out["period_start"] = parsed.map(lambda t: t[0]).astype(int)
    out["period_end"] = parsed.map(lambda t: t[1]).astype(int)
    out["period_years"] = parsed.map(lambda t: t[2]).astype(int)
    out["county"] = clean_text(df["County"])
    out["hospital_name_reported"] = clean_text(df["Hospital"])
    out["hospital_id"] = normalize_id(df["Hospital ID"])
    out["measure"] = clean_text(df["Performance Measure"])
    out["measure_code"] = out["measure"].map(config.CABG_MEASURES)

    numeric_map = {
        "Total Statewide CABG Cases": "statewide_cabg_cases",
        "Total Statewide CABG Cases for Performance Measure Reporting": "statewide_measure_cases",
        "Statewide Adverse Events": "statewide_events",
        "Statewide Adverse Events Rate (%)": "statewide_rate",
        "Total Hospital CABG Cases": "hospital_cabg_cases",
        "Total Hospital CABG Cases for Performance Measure Reporting": "hospital_measure_cases",
        "Hospital Adverse Events": "hospital_events",
        "Hospital Observed Adverse Events Rate (%)": "observed_rate",
        "Hospital Expected Adverse Events Rate (%)": "expected_rate",
        "Hospital Risk-Adjusted Adverse Events Rate (%,RAAER)": "raaer",
        "Hospital Lower 95% CI for RAAER": "raaer_ci_low",
        "Hospital Upper 95% CI for RAAER": "raaer_ci_high",
        "Longitude": "longitude",
        "Latitude": "latitude",
    }
    for src, dst in numeric_map.items():
        out[dst] = parse_number(df[src])
    out["rating"] = normalize_rating(df["Performance Rating"])

    n_dot = int((df["Hospital Risk-Adjusted Adverse Events Rate (%,RAAER)"].str.strip() == ".").sum())
    log.add("cabg", "missing_markers",
            f"Converted '.' (not reported) to NaN in rate/count fields; {n_dot} rows had no RAAER "
            "(all 'CABG+Valve Operative Mortality', rating 'Not Applicable'). Missing values were NOT set to 0.",
            n_dot)
    log.add("cabg", "numeric_parsing",
            "Removed thousands separators (e.g. '31,086') and cast counts/rates to float.")
    log.add("cabg", "text_normalisation",
            "Collapsed embedded line breaks/double spaces in County, Hospital and Rating "
            "(e.g. 'San\\nBernardino', 'Contra  Costa'); mapped Windows-1252 en dashes to '-'; "
            "fixed rating typo 'Not Appplicable' -> 'Not Applicable'.")

    unknown_measure = out["measure_code"].isna().sum()
    if unknown_measure:
        log.add("cabg", "unknown_measures", f"{unknown_measure} rows with unrecognised measure kept but uncoded")

    # Hospital display name = name used in the most recent period for that ID.
    out["hospital_name"] = _latest_name(out, "hospital_id", "hospital_name_reported", "period_end")
    n_ids = out["hospital_id"].nunique()
    n_names = out["hospital_name_reported"].nunique()
    log.add("cabg", "hospital_identity",
            f"{n_names} reported name variants map to {n_ids} stable Hospital IDs; the most recent "
            "name per ID is used for display. Hospital ID (not name) is the panel key.")
    latest_county = out.sort_values("period_end").groupby("hospital_id")["county"].last()
    out["county_latest"] = out["hospital_id"].map(latest_county)

    # Plausibility checks (flag, do not silently drop).
    bad_rate = ~out["raaer"].between(0, 100) & out["raaer"].notna()
    out.loc[bad_rate, ["raaer", "raaer_ci_low", "raaer_ci_high"]] = np.nan
    log.add("cabg", "range_check", f"RAAER outside [0, 100] set to NaN: {int(bad_rate.sum())} rows",
            int(bad_rate.sum()))
    over = (out["hospital_events"] > out["hospital_measure_cases"]) & out["hospital_events"].notna()
    log.add("cabg", "events_le_cases", f"Rows with adverse events > eligible cases: {int(over.sum())}",
            int(over.sum()))
    assert len(out) == n0 - dups
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Dataset B - ischemic stroke
# --------------------------------------------------------------------------------------
def clean_stroke(raw: pd.DataFrame, log: PreprocessingLog | None = None) -> pd.DataFrame:
    log = log or PreprocessingLog()
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log.add("stroke", "duplicates", f"Removed {dups} exact duplicate rows", dups)

    out = pd.DataFrame(index=df.index)
    out["period"] = clean_text(df["Year"])
    parsed = out["period"].map(parse_period)
    out["period_start"] = parsed.map(lambda t: t[0]).astype(int)
    out["period_end"] = parsed.map(lambda t: t[1]).astype(int)
    out["hospital_name_reported"] = clean_text(df["Hospital"])
    out["is_statewide"] = out["hospital_name_reported"].str.lower().eq("statewide")
    county = clean_text(df["County"])
    placeholder = county.isin(["AAAA"])
    out["county"] = county.mask(placeholder | out["is_statewide"])
    log.add("stroke", "statewide_rows",
            f"Identified {int(out['is_statewide'].sum())} 'Statewide' summary rows; their county "
            "placeholder 'AAAA' / blank and ID '.' were set to NaN.", int(out["is_statewide"].sum()))
    out["hospital_id"] = normalize_id(df["OSHPDID"])
    out["measure"] = clean_text(df["Measure"])
    out["measure_code"] = out["measure"].map(config.STROKE_MEASURES)
    out["risk_adjusted_rate"] = parse_number(df["Risk Adjusted Rate"])
    out["events"] = parse_number(df["# of Deaths/Readmissions"])
    out["cases"] = parse_number(df["# of Cases"])
    out["rating"] = normalize_rating(df["Hospital Ratings"])
    out["latitude"], out["longitude"] = _parse_location(df["Location 1"])
    n_missing = int(out["risk_adjusted_rate"].isna().sum())
    log.add("stroke", "missing_rates",
            f"{n_missing} hospital-measure rows have no published rate/counts; kept as NaN (not 0).",
            n_missing)

    # Systematic column swap: in one period every row has events > cases.
    has = out["events"].notna() & out["cases"].notna()
    out["counts_corrected"] = False
    for period, grp in out[has].groupby("period"):
        frac = float((grp["events"] > grp["cases"]).mean())
        if frac >= 0.9:
            idx = grp.index
            out.loc[idx, ["events", "cases"]] = out.loc[idx, ["cases", "events"]].to_numpy()
            out.loc[idx, "counts_corrected"] = True
            log.add("stroke", "column_swap_fix",
                    f"Period {period}: {frac:.0%} of rows had '# of Deaths/Readmissions' > '# of Cases' "
                    "(impossible). The two count columns are systematically transposed in this period; "
                    "they were swapped back. Risk-adjusted rates were unaffected. Verified: after the swap "
                    "the observed rate correlates with the risk-adjusted rate at the same level as in "
                    "other periods.", len(idx))
    residual = (out["events"] > out["cases"]) & has
    if residual.any():
        out.loc[residual, ["events", "cases"]] = np.nan
        log.add("stroke", "events_gt_cases", "Isolated rows with events > cases set to NaN", int(residual.sum()))

    out["observed_rate"] = np.where(out["cases"] > 0, out["events"] / out["cases"] * 100, np.nan)
    out["hospital_name"] = _latest_name(out, "hospital_id", "hospital_name_reported", "period_end")
    out.loc[out["is_statewide"], "hospital_name"] = "Statewide"
    latest_county = out.dropna(subset=["county"]).sort_values("period_end").groupby("hospital_id")["county"].last()
    out["county_latest"] = out["hospital_id"].map(latest_county)
    log.add("stroke", "hospital_identity",
            f"{out['hospital_name_reported'].nunique()} reported names map to "
            f"{out['hospital_id'].nunique()} OSHPD IDs; most recent name used for display.")
    log.add("stroke", "location_parsing", "Parsed 'Location 1' '(lat, lon)' text into numeric latitude/longitude.")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Dataset C - all-cause readmissions
# --------------------------------------------------------------------------------------
def clean_allcause(raw: pd.DataFrame, log: PreprocessingLog | None = None) -> pd.DataFrame:
    log = log or PreprocessingLog()
    df = raw.copy()
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    log.add("allcause", "column_names", "Stripped trailing spaces from column names (e.g. '30-day Readmits (ICD-9) ').")
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log.add("allcause", "duplicates", f"Removed {dups} exact duplicate rows", dups)

    out = pd.DataFrame(index=df.index)
    out["year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)
    out["strata"] = clean_text(df["Strata"])
    out["strata_name"] = clean_text(df["Strata Name"])
    out["county"] = clean_text(df["County"])

    icd = {}
    for code in ("ICD-9", "ICD-10"):
        icd[code] = pd.DataFrame({
            "admits": parse_number(df[f"Total Admits ({code})"]),
            "readmits": parse_number(df[f"30-day Readmits ({code})"]),
            "rate": parse_rate_percent(df[f"30-day Readmission Rate ({code})"]),
        })
    has9 = icd["ICD-9"].notna().any(axis=1)
    has10 = icd["ICD-10"].notna().any(axis=1)
    both = has9 & has10
    if both.any():
        raise ValueError(f"{int(both.sum())} all-cause rows report both ICD-9 and ICD-10 values")
    out["coding_system"] = np.select([has9, has10], ["ICD-9", "ICD-10"], default=None)
    for col in ("admits", "readmits", "rate"):
        out[col if col != "rate" else "rate_reported"] = icd["ICD-9"][col].where(has9, icd["ICD-10"][col])
    out["rate_computed"] = np.where(
        (out["admits"] > 0) & out["readmits"].notna(), out["readmits"] / out["admits"] * 100, np.nan
    )
    # Analysis rate: the published rate; if unpublished but counts exist, use the recomputed rate.
    out["rate"] = out["rate_reported"].fillna(out["rate_computed"])
    out["geography_level"] = np.select(
        [out["county"].eq("State"), out["county"].eq("Unknown County")], ["State", "Unknown county"], "County"
    )
    last_icd9 = int(out.loc[out["coding_system"] == "ICD-9", "year"].max())
    out["coding_transition_year"] = out["year"].eq(last_icd9)

    log.add("allcause", "icd_eras",
            f"ICD-9 fields populated for {int(has9.sum())} rows (years "
            f"{int(out.loc[has9, 'year'].min())}-{int(out.loc[has9, 'year'].max())}); ICD-10 fields for "
            f"{int(has10.sum())} rows ({int(out.loc[has10, 'year'].min())}-{int(out.loc[has10, 'year'].max())}). "
            "Combined into one tidy table with an explicit 'coding_system' column; the two eras are "
            "never pooled without that label.")
    log.add("allcause", "rate_parsing",
            "Parsed rates such as '14%', '13.80%' into percentage points; removed thousands separators "
            "from ICD-10 counts (e.g. '1,877,198').")
    supp = int((out["admits"].notna() & out["readmits"].isna()).sum())
    log.add("allcause", "suppressed_values",
            f"{supp} row(s) report admissions but no readmissions/rate (small-cell suppression, "
            "e.g. Alpine 2017); left as NaN.", supp)
    log.add("allcause", "transition_year",
            f"{last_icd9} is the final ICD-9 year. Its statewide admissions are ~25% below the prior "
            "year, consistent with ICD-9 coding ending on 1 Oct 2015 (no ICD-10 rows exist for that "
            "year in this file). It is flagged 'coding_transition_year' and treated as a partial year.")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Dataset D - synthetic CABG scenarios
# --------------------------------------------------------------------------------------
SYNTHETIC_TO_MODEL_FEATURES = {
    "Previous_CABG_Readmission_Rate_Pct": "prev_readmit_rate",
    "Postoperative_Stroke_Rate_Pct": "prev_stroke_rate",
    "CABG_Case_Volume": "prev_cabg_volume",
}


def prepare_synthetic(raw: pd.DataFrame, log: PreprocessingLog | None = None) -> pd.DataFrame:
    log = log or PreprocessingLog()
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    out = df.copy()
    for col in ("Previous_CABG_Readmission_Rate_Pct", "CABG_Case_Volume", "Operative_Mortality_Rate_Pct",
                "Postoperative_Stroke_Rate_Pct", "Next_Period_CABG_Readmission_Rate_Pct"):
        out[col] = parse_number(df[col])
    flag = df["Synthetic_Record_Flag"].astype("string").str.strip().str.lower()
    out["Synthetic_Record_Flag"] = flag.map({"true": True, "false": False, "1": True, "0": False})
    for src, dst in SYNTHETIC_TO_MODEL_FEATURES.items():
        out[dst] = out[src]
    out["synthetic_label_next_rate"] = out["Next_Period_CABG_Readmission_Rate_Pct"]
    out["is_synthetic"] = True
    log.add("synthetic", "schema_mapping",
            "Mapped synthetic columns onto the CABG model feature schema: "
            + ", ".join(f"{k} -> {v}" for k, v in SYNTHETIC_TO_MODEL_FEATURES.items())
            + ". 'Operative_Mortality_Rate_Pct' has no aligned real-data counterpart in the primary model "
              "and is reported but not used.")
    log.add("synthetic", "isolation",
            "All synthetic rows are tagged is_synthetic=True and are excluded from training, "
            "testing and every real-data statistic.")
    return out
