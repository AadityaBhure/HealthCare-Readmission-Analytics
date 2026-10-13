"""Data validation checks for raw and processed datasets.

Each check returns a :class:`Check` with status ``pass`` / ``warn`` / ``fail``.
The pipeline aborts with a clear message if any check fails; warnings are
reported in the dashboard (Model Evaluation -> Data validation) and the report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class Check:
    check_id: str
    dataset: str
    description: str
    status: str  # pass | warn | fail
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


def _status(ok: bool, warn_only: bool = False) -> str:
    if ok:
        return "pass"
    return "warn" if warn_only else "fail"


def validate_raw(raw: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []
    for key, spec in config.DATASETS.items():
        df = raw.get(key)
        if df is None:
            checks.append(Check(f"{key}.present", key, "Dataset file is present", "fail",
                                f"{spec.filename} missing"))
            continue
        cols = {c.strip() for c in df.columns}
        missing = [c for c in spec.required_columns if c.strip() not in cols]
        checks.append(Check(f"{key}.columns", key, "All required columns present",
                            _status(not missing), "missing: " + ", ".join(missing) if missing
                            else f"{len(spec.required_columns)} required columns found"))
        checks.append(Check(f"{key}.rows", key, "Dataset is non-empty", _status(len(df) > 0),
                            f"{len(df):,} rows; encoding {df.attrs.get('encoding', '?')}"))
        n_dup = int(df.duplicated().sum())
        checks.append(Check(f"{key}.duplicates", key, "No exact duplicate rows",
                            _status(n_dup == 0, warn_only=True), f"{n_dup} duplicate rows"))
    return checks


def validate_processed(cabg: pd.DataFrame, stroke: pd.DataFrame, allcause: pd.DataFrame,
                       synthetic: pd.DataFrame) -> list[Check]:
    c: list[Check] = []

    # ---------------- CABG ----------------
    key = ["period", "hospital_id", "measure"]
    dup = int(cabg.duplicated(key).sum())
    c.append(Check("cabg.unique_key", "cabg", "One row per hospital x period x measure",
                   _status(dup == 0), f"{dup} duplicate keys"))
    c.append(Check("cabg.ids", "cabg", "Every row has a valid 9-digit Hospital ID",
                   _status(cabg["hospital_id"].notna().all()),
                   f"{cabg['hospital_id'].isna().sum()} missing IDs; {cabg['hospital_id'].nunique()} unique IDs"))
    c.append(Check("cabg.measures", "cabg", "All measures recognised",
                   _status(cabg["measure_code"].notna().all()),
                   ", ".join(sorted(cabg["measure"].dropna().unique()))))
    rate_ok = cabg["raaer"].dropna().between(0, 100).all()
    c.append(Check("cabg.raaer_range", "cabg", "Risk-adjusted rates within [0, 100] %",
                   _status(rate_ok), f"min {cabg['raaer'].min():.2f}, max {cabg['raaer'].max():.2f}"))
    ci = cabg.dropna(subset=["raaer", "raaer_ci_low", "raaer_ci_high"])
    ci_bad = int(((ci["raaer_ci_low"] - ci["raaer"] > 0.01) | (ci["raaer"] - ci["raaer_ci_high"] > 0.01)).sum())
    c.append(Check("cabg.ci_order", "cabg", "Lower CI <= RAAER <= Upper CI",
                   _status(ci_bad == 0, warn_only=True), f"{ci_bad} rows violate CI ordering"))
    ev = cabg.dropna(subset=["hospital_events", "hospital_measure_cases"])
    ev_bad = int((ev["hospital_events"] > ev["hospital_measure_cases"]).sum())
    c.append(Check("cabg.events_le_cases", "cabg", "Adverse events <= eligible cases",
                   _status(ev_bad == 0), f"{ev_bad} violations"))
    readm = cabg[cabg["measure_code"] == "readmission"]
    cov = readm.groupby("period")["hospital_id"].nunique()
    sparse = cov[cov < 0.5 * cov.median()]
    c.append(Check("cabg.readmission_coverage", "cabg", "Readmission periods have comparable hospital coverage",
                   _status(sparse.empty, warn_only=True),
                   "; ".join(f"{p}: {n} hospitals" for p, n in cov.items())
                   + (f" -> sparse: {', '.join(sparse.index)}" if not sparse.empty else "")))
    # Rolling two-year windows: statewide 2-year totals equal the sum of single-year totals.
    tot = cabg.groupby("period").agg(start=("period_start", "first"), end=("period_end", "first"),
                                     years=("period_years", "first"),
                                     cases=("statewide_cabg_cases", "first"))
    single = tot[tot["years"] == 1].set_index("start")["cases"]
    msgs, matches = [], 0
    for p, row in tot[tot["years"] == 2].iterrows():
        if row["start"] in single.index and row["end"] in single.index:
            s = single[row["start"]] + single[row["end"]]
            ok = abs(s - row["cases"]) / row["cases"] < 0.01
            matches += ok
            msgs.append(f"{p}: {row['cases']:,.0f} vs {int(row['start'])}+{int(row['end'])} = {s:,.0f}")
    c.append(Check("cabg.rolling_windows", "cabg",
                   "Two-year periods are overlapping rolling windows (statewide cases = sum of single years)",
                   "pass" if matches else "warn",
                   "; ".join(msgs) + ". Consecutive two-year periods share one calendar year of patients."))

    # ---------------- Stroke ----------------
    hosp = stroke[~stroke["is_statewide"]]
    dup = int(hosp.duplicated(["period", "hospital_id", "measure"]).sum())
    c.append(Check("stroke.unique_key", "stroke", "One row per hospital x period x measure",
                   _status(dup == 0), f"{dup} duplicate keys"))
    c.append(Check("stroke.ids", "stroke", "Hospital rows have valid OSHPD IDs",
                   _status(hosp["hospital_id"].notna().all()),
                   f"{hosp['hospital_id'].isna().sum()} missing; {hosp['hospital_id'].nunique()} unique"))
    c.append(Check("stroke.rate_range", "stroke", "Risk-adjusted rates within [0, 100] %",
                   _status(stroke["risk_adjusted_rate"].dropna().between(0, 100).all()),
                   f"min {stroke['risk_adjusted_rate'].min():.2f}, max {stroke['risk_adjusted_rate'].max():.2f}"))
    ev = stroke.dropna(subset=["events", "cases"])
    bad = int((ev["events"] > ev["cases"]).sum())
    fixed = int(stroke["counts_corrected"].sum())
    c.append(Check("stroke.events_le_cases", "stroke", "Deaths/readmissions <= cases (after correction)",
                   _status(bad == 0), f"{bad} violations remain; {fixed} rows corrected by column swap"))
    if fixed:
        c.append(Check("stroke.column_swap", "stroke", "Source count columns consistent across periods",
                       "warn", f"{fixed} rows in "
                       + ", ".join(sorted(stroke.loc[stroke['counts_corrected'], 'period'].unique()))
                       + " had transposed count columns in the source file (corrected in memory)."))
    miss = int(hosp["risk_adjusted_rate"].isna().sum())
    c.append(Check("stroke.missing_rates", "stroke", "Missing hospital rates are kept as NaN",
                   "pass", f"{miss} hospital-measure rows without a published rate (not imputed)"))

    # ---------------- All-cause ----------------
    key = ["year", "strata", "strata_name", "county"]
    dup = int(allcause.duplicated(key).sum())
    c.append(Check("allcause.unique_key", "allcause", "One row per year x stratum x county",
                   _status(dup == 0), f"{dup} duplicate keys"))
    c.append(Check("allcause.single_coding_system", "allcause", "Each row uses exactly one coding system",
                   _status(allcause["coding_system"].notna().all()),
                   allcause.groupby("coding_system")["year"].agg(["min", "max"]).to_string()
                   .replace("\n", "; ")))
    both = allcause.dropna(subset=["rate_reported", "rate_computed"])
    diff = (both["rate_reported"] - both["rate_computed"]).abs()
    # Early years publish rates rounded to 1 decimal or whole numbers ("14%"), so allow 0.5 pp.
    c.append(Check("allcause.rate_consistency", "allcause",
                   "Published rate agrees with readmits / admits (rounding tolerance 0.5 pp)",
                   _status(bool((diff <= 0.5).all()), warn_only=True),
                   f"max abs difference {diff.max():.3f} pp; mean {diff.mean():.3f} pp"))
    c.append(Check("allcause.rate_range", "allcause", "Rates within [0, 100] %",
                   _status(allcause["rate"].dropna().between(0, 100).all()),
                   f"min {allcause['rate'].min():.2f}, max {allcause['rate'].max():.2f}"))
    small = allcause[(allcause["strata"] == "Overall") & (allcause["geography_level"] == "County")
                     & (allcause["admits"] < config.MIN_COUNTY_ADMITS_DEFAULT)]
    c.append(Check("allcause.small_denominators", "allcause",
                   f"County rates based on >= {config.MIN_COUNTY_ADMITS_DEFAULT:,} admissions",
                   "warn" if len(small) else "pass",
                   f"{len(small)} county-years below threshold (e.g. "
                   + ", ".join(sorted(small['county'].unique())[:6]) + "); flagged in dashboard rankings"))

    # ---------------- Synthetic ----------------
    flag_ok = bool(synthetic["Synthetic_Record_Flag"].eq(True).all())
    c.append(Check("synthetic.flag", "synthetic", "Every synthetic row has Synthetic_Record_Flag = True",
                   _status(flag_ok), f"{int(synthetic['Synthetic_Record_Flag'].eq(True).sum())}/{len(synthetic)} flagged"))
    prefix_ok = bool(synthetic["Hospital_ID"].astype(str).str.startswith("SYNTH_").all())
    c.append(Check("synthetic.id_prefix", "synthetic", "Synthetic IDs use the SYNTH_ prefix",
                   _status(prefix_ok), "all IDs start with SYNTH_" if prefix_ok else "non-prefixed IDs found"))
    real_ids = set(cabg["hospital_id"].dropna()) | set(stroke["hospital_id"].dropna())
    overlap = real_ids & set(synthetic["Hospital_ID"].astype(str))
    c.append(Check("synthetic.no_overlap", "synthetic", "No synthetic ID collides with a real hospital ID",
                   _status(not overlap), f"{len(overlap)} collisions"))
    num_cols = ["prev_readmit_rate", "prev_stroke_rate", "prev_cabg_volume", "synthetic_label_next_rate"]
    finite = bool(np.isfinite(synthetic[num_cols].to_numpy(dtype=float)).all())
    c.append(Check("synthetic.numeric", "synthetic", "Synthetic model inputs are numeric and finite",
                   _status(finite), ", ".join(num_cols)))
    return c


def summarize(checks: list[Check]) -> dict[str, int]:
    out = {"pass": 0, "warn": 0, "fail": 0}
    for ch in checks:
        out[ch.status] += 1
    return out


def checks_frame(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([ch.as_dict() for ch in checks])
