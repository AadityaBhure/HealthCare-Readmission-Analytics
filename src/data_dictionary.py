"""Dataset audit + data dictionary generator (writes DATA_DICTIONARY.md and dataset_audit.json)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

RAW_COLUMN_DESCRIPTIONS = {
    "cabg": {
        "Year": "Reporting period: two-year rolling window ('2018-2019') or single year ('2019', operative mortality only)",
        "County": "County of the hospital",
        "Hospital": "Hospital name as published in that period (names change over time)",
        "Hospital ID": "9-digit OSHPD/HCAI facility identifier - stable panel key",
        "Performance Measure": "One of 4 CABG measures (30-day readmission, post-op stroke, isolated CABG mortality, CABG+valve mortality)",
        "Total Statewide CABG Cases": "All CABG cases statewide in the period",
        "Total Statewide CABG Cases for Performance Measure Reporting": "Statewide cases eligible for this measure",
        "Statewide Adverse Events": "Statewide count of the adverse event",
        "Statewide Adverse Events Rate (%)": "Statewide observed rate for the measure (%)",
        "Total Hospital CABG Cases": "Hospital CABG volume in the period (used as case-volume feature)",
        "Total Hospital CABG Cases for Performance Measure Reporting": "Hospital cases eligible for the measure (denominator)",
        "Hospital Adverse Events": "Hospital adverse-event count ('.' = not reported)",
        "Hospital Observed Adverse Events Rate (%)": "Events / eligible cases (%)",
        "Hospital Expected Adverse Events Rate (%)": "Rate expected from the hospital's patient case mix (%)",
        "Hospital Risk-Adjusted Adverse Events Rate (%,RAAER)": "Risk-adjusted rate (%) - the modelling target/feature",
        "Hospital Lower 95% CI for RAAER": "Lower 95% confidence limit of RAAER",
        "Hospital Upper 95% CI for RAAER": "Upper 95% confidence limit of RAAER",
        "Performance Rating": "Better / Average / Worse than state average, or Not Applicable",
        "Longitude": "Hospital longitude", "Latitude": "Hospital latitude",
    },
    "stroke": {
        "Year": "Two-year rolling reporting window (2011-2012 ... 2014-2015)",
        "County": "County (blank / 'AAAA' on statewide rows)",
        "Hospital": "Hospital name or 'Statewide'",
        "OSHPDID": "9-digit OSHPD facility identifier ('.' on statewide rows)",
        "Measure": "30-day Readmission or 30-day Mortality",
        "Risk Adjusted Rate": "Risk-adjusted rate (%)",
        "# of Deaths/Readmissions": "Event count (columns transposed in source for 2013-2014; corrected)",
        "# of Cases": "Eligible ischemic stroke cases (denominator)",
        "Hospital Ratings": "Better / As Expected / Worse",
        "Location 1": "'(lat, lon)' text",
    },
    "allcause": {
        "Year": "Calendar year 2011-2024",
        "Strata": "Overall, Age, Sex, Race-Ethnicity or Expected Payer",
        "Strata Name": "Stratum value (e.g. '65 years and above', 'Medi-Cal')",
        "County": "County name, 'State' or 'Unknown County' (non-Overall strata are State only)",
        "Total Admits (ICD-9)": "Index admissions coded in ICD-9 (2011-2015)",
        "30-day Readmits (ICD-9)": "Unplanned 30-day readmissions (ICD-9 era)",
        "30-day Readmission Rate (ICD-9)": "Readmission rate text, e.g. '13.80%' (ICD-9 era)",
        "Total Admits (ICD-10)": "Index admissions coded in ICD-10 (2016-2024), text with thousands separators",
        "30-day Readmits (ICD-10)": "Unplanned 30-day readmissions (ICD-10 era)",
        "30-day Readmission Rate (ICD-10)": "Readmission rate text (ICD-10 era)",
    },
    "synthetic": {
        "Hospital_ID": "Synthetic ID (SYNTH_HOSP_###)",
        "Hospital_Name": "Synthetic name",
        "Input_Period": "Always 'Synthetic prior period'",
        "Target_Period": "Always 'Synthetic next period'",
        "Previous_CABG_Readmission_Rate_Pct": "-> prev_readmit_rate",
        "CABG_Case_Volume": "-> prev_cabg_volume",
        "Operative_Mortality_Rate_Pct": "Not used by the primary model (no aligned real-data feature)",
        "Postoperative_Stroke_Rate_Pct": "-> prev_stroke_rate",
        "Next_Period_CABG_Readmission_Rate_Pct": "Synthetic label (agreement check only, never performance evidence)",
        "Synthetic_Record_Flag": "Must be True for every row",
    },
}

PROCESSED_TABLES = {
    "cabg_clean.csv": "Tidy CABG table: one row per hospital x period x measure; typed numerics; measure_code; "
                      "period_start/end/years; hospital_name = latest name for the ID.",
    "cabg_panel.csv": "One row per hospital x two-year window with readmission, stroke, CABG+valve mortality, "
                      "final-year isolated mortality and volume side by side.",
    "stroke_clean.csv": "Tidy stroke table; is_statewide flag; counts_corrected flag for the 2013-2014 swap fix.",
    "stroke_panel.csv": "One row per hospital x window with readmission and mortality side by side.",
    "allcause_clean.csv": "Tidy all-cause table with one rate column plus coding_system (ICD-9 / ICD-10), "
                          "rate_reported, rate_computed, geography_level, coding_transition_year.",
    "synthetic_clean.csv": "Synthetic rows with model-schema columns added; is_synthetic=True.",
}


def audit(raw: dict[str, pd.DataFrame], clean: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for key, spec in config.DATASETS.items():
        r = raw[key]
        stripped = r.apply(lambda s: s.astype("string").str.strip())
        blank = (stripped.isna() | stripped.eq("") | stripped.eq(".")).sum()
        info = {
            "title": spec.title, "file": spec.filename, "role": spec.role, "granularity": spec.granularity,
            "rows": int(len(r)), "columns": int(r.shape[1]), "encoding": r.attrs.get("encoding"),
            "bytes": r.attrs.get("bytes"), "sha256": r.attrs.get("sha256"),
            "duplicate_rows": int(r.duplicated().sum()),
            "blank_or_dot_by_column": {k.strip(): int(v) for k, v in blank.items() if v},
            "is_synthetic": spec.is_synthetic,
        }
        c = clean[key]
        if key == "cabg":
            info["periods"] = sorted(c["period"].unique().tolist())
            info["hospital_ids"] = int(c["hospital_id"].nunique())
            info["name_variants"] = int(c["hospital_name_reported"].nunique())
            info["counties"] = int(c["county"].nunique())
            info["rows_by_period_measure"] = (c.groupby(["period", "measure"]).size()
                                              .rename("rows").reset_index().to_dict(orient="records"))
        elif key == "stroke":
            info["periods"] = sorted(c["period"].unique().tolist())
            info["hospital_ids"] = int(c["hospital_id"].nunique())
            info["counties"] = int(c["county"].nunique())
            info["rows_corrected_for_column_swap"] = int(c["counts_corrected"].sum())
        elif key == "allcause":
            info["years"] = [int(c["year"].min()), int(c["year"].max())]
            info["strata"] = {k: sorted(v.unique().tolist()) for k, v in c.groupby("strata")["strata_name"]}
            info["counties_excluding_state"] = int(c.loc[c["geography_level"] == "County", "county"].nunique())
            info["coding_system_years"] = {k: [int(v.min()), int(v.max())] for k, v in c.groupby("coding_system")["year"]}
        else:
            info["flagged_synthetic"] = int(c["Synthetic_Record_Flag"].eq(True).sum())
        out[key] = info
    return out


def write_markdown(audit_info: dict, prep_log: pd.DataFrame, checks: pd.DataFrame, path: Path) -> None:
    L = ["# Data Dictionary and Dataset Audit", "",
         "_Generated automatically by `run_pipeline.py` from the files in `data/raw/`. Do not edit by hand._", "",
         "The four datasets are analysed in **separate workflows**; they are not merged into one master table "
         "because they differ in clinical condition, geography, population and reporting period.", ""]
    for key, info in audit_info.items():
        L += [f"## {info['title']}", "",
              f"- **File:** `{info['file']}` ({info['bytes']:,} bytes, encoding `{info['encoding']}`)",
              f"- **Role:** {info['role']}",
              f"- **Granularity:** {info['granularity']}",
              f"- **Rows x columns:** {info['rows']:,} x {info['columns']}; exact duplicate rows: {info['duplicate_rows']}",
              f"- **SHA-256:** `{info['sha256']}`"]
        if "periods" in info:
            L.append(f"- **Reporting periods:** {', '.join(info['periods'])}")
        if "hospital_ids" in info:
            L.append(f"- **Unique hospital IDs:** {info['hospital_ids']}"
                     + (f" ({info['name_variants']} name variants)" if "name_variants" in info else ""))
        if "years" in info:
            L.append(f"- **Years:** {info['years'][0]}-{info['years'][1]}; coding systems: "
                     + "; ".join(f"{k} {v[0]}-{v[1]}" for k, v in info["coding_system_years"].items()))
            L.append(f"- **Counties:** {info['counties_excluding_state']} (plus 'State' and, in 2019 only, 'Unknown County')")
        if info.get("is_synthetic"):
            L.append(f"- **Synthetic flag:** {info['flagged_synthetic']}/{info['rows']} rows flagged True. "
                     "**Never used for training or real-data evaluation.**")
        if info["blank_or_dot_by_column"]:
            L.append("- **Blank / '.' values by column:** "
                     + ", ".join(f"{k}: {v}" for k, v in info["blank_or_dot_by_column"].items()))
        L += ["", "| Column | Meaning |", "|---|---|"]
        for col, desc in RAW_COLUMN_DESCRIPTIONS[key].items():
            L.append(f"| `{col}` | {desc} |")
        L.append("")
    L += ["## Processed tables (`data/processed/`)", "", "| File | Contents |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in PROCESSED_TABLES.items()]
    L += ["", "## Preprocessing decisions (log)", "", "| Dataset | Step | Rows | Detail |", "|---|---|---|---|"]
    for _, r in prep_log.iterrows():
        rows = "" if pd.isna(r["rows_affected"]) else f"{int(r['rows_affected']):,}"
        L.append(f"| {r['dataset']} | {r['step']} | {rows} | {str(r['detail']).replace('|', '/')} |")
    L += ["", "## Validation checks", "", "| Status | Check | Detail |", "|---|---|---|"]
    for _, r in checks.iterrows():
        L.append(f"| {r['status'].upper()} | {r['description']} | {str(r['detail']).replace('|', '/')} |")
    L += ["", "## Joins performed", "",
          "- **CABG measures -> CABG panel:** joined on `hospital_id` + identical two-year `period` "
          "(readmission, post-operative stroke, CABG+valve mortality). Isolated CABG operative mortality "
          "(single-year) is joined on `hospital_id` + the input window's final calendar year and is used only "
          "in the labelled 'extended feature' sensitivity analysis.",
          "- **Stroke measures -> stroke panel:** joined on `hospital_id` + identical `period`.",
          "- **All-cause counties -> county boundaries:** county name to Census county `NAME` "
          "(58/58 matched exactly).",
          "- **Not joined:** CABG with stroke (no overlapping years), hospital datasets with all-cause "
          "population data (different units of analysis), synthetic with any real data.", ""]
    path.write_text("\n".join(L), encoding="utf-8")
