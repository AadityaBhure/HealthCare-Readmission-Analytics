"""Central configuration: paths, dataset specifications, data-source settings and constants.

Every other module imports paths and dataset metadata from here so that the local
pipeline, the Streamlit dashboard, the tests and GitHub Actions all agree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GEO_DIR = DATA_DIR / "geo"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SCREENSHOTS_DIR = FIGURES_DIR / "screenshots"

COUNTY_GEOJSON = GEO_DIR / "california_counties.geojson"

RANDOM_STATE = 42

# --------------------------------------------------------------------------------------
# Optional remote data source.
# By default the CSVs committed under data/raw/ are used (works locally and on
# Streamlit Community Cloud, which checks out the repository). Set
#   READMISSION_DATA_SOURCE=github
#   READMISSION_DATA_BASE_URL=https://raw.githubusercontent.com/<user>/<repo>/main/data/raw
# (environment variable or a root-level key in .streamlit/secrets.toml) to read the
# raw files over HTTPS instead. No credentials are needed or stored for public repos.
# --------------------------------------------------------------------------------------
DATA_SOURCE = os.environ.get("READMISSION_DATA_SOURCE", "local").strip().lower()
DATA_BASE_URL = os.environ.get("READMISSION_DATA_BASE_URL", "").strip().rstrip("/")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    filename: str
    title: str
    role: str
    granularity: str
    required_columns: tuple[str, ...]
    is_synthetic: bool = False
    encodings: tuple[str, ...] = ("utf-8", "cp1252")
    notes: tuple[str, ...] = field(default_factory=tuple)


DATASETS: dict[str, DatasetSpec] = {
    "cabg": DatasetSpec(
        key="cabg",
        filename="cabg_hospital_performance.csv",
        title="Dataset A - California CABG hospital performance",
        role="Primary supervised ML experiment (next-period isolated CABG readmission rate)",
        granularity="Hospital x reporting period x performance measure",
        required_columns=(
            "Year", "County", "Hospital", "Hospital ID", "Performance Measure",
            "Total Statewide CABG Cases",
            "Total Statewide CABG Cases for Performance Measure Reporting",
            "Statewide Adverse Events", "Statewide Adverse Events Rate (%)",
            "Total Hospital CABG Cases",
            "Total Hospital CABG Cases for Performance Measure Reporting",
            "Hospital Adverse Events", "Hospital Observed Adverse Events Rate (%)",
            "Hospital Expected Adverse Events Rate (%)",
            "Hospital Risk-Adjusted Adverse Events Rate (%,RAAER)",
            "Hospital Lower 95% CI for RAAER", "Hospital Upper 95% CI for RAAER",
            "Performance Rating", "Longitude", "Latitude",
        ),
    ),
    "stroke": DatasetSpec(
        key="stroke",
        filename="ischemic_stroke_outcomes.csv",
        title="Dataset B - California ischemic stroke 30-day outcomes",
        role="Separate stroke-specific analytics and temporal model",
        granularity="Hospital x reporting period x measure (readmission / mortality)",
        required_columns=(
            "Year", "County", "Hospital", "OSHPDID", "Measure", "Risk Adjusted Rate",
            "# of Deaths/Readmissions", "# of Cases", "Hospital Ratings", "Location 1",
        ),
    ),
    "allcause": DatasetSpec(
        key="allcause",
        filename="california_allcause_readmissions.csv",
        title="Dataset C - California all-cause unplanned 30-day readmissions",
        role="Population-level trends, county comparison and strata breakdowns",
        granularity="Year x stratum x county (strata other than 'Overall' are statewide only)",
        required_columns=(
            "Year", "Strata", "Strata Name", "County",
            "Total Admits (ICD-9)", "30-day Readmits (ICD-9)", "30-day Readmission Rate (ICD-9)",
            "Total Admits (ICD-10)", "30-day Readmits (ICD-10)", "30-day Readmission Rate (ICD-10)",
        ),
    ),
    "synthetic": DatasetSpec(
        key="synthetic",
        filename="synthetic_cabg_test.csv",
        title="Dataset D - Synthetic CABG scenario data",
        role="Schema validation, smoke tests and hypothetical scenarios ONLY (never training/evaluation)",
        granularity="One synthetic hospital per row",
        required_columns=(
            "Hospital_ID", "Hospital_Name", "Input_Period", "Target_Period",
            "Previous_CABG_Readmission_Rate_Pct", "CABG_Case_Volume",
            "Operative_Mortality_Rate_Pct", "Postoperative_Stroke_Rate_Pct",
            "Next_Period_CABG_Readmission_Rate_Pct", "Synthetic_Record_Flag",
        ),
        is_synthetic=True,
    ),
}

# CABG measure names (as published) mapped to short codes used throughout the code.
CABG_MEASURES = {
    "Isolated CABG 30-Day Readmission": "readmission",
    "Isolated CABG Post-Operative Stroke": "postop_stroke",
    "Isolated CABG Operative Mortality": "iso_mortality",
    "CABG+Valve Operative Mortality": "cabg_valve_mortality",
}
STROKE_MEASURES = {"30-day Readmission": "readmission", "30-day Mortality": "mortality"}

# Values that represent "not reported" in the source files. They become NaN, never 0.
MISSING_TOKENS = {"", ".", "-", "--", "na", "n/a", "nan", "none", "null", "suppressed", "*"}

# Minimum admissions for a county to be ranked on the population page (small
# denominators give unstable rates, e.g. Alpine County has ~10 admissions/year).
MIN_COUNTY_ADMITS_DEFAULT = 1000

APP_TITLE = "California Readmission Analytics"
PROJECT_TITLE = (
    "Explainable California Hospital Readmission Analytics: CABG Prediction, "
    "Stroke Outcome Analysis, and Population-Level Trends"
)

# Replace after deploying (see README). Kept as an explicit placeholder on purpose.
LIVE_DASHBOARD_URL = os.environ.get("LIVE_DASHBOARD_URL", "https://<your-app-name>.streamlit.app")
GITHUB_REPO_URL = os.environ.get(
    "GITHUB_REPO_URL", "https://github.com/<your-github-username>/healthcare-readmission-analytics"
)


# Files that must exist for the dashboard to start without re-running the pipeline.
REQUIRED_ARTIFACTS = [
    MODELS_DIR / "cabg_models.joblib",
    MODELS_DIR / "cabg_feature_schema.json",
    MODELS_DIR / "stroke_models.joblib",
    ARTIFACTS_DIR / "key_findings.json",
    ARTIFACTS_DIR / "cabg" / "hospital_predictions.csv",
    ARTIFACTS_DIR / "population" / "county.csv",
    PROCESSED_DIR / "cabg_panel.csv",
]


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR, FIGURES_DIR, SCREENSHOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
