"""Cached data access for the Streamlit app.

The app reads the artifacts written by ``run_pipeline.py`` (committed to the repository so
Streamlit Community Cloud can serve them immediately). If they are missing - e.g. a fresh
clone where artifacts were not committed - the pipeline is executed once in-process.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from . import cabg_model, config, model_store

REQUIRED = config.REQUIRED_ARTIFACTS
ID_DTYPES = {"hospital_id": str, "Hospital_ID": str}


@st.cache_resource(show_spinner="Running the analysis pipeline for the first time (about a minute)...")
def ensure_artifacts() -> str:
    if all(p.exists() for p in REQUIRED):
        return "artifacts loaded from repository"
    import run_pipeline  # local import: only needed on a fresh checkout

    rc = run_pipeline.main([])
    if rc != 0:
        raise RuntimeError("The analysis pipeline failed - see the server log for the failed validation check.")
    return "artifacts generated at start-up"


@st.cache_data(show_spinner=False)
def artifact_csv(rel: str) -> pd.DataFrame:
    ensure_artifacts()
    return pd.read_csv(config.ARTIFACTS_DIR / rel, dtype=ID_DTYPES)


@st.cache_data(show_spinner=False)
def processed_csv(name: str) -> pd.DataFrame:
    ensure_artifacts()
    return pd.read_csv(config.PROCESSED_DIR / f"{name}.csv", dtype=ID_DTYPES)


@st.cache_data(show_spinner=False)
def artifact_json(rel: str):
    ensure_artifacts()
    return json.loads((config.ARTIFACTS_DIR / rel).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def schema_json(prefix: str) -> dict:
    ensure_artifacts()
    return json.loads((config.MODELS_DIR / f"{prefix}_feature_schema.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def cabg_bundle() -> tuple[dict, dict]:
    """Load the evaluated CABG models. If the pickle cannot be loaded (e.g. scikit-learn version
    mismatch on the server), retrain deterministically via the pipeline and load again."""
    ensure_artifacts()
    try:
        return cabg_model.load_bundle()
    except Exception:  # pragma: no cover - environment-dependent
        import run_pipeline

        run_pipeline.main([])
        st.cache_data.clear()
        return cabg_model.load_bundle()


@st.cache_data(show_spinner=False)
def integrity_check() -> dict:
    models, schema = cabg_bundle()
    preds = artifact_csv("cabg/test_predictions.csv")
    return model_store.verify_bundle(models, schema, preds, tol=1e-6)
