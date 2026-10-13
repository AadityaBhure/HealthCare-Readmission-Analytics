"""Save / load trained model bundles together with their feature schema, so the dashboard
always runs exactly the estimator objects that were evaluated by the pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .modeling import MODEL_LABELS, ExperimentResult


def save_model_bundle(prefix: str, result: ExperimentResult, feature_info: dict, target_info: tuple,
                      models_dir: Path, data_hashes: dict[str, str], experiment: str) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.models, models_dir / f"{prefix}_models.joblib")
    feats = result.config.features
    schema = {
        "experiment": experiment,
        "features": feats,
        "feature_info": {f: {"description": feature_info[f][0], "unit": feature_info[f][1]} for f in feats},
        "target": result.config.target,
        "target_info": {"description": target_info[0], "unit": target_info[1]},
        "models": {k: MODEL_LABELS[k] for k in result.models},
        "recommended_model": result.recommended_model,
        "recommended_model_rule": "lowest mean MAE in hospital-grouped 5-fold CV on the training transitions",
        "rf_best_params": {k: (v if v is not None else "None") for k, v in result.rf_best_params.items()},
        "train_summary": result.train_summary,
        "test_metrics": result.test_metrics[["model", "MAE", "RMSE", "R2", "n_test"]].to_dict(orient="records"),
        "sklearn_version": sklearn.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256": data_hashes,
    }
    (models_dir / f"{prefix}_feature_schema.json").write_text(json.dumps(schema, indent=2, default=str),
                                                              encoding="utf-8")
    return schema


def load_model_bundle(prefix: str, models_dir: Path) -> tuple[dict, dict]:
    models = joblib.load(models_dir / f"{prefix}_models.joblib")
    schema = json.loads((models_dir / f"{prefix}_feature_schema.json").read_text(encoding="utf-8"))
    return models, schema


def verify_bundle(models: dict, schema: dict, test_predictions: pd.DataFrame, tol: float = 1e-6) -> dict:
    """Integrity check: re-predict the stored test set with the loaded models and compare with the
    predictions written by the pipeline. Confirms the dashboard uses the evaluated models."""
    X = test_predictions[schema["features"]]
    out = {}
    for name, model in models.items():
        col = f"pred_{name}"
        if col not in test_predictions:
            continue
        diff = float(np.max(np.abs(model.predict(X) - test_predictions[col].to_numpy(dtype=float))))
        out[name] = {"max_abs_diff": diff, "match": diff <= tol}
    return out
