"""Evaluation utilities: metrics, paired bootstrap, hospital-grouped cross-validation,
permutation importance and error analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.parallel import Parallel, delayed


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "Bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def paired_bootstrap_mae_diff(y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    """Bootstrap the difference MAE(a) - MAE(b) by resampling evaluation rows (hospitals) with
    replacement. Negative values mean model *a* has lower error than *b*."""
    y = np.asarray(y_true, float)
    ea = np.abs(np.asarray(pred_a, float) - y)
    eb = np.abs(np.asarray(pred_b, float) - y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = ea[idx].mean(axis=1) - eb[idx].mean(axis=1)
    point = float(ea.mean() - eb.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mae_diff": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "prob_a_better": float(np.mean(diffs < 0)),
        "relative_change_pct": float(point / eb.mean() * 100) if eb.mean() > 0 else float("nan"),
        "n_boot": int(n_boot),
    }


def _fit_fold(models: dict, X: pd.DataFrame, y: pd.Series, tr, va, r: int, f: int) -> list[dict]:
    rows = []
    for name, est in models.items():
        m = clone(est)
        m.fit(X.iloc[tr], y.iloc[tr])
        met = regression_metrics(y.iloc[va], m.predict(X.iloc[va]))
        rows.append({"repeat": r, "fold": f, "model": name, **met})
    return rows


def grouped_cv(models: dict, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5,
               n_repeats: int = 5, seed: int = 42, n_jobs: int = -1) -> pd.DataFrame:
    """Repeated hospital-grouped K-fold CV: no hospital appears in both the training and the
    validation fold. Every model sees identical folds (paired comparison). One row per
    (repeat, fold, model)."""
    n_splits = min(n_splits, groups.nunique())
    tasks = []
    for r in range(n_repeats):
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, va) in enumerate(cv.split(X, y, groups)):
            tasks.append((tr, va, r, f))
    results = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(models, X, y, tr, va, r, f) for tr, va, r, f in tasks)
    return pd.DataFrame([row for fold_rows in results for row in fold_rows])


def summarize_cv(cv_rows: pd.DataFrame, model_order: list[str] | None = None) -> pd.DataFrame:
    agg = cv_rows.groupby("model").agg(
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        folds=("MAE", "size"), n_val_per_fold=("n", "mean"),
    )
    if model_order:
        agg = agg.reindex([m for m in model_order if m in agg.index])
    return agg.reset_index()


def paired_cv_difference(cv_rows: pd.DataFrame, model: str, reference: str) -> dict[str, float]:
    """Fold-wise paired comparison of MAE (model - reference) over identical validation folds."""
    a = cv_rows[cv_rows["model"] == model].set_index(["repeat", "fold"])["MAE"]
    b = cv_rows[cv_rows["model"] == reference].set_index(["repeat", "fold"])["MAE"]
    d = (a - b).dropna()
    return {"model": model, "reference": reference, "mean_mae_diff": float(d.mean()),
            "sd_mae_diff": float(d.std(ddof=1)), "folds": int(len(d)),
            "share_folds_model_better": float((d < 0).mean())}


def permutation_importance_table(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 30,
                                 seed: int = 42) -> pd.DataFrame:
    """Increase in MAE (percentage points) when each feature is randomly shuffled on held-out data."""
    res = permutation_importance(model, X, y, scoring="neg_mean_absolute_error",
                                 n_repeats=n_repeats, random_state=seed)
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mae_increase": res.importances_mean,
        "importance_sd": res.importances_std,
    }).sort_values("importance_mae_increase", ascending=False).reset_index(drop=True)


def error_by_group(df: pd.DataFrame, actual: str, pred_cols: dict[str, str], group_col: str) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        for model, col in pred_cols.items():
            rows.append({group_col: g, "model": model,
                         "MAE": float(np.mean(np.abs(sub[col] - sub[actual]))), "n": len(sub)})
    return pd.DataFrame(rows)
