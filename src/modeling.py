"""Model definitions and a generic, leakage-aware temporal experiment runner shared by the
CABG (primary) and ischemic-stroke (secondary) experiments.

Protocol (identical for both conditions):
1. Forward temporal hold-out: models are fitted on earlier transitions only and evaluated
   once on the latest transition (the test set is never used for fitting or tuning).
2. All learned preprocessing (median imputation, log transform, scaling) lives inside a
   scikit-learn Pipeline, so it is fitted on training rows only.
3. Random-forest hyper-parameters and the "recommended" model are chosen with
   hospital-grouped cross-validation *inside the training set*.
4. Every model is compared with a persistence baseline (next rate = previous rate) and a
   training-mean baseline on exactly the same test rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config, evaluation

MODEL_LABELS = {
    "mean_baseline": "Training-mean baseline",
    "persistence": "Persistence baseline",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
}
LEARNED_MODELS = ("linear_regression", "random_forest")


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Naive forecast: the next-period rate equals the previously observed rate."""

    def __init__(self, feature: str = "prev_readmit_rate"):
        self.feature = feature

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature].to_numpy(dtype=float)
        idx = list(self.feature_names_in_).index(self.feature)
        return np.asarray(X, dtype=float)[:, idx]


def _preprocessor(features: list[str], log_features: list[str]) -> ColumnTransformer:
    log_cols = [f for f in features if f in log_features]
    other = [f for f in features if f not in log_features]
    transformers = []
    if log_cols:
        transformers.append(("log_volume", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ]), log_cols))
    if other:
        transformers.append(("rates", SimpleImputer(strategy="median"), other))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_models(features: list[str], log_features: list[str], persistence_feature: str,
                rf_params: dict | None = None) -> dict[str, BaseEstimator]:
    rf_params = rf_params or {}
    # n_jobs=1: the data are tiny (~120 rows), so thread start-up costs more than it saves and
    # nested parallelism inside GridSearchCV is avoided.
    rf = RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=1,
                               **{"min_samples_leaf": 5, **rf_params})
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "persistence": PersistenceRegressor(feature=persistence_feature),
        "linear_regression": Pipeline([
            ("prep", _preprocessor(features, log_features)),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor(features, log_features)),
            ("model", rf),
        ]),
    }


RF_PARAM_GRID = {
    "model__min_samples_leaf": [5, 10, 20],
    "model__max_depth": [3, 5, None],
    "model__max_features": [1.0, "sqrt"],
}


@dataclass
class ExperimentConfig:
    name: str
    features: list[str]
    target: str
    persistence_feature: str
    log_features: list[str] = field(default_factory=list)
    group_col: str = "hospital_id"
    volume_col: str | None = None  # target-period cohort size, for error analysis only
    n_bootstrap: int = 2000
    n_perm_repeats: int = 30
    cv_splits: int = 5
    cv_repeats: int = 5
    tune_rf: bool = True


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    models: dict
    rf_best_params: dict
    train_cv_selection: pd.DataFrame
    recommended_model: str
    test_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    bootstrap: pd.DataFrame
    permutation_importance: pd.DataFrame
    model_native_importance: pd.DataFrame
    error_by_volume: pd.DataFrame
    learned_curve: pd.DataFrame
    train_summary: dict


def run_temporal_experiment(train: pd.DataFrame, test: pd.DataFrame, cfg: ExperimentConfig,
                            fast: bool = False) -> ExperimentResult:
    feats, tgt = cfg.features, cfg.target
    if tgt in feats:
        raise ValueError("Target leakage: target listed as a feature")
    Xtr, ytr, gtr = train[feats], train[tgt], train[cfg.group_col]
    Xte, yte = test[feats], test[tgt]
    n_boot = 300 if fast else cfg.n_bootstrap
    n_perm = 5 if fast else cfg.n_perm_repeats

    # 1. Tune RF on the training set with hospital-grouped CV.
    base = make_models(feats, cfg.log_features, cfg.persistence_feature)
    rf_best = {"model__min_samples_leaf": 5, "model__max_depth": None, "model__max_features": 1.0}
    if cfg.tune_rf:
        grid = RF_PARAM_GRID if not fast else {"model__min_samples_leaf": [5, 10]}
        rf_search = clone(base["random_forest"])
        rf_search.set_params(model__n_estimators=150 if not fast else 50)
        gs = GridSearchCV(rf_search, grid, cv=GroupKFold(n_splits=cfg.cv_splits),
                          scoring="neg_mean_absolute_error", n_jobs=-1)
        gs.fit(Xtr, ytr, groups=gtr)
        rf_best = {**rf_best, **gs.best_params_}
    rf_params = {k.replace("model__", ""): v for k, v in rf_best.items()}
    models = make_models(feats, cfg.log_features, cfg.persistence_feature, rf_params)
    if fast:
        models["random_forest"].set_params(model__n_estimators=100)

    # 2. Model selection by grouped CV inside the training set only.
    sel_rows = evaluation.grouped_cv(models, Xtr, ytr, gtr, n_splits=cfg.cv_splits,
                                     n_repeats=1 if fast else 3)
    selection = evaluation.summarize_cv(sel_rows, list(MODEL_LABELS))
    recommended = str(selection.sort_values("MAE_mean").iloc[0]["model"])

    # 3. Fit on all training rows; evaluate once on the held-out later transition.
    preds = test[[cfg.group_col, "input_period", "target_period", *feats, tgt]].copy()
    if cfg.volume_col and cfg.volume_col in test:
        preds[cfg.volume_col] = test[cfg.volume_col].to_numpy()
    rows = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        preds[f"pred_{name}"] = p
        rows.append({"model": name, "label": MODEL_LABELS[name], **evaluation.regression_metrics(yte, p),
                     "n_train": int(len(Xtr)), "n_test": int(len(Xte))})
    metrics = pd.DataFrame(rows)
    pers_mae = float(metrics.loc[metrics["model"] == "persistence", "MAE"].iloc[0])
    metrics["MAE_vs_persistence_pct"] = (metrics["MAE"] - pers_mae) / pers_mae * 100
    metrics["recommended"] = metrics["model"].eq(recommended)

    # 4. Paired bootstrap of MAE differences (resampling hospitals in the test set).
    boot_rows = []
    pairs = [(m, "persistence") for m in ("linear_regression", "random_forest", "mean_baseline")]
    pairs.append(("random_forest", "linear_regression"))
    for a, b in pairs:
        res = evaluation.paired_bootstrap_mae_diff(yte, preds[f"pred_{a}"], preds[f"pred_{b}"],
                                                   n_boot=n_boot, seed=config.RANDOM_STATE)
        boot_rows.append({"model": a, "reference": b, **res})
    boot = pd.DataFrame(boot_rows)

    # 5. Explainability: permutation importance on held-out data + native importances.
    perm = []
    for name in LEARNED_MODELS:
        t = evaluation.permutation_importance_table(models[name], Xte, yte, n_repeats=n_perm,
                                                    seed=config.RANDOM_STATE)
        t.insert(0, "model", name)
        perm.append(t)
    perm = pd.concat(perm, ignore_index=True)
    lr = models["linear_regression"]
    names_out = list(lr.named_steps["prep"].get_feature_names_out())
    native = [pd.DataFrame({"model": "linear_regression", "feature": names_out,
                            "value": lr.named_steps["model"].coef_,
                            "kind": "standardised coefficient (pp per 1 SD; volume on log scale)"})]
    rf = models["random_forest"]
    native.append(pd.DataFrame({"model": "random_forest",
                                "feature": list(rf.named_steps["prep"].get_feature_names_out()),
                                "value": rf.named_steps["model"].feature_importances_,
                                "kind": "impurity importance (share of variance reduction)"}))
    native = pd.concat(native, ignore_index=True)

    # 6. Error analysis by target-period cohort size (volume terciles).
    err = pd.DataFrame()
    if cfg.volume_col and cfg.volume_col in preds:
        preds["volume_tercile"] = pd.qcut(preds[cfg.volume_col], 3,
                                          labels=["Low volume", "Medium volume", "High volume"])
        err = evaluation.error_by_group(preds, tgt, {m: f"pred_{m}" for m in models}, "volume_tercile")
        rng = preds.groupby("volume_tercile", observed=True)[cfg.volume_col].agg(["min", "max"])
        err["volume_range"] = err["volume_tercile"].map(
            lambda t: f"{int(rng.loc[t, 'min'])}-{int(rng.loc[t, 'max'])} cases")

    # 7. Learned relationship between previous and next rate (other features at training median).
    grid = np.linspace(np.nanpercentile(Xtr[cfg.persistence_feature], 1),
                       np.nanpercentile(Xtr[cfg.persistence_feature], 99), 60)
    base_row = Xtr.median(numeric_only=True)
    curve = pd.DataFrame([base_row.to_dict()] * len(grid))
    curve[cfg.persistence_feature] = grid
    curve = curve[feats]
    learned = pd.DataFrame({cfg.persistence_feature: grid})
    for name in ("persistence", "linear_regression", "random_forest", "mean_baseline"):
        learned[f"pred_{name}"] = models[name].predict(curve)

    train_summary = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "train_hospitals": int(train[cfg.group_col].nunique()),
        "test_hospitals": int(test[cfg.group_col].nunique()),
        "train_target_periods": sorted(train["target_period"].unique().tolist()),
        "test_target_periods": sorted(test["target_period"].unique().tolist()),
        "train_input_periods": sorted(train["input_period"].unique().tolist()),
        "test_input_periods": sorted(test["input_period"].unique().tolist()),
        "feature_ranges": {f: {"min": float(Xtr[f].min()), "max": float(Xtr[f].max()),
                               "median": float(Xtr[f].median()), "p05": float(Xtr[f].quantile(.05)),
                               "p95": float(Xtr[f].quantile(.95))} for f in feats},
        "target_train_mean": float(ytr.mean()),
    }
    return ExperimentResult(cfg, models, rf_best, selection, recommended, metrics, preds, boot, perm,
                            native, err, learned, train_summary)


def linear_contributions(pipe: Pipeline, x_row: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact additive decomposition of a Linear Regression prediction.

    prediction = intercept (prediction for a hospital at the training mean of every
    transformed feature) + sum(coef_j * z_j), where z_j is the standardised feature value.
    """
    prep, scale, lr = pipe.named_steps["prep"], pipe.named_steps["scale"], pipe.named_steps["model"]
    z = scale.transform(prep.transform(x_row))[0]
    names = list(prep.get_feature_names_out())
    df = pd.DataFrame({"feature": names, "standardised_value": z, "contribution": lr.coef_ * z})
    return df, float(lr.intercept_)
