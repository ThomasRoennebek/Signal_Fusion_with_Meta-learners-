"""
baselines.py — Stage 1 baseline utilities for weak-only, gold-only, and hybrid training.

Scope
-----
This module keeps notebook code focused on experiment orchestration rather than
model boilerplate. It provides:

- config dataclasses for Elastic Net and XGBoost
- preprocessing builder for tabular baselines
- mean predictor
- Elastic Net regressor
- XGBoost regressor
- shared prediction helpers
- feature importance extraction for XGBoost
- one wrapper to fit the full tabular baseline suite

Notes
-----
- These baselines are treated as score predictors in [0, 1].
- Weak-label distillation should be evaluated separately from gold-label validity.
- MLP is intentionally kept in `mlp.py`, since it has its own training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from master_thesis.metrics import evaluate_predictions


# =============================================================================
# Config objects
# =============================================================================

@dataclass
class ElasticNetConfig:
    alpha: float = 0.01
    l1_ratio: float = 0.5
    max_iter: int = 10000
    random_state: int = 42


@dataclass
class XGBoostConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class BaselineTrainingConfig:
    scale_numeric_for_elastic_net: bool = True
    scale_numeric_for_xgboost: bool = False
    add_numeric_missing_indicator: bool = True
    elastic_net: ElasticNetConfig = field(default_factory=ElasticNetConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)


# =============================================================================
# Column helpers
# =============================================================================

def get_numeric_and_categorical_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Infer numeric and categorical columns from a pandas DataFrame.
    """
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


# =============================================================================
# Preprocessing
# =============================================================================

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = True,
    add_numeric_missing_indicator: bool = True,
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline shared by Elastic Net and XGBoost.
    """
    numeric_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=add_numeric_missing_indicator,
            ),
        )
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


# =============================================================================
# Individual models
# =============================================================================

def fit_mean_predictor(
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray,
    model_name: str = "Mean Predictor",
) -> tuple[dict[str, float], np.ndarray, pd.DataFrame]:
    """
    Fit a mean predictor baseline.
    """
    train_mean = float(np.mean(y_train))
    y_pred = np.full(shape=len(y_val), fill_value=train_mean)

    df_results = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        model_name=model_name,
    )

    model = {"train_mean": train_mean}
    return model, y_pred, df_results


def fit_elastic_net(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: Optional[ElasticNetConfig] = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    add_numeric_missing_indicator: bool = True,
    scale_numeric: bool = True,
    model_name: str = "Elastic Net",
) -> tuple[Pipeline, np.ndarray, pd.DataFrame]:
    """
    Fit Elastic Net with optional sample weights.
    """
    if config is None:
        config = ElasticNetConfig()

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=scale_numeric,
        add_numeric_missing_indicator=add_numeric_missing_indicator,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                ElasticNet(
                    alpha=config.alpha,
                    l1_ratio=config.l1_ratio,
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)

    y_pred = np.clip(model.predict(X_val), 0, 1)

    df_results = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        model_name=model_name,
    )

    return model, y_pred, df_results


def fit_xgboost_regressor(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: Optional[XGBoostConfig] = None,
    sample_weight: pd.Series | np.ndarray | None = None,
    add_numeric_missing_indicator: bool = True,
    scale_numeric: bool = False,
    model_name: str = "XGBoost",
) -> tuple[Pipeline, np.ndarray, pd.DataFrame]:
    """
    Fit XGBoost regressor with optional sample weights.
    """
    if config is None:
        config = XGBoostConfig()

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=scale_numeric,
        add_numeric_missing_indicator=add_numeric_missing_indicator,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    max_depth=config.max_depth,
                    min_child_weight=config.min_child_weight,
                    subsample=config.subsample,
                    colsample_bytree=config.colsample_bytree,
                    reg_alpha=config.reg_alpha,
                    reg_lambda=config.reg_lambda,
                    random_state=config.random_state,
                    n_jobs=config.n_jobs,
                ),
            ),
        ]
    )

    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)

    y_pred = np.clip(model.predict(X_val), 0, 1)

    df_results = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        model_name=model_name,
    )

    return model, y_pred, df_results


# =============================================================================
# Prediction helpers
# =============================================================================

def predict_baseline_scores(
    fitted_model,
    X_input: pd.DataFrame | np.ndarray,
    model_type: str,
) -> np.ndarray:
    """
    Predict scores from a fitted baseline object.

    Parameters
    ----------
    fitted_model
        Either:
        - dict with `train_mean` for the mean predictor
        - sklearn Pipeline for Elastic Net / XGBoost
    X_input
        Input features for scoring. Ignored for mean predictor except for row count.
    model_type
        One of {"mean", "elastic_net", "xgboost"}.

    Returns
    -------
    np.ndarray
        1D array of scores clipped to [0, 1].
    """
    model_type = model_type.lower()

    if model_type == "mean":
        return np.full(shape=len(X_input), fill_value=float(fitted_model["train_mean"]))

    if model_type in {"elastic_net", "xgboost"}:
        y_pred = fitted_model.predict(X_input)
        return np.clip(np.asarray(y_pred).flatten(), 0, 1)

    raise ValueError(f"Unsupported model_type: {model_type}")


def predict_tabular_baselines(
    fitted_models: dict[str, Any],
    X_input: pd.DataFrame | np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Predict from the standard non-neural baseline suite.
    """
    return {
        "Mean Predictor": predict_baseline_scores(
            fitted_models["Mean Predictor"],
            X_input,
            model_type="mean",
        ),
        "Elastic Net": predict_baseline_scores(
            fitted_models["Elastic Net"],
            X_input,
            model_type="elastic_net",
        ),
        "XGBoost": predict_baseline_scores(
            fitted_models["XGBoost"],
            X_input,
            model_type="xgboost",
        ),
    }


# =============================================================================
# Inspection helper
# =============================================================================

def extract_xgb_feature_importance(
    xgb_pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Extract feature importance from a fitted XGBoost pipeline.
    """
    xgb_fitted = xgb_pipeline.named_steps["model"]
    xgb_preprocessor = xgb_pipeline.named_steps["preprocessor"]

    feature_names = xgb_preprocessor.get_feature_names_out()

    df_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": xgb_fitted.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return df_importance


# =============================================================================
# Full tabular suite
# =============================================================================

def fit_tabular_baselines(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    sample_weight: pd.Series | np.ndarray | None = None,
    config: Optional[BaselineTrainingConfig] = None,
    condition_name: str = "condition",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Fit all three non-neural baselines and aggregate their weak-distillation results.

    Returns
    -------
    fitted_models : dict[str, Any]
        Fitted model objects for Mean Predictor, Elastic Net, and XGBoost.
    df_results : pd.DataFrame
        One-row-per-model regression metric table on the provided validation set.
    """
    if config is None:
        config = BaselineTrainingConfig()

    fitted_models: dict[str, Any] = {}
    result_frames: list[pd.DataFrame] = []

    mean_model, _, df_mean = fit_mean_predictor(
        y_train=y_train,
        y_val=y_val,
        model_name=f"Mean Predictor_{condition_name}",
    )
    fitted_models["Mean Predictor"] = mean_model
    result_frames.append(df_mean)

    elastic_model, _, df_elastic = fit_elastic_net(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        config=config.elastic_net,
        sample_weight=sample_weight,
        add_numeric_missing_indicator=config.add_numeric_missing_indicator,
        scale_numeric=config.scale_numeric_for_elastic_net,
        model_name=f"Elastic Net_{condition_name}",
    )
    fitted_models["Elastic Net"] = elastic_model
    result_frames.append(df_elastic)

    xgb_model, _, df_xgb = fit_xgboost_regressor(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        config=config.xgboost,
        sample_weight=sample_weight,
        add_numeric_missing_indicator=config.add_numeric_missing_indicator,
        scale_numeric=config.scale_numeric_for_xgboost,
        model_name=f"XGBoost_{condition_name}",
    )
    fitted_models["XGBoost"] = xgb_model
    result_frames.append(df_xgb)

    df_results = pd.concat(result_frames, ignore_index=True)
    df_results["condition"] = condition_name

    return fitted_models, df_results