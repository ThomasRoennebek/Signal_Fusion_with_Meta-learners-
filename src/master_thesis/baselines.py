"""
baselines.py — Stage 1 baseline models (XGBoost, ElasticNet, Mean Predictor).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "model",
) -> pd.DataFrame:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return pd.DataFrame(
        {
            "model": [model_name],
            "rmse": [rmse],
            "mae": [mae],
            "r2": [r2],
            "pred_mean": [np.mean(y_pred)],
            "true_mean": [np.mean(y_true)],
        }
    )


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

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


def fit_mean_predictor(
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    train_mean = np.mean(y_train)
    y_pred = np.full(shape=len(y_val), fill_value=train_mean)

    results = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        model_name="Mean Predictor",
    )
    return y_pred, results


def fit_elastic_net(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 10000,
    random_state: int = 42,
) -> tuple[Pipeline, np.ndarray, pd.DataFrame]:
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=True,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_val), 0, 1)

    results = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        model_name="Elastic Net",
    )

    return model, y_pred, results


def fit_xgboost_regressor(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_val: pd.Series | np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    min_child_weight: int = 3,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[Pipeline, np.ndarray, pd.DataFrame]:
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=False,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_val), 0, 1)

    results = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        model_name="XGBoost",
    )

    return model, y_pred, results


def extract_xgb_feature_importance(
    xgb_pipeline: Pipeline,
) -> pd.DataFrame:
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
