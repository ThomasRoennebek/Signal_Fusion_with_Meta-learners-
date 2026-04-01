"""
stage1.py — High-level orchestration for training all Stage 1 models together.

Purpose
-------
This module sits above:
- baselines.py
- mlp.py
- data_utils.py
- metrics.py

It is responsible for:
- fitting all four Stage 1 models under one condition
- predicting with all four models
- evaluating weak-label distillation and gold-label validity
- saving the Stage 1 MLP as a pretrained artifact for Stage 2
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from master_thesis.baselines import (
    BaselineTrainingConfig,
    build_preprocessor,
    fit_tabular_baselines,
    get_numeric_and_categorical_columns,
    predict_tabular_baselines,
)
from master_thesis.metrics import (
    evaluate_on_gold_binary,
    evaluate_predictions,
)
from master_thesis.mlp import (
    MLPTrainingConfig,
    fit_mlp_model,
    predict_mlp_bundle,
)


# =============================================================================
# Config
# =============================================================================

@dataclass
class Stage1ModelConfig:
    baseline_config: BaselineTrainingConfig = field(default_factory=BaselineTrainingConfig)
    mlp_config: MLPTrainingConfig = field(default_factory=MLPTrainingConfig)


# =============================================================================
# Core training
# =============================================================================

def fit_all_stage1_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train,
    y_val,
    sample_weight=None,
    condition_name: str = "condition",
    config: Optional[Stage1ModelConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train all four Stage 1 models for one condition.

    Models
    ------
    - Mean Predictor
    - Elastic Net
    - XGBoost
    - MLP

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_train : array-like
        Training target.
    y_val : array-like
        Validation target.
    sample_weight : array-like or None
        Optional sample weights, e.g. for hybrid training.
    condition_name : str
        Label for the current training condition.
    config : Stage1ModelConfig or None
        Combined baseline + MLP config.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print MLP training logs.

    Returns
    -------
    dict
        Bundle containing trained models, configs, and weak-distillation results.
    """
    if config is None:
        config = Stage1ModelConfig()

    numeric_cols, categorical_cols = get_numeric_and_categorical_columns(X_train)

    y_train = np.asarray(y_train).astype(float)
    y_val = np.asarray(y_val).astype(float)

    df_tabular_models, df_tabular_results = fit_tabular_baselines(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        sample_weight=sample_weight,
        config=config.baseline_config,
        condition_name=condition_name,
    )

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=True,
        add_numeric_missing_indicator=True,
    )

    mlp_bundle = fit_mlp_model(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        preprocessor=preprocessor,
        sample_weight=sample_weight,
        config=config.mlp_config,
        seed=seed,
        verbose=verbose,
    )

    return {
        "condition_name": condition_name,
        "tabular_models": df_tabular_models,
        "tabular_results": df_tabular_results,
        "mlp_bundle": mlp_bundle,
        "feature_columns": X_train.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "config": {
            "baseline_config": asdict(config.baseline_config),
            "mlp_config": asdict(config.mlp_config),
        },
    }


# =============================================================================
# Prediction
# =============================================================================

def predict_all_stage1_models(
    trained_bundle: dict,
    X_input: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Predict scores from all four Stage 1 models at once.
    """
    tabular_preds = predict_tabular_baselines(
        fitted_models=trained_bundle["tabular_models"],
        X_input=X_input,
    )

    mlp_preds = predict_mlp_bundle(
        mlp_bundle=trained_bundle["mlp_bundle"],
        X_input=X_input,
    )

    all_preds = dict(tabular_preds)
    all_preds["MLP"] = mlp_preds
    return all_preds


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_stage1_on_weak_labels(
    trained_bundle: dict,
    X_eval: pd.DataFrame,
    y_true_soft,
    condition_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evaluate all four models on weak labels.

    This is weak-label distillation only, not real predictive validity.
    """
    if condition_name is None:
        condition_name = trained_bundle["condition_name"]

    y_true_soft = np.asarray(y_true_soft).astype(float)
    preds = predict_all_stage1_models(trained_bundle, X_eval)

    df_results = []
    for model_name, y_pred in preds.items():
        df_result = evaluate_predictions(
            y_true=y_true_soft,
            y_pred=y_pred,
            model_name=f"{model_name}_{condition_name}",
        )
        df_result["condition"] = condition_name
        df_result["model_base"] = model_name
        df_results.append(df_result)

    return pd.concat(df_results, ignore_index=True)


def evaluate_stage1_on_gold_labels(
    trained_bundle: dict,
    X_eval: pd.DataFrame,
    y_true_gold,
    condition_name: Optional[str] = None,
    threshold: float = 0.5,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Evaluate all four models on gold labels.

    This is the primary Stage 1 evaluation layer.
    """
    if condition_name is None:
        condition_name = trained_bundle["condition_name"]

    y_true_gold = np.asarray(y_true_gold).astype(int)
    preds = predict_all_stage1_models(trained_bundle, X_eval)

    df_results = []
    for model_name, y_pred in preds.items():
        df_result = evaluate_on_gold_binary(
            y_true_gold=y_true_gold,
            y_pred_prob=y_pred,
            model_name=f"{model_name}_{condition_name}",
            threshold=threshold,
            k_values=k_values,
        )
        df_result["condition"] = condition_name
        df_result["model_base"] = model_name
        df_results.append(df_result)

    return pd.concat(df_results, ignore_index=True)


def evaluate_stage1_bundle(
    trained_bundle: dict,
    X_weak_eval: pd.DataFrame,
    y_weak_eval,
    X_gold_eval: pd.DataFrame,
    y_gold_eval,
    threshold: float = 0.5,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, pd.DataFrame]:
    """
    Evaluate one trained Stage 1 bundle on both:
    - weak-label distillation
    - gold-label validity
    """
    df_weak_results = evaluate_stage1_on_weak_labels(
        trained_bundle=trained_bundle,
        X_eval=X_weak_eval,
        y_true_soft=y_weak_eval,
    )

    df_gold_results = evaluate_stage1_on_gold_labels(
        trained_bundle=trained_bundle,
        X_eval=X_gold_eval,
        y_true_gold=y_gold_eval,
        threshold=threshold,
        k_values=k_values,
    )

    return {
        "df_weak_results": df_weak_results,
        "df_gold_results": df_gold_results,
    }


# =============================================================================
# Saving pretrained MLP artifacts
# =============================================================================

def save_pretrained_mlp_bundle(
    mlp_bundle: dict,
    output_dir: str | Path,
    stem: str = "mlp_pretrained",
) -> dict[str, Path]:
    """
    Save Stage 1 MLP artifacts for Stage 2 reuse.

    Saves
    -----
    - model weights (.pt)
    - fitted preprocessor (.joblib)
    - training history (.csv)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / f"{stem}.pt"
    preprocessor_path = output_dir / f"{stem}_preprocessor.joblib"
    history_path = output_dir / f"{stem}_history.csv"

    torch.save(mlp_bundle["model"].state_dict(), weights_path)
    joblib.dump(mlp_bundle["preprocessor"], preprocessor_path)
    mlp_bundle["history"].to_csv(history_path, index=False)

    return {
        "weights_path": weights_path,
        "preprocessor_path": preprocessor_path,
        "history_path": history_path,
    }


# =============================================================================
# Saving whole Stage 1 condition bundle
# =============================================================================

def save_stage1_condition_artifacts(
    trained_bundle: dict,
    output_dir: str | Path,
    save_xgb_importance: bool = False,
) -> dict[str, Path]:
    """
    Save the main artifacts for one trained Stage 1 condition.

    Saves
    -----
    - pretrained MLP bundle
    - tabular weak-distillation result table

    Optionally also saves:
    - XGBoost feature importance
    """
    from master_thesis.baselines import extract_xgb_feature_importance

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    mlp_paths = save_pretrained_mlp_bundle(
        mlp_bundle=trained_bundle["mlp_bundle"],
        output_dir=output_dir,
        stem="mlp_pretrained",
    )
    saved_paths.update(mlp_paths)

    tabular_results_path = output_dir / "tabular_results.csv"
    trained_bundle["tabular_results"].to_csv(tabular_results_path, index=False)
    saved_paths["tabular_results_path"] = tabular_results_path

    if save_xgb_importance:
        xgb_model = trained_bundle["tabular_models"]["XGBoost"]
        df_importance = extract_xgb_feature_importance(xgb_model)

        importance_path = output_dir / "xgboost_feature_importance.csv"
        df_importance.to_csv(importance_path, index=False)
        saved_paths["xgboost_feature_importance_path"] = importance_path

    return saved_paths


# =============================================================================
# Multi-condition runner
# =============================================================================

def fit_stage1_conditions(
    conditions: dict[str, dict],
    X_val: pd.DataFrame,
    y_val,
    config: Optional[Stage1ModelConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Fit multiple Stage 1 conditions in one call.

    Expected condition payload format
    -------------------------------
    {
        "A_weak_only": {
            "X_train_df": ...,
            "y_train": ...,
            "sample_weight": None,
        },
        ...
    }
    """
    trained_bundles: dict[str, dict] = {}

    for condition_name, payload in conditions.items():
        trained_bundles[condition_name] = fit_all_stage1_models(
            X_train=payload["X_train_df"],
            X_val=X_val,
            y_train=payload["y_train"],
            y_val=y_val,
            sample_weight=payload.get("sample_weight"),
            condition_name=condition_name,
            config=config,
            seed=seed,
            verbose=verbose,
        )

    return trained_bundles


def evaluate_stage1_conditions(
    trained_bundles: dict[str, dict],
    X_weak_eval: pd.DataFrame,
    y_weak_eval,
    X_gold_eval: pd.DataFrame,
    y_gold_eval,
    threshold: float = 0.5,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, pd.DataFrame]:
    """
    Evaluate multiple Stage 1 conditions and aggregate results.
    """
    df_weak_all = []
    df_gold_all = []

    for condition_name, trained_bundle in trained_bundles.items():
        df_weak_results = evaluate_stage1_on_weak_labels(
            trained_bundle=trained_bundle,
            X_eval=X_weak_eval,
            y_true_soft=y_weak_eval,
            condition_name=condition_name,
        )
        df_gold_results = evaluate_stage1_on_gold_labels(
            trained_bundle=trained_bundle,
            X_eval=X_gold_eval,
            y_true_gold=y_gold_eval,
            condition_name=condition_name,
            threshold=threshold,
            k_values=k_values,
        )

        df_weak_all.append(df_weak_results)
        df_gold_all.append(df_gold_results)

    return {
        "df_weak_results_all": pd.concat(df_weak_all, ignore_index=True),
        "df_gold_results_all": pd.concat(df_gold_all, ignore_index=True),
    }