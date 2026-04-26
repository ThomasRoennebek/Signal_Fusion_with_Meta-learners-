"""
stage1.py — High-level orchestration for training all Stage 1 models together.

Purpose
-------
This module sits above baselines.py, mlp.py, data_utils.py, and metrics.py.
It is responsible for:
- fitting all four Stage 1 models under one condition
- predicting with all four models
- evaluating weak-label distillation and gold-label validity
- saving the Stage 1 MLP as a pretrained artifact for Stage 2
- saving per-condition metadata for reproducibility

Stage 1 Conditions
------------------
Four training conditions isolate the contribution of each supervision source:

  A — Weak-only (GLOBAL)
      Target : Snorkel renegotiation_prob (soft, 0-1) for every contract.
      Weight : None (uniform).
      Scope  : Global — independent of the target department.
      Purpose: Main pre-training path; principal Stage 2 initialisation.

  B — Gold-only (TARGET-AWARE)
      Target : Hard binary gold_y for non-target-department contracts only.
      Weight : None (uniform).
      Scope  : Target-specific — excludes the current target department.
      Purpose: Ablation — how far scarce gold supervision goes alone.

  C — Hybrid weighted (TARGET-AWARE)
      Target : Concatenation of weak labels (all rows) + gold labels
               (non-target-department rows).  C and D share the same
               X_train and y_train; they differ only in sample_weight.
      Weight : weak rows -> HYBRID_WEAK_WEIGHT (e.g. 1.0),
               gold rows -> HYBRID_GOLD_WEIGHT (e.g. 5.0).
      Scope  : Target-specific — gold rows exclude the target department.
      Purpose: Ablation — does explicit weighting of gold signal help?

  D — Hybrid unweighted (TARGET-AWARE)
      Target : Identical concatenation to C (same row-level target logic).
      Weight : None (uniform, no WeightedRandomSampler).
      Scope  : Target-specific — same exclusion rule as C.
      Purpose: Naive baseline — does simply pooling all labels work?

KEY POINT: C and D use the same mixed target.  The difference between them
is the *weighting scheme*, not a different conceptual target definition.

Artifact path convention
------------------------
  models/stage_1/global/A_weak_only/        — condition A (reusable)
  models/stage_1/{target_dept}/B_gold_only/  — condition B
  models/stage_1/{target_dept}/C_hybrid/     — condition C
  models/stage_1/{target_dept}/D_hybrid_unweighted/ — condition D

This layout prevents silent overwrites when Stage 1 is re-run for a
different target department.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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

    print(f"\n{'='*50}")
    print(f"========== Starting Condition: {condition_name} ==========")
    print(f"{'='*50}")
    
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

    print(f"[{condition_name}] Fitting PyTorch MLP...")
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
    print(f"[{condition_name}] PyTorch MLP fitting complete!")

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


def make_stage1_predictions_df(
    trained_bundle: dict,
    X_eval: pd.DataFrame,
    y_true,
    condition_name: Optional[str] = None,
    index_col: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Return a long-format DataFrame of per-sample predicted probabilities.

    One row per (sample, model) combination, suitable for saving as
    ``predictions.csv`` and for later calibration or threshold analysis.

    Columns
    -------
    - ``condition``    : training condition name
    - ``model``        : model name (MeanPredictor, ElasticNet, XGBoost, MLP)
    - ``sample_index`` : integer position in X_eval (or values from index_col)
    - ``y_true``       : true target value (weak-label float or gold-label 0/1)
    - ``y_pred``       : predicted probability score

    Parameters
    ----------
    trained_bundle : dict
        Bundle returned by fit_all_stage1_models().
    X_eval : pd.DataFrame
        Evaluation feature matrix.
    y_true : array-like
        Ground-truth labels (weak soft labels or binary gold labels).
    condition_name : str or None
        Condition label; defaults to trained_bundle["condition_name"].
    index_col : pd.Series or None
        Optional series of contract IDs or other identifiers to use instead
        of integer position.  Must have the same length as X_eval.
    """
    if condition_name is None:
        condition_name = trained_bundle["condition_name"]

    y_true = np.asarray(y_true)
    preds = predict_all_stage1_models(trained_bundle, X_eval)

    rows = []
    sample_index = (
        index_col.values if index_col is not None
        else np.arange(len(X_eval))
    )
    for model_name, y_pred in preds.items():
        for i, (idx, yt, yp) in enumerate(zip(sample_index, y_true, y_pred)):
            rows.append({
                "condition": condition_name,
                "model": model_name,
                "sample_index": idx,
                "y_true": float(yt),
                "y_pred": float(yp),
            })

    return pd.DataFrame(rows)


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

def save_stage1_gold_results(
    df_gold_results: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Save Stage 1 gold-label evaluation results as a persistent artifact.

    The file is written to ``{output_dir}/gold_results.csv`` alongside the
    other Stage 1 condition artifacts.  One row per model per condition; columns
    include AUROC, Log-Loss, F1, NDCG@k, ECE, and condition metadata.

    Parameters
    ----------
    df_gold_results : pd.DataFrame
        DataFrame returned by evaluate_stage1_on_gold_labels() or
        evaluate_stage1_conditions()["df_gold_results_all"].
    output_dir : str | Path
        Destination directory (same directory as tabular_results.csv).

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "gold_results.csv"
    df_gold_results.to_csv(gold_path, index=False)
    return gold_path


def save_stage1_feature_names(
    feature_columns: list[str],
    output_dir: str | Path,
) -> Path:
    """
    Save the input feature names used for a Stage 1 condition.

    The file ``feature_names.json`` records the ordered list of column names
    that were passed to every model in the condition.  This is the Stage 2
    handoff artifact: the Stage 2 preprocessor must receive data in the same
    feature space to produce embeddings that are compatible with the Stage 1
    MLP weights.

    Parameters
    ----------
    feature_columns : list[str]
        Ordered list of input feature column names.
    output_dir : str | Path
        Condition-specific artifact directory.

    Returns
    -------
    Path
        Path to the saved feature_names.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_features": len(feature_columns),
        "feature_names": feature_columns,
    }
    path = output_dir / "feature_names.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def save_stage1_condition_artifacts(
    trained_bundle: dict,
    output_dir: str | Path,
    save_xgb_importance: bool = False,
    df_gold_results: Optional[pd.DataFrame] = None,
    df_predictions: Optional[pd.DataFrame] = None,
) -> dict[str, Path]:
    """
    Save the main artifacts for one trained Stage 1 condition.

    Always saves
    ------------
    - pretrained MLP bundle (weights, preprocessor, history)
    - tabular weak-distillation result table (tabular_results.csv)
    - feature_names.json (ordered input feature list for Stage 2 handoff)

    Saves when provided
    -------------------
    - per-sample validation predictions (predictions.csv)
      Pass the DataFrame from make_stage1_predictions_df().
    - gold-label evaluation table (gold_results.csv)
      Pass the DataFrame from evaluate_stage1_on_gold_labels().
    - XGBoost feature importance (xgboost_feature_importance.csv)
      Set save_xgb_importance=True.

    Parameters
    ----------
    trained_bundle : dict
        Bundle returned by fit_all_stage1_models().
    output_dir : str | Path
        Destination directory, e.g. models/stage_1/global/A_weak_only/.
    save_xgb_importance : bool
        Whether to also save XGBoost feature importances.
    df_gold_results : pd.DataFrame or None
        Gold-label evaluation results; saved as gold_results.csv.
    df_predictions : pd.DataFrame or None
        Long-format per-sample predictions; saved as predictions.csv.
        Produced by make_stage1_predictions_df().
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

    # Always save feature names — critical for Stage 2 handoff.
    feature_names_path = save_stage1_feature_names(
        feature_columns=trained_bundle["feature_columns"],
        output_dir=output_dir,
    )
    saved_paths["feature_names_path"] = feature_names_path

    if df_predictions is not None:
        predictions_path = output_dir / "predictions.csv"
        df_predictions.to_csv(predictions_path, index=False)
        saved_paths["predictions_path"] = predictions_path

    if df_gold_results is not None:
        gold_results_path = save_stage1_gold_results(df_gold_results, output_dir)
        saved_paths["gold_results_path"] = gold_results_path

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

    Returns
    -------
    dict with keys:
    - ``df_weak_results_all``: combined weak-label metrics across all conditions
    - ``df_gold_results_all``: combined gold-label metrics across all conditions
    - ``df_gold_results_by_condition``: dict mapping condition_name to its gold DataFrame
      (convenient for passing per-condition results to save_stage1_condition_artifacts)
    """
    df_weak_all = []
    df_gold_all = []
    df_gold_by_condition: dict[str, pd.DataFrame] = {}

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
        df_gold_by_condition[condition_name] = df_gold_results

    return {
        "df_weak_results_all": pd.concat(df_weak_all, ignore_index=True),
        "df_gold_results_all": pd.concat(df_gold_all, ignore_index=True),
        "df_gold_results_by_condition": df_gold_by_condition,
    }


def save_all_stage1_gold_results(
    df_gold_results_by_condition: dict[str, pd.DataFrame],
    base_output_dir: str | Path,
) -> dict[str, Path]:
    """
    Persist gold evaluation results for all conditions at once.

    Writes ``gold_results.csv`` into each condition's subdirectory under
    ``base_output_dir``.  Call this after evaluate_stage1_conditions() using
    the ``df_gold_results_by_condition`` key from the returned dict.

    Parameters
    ----------
    df_gold_results_by_condition : dict
        Mapping of condition_name -> gold results DataFrame, as returned by
        evaluate_stage1_conditions()["df_gold_results_by_condition"].
    base_output_dir : str | Path
        Root Stage 1 models directory, e.g. ``models/stage_1/``.

    Returns
    -------
    dict mapping condition_name -> saved gold_results.csv path.
    """
    base_output_dir = Path(base_output_dir)
    saved: dict[str, Path] = {}
    for condition_name, df_gold in df_gold_results_by_condition.items():
        condition_dir = base_output_dir / condition_name
        gold_path = save_stage1_gold_results(df_gold, condition_dir)
        saved[condition_name] = gold_path
        print(f"[{condition_name}] Saved gold_results.csv -> {gold_path}")
    return saved


def save_stage1_condition_summary(
    df_gold_results_all: pd.DataFrame,
    output_path: str | Path,
    metric_cols: tuple[str, ...] = (
        "gold_auroc",
        "gold_logloss",
        "gold_f1",
        "gold_ece",
        "gold_ap",
        "gold_brier",
        "gold_accuracy",
    ),
) -> Path:
    """
    Save a cross-condition summary table as condition_summary.csv.

    Aggregates gold-label evaluation metrics across conditions and models,
    pivoted so that each row is one (condition, model) pair with one column
    per metric.  This is the table that feeds thesis comparison figures and
    the cross-condition leaderboard in the analysis dashboard.

    Parameters
    ----------
    df_gold_results_all : pd.DataFrame
        Combined gold results for all conditions, as returned by
        evaluate_stage1_conditions()["df_gold_results_all"].
    output_path : str | Path
        Full file path to write, e.g. ``models/stage_1/condition_summary.csv``.
    metric_cols : tuple[str, ...]
        Which metric columns to include in the summary.  Must match the
        column names produced by evaluate_on_gold_binary() — i.e. prefixed
        with ``gold_`` (gold_auroc, gold_logloss, gold_f1, gold_ece, etc.).

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available = [c for c in metric_cols if c in df_gold_results_all.columns]
    group_cols = [c for c in ("condition", "model_base") if c in df_gold_results_all.columns]

    if not group_cols or not available:
        df_summary = df_gold_results_all.copy()
    else:
        df_summary = (
            df_gold_results_all
            .groupby(group_cols, sort=False)[available]
            .mean()
            .reset_index()
        )

    df_summary.to_csv(output_path, index=False)
    print(f"Saved condition_summary.csv -> {output_path}")
    return output_path


# =============================================================================
# Target-aware artifact path helpers
# =============================================================================

# Conditions that are global (independent of target department).
GLOBAL_CONDITIONS = {"A_weak_only"}


def get_stage1_condition_dir(
    base_dir: str | Path,
    condition_name: str,
    target_department: str = "Logistics",
) -> Path:
    """
    Return the canonical artifact directory for a Stage 1 condition.

    Layout
    ------
    - A_weak_only:          ``{base_dir}/global/A_weak_only/``
    - B/C/D (target-aware): ``{base_dir}/{target_department}/B_gold_only/``

    This layout prevents artifact overwrites when Stage 1 is re-run for a
    different target department.
    """
    base_dir = Path(base_dir)
    if condition_name in GLOBAL_CONDITIONS:
        return base_dir / "global" / condition_name
    return base_dir / target_department / condition_name


# =============================================================================
# Per-condition metadata
# =============================================================================

def save_stage1_metadata(
    output_dir: str | Path,
    condition_name: str,
    target_department: str,
    feature_count: int,
    config_snapshot: dict,
    model_families: tuple[str, ...] = (
        "MeanPredictor", "ElasticNet", "XGBoost", "MLP",
    ),
    extra: Optional[dict] = None,
) -> Path:
    """
    Save a self-describing metadata.json for one Stage 1 condition.

    This file records the provenance of every artifact in the directory,
    making the Stage 1 output fully reproducible and easy to audit.

    Parameters
    ----------
    output_dir : str | Path
        Condition-specific artifact directory.
    condition_name : str
        One of A_weak_only, B_gold_only, C_hybrid, D_hybrid_unweighted.
    target_department : str
        The department whose gold labels were excluded (or "global").
    feature_count : int
        Number of input features used for training.
    config_snapshot : dict
        Serialisable copy of the experiment config.
    model_families : tuple[str, ...]
        Names of the model families trained in this condition.
    extra : dict or None
        Any additional key-value pairs to include.

    Returns
    -------
    Path
        Path to the saved metadata.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_global = condition_name in GLOBAL_CONDITIONS
    exclusion_rule = (
        "none (global condition, no department excluded)"
        if is_global
        else f"gold labels from '{target_department}' excluded from training"
    )

    metadata = {
        "condition": condition_name,
        "target_department": "global" if is_global else target_department,
        "is_global": is_global,
        "exclusion_rule": exclusion_rule,
        "feature_count": feature_count,
        "model_families": list(model_families),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config_snapshot,
    }

    # Include MLP architecture details if present in the config.
    mlp_cfg = config_snapshot.get("mlp_config", {})
    if mlp_cfg:
        metadata["mlp_architecture"] = {
            "hidden_dim_1": mlp_cfg.get("hidden_dim_1"),
            "hidden_dim_2": mlp_cfg.get("hidden_dim_2"),
            "dropout": mlp_cfg.get("dropout"),
        }

    if extra:
        metadata.update(extra)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata_path