"""
metrics.py — Evaluation metrics for Stage 1 weak-label distillation,
gold-label binary classification, and ranking.

Design intent
-------------
This module separates evaluation into three explicit streams:

1) Weak-label distillation
   Soft target vs soft prediction evaluation, e.g. model predictions
   against Snorkel's renegotiation_prob.

2) Gold-label classification
   Real predictive validity on held-out human-validated binary labels.

3) Ranking
   Top-K prioritization quality for decision support, where identifying
   the most relevant contracts first is often more important than only
   optimizing a fixed threshold classification metric.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# =============================================================================
# Internal helpers
# =============================================================================

def _to_1d_array(x, dtype=None) -> np.ndarray:
    """
    Convert input to a flattened 1D numpy array.
    """
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr.flatten()


def _safe_binary_probabilities(y_pred_proba) -> np.ndarray:
    """
    Ensure probabilities are 1D float array clipped away from 0 and 1
    for numerically stable log-loss and calibration calculations.
    """
    y_pred_proba = _to_1d_array(y_pred_proba, dtype=float)
    return np.clip(y_pred_proba, 1e-8, 1 - 1e-8)


def _has_both_classes(y_true: np.ndarray) -> bool:
    """
    Check whether binary labels contain both 0 and 1.
    """
    return len(np.unique(y_true)) > 1


# =============================================================================
# Stage 1 — Weak label distillation / regression evaluation
# =============================================================================

def evaluate_predictions(
    y_true,
    y_pred,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Evaluate continuous predictions against a continuous target such as
    Snorkel's renegotiation_prob.

    Interpretation:
    These metrics measure weak-label distillation quality, not real predictive
    validity against gold labels.
    """
    y_true = _to_1d_array(y_true, dtype=float)
    y_pred = _to_1d_array(y_pred, dtype=float)

    return pd.DataFrame({
        "model": [model_name],
        "rmse": [mean_squared_error(y_true, y_pred) ** 0.5],
        "mae": [mean_absolute_error(y_true, y_pred)],
        "r2": [r2_score(y_true, y_pred)],
        "pred_mean": [float(np.mean(y_pred))],
        "true_mean": [float(np.mean(y_true))],
    })


def regression_metrics(
    y_true,
    y_pred,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Alias for evaluate_predictions().
    """
    return evaluate_predictions(y_true=y_true, y_pred=y_pred, model_name=model_name)


def weak_label_distillation_metrics(
    y_true_soft,
    y_pred_soft,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Explicit wrapper for Stage 1 weak-label distillation evaluation.

    Parameters
    ----------
    y_true_soft : array-like
        Soft targets, e.g. Snorkel LabelModel probabilities.
    y_pred_soft : array-like
        Predicted soft scores from the student model.
    model_name : str
        Name of the model.

    Returns
    -------
    pd.DataFrame
        Single-row metric table.
    """
    return evaluate_predictions(
        y_true=y_true_soft,
        y_pred=y_pred_soft,
        model_name=model_name,
    )


# =============================================================================
# Stage 1 / Stage 2 — Binary classification / gold label evaluation
# =============================================================================

def expected_calibration_error(
    y_true,
    y_pred_proba,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary classification.

    ECE bins predictions by confidence and compares average confidence
    to empirical accuracy in each bin.

    Returns np.nan if input is empty.
    """
    y_true = _to_1d_array(y_true, dtype=int)
    y_pred_proba = _safe_binary_probabilities(y_pred_proba)

    if len(y_true) == 0:
        return np.nan

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_pred_proba >= left) & (y_pred_proba <= right)
        else:
            mask = (y_pred_proba >= left) & (y_pred_proba < right)

        if not np.any(mask):
            continue

        bin_true = y_true[mask]
        bin_pred = y_pred_proba[mask]

        acc = np.mean(bin_true)
        conf = np.mean(bin_pred)
        ece += (len(bin_true) / n) * abs(acc - conf)

    return float(ece)


def calibration_table(
    y_true,
    y_pred_proba,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Create a reliability-table scaffold for calibration analysis.

    Useful for reliability diagrams in notebooks or reports.
    """
    y_true = _to_1d_array(y_true, dtype=int)
    y_pred_proba = _safe_binary_probabilities(y_pred_proba)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_pred_proba >= left) & (y_pred_proba <= right)
        else:
            mask = (y_pred_proba >= left) & (y_pred_proba < right)

        count = int(np.sum(mask))

        if count == 0:
            rows.append({
                "bin": i + 1,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": 0,
                "mean_pred": np.nan,
                "event_rate": np.nan,
                "abs_gap": np.nan,
            })
            continue

        mean_pred = float(np.mean(y_pred_proba[mask]))
        event_rate = float(np.mean(y_true[mask]))

        rows.append({
            "bin": i + 1,
            "bin_left": float(left),
            "bin_right": float(right),
            "count": count,
            "mean_pred": mean_pred,
            "event_rate": event_rate,
            "abs_gap": abs(mean_pred - event_rate),
        })

    return pd.DataFrame(rows)


def classification_metrics(
    y_true,
    y_pred_proba,
    threshold: float = 0.5,
    model_name: str = "model",
    verbose: bool = True,
) -> dict:
    """
    Compute standard binary classification metrics.

    Returns
    -------
    dict
        Dictionary with classification metrics.
    """
    y_true = _to_1d_array(y_true, dtype=int)
    y_pred_proba = _safe_binary_probabilities(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    results = {
        "model": model_name,
        "auc": roc_auc_score(y_true, y_pred_proba) if _has_both_classes(y_true) else np.nan,
        "ap": average_precision_score(y_true, y_pred_proba) if _has_both_classes(y_true) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_pred_proba),
        "ece": expected_calibration_error(y_true, y_pred_proba),
    }

    if verbose:
        print(classification_report(y_true, y_pred, zero_division=0))

    return results


def evaluate_on_gold_binary(
    y_true_gold,
    y_pred_prob,
    model_name: str = "model",
    threshold: float = 0.5,
    k_values: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Evaluate predicted probabilities on held-out gold binary labels.

    Recommended Stage 1 and Stage 2 gold-label metric table for real model
    validity.

    Includes:
    - AUROC
    - Average Precision
    - Log-loss
    - Brier score
    - ECE
    - Thresholded F1 / accuracy
    - Ranking quality at selected K values
    """
    y_true_gold = _to_1d_array(y_true_gold, dtype=int)
    y_pred_prob = _safe_binary_probabilities(y_pred_prob)
    y_pred = (y_pred_prob >= threshold).astype(int)

    result = {
        "model": model_name,
        "gold_auroc": np.nan,
        "gold_ap": np.nan,
        "gold_logloss": np.nan,
        "gold_brier": np.nan,
        "gold_ece": np.nan,
        "gold_f1": np.nan,
        "gold_accuracy": np.nan,
        "gold_pred_mean": float(np.mean(y_pred_prob)),
        "gold_true_mean": float(np.mean(y_true_gold)),
    }

    if _has_both_classes(y_true_gold):
        result["gold_auroc"] = roc_auc_score(y_true_gold, y_pred_prob)
        result["gold_ap"] = average_precision_score(y_true_gold, y_pred_prob)

    result["gold_logloss"] = log_loss(
        y_true_gold,
        y_pred_prob,
        labels=[0, 1],
    )
    result["gold_brier"] = brier_score_loss(y_true_gold, y_pred_prob)
    result["gold_ece"] = expected_calibration_error(y_true_gold, y_pred_prob)
    result["gold_f1"] = f1_score(y_true_gold, y_pred, zero_division=0)
    result["gold_accuracy"] = accuracy_score(y_true_gold, y_pred)

    for k in k_values:
        result[f"precision_at_{k}"] = precision_at_k(y_true_gold, y_pred_prob, k=k)
        result[f"recall_at_{k}"] = recall_at_k(y_true_gold, y_pred_prob, k=k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(y_true_gold, y_pred_prob, k=k)

    return pd.DataFrame([result])


def gold_label_classification_metrics(
    y_true_gold,
    y_pred_proba,
    model_name: str = "model",
    threshold: float = 0.5,
    k_values: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Explicit wrapper for gold label binary evaluation.
    """
    return evaluate_on_gold_binary(
        y_true_gold=y_true_gold,
        y_pred_prob=y_pred_proba,
        model_name=model_name,
        threshold=threshold,
        k_values=k_values,
    )


# =============================================================================
# Ranking metrics
# =============================================================================

def precision_at_k(
    y_true,
    y_score,
    k: int = 10,
) -> float:
    """
    Precision@K for ranking tasks.
    """
    y_true = _to_1d_array(y_true, dtype=int)
    y_score = _to_1d_array(y_score, dtype=float)

    if len(y_true) == 0:
        return np.nan

    k = min(k, len(y_true))
    top_k_idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[top_k_idx]))


def recall_at_k(
    y_true,
    y_score,
    k: int = 10,
) -> float:
    """
    Recall@K for ranking tasks.
    """
    y_true = _to_1d_array(y_true, dtype=int)
    y_score = _to_1d_array(y_score, dtype=float)

    positives = np.sum(y_true)
    if positives == 0:
        return np.nan

    k = min(k, len(y_true))
    top_k_idx = np.argsort(-y_score)[:k]
    return float(np.sum(y_true[top_k_idx]) / positives)


def dcg_at_k(
    y_true,
    y_score,
    k: int = 10,
) -> float:
    """
    Discounted Cumulative Gain (DCG)@K for binary relevance.
    """
    y_true = _to_1d_array(y_true, dtype=int)
    y_score = _to_1d_array(y_score, dtype=float)

    if len(y_true) == 0:
        return np.nan

    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(
    y_true,
    y_score,
    k: int = 10,
) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG)@K.

    More informative than Precision@K when correct hits near the very top
    of the ranking should matter more than lower ranked hits.
    """
    y_true = _to_1d_array(y_true, dtype=int)
    y_score = _to_1d_array(y_score, dtype=float)

    if len(y_true) == 0:
        return np.nan

    actual_dcg = dcg_at_k(y_true, y_score, k=k)

    ideal_scores = y_true.astype(float)
    ideal_dcg = dcg_at_k(y_true, ideal_scores, k=k)

    if ideal_dcg == 0:
        return np.nan

    return float(actual_dcg / ideal_dcg)


def map_score(
    y_true,
    y_score,
) -> float:
    """
    Mean Average Precision for a single ranked list.

    For a single ranking task, this is numerically equivalent to
    Average Precision.
    """
    y_true = _to_1d_array(y_true, dtype=int)
    y_score = _to_1d_array(y_score, dtype=float)

    if len(y_true) == 0 or not _has_both_classes(y_true):
        return np.nan

    return float(average_precision_score(y_true, y_score))


def ranking_metrics(
    y_true_gold,
    y_score,
    model_name: str = "model",
    k_values: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Explicit ranking evaluation API.

    Parameters
    ----------
    y_true_gold : array-like
        Binary gold labels.
    y_score : array-like
        Ranking scores or probabilities.
    model_name : str
        Model name.
    k_values : sequence of int
        K values for top-K ranking metrics.

    Returns
    -------
    pd.DataFrame
        Single-row ranking metric table.
    """
    y_true_gold = _to_1d_array(y_true_gold, dtype=int)
    y_score = _to_1d_array(y_score, dtype=float)

    result = {
        "model": model_name,
        "map": map_score(y_true_gold, y_score),
        "ranking_ap": average_precision_score(y_true_gold, y_score) if _has_both_classes(y_true_gold) else np.nan,
    }

    for k in k_values:
        result[f"precision_at_{k}"] = precision_at_k(y_true_gold, y_score, k=k)
        result[f"recall_at_{k}"] = recall_at_k(y_true_gold, y_score, k=k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(y_true_gold, y_score, k=k)

    return pd.DataFrame([result])


# =============================================================================
# per task aggregation for Stage 2
# =============================================================================

def evaluate_grouped_gold_binary(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    y_pred_col: str,
    threshold: float = 0.5,
    k_values: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Evaluate gold label binary performance per task/group  per department.

   Stage 2 meta-learning evaluation with both:
    - per-task metrics
    - aggregate metrics across tasks

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing group labels, gold labels, and prediction scores.
    group_col : str
        Column identifying the task/group, e.g. department.
    y_true_col : str
        Column with binary gold labels.
    y_pred_col : str
        Column with predicted probabilities or ranking scores.
    threshold : float
        Threshold for threshold-dependent metrics.
    k_values : sequence of int
        K values for ranking metrics.

    Returns
    -------
    pd.DataFrame
        One row per group.
    """
    rows = []

    for group_name, group_df in df.groupby(group_col):
        metrics_df = evaluate_on_gold_binary(
            y_true_gold=group_df[y_true_col].values,
            y_pred_prob=group_df[y_pred_col].values,
            model_name=str(group_name),
            threshold=threshold,
            k_values=k_values,
        )
        metrics_df.insert(0, group_col, group_name)
        rows.append(metrics_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def summarize_grouped_metrics(
    grouped_metrics_df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Aggregate numeric grouped metrics across tasks/groups.

    Returns mean, std, min, and max across groups for thesis reporting.
    """
    if grouped_metrics_df.empty:
        return pd.DataFrame()

    numeric_cols = grouped_metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    summary = grouped_metrics_df[numeric_cols].agg(["mean", "std", "min", "max"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})
    summary.insert(0, "grouping", group_col)
    return summary