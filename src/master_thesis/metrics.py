"""
metrics.py — Evaluation metrics for regression, classification, and ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    classification_report,
    log_loss,
)


# ── Weak label distillation / regression evaluation ──────────────────

def evaluate_predictions(
    y_true,
    y_pred,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Evaluate continuous predictions against a continuous target
    such as Snorkel's renegotiation_prob.

    This should be interpreted as weak-label distillation quality,
    not real predictive validity.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

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


# ── Binary classification / gold label evaluation ──────────────────────────

def classification_metrics(
    y_true,
    y_pred_proba,
    threshold: float = 0.5,
    model_name: str = "model",
    verbose: bool = True,
) -> dict:
    """
    Compute standard binary classification metrics.

    Returns a dict with:
    - model
    - auc
    - ap
    - f1
    - accuracy
    """
    y_true = np.asarray(y_true).astype(int).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    results = {
        "model": model_name,
        "auc": roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "ap": average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    if verbose:
        print(classification_report(y_true, y_pred, zero_division=0))

    return results


def evaluate_on_gold_binary(
    y_true_gold,
    y_pred_prob,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Evaluate predicted probabilities on held-out gold binary labels.

    Recommended primary Stage 1 metric table for real model validity.
    """
    y_true_gold = np.asarray(y_true_gold).astype(int).flatten()
    y_pred_prob = np.asarray(y_pred_prob).astype(float).flatten()

    result = {
        "model": model_name,
        "gold_auroc": np.nan,
        "gold_ap": np.nan,
        "gold_logloss": np.nan,
        "gold_pred_mean": float(np.mean(y_pred_prob)),
        "gold_true_mean": float(np.mean(y_true_gold)),
    }

    if len(np.unique(y_true_gold)) > 1:
        result["gold_auroc"] = roc_auc_score(y_true_gold, y_pred_prob)
        result["gold_ap"] = average_precision_score(y_true_gold, y_pred_prob)

    result["gold_logloss"] = log_loss(
        y_true_gold,
        np.clip(y_pred_prob, 1e-8, 1 - 1e-8),
        labels=[0, 1],
    )

    return pd.DataFrame([result])


# ── Ranking metrics ────────────────────────────────────────────────────────

def precision_at_k(
    y_true,
    y_score,
    k: int = 10,
) -> float:
    """
    Precision@K for ranking tasks.
    """
    y_true = np.asarray(y_true).astype(int).flatten()
    y_score = np.asarray(y_score).flatten()

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
    y_true = np.asarray(y_true).astype(int).flatten()
    y_score = np.asarray(y_score).flatten()

    positives = np.sum(y_true)
    if positives == 0:
        return np.nan

    k = min(k, len(y_true))
    top_k_idx = np.argsort(-y_score)[:k]
    return float(np.sum(y_true[top_k_idx]) / positives)