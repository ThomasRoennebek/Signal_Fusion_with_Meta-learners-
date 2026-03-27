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
)


# ── Regression ─────────────────────────────────────────────────────────────

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Compute standard regression metrics.

    Returns a single-row DataFrame with: model, rmse, mae, r2, pred_mean, true_mean.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return pd.DataFrame({
        "model":     [model_name],
        "rmse":      [mean_squared_error(y_true, y_pred) ** 0.5],
        "mae":       [mean_absolute_error(y_true, y_pred)],
        "r2":        [r2_score(y_true, y_pred)],
        "pred_mean": [float(np.mean(y_pred))],
        "true_mean": [float(np.mean(y_true))],
    })


# ── Classification ─────────────────────────────────────────────────────────

def classification_metrics(
    y_true,
    y_pred_proba,
    threshold: float = 0.5,
    model_name: str = "model",
    verbose: bool = True,
) -> dict:
    """
    Compute standard binary classification metrics.

    Returns a dict with: model, auc, ap, f1, accuracy.
    """
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    results = {
        "model":    model_name,
        "auc":      roc_auc_score(y_true, y_pred_proba),
        "ap":       average_precision_score(y_true, y_pred_proba),
        "f1":       f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if verbose:
        print(classification_report(y_true, y_pred))
    return results


# ── Back-compat alias ──────────────────────────────────────────────────────
evaluate = classification_metrics
