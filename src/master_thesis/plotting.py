"""
plotting.py — Reusable plotting utilities for EDA and model evaluation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from master_thesis.config import FIGURES


# ---------------------------------------
# Generic save helper
# ---------------------------------------

def save_fig(
    fig,
    filename: str,
    output_dir: Path | None = None,
    dpi: int = 300,
) -> Path:
    """
    Save a matplotlib figure to disk.
    """
    if output_dir is None:
        output_dir = FIGURES

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    print(f"Figure saved to: {output_path}")
    return output_path


# ---------------------------------------
# Baseline plots
# ---------------------------------------

def plot_baseline_metric(
    baseline_results: pd.DataFrame,
    metric: str = "mae",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """
    Plot a bar chart for a chosen baseline metric.
    """
    plot_df = baseline_results.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(plot_df["model"], plot_df[metric])
    ax.set_ylabel(f"Validation {metric.upper()}")
    ax.set_title(title if title is not None else "Baseline Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()

    return fig, ax


def plot_predicted_vs_true(
    y_true,
    y_pred,
    model_name: str = "Model",
    figsize: tuple[int, int] = (6, 6),
    alpha: float = 0.4,
):
    """
    Scatter plot of predicted vs true values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=alpha)
    ax.set_xlabel("True renegotiation_prob")
    ax.set_ylabel("Predicted renegotiation_prob")
    ax.set_title(f"{model_name}: Predicted vs True")
    plt.tight_layout()

    return fig, ax


