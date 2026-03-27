"""
plotting.py — Reusable plotting utilities for EDA and model evaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from master_thesis.config import FIGURES


#---------------------------------------
# Baseline plots
#---------------------------------------

def plot_baseline_metric(
    baseline_results: pd.DataFrame,
    metric: str = "mae",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """
    Plot a bar chart for a chosen baseline metric.
    """
    plot_df = baseline_results.sort_values(metric, ascending=True)

    plt.figure(figsize=figsize)
    plt.bar(plot_df["model"], plot_df[metric])
    plt.ylabel(f"Validation {metric.upper()}")
    plt.title(title if title is not None else "Baseline Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_predicted_vs_true(
    y_true,
    y_pred,
    model_name: str = "Model",
    figsize: tuple[int, int] = (6, 6),
    alpha: float = 0.4,
) -> None:
    """
    Scatter plot of predicted vs true values.
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=alpha)
    plt.xlabel("True renegotiation_prob")
    plt.ylabel("Predicted renegotiation_prob")
    plt.title(f"{model_name}: Predicted vs True")
    plt.tight_layout()
    plt.show()




