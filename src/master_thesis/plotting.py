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


# ---------------------------------------
# Feature importance plots
# ---------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot top-N feature importances as a horizontal bar chart.
    """
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    return fig, ax


# ---------------------------------------
# Label distribution plots
# ---------------------------------------

def plot_label_distribution_by_department(
    df: pd.DataFrame,
    department_col: str = "department",
    label_col: str = "target_renegotiate",
    contract_id_col: str = "contract_id",
    title: str = "Label Distribution by Department",
    figsize: tuple[int, int] = (14, 6),
):
    """
    Plot label counts by department using one row per unique contract.
    """
    plot_df = (
        df[[contract_id_col, department_col, label_col]]
        .drop_duplicates(subset=[contract_id_col])
        .copy()
    )

    plot_df[department_col] = plot_df[department_col].fillna("Unknown")

    counts = (
        plot_df.groupby([department_col, label_col])
        .size()
        .unstack(fill_value=0)
    )

    if 0 not in counts.columns:
        counts[0] = 0
    if 1 not in counts.columns:
        counts[1] = 0

    counts = counts[[0, 1]]

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Contracts")
    ax.legend(["Label 0", "Label 1"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig, ax


# ---------------------------------------
# Condition comparison plots
# ---------------------------------------

def plot_condition_metric(
    results_df: pd.DataFrame,
    metric: str = "gold_auroc",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 5),
):
    """
    Compare Stage 1 conditions (weak_only, gold_only, hybrid) by model.
    Expects columns: model, condition, metric.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=results_df, x="model", y=metric, hue="condition", ax=ax)
    ax.set_title(title if title is not None else f"Comparison on {metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()

    return fig, ax


# ---------------------------------------
# Classification curve plots
# ---------------------------------------

def plot_roc_curve(
    y_true,
    y_score,
    model_name: str = "Model",
    figsize: tuple[int, int] = (6, 6),
):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    plt.tight_layout()

    return fig, ax


def plot_precision_recall_curve(
    y_true,
    y_score,
    model_name: str = "Model",
    figsize: tuple[int, int] = (6, 6),
):
    """
    Plot precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, label=model_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    plt.tight_layout()

    return fig, ax

# ---------------------------------------
# Meta-Learning plots
# ---------------------------------------

def plot_adaptation_curve(
    loss_histories: dict[str, list[float]],
    title: str = "Inner-Loop Adaptation Loss",
    figsize: tuple[int, int] = (8, 5),
):
    """
    Plot the inner-loop adaptation loss across steps for multiple algorithms.

    Parameters
    ----------
    loss_histories : dict
        A dictionary mapping algorithm names to their list of loss values per step.
        e.g., {"MAML": [0.8, 0.5, 0.3], "Fine-Tuning": [0.8, 0.7, 0.65]}
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, losses in loss_histories.items():
        steps = range(1, len(losses) + 1)
        ax.plot(steps, losses, marker="o", label=model_name)

    ax.set_xlabel("Inner Gradient Step")
    ax.set_ylabel("Adaptation Loss")
    ax.set_title(title)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig, ax

def plot_metric_boxplots(
    metrics_df: pd.DataFrame,
    metric_col: str = "gold_auroc",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 6)
):
    """
    Plot the mathematical distribution (variance) of an evaluation metric across multiple random splits.
    This visually proves which Meta-Learning algorithm is the most stable.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=metrics_df, x="model", y=metric_col, hue="model", ax=ax, palette="Set2")
    
    if title is None:
        title = f"Algorithm Stability across Multiple Splits ({metric_col.replace('gold_', '').upper()})"
    
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(metric_col.replace('gold_', '').upper())
    ax.set_xlabel("Algorithm")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    return fig, ax

def plot_aggregated_metrics(
    aggregated_df: pd.DataFrame,
    metric_col: str = "gold_auroc",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 6)
):
    """
    Plot a Bar Chart proving the Mean performance, with Error Bars representing the Standard Deviation.
    """
    means = aggregated_df[(metric_col, "mean")]
    stds = aggregated_df[(metric_col, "std")]
    models = aggregated_df.index
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(models))
    # Note: Use a generic color cycle rather than hard-coding 3 colors in case more models are added
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.8, ecolor='black', capsize=8, color=sns.color_palette("muted", len(models)))
    
    if title is None:
        title = f"Mean Performance vs Variance ({metric_col.replace('gold_', '').upper()})"
        
    ax.set_ylabel(metric_col.replace('gold_', '').upper())
    ax.set_xlabel("Algorithm")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_title(title, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def plot_calibration_reliability(
    calib_df: pd.DataFrame,
    model_name: str = "Model",
    title: str | None = None,
    figsize: tuple[int, int] = (6, 6)
):
    """
    Plot a Reliability Diagram (Calibration Curve) using the binned calibration_table.
    Perfectly calibrated models should fall perfectly on the y=x dashed line.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract the numeric valid bins
    valid_bins = calib_df.dropna(subset=["mean_pred", "event_rate"])
    mean_preds = valid_bins["mean_pred"]
    empirical_rates = valid_bins["event_rate"]
    
    ax.plot(mean_preds, empirical_rates, marker='o', linewidth=2, label=model_name)
    
    # Plot the ideal y=x perfectly calibrated line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly Calibrated")
    
    if title is None:
        title = f"Calibration Reliability Diagram - {model_name}"
        
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Event Rate (True Probability)")
    ax.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_k_shot_curve(
    k_shot_df: pd.DataFrame,
    metric_col: str = "gold_auroc",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 6)
):
    """
    Plot a Line Graph demonstrating how Model Performance scales as the number of Support Examples increases.
    Expects k_shot_df to have columns: 'k_shots', 'model', and metric_col.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.lineplot(data=k_shot_df, x="k_shots", y=metric_col, hue="model", marker="o", ax=ax, linewidth=2)
    
    if title is None:
        title = f"K-Shot Adaptation Scaling Curve ({metric_col.replace('gold_', '').upper()})"
        
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Number of Support Examples (K-shots)")
    ax.set_ylabel(metric_col.replace('gold_', '').upper())
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    return fig, ax

# ---------------------------------------
# Meta-Learning: Comparisons
# ---------------------------------------

def plot_grouped_model_comparison(
    master_df: pd.DataFrame,
    metrics: list[str] = ["gold_auroc", "gold_ap"],
    title: str = "Mean Metrics Comparison",
    figsize: tuple[int, int] = (10, 6)
):
    """
    A Grouped Bar Chart comparing Mean AUROC/F1/ECE across methods.
    """
    mean_df = master_df.groupby("model")[metrics].mean().reset_index()
    melted_df = pd.melt(mean_df, id_vars="model", var_name="Metric", value_name="Score")
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=melted_df, x="Metric", y="Score", hue="model", ax=ax, palette="Set2")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig, ax

def plot_boxplots_repeats(
    master_df: pd.DataFrame,
    metric: str = "gold_auroc",
    title: str = "Variance across 20-Split Iterations",
    figsize: tuple[int, int] = (10, 6)
):
    """
    A Boxplot generator to visualize the variance (Mean/Std) across the iterated episodes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=master_df, x="model", y=metric, ax=ax, palette="Pastel1", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(data=master_df, x="model", y=metric, ax=ax, color="black", alpha=0.3, jitter=True)
    ax.set_title(title)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Algorithms")
    plt.tight_layout()
    return fig, ax
