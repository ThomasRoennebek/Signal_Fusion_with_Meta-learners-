"""
plotting.py — Reusable plotting utilities for EDA, model evaluation, and thesis figures.

This module centralizes plotting style, saving, and common figure types so that
all notebooks produce consistent, publication-ready visualizations.
"""
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from master_thesis.config import FIGURES


# -----------------------------------------------------------------------------
# Thesis style and color mappings
# -----------------------------------------------------------------------------

THESIS_PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#999999",
    "light_grey": "#DDDDDD",
}

METHOD_COLORS = {
    "zero_shot": THESIS_PALETTE["grey"],
    "zero-shot": THESIS_PALETTE["grey"],
    "pretrained": THESIS_PALETTE["grey"],
    "finetune": THESIS_PALETTE["blue"],
    "fine_tune": THESIS_PALETTE["blue"],
    "fine-tune": THESIS_PALETTE["blue"],
    "anil": THESIS_PALETTE["green"],
    "fomaml": THESIS_PALETTE["orange"],
    "maml": THESIS_PALETTE["vermillion"],
    "xgboost": THESIS_PALETTE["purple"],
    "elastic_net": THESIS_PALETTE["sky_blue"],
    "elastic net": THESIS_PALETTE["sky_blue"],
    "mean_predictor": THESIS_PALETTE["grey"],
    "mean predictor": THESIS_PALETTE["grey"],
}

VIEW_COLORS = {
    "contract_core": THESIS_PALETTE["blue"],
    "financial": THESIS_PALETTE["vermillion"],
    "esg": THESIS_PALETTE["green"],
    "news": THESIS_PALETTE["orange"],
    "market": THESIS_PALETTE["purple"],
    "macro_logistics": THESIS_PALETTE["sky_blue"],
    "labels": THESIS_PALETTE["grey"],
}


def set_thesis_style(
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    font_scale: float = 1.15,
    use_latex_if_available: bool = False,
    grid: bool = False,
) -> dict[str, str]:
    """
    Apply a clean, publication-ready plotting style for all thesis figures.

    The default style is intentionally minimal: white background, no top/right
    spines, colorblind-safe colors, embedded fonts for PDF export, and high-DPI
    saving. It works for EDA plots, heatmaps, model evaluation, and SHAP figures.

    Returns
    -------
    dict[str, str]
        Method-specific color mapping that can be reused in plots.
    """
    latex_available = shutil.which("latex") is not None
    use_latex = use_latex_if_available and latex_available

    sns.set_theme(
        context="paper",
        style="whitegrid" if grid else "white",
        font=font_name,
        font_scale=font_scale,
    )

    plt.rcParams.update({
        # Resolution
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # Fonts
        "font.family": font_family,
        "font.serif": [font_name, "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "text.usetex": use_latex,
        # Font sizes
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "figure.titlesize": 14,
        # Lines and markers
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "patch.linewidth": 0.8,
        # Clean visual style
        "axes.grid": grid,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        # Saving
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Color cycle: Okabe-Ito inspired
        "axes.prop_cycle": plt.cycler(color=[
            THESIS_PALETTE["blue"],
            THESIS_PALETTE["vermillion"],
            THESIS_PALETTE["green"],
            THESIS_PALETTE["orange"],
            THESIS_PALETTE["sky_blue"],
            THESIS_PALETTE["purple"],
            THESIS_PALETTE["grey"],
        ]),
    })

    return METHOD_COLORS


def despine_thesis(ax=None) -> None:
    """Remove unnecessary plot spines and standardize tick styling."""
    if ax is None:
        ax = plt.gca()
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(axis="both", which="major", length=4, width=0.8)


def _normalise_method_name(name: str) -> str:
    return str(name).lower().replace(" ", "_")


def method_palette(methods: Iterable[str]) -> dict[str, str]:
    """Return a robust palette for a given list of method names."""
    colors = {}
    fallback = sns.color_palette("colorblind", n_colors=max(1, len(list(methods))))
    for i, method in enumerate(methods):
        key = _normalise_method_name(method)
        colors[method] = METHOD_COLORS.get(key, fallback[i % len(fallback)])
    return colors


# -----------------------------------------------------------------------------
# Saving helpers
# -----------------------------------------------------------------------------

def save_fig(
    fig,
    filename: str,
    output_dir: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Save a matplotlib figure to disk."""
    if output_dir is None:
        output_dir = FIGURES

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    print(f"Figure saved to: {output_path}")
    return output_path


def save_thesis_fig(
    fig,
    filename: str,
    output_dir: str | Path = "reports/figures",
    width: str = "double",
    height: float | None = None,
    filetypes: tuple[str, ...] = ("png", "pdf"),
) -> dict[str, Path]:
    """
    Save a figure in thesis/publication-ready dimensions.

    width options:
    - single: approx. single-column journal width
    - double: approx. double-column journal width
    - wide: wider thesis/report figure
    """
    width_map = {
        "single": 3.5,
        "double": 7.2,
        "wide": 9.0,
    }
    if width not in width_map:
        raise ValueError("width must be one of: 'single', 'double', 'wide'")

    fig_width = width_map[width]
    fig_height = height if height is not None else fig_width * 0.58
    fig.set_size_inches(fig_width, fig_height)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for filetype in filetypes:
        path = output_dir / f"{filename}.{filetype}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        saved_paths[filetype] = path
        print(f"Saved figure: {path}")

    return saved_paths


# -----------------------------------------------------------------------------
# Generic EDA plots
# -----------------------------------------------------------------------------

def plot_distribution(
    df: pd.DataFrame,
    column: str,
    hue: str | None = None,
    bins: int = 40,
    kde: bool = True,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """Plot a numeric distribution using a histogram and optional KDE curve."""
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=df, x=column, hue=hue, bins=bins, kde=kde, ax=ax)
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_categorical_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int | None = 20,
    horizontal: bool = True,
    title: str | None = None,
    figsize: tuple[int, int] = (9, 6),
):
    """Plot counts for a categorical column."""
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    counts = df[column].fillna("Missing").value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    plot_df = counts.rename_axis(column).reset_index(name="count")

    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        sns.barplot(data=plot_df, y=column, x="count", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel(column)
    else:
        sns.barplot(data=plot_df, x=column, y="count", ax=ax)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    ax.set_title(title or f"Counts of {column}")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_numeric_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    order: Sequence[str] | None = None,
    figsize: tuple[int, int] = (10, 6),
    show_points: bool = False,
):
    """Plot a numeric variable by category using a boxplot."""
    for col in [x, y]:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y, order=order, ax=ax)
    if show_points:
        sns.stripplot(data=df, x=x, y=y, order=order, color="black", alpha=0.25, jitter=True, ax=ax)
    ax.set_title(title or f"{y} by {x}")
    ax.tick_params(axis="x", rotation=35)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_missingness_by_column(
    df: pd.DataFrame,
    top_n: int = 30,
    title: str = "Missingness by Column",
    figsize: tuple[int, int] = (9, 7),
):
    """Plot the columns with the highest missingness percentages."""
    missing = df.isna().mean().mul(100).sort_values(ascending=False).head(top_n)
    plot_df = missing.rename_axis("feature").reset_index(name="missing_pct")

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_df, y="feature", x="missing_pct", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Feature")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_missingness_heatmap(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    sample_n: int | None = 500,
    title: str = "Missingness Heatmap",
    figsize: tuple[int, int] = (10, 6),
):
    """Plot a binary missingness heatmap for selected columns and rows."""
    plot_df = df.copy()
    if columns is not None:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        plot_df = plot_df[list(columns)]

    if sample_n is not None and len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(plot_df.isna(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Rows")
    plt.tight_layout()
    return fig, ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    title: str = "Pearson Correlation Heatmap",
    figsize: tuple[int, int] = (10, 8),
    method: str = "pearson",
    annot: bool = False,
    mask_upper: bool = True,
    cmap: str = "vlag",
    vmin: float = -1,
    vmax: float = 1,
):
    """Plot a correlation heatmap for numeric columns."""
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        numeric_df = df[list(columns)].select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric columns available for correlation heatmap.")

    corr = numeric_df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        annot=annot,
        fmt=".2f",
        linewidths=0.2,
        cbar_kws={"label": f"{method.title()} correlation"},
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_view_missingness(
    view_summary: pd.DataFrame,
    view_col: str = "view",
    missing_col: str = "avg_missing_pct",
    title: str = "Average Missingness by Feature View",
    figsize: tuple[int, int] = (8, 5),
):
    """Plot average missingness by data view."""
    for col in [view_col, missing_col]:
        if col not in view_summary.columns:
            raise ValueError(f"Column not found: {col}")

    fig, ax = plt.subplots(figsize=figsize)
    palette = [VIEW_COLORS.get(v, THESIS_PALETTE["grey"]) for v in view_summary[view_col]]
    sns.barplot(data=view_summary, x=view_col, y=missing_col, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Data view")
    ax.set_ylabel("Average missingness (%)")
    ax.tick_params(axis="x", rotation=25)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Baseline and model evaluation plots
# -----------------------------------------------------------------------------

def plot_baseline_metric(
    baseline_results: pd.DataFrame,
    metric: str = "mae",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """Plot a bar chart for a chosen baseline metric."""
    plot_df = baseline_results.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(plot_df["model"], plot_df[metric])
    ax.set_ylabel(f"Validation {metric.upper()}")
    ax.set_title(title if title is not None else "Baseline Comparison")
    ax.tick_params(axis="x", rotation=25)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_predicted_vs_true(
    y_true,
    y_pred,
    model_name: str = "Model",
    figsize: tuple[int, int] = (6, 6),
    alpha: float = 0.4,
):
    """Scatter plot of predicted vs. true values."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=alpha)
    ax.set_xlabel("True renegotiation_prob")
    ax.set_ylabel("Predicted renegotiation_prob")
    ax.set_title(f"{model_name}: Predicted vs. True")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: tuple[int, int] = (10, 6),
):
    """Plot top-N feature importances as a horizontal bar chart."""
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Label and department plots
# -----------------------------------------------------------------------------

def plot_label_distribution_by_department(
    df: pd.DataFrame,
    department_col: str = "department",
    label_col: str = "target_renegotiate",
    contract_id_col: str = "contract_id",
    title: str = "Label Distribution by Department",
    figsize: tuple[int, int] = (14, 6),
):
    """Plot label counts by department using one row per unique contract."""
    plot_df = df[[contract_id_col, department_col, label_col]].drop_duplicates(subset=[contract_id_col]).copy()
    plot_df[department_col] = plot_df[department_col].fillna("Unknown")

    counts = plot_df.groupby([department_col, label_col]).size().unstack(fill_value=0)
    if 0 not in counts.columns:
        counts[0] = 0
    if 1 not in counts.columns:
        counts[1] = 0
    counts = counts[[0, 1]].sort_values(1, ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Contracts")
    ax.legend(["Label 0", "Label 1"])
    ax.tick_params(axis="x", rotation=45)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_contracts_by_department(
    df: pd.DataFrame,
    department_col: str = "department",
    contract_id_col: str = "contract_id",
    title: str = "Unique Contracts per Department",
    figsize: tuple[int, int] = (10, 6),
):
    """Plot unique contract counts by department."""
    counts = (
        df.drop_duplicates(subset=[contract_id_col])
        .groupby(department_col)[contract_id_col]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_unique_contracts")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=counts, y=department_col, x="n_unique_contracts", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Unique contracts")
    ax.set_ylabel("Department")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_gold_label_distribution(
    gold_df: pd.DataFrame,
    department_col: str = "department",
    label_col: str = "gold_y",
    title: str = "Gold Label Distribution by Department",
    figsize: tuple[int, int] = (10, 6),
):
    """Plot gold yes/no labels per department as a stacked horizontal bar chart."""
    if label_col not in gold_df.columns:
        raise ValueError(f"Column not found: {label_col}")

    plot_df = gold_df[gold_df[label_col].notna()].copy()
    counts = plot_df.groupby([department_col, label_col]).size().unstack(fill_value=0)
    if 0 not in counts.columns:
        counts[0] = 0
    if 1 not in counts.columns:
        counts[1] = 0
    counts = counts[[0, 1]]
    counts["total"] = counts[0] + counts[1]
    counts = counts.sort_values("total", ascending=True).drop(columns="total")

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind="barh", stacked=True, ax=ax, color=[THESIS_PALETTE["vermillion"], THESIS_PALETTE["green"]])
    ax.set_title(title)
    ax.set_xlabel("Gold labels")
    ax.set_ylabel("Department")
    ax.legend(["No", "Yes"], title="Gold label")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_snorkel_probability_distribution(
    df: pd.DataFrame,
    prob_col: str = "renegotiation_prob",
    department_col: str | None = None,
    title: str = "Distribution of Snorkel Probabilities",
    figsize: tuple[int, int] = (8, 5),
):
    """Plot Snorkel probability distribution overall or by department."""
    if prob_col not in df.columns:
        raise ValueError(f"Column not found: {prob_col}")

    fig, ax = plt.subplots(figsize=figsize)
    if department_col is None:
        sns.histplot(data=df, x=prob_col, bins=40, kde=True, ax=ax)
        ax.set_ylabel("Count")
    else:
        sns.boxplot(data=df, y=department_col, x=prob_col, ax=ax)
        ax.set_ylabel("Department")
    ax.set_title(title)
    ax.set_xlabel(prob_col)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Condition comparison and classification curves
# -----------------------------------------------------------------------------

def plot_condition_metric(
    results_df: pd.DataFrame,
    metric: str = "gold_auroc",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 5),
):
    """Compare Stage 1 conditions by model."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=results_df, x="model", y=metric, hue="condition", ax=ax)
    ax.set_title(title if title is not None else f"Comparison on {metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_roc_curve(
    y_true,
    y_score,
    model_name: str = "Model",
    figsize: tuple[int, int] = (6, 6),
):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color=THESIS_PALETTE["grey"], label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_precision_recall_curve(
    y_true,
    y_score,
    model_name: str = "Model",
    figsize: tuple[int, int] = (6, 6),
):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, label=model_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_multiple_roc_curves(
    curve_df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_score_col: str = "y_score",
    method_col: str = "method",
    title: str = "ROC Curves by Method",
    figsize: tuple[int, int] = (6, 6),
):
    """Plot ROC curves for multiple methods from a long dataframe."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = method_palette(curve_df[method_col].unique())
    for method, group in curve_df.groupby(method_col):
        fpr, tpr, _ = roc_curve(group[y_true_col], group[y_score_col])
        ax.plot(fpr, tpr, label=method, color=colors.get(method))
    ax.plot([0, 1], [0, 1], linestyle="--", color=THESIS_PALETTE["grey"], label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Meta-learning plots
# -----------------------------------------------------------------------------

def plot_adaptation_curve(
    loss_histories: dict[str, list[float]],
    title: str = "Inner-Loop Adaptation Loss",
    figsize: tuple[int, int] = (8, 5),
):
    """Plot inner-loop adaptation loss across gradient steps for multiple algorithms."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = method_palette(loss_histories.keys())

    for model_name, losses in loss_histories.items():
        steps = range(1, len(losses) + 1)
        ax.plot(steps, losses, marker="o", label=model_name, color=colors.get(model_name))

    ax.set_xlabel("Inner Gradient Step")
    ax.set_ylabel("Adaptation Loss")
    ax.set_title(title)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_metric_boxplots(
    metrics_df: pd.DataFrame,
    metric_col: str = "gold_auroc",
    method_col: str = "model",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 6),
):
    """Plot the distribution of an evaluation metric across repeated episodes."""
    fig, ax = plt.subplots(figsize=figsize)
    palette = method_palette(metrics_df[method_col].unique())
    sns.boxplot(data=metrics_df, x=method_col, y=metric_col, ax=ax, palette=palette, showmeans=True)
    sns.stripplot(data=metrics_df, x=method_col, y=metric_col, ax=ax, color="black", alpha=0.25, jitter=True)

    if title is None:
        title = f"Repeated-Episode Distribution ({metric_col.replace('gold_', '').upper()})"
    ax.set_title(title)
    ax.set_ylabel(metric_col.replace("gold_", "").upper())
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=20)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_aggregated_metrics(
    aggregated_df: pd.DataFrame,
    metric_col: str = "gold_auroc",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 6),
):
    """Plot mean performance with standard-deviation error bars from a grouped dataframe."""
    means = aggregated_df[(metric_col, "mean")]
    stds = aggregated_df[(metric_col, "std")]
    models = list(aggregated_df.index)

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(models))
    colors = [method_palette(models).get(m, THESIS_PALETTE["grey"]) for m in models]
    ax.bar(x_pos, means, yerr=stds, align="center", ecolor="black", capsize=6, color=colors)

    if title is None:
        title = f"Mean Performance with Standard Deviation ({metric_col.replace('gold_', '').upper()})"

    ax.set_ylabel(metric_col.replace("gold_", "").upper())
    ax.set_xlabel("Method")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=20)
    ax.set_title(title)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_calibration_reliability(
    calib_df: pd.DataFrame,
    model_name: str = "Model",
    title: str | None = None,
    figsize: tuple[int, int] = (6, 6),
):
    """Plot a reliability diagram using a binned calibration table."""
    fig, ax = plt.subplots(figsize=figsize)
    valid_bins = calib_df.dropna(subset=["mean_pred", "event_rate"])

    ax.plot(valid_bins["mean_pred"], valid_bins["event_rate"], marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color=THESIS_PALETTE["grey"], label="Perfect calibration")

    ax.set_title(title or f"Calibration Reliability Diagram — {model_name}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Event Rate")
    ax.legend(frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_k_shot_curve(
    k_shot_df: pd.DataFrame,
    metric_col: str = "gold_auroc",
    k_col: str = "k_shots",
    method_col: str = "model",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 6),
):
    """Plot how model performance scales as support examples increase."""
    fig, ax = plt.subplots(figsize=figsize)
    palette = method_palette(k_shot_df[method_col].unique())
    sns.lineplot(data=k_shot_df, x=k_col, y=metric_col, hue=method_col, marker="o", ax=ax, linewidth=2, palette=palette)

    ax.set_title(title or f"K-Shot Adaptation Scaling Curve ({metric_col.replace('gold_', '').upper()})")
    ax.set_xlabel("Support examples per class (k-shot)")
    ax.set_ylabel(metric_col.replace("gold_", "").upper())
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_grouped_model_comparison(
    master_df: pd.DataFrame,
    metrics: list[str] | None = None,
    method_col: str = "model",
    title: str = "Mean Metrics Comparison",
    figsize: tuple[int, int] = (10, 6),
):
    """Grouped bar chart comparing mean metrics across methods."""
    if metrics is None:
        metrics = ["gold_auroc", "gold_ap"]

    mean_df = master_df.groupby(method_col)[metrics].mean().reset_index()
    melted_df = pd.melt(mean_df, id_vars=method_col, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=figsize)
    palette = method_palette(mean_df[method_col].unique())
    sns.barplot(data=melted_df, x="Metric", y="Score", hue=method_col, ax=ax, palette=palette)
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean score")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_boxplots_repeats(
    master_df: pd.DataFrame,
    metric: str = "gold_auroc",
    method_col: str = "model",
    title: str = "Variance across Repeated Episodes",
    figsize: tuple[int, int] = (10, 6),
):
    """Boxplot generator for repeated-episode variance."""
    return plot_metric_boxplots(master_df, metric_col=metric, method_col=method_col, title=title, figsize=figsize)


def plot_interaction_heatmap(
    pivot_df: pd.DataFrame,
    title: str = "Hyperparameter Interaction Heatmap",
    metric_label: str = "AUROC",
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "viridis",
):
    """Plot a heatmap showing interaction between two hyperparameters."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar_kws={"label": metric_label},
        ax=ax,
    )
    ax.set_title(title, pad=16)
    plt.tight_layout()
    return fig, ax


def plot_kshot_heatmap(
    results_df: pd.DataFrame,
    value_col: str = "gold_auroc",
    k_col: str = "k_shots",
    step_col: str = "inner_steps",
    method: str | None = None,
    method_col: str = "method",
    aggfunc: str = "mean",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 6),
):
    """Plot a k-shot × inner-step heatmap for one method or pooled results."""
    plot_df = results_df.copy()
    if method is not None:
        plot_df = plot_df[plot_df[method_col] == method]

    pivot = plot_df.pivot_table(index=step_col, columns=k_col, values=value_col, aggfunc=aggfunc)
    fig, ax = plot_interaction_heatmap(
        pivot,
        title=title or f"{method or 'All methods'}: {value_col} by k-shot and inner steps",
        metric_label=value_col,
        figsize=figsize,
    )
    ax.set_xlabel("k-shot")
    ax.set_ylabel("Inner steps")
    return fig, ax

# -----------------------------------------------------------------------------
# Thesis-specific data landscape and diagnostic plots
# -----------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: Sequence[str], df_name: str = "df") -> None:
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def _unique_contract_frame(
    df: pd.DataFrame,
    contract_id_col: str = "contract_id",
) -> pd.DataFrame:
    """Return one row per unique contract for contract-level diagnostics."""
    if contract_id_col not in df.columns:
        raise ValueError(f"Column not found: {contract_id_col}")
    return df.drop_duplicates(subset=[contract_id_col]).copy()


def plot_department_contract_distribution(
    df: pd.DataFrame,
    department_col: str = "department",
    contract_id_col: str = "contract_id",
    top_n: int | None = None,
    title: str = "Unique Contracts by Department",
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot unique contract counts by department.

    Thesis use: documents the department-level structure of the contract universe
    and motivates department-aware evaluation.
    """
    _require_columns(df, [department_col, contract_id_col])

    plot_df = _unique_contract_frame(df, contract_id_col)
    counts = (
        plot_df.groupby(department_col, dropna=False)[contract_id_col]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_unique_contracts")
        .reset_index()
    )
    counts[department_col] = counts[department_col].fillna("Missing")
    if top_n is not None:
        counts = counts.head(top_n)
    counts = counts.sort_values("n_unique_contracts", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=counts, y=department_col, x="n_unique_contracts", ax=ax, color=THESIS_PALETTE["blue"])
    ax.set_title(title)
    ax.set_xlabel("Unique contracts")
    ax.set_ylabel("Department")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_gold_label_balance_by_department(
    df: pd.DataFrame,
    department_col: str = "department",
    label_col: str = "gold_y",
    contract_id_col: str | None = "contract_id",
    title: str = "Gold Label Balance by Department",
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot yes/no gold labels by department as stacked bars.

    Thesis use: shows whether each department has enough class balance for
    support/query construction in few-shot adaptation.
    """
    _require_columns(df, [department_col, label_col])

    plot_df = df[df[label_col].notna()].copy()
    if contract_id_col is not None and contract_id_col in plot_df.columns:
        plot_df = plot_df.drop_duplicates(subset=[contract_id_col, label_col])

    counts = plot_df.groupby([department_col, label_col]).size().unstack(fill_value=0)
    if 0 not in counts.columns:
        counts[0] = 0
    if 1 not in counts.columns:
        counts[1] = 0
    counts = counts[[0, 1]]
    counts["total"] = counts[0] + counts[1]
    counts = counts.sort_values("total", ascending=True).drop(columns="total")

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[THESIS_PALETTE["vermillion"], THESIS_PALETTE["green"]],
        width=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("Gold labels")
    ax.set_ylabel("Department")
    ax.legend(["No", "Yes"], title="Gold label", frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_gold_coverage_by_department(
    df: pd.DataFrame,
    department_col: str = "department",
    contract_id_col: str = "contract_id",
    label_col: str = "gold_y",
    title: str = "Gold Label Coverage by Department",
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot gold-label coverage percentage by department.

    Thesis use: visualizes the gap between the unlabeled contract universe and
    expert-reviewed labels, motivating weak supervision.
    """
    _require_columns(df, [department_col, contract_id_col, label_col])

    contracts = (
        _unique_contract_frame(df, contract_id_col)
        .groupby(department_col, dropna=False)[contract_id_col]
        .nunique()
        .rename("n_unique_contracts")
    )
    gold = (
        df[df[label_col].notna()]
        .drop_duplicates(subset=[contract_id_col])
        .groupby(department_col, dropna=False)[contract_id_col]
        .nunique()
        .rename("gold_total")
    )
    summary = pd.concat([contracts, gold], axis=1).fillna(0).reset_index()
    summary["gold_coverage_pct"] = 100 * summary["gold_total"] / summary["n_unique_contracts"].replace(0, np.nan)
    summary = summary.sort_values("gold_coverage_pct", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=summary, y=department_col, x="gold_coverage_pct", ax=ax, color=THESIS_PALETTE["orange"])
    ax.set_title(title)
    ax.set_xlabel("Gold coverage (%)")
    ax.set_ylabel("Department")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_stage2_task_validity_heatmap(
    validity_df: pd.DataFrame,
    department_col: str = "department",
    k_cols: Sequence[str] = ("valid_k2", "valid_k5", "valid_k10"),
    title: str = "Stage 2 Task Validity by k-shot Setting",
    figsize: tuple[int, int] = (7, 7),
):
    """
    Plot a heatmap showing which departments are valid for k-shot episodes.

    Expects boolean or 0/1 columns such as valid_k2, valid_k5, valid_k10.
    """
    _require_columns(validity_df, [department_col, *k_cols])

    plot_df = validity_df[[department_col, *k_cols]].copy()
    plot_df = plot_df.set_index(department_col)
    heatmap_data = plot_df.astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        cmap=sns.color_palette([THESIS_PALETTE["light_grey"], THESIS_PALETTE["green"]], as_cmap=True),
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot=True,
        fmt="d",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("k-shot validity rule")
    ax.set_ylabel("Department")
    plt.tight_layout()
    return fig, ax


def plot_lpi_threshold_distribution(
    df: pd.DataFrame,
    lpi_col: str = "LPI_Score",
    flag_col: str = "lpi_below_supplier_median",
    threshold: float | None = None,
    title: str = "Supplier-Country LPI Distribution",
    figsize: tuple[int, int] = (8, 5),
):
    """
    Plot LPI score distribution with the dataset-relative median threshold.

    Thesis use: justifies using supplier-population-relative logistics risk rather
    than a generic global LPI threshold.
    """
    _require_columns(df, [lpi_col])
    values = df[lpi_col].dropna()
    if values.empty:
        raise ValueError(f"No non-missing values available in {lpi_col}.")

    threshold_value = float(values.median()) if threshold is None else threshold

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(values, bins=30, kde=True, ax=ax, color=THESIS_PALETTE["sky_blue"])
    ax.axvline(
        threshold_value,
        color=THESIS_PALETTE["vermillion"],
        linestyle="--",
        linewidth=2,
        label=f"Supplier median = {threshold_value:.2f}",
    )
    ax.set_title(title)
    ax.set_xlabel(lpi_col)
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_public_supplier_coverage(
    df: pd.DataFrame,
    public_col: str = "supplier_is_publicly_listed",
    contract_id_col: str | None = "contract_id",
    title: str = "Publicly Listed Supplier Coverage",
    figsize: tuple[int, int] = (6, 4),
):
    """
    Plot the share of contracts/suppliers with stock-market data availability.

    Thesis use: shows that market data missingness is structurally informative
    because many suppliers are private rather than listed.
    """
    _require_columns(df, [public_col])

    plot_df = df.copy()
    if contract_id_col is not None and contract_id_col in plot_df.columns:
        plot_df = plot_df.drop_duplicates(subset=[contract_id_col])

    counts = plot_df[public_col].fillna(0).astype(int).value_counts().reindex([0, 1], fill_value=0)
    labels = ["No listed supplier data", "Publicly listed supplier"]
    plot_counts = pd.DataFrame({"status": labels, "count": [counts.loc[0], counts.loc[1]]})

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_counts, x="status", y="count", ax=ax, palette=[THESIS_PALETTE["grey"], THESIS_PALETTE["purple"]])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Unique contracts" if contract_id_col else "Rows")
    ax.tick_params(axis="x", rotation=20)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_stage1_metric_comparison(
    results_df: pd.DataFrame,
    metrics: Sequence[str] = ("gold_auroc", "gold_logloss", "gold_ece", "ndcg_at_10"),
    condition_col: str = "condition",
    model_col: str = "model_base",
    title: str = "Stage 1 Baseline Comparison",
    figsize: tuple[int, int] = (11, 6),
):
    """Plot Stage 1 metrics across model families and supervision conditions."""
    _require_columns(results_df, [condition_col, model_col, *metrics])
    plot_df = results_df[[condition_col, model_col, *metrics]].copy()
    long_df = plot_df.melt(id_vars=[condition_col, model_col], value_vars=list(metrics), var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=long_df, x="metric", y="value", hue=condition_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.legend(title="Condition", frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_stage2_metric_comparison(
    results_df: pd.DataFrame,
    metrics: Sequence[str] = ("gold_auroc", "gold_logloss", "gold_ece", "ndcg_at_10"),
    method_col: str = "method",
    title: str = "Stage 2 Adaptation Benchmark",
    figsize: tuple[int, int] = (11, 6),
):
    """
    Plot Stage 2 method comparison across core metrics.

    If the input contains repeated episodes, the bar height is the mean and the
    error bar is the standard deviation computed by seaborn.
    """
    _require_columns(results_df, [method_col, *metrics])
    long_df = results_df[[method_col, *metrics]].melt(id_vars=method_col, value_vars=list(metrics), var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=figsize)
    palette = method_palette(results_df[method_col].unique())
    sns.barplot(data=long_df, x="metric", y="value", hue=method_col, palette=palette, errorbar="sd", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean value")
    ax.legend(title="Method", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_repeated_episode_boxplot(
    results_df: pd.DataFrame,
    metric_col: str = "gold_auroc",
    method_col: str = "method",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """Alias for repeated-episode distributions using thesis terminology."""
    return plot_metric_boxplots(
        results_df,
        metric_col=metric_col,
        method_col=method_col,
        title=title or f"Repeated-Episode Distribution: {metric_col}",
        figsize=figsize,
    )


def plot_inner_step_curve(
    results_df: pd.DataFrame,
    metric_col: str = "gold_auroc",
    step_col: str = "inner_steps",
    method_col: str = "method",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """Plot performance as a function of inner-loop adaptation steps."""
    _require_columns(results_df, [metric_col, step_col, method_col])

    fig, ax = plt.subplots(figsize=figsize)
    palette = method_palette(results_df[method_col].unique())
    sns.lineplot(data=results_df, x=step_col, y=metric_col, hue=method_col, marker="o", errorbar="sd", palette=palette, ax=ax)
    ax.set_title(title or f"Inner-Step Adaptation Curve ({metric_col})")
    ax.set_xlabel("Inner-loop gradient steps")
    ax.set_ylabel(metric_col)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(title="Method", frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_prediction_probability_distribution(
    predictions_df: pd.DataFrame,
    prob_col: str = "y_score",
    method_col: str = "method",
    label_col: str | None = None,
    title: str = "Prediction Probability Distribution by Method",
    figsize: tuple[int, int] = (8, 5),
):
    """Plot predicted probability distributions to diagnose overconfidence."""
    _require_columns(predictions_df, [prob_col, method_col])

    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(data=predictions_df, x=prob_col, hue=method_col, common_norm=False, fill=False, ax=ax)
    if label_col is not None and label_col in predictions_df.columns:
        base_rate = predictions_df[label_col].mean()
        ax.axvline(base_rate, color=THESIS_PALETTE["grey"], linestyle="--", label=f"Base rate = {base_rate:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Predicted renegotiation probability")
    ax.set_ylabel("Density")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_shap_delta_importance(
    shap_delta_df: pd.DataFrame,
    feature_col: str = "feature",
    delta_col: str = "delta_importance",
    top_n: int = 20,
    title: str = "Largest SHAP Importance Shifts After Adaptation",
    figsize: tuple[int, int] = (9, 7),
):
    """Plot the largest absolute SHAP importance shifts pre/post adaptation."""
    _require_columns(shap_delta_df, [feature_col, delta_col])
    plot_df = shap_delta_df.copy()
    plot_df["abs_delta"] = plot_df[delta_col].abs()
    plot_df = plot_df.sort_values("abs_delta", ascending=False).head(top_n).sort_values(delta_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [THESIS_PALETTE["green"] if val >= 0 else THESIS_PALETTE["vermillion"] for val in plot_df[delta_col]]
    ax.barh(plot_df[feature_col], plot_df[delta_col], color=colors)
    ax.axvline(0, color=THESIS_PALETTE["black"], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Post-adaptation importance minus pre-adaptation importance")
    ax.set_ylabel("Feature")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_view_level_shap_importance(
    view_importance_df: pd.DataFrame,
    view_col: str = "view",
    importance_col: str = "mean_abs_shap",
    title: str = "View-Level SHAP Importance",
    figsize: tuple[int, int] = (8, 5),
):
    """Plot SHAP importance aggregated by data view."""
    _require_columns(view_importance_df, [view_col, importance_col])
    plot_df = view_importance_df.sort_values(importance_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [VIEW_COLORS.get(view, THESIS_PALETTE["grey"]) for view in plot_df[view_col]]
    ax.barh(plot_df[view_col], plot_df[importance_col], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Data view")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax

# -----------------------------------------------------------------------------
# Representation / embedding diagnostics
# -----------------------------------------------------------------------------

def compute_2d_projection(
    X,
    method: str = "pca",
    random_state: int = 42,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> pd.DataFrame:
    """
    Reduce a feature matrix or model embedding matrix to two dimensions.

    This helper is intended for qualitative diagnostics such as:
    - checking whether departments form distinct task clusters
    - checking whether gold-label classes separate before/after adaptation
    - visualizing whether a learned representation changes after adaptation

    Parameters
    ----------
    X:
        Array-like matrix with shape (n_samples, n_features).
    method:
        One of {"pca", "tsne", "umap"}. UMAP requires the optional `umap-learn`
        package. If UMAP is unavailable, use PCA or t-SNE instead.
    random_state:
        Random seed for reproducibility.
    perplexity:
        t-SNE perplexity. Automatically capped for very small datasets.
    n_neighbors:
        UMAP neighborhood size.
    min_dist:
        UMAP minimum distance.

    Returns
    -------
    pd.DataFrame
        Columns: dim_1, dim_2.
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional. Got shape {X_arr.shape}.")
    if X_arr.shape[0] < 2:
        raise ValueError("At least two samples are required for a 2D projection.")

    method = method.lower()

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(X_arr)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        safe_perplexity = min(float(perplexity), max(1.0, (X_arr.shape[0] - 1) / 3))
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=safe_perplexity,
            init="pca",
            learning_rate="auto",
        )
        coords = reducer.fit_transform(X_arr)

    elif method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "UMAP projection requires the optional package `umap-learn`. "
                "Install it or use method='pca' or method='tsne'."
            ) from exc

        safe_neighbors = min(int(n_neighbors), max(2, X_arr.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=safe_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        coords = reducer.fit_transform(X_arr)

    else:
        raise ValueError("method must be one of: 'pca', 'tsne', 'umap'")

    return pd.DataFrame(coords, columns=["dim_1", "dim_2"])


def plot_embedding_projection(
    projection_df: pd.DataFrame,
    hue_col: str,
    style_col: str | None = None,
    size_col: str | None = None,
    title: str = "2D Representation Projection",
    subtitle: str | None = None,
    palette: dict | str | None = None,
    alpha: float = 0.85,
    s: int = 55,
    figsize: tuple[int, int] = (8, 6),
):
    """
    Plot a 2D PCA/t-SNE/UMAP projection with thesis styling.

    The function expects a DataFrame containing `dim_1`, `dim_2`, and a hue
    column. It does not compute the projection itself; use `compute_2d_projection`
    first when needed.
    """
    _require_columns(projection_df, ["dim_1", "dim_2", hue_col])

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=projection_df,
        x="dim_1",
        y="dim_2",
        hue=hue_col,
        style=style_col,
        size=size_col,
        palette=palette,
        alpha=alpha,
        s=s,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_title(title)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555")
    ax.set_xlabel("Projection dimension 1")
    ax.set_ylabel("Projection dimension 2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_department_embedding_projection(
    X,
    metadata_df: pd.DataFrame,
    department_col: str = "department",
    method: str = "pca",
    title: str | None = None,
    random_state: int = 42,
    figsize: tuple[int, int] = (9, 6),
):
    """
    Project feature/model embeddings and color contracts by department.

    Thesis use:
    This plot is a qualitative diagnostic of task heterogeneity. If departments
    form distinct clusters, it supports the claim that procurement departments
    behave like different tasks and that department-specific adaptation may be
    useful. If departments strongly overlap, it suggests that Stage 1 weak
    supervision may already capture much of the shared contract structure.
    """
    _require_columns(metadata_df, [department_col])
    projection_df = compute_2d_projection(X, method=method, random_state=random_state)
    projection_df[department_col] = metadata_df[department_col].reset_index(drop=True).astype(str)

    return plot_embedding_projection(
        projection_df=projection_df,
        hue_col=department_col,
        title=title or f"{method.upper()} Projection of Contract Representation by Department",
        subtitle="Qualitative diagnostic of department/task structure; not a performance metric.",
        palette=None,
        figsize=figsize,
    )


def plot_gold_label_embedding_projection(
    X,
    metadata_df: pd.DataFrame,
    label_col: str = "gold_y",
    department_col: str | None = "department",
    target_department: str | None = None,
    method: str = "pca",
    title: str | None = None,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 6),
):
    """
    Project feature/model embeddings and color points by gold labels.

    Optionally filters to one department, for example Logistics. This is useful
    for checking whether Stage 1 already separates the target department's gold
    labels or whether adaptation is needed.
    """
    required = [label_col]
    if target_department is not None and department_col is not None:
        required.append(department_col)
    _require_columns(metadata_df, required)

    meta = metadata_df.reset_index(drop=True).copy()
    X_arr = np.asarray(X)

    if target_department is not None and department_col is not None:
        mask = meta[department_col].astype(str).eq(str(target_department)).to_numpy()
        meta = meta.loc[mask].reset_index(drop=True)
        X_arr = X_arr[mask]

    projection_df = compute_2d_projection(X_arr, method=method, random_state=random_state)
    projection_df[label_col] = meta[label_col].map({0: "No", 1: "Yes"}).fillna(meta[label_col].astype(str))
    if department_col is not None and department_col in meta.columns:
        projection_df[department_col] = meta[department_col].astype(str)

    label_palette = {"No": THESIS_PALETTE["blue"], "Yes": THESIS_PALETTE["vermillion"]}
    dept_text = f" — {target_department}" if target_department else ""

    return plot_embedding_projection(
        projection_df=projection_df,
        hue_col=label_col,
        title=title or f"{method.upper()} Projection by Gold Label{dept_text}",
        subtitle="Qualitative diagnostic of label separability; main evidence should come from repeated metrics.",
        palette=label_palette,
        figsize=figsize,
    )


def plot_pre_post_adaptation_projection(
    pre_embeddings,
    post_embeddings,
    metadata_df: pd.DataFrame,
    label_col: str = "gold_y",
    method: str = "pca",
    title: str = "Representation Shift Before and After Adaptation",
    random_state: int = 42,
    figsize: tuple[int, int] = (12, 5),
):
    """
    Compare pre-adaptation and post-adaptation embeddings in a shared 2D space.

    The reducer is fitted on the concatenated pre/post embeddings so both panels
    share the same coordinate system. This makes the figure useful for asking:
    did few-shot adaptation change the representation in a way that makes the
    target labels more separable?

    This is a qualitative representation diagnostic and should be interpreted
    together with AUROC, log-loss, ECE, Precision@K and NDCG@K.
    """
    _require_columns(metadata_df, [label_col])
    pre_arr = np.asarray(pre_embeddings)
    post_arr = np.asarray(post_embeddings)
    if pre_arr.shape != post_arr.shape:
        raise ValueError(f"pre_embeddings and post_embeddings must have the same shape. Got {pre_arr.shape} and {post_arr.shape}.")

    combined = np.vstack([pre_arr, post_arr])
    projection = compute_2d_projection(combined, method=method, random_state=random_state)
    n = pre_arr.shape[0]

    meta = metadata_df.reset_index(drop=True).copy()
    label_values = meta[label_col].map({0: "No", 1: "Yes"}).fillna(meta[label_col].astype(str))

    plot_df = pd.concat([
        projection.iloc[:n].assign(stage="Pre-adaptation", label=label_values.values),
        projection.iloc[n:].assign(stage="Post-adaptation", label=label_values.values),
    ], ignore_index=True)

    label_palette = {"No": THESIS_PALETTE["blue"], "Yes": THESIS_PALETTE["vermillion"]}

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, stage in zip(axes, ["Pre-adaptation", "Post-adaptation"]):
        stage_df = plot_df.loc[plot_df["stage"] == stage]
        sns.scatterplot(
            data=stage_df,
            x="dim_1",
            y="dim_2",
            hue="label",
            palette=label_palette,
            alpha=0.9,
            s=60,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
        )
        ax.set_title(stage)
        ax.set_xlabel("Projection dimension 1")
        ax.set_ylabel("Projection dimension 2")
        despine_thesis(ax)
        ax.legend(title=label_col, frameon=False)

    fig.suptitle(title)
    plt.tight_layout()
    return fig, axes


def plot_embedding_shift_magnitude(
    pre_embeddings,
    post_embeddings,
    metadata_df: pd.DataFrame | None = None,
    group_col: str | None = None,
    title: str = "Embedding Shift Magnitude After Adaptation",
    figsize: tuple[int, int] = (8, 5),
):
    """
    Plot how much each observation moved in embedding space after adaptation.

    If `group_col` is provided, shift magnitudes are shown by group, for example
    by gold label or department. This can reveal whether adaptation changes the
    representation more strongly for one class or task subset.
    """
    pre_arr = np.asarray(pre_embeddings)
    post_arr = np.asarray(post_embeddings)
    if pre_arr.shape != post_arr.shape:
        raise ValueError(f"pre_embeddings and post_embeddings must have the same shape. Got {pre_arr.shape} and {post_arr.shape}.")

    shift = np.linalg.norm(post_arr - pre_arr, axis=1)
    plot_df = pd.DataFrame({"shift_magnitude": shift})

    if metadata_df is not None and group_col is not None:
        _require_columns(metadata_df, [group_col])
        plot_df[group_col] = metadata_df.reset_index(drop=True)[group_col].values

    fig, ax = plt.subplots(figsize=figsize)
    if group_col is not None and group_col in plot_df.columns:
        sns.boxplot(data=plot_df, x=group_col, y="shift_magnitude", ax=ax, showfliers=False)
        sns.stripplot(data=plot_df, x=group_col, y="shift_magnitude", ax=ax, color="black", alpha=0.45, size=3, jitter=True)
        ax.set_xlabel(group_col)
    else:
        sns.histplot(data=plot_df, x="shift_magnitude", bins=30, kde=True, ax=ax)
        ax.set_xlabel("Embedding shift magnitude")
    ax.set_title(title)
    ax.set_ylabel("Shift magnitude")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax
