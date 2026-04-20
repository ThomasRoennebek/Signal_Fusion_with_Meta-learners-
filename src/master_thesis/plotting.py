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

DTU_PALETTE = {
    "primary": "#990000",      # Corporate red
    "blue": "#2F3EEA",         # DTU blue
    "navy": "#030F4F",         # DTU navy blue
    "green": "#1FD082",        # Bright green
    "yellow": "#F6D04D",       # Yellow
    "orange": "#FC7634",       # Orange
    "pink": "#F7BBB1",         # Pink
    "red": "#E83F48",          # Secondary red
    "dark_green": "#008835",   # Green
    "purple": "#79238E",       # Purple
    "grey": "#DADADA",         # Grey
    "black": "#000000",
    "white": "#FFFFFF",
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

# Color mapping for data view labels used in SHAP view-level importance plots.
VIEW_COLORS: dict = {
    "financial": THESIS_PALETTE["blue"],
    "procurement": THESIS_PALETTE["green"],
    "supplier": THESIS_PALETTE["orange"],
    "contract": THESIS_PALETTE["vermillion"],
    "temporal": THESIS_PALETTE["sky_blue"],
    "text": THESIS_PALETTE["purple"],
    "meta": THESIS_PALETTE["grey"],
}


def set_thesis_style(
    palette: str = "thesis",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    font_scale: float = 1.15,
    use_latex_if_available: bool = False,
    grid: bool = False,
) -> dict[str, str]:
    """
    Apply a clean, publication-ready plotting style for all thesis figures.

    Parameters
    ----------
    palette : str
        The color palette to use. Options: "thesis" (standard colorblind-safe) 
        or "dtu" (Technical University of Denmark branding).
    ...
    """
    latex_available = shutil.which("latex") is not None
    use_latex = use_latex_if_available and latex_available

    sns.set_theme(
        context="paper",
        style="whitegrid" if grid else "white",
        font=font_name,
        font_scale=font_scale,
    )

    # Determine color cycle based on palette
    if palette.lower() == "dtu":
        color_cycle = [
            DTU_PALETTE["primary"],
            DTU_PALETTE["blue"],
            DTU_PALETTE["green"],
            DTU_PALETTE["orange"],
            DTU_PALETTE["yellow"],
            DTU_PALETTE["purple"],
            DTU_PALETTE["navy"],
            DTU_PALETTE["grey"],
        ]
        METHOD_COLORS.update({
            "finetune": DTU_PALETTE["blue"],
            "fine_tune": DTU_PALETTE["blue"],
            "fine-tune": DTU_PALETTE["blue"],
            "anil": DTU_PALETTE["green"],
            "fomaml": DTU_PALETTE["orange"],
            "maml": DTU_PALETTE["primary"],
            "zero_shot": DTU_PALETTE["grey"],
            "zero-shot": DTU_PALETTE["grey"],
            "pretrained": DTU_PALETTE["navy"],
        })
    else:
        color_cycle = [
            THESIS_PALETTE["blue"],
            THESIS_PALETTE["vermillion"],
            THESIS_PALETTE["green"],
            THESIS_PALETTE["orange"],
            THESIS_PALETTE["sky_blue"],
            THESIS_PALETTE["purple"],
            THESIS_PALETTE["grey"],
        ]

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
        # Color cycle
        "axes.prop_cycle": plt.cycler(color=color_cycle),
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
    group_col: str = "method",
    method_col: str | None = None,
    title: str = "Stage 2 Adaptation Benchmark",
    figsize: tuple[int, int] = (11, 6),
):
    """
    Plot Stage 2 group comparison across core metrics.

    Parameters
    ----------
    results_df
        Summary DataFrame with one row per experiment.
    metrics
        Metric column names to include.
    group_col
        Column to use as the grouping variable (hue). Default is "method".
        Accepts any categorical column, e.g. "method", "init_name", "support_label".
    method_col
        Deprecated alias for group_col. If provided, takes precedence over group_col
        for backward compatibility.
    title
        Plot title.
    figsize
        Figure size tuple (width, height).
    """
    # backward-compat alias
    if method_col is not None:
        group_col = method_col

    available_metrics = [m for m in metrics if m in results_df.columns]
    if not available_metrics:
        raise ValueError(f"None of the requested metrics are present in results_df. Requested: {list(metrics)}")
    _require_columns(results_df, [group_col])

    long_df = results_df[[group_col, *available_metrics]].melt(
        id_vars=group_col,
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=figsize)
    unique_groups = results_df[group_col].unique()
    # Use method palette when grouping by method, otherwise fall back to tab palette.
    if group_col == "method":
        palette = method_palette(unique_groups)
    else:
        palette = dict(zip(unique_groups, sns.color_palette("tab10", n_colors=len(unique_groups))))

    sns.barplot(data=long_df, x="metric", y="value", hue=group_col, palette=palette, errorbar="sd", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean value")
    legend_title = group_col.replace("_", " ").title()
    ax.legend(title=legend_title, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
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
    method_col: str | None = "method",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """Plot performance as a function of inner-loop adaptation steps.

    Parameters
    ----------
    results_df
        Summary DataFrame.
    metric_col
        Metric column to plot on the y-axis.
    step_col
        Column containing inner-step counts (x-axis).
    method_col
        Optional column for grouping lines by method. If None or not present
        in the DataFrame, a single line is drawn without grouping.
    """
    required = [metric_col, step_col]
    use_hue = method_col is not None and method_col in results_df.columns
    if use_hue:
        required.append(method_col)
    _require_columns(results_df, required)

    fig, ax = plt.subplots(figsize=figsize)
    if use_hue:
        palette = method_palette(results_df[method_col].unique())
        sns.lineplot(data=results_df, x=step_col, y=metric_col, hue=method_col,
                     marker="o", errorbar="sd", palette=palette, ax=ax)
        ax.legend(title="Method", frameon=False)
    else:
        sns.lineplot(data=results_df, x=step_col, y=metric_col,
                     marker="o", errorbar="sd", ax=ax)
    ax.set_title(title or f"Inner-Step Adaptation Curve ({metric_col})")
    ax.set_xlabel("Inner-loop gradient steps")
    ax.set_ylabel(metric_col)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_prediction_probability_distribution(
    predictions_df: pd.DataFrame,
    prob_col: str = "y_score",
    method_col: str | None = "method",
    label_col: str | None = None,
    title: str = "Prediction Probability Distribution",
    figsize: tuple[int, int] = (8, 5),
):
    """Plot predicted probability distributions to diagnose overconfidence.

    Parameters
    ----------
    predictions_df
        DataFrame with at least a predicted probability column.
    prob_col
        Column containing predicted probabilities.
    method_col
        Optional column for grouping by method. If None or not present in the
        DataFrame, predictions are plotted as a single distribution.
    label_col
        Optional binary label column used to draw a base-rate reference line.
    """
    _require_columns(predictions_df, [prob_col])
    use_hue = method_col is not None and method_col in predictions_df.columns

    fig, ax = plt.subplots(figsize=figsize)
    if use_hue:
        sns.kdeplot(data=predictions_df, x=prob_col, hue=method_col, common_norm=False, fill=False, ax=ax)
    else:
        sns.kdeplot(data=predictions_df, x=prob_col, common_norm=False, fill=False, ax=ax)
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


def plot_support_size_curve(
    results_df: pd.DataFrame,
    metric_col: str = "gold_auroc_mean",
    support_pos_col: str = "n_support_pos",
    support_neg_col: str = "n_support_neg",
    method_col: str = "method",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """
    Plot performance as a function of support set size.

    Constructs a label 'k_pos/k_neg' from support columns and plots a
    line per method.
    """
    _require_columns(results_df, [metric_col, support_pos_col, support_neg_col, method_col])
    plot_df = results_df.copy()
    plot_df["support_label"] = (
        plot_df[support_pos_col].astype(str) + "/" + plot_df[support_neg_col].astype(str)
    )

    fig, ax = plt.subplots(figsize=figsize)
    palette = method_palette(plot_df[method_col].unique())
    sns.lineplot(
        data=plot_df,
        x="support_label",
        y=metric_col,
        hue=method_col,
        marker="o",
        errorbar="sd",
        palette=palette,
        ax=ax,
    )
    ax.set_title(title or f"Support-Size Sensitivity ({metric_col})")
    ax.set_xlabel("Support size (pos/neg contracts)")
    ax.set_ylabel(metric_col)
    ax.legend(title="Method", frameon=False)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Stage 2 diagnostics — training dynamics, collapse, embeddings, SHAP
# -----------------------------------------------------------------------------

def plot_meta_training_curve(
    history_df: pd.DataFrame,
    iter_col: str = "iteration",
    loss_col: str = "meta_loss",
    method_col: str | None = "method",
    smooth_window: int = 5,
    title: str = "Meta-Training Loss",
    figsize: tuple[int, int] = (9, 4.5),
):
    """
    Plot outer-loop meta-loss across iterations with optional rolling smoothing.

    Accepts either a single-run history (no method column) or a concatenated
    history across methods (with a ``method`` column).
    """
    _require_columns(history_df, [iter_col, loss_col])
    plot_df = history_df.copy()

    if smooth_window and smooth_window > 1:
        group_keys = [method_col] if (method_col and method_col in plot_df.columns) else None
        if group_keys:
            plot_df["_smoothed"] = (
                plot_df.groupby(group_keys)[loss_col]
                .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
            )
        else:
            plot_df["_smoothed"] = plot_df[loss_col].rolling(smooth_window, min_periods=1).mean()
        y_col = "_smoothed"
        ylabel = f"Meta-loss (rolling mean, window={smooth_window})"
    else:
        y_col = loss_col
        ylabel = "Meta-loss"

    fig, ax = plt.subplots(figsize=figsize)
    if method_col and method_col in plot_df.columns:
        palette = method_palette(plot_df[method_col].unique())
        sns.lineplot(
            data=plot_df, x=iter_col, y=y_col, hue=method_col, ax=ax, palette=palette, linewidth=1.8
        )
        ax.legend(title="Method", frameon=False)
    else:
        ax.plot(plot_df[iter_col], plot_df[y_col], color=THESIS_PALETTE["blue"], linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(ylabel)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_collapse_rate(
    metrics_df: pd.DataFrame,
    method_col: str = "method",
    prob_mean_col: str = "pred_mean",
    prob_std_col: str = "pred_std",
    collapse_std_threshold: float = 0.01,
    title: str = "Episode Collapse Rate by Method",
    figsize: tuple[int, int] = (8, 5),
):
    """
    Bar chart of the fraction of episodes where predictions collapse to a
    near-constant value (``pred_std < collapse_std_threshold``).

    Expects a long-form episode-level metrics frame with a ``method`` column and
    per-episode prediction statistics. If ``pred_std`` is missing, falls back to
    flagging episodes where ``pred_mean`` is extreme (< 0.01 or > 0.99).
    """
    plot_df = metrics_df.copy()

    if prob_std_col in plot_df.columns:
        plot_df["_collapsed"] = (plot_df[prob_std_col] < collapse_std_threshold).astype(int)
    elif prob_mean_col in plot_df.columns:
        plot_df["_collapsed"] = (
            (plot_df[prob_mean_col] < 0.01) | (plot_df[prob_mean_col] > 0.99)
        ).astype(int)
    else:
        raise ValueError(
            f"Neither '{prob_std_col}' nor '{prob_mean_col}' present in metrics_df."
        )

    if method_col not in plot_df.columns:
        raise ValueError(f"Column '{method_col}' not in metrics_df.")

    agg = (
        plot_df.groupby(method_col)
        .agg(collapse_rate=("_collapsed", "mean"), n_episodes=("_collapsed", "count"))
        .reset_index()
        .sort_values("collapse_rate", ascending=True)
    )

    fig, ax = plt.subplots(figsize=figsize)
    palette = method_palette(agg[method_col].unique())
    colors = [palette.get(_normalise_method_name(m), THESIS_PALETTE["grey"]) for m in agg[method_col]]
    ax.barh(agg[method_col], agg["collapse_rate"], color=colors)
    for i, (rate, n) in enumerate(zip(agg["collapse_rate"], agg["n_episodes"])):
        ax.text(rate + 0.01, i, f"{rate:.0%} (n={n})", va="center", fontsize=9)
    ax.set_xlim(0, max(1.0, float(agg["collapse_rate"].max() + 0.1)))
    ax.set_title(title)
    ax.set_xlabel("Fraction of collapsed episodes")
    ax.set_ylabel("Method")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_episode_prediction_stats(
    predictions_df: pd.DataFrame,
    method_col: str = "method",
    episode_col: str = "episode_idx",
    prob_col: str = "y_prob",
    title: str = "Per-Episode Prediction Spread",
    figsize: tuple[int, int] = (10, 5),
):
    """
    Boxplot of predicted probabilities per episode grouped by method.

    Useful to diagnose episodes where predictions collapse to a constant.
    """
    _require_columns(predictions_df, [method_col, episode_col, prob_col])

    fig, ax = plt.subplots(figsize=figsize)
    palette = method_palette(predictions_df[method_col].unique())
    sns.boxplot(
        data=predictions_df,
        x=episode_col,
        y=prob_col,
        hue=method_col,
        ax=ax,
        palette=palette,
        fliersize=2,
    )
    ax.set_title(title)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Predicted probability")
    ax.legend(title="Method", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def _scatter_2d_by_group(
    coords: np.ndarray,
    groups,
    ax,
    palette: dict | None = None,
    alpha: float = 0.75,
    s: float = 32.0,
    legend_title: str | None = None,
):
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    unique_groups = list(dict.fromkeys(groups.tolist()))
    if palette is None:
        base_colors = [
            THESIS_PALETTE["blue"],
            THESIS_PALETTE["vermillion"],
            THESIS_PALETTE["green"],
            THESIS_PALETTE["orange"],
            THESIS_PALETTE["purple"],
            THESIS_PALETTE["sky_blue"],
            THESIS_PALETTE["yellow"],
            THESIS_PALETTE["grey"],
        ]
        palette = {g: base_colors[i % len(base_colors)] for i, g in enumerate(unique_groups)}
    for g in unique_groups:
        mask = (groups == g).to_numpy()
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=palette.get(g, THESIS_PALETTE["grey"]),
            label=str(g),
            alpha=alpha,
            s=s,
            edgecolor="white",
            linewidths=0.4,
        )
    if legend_title:
        ax.legend(title=legend_title, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")


def plot_department_embedding_projection(
    coords: np.ndarray,
    departments: Sequence[str],
    projection_method: str = "pca",
    title: str | None = None,
    figsize: tuple[int, int] = (8, 6),
):
    """
    Scatter of 2D embeddings coloured by department.

    ``coords`` must be an (n, 2) array from ``explainability.compute_2d_projection``.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got {coords.shape}")
    fig, ax = plt.subplots(figsize=figsize)
    _scatter_2d_by_group(coords, departments, ax, legend_title="Department")
    ax.set_title(title or f"Stage 1 Embedding Projection ({projection_method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_gold_label_embedding_projection(
    coords: np.ndarray,
    labels: Sequence[int],
    projection_method: str = "pca",
    title: str | None = None,
    figsize: tuple[int, int] = (7, 6),
):
    """
    Scatter of 2D embeddings coloured by gold label (0/1).
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got {coords.shape}")
    palette = {
        "0": THESIS_PALETTE["blue"],
        "1": THESIS_PALETTE["vermillion"],
        "nan": THESIS_PALETTE["light_grey"],
    }
    fig, ax = plt.subplots(figsize=figsize)
    _scatter_2d_by_group(coords, labels, ax, palette=palette, legend_title="gold_y")
    ax.set_title(title or f"Gold-Label Separability ({projection_method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_pre_post_adaptation_projection(
    coords_pre: np.ndarray,
    coords_post: np.ndarray,
    groups: Sequence,
    group_title: str = "gold_y",
    projection_method: str = "pca",
    title: str = "Representation Shift Before/After Adaptation",
    figsize: tuple[int, int] = (12, 5),
):
    """
    Two-panel scatter of 2D embeddings before and after adaptation.

    Both panels are coloured by the same ``groups`` (usually gold labels or
    department). Fit one projection on the stacked pre+post embeddings upstream
    and split here so the two panels share a comparable coordinate space.
    """
    if coords_pre.shape != coords_post.shape:
        raise ValueError(
            f"Shape mismatch: pre={coords_pre.shape}, post={coords_post.shape}"
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    _scatter_2d_by_group(coords_pre, groups, axes[0], legend_title=None)
    _scatter_2d_by_group(coords_post, groups, axes[1], legend_title=group_title)
    axes[0].set_title("Pre-adaptation")
    axes[1].set_title("Post-adaptation")
    for ax in axes:
        ax.set_xlabel("Component 1")
        despine_thesis(ax)
    axes[0].set_ylabel("Component 2")
    fig.suptitle(f"{title} ({projection_method.upper()})", y=1.02)
    plt.tight_layout()
    return fig, axes


def plot_embedding_shift_magnitude(
    shifts: np.ndarray,
    labels: Sequence | None = None,
    title: str = "Embedding Shift Magnitude After Adaptation",
    figsize: tuple[int, int] = (8, 5),
    bins: int = 30,
):
    """
    Histogram (optionally split by label) of Euclidean embedding shifts.
    """
    shifts = np.asarray(shifts).ravel()
    fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        ax.hist(shifts, bins=bins, color=THESIS_PALETTE["blue"], alpha=0.85, edgecolor="white")
        ax.set_ylabel("Number of samples")
    else:
        labels_s = pd.Series(labels).astype(str)
        palette = method_palette(labels_s.unique())
        for g in labels_s.unique():
            ax.hist(
                shifts[labels_s.values == g],
                bins=bins,
                alpha=0.55,
                label=str(g),
                color=palette.get(g, THESIS_PALETTE["grey"]),
                edgecolor="white",
            )
        ax.legend(title="Group", frameon=False)
        ax.set_ylabel("Number of samples")

    ax.set_xlabel("Euclidean distance between pre and post embeddings")
    ax.set_title(title)
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_shap_summary_bar(
    feature_ranking_df: pd.DataFrame,
    feature_col: str = "feature",
    importance_col: str = "mean_abs_shap",
    top_n: int = 20,
    title: str = "Global SHAP Importance",
    figsize: tuple[int, int] = (8, 7),
):
    """
    Horizontal bar chart of mean |SHAP| per feature (global importance).

    Designed to consume the output of ``ShapResult.mean_abs()`` or
    ``explainability.shap_feature_ranking``.
    """
    _require_columns(feature_ranking_df, [feature_col, importance_col])
    plot_df = feature_ranking_df.copy().head(top_n).sort_values(importance_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(plot_df[feature_col], plot_df[importance_col], color=THESIS_PALETTE["blue"])
    ax.set_title(title)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


def plot_shap_local_bar(
    shap_values_row: np.ndarray,
    feature_names: Sequence[str],
    feature_values: Sequence | None = None,
    top_n: int = 15,
    title: str = "Local SHAP Attribution",
    figsize: tuple[int, int] = (9, 6),
):
    """
    Single-sample SHAP explanation as a signed horizontal bar chart.

    Parameters
    ----------
    shap_values_row:
        (n_features,) SHAP values for one sample.
    feature_names:
        Ordered feature names, same length as ``shap_values_row``.
    feature_values:
        Optional original feature values to annotate on the y-axis.
    """
    shap_values_row = np.asarray(shap_values_row).ravel()
    if len(shap_values_row) != len(feature_names):
        raise ValueError(
            f"Length mismatch: shap_values_row={len(shap_values_row)}, "
            f"feature_names={len(feature_names)}"
        )

    df = pd.DataFrame({"feature": list(feature_names), "shap": shap_values_row})
    if feature_values is not None and len(feature_values) == len(feature_names):
        df["value"] = list(feature_values)
        df["label"] = df["feature"] + " = " + df["value"].astype(str)
    else:
        df["label"] = df["feature"]
    df["abs_shap"] = df["shap"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).sort_values("shap", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [
        THESIS_PALETTE["green"] if v >= 0 else THESIS_PALETTE["vermillion"] for v in df["shap"]
    ]
    ax.barh(df["label"], df["shap"], color=colors)
    ax.axvline(0, color=THESIS_PALETTE["black"], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("SHAP contribution (logit space)")
    ax.set_ylabel("Feature")
    despine_thesis(ax)
    plt.tight_layout()
    return fig, ax


# =============================================================================
# Stage 2 — Thesis benchmark plotting functions
# =============================================================================

def plot_method_comparison_bars(
    summary_df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    metric_labels: dict[str, str] | None = None,
    method_order: Sequence[str] | None = None,
    zero_shot_ref_row: pd.Series | None = None,
    figsize: tuple[int, int] = (14, 9),
    title: str = "Method Comparison — Stage 2 Benchmark",
    save_path: str | None = None,
) -> tuple:
    """
    Four-panel grouped bar chart comparing methods across key metrics.

    Each panel shows one metric (AUROC, NDCG@10, Log-Loss, ECE) with one
    bar per method.  Error bars represent ±1 std across evaluation episodes.
    A dashed horizontal line marks the zero-shot reference where applicable.

    Parameters
    ----------
    summary_df:
        DataFrame with columns ``method``, ``inner_steps``, and metric columns
        ending in ``_mean`` / ``_std`` (e.g. ``auroc_mean``, ``auroc_std``).
        When multiple rows exist per method, the best row by ``auroc_mean`` is
        used automatically.
    metrics:
        Metric base names to plot.  Defaults to
        ``["auroc", "ndcg_at_10", "log_loss", "ece"]``.
    metric_labels:
        Human-readable axis labels keyed by metric base name.
    method_order:
        Ordered list of method names for the x-axis.
    zero_shot_ref_row:
        Optional single-row Series with zero-shot reference values (same
        ``_mean`` column scheme); adds a dashed reference line to each panel.
    save_path:
        If provided, save figure to this path at 300 dpi.

    Returns
    -------
    fig, axes : tuple
    """
    if metrics is None:
        metrics = ["auroc", "ndcg_at_10", "log_loss", "ece"]
    if metric_labels is None:
        metric_labels = {
            "auroc": "AUROC",
            "ndcg_at_10": "NDCG@10",
            "log_loss": "Log-Loss",
            "ece": "ECE",
        }

    df = summary_df.copy()

    # Pick best row per method (highest auroc_mean)
    auroc_col = "auroc_mean" if "auroc_mean" in df.columns else "auroc"
    if "method" in df.columns:
        best_idx = df.groupby("method")[auroc_col].idxmax()
        df = df.loc[best_idx].reset_index(drop=True)

    if method_order is None:
        # Maintain canonical order; filter to present methods
        canonical = ["zero_shot", "finetune", "anil", "fomaml", "maml"]
        present = set(df["method"].tolist()) if "method" in df.columns else set()
        method_order = [m for m in canonical if m in present] + [
            m for m in present if m not in canonical
        ]

    palette = method_palette(method_order)
    n_methods = len(method_order)
    x = np.arange(n_methods)
    bar_colors = [palette.get(m, THESIS_PALETTE["grey"]) for m in method_order]

    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).ravel()

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        mean_col = f"{metric}_mean" if f"{metric}_mean" in df.columns else metric
        std_col = f"{metric}_std" if f"{metric}_std" in df.columns else None

        means, stds = [], []
        for m in method_order:
            row = df[df["method"] == m] if "method" in df.columns else pd.DataFrame()
            if len(row) == 0:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row[mean_col].iloc[0]) if mean_col in row.columns else np.nan)
                stds.append(float(row[std_col].iloc[0]) if std_col and std_col in row.columns else 0.0)

        means_arr = np.array(means)
        stds_arr = np.array(stds)

        bars = ax.bar(
            x, means_arr, color=bar_colors, edgecolor="white",
            linewidth=0.8, alpha=0.88,
            yerr=stds_arr, capsize=4, error_kw={"linewidth": 1.2, "ecolor": THESIS_PALETTE["black"]},
        )

        # Zero-shot reference line
        if zero_shot_ref_row is not None:
            ref_col = f"{metric}_mean" if f"{metric}_mean" in zero_shot_ref_row.index else metric
            if ref_col in zero_shot_ref_row.index:
                ref_val = float(zero_shot_ref_row[ref_col])
                ax.axhline(ref_val, color=THESIS_PALETTE["grey"], linestyle="--",
                           linewidth=1.4, alpha=0.8, zorder=0, label="Zero-shot ref.")

        ax.set_title(metric_labels.get(metric, metric), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", " ").title() for m in method_order],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=9)
        despine_thesis(ax)

    # Hide any spare axes
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_lr_sweep_comparison(
    summary_df: pd.DataFrame,
    methods: Sequence[str] | None = None,
    metric: str = "auroc",
    metric_label: str | None = None,
    inner_lr_col: str = "inner_lr",
    inner_steps_col: str = "inner_steps",
    steps_to_show: Sequence[int] | None = None,
    zero_shot_ref: float | None = None,
    figsize: tuple[int, int] = (12, 4),
    title: str | None = None,
    save_path: str | None = None,
) -> tuple:
    """
    Line plots of metric vs. inner_lr, facetted by method and coloured by
    inner_steps.

    Intended to visualise the LR-sweep results (FOMAML, MAML, ANIL) and show
    how the optimal inner learning rate changes with gradient-step count.

    Parameters
    ----------
    summary_df:
        DataFrame containing ``method``, ``inner_lr``, ``inner_steps``, and
        metric columns (``{metric}_mean`` / ``{metric}_std`` or plain
        ``{metric}``).
    methods:
        Which methods to plot.  Defaults to ``["anil", "fomaml", "maml"]``.
    metric:
        Base metric name (e.g. ``"auroc"``).
    zero_shot_ref:
        Optional scalar — draws a dashed horizontal reference line on each
        panel (e.g. zero-shot AUROC=0.634).
    steps_to_show:
        Inner-step counts to include.  Defaults to all unique values.

    Returns
    -------
    fig, axes : tuple
    """
    if methods is None:
        methods = ["anil", "fomaml", "maml"]
    if metric_label is None:
        metric_label = metric.upper().replace("_", " ").replace("AT", "@")
    if title is None:
        title = f"{metric_label} vs. Inner LR — LR Sweep Benchmark"

    mean_col = f"{metric}_mean" if f"{metric}_mean" in summary_df.columns else metric
    std_col = f"{metric}_std" if f"{metric}_std" in summary_df.columns else None

    df = summary_df.copy()

    # Normalise inner_lr to float
    if inner_lr_col in df.columns:
        df[inner_lr_col] = df[inner_lr_col].astype(float)

    if steps_to_show is None and inner_steps_col in df.columns:
        steps_to_show = sorted(df[inner_steps_col].dropna().unique().tolist())
    elif steps_to_show is None:
        steps_to_show = []

    step_colors = {
        s: c for s, c in zip(
            steps_to_show,
            [THESIS_PALETTE["blue"], THESIS_PALETTE["orange"], THESIS_PALETTE["vermillion"],
             THESIS_PALETTE["green"], THESIS_PALETTE["purple"]],
        )
    }

    methods_present = [m for m in methods if "method" not in df.columns or m in df["method"].values]
    n = len(methods_present)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_present):
        mdf = df[df["method"] == method].copy() if "method" in df.columns else df.copy()

        for steps in steps_to_show:
            sdf = mdf[mdf[inner_steps_col] == steps].copy() if inner_steps_col in mdf.columns else mdf.copy()
            if sdf.empty:
                continue
            sdf = sdf.sort_values(inner_lr_col)
            # Drop rows where inner_lr is NaN or non-positive (safe for log scale)
            sdf = sdf[sdf[inner_lr_col].notna() & (sdf[inner_lr_col] > 0)]
            if sdf.empty:
                continue
            lr_vals = sdf[inner_lr_col].values
            mean_vals = sdf[mean_col].values
            std_vals = sdf[std_col].values if std_col and std_col in sdf.columns else np.zeros_like(mean_vals)

            color = step_colors.get(steps, THESIS_PALETTE["grey"])
            ax.plot(lr_vals, mean_vals, marker="o", color=color, linewidth=2,
                    markersize=6, label=f"steps={steps}")
            ax.fill_between(lr_vals, mean_vals - std_vals, mean_vals + std_vals,
                            alpha=0.15, color=color)

        if zero_shot_ref is not None:
            ax.axhline(zero_shot_ref, color=THESIS_PALETTE["grey"], linestyle="--",
                       linewidth=1.4, alpha=0.8, label="Zero-shot ref.")

        # Only use log scale when the axis has data with positive x-values
        if ax.lines and any(
            line.get_xdata() is not None and len(line.get_xdata()) > 0
            for line in ax.lines
        ):
            ax.set_xscale("log")
        ax.set_xlabel("Inner LR", fontsize=10)
        ax.set_title(method.upper(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)
        despine_thesis(ax)

    axes[0].set_ylabel(metric_label, fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    try:
        plt.tight_layout()
    except ValueError:
        pass  # tight_layout can fail on empty log-scale axes; layout is still usable

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_collapse_rate_heatmap(
    summary_df: pd.DataFrame,
    method: str = "fomaml",
    collapse_col: str = "collapse_rate",
    inner_lr_col: str = "inner_lr",
    inner_steps_col: str = "inner_steps",
    figsize: tuple[int, int] = (6, 4),
    title: str | None = None,
    save_path: str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> tuple:
    """
    Heatmap of prediction-collapse rate across inner_steps × inner_lr for one
    method.

    Collapse rate = fraction of evaluation episodes where all query
    predictions are nearly constant (pred_std < 0.01).  A value of 1.0 means
    the method collapsed in every episode; 0.0 means it never did.

    Parameters
    ----------
    summary_df:
        DataFrame with ``method``, ``inner_lr``, ``inner_steps``, and
        ``collapse_rate`` columns.
    method:
        The method to visualise (e.g. ``"fomaml"``).
    figsize:
        Figure size in inches.
    vmin / vmax:
        Colour scale limits (default 0–1).

    Returns
    -------
    fig, ax : tuple
    """
    if title is None:
        title = f"Collapse Rate — {method.upper()} (steps × inner_lr)"

    df = summary_df.copy()
    if "method" in df.columns:
        df = df[df["method"] == method].copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for method='{method}'",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax

    if inner_lr_col in df.columns:
        df[inner_lr_col] = df[inner_lr_col].astype(float)

    # Build pivot: rows = inner_steps, cols = inner_lr
    try:
        pivot = df.pivot_table(
            index=inner_steps_col, columns=inner_lr_col, values=collapse_col, aggfunc="mean"
        )
        pivot = pivot.sort_index().sort_index(axis=1)
    except Exception:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Could not pivot collapse-rate data.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pivot.values, aspect="auto", interpolation="nearest",
        vmin=vmin, vmax=vmax,
        cmap="RdYlGn_r",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{lr:.4g}" for lr in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(s) for s in pivot.index])
    ax.set_xlabel("Inner LR", fontsize=10)
    ax.set_ylabel("Inner Steps", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate cells
    for ri in range(len(pivot.index)):
        for ci in range(len(pivot.columns)):
            val = pivot.values[ri, ci]
            if not np.isnan(val):
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if val < 0.7 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Collapse Rate", fontsize=9)

    despine_thesis(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def normalize_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add short-form column aliases to a Stage 2 experiment summary DataFrame.

    The raw CSV uses ``gold_`` prefixed names for evaluation metrics
    (e.g. ``gold_auroc_mean``).  This function adds unprefixed aliases
    (e.g. ``auroc_mean``) so that downstream plotting functions — which
    expect the shorter names — work without modification.

    The function is idempotent: if the alias already exists it is not
    overwritten.

    Parameters
    ----------
    df : pd.DataFrame
        Raw experiment summary as loaded from ``experiment_summary.csv``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with additional alias columns (in-place copy).
    """
    df = df.copy()
    aliases = {
        "gold_auroc_mean":    "auroc_mean",
        "gold_auroc_std":     "auroc_std",
        "gold_logloss_mean":  "log_loss_mean",
        "gold_logloss_std":   "log_loss_std",
        "gold_ece_mean":      "ece_mean",
        "gold_ece_std":       "ece_std",
        "gold_ap_mean":       "ap_mean",
        "gold_ap_std":        "ap_std",
        "gold_brier_mean":    "brier_mean",
        "gold_brier_std":     "brier_std",
        "gold_f1_mean":       "f1_mean",
        "gold_f1_std":        "f1_std",
        "pred_mean_mean":     "pred_mean",
        "pred_std_mean":      "pred_std",
    }
    for src, dst in aliases.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    return df


def plot_pred_std_comparison(
    summary_df: pd.DataFrame,
    method_order: Sequence[str] | None = None,
    pred_std_col: str = "pred_std",
    collapse_rate_col: str = "collapse_rate",
    figsize: tuple[int, int] = (11, 4),
    title: str = "Prediction Spread vs. Collapse Rate by Method",
    save_path: str | None = None,
) -> tuple:
    """
    Two-panel comparison of prediction spread and collapse rate per method.

    Left panel: bar chart of mean ``pred_std`` (higher = more spread,
    lower = collapsed predictions).
    Right panel: bar chart of ``collapse_rate`` (fraction of episodes where
    all query predictions converge to a near-constant value).

    Both panels use DTU-palette method colours. The best row per method
    (highest AUROC) is selected when multiple rows exist.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with ``method``, ``pred_std``, ``collapse_rate``,
        and optionally ``auroc_mean``.
    method_order : list[str], optional
        Ordered method names for the x-axis.
    pred_std_col : str
        Column name for mean prediction standard deviation.
    collapse_rate_col : str
        Column name for per-experiment collapse rate.
    save_path : str, optional
        If provided, save figure at 300 dpi.

    Returns
    -------
    fig, axes : tuple
    """
    df = summary_df.copy()

    if method_order is None:
        canonical = ["zero_shot", "finetune", "anil", "fomaml", "maml"]
        present = set(df["method"].tolist()) if "method" in df.columns else set()
        method_order = [m for m in canonical if m in present] + [
            m for m in present if m not in canonical
        ]

    # Best row per method
    auroc_col = "auroc_mean" if "auroc_mean" in df.columns else "gold_auroc_mean"
    if "method" in df.columns and auroc_col in df.columns:
        best_idx = df.groupby("method")[auroc_col].idxmax()
        df = df.loc[best_idx]

    palette = method_palette(method_order)
    colors = [palette.get(m, THESIS_PALETTE["grey"]) for m in method_order]
    x = np.arange(len(method_order))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: pred_std
    ax = axes[0]
    if pred_std_col in df.columns and "method" in df.columns:
        vals = [
            float(df[df["method"] == m][pred_std_col].iloc[0])
            if m in df["method"].values else 0.0
            for m in method_order
        ]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, alpha=0.88)
        ax.axhline(0.01, color=THESIS_PALETTE["vermillion"], linestyle="--",
                   linewidth=1.2, label="Collapse threshold (0.01)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", " ").title() for m in method_order],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_ylabel("Mean Prediction Std. Dev.", fontsize=9)
        ax.set_title("Prediction Spread", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)
        despine_thesis(ax)
    else:
        ax.text(0.5, 0.5, f"Column '{pred_std_col}' not found",
                ha="center", va="center", transform=ax.transAxes)

    # Right: collapse_rate
    ax = axes[1]
    if collapse_rate_col in df.columns and "method" in df.columns:
        cr_vals = [
            float(df[df["method"] == m][collapse_rate_col].iloc[0])
            if m in df["method"].values else np.nan
            for m in method_order
        ]
        ax.bar(x, cr_vals, color=colors, edgecolor="white", linewidth=0.8, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", " ").title() for m in method_order],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_ylabel("Collapse Rate", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Collapse Rate per Method", fontsize=11, fontweight="bold")
        despine_thesis(ax)
    else:
        ax.text(0.5, 0.5, f"Column '{collapse_rate_col}' not found",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_ece_auroc_scatter(
    summary_df: pd.DataFrame,
    auroc_col: str = "auroc_mean",
    ece_col: str = "ece_mean",
    method_col: str = "method",
    inner_steps_col: str = "inner_steps",
    inner_lr_col: str = "inner_lr",
    method_order: Sequence[str] | None = None,
    figsize: tuple[int, int] = (7, 5),
    title: str = "Calibration vs. Discrimination Tradeoff",
    save_path: str | None = None,
) -> tuple:
    """
    Scatter plot of AUROC (x) vs. ECE (y) per experiment configuration.

    Each point represents one (method, steps, lr) combination.  Colour
    encodes method; shape optionally encodes inner_steps.  A good operating
    point sits in the top-right quadrant (high AUROC, low ECE).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with method, AUROC, and ECE columns.
    auroc_col, ece_col : str
        Column names for AUROC and ECE (after normalization).
    figsize : tuple
        Figure dimensions in inches.
    save_path : str, optional
        If provided, save at 300 dpi.

    Returns
    -------
    fig, ax : tuple
    """
    df = summary_df.copy()
    df = df.dropna(subset=[auroc_col, ece_col]) if auroc_col in df.columns and ece_col in df.columns else df

    if method_order is None:
        canonical = ["zero_shot", "finetune", "anil", "fomaml", "maml"]
        present = set(df[method_col].tolist()) if method_col in df.columns else set()
        method_order = [m for m in canonical if m in present] + [
            m for m in present if m not in canonical
        ]

    palette = method_palette(method_order)
    markers = {1: "o", 3: "s", 5: "^", 0: "D"}

    fig, ax = plt.subplots(figsize=figsize)

    for method in method_order:
        mdf = df[df[method_col] == method] if method_col in df.columns else df
        if mdf.empty:
            continue
        color = palette.get(method, THESIS_PALETTE["grey"])

        for _, row in mdf.iterrows():
            x_val = row.get(auroc_col, np.nan)
            y_val = row.get(ece_col, np.nan)
            if np.isnan(x_val) or np.isnan(y_val):
                continue
            steps = int(row.get(inner_steps_col, 0)) if inner_steps_col in row.index else 0
            marker = markers.get(steps, "o")
            ax.scatter(x_val, y_val, color=color, marker=marker, s=60, alpha=0.85, zorder=3)

    # Legend for methods
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette.get(m, THESIS_PALETTE["grey"]),
               markersize=8, label=m.replace("_", " ").title())
        for m in method_order
    ]
    # Add step-size markers
    for steps, marker in sorted(markers.items()):
        if steps > 0:
            legend_elements.append(
                Line2D([0], [0], marker=marker, color="grey", markersize=7,
                       label=f"steps={steps}", linestyle="none")
            )
    ax.legend(handles=legend_elements, fontsize=8, frameon=False,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.set_xlabel("AUROC (higher is better)", fontsize=10)
    ax.set_ylabel("ECE (lower is better)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate quadrant
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1] * 0.97, ylim[0] + 0.01, "Better discrimination",
            ha="right", va="bottom", fontsize=7, color=THESIS_PALETTE["grey"])
    ax.text(xlim[0] + 0.005, ylim[0] + 0.01, "Better calibration",
            ha="left", va="bottom", fontsize=7, color=THESIS_PALETTE["grey"])

    despine_thesis(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


# -----------------------------------------------------------------------------
# Experiment-group plotting utilities (Groups A-E)
# Added to support the structured evaluation battery defined in stage2_config.yaml.
# -----------------------------------------------------------------------------

def plot_kshot_sensitivity(
    df,
    auroc_col="auroc_mean",
    ndcg_col="ndcg_at_10_mean",
    k_col="n_support_pos",
    method_col="method",
    title="K-Shot Sensitivity (Logistics)",
    save_path=None,
    palette=None,
):
    """Plot AUROC and NDCG@10 as a function of support-set size k.

    Parameters
    ----------
    df : pd.DataFrame
        One row per experiment; must contain k_col, method_col,
        and at least one of auroc_col / ndcg_col.
    auroc_col, ndcg_col : str
        Metric column names (accept both normalized and gold-prefixed).
    k_col : str
        Column containing the integer number of positive support examples.
    title : str
        Figure title.
    save_path : path-like, optional
        If given, save the figure at 300 dpi.
    palette : dict, optional
        Method -> hex-color mapping. Defaults to THESIS_PALETTE.

    Returns
    -------
    (fig, axes)
    """
    if df.empty:
        return None, None

    _palette = palette or THESIS_PALETTE
    method_order = df[method_col].unique().tolist() if method_col in df.columns else []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = list(_palette.values())

    for ax_idx, (col, ylabel) in enumerate([
        (auroc_col, "AUROC (mean)"),
        (ndcg_col, "NDCG@10 (mean)"),
    ]):
        if col not in df.columns:
            continue
        ax = axes[ax_idx]
        for mi, method in enumerate(method_order):
            sub = (
                df[df[method_col] == method]
                .groupby(k_col)[col].mean()
                .reset_index()
                .sort_values(k_col)
            )
            if sub.empty:
                continue
            color = _palette.get(method, colors[mi % len(colors)])
            ax.plot(sub[k_col], sub[col], marker="o", label=method, color=color)
        ax.set_xlabel("k (support examples per class)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        despine_thesis(ax)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_init_ablation_bars(
    df,
    auroc_col="auroc_mean",
    ndcg_col="ndcg_at_10_mean",
    ece_col="ece_mean",
    method_col="method",
    init_col="init_name",
    title="Initialization Ablation",
    save_path=None,
):
    """Grouped bar chart comparing A_weak_only vs C_hybrid per method.

    Parameters
    ----------
    df : pd.DataFrame
        Rows from the initialization ablation group (Group C).
    auroc_col, ndcg_col, ece_col : str
        Metric column names.
    method_col, init_col : str
        Column names for method and initialization respectively.
    title : str
        Figure title.
    save_path : path-like, optional
        If given, save the figure at 300 dpi.

    Returns
    -------
    (fig, axes)
    """
    if df.empty or method_col not in df.columns or init_col not in df.columns:
        return None, None

    methods = sorted(df[method_col].unique())
    inits = sorted(df[init_col].unique())
    x = np.arange(len(methods))
    width = 0.8 / max(len(inits), 1)

    _avail_metrics = [(c, lbl) for c, lbl in [
        (auroc_col, "AUROC"), (ndcg_col, "NDCG@10"), (ece_col, "ECE"),
    ] if c in df.columns]
    n_panels = len(_avail_metrics)
    if n_panels == 0:
        return None, None

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    init_colors = {
        "A_weak_only": THESIS_PALETTE.get("blue", "#0072B2"),
        "C_hybrid":    THESIS_PALETTE.get("green", "#009E73"),
    }
    default_colors = list(THESIS_PALETTE.values())

    for panel, (col, lbl) in enumerate(_avail_metrics):
        ax = axes[panel]
        for ii, init_name in enumerate(inits):
            vals = [
                df[(df[method_col] == m) & (df[init_col] == init_name)][col].mean()
                for m in methods
            ]
            offset = (ii - (len(inits) - 1) / 2) * width
            color = init_colors.get(init_name, default_colors[ii % len(default_colors)])
            ax.bar(x + offset, vals, width, label=init_name, alpha=0.85, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        if lbl != "ECE":
            ax.set_ylim(0, 1)
        ax.legend(title="init", fontsize=7)
        despine_thesis(ax)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_department_comparison(
    df,
    auroc_col="auroc_mean",
    ndcg_col="ndcg_at_10_mean",
    ece_col="ece_mean",
    method_col="method",
    dept_col="target_department",
    title="Department Comparison",
    save_path=None,
    palette=None,
):
    """Grouped bar chart comparing performance across target departments.

    Useful for Group E (richer department check) where multiple target
    departments are evaluated side-by-side against the Logistics reference.

    Parameters
    ----------
    df : pd.DataFrame
        Rows with different target_department values.
    auroc_col, ndcg_col, ece_col : str
        Metric column names.
    method_col, dept_col : str
        Column names for method and department respectively.
    title : str
        Figure title.
    save_path : path-like, optional
        If given, save the figure at 300 dpi.
    palette : dict, optional
        Method -> hex-color mapping. Defaults to THESIS_PALETTE.

    Returns
    -------
    (fig, axes)
    """
    if df.empty or dept_col not in df.columns or method_col not in df.columns:
        return None, None

    _palette = palette or THESIS_PALETTE
    depts   = sorted(df[dept_col].unique())
    methods = sorted(df[method_col].unique())
    n_depts = len(depts)
    x = np.arange(n_depts)
    width = 0.8 / max(len(methods), 1)

    _avail = [(c, lbl) for c, lbl in [
        (auroc_col, "AUROC"), (ndcg_col, "NDCG@10"),
    ] if c in df.columns]
    n_panels = len(_avail)
    if n_panels == 0:
        return None, None

    colors = list(_palette.values())
    fig, axes = plt.subplots(1, n_panels, figsize=(max(8, n_depts * 2) * n_panels, 4),
                              constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for panel, (col, lbl) in enumerate(_avail):
        ax = axes[panel]
        for mi, method in enumerate(methods):
            vals = [
                df[(df[method_col] == method) & (df[dept_col] == d)][col].mean()
                for d in depts
            ]
            offset = (mi - (len(methods) - 1) / 2) * width
            color = _palette.get(method, colors[mi % len(colors)])
            ax.bar(x + offset, vals, width, label=method, alpha=0.85, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(depts, rotation=25, ha="right")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        if lbl != "ECE":
            ax.set_ylim(0, 1)
        ax.legend(title="method", fontsize=7)
        despine_thesis(ax)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Group E helpers — added for Group E richer-department analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_collapse_by_department(
    df: pd.DataFrame,
    method_col: str = "method",
    dept_col: str = "target_department",
    collapse_col: str = "collapse_rate",
    threshold: float = 0.20,
    title: str = "Collapse Rate by Department and Method",
    figsize: tuple = (10, 4),
    save_path: str | None = None,
) -> tuple:
    """
    Grouped bar chart of collapse_rate across departments, one bar per method.

    Bars above *threshold* are filled red (unstable); bars at or below are
    filled green so stability is visible at a glance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``method_col``, ``dept_col``, and ``collapse_col``.
    threshold : float
        Stability boundary; a dashed orange line marks this level.
    save_path : path-like, optional
        Save figure at 300 dpi when provided.

    Returns
    -------
    fig, ax : tuple
    """
    if df.empty or collapse_col not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title}\n(no data)")
        return fig, ax

    depts   = sorted(df[dept_col].unique()) if dept_col in df.columns else []
    methods = sorted(df[method_col].unique()) if method_col in df.columns else []
    if not depts or not methods:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title}\n(missing columns)")
        return fig, ax

    x     = np.arange(len(depts))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=figsize)

    _stable_color   = THESIS_PALETTE.get("green",      "#009E73")
    _unstable_color = THESIS_PALETTE.get("vermillion", "#D55E00")

    for mi, method in enumerate(methods):
        vals = [
            df[(df[method_col] == method) & (df[dept_col] == d)][collapse_col].mean()
            for d in depts
        ]
        offset = (mi - (len(methods) - 1) / 2) * width
        colors_bar = [_unstable_color if v > threshold else _stable_color for v in vals]
        bars = ax.bar(x + offset, vals, width, label=method,
                      alpha=0.85, color=colors_bar)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.axhline(threshold, color=THESIS_PALETTE.get("orange", "#E69F00"),
               linestyle="--", linewidth=1.4, alpha=0.9,
               label=f"stability threshold ({threshold})")
    ax.set_xticks(x)
    ax.set_xticklabels(depts, rotation=20, ha="right")
    ax.set_ylabel("Collapse Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    despine_thesis(ax)
    try:
        plt.tight_layout()
    except ValueError:
        pass
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_grp_a_b_e_comparison(
    grp_a_df: pd.DataFrame,
    grp_b_df: pd.DataFrame,
    grp_e_df: pd.DataFrame,
    auroc_col: str = "auroc_mean",
    ndcg_col: str = "ndcg_at_10_mean",
    collapse_col: str = "collapse_rate",
    fomaml_b_steps: int = 5,
    fomaml_b_lr: float = 0.001,
    maml_b_steps: int = 3,
    maml_b_lr: float = 0.001,
    title: str = "Groups A / B / E — Cross-Group Snapshot",
    figsize: tuple = (14, 5),
    save_path: str | None = None,
) -> tuple:
    """
    Three-panel bar chart comparing Group A (Logistics baseline), Group B
    (stabilized FOMAML/MAML on Logistics), and Group E (richer departments).

    Panels: AUROC | NDCG@10 | Collapse Rate.

    For Group A the best-AUROC row per method is used.
    For Group B only the two target rows (fomaml at *fomaml_b_steps* /
    *fomaml_b_lr* and maml at *maml_b_steps* / *maml_b_lr*) are included;
    zero_shot / finetune / anil are taken from Group A.
    For Group E the best-AUROC row per method (across departments) is shown.

    Returns
    -------
    fig, axes : tuple
    """
    methods_order = ["zero_shot", "finetune", "anil", "fomaml", "maml"]

    def _best_per_method(df, label):
        if df.empty or auroc_col not in df.columns:
            return pd.DataFrame()
        best = (df.sort_values(auroc_col, ascending=False)
                  .groupby("method").first().reset_index())
        best["_group"] = label
        return best

    snap_a = _best_per_method(grp_a_df, "A: Logistics baseline")

    # Group B: the two stabilized adaptive rows + ref methods from Group A
    snap_b_rows = []
    if not grp_b_df.empty:
        for meth, s, lr in [("fomaml", fomaml_b_steps, fomaml_b_lr),
                             ("maml",   maml_b_steps,  maml_b_lr)]:
            mask = (
                (grp_b_df.get("method",      pd.Series(dtype=str)) == meth) &
                (grp_b_df.get("inner_steps", pd.Series(dtype=int)) == s)    &
                (grp_b_df.get("inner_lr",    pd.Series(dtype=float))
                 .between(lr * 0.9, lr * 1.1))
            )
            sub = grp_b_df[mask]
            if not sub.empty:
                row = sub.iloc[[0]].copy()
                row["_group"] = "B: Logistics stabilized"
                snap_b_rows.append(row)
    for meth in ["zero_shot", "finetune", "anil"]:
        sub_a = snap_a[snap_a["method"] == meth] if not snap_a.empty else pd.DataFrame()
        if not sub_a.empty:
            row = sub_a.iloc[[0]].copy()
            row["_group"] = "B: Logistics stabilized"
            snap_b_rows.append(row)
    snap_b = pd.concat(snap_b_rows, ignore_index=True) if snap_b_rows else pd.DataFrame()

    snap_e = _best_per_method(grp_e_df, "E: Richer departments")

    all_snaps = pd.concat([snap_a, snap_b, snap_e], ignore_index=True)
    if all_snaps.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title}\n(no data yet)")
        return fig, ax

    groups_present  = list(dict.fromkeys(all_snaps["_group"].tolist()))
    methods_present = [m for m in methods_order if m in all_snaps["method"].values]

    x     = np.arange(len(methods_present))
    width = 0.8 / max(len(groups_present), 1)
    group_colors = [
        THESIS_PALETTE.get("blue",   "#0072B2"),
        THESIS_PALETTE.get("orange", "#E69F00"),
        THESIS_PALETTE.get("green",  "#009E73"),
    ]

    panels = [(c, lbl) for c, lbl in [
        (auroc_col,   "AUROC"),
        (ndcg_col,    "NDCG@10"),
        (collapse_col, "Collapse Rate"),
    ] if c in all_snaps.columns]

    n_panels = max(len(panels), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for pi, (col, lbl) in enumerate(panels):
        ax = axes[pi]
        for gi, grp_label in enumerate(groups_present):
            sub = all_snaps[all_snaps["_group"] == grp_label]
            vals = [
                sub[sub["method"] == m][col].mean()
                if m in sub["method"].values else np.nan
                for m in methods_present
            ]
            offset = (gi - (len(groups_present) - 1) / 2) * width
            color  = group_colors[gi % len(group_colors)]
            ax.bar(x + offset, vals, width, label=grp_label, alpha=0.85, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", " ").title() for m in methods_present],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        if col == collapse_col:
            ax.set_ylim(0, 1.05)
            ax.axhline(0.20, color=THESIS_PALETTE.get("grey", "#999999"),
                       linestyle="--", linewidth=1.2, alpha=0.7)
        else:
            ax.set_ylim(0, 1)
        despine_thesis(ax)

    axes[0].legend(title="group", fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    try:
        plt.tight_layout()
    except ValueError:
        pass
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_lr_sweep_heatmaps(
    df,
    metrics=None,
    method_col="method",
    steps_col="inner_steps",
    lr_col="inner_lr",
    title="LR Sweep",
    save_path=None,
    figsize_per_panel=(5.0, 3.5),
):
    """Heatmap grid: one column per method, rows = (collapse_rate, AUROC, NDCG@10).

    Extends plot_collapse_rate_heatmap to show multiple metrics at once,
    designed for Group B (maml_fomaml_lr_sweep) analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of the registry for the LR sweep group.
    metrics : list of (col_name, display_name) tuples
        Metrics to render as heatmap rows. Defaults to collapse_rate, AUROC, NDCG@10.
    method_col, steps_col, lr_col : str
        Column names.
    title : str
        Figure suptitle.
    save_path : path-like, optional
        If given, save the figure at 300 dpi.
    figsize_per_panel : (float, float)
        Width x height per (metric, method) panel.

    Returns
    -------
    (fig, axes)
    """
    if df.empty:
        return None, None

    if metrics is None:
        metrics = [
            ("collapse_rate",    "Collapse Rate"),
            ("auroc_mean",       "AUROC"),
            ("ndcg_at_10_mean",  "NDCG@10"),
        ]
    _avail_metrics = []
    for col, lbl in metrics:
        if col in df.columns:
            _avail_metrics.append((col, lbl))
        elif "gold_" + col in df.columns:
            _avail_metrics.append(("gold_" + col, lbl))

    if not _avail_metrics or steps_col not in df.columns or lr_col not in df.columns:
        return None, None

    methods = sorted(df[method_col].unique()) if method_col in df.columns else [None]
    n_methods = len(methods)
    n_rows = len(_avail_metrics)

    fw, fh = figsize_per_panel
    fig, axes = plt.subplots(
        n_rows, n_methods,
        figsize=(fw * n_methods, fh * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    cmaps = ["Reds", "Blues_r", "Blues_r"]
    vmins = [0.0, 0.4, 0.0]
    vmaxs = [1.0, 0.9, 1.0]

    for ri, (col, lbl) in enumerate(_avail_metrics):
        cmap = cmaps[ri] if ri < len(cmaps) else "viridis"
        vmin = vmins[ri] if ri < len(vmins) else None
        vmax = vmaxs[ri] if ri < len(vmaxs) else None

        for mi, method in enumerate(methods):
            sub = df[df[method_col] == method].copy() if method_col in df.columns else df.copy()
            ax = axes[ri, mi]

            try:
                pivot = sub.pivot_table(
                    index=steps_col, columns=lr_col,
                    values=col, aggfunc="mean"
                )
            except Exception as exc:
                ax.set_title(f"{method} | {lbl}\n(pivot failed: {exc})")
                continue

            im = ax.imshow(
                pivot.values, cmap=cmap,
                vmin=vmin, vmax=vmax, aspect="auto"
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.4g}" for v in pivot.columns],
                               rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_xlabel("inner_lr", fontsize=8)
            ax.set_ylabel("inner_steps", fontsize=8)
            ax.set_title(f"{method} -- {lbl}", fontsize=9)

            for rr in range(len(pivot.index)):
                for cc in range(len(pivot.columns)):
                    v = pivot.values[rr, cc]
                    if not np.isnan(v):
                        ax.text(cc, rr, f"{v:.2f}", ha="center", va="center",
                                fontsize=7,
                                color="white" if (v > (vmax or 1) * 0.65) else "black")
            plt.colorbar(im, ax=ax)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes
