"""
synthetic_plots.py — Figures and tables for the synthetic-augmentation experiment.

This module is the visualization layer that consumes the long-form per-episode
results CSV produced by the Stage 2b synthetic-support experiment runner.
Every public function takes a DataFrame (or a path to one) that conforms to
``master_thesis.stage2.EPISODIC_RESULT_ROW_SCHEMA``.

Public API
----------
plot_auroc_vs_real_k(...)                Figure 1 — does augmentation lift the k-shot curve?
plot_auroc_vs_syn_prop(...)              Figure 2 — what is the best synthetic proportion?
plot_realism_vs_syn_prop(...)            Figure 3 — how realistic are the synthetic rows?
plot_stage2b_adaptation_curve(...)       Figure 4 — adaptation curve (metric vs real_k)
plot_stage2b_shot_matrix(...)            Figure 5 — shot/augmentation heatmap matrix
build_summary_table(...)                 Table 1 — one row per (method, real_k, gen, syn_prop)
summarize_stage2b_results(...)           Table 2 — mean/std/count grouped summary
load_results(...)                        Convenience loader that validates required columns.

Design notes
------------
- The plots aggregate by mean ± std-error across episodes, treating each
  episode as an i.i.d. draw from the per-cell distribution.  This matches
  the n_repeats convention in the Stage 2 runner.
- All plots use the DTU palette from plotting.py for thesis consistency.
- All functions are tolerant to NaN-laden rows (e.g. cells where the method
  was not yet wired); they drop them before aggregation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# matplotlib is imported lazily inside the plotting helpers so that callers
# who only want `build_summary_table` don't pay the import cost.

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# DTU colour palette — reusable constants for all Stage 2b plots.
# These match plotting.DTU_PALETTE; duplicated here to avoid a circular import
# when synthetic_plots is used standalone.
# ---------------------------------------------------------------------------
DTU_RED = "#990000"
DTU_BLUE = "#2F3EEA"
DTU_NAVY = "#030F4F"
DTU_BRIGHT_GREEN = "#1FD082"
DTU_ORANGE = "#FC7634"
DTU_YELLOW = "#F6D04D"
DTU_GREY = "#DADADA"
DTU_DARK_GREY = "#999999"
DTU_PURPLE = "#79238E"

# Method → colour mapping (DTU palette)
_METHOD_COLORS: dict[str, str] = {
    "zero_shot": DTU_DARK_GREY,
    "finetune": DTU_BLUE,
    "anil": DTU_BRIGHT_GREEN,
    "fomaml": DTU_ORANGE,
    "maml": DTU_RED,
}

# Augmentation condition → colour
_AUG_COLORS: dict[str, str] = {
    "none": DTU_NAVY,
    "smote_nc": DTU_ORANGE,
    "gaussian_noise": DTU_PURPLE,
    "mixup": DTU_YELLOW,
}

# Ordered palette for generic use
_DTU_CYCLE: list[str] = [
    DTU_RED, DTU_BLUE, DTU_BRIGHT_GREEN, DTU_ORANGE,
    DTU_PURPLE, DTU_NAVY, DTU_YELLOW, DTU_DARK_GREY,
]


def apply_dtu_style() -> None:
    """Apply DTU-consistent matplotlib rcParams for thesis figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(color=_DTU_CYCLE),
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def get_dtu_palette() -> dict[str, str]:
    """Return a dictionary of named DTU colors."""
    return {
        "red": DTU_RED,
        "blue": DTU_BLUE,
        "navy": DTU_NAVY,
        "green": DTU_BRIGHT_GREEN,
        "orange": DTU_ORANGE,
        "yellow": DTU_YELLOW,
        "grey": DTU_GREY,
        "dark_grey": DTU_DARK_GREY,
        "purple": DTU_PURPLE,
    }


def _method_color(method: str) -> str:
    """Look up DTU color for a method name, with fallback."""
    return _METHOD_COLORS.get(method.lower(), DTU_DARK_GREY)


def _aug_color(aug_method: str) -> str:
    """Look up DTU color for an augmentation method, with fallback."""
    return _AUG_COLORS.get(aug_method.lower(), DTU_DARK_GREY)


# Canonical aggregation columns used by the plots and the summary table.
GROUP_COLS_BASE: Tuple[str, ...] = (
    "department", "init_name", "method", "real_k", "gen_method", "syn_prop",
)

# Required-on-input columns.  We don't enforce the full schema here because
# the runner may legitimately leave some columns NaN for skipped cells.
REQUIRED_COLS: Tuple[str, ...] = (
    "department", "method", "real_k", "syn_prop", "gen_method",
    "auroc", "ap", "ndcg_at_10", "ece",
    "pred_std", "collapsed",
    "dist_to_nearest_real_mean", "ks_max", "cat_mode_preservation_mean",
    "n_synthetic_support_rows",
)


# ---------------------------------------------------------------------------
# IO + validation
# ---------------------------------------------------------------------------
def load_results(path_or_df: Union[PathLike, pd.DataFrame]) -> pd.DataFrame:
    """Load and validate a long-form per-episode results frame.

    Accepts either a DataFrame (returned untouched after validation) or a path
    to a CSV.  Raises ``ValueError`` when required columns are missing.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in results frame: {missing}. "
            f"Expected at least {REQUIRED_COLS}"
        )
    return df


def _ensure_df(path_or_df: Union[PathLike, pd.DataFrame]) -> pd.DataFrame:
    return load_results(path_or_df)


def _drop_na_for_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df[df[metric].notna()].copy()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _mean_sem(series: pd.Series) -> Tuple[float, float, float, int]:
    """Return ``(mean, std, standard_error_of_mean, n)`` ignoring NaNs."""
    s = series.dropna()
    n = int(len(s))
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    if n == 1:
        return (float(s.iloc[0]), 0.0, 0.0, 1)
    std = float(s.std(ddof=1))
    sem = std / np.sqrt(n)
    return (float(s.mean()), std, sem, n)


def aggregate_by(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """Mean / std / sem / count aggregation, one row per group cell.

    For each metric ``m`` the output contains:
    - ``{m}_mean`` — mean across episodes
    - ``{m}_std``  — standard deviation (ddof=1)
    - ``{m}_sem``  — standard error of the mean
    - ``{m}_n``    — number of non-null observations
    """
    df = df.copy()
    for c in group_cols:
        if c not in df.columns:
            df[c] = np.nan
    rows: List[dict] = []
    grouped = df.groupby(list(group_cols), dropna=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_episodes"] = int(len(sub))
        for m in metric_cols:
            if m not in sub.columns:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_sem"] = np.nan
                row[f"{m}_n"] = 0
                continue
            mean, std, sem, n = _mean_sem(sub[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_sem"] = sem
            row[f"{m}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1 — AUROC vs real_k, real-only vs augmented
# ---------------------------------------------------------------------------
def plot_auroc_vs_real_k(
    results: Union[PathLike, pd.DataFrame],
    *,
    method: Optional[str] = None,
    department: Optional[str] = None,
    save_to: Optional[PathLike] = None,
    figsize: Tuple[float, float] = (8.0, 5.0),
):
    """Figure 1 — AUROC vs real_k with one line per (method, gen_method).

    The "real-only" condition (``gen_method == 'none'``, ``syn_prop == 0``) is
    drawn solid; augmented conditions are dashed.  Error bars are SEM across
    episodes inside each (method, gen_method, real_k) cell.
    Uses DTU palette throughout.
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    df = _ensure_df(results)
    df = _drop_na_for_metric(df, "auroc")
    if method is not None:
        df = df[df["method"] == method]
    if department is not None:
        df = df[df["department"] == department]

    if df.empty:
        raise ValueError("No rows left after filtering for plot_auroc_vs_real_k.")

    agg = aggregate_by(
        df, group_cols=("method", "gen_method", "real_k"), metric_cols=("auroc",),
    )
    agg = agg.sort_values(["method", "gen_method", "real_k"])

    fig, ax = plt.subplots(figsize=figsize)
    for (method_name, gen_name), sub in agg.groupby(["method", "gen_method"], dropna=False):
        sub = sub.sort_values("real_k")
        is_real_only = (gen_name == "none")
        linestyle = "-" if is_real_only else "--"
        marker = "o" if is_real_only else "s"
        color = _method_color(str(method_name))
        label = f"{method_name} · {gen_name}"
        ax.errorbar(
            sub["real_k"], sub["auroc_mean"], yerr=sub["auroc_sem"],
            linestyle=linestyle, marker=marker, capsize=3, label=label,
            color=color, alpha=0.6 if not is_real_only else 1.0,
        )

    ax.set_xlabel("Real support size per class (real_k)")
    ax.set_ylabel("AUROC (query, real-only)")
    title = "AUROC vs real_k"
    if department:
        title += f"  ·  {department}"
    if method:
        title += f"  ·  {method}"
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color=DTU_DARK_GREY, linewidth=0.8, alpha=0.6, linestyle=":")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — AUROC vs synthetic proportion, faceted by real_k
# ---------------------------------------------------------------------------
def plot_auroc_vs_syn_prop(
    results: Union[PathLike, pd.DataFrame],
    *,
    method: Optional[str] = None,
    department: Optional[str] = None,
    save_to: Optional[PathLike] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Figure 2 — AUROC vs syn_prop, one subplot per real_k.

    Useful for reading off the optimal augmentation ratio at each support size.
    Uses DTU palette.
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    df = _ensure_df(results)
    df = _drop_na_for_metric(df, "auroc")
    if method is not None:
        df = df[df["method"] == method]
    if department is not None:
        df = df[df["department"] == department]

    if df.empty:
        raise ValueError("No rows left after filtering for plot_auroc_vs_syn_prop.")

    agg = aggregate_by(
        df, group_cols=("method", "real_k", "syn_prop"), metric_cols=("auroc",),
    )
    real_k_vals = sorted(agg["real_k"].dropna().unique().tolist())
    n = max(1, len(real_k_vals))
    if figsize is None:
        figsize = (4.0 * n, 4.0)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True, squeeze=False)

    for col_idx, k_val in enumerate(real_k_vals):
        ax = axes[0, col_idx]
        sub_k = agg[agg["real_k"] == k_val].sort_values(["method", "syn_prop"])
        for method_name, sub in sub_k.groupby("method"):
            sub = sub.sort_values("syn_prop")
            ax.errorbar(
                sub["syn_prop"], sub["auroc_mean"], yerr=sub["auroc_sem"],
                marker="o", capsize=3, label=str(method_name),
                color=_method_color(str(method_name)),
            )
        ax.set_title(f"real_k = {int(k_val) if pd.notna(k_val) else 'NA'}")
        ax.set_xlabel("synthetic_proportion")
        if col_idx == 0:
            ax.set_ylabel("AUROC (query, real-only)")
        ax.set_ylim(0.0, 1.05)
        ax.axhline(0.5, color=DTU_DARK_GREY, linewidth=0.8, alpha=0.6, linestyle=":")
        ax.legend(fontsize=8, loc="best")

    suptitle = "AUROC vs syn_prop"
    if department:
        suptitle += f"  ·  {department}"
    if method:
        suptitle += f"  ·  {method}"
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — realism diagnostics vs synthetic proportion
# ---------------------------------------------------------------------------
def plot_realism_vs_syn_prop(
    results: Union[PathLike, pd.DataFrame],
    *,
    department: Optional[str] = None,
    save_to: Optional[PathLike] = None,
    figsize: Tuple[float, float] = (12.0, 4.0),
):
    """Figure 3 — three panels of realism diagnostics against syn_prop.

    Panels: (a) distance to nearest real, (b) KS-max across features,
    (c) categorical mode preservation (mean across categorical features).
    Uses DTU palette.
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    df = _ensure_df(results)
    if department is not None:
        df = df[df["department"] == department]

    # Realism is only meaningful when we actually generated synthetic rows.
    df = df[df["n_synthetic_support_rows"] > 0].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5,
                "No synthetic rows in results — nothing to plot.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        if save_to is not None:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_to, bbox_inches="tight", dpi=300)
        return fig

    metrics = ("dist_to_nearest_real_mean", "ks_max", "cat_mode_preservation_mean")
    agg = aggregate_by(
        df, group_cols=("real_k", "syn_prop"), metric_cols=metrics,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    titles = {
        "dist_to_nearest_real_mean": "Distance to nearest real (lower = closer)",
        "ks_max": "KS-max across features (lower = better)",
        "cat_mode_preservation_mean": "Categorical mode preservation (1.0 = perfect)",
    }
    for ax, metric in zip(axes, metrics):
        for i, (k_val, sub) in enumerate(agg.groupby("real_k")):
            sub = sub.sort_values("syn_prop")
            ax.errorbar(
                sub["syn_prop"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_sem"],
                marker="o", capsize=3,
                label=f"real_k={int(k_val) if pd.notna(k_val) else 'NA'}",
                color=_DTU_CYCLE[i % len(_DTU_CYCLE)],
            )
        ax.set_title(titles[metric], fontsize=10)
        ax.set_xlabel("synthetic_proportion")
        ax.legend(fontsize=8, loc="best")

    suptitle = "Realism diagnostics vs syn_prop"
    if department:
        suptitle += f"  ·  {department}"
    fig.suptitle(suptitle, y=1.04)
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Stage 2b adaptation curve (metric vs real_k)
# ---------------------------------------------------------------------------
def plot_stage2b_adaptation_curve(
    results_df: pd.DataFrame,
    metric: str = "auroc",
    group_col: str = "department",
    hue_col: str = "augmentation_method_actual",
    output_path: Union[str, Path, None] = None,
    figsize: Tuple[float, float] = (9.0, 5.5),
):
    """Adaptation curve: metric vs real_k, coloured by augmentation condition.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form results from ``run_stage2b_grid``.
    metric : str
        Metric column to plot (e.g. ``"auroc"``, ``"ap"``, ``"ece"``).
    group_col : str
        Column to facet by (e.g. ``"department"`` or ``"method"``).
    hue_col : str
        Column for line colour/style.  Defaults to
        ``"augmentation_method_actual"`` — set to ``"method"`` to compare
        adaptation methods instead.
    output_path : str | Path | None
        If given, save figure to this path.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    df = results_df.copy()
    df = _drop_na_for_metric(df, metric)

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for metric '{metric}'.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Resolve hue column: fall back to gen_method if the requested column
    # doesn't exist (backward compat with older result CSVs).
    if hue_col not in df.columns:
        if "gen_method" in df.columns:
            hue_col = "gen_method"
        else:
            df[hue_col] = "unknown"

    groups = sorted(df[group_col].dropna().unique()) if group_col in df.columns else ["all"]
    n_groups = max(1, len(groups))
    ncols = min(n_groups, 3)
    nrows = (n_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows / 1.5),
                             squeeze=False, sharey=True)

    for idx, grp in enumerate(groups):
        ax = axes[idx // ncols, idx % ncols]
        if group_col in df.columns:
            sub = df[df[group_col] == grp]
        else:
            sub = df

        hue_vals = sorted(sub[hue_col].dropna().unique())
        for hi, hue_val in enumerate(hue_vals):
            h_sub = sub[sub[hue_col] == hue_val]
            agg = aggregate_by(
                h_sub, group_cols=("real_k",), metric_cols=(metric,),
            ).sort_values("real_k")

            is_real_only = str(hue_val).lower() in ("none", "real_only", "nan")
            ls = "-" if is_real_only else "--"
            color = _aug_color(str(hue_val)) if hue_col.startswith("aug") or hue_col == "gen_method" \
                else _DTU_CYCLE[hi % len(_DTU_CYCLE)]

            ax.errorbar(
                agg["real_k"], agg[f"{metric}_mean"], yerr=agg[f"{metric}_sem"],
                marker="o" if is_real_only else "s",
                linestyle=ls, capsize=3, label=str(hue_val), color=color,
            )

        ax.set_title(str(grp), fontsize=11)
        ax.set_xlabel("real_k")
        if idx % ncols == 0:
            ax.set_ylabel(metric.upper())
        ax.axhline(0.5, color=DTU_DARK_GREY, linewidth=0.6, alpha=0.5, linestyle=":")
        ax.legend(fontsize=7, loc="best")

    # Hide unused subplots
    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"Adaptation curve: {metric.upper()} vs real_k", fontsize=13, y=1.02)
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — shot / augmentation heatmap matrix
# ---------------------------------------------------------------------------
def plot_stage2b_shot_matrix(
    results_df: pd.DataFrame,
    metric: str = "auroc",
    department: Optional[str] = None,
    method: Optional[str] = None,
    init_name: Optional[str] = None,
    output_path: Union[str, Path, None] = None,
    figsize: Tuple[float, float] = (7.0, 5.0),
):
    """Shot/augmentation heatmap: rows = real_k, columns = augmentation setting.

    Cell value is the mean metric across repeats.  Supports the stakeholder
    interpretation: "If you provide k real Yes/No labels, expected performance
    is X without synthetic and Y with synthetic support."

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form results.
    metric : str
        Metric column to display in the cells.
    department, method, init_name : str | None
        Optional filters.
    output_path : str | Path | None
        If given, save figure.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    apply_dtu_style()

    df = results_df.copy()
    if department is not None:
        df = df[df["department"] == department]
    if method is not None:
        df = df[df["method"] == method]
    if init_name is not None:
        df = df[df["init_name"] == init_name]
    df = _drop_na_for_metric(df, metric)

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for shot matrix ({metric}).",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Determine augmentation column
    aug_col = "augmentation_method_actual" if "augmentation_method_actual" in df.columns else "gen_method"

    agg = aggregate_by(
        df, group_cols=("real_k", aug_col), metric_cols=(metric,),
    )

    real_ks = sorted(agg["real_k"].dropna().unique())
    aug_methods = sorted(agg[aug_col].dropna().unique())

    # Build matrix
    matrix = np.full((len(real_ks), len(aug_methods)), np.nan)
    for i, k in enumerate(real_ks):
        for j, aug in enumerate(aug_methods):
            mask = (agg["real_k"] == k) & (agg[aug_col] == aug)
            vals = agg.loc[mask, f"{metric}_mean"]
            if len(vals) > 0:
                matrix[i, j] = vals.iloc[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Custom DTU-tinted colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dtu_heat", [DTU_NAVY, DTU_BLUE, DTU_BRIGHT_GREEN, DTU_YELLOW], N=256,
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(aug_methods)))
    ax.set_xticklabels([str(a) for a in aug_methods], fontsize=9)
    ax.set_yticks(range(len(real_ks)))
    ax.set_yticklabels([f"k={int(k)}" for k in real_ks], fontsize=9)
    ax.set_xlabel("Augmentation condition")
    ax.set_ylabel("Real support per class (real_k)")

    # Annotate cells
    for i in range(len(real_ks)):
        for j in range(len(aug_methods)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    title = f"Shot matrix: {metric.upper()}"
    parts = []
    if department:
        parts.append(department)
    if method:
        parts.append(method)
    if init_name:
        parts.append(init_name)
    if parts:
        title += f"  ·  {' · '.join(parts)}"
    ax.set_title(title, fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.8, label=metric.upper())
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Table 1 — main summary table (backward-compatible)
# ---------------------------------------------------------------------------
def build_summary_table(
    results: Union[PathLike, pd.DataFrame],
    *,
    extra_group_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Build the main summary table for the synthetic-augmentation experiment.

    One row per ``(department, init_name, method, real_k, gen_method, syn_prop)``
    cell with mean / sem / n for the headline metrics, plus collapse rate and
    a compact view of the realism diagnostics.
    """
    df = _ensure_df(results)
    group_cols: List[str] = list(GROUP_COLS_BASE)
    if extra_group_cols:
        for c in extra_group_cols:
            if c not in group_cols:
                group_cols.append(c)

    metric_cols = (
        "auroc", "ap", "ndcg_at_10", "ece",
        "pred_mean", "pred_std",
        "n_real_support_rows", "n_synthetic_support_rows",
        "dist_to_nearest_real_mean", "dist_to_nearest_real_p95",
        "ks_max", "ks_median",
        "cat_mode_preservation_mean", "cat_mode_preservation_min",
    )
    table = aggregate_by(df, group_cols=group_cols, metric_cols=metric_cols)

    # Add collapse rate per cell.
    if "collapsed" in df.columns:
        collapse = (
            df.groupby(group_cols, dropna=False)["collapsed"]
            .apply(lambda s: float(s.dropna().astype(bool).mean()) if len(s.dropna()) else np.nan)
            .reset_index(name="collapse_rate")
        )
        table = table.merge(collapse, on=group_cols, how="left")

    sort_cols = [c for c in ("department", "real_k", "method", "syn_prop") if c in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return table


# ---------------------------------------------------------------------------
# Table 2 — summarize_stage2b_results
# ---------------------------------------------------------------------------
def summarize_stage2b_results(
    results_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Grouped summary for Stage 2b results.

    Groups by department, real_k, method, init_name, augmentation_method_actual,
    and target_effective_k.  Returns mean/std/count for each metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form results.
    metrics : list[str] | None
        Metric columns to summarize.  Defaults to core metrics.

    Returns
    -------
    pd.DataFrame
        One row per group with ``{metric}_mean``, ``{metric}_std``,
        ``{metric}_count`` columns.
    """
    if metrics is None:
        metrics = ["auroc", "ap", "average_precision", "log_loss",
                    "brier", "ece", "ndcg_at_10", "precision_at_10"]

    # Only keep metrics that actually exist
    metrics = [m for m in metrics if m in results_df.columns]

    group_cols = ["department", "real_k", "method", "init_name"]

    # Add optional columns if present
    for opt in ("augmentation_method_actual", "gen_method", "target_effective_k"):
        if opt in results_df.columns:
            group_cols.append(opt)

    # Remove duplicates, preserve order
    seen: set[str] = set()
    unique_group_cols: list[str] = []
    for c in group_cols:
        if c not in seen and c in results_df.columns:
            seen.add(c)
            unique_group_cols.append(c)
    group_cols = unique_group_cols

    agg_dicts: list[dict] = []
    grouped = results_df.groupby(group_cols, dropna=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["count"] = len(sub)
        for m in metrics:
            vals = sub[m].dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_count"] = len(vals)
        agg_dicts.append(row)

    summary = pd.DataFrame(agg_dicts)
    sort_cols = [c for c in ("department", "real_k", "method", "init_name") if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return summary


# ============================================================================
# THESIS ANALYSIS UTILITIES — Stage 2 result loading, grouping, and plots
# ============================================================================


def load_clean_stage2_results(experiment_root: PathLike, stale_aware: bool = True) -> dict:
    """Load experiment_summary.csv and split into clean/diagnostic/failed/stale subsets.

    Parameters
    ----------
    experiment_root : Path or str
        Directory containing experiment_summary.csv and experiment subfolders.
    stale_aware : bool
        If True, Stage 2b rows without stage2b_config.json are marked stale.

    Returns
    -------
    dict
        Keys:
        - "all"         : full DataFrame
        - "valid"       : rows where status=="success" AND n_query_neg > 0 (both classes in query)
        - "diagnostic"  : rows where status=="success" AND n_query_neg == 0 (one-class query)
        - "failed"      : rows where status != "success" or status is missing
        - "stale"       : Stage 2b rows lacking stage2b_config.json (pre-fix)
        - "s2b_current" : Stage 2b rows with stage2b_config.json (post-fix)
    """
    from pathlib import Path

    root = Path(experiment_root)
    csv_path = root / "experiment_summary.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"experiment_summary.csv not found at {csv_path}")

    df_all = pd.read_csv(csv_path)
    print(f"Loaded {len(df_all)} total rows from experiment_summary.csv")

    # Split by status
    df_failed = df_all[
        (df_all["status"].isna()) | (df_all["status"] != "success")
    ].copy()
    df_success = df_all[df_all["status"] == "success"].copy()

    # Split successful rows by n_query_neg if column exists
    if "n_query_neg" in df_success.columns:
        df_valid = df_success[df_success["n_query_neg"] > 0].copy()
        df_diagnostic = df_success[df_success["n_query_neg"] == 0].copy()
    else:
        df_valid = df_success.copy()
        df_diagnostic = pd.DataFrame(columns=df_success.columns)

    # Check for stale rows (Stage 2b without stage2b_config.json)
    df_stale = pd.DataFrame(columns=df_all.columns)
    df_s2b_current = pd.DataFrame(columns=df_all.columns)

    if stale_aware and "augmentation_method" in df_all.columns:
        stage2b_mask = df_all["augmentation_method"].notna()
        if stage2b_mask.any():
            for idx in df_all[stage2b_mask].index:
                row = df_all.loc[idx]
                # Try to infer experiment folder name
                folder_name = None
                if "experiment_id" in row and pd.notna(row["experiment_id"]):
                    folder_name = str(row["experiment_id"])
                elif "experiment_folder" in row and pd.notna(row["experiment_folder"]):
                    folder_name = str(row["experiment_folder"])

                if folder_name:
                    config_path = root / folder_name / "stage2b_config.json"
                    if config_path.exists():
                        df_s2b_current = pd.concat([df_s2b_current, df_all.loc[[idx]]], ignore_index=True)
                    else:
                        df_stale = pd.concat([df_stale, df_all.loc[[idx]]], ignore_index=True)
                else:
                    df_stale = pd.concat([df_stale, df_all.loc[[idx]]], ignore_index=True)

    print(f"  - Valid (success + both classes in query): {len(df_valid)}")
    print(f"  - Diagnostic (success + one class in query): {len(df_diagnostic)}")
    print(f"  - Failed: {len(df_failed)}")
    if stale_aware:
        print(f"  - Stale Stage 2b: {len(df_stale)}")
        print(f"  - Current Stage 2b: {len(df_s2b_current)}")

    return {
        "all": df_all,
        "valid": df_valid,
        "diagnostic": df_diagnostic,
        "failed": df_failed,
        "stale": df_stale,
        "s2b_current": df_s2b_current,
    }


def summarize_metric_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Optional[Sequence[str]] = None,
    min_count: int = 1,
) -> pd.DataFrame:
    """Group df by group_cols and compute mean ± std ± count for each metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_cols : Sequence[str]
        Columns to group by.
    metric_cols : Sequence[str] | None
        Metric columns to summarize. If None, defaults to
        ["gold_auroc_mean", "gold_logloss_mean", "auroc_mean", "logloss_mean"]
        (filtered to columns that exist).
    min_count : int
        Exclude groups with fewer than min_count observations.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns like {metric}_mean, {metric}_std, {metric}_count.
    """
    if metric_cols is None:
        default_metrics = [
            "gold_auroc_mean", "gold_logloss_mean", "auroc_mean", "logloss_mean"
        ]
        metric_cols = [m for m in default_metrics if m in df.columns]

    if not metric_cols:
        print("Warning: no recognized metric columns found in df.")
        return pd.DataFrame()

    metric_cols = [m for m in metric_cols if m in df.columns]

    agg = aggregate_by(df, group_cols=group_cols, metric_cols=metric_cols)
    agg = agg[agg["n_episodes"] >= min_count].copy()

    return agg


def plot_method_comparison(
    df: pd.DataFrame,
    metric: str = "auroc",
    department: Optional[str] = None,
    init_name: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Bar plot comparing methods on a chosen metric.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    metric : str
        Metric column to plot.
    department : str | None
        If provided, filter to this department.
    init_name : str | None
        If provided, filter to this init_name.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    work_df = df.copy()

    if department is not None:
        if "department" in work_df.columns or "target_department" in work_df.columns:
            dep_col = "target_department" if "target_department" in work_df.columns else "department"
            work_df = work_df[work_df[dep_col] == department]

    if init_name is not None:
        if "init_name" in work_df.columns:
            work_df = work_df[work_df["init_name"] == init_name]

    if metric not in work_df.columns:
        print(f"Warning: metric '{metric}' not found in DataFrame.")
        return None

    work_df = _drop_na_for_metric(work_df, metric)

    if work_df.empty:
        print(f"Warning: no data left after filtering for method comparison.")
        return None

    # Aggregate by method
    method_col = "method" if "method" in work_df.columns else "model"
    if method_col not in work_df.columns:
        print(f"Warning: no method column found.")
        return None

    agg = aggregate_by(work_df, group_cols=(method_col,), metric_cols=(metric,))
    agg = agg.sort_values(f"{metric}_mean", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)

    methods = agg[method_col].dropna().astype(str).tolist()
    means = agg[f"{metric}_mean"].values
    stds = agg[f"{metric}_std"].values
    colors = [_method_color(m) for m in methods]

    ax.bar(range(len(methods)), means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="#333333")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, 1.05)

    title = f"Method Comparison — {metric.upper()} by Method"
    if department:
        title += f"  ·  {department}"
    if init_name:
        title += f"  ·  {init_name}"
    ax.set_title(title)

    ax.axhline(0.5, color=DTU_DARK_GREY, linewidth=0.8, alpha=0.6, linestyle=":")
    fig.tight_layout()

    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_maml_family_comparison(
    df: pd.DataFrame,
    metric: str = "auroc",
    group_by: str = "target_department",
    figsize: Tuple[float, float] = (12.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Compare ANIL, FOMAML, MAML (maml-family methods only).

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    metric : str
        Metric to plot.
    group_by : str
        Column to use for grouping/faceting (e.g., "target_department", "init_name").
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    work_df = df.copy()

    # Filter to maml-family methods
    if "method" in work_df.columns:
        maml_methods = ["anil", "fomaml", "maml"]
        work_df = work_df[work_df["method"].str.lower().isin(maml_methods)]
    else:
        print("Warning: 'method' column not found.")
        return None

    if metric not in work_df.columns:
        print(f"Warning: metric '{metric}' not found.")
        return None

    work_df = _drop_na_for_metric(work_df, metric)

    if work_df.empty:
        print("Warning: no maml-family data after filtering.")
        return None

    # Check if group_by column exists
    if group_by not in work_df.columns:
        print(f"Warning: group column '{group_by}' not found. Using no grouping.")
        group_by = None

    if group_by:
        groups = sorted(work_df[group_by].dropna().unique())
        ncols = min(len(groups), 3)
        nrows = (len(groups) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows / 2), squeeze=False)
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize, squeeze=False)
        groups = [None]

    for idx, grp in enumerate(groups):
        ax = axes[idx // (ncols if group_by else 1), idx % (ncols if group_by else 1)]

        if group_by and grp is not None:
            sub = work_df[work_df[group_by] == grp]
        else:
            sub = work_df

        agg = aggregate_by(sub, group_cols=("method",), metric_cols=(metric,))
        agg = agg.sort_values("method")

        methods = agg["method"].dropna().astype(str).tolist()
        means = agg[f"{metric}_mean"].values
        stds = agg[f"{metric}_std"].values
        colors = [_method_color(m) for m in methods]

        ax.bar(range(len(methods)), means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="#333333")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0.0, 1.05)

        title = f"{metric.upper()} by MAML Family"
        if group_by and grp is not None:
            title += f"  ·  {grp}"
        ax.set_title(title)
        ax.axhline(0.5, color=DTU_DARK_GREY, linewidth=0.8, alpha=0.6, linestyle=":")

    # Hide unused subplots
    if group_by:
        for idx in range(len(groups), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_initialization_comparison(
    df: pd.DataFrame,
    metric: str = "auroc",
    method: Optional[str] = None,
    figsize: Tuple[float, float] = (12.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Compare initialization strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    metric : str
        Metric to plot.
    method : str | None
        If provided, filter to this method.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    work_df = df.copy()

    if method is not None and "method" in work_df.columns:
        work_df = work_df[work_df["method"] == method]

    if metric not in work_df.columns:
        print(f"Warning: metric '{metric}' not found.")
        return None

    if "init_name" not in work_df.columns:
        print("Warning: 'init_name' column not found.")
        return None

    work_df = _drop_na_for_metric(work_df, metric)

    if work_df.empty:
        print("Warning: no data for initialization comparison.")
        return None

    # Init color mapping
    init_colors_map = {
        "A_weak_only": DTU_BLUE,
        "B_gold_only": DTU_RED,
        "C_hybrid": DTU_BRIGHT_GREEN,
        "D_hybrid_unweighted": DTU_YELLOW,
        "random": DTU_DARK_GREY,
    }

    agg = aggregate_by(work_df, group_cols=("init_name",), metric_cols=(metric,))
    agg = agg.sort_values("init_name")

    fig, ax = plt.subplots(figsize=figsize)

    inits = agg["init_name"].dropna().astype(str).tolist()
    means = agg[f"{metric}_mean"].values
    stds = agg[f"{metric}_std"].values
    colors = [init_colors_map.get(init, DTU_DARK_GREY) for init in inits]

    ax.bar(range(len(inits)), means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="#333333")
    ax.set_xticks(range(len(inits)))
    ax.set_xticklabels(inits, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, 1.05)

    title = f"Initialization Comparison — {metric.upper()}"
    if method:
        title += f"  ·  {method}"
    ax.set_title(title)

    ax.axhline(0.5, color=DTU_DARK_GREY, linewidth=0.8, alpha=0.6, linestyle=":")
    fig.tight_layout()

    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_k_shot_curve(
    df: pd.DataFrame,
    metric: str = "auroc",
    methods: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Line plot of metric vs k (n_support_pos or derived k value).

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    metric : str
        Metric to plot.
    methods : Sequence[str] | None
        If provided, filter to these methods.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    work_df = df.copy()

    # Find k column
    k_col = None
    for c in ["n_support_pos", "real_k", "k"]:
        if c in work_df.columns:
            k_col = c
            break

    if k_col is None:
        print("Warning: no k column (n_support_pos, real_k, or k) found.")
        return None

    if metric not in work_df.columns:
        print(f"Warning: metric '{metric}' not found.")
        return None

    if methods is not None and "method" in work_df.columns:
        work_df = work_df[work_df["method"].isin(methods)]

    work_df = _drop_na_for_metric(work_df, metric)

    if work_df.empty:
        print("Warning: no data for k-shot curve.")
        return None

    # Aggregate by k and method (if available)
    group_cols = [k_col]
    if "method" in work_df.columns and "init_name" in work_df.columns:
        group_cols.append("method")
    elif "init_name" in work_df.columns:
        group_cols.append("init_name")

    agg = aggregate_by(work_df, group_cols=tuple(group_cols), metric_cols=(metric,))

    fig, ax = plt.subplots(figsize=figsize)

    if len(group_cols) > 1:
        grouping_col = group_cols[1]
        for gval in sorted(agg[grouping_col].dropna().unique()):
            sub = agg[agg[grouping_col] == gval].sort_values(k_col)
            color = _method_color(str(gval)) if grouping_col == "method" else DTU_BLUE
            ax.plot(sub[k_col], sub[f"{metric}_mean"], marker="o", label=str(gval), color=color)
            ax.fill_between(
                sub[k_col],
                sub[f"{metric}_mean"] - sub[f"{metric}_sem"],
                sub[f"{metric}_mean"] + sub[f"{metric}_sem"],
                alpha=0.2, color=color,
            )
    else:
        agg = agg.sort_values(k_col)
        ax.plot(agg[k_col], agg[f"{metric}_mean"], marker="o", color=DTU_BLUE)
        ax.fill_between(
            agg[k_col],
            agg[f"{metric}_mean"] - agg[f"{metric}_sem"],
            agg[f"{metric}_mean"] + agg[f"{metric}_sem"],
            alpha=0.2, color=DTU_BLUE,
        )

    ax.set_xlabel(f"k (contracts per class)")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{metric.upper()} vs k-shot")
    ax.legend(fontsize=9, loc="best")
    ax.axhline(0.5, color=DTU_DARK_GREY, linewidth=0.8, alpha=0.6, linestyle=":")

    # Add note
    note = "k = contracts per class; model input may use more rows (one row per observation/view)."
    fig.text(0.5, 0.02, note, ha="center", fontsize=8, style="italic", color="#555555")

    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_stage2b_performance_comparison(
    df: pd.DataFrame,
    metric: str = "auroc",
    figsize: Tuple[float, float] = (12.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Compare none vs smote_nc augmentation on a chosen metric.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame (Stage 2b results).
    metric : str
        Metric to plot.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    work_df = df.copy()

    # Filter to Stage 2b rows
    aug_col = None
    for c in ["augmentation_method", "augmentation_method_actual"]:
        if c in work_df.columns:
            aug_col = c
            break

    if aug_col is None:
        print("Warning: no augmentation_method column found.")
        return None

    work_df = work_df[work_df[aug_col].notna()].copy()

    if metric not in work_df.columns:
        print(f"Warning: metric '{metric}' not found.")
        return None

    work_df = _drop_na_for_metric(work_df, metric)

    if work_df.empty:
        print("Warning: no Stage 2b data for augmentation comparison.")
        return None

    # Group by department and k, compare none vs smote_nc
    group_cols = ["target_department"] if "target_department" in work_df.columns else ["department"]
    if "n_support_pos" in work_df.columns:
        group_cols.append("n_support_pos")
    group_cols.append(aug_col)

    agg = aggregate_by(work_df, group_cols=tuple(group_cols), metric_cols=(metric,))

    fig, ax = plt.subplots(figsize=figsize)

    # Create grouped bars
    dep_col = "target_department" if "target_department" in work_df.columns else "department"
    dep_vals = sorted(agg[dep_col].dropna().unique())

    x_pos = 0
    tick_positions = []
    tick_labels = []

    for dep in dep_vals:
        dep_data = agg[agg[dep_col] == dep]

        for aug_val in sorted(dep_data[aug_col].dropna().unique()):
            mask = (agg[dep_col] == dep) & (agg[aug_col] == aug_val)
            row = agg[mask]

            if len(row) > 0:
                mean_val = row[f"{metric}_mean"].iloc[0]
                std_val = row[f"{metric}_std"].iloc[0]
                color = DTU_DARK_GREY if aug_val == "none" else DTU_BRIGHT_GREEN

                ax.bar(x_pos, mean_val, yerr=std_val, capsize=5, color=color, alpha=0.8, edgecolor="#333333", width=0.6)
                x_pos += 1

        tick_positions.append(x_pos - 1.5)
        tick_labels.append(str(dep))
        x_pos += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Stage 2b Augmentation Comparison — {metric.upper()}")
    ax.axhline(0.5, color=DTU_DARK_GREY, linewidth=0.8, alpha=0.6, linestyle=":")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DTU_DARK_GREY, alpha=0.8, label="none"),
        Patch(facecolor=DTU_BRIGHT_GREEN, alpha=0.8, label="smote_nc"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="best")

    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_synthetic_support_counts(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (12.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Stacked or side-by-side bar chart of real vs synthetic support rows.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    work_df = df.copy()

    # Filter to smote_nc rows
    aug_col = None
    for c in ["augmentation_method", "augmentation_method_actual"]:
        if c in work_df.columns:
            aug_col = c
            break

    if aug_col is None:
        print("Warning: no augmentation_method column found.")
        return None

    work_df = work_df[work_df[aug_col] == "smote_nc"].copy()

    if "n_real_support_rows" not in work_df.columns or "n_synthetic_support_rows" not in work_df.columns:
        print("Warning: n_real_support_rows or n_synthetic_support_rows not found.")
        return None

    if work_df.empty:
        print("Warning: no smote_nc data found.")
        return None

    # Aggregate
    group_cols = ["target_department"] if "target_department" in work_df.columns else ["department"]
    if "n_support_pos" in work_df.columns:
        group_cols.append("n_support_pos")

    agg = aggregate_by(
        work_df,
        group_cols=tuple(group_cols),
        metric_cols=("n_real_support_rows", "n_synthetic_support_rows"),
    )

    fig, ax = plt.subplots(figsize=figsize)

    dep_col = "target_department" if "target_department" in work_df.columns else "department"
    dep_vals = sorted(agg[dep_col].dropna().unique())

    x_pos = 0
    width = 0.35
    tick_positions = []
    tick_labels = []

    for dep in dep_vals:
        dep_data = agg[agg[dep_col] == dep].sort_values("n_support_pos") if "n_support_pos" in agg.columns else agg[agg[dep_col] == dep]

        for _, row in dep_data.iterrows():
            real_mean = row.get("n_real_support_rows_mean", 0)
            syn_mean = row.get("n_synthetic_support_rows_mean", 0)

            ax.bar(x_pos, real_mean, width, color=DTU_BLUE, alpha=0.8, label="Real" if x_pos == 0 else "", edgecolor="#333333")
            ax.bar(x_pos, syn_mean, width, bottom=real_mean, color=DTU_BRIGHT_GREEN, alpha=0.8, label="Synthetic" if x_pos == 0 else "", edgecolor="#333333")

            x_pos += 1

        tick_positions.append(x_pos - 0.5)
        tick_labels.append(str(dep))
        x_pos += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel("Number of rows (mean)")
    ax.set_title("Real vs Synthetic Support Rows per Episode")
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_synthetic_support_proportion(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (10.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Plot synthetic_support_proportion = n_synthetic / n_effective support rows.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    work_df = df.copy()

    # Compute proportion if not present
    if "synthetic_support_proportion" not in work_df.columns:
        if "n_synthetic_support_rows" in work_df.columns and "n_effective_support_rows" in work_df.columns:
            work_df["synthetic_support_proportion"] = (
                work_df["n_synthetic_support_rows"] / work_df["n_effective_support_rows"]
            )
        elif "n_synthetic_support_rows" in work_df.columns and "n_real_support_rows" in work_df.columns:
            work_df["synthetic_support_proportion"] = (
                work_df["n_synthetic_support_rows"] /
                (work_df["n_real_support_rows"] + work_df["n_synthetic_support_rows"])
            )
        else:
            print("Warning: cannot compute synthetic_support_proportion.")
            return None

    work_df = work_df[work_df["synthetic_support_proportion"].notna()].copy()

    if work_df.empty:
        print("Warning: no synthetic proportion data.")
        return None

    # Aggregate
    group_cols = ["target_department"] if "target_department" in work_df.columns else ["department"]
    if "n_support_pos" in work_df.columns:
        group_cols.append("n_support_pos")

    agg = aggregate_by(
        work_df,
        group_cols=tuple(group_cols),
        metric_cols=("synthetic_support_proportion",),
    )

    fig, ax = plt.subplots(figsize=figsize)

    dep_col = "target_department" if "target_department" in work_df.columns else "department"
    dep_vals = sorted(agg[dep_col].dropna().unique())

    x_pos = 0
    tick_positions = []
    tick_labels = []

    for dep in dep_vals:
        dep_data = agg[agg[dep_col] == dep]

        means = dep_data["synthetic_support_proportion_mean"].values
        stds = dep_data["synthetic_support_proportion_std"].values

        ax.bar(
            range(x_pos, x_pos + len(means)),
            means,
            yerr=stds,
            capsize=5,
            color=DTU_BRIGHT_GREEN,
            alpha=0.8,
            edgecolor="#333333",
        )

        tick_positions.append(x_pos + len(means) / 2 - 0.5)
        tick_labels.append(str(dep))
        x_pos += len(means) + 1

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel("Synthetic Proportion")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Synthetic Support Proportion by Department and k")

    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_effective_support_size(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (10.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Compare contract-level k vs row-level effective support size.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    work_df = df.copy()

    # Require k and row count columns
    k_col = None
    for c in ["n_support_pos", "real_k"]:
        if c in work_df.columns:
            k_col = c
            break

    if k_col is None or "n_real_support_rows" not in work_df.columns:
        print("Warning: missing n_support_pos/real_k or n_real_support_rows.")
        return None

    work_df = work_df[[k_col, "n_real_support_rows"]].dropna().copy()

    if work_df.empty:
        print("Warning: no data for effective support size plot.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: contract-level k distribution
    ax1.scatter(range(len(work_df)), work_df[k_col], alpha=0.5, color=DTU_BLUE, s=20)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("k (contracts per class)")
    ax1.set_title(f"Contract-level k Distribution")
    ax1.grid(True, alpha=0.3)

    # Right: row-level support rows distribution
    ax2.scatter(range(len(work_df)), work_df["n_real_support_rows"], alpha=0.5, color=DTU_BRIGHT_GREEN, s=20)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("n_real_support_rows")
    ax2.set_title("Row-level Support Size Distribution")
    ax2.grid(True, alpha=0.3)

    # Add note
    note = "Each contract contributes multiple feature rows (one per observation/view)."
    fig.text(0.5, 0.02, note, ha="center", fontsize=8, style="italic", color="#555555")

    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_skip_reasons(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (8.0, 4.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Bar chart of gen_skip_reason value counts.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    if "gen_skip_reason" not in df.columns:
        print("Warning: gen_skip_reason column not found.")
        return None

    skip_reasons = df[df["gen_skip_reason"].notna() & (df["gen_skip_reason"] != "")]["gen_skip_reason"].value_counts()

    if skip_reasons.empty:
        print("No skip reasons found.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    reasons = skip_reasons.index.astype(str).tolist()
    counts = skip_reasons.values
    colors = [
        DTU_RED if "real_support_exceeds_target_effective_k" in r else DTU_YELLOW
        for r in reasons
    ]

    bars = ax.bar(range(len(reasons)), counts, color=colors, alpha=0.8, edgecolor="#333333")

    # Annotate bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(int(count)),
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(reasons)))
    ax.set_xticklabels(reasons, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("SMOTENC Skip Reason Distribution")

    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def plot_prediction_collapse(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (10.0, 5.0),
    save_to: Optional[PathLike] = None,
) -> Optional:
    """Show prediction mean distribution and collapse rate by method.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    figsize : tuple
        Figure size.
    save_to : PathLike | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure | None
    """
    import matplotlib.pyplot as plt
    apply_dtu_style()

    if "pred_mean_mean" not in df.columns:
        print("Warning: pred_mean_mean column not found.")
        return None

    work_df = df.copy()

    # Define collapse
    if "pred_mean_mean" in work_df.columns:
        work_df["collapsed"] = (
            (work_df["pred_mean_mean"] < 0.05) | (work_df["pred_mean_mean"] > 0.95)
        )

    if work_df.empty:
        print("Warning: no data for prediction collapse plot.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: distribution of pred_mean by method
    if "method" in work_df.columns:
        methods = sorted(work_df["method"].dropna().unique())
        for method in methods:
            sub = work_df[work_df["method"] == method]["pred_mean_mean"].dropna()
            if len(sub) > 0:
                color = _method_color(method)
                ax1.hist(sub, bins=20, alpha=0.5, label=method, color=color)
    else:
        ax1.hist(work_df["pred_mean_mean"].dropna(), bins=20, color=DTU_BLUE, alpha=0.7)

    ax1.set_xlabel("Prediction mean")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Prediction Mean")
    ax1.legend(fontsize=8, loc="best")
    ax1.axvline(0.5, color=DTU_DARK_GREY, linestyle="--", linewidth=0.8, alpha=0.6)

    # Right: collapse rate by method
    if "method" in work_df.columns:
        collapse_rates = []
        method_names = []
        for method in sorted(work_df["method"].dropna().unique()):
            sub = work_df[work_df["method"] == method]
            if "collapsed" in sub.columns:
                rate = float(sub["collapsed"].dropna().astype(int).mean())
            else:
                rate = float(
                    ((sub["pred_mean_mean"] < 0.05) | (sub["pred_mean_mean"] > 0.95))
                    .astype(int).mean()
                )
            collapse_rates.append(rate)
            method_names.append(method)

        colors = [_method_color(m) for m in method_names]
        ax2.bar(range(len(method_names)), collapse_rates, color=colors, alpha=0.8, edgecolor="#333333")
        ax2.set_xticks(range(len(method_names)))
        ax2.set_xticklabels(method_names, rotation=45, ha="right")

    ax2.set_ylabel("Collapse rate")
    ax2.set_ylim(0.0, 1.1)
    ax2.set_title("Prediction Collapse Rate by Method")

    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig


def make_thesis_summary_table(
    df: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    metric_cols: Optional[Sequence[str]] = None,
    save_to: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Create a formatted summary table suitable for thesis inclusion.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    group_cols : Sequence[str] | None
        Columns to group by. Defaults to ["method", "init_name", "target_department", "n_support_pos"].
    metric_cols : Sequence[str] | None
        Metrics to summarize. Defaults to gold_auroc_mean, gold_logloss_mean, auroc_mean, logloss_mean (filtered).
    save_to : PathLike | None
        If provided, save table as CSV.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per group and formatted metric columns.
    """
    if group_cols is None:
        group_cols = ["method", "init_name", "target_department", "n_support_pos"]
        # Filter to columns that actually exist
        group_cols = [c for c in group_cols if c in df.columns]

    if not group_cols:
        group_cols = ["method"]

    if metric_cols is None:
        candidate_metrics = ["gold_auroc_mean", "gold_logloss_mean", "auroc_mean", "logloss_mean"]
        metric_cols = [m for m in candidate_metrics if m in df.columns]

    if not metric_cols:
        print("Warning: no recognized metric columns found.")
        return pd.DataFrame()

    # Aggregate
    agg = aggregate_by(df, group_cols=group_cols, metric_cols=metric_cols)

    # Filter to rows with n >= 3
    agg = agg[agg["n_episodes"] >= 3].copy()

    # Format metrics as "mean ± std"
    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in agg.columns and std_col in agg.columns:
            agg[metric] = agg.apply(
                lambda row: f"{row[mean_col]:.3f} ± {row[std_col]:.3f}"
                if pd.notna(row[mean_col]) else "NA",
                axis=1,
            )

    # Select columns for output
    out_cols = list(group_cols) + [m for m in metric_cols if m in agg.columns] + ["n_episodes"]
    summary = agg[out_cols].copy()

    # Sort
    sort_cols = [c for c in ["method", "init_name", "target_department", "n_support_pos"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    if save_to is not None:
        summary.to_csv(save_to, index=False)
        print(f"Summary table saved to {save_to}")

    return summary


__all__ = [
    "load_results",
    "aggregate_by",
    "plot_auroc_vs_real_k",
    "plot_auroc_vs_syn_prop",
    "plot_realism_vs_syn_prop",
    "plot_stage2b_adaptation_curve",
    "plot_stage2b_shot_matrix",
    "build_summary_table",
    "summarize_stage2b_results",
    "apply_dtu_style",
    "get_dtu_palette",
    "load_clean_stage2_results",
    "summarize_metric_by_group",
    "plot_method_comparison",
    "plot_maml_family_comparison",
    "plot_initialization_comparison",
    "plot_k_shot_curve",
    "plot_stage2b_performance_comparison",
    "plot_synthetic_support_counts",
    "plot_synthetic_support_proportion",
    "plot_effective_support_size",
    "plot_skip_reasons",
    "plot_prediction_collapse",
    "make_thesis_summary_table",
    "REQUIRED_COLS",
    "GROUP_COLS_BASE",
    "DTU_RED",
    "DTU_BLUE",
    "DTU_NAVY",
    "DTU_BRIGHT_GREEN",
    "DTU_ORANGE",
    "DTU_YELLOW",
    "DTU_GREY",
    "DTU_DARK_GREY",
    "DTU_PURPLE",
]
