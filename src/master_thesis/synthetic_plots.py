"""
synthetic_plots.py — Figures and tables for the synthetic-augmentation experiment.

This module is the visualization layer that consumes the long-form per-episode
results CSV produced by `Notebooks/19.0_synthetic_augmentation_experiment.ipynb`.
Every public function takes a DataFrame (or a path to one) that conforms to
`master_thesis.stage2.EPISODIC_RESULT_ROW_SCHEMA`.

Public API
----------
plot_auroc_vs_real_k(...)       Figure 1 — does augmentation lift the k-shot curve?
plot_auroc_vs_syn_prop(...)     Figure 2 — what is the best synthetic proportion?
plot_realism_vs_syn_prop(...)   Figure 3 — how realistic are the synthetic rows?
build_summary_table(...)        Table 1 — one row per (method, real_k, gen, syn_prop)
load_results(...)               Convenience loader that validates required columns.

Design notes
------------
- The plots aggregate by mean ± std-error across episodes, treating each
  episode as an i.i.d. draw from the per-cell distribution.  This matches
  the n_repeats convention in the Stage 2 runner.
- We deliberately keep matplotlib styling minimal (no custom palettes, no
  rcParams overrides) so the figures render cleanly in any environment and
  remain readable when copied into the thesis manuscript.
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
def _mean_sem(series: pd.Series) -> Tuple[float, float, int]:
    """Return ``(mean, standard_error_of_mean, n)`` ignoring NaNs."""
    s = series.dropna()
    n = int(len(s))
    if n == 0:
        return (np.nan, np.nan, 0)
    if n == 1:
        return (float(s.iloc[0]), 0.0, 1)
    return (float(s.mean()), float(s.std(ddof=1) / np.sqrt(n)), n)


def aggregate_by(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """Mean / std / sem / count aggregation, one row per group cell."""
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
                row[f"{m}_sem"] = np.nan
                row[f"{m}_n"] = 0
                continue
            mean, sem, n = _mean_sem(sub[m])
            row[f"{m}_mean"] = mean
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
    drawn solid; augmented conditions are dashed.  Errors bars are SEM across
    episodes inside each (method, gen_method, real_k) cell.
    """
    import matplotlib.pyplot as plt

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
        label = f"{method_name} · {gen_name}"
        ax.errorbar(
            sub["real_k"], sub["auroc_mean"], yerr=sub["auroc_sem"],
            linestyle=linestyle, marker=marker, capsize=3, label=label,
        )

    ax.set_xlabel("Real support size per class (real_k)")
    ax.set_ylabel("AUROC (query, real-only)")
    title = "Fig 1 — AUROC vs real_k"
    if department:
        title += f"  ·  {department}"
    if method:
        title += f"  ·  {method}"
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="grey", linewidth=0.8, alpha=0.6, linestyle=":")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=200)
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
    """
    import matplotlib.pyplot as plt

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
            )
        ax.set_title(f"real_k = {int(k_val) if pd.notna(k_val) else 'NA'}")
        ax.set_xlabel("synthetic_proportion")
        if col_idx == 0:
            ax.set_ylabel("AUROC (query, real-only)")
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="grey", linewidth=0.8, alpha=0.6, linestyle=":")
        ax.legend(fontsize=8, loc="best")

    suptitle = "Fig 2 — AUROC vs syn_prop"
    if department:
        suptitle += f"  ·  {department}"
    if method:
        suptitle += f"  ·  {method}"
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=200)
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
    Aggregated over methods because realism is a generation-side property
    that does not depend on the meta-learner.
    """
    import matplotlib.pyplot as plt

    df = _ensure_df(results)
    if department is not None:
        df = df[df["department"] == department]

    # Realism is only meaningful when we actually generated synthetic rows.
    df = df[df["n_synthetic_support_rows"] > 0].copy()
    if df.empty:
        # Fallback: return an empty figure with an explanatory annotation so
        # the calling notebook does not crash on a real-only run.
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5,
                "No synthetic rows in results — nothing to plot.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        if save_to is not None:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_to, bbox_inches="tight", dpi=200)
        return fig

    metrics = ("dist_to_nearest_real_mean", "ks_max", "cat_mode_preservation_mean")
    agg = aggregate_by(
        df, group_cols=("real_k", "syn_prop"), metric_cols=metrics,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    titles = {
        "dist_to_nearest_real_mean": "Distance to nearest real (lower = closer)",
        "ks_max": "KS_max across features (lower = closer to real dist.)",
        "cat_mode_preservation_mean": "Categorical mode preservation (1.0 = perfect)",
    }
    for ax, metric in zip(axes, metrics):
        for k_val, sub in agg.groupby("real_k"):
            sub = sub.sort_values("syn_prop")
            ax.errorbar(
                sub["syn_prop"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_sem"],
                marker="o", capsize=3,
                label=f"real_k={int(k_val) if pd.notna(k_val) else 'NA'}",
            )
        ax.set_title(titles[metric], fontsize=10)
        ax.set_xlabel("synthetic_proportion")
        ax.legend(fontsize=8, loc="best")

    suptitle = "Fig 3 — Realism diagnostics vs syn_prop"
    if department:
        suptitle += f"  ·  {department}"
    fig.suptitle(suptitle, y=1.04)
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Table 1 — main summary table
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

    Parameters
    ----------
    results : DataFrame or path
    extra_group_cols : optional iterable of column names appended to the
        canonical grouping keys (e.g. ``("seed",)`` to inspect per-seed
        variability).
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

    # Add collapse rate per cell (fraction of episodes whose pred_std<0.05).
    if "collapsed" in df.columns:
        collapse = (
            df.groupby(group_cols, dropna=False)["collapsed"]
            .apply(lambda s: float(s.dropna().astype(bool).mean()) if len(s.dropna()) else np.nan)
            .reset_index(name="collapse_rate")
        )
        table = table.merge(collapse, on=group_cols, how="left")

    # Sort: by department → real_k → method → syn_prop ascending.  This is the
    # ordering the thesis tables use.
    sort_cols = [c for c in ("department", "real_k", "method", "syn_prop") if c in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return table


__all__ = [
    "load_results",
    "aggregate_by",
    "plot_auroc_vs_real_k",
    "plot_auroc_vs_syn_prop",
    "plot_realism_vs_syn_prop",
    "build_summary_table",
    "REQUIRED_COLS",
    "GROUP_COLS_BASE",
]
