"""
Realism diagnostics for synthetic support augmentation.

Computed per episode alongside the standard performance metrics
(AUROC / NDCG@K / ECE) so the thesis can answer:

    "How do you know the synthetic data isn't garbage?"

Three independent diagnostics
-----------------------------
1. distance_to_nearest_real
   For each synthetic row, Euclidean distance in numeric-feature space
   to its nearest same-class real row. Returns mean and p95 per class.
   Sanity: should not be 0 (synthetic ≠ real clone) and should not be
   wildly larger than the typical real-real intra-class distance.

2. ks_per_feature
   For each numeric feature, KS statistic between the real-support
   distribution and the synthetic-support distribution (per class).
   Returns the per-feature dict, max, and median KS.
   High max KS for any one feature flags suspicious feature drift.

3. categorical_mode_preservation
   For each categorical feature, share of synthetic rows whose value
   appears in the real support's mode set for that feature (per class).
   Should be ~1.0 for SMOTE-NC by design.

Public entry point
------------------
compute_realism_diagnostics(real_support_df, synth_support_df, ...)
    Returns a flat dict suitable for joining onto the per-episode
    result row in reports/tables/synthetic_augmentation_results.csv.

All functions are NaN-safe: numeric NaNs are filled with the real
support's per-column median before distance / KS computations, the
same convention SMOTE-NC uses internally.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pure-numpy replacements for scipy.spatial.distance.cdist and scipy.stats.ks_2samp.
# Kept in-module so this file has no scipy dependency. Both produce results
# identical to scipy on the sizes used in episodic diagnostics.
# ---------------------------------------------------------------------------

def _euclidean_cdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance, shape (len(A), len(B))."""
    if A.size == 0 or B.size == 0:
        return np.zeros((len(A), len(B)), dtype=float)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def _ks_2samp_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max |F_a - F_b|)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_num_matrix(
    df: pd.DataFrame,
    num_cols: Sequence[str],
    fill_medians: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    Convert numeric columns to a float matrix, filling NaN with `fill_medians`
    if provided, else with column medians of `df` itself.
    """
    if not num_cols:
        return np.zeros((len(df), 0), dtype=float)

    sub = df[list(num_cols)].copy()
    if fill_medians is None:
        fill_medians = sub.median(numeric_only=True)
    sub = sub.fillna(fill_medians)
    # Any cols that are still NaN (entire column missing in real) → 0
    sub = sub.fillna(0.0)
    return sub.to_numpy(dtype=float)


def _modes(series: pd.Series) -> set:
    """Return the set of modes (most-frequent values) of a series."""
    if len(series) == 0:
        return set()
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return set()
    top = counts.iloc[0]
    return set(counts[counts == top].index.tolist())


# ---------------------------------------------------------------------------
# 1. Distance to nearest real
# ---------------------------------------------------------------------------

def distance_to_nearest_real(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    num_cols: Sequence[str],
    target_col: str = "gold_y",
    standardize: bool = True,
) -> dict:
    """
    For each synthetic row, Euclidean distance to its nearest same-class
    real row, computed on `num_cols` only.

    Parameters
    ----------
    synth_df, real_df:
        Synthetic and real support rows (must contain `target_col` and
        all `num_cols`).
    num_cols:
        Numeric feature columns used for distance.
    target_col:
        Class column. Distances are computed within-class.
    standardize:
        If True (default), z-score every numeric column using the
        per-class real-support mean and std BEFORE computing pairwise
        distances.  Without this, the metric is dominated by whichever
        feature has the largest absolute scale (e.g. revenue in millions
        next to ratios in [0, 1]) and the resulting numbers are not
        comparable across cells or across departments.

        Set ``standardize=False`` only if you explicitly want raw-units
        distance (e.g. for sanity-checking against a hand calculation).

    Returns
    -------
    dict with keys:
        - dist_to_nearest_real_mean : mean across all synthetic rows
        - dist_to_nearest_real_p95  : 95th percentile across all synth rows
        - dist_to_nearest_real_per_class : {class: {mean, p95, n}}

    Returns NaN for mean/p95 if there are no synthetic rows.
    """
    if len(synth_df) == 0 or not num_cols:
        return {
            "dist_to_nearest_real_mean": float("nan"),
            "dist_to_nearest_real_p95": float("nan"),
            "dist_to_nearest_real_per_class": {},
        }

    all_dists: list[float] = []
    per_class: dict = {}

    classes = sorted(synth_df[target_col].unique())
    for cls in classes:
        s = synth_df[synth_df[target_col] == cls]
        r = real_df[real_df[target_col] == cls]

        if len(s) == 0 or len(r) == 0:
            per_class[cls] = {"mean": float("nan"), "p95": float("nan"), "n": len(s)}
            continue

        # Within-class imputation — matches SMOTE-NC's per-class median fill.
        class_medians = r[list(num_cols)].median(numeric_only=True)
        S = _to_num_matrix(s, num_cols, fill_medians=class_medians)
        R = _to_num_matrix(r, num_cols, fill_medians=class_medians)

        if standardize:
            # Fit z-score on the REAL support (not on the synthetic rows).
            # This makes a unit move in any numeric column count the same
            # toward distance, so the result is comparable across cells.
            mu = R.mean(axis=0)
            sd = R.std(axis=0, ddof=0)
            # Guard against zero-variance columns (e.g. a numeric flag
            # that happens to be constant within this class).  Replacing
            # sd=0 with 1 means those columns contribute 0 to the
            # distance, which is the right behaviour: a constant column
            # carries no information about row-to-row similarity.
            sd_safe = np.where(sd > 0, sd, 1.0)
            S = (S - mu) / sd_safe
            R = (R - mu) / sd_safe

        # Pairwise distances, shape (len_s, len_r); min along axis=1 = nearest real
        d = _euclidean_cdist(S, R).min(axis=1)

        per_class[cls] = {
            "mean": float(d.mean()),
            "p95": float(np.percentile(d, 95)),
            "n": int(len(s)),
        }
        all_dists.extend(d.tolist())

    if not all_dists:
        return {
            "dist_to_nearest_real_mean": float("nan"),
            "dist_to_nearest_real_p95": float("nan"),
            "dist_to_nearest_real_per_class": per_class,
        }

    arr = np.asarray(all_dists)
    return {
        "dist_to_nearest_real_mean": float(arr.mean()),
        "dist_to_nearest_real_p95": float(np.percentile(arr, 95)),
        "dist_to_nearest_real_per_class": per_class,
    }


# ---------------------------------------------------------------------------
# 2. KS statistic per feature
# ---------------------------------------------------------------------------

def ks_per_feature(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    num_cols: Sequence[str],
    target_col: str = "gold_y",
) -> dict:
    """
    Two-sample KS statistic per numeric feature, comparing real-support
    and synthetic-support distributions WITHIN each class.

    KS is computed per (feature, class) and aggregated across all
    (feature, class) pairs into max and median.

    Returns
    -------
    dict with keys:
        - ks_max     : worst-case KS across (feature, class) pairs
        - ks_median  : median KS across (feature, class) pairs
        - ks_per_feature : {feature: {class: ks_stat}}

    NaN if synth or real is empty in a given class.
    """
    if len(synth_df) == 0 or len(real_df) == 0 or not num_cols:
        return {"ks_max": float("nan"), "ks_median": float("nan"), "ks_per_feature": {}}

    classes = sorted(set(synth_df[target_col].unique()) & set(real_df[target_col].unique()))
    per_feature: dict = {col: {} for col in num_cols}
    flat: list[float] = []

    for col in num_cols:
        for cls in classes:
            r_vals = real_df.loc[real_df[target_col] == cls, col].dropna().to_numpy()
            s_vals = synth_df.loc[synth_df[target_col] == cls, col].dropna().to_numpy()
            if len(r_vals) < 2 or len(s_vals) < 2:
                per_feature[col][cls] = float("nan")
                continue
            try:
                stat = _ks_2samp_stat(r_vals, s_vals)
            except Exception:
                stat = float("nan")
            per_feature[col][cls] = stat
            if not np.isnan(stat):
                flat.append(stat)

    if not flat:
        return {"ks_max": float("nan"), "ks_median": float("nan"), "ks_per_feature": per_feature}

    arr = np.asarray(flat)
    return {
        "ks_max": float(arr.max()),
        "ks_median": float(np.median(arr)),
        "ks_per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# 3. Categorical mode preservation
# ---------------------------------------------------------------------------

def categorical_mode_preservation(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cat_cols: Sequence[str],
    target_col: str = "gold_y",
) -> dict:
    """
    For each categorical feature, share of synthetic rows whose value is
    in the real-support's mode set for that feature (per class), then
    averaged across (feature, class) pairs.

    SMOTE-NC picks categorical values via mode of (anchor + k neighbors),
    so this share should be close to 1.0 by construction. Values noticeably
    below 1.0 indicate either a sparse class or a generator quirk.

    Returns
    -------
    dict with keys:
        - cat_mode_preservation_mean : average share across (feature, class)
        - cat_mode_preservation_min  : worst (feature, class)
        - cat_mode_preservation_per_feature : {feature: {class: share}}
    """
    if len(synth_df) == 0 or len(real_df) == 0 or not cat_cols:
        return {
            "cat_mode_preservation_mean": float("nan"),
            "cat_mode_preservation_min": float("nan"),
            "cat_mode_preservation_per_feature": {},
        }

    classes = sorted(set(synth_df[target_col].unique()) & set(real_df[target_col].unique()))
    per_feature: dict = {col: {} for col in cat_cols}
    flat: list[float] = []

    for col in cat_cols:
        for cls in classes:
            r_vals = real_df.loc[real_df[target_col] == cls, col]
            s_vals = synth_df.loc[synth_df[target_col] == cls, col]
            if len(r_vals) == 0 or len(s_vals) == 0:
                per_feature[col][cls] = float("nan")
                continue
            modes = _modes(r_vals)
            share = float(s_vals.isin(modes).mean())
            per_feature[col][cls] = share
            flat.append(share)

    if not flat:
        return {
            "cat_mode_preservation_mean": float("nan"),
            "cat_mode_preservation_min": float("nan"),
            "cat_mode_preservation_per_feature": per_feature,
        }

    arr = np.asarray(flat)
    return {
        "cat_mode_preservation_mean": float(arr.mean()),
        "cat_mode_preservation_min": float(arr.min()),
        "cat_mode_preservation_per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_realism_diagnostics(
    real_support_df: pd.DataFrame,
    synth_support_df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    target_col: str = "gold_y",
    include_breakdowns: bool = False,
) -> dict:
    """
    Compute all three realism diagnostics for ONE episode and return a
    flat dict suitable for appending to a per-episode result row.

    Parameters
    ----------
    real_support_df:
        Real support rows used in the episode.
    synth_support_df:
        Synthetic support rows generated for the episode.
        May be empty (real-only baseline) — function returns NaN diagnostics.
    num_cols, cat_cols:
        Feature columns. Should match what the augmentation method saw.
    target_col:
        Class label column.
    include_breakdowns:
        If True, also include the nested per-class / per-feature dicts
        (useful for the appendix; noisy for the main results table).

    Returns
    -------
    dict with the headline scalars:
        dist_to_nearest_real_mean, dist_to_nearest_real_p95,
        ks_max, ks_median,
        cat_mode_preservation_mean, cat_mode_preservation_min,
        n_synthetic_rows
    Optionally also the nested breakdowns when include_breakdowns=True.
    """
    out: dict = {"n_synthetic_rows": int(len(synth_support_df))}

    dist = distance_to_nearest_real(
        synth_support_df, real_support_df, num_cols, target_col=target_col
    )
    ks = ks_per_feature(
        real_support_df, synth_support_df, num_cols, target_col=target_col
    )
    cat = categorical_mode_preservation(
        real_support_df, synth_support_df, cat_cols, target_col=target_col
    )

    out["dist_to_nearest_real_mean"] = dist["dist_to_nearest_real_mean"]
    out["dist_to_nearest_real_p95"] = dist["dist_to_nearest_real_p95"]
    out["ks_max"] = ks["ks_max"]
    out["ks_median"] = ks["ks_median"]
    out["cat_mode_preservation_mean"] = cat["cat_mode_preservation_mean"]
    out["cat_mode_preservation_min"] = cat["cat_mode_preservation_min"]

    if include_breakdowns:
        out["dist_to_nearest_real_per_class"] = dist["dist_to_nearest_real_per_class"]
        out["ks_per_feature"] = ks["ks_per_feature"]
        out["cat_mode_preservation_per_feature"] = cat["cat_mode_preservation_per_feature"]

    return out
