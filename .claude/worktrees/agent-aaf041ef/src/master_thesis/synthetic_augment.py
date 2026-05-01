"""
Synthetic support augmentation for Stage 2b episodic few-shot learning.

This module generates synthetic tabular feature vectors to augment the support
set of a department episode when real labeled contracts are scarce.

Design rules
------------
- Synthetic data is generated PER CLASS (Yes / No separately).
- Synthetic rows are NEVER placed in the query set.
- Every synthetic row is tagged with is_synthetic=True and label_source.
- The department column is always preserved exactly — never interpolated.
- contract_id for synthetic rows uses a UUID prefix to avoid collisions.

Public API
----------
augment_support(support_df, feature_cols, ...) -> pd.DataFrame

Parameters (key)
----------------
target_per_class : int, optional
    Fill each class up to exactly this many rows (real + synthetic).
    Use this for the "5-shot requirement" experiment.
    Takes priority over synthetic_proportion when set.

synthetic_proportion : float, optional
    Fraction of final support that is synthetic:
    proportion = n_synthetic / (n_real + n_synthetic).
    Use this for the shots-vs-performance sweep (0.0, 0.5, 0.75).

Methods
-------
"none"           : passthrough, no augmentation
"gaussian_noise" : clone real rows + Gaussian noise on numeric cols
"mixup"          : convex combinations of two same-class rows
"smote_nc"       : k-NN interpolation (numeric) + mode (categorical)
"""

from __future__ import annotations

import uuid
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_rng(random_state: Optional[int]) -> np.random.Generator: # This creates a reproducible random generator.
    return np.random.default_rng(random_state)


def _detect_categorical_cols(
    df: pd.DataFrame,
    interp_feature_cols: Sequence[str],
    explicit_categorical_cols: Optional[Sequence[str]],
    protected: set[str],
) -> tuple[list[str], list[str]]:
    """
    Return (numeric_cols, categorical_cols) from interp_feature_cols.

    Protected columns (department, contract_id, target) are always excluded.
    If explicit_categorical_cols is provided, use that (minus protected cols).
    Otherwise auto-detect: object/category dtypes → categorical.
    """
    if explicit_categorical_cols is not None:
        # Fix 2: exclude protected cols that may have slipped in
        cat_cols = [
            c for c in explicit_categorical_cols
            if c in interp_feature_cols and c not in protected
        ]
        # Anything not explicitly categorical AND not numeric is also
        # categorical — protects us from string/object cols leaking into
        # num_cols and breaking .median().
        num_cols = [
            c for c in interp_feature_cols
            if c not in cat_cols
            and c not in protected
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        # Re-classify any non-numeric leftover as categorical.
        leftovers = [
            c for c in interp_feature_cols
            if c not in cat_cols
            and c not in protected
            and c not in num_cols
        ]
        cat_cols = cat_cols + leftovers
    else:
        # Auto-detect: anything that isn't a numeric dtype is categorical.
        # This is robust to pandas StringDtype, object, category, and bool —
        # the previous check missed `string` dtype and silently let it fall
        # into num_cols, which then crashed `.median()` downstream.
        num_cols = [
            c for c in interp_feature_cols
            if c not in protected
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        cat_cols = [
            c for c in interp_feature_cols
            if c not in protected
            and c not in num_cols
        ]

    return num_cols, cat_cols


def _make_synthetic_id() -> str:                  # This creates a new fake contract ID.
    return f"SYNTH_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Per-method generators (operate on one class at a time)
# ---------------------------------------------------------------------------

# Generator 1: simplest method: clone real rows and add Gaussian noise to numeric cols.
# keep categorical cols the same. 
# give it a synthetic contract_id.
# Baseline that tests: Does simply adding small variation help?
def _gaussian_noise(
    class_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_generate: int,
    noise_scale: float,
    rng: np.random.Generator,
    contract_id_col: str,
) -> pd.DataFrame:
    """
    Clone real rows uniformly at random and add Gaussian noise to numeric cols.
    Categorical cols are copied exactly.
    """
    if len(class_df) == 0 or n_generate == 0:
        return pd.DataFrame(columns=class_df.columns)

    # Fill NaN with column median before cloning
    class_df = class_df.copy()
    if num_cols:
        class_df[num_cols] = class_df[num_cols].fillna(class_df[num_cols].median())

    idx = rng.integers(0, len(class_df), size=n_generate)
    synthetic = class_df.iloc[idx].copy().reset_index(drop=True)

    for col in num_cols:
        col_std = class_df[col].std(ddof=0)
        if pd.isna(col_std) or col_std == 0.0:
            col_std = 1.0
        noise = rng.normal(0, noise_scale * col_std, size=n_generate)
        synthetic[col] = synthetic[col] + noise

    if contract_id_col in synthetic.columns:
        synthetic[contract_id_col] = [_make_synthetic_id() for _ in range(n_generate)]

    return synthetic

# Generator 2: mixup,combines two same-class rows.
def _mixup(
    class_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_generate: int,
    rng: np.random.Generator,
    contract_id_col: str,
) -> pd.DataFrame:
    """
    Convex combination of two randomly chosen same-class rows.
    λ ~ Uniform(0, 1) applied to numeric cols.
    Categorical cols: taken from the row with the higher λ weight.
    """
    if len(class_df) < 2 or n_generate == 0: # If there is only one row in the class, mixup cannot work, so it falls back to Gaussian noise.
        # Fix 5 / Fix 6: log fallback clearly
        warnings.warn(
            f"mixup: only {len(class_df)} row(s) in class — falling back to gaussian_noise. "
            "Synthetic rows will be tagged augmentation_fallback=True.",
            UserWarning,
            stacklevel=3,
        )
        result = _gaussian_noise(
            class_df, num_cols, cat_cols, n_generate,
            noise_scale=0.05, rng=rng, contract_id_col=contract_id_col,
        )
        result["augmentation_fallback"] = True
        return result

    # Fix 4: fill NaN with median before blending
    class_df = class_df.copy()
    if num_cols:
        class_df[num_cols] = class_df[num_cols].fillna(class_df[num_cols].median())

    rows = []
    n = len(class_df)
    for _ in range(n_generate):
        i, j = rng.choice(n, size=2, replace=False)
        a = class_df.iloc[i]
        b = class_df.iloc[j]
        lam = float(rng.uniform(0, 1))

        new_row = {}
        for col in num_cols:
            new_row[col] = lam * float(a[col]) + (1 - lam) * float(b[col])
        for col in cat_cols:
            new_row[col] = a[col] if lam >= 0.5 else b[col]

        dominant = a if lam >= 0.5 else b
        for col in class_df.columns:
            if col not in new_row:
                new_row[col] = dominant[col]

        if contract_id_col in new_row:
            new_row[contract_id_col] = _make_synthetic_id()

        rows.append(new_row)

    return pd.DataFrame(rows, columns=class_df.columns)

# Generator 3: SMOTE-NC, in
def _smote_nc(
    class_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_generate: int,
    k_neighbors: int,
    rng: np.random.Generator,
    contract_id_col: str,
) -> pd.DataFrame:
    """
    SMOTE-NC: interpolate along k-nearest-neighbor edges (numeric),
    mode selection for categorical cols.

    **Option B (same-contract exclusion):** When selecting k-nearest
    neighbors for row i, any row j with the same contract_id is excluded.
    This ensures interpolation always happens between genuinely different
    contracts, not between year-rows of the same contract.

    If the class contains fewer than 2 unique contracts, falls back to
    Gaussian noise (tagged with augmentation_fallback=True).

    Distance metric:
        d(x_i, x_j) = sqrt( sum((x_i[r] - x_j[r])^2 for r in num_cols) )
                     + sigma_R * |{f in cat_cols : x_i[f] != x_j[f]}|
    where sigma_R = median std of numeric cols (SMOTE-NC original paper).
    """
    n = len(class_df)

    # Option B: check unique contracts, not just row count
    contract_ids = class_df[contract_id_col].values if contract_id_col in class_df.columns else None
    n_unique_contracts = len(set(contract_ids)) if contract_ids is not None else n

    if n < 2 or n_generate == 0 or n_unique_contracts < 2: # If there are fewer than 2 unique contracts in that class, SMOTE-NC cannot run. It needs another contract to interpolate with.
        reason = (
            f"only {n_unique_contracts} unique contract(s)"
            if n_unique_contracts < 2
            else f"only {n} row(s)"
        )
        warnings.warn(
            f"smote_nc: {reason} in class — falling back to gaussian_noise. "
            "Synthetic rows will be tagged augmentation_fallback=True.",
            UserWarning,
            stacklevel=3,
        )
        result = _gaussian_noise(
            class_df, num_cols, cat_cols, n_generate,
            noise_scale=0.05, rng=rng, contract_id_col=contract_id_col,
        )
        result["augmentation_fallback"] = True
        return result

    # Build numeric matrix (fill NaN with col median). Before computing distances, missing numeric values are filled with the class median. This avoids broken distance calculations.
    class_df = class_df.copy()
    if num_cols:
        class_df[num_cols] = class_df[num_cols].fillna(class_df[num_cols].median())

    X_num = class_df[num_cols].to_numpy(dtype=float) \
        if num_cols else np.zeros((n, 0))

    if X_num.shape[1] > 0:
        stds = X_num.std(axis=0)
        sigma_R = float(np.median(stds[stds > 0])) if (stds > 0).any() else 1.0
    else:
        sigma_R = 1.0

    X_cat = class_df[cat_cols].to_numpy(dtype=object) if cat_cols else np.zeros((n, 0), dtype=object)

    # Pairwise distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            num_d = float(np.sqrt(np.sum((X_num[i] - X_num[j]) ** 2))) if X_num.shape[1] > 0 else 0.0 # Euclidean distance between numeric vectors of row i and row j.
            cat_d = float(np.sum(X_cat[i] != X_cat[j])) * sigma_R if X_cat.shape[1] > 0 else 0.0 # Number of disagreeing categorical values between row i and row j, weighted by sigma_R.
            dist[i, j] = dist[j, i] = num_d + cat_d

    rows = []
    class_df_reset = class_df.reset_index(drop=True)

    for _ in range(n_generate):
        i = int(rng.integers(0, n))

        # --- Option B: exclude same-contract neighbors ---
        # Sort all rows by distance to i, then filter out rows
        # belonging to the same contract as row i.
        sorted_indices = np.argsort(dist[i])  # includes self at position 0
        i_contract = contract_ids[i] if contract_ids is not None else i

        cross_contract_neighbors = [
            idx for idx in sorted_indices
            if idx != i and (
                contract_ids is None or contract_ids[idx] != i_contract # forces SMOTE-NC to interpolate across different contracts, not just different rows of the same contract.
            )
        ]

        # Take top-k cross-contract neighbors
        k_actual = min(k_neighbors, len(cross_contract_neighbors))
        neighbor_ids = cross_contract_neighbors[:k_actual]

        j = int(rng.choice(neighbor_ids))
        # --- End Option B ---

        x_i = class_df_reset.iloc[i]
        lam = float(rng.uniform(0, 1))

        new_row = {}

        for col_idx, col in enumerate(num_cols):
            vi = X_num[i, col_idx]
            vj = X_num[j, col_idx]
            new_row[col] = vi + lam * (vj - vi) # For numeric columns, the new value is a linear interpolation between the anchor row and its nearest neighbor.

        # Categorical: mode across x_i and its cross-contract k neighbors
        for col_idx, col in enumerate(cat_cols):
            neighbor_vals = [X_cat[nid, col_idx] for nid in neighbor_ids]
            all_vals = [X_cat[i, col_idx]] + neighbor_vals
            counts: dict = {}
            for v in all_vals:
                counts[v] = counts.get(v, 0) + 1
            new_row[col] = max(counts, key=lambda v: counts[v])

        for col in class_df.columns:
            if col not in new_row:
                new_row[col] = x_i[col]

        if contract_id_col in new_row:
            new_row[contract_id_col] = _make_synthetic_id()

        rows.append(new_row)

    return pd.DataFrame(rows, columns=class_df.columns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_support(
    support_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "gold_y",
    method: str = "smote_nc",
    synthetic_proportion: Optional[float] = None,
    target_per_class: Optional[int] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    contract_id_col: str = "contract_id",
    department_col: str = "department",
    k_neighbors: int = 5,
    gaussian_noise_scale: float = 0.05,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Augment a support set with synthetic tabular feature vectors.

    Synthetic data is generated PER CLASS so that Yes-pattern contracts
    only interpolate with other Yes-pattern contracts (and likewise for No).

    Parameters
    ----------
    support_df:
        Real support set for one department episode.
    feature_cols:
        Feature columns used by the meta-learner.
    target_col:
        Binary label column (gold_y).
    method:
        One of "none", "gaussian_noise", "mixup", "smote_nc".
    synthetic_proportion:
        Fraction of final support that is synthetic.
        proportion = n_synthetic / (n_real + n_synthetic).
        Use for the shots-vs-performance sweep (0.0, 0.5, 0.75).
        Ignored if target_per_class is set.
    target_per_class:
        Fill each class up to exactly this many rows (real + synthetic).
        Use for the "5-shot requirement" experiment.
        Takes priority over synthetic_proportion.
        Example: target_per_class=5 with 2 real Yes → generates 3 synthetic Yes.
    categorical_cols:
        Explicit list of categorical feature columns (protected cols excluded).
        If None, auto-detected from dtype (object / category).
    contract_id_col:
        Column holding contract identifiers.
    department_col:
        Department column — always frozen, never interpolated.
    k_neighbors:
        Number of nearest neighbors for smote_nc (capped at class size - 1).
    gaussian_noise_scale:
        Std multiplier for gaussian_noise method (fraction of each col's std).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Augmented support set (real rows first, then synthetic rows).
        Synthetic rows have:
          - is_synthetic = True
          - label_source = "SYNTHETIC_{method}"
          - augmentation_fallback = True  (only if method fell back to gaussian_noise)
    """
    # Validate method
    valid_methods = ("none", "gaussian_noise", "mixup", "smote_nc")
    if method not in valid_methods:
        raise ValueError(
            f"Unknown augmentation method '{method}'. "
            f"Choose from: {valid_methods}."
        )

    # Passthrough: no augmentation
    if method == "none":
        df = support_df.copy()
        df["is_synthetic"] = False
        df["label_source"] = "REAL"
        return df

    # Validate that at least one of the two sizing modes is set
    if target_per_class is None and synthetic_proportion is None:
        raise ValueError(
            "Provide either target_per_class or synthetic_proportion."
        )

    if synthetic_proportion is not None and not (0.0 <= synthetic_proportion < 1.0):
        raise ValueError(
            f"synthetic_proportion must be in [0, 1), got {synthetic_proportion}."
        )

    # Passthrough: proportion = 0
    if synthetic_proportion is not None and synthetic_proportion <= 0.0:
        df = support_df.copy()
        df["is_synthetic"] = False
        df["label_source"] = "REAL"
        return df

    rng = _get_rng(random_state)

    # Fix 2: exclude protected cols from interpolation feature set
    protected = {contract_id_col, target_col, department_col}
    interp_feature_cols = [c for c in feature_cols if c not in protected]

    num_cols, cat_cols = _detect_categorical_cols(
        support_df, interp_feature_cols, categorical_cols, protected
    )

    n_real = len(support_df)
    classes = sorted(support_df[target_col].unique())
    synthetic_parts = []

    for cls in classes:
        class_df = support_df[support_df[target_col] == cls].copy()
        n_class_real = len(class_df)

        # Fix 1: target_per_class takes priority over proportion
        if target_per_class is not None:
            n_class_synth = max(0, target_per_class - n_class_real)
        else:
            # proportion-based: allocate proportionally across classes
            n_synthetic_total = int(
                round(synthetic_proportion / (1 - synthetic_proportion) * n_real)
            )
            n_class_synth = int(round(n_class_real / n_real * n_synthetic_total))

        if n_class_synth == 0:
            continue

        if method == "gaussian_noise":
            synth = _gaussian_noise(
                class_df, num_cols, cat_cols, n_class_synth,
                noise_scale=gaussian_noise_scale, rng=rng,
                contract_id_col=contract_id_col,
            )
        elif method == "mixup":
            synth = _mixup(
                class_df, num_cols, cat_cols, n_class_synth,
                rng=rng, contract_id_col=contract_id_col,
            )
        elif method == "smote_nc":
            synth = _smote_nc(
                class_df, num_cols, cat_cols, n_class_synth,
                k_neighbors=k_neighbors, rng=rng,
                contract_id_col=contract_id_col,
            )

        synth[target_col] = cls
        # Fix 3: label_source for every synthetic row
        synth["label_source"] = f"SYNTHETIC_{method.upper()}"
        synthetic_parts.append(synth)

    if not synthetic_parts:
        df = support_df.copy()
        df["is_synthetic"] = False
        df["label_source"] = "REAL"
        return df

    synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
    synthetic_df["is_synthetic"] = True
    if "augmentation_fallback" not in synthetic_df.columns:
        synthetic_df["augmentation_fallback"] = False

    real_df = support_df.copy()
    real_df["is_synthetic"] = False
    real_df["label_source"] = "REAL"
    real_df["augmentation_fallback"] = False

    augmented = pd.concat([real_df, synthetic_df], ignore_index=True)

    # Fix 6: raise hard error if department changed (must never happen)
    if department_col in augmented.columns:
        original_depts = set(support_df[department_col].unique())
        augmented_depts = set(augmented[department_col].unique())
        if not augmented_depts.issubset(original_depts):
            raise ValueError(
                f"Department changed during augmentation! "
                f"Original: {original_depts}, Augmented: {augmented_depts}. "
                "This is a leakage bug — department must always be frozen."
            )

    return augmented
