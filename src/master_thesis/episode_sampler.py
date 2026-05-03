"""
Utilities for Stage 2 episodic task construction.

This module is responsible for:
- building the department-level gold-label task table
- summarizing label availability per department
- filtering valid departments for few-shot meta-learning
- sampling contract-aware support/query splits
- sampling meta-batches of department tasks
- constructing held-out target department meta-test episodes

This module prepares episodic task data for:
- ANIL
- FOMAML
- fine-tuning baseline
- adaptation curves
- ablations
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Private helpers
# ============================================================================

def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: Sequence[str],
    df_name: str,
) -> None:
    """
    Raise a clear error if required columns are missing from a dataframe.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {df_name}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _get_rng(random_state: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy random generator from an optional integer seed.
    """
    return np.random.default_rng(random_state)


def _extract_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target vector y from a dataframe.
    """
    _validate_required_columns(df, list(feature_cols) + [target_col], "episode dataframe")

    X = df.loc[:, feature_cols].to_numpy()
    y = df.loc[:, target_col].to_numpy(dtype=int)

    return X, y


def _make_episode_dict(
    department: str,
    support_df: pd.DataFrame,
    query_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    contract_id_col: str,
) -> Dict[str, Any]:
    """
    Standardize the output format of one episode.
    """
    X_support, y_support = _extract_xy(support_df, feature_cols, target_col)
    X_query, y_query = _extract_xy(query_df, feature_cols, target_col)

    return {
        "department": department,
        "support_df": support_df.copy(),
        "query_df": query_df.copy(),
        "support_ids": support_df[contract_id_col].tolist() if contract_id_col in support_df.columns else [],
        "query_ids": query_df[contract_id_col].tolist() if contract_id_col in query_df.columns else [],
        "X_support": X_support,
        "y_support": y_support,
        "X_query": X_query,
        "y_query": y_query,
    }


def _sample_rows(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample n rows without replacement from a dataframe.

    Note:
    This helper is currently not used in the contract-aware episode sampler,
    but is kept as a utility in case fixed-size row-level sampling is needed later.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n > len(df):
        raise ValueError(f"Cannot sample {n} rows from dataframe of size {len(df)}.")

    if n == 0:
        return df.iloc[0:0].copy()

    sampled_idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[sampled_idx].copy()


# ============================================================================
# Public API: task table construction and summaries
# ============================================================================

def build_department_task_table(
    features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    contract_id_col: str = "contract_id",
    department_col: str = "department",
    target_col: str = "gold_y",
    observation_year_col: str = "observation_year",
    drop_missing_features: bool = False,
) -> pd.DataFrame:
    """
    Build the master Stage 2 task table.

    Behavior:
    - keep only gold-labeled rows
    - drop rows without department
    - retain contract-year structure
    - keep selected feature columns
    - optionally drop rows with missing feature values

    Returns
    -------
    pd.DataFrame
        Task table containing:
        - contract_id
        - department
        - observation_year
        - gold_y
        - feature columns
    """
    required_cols = [
        contract_id_col,
        department_col,
        observation_year_col,
        target_col,
    ] + list(feature_cols)

    _validate_required_columns(
        features_df,
        required_cols,
        "features_df",
    )

    # Keep only gold-labeled rows
    task_df = features_df.loc[
        features_df[target_col].notna()
    ].copy()

    # Drop rows without department
    task_df = task_df.loc[
        task_df[department_col].notna()
    ].copy()

    # Ensure binary integer target
    task_df[target_col] = task_df[target_col].astype(int)

    # Optionally drop rows with missing features
    if drop_missing_features:
        task_df = task_df.dropna(subset=list(feature_cols)).copy()

    # Keep only required columns
    keep_cols = [
        contract_id_col,
        department_col,
        observation_year_col,
        target_col,
    ] + list(feature_cols)

    # Deduplicate keeping order to prevent groupby errors if a primary col is also a feature
    keep_cols = list(dict.fromkeys(keep_cols))

    task_df = task_df.loc[:, keep_cols].copy()
    task_df = task_df.reset_index(drop=True)

    print("Stage 2 task table created")
    print("Rows:", len(task_df))
    print("Unique contracts:", task_df[contract_id_col].nunique())
    print("Departments:", task_df[department_col].nunique())
    print("Positive labels:", int((task_df[target_col] == 1).sum()))
    print("Negative labels:", int((task_df[target_col] == 0).sum()))

    return task_df


def summarize_department_tasks(
    task_df: pd.DataFrame,
    department_col: str = "department",
    target_col: str = "gold_y",
    contract_id_col: str = "contract_id",
) -> pd.DataFrame:
    """
    Summarize gold-label availability per department.

    Includes both row-level and contract-level counts.

    Output columns include:
    - department
    - n_total_rows
    - n_pos_rows
    - n_neg_rows
    - n_total_contracts
    - n_pos_contracts
    - n_neg_contracts
    - pos_rate_rows
    - pos_rate_contracts
    """
    _validate_required_columns(
        task_df,
        [department_col, target_col, contract_id_col],
        "task_df",
    )

    # Row-level summary
    row_summary = (
        task_df
        .groupby(department_col)
        .agg(
            n_total_rows=(target_col, "count"),
            n_pos_rows=(target_col, lambda x: int((x == 1).sum())),
            n_neg_rows=(target_col, lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )

    # Contract-level summary
    contract_level = (
        task_df[[department_col, contract_id_col, target_col]]
        .drop_duplicates(subset=[department_col, contract_id_col])
        .groupby(department_col)
        .agg(
            n_total_contracts=(contract_id_col, "count"),
            n_pos_contracts=(target_col, lambda x: int((x == 1).sum())),
            n_neg_contracts=(target_col, lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )

    summary = row_summary.merge(
        contract_level,
        on=department_col,
        how="left",
    )

    summary["pos_rate_rows"] = summary["n_pos_rows"] / summary["n_total_rows"]
    summary["pos_rate_contracts"] = summary["n_pos_contracts"] / summary["n_total_contracts"]

    summary = summary.sort_values(
        by=["n_total_contracts", "n_total_rows"],
        ascending=[False, False],
    ).reset_index(drop=True)

    print("Department task summary created")
    print("Departments:", len(summary))
    print("Total labeled rows:", int(summary["n_total_rows"].sum()))
    print("Total positive rows:", int(summary["n_pos_rows"].sum()))
    print("Total negative rows:", int(summary["n_neg_rows"].sum()))
    print("Total labeled contracts:", int(summary["n_total_contracts"].sum()))
    print("Total positive contracts:", int(summary["n_pos_contracts"].sum()))
    print("Total negative contracts:", int(summary["n_neg_contracts"].sum()))

    return summary


def filter_valid_departments(
    task_df: pd.DataFrame,
    department_col: str = "department",
    target_col: str = "gold_y",
    contract_id_col: str = "contract_id",
    n_support_pos: int = 2,
    n_support_neg: int = 2,
    min_query_contracts: int = 1,
    require_both_query_classes: bool = False,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Filter departments that can support the requested few-shot episode setup.

    Validation is contract-aware:
    support/query feasibility is checked at the contract level.

    Parameters
    ----------
    task_df:
        Stage 2 task table.
    department_col:
        Department/task column.
    target_col:
        Binary target column.
    contract_id_col:
        Contract ID used for contract-aware validation.
    n_support_pos:
        Number of positive support contracts required.
    n_support_neg:
        Number of negative support contracts required.
    min_query_contracts:
        Minimum number of query contracts remaining after support selection.
    require_both_query_classes:
        If True, require at least one positive and one negative contract to remain in query.

    Returns
    -------
    filtered_task_df:
        Only rows from valid departments.
    valid_departments:
        List of valid department names.
    validity_summary_df:
        Department-level summary with validity flags and reasons.
    """
    _validate_required_columns(
        task_df,
        [department_col, target_col, contract_id_col],
        "task_df",
    )

    summary = summarize_department_tasks(
        task_df=task_df,
        department_col=department_col,
        target_col=target_col,
        contract_id_col=contract_id_col,
    ).copy()

    required_support_contracts = n_support_pos + n_support_neg

    summary["remaining_contracts_after_support"] = (
        summary["n_total_contracts"] - required_support_contracts
    )

    summary["enough_pos_for_support"] = summary["n_pos_contracts"] >= n_support_pos
    summary["enough_neg_for_support"] = summary["n_neg_contracts"] >= n_support_neg
    summary["enough_total_for_query"] = (
        summary["remaining_contracts_after_support"] >= min_query_contracts
    )

    if require_both_query_classes:
        summary["enough_pos_for_query"] = (
            summary["n_pos_contracts"] - n_support_pos
        ) >= 1
        summary["enough_neg_for_query"] = (
            summary["n_neg_contracts"] - n_support_neg
        ) >= 1

        summary["valid_department"] = (
            summary["enough_pos_for_support"]
            & summary["enough_neg_for_support"]
            & summary["enough_total_for_query"]
            & summary["enough_pos_for_query"]
            & summary["enough_neg_for_query"]
        )
    else:
        summary["valid_department"] = (
            summary["enough_pos_for_support"]
            & summary["enough_neg_for_support"]
            & summary["enough_total_for_query"]
        )

    def _build_reason(row: pd.Series) -> str:
        reasons = []

        if not row["enough_pos_for_support"]:
            reasons.append("insufficient_positive_contracts_for_support")
        if not row["enough_neg_for_support"]:
            reasons.append("insufficient_negative_contracts_for_support")
        if not row["enough_total_for_query"]:
            reasons.append("insufficient_query_contracts")

        if require_both_query_classes:
            if not row["enough_pos_for_query"]:
                reasons.append("no_positive_contract_left_for_query")
            if not row["enough_neg_for_query"]:
                reasons.append("no_negative_contract_left_for_query")

        return "valid" if not reasons else "|".join(reasons)

    summary["validity_reason"] = summary.apply(_build_reason, axis=1)

    valid_departments = (
        summary.loc[summary["valid_department"], department_col]
        .astype(str)
        .tolist()
    )

    filtered_task_df = task_df.loc[
        task_df[department_col].astype(str).isin(valid_departments)
    ].copy()

    print("Department validity filtering completed")
    print("Valid departments:", len(valid_departments))
    print("Invalid departments:", len(summary) - len(valid_departments))

    return filtered_task_df, valid_departments, summary


# ============================================================================
# Public API: support/query and episode sampling
# ============================================================================

def sample_support_query_split(
    dept_df: pd.DataFrame,
    feature_cols: Sequence[str],
    department_name: Optional[str] = None,
    target_col: str = "gold_y",
    department_col: str = "department",
    contract_id_col: str = "contract_id",
    n_support_pos: int = 2,
    n_support_neg: int = 2,
    query_strategy: str = "remainder",
    random_state: Optional[int] = None,
    require_both_query_classes: bool = False,
) -> Dict[str, Any]:
    """
    Sample one contract-aware support/query episode for a single department.

    Support/query splitting is done at the contract level:
    the same contract_id will never appear in both support and query.

    Support set construction:
    - sample n_support_pos positive contracts
    - sample n_support_neg negative contracts
    - include all rows of those contracts in support

    Query set construction:
    - all remaining contracts and their rows

    Parameters
    ----------
    require_both_query_classes:
        If True, raise ValueError when the query set does not contain at least
        one positive and one negative contract (checked at contract level).

    Returns
    -------
    Dict[str, Any]
        Episode dictionary containing support/query dataframes and X/y arrays.
    """
    _validate_required_columns(
        dept_df,
        [department_col, target_col, contract_id_col] + list(feature_cols),
        "dept_df",
    )

    if department_name is None:
        dept_values = dept_df[department_col].dropna().unique().tolist()
        if len(dept_values) != 1:
            raise ValueError(
                f"dept_df must contain exactly one department. Found: {dept_values}"
            )
        department_name = str(dept_values[0])

    if query_strategy != "remainder":
        raise ValueError("Currently only query_strategy='remainder' is supported.")

    rng = _get_rng(random_state)

    # Contract-level labels
    contract_labels = (
        dept_df[[contract_id_col, target_col]]
        .drop_duplicates(subset=[contract_id_col])
        .copy()
    )

    pos_contracts = contract_labels.loc[
        contract_labels[target_col] == 1, contract_id_col
    ].to_numpy()

    neg_contracts = contract_labels.loc[
        contract_labels[target_col] == 0, contract_id_col
    ].to_numpy()

    if len(pos_contracts) < n_support_pos:
        raise ValueError(
            f"Department '{department_name}' has only {len(pos_contracts)} positive contracts, "
            f"but {n_support_pos} are required."
        )

    if len(neg_contracts) < n_support_neg:
        raise ValueError(
            f"Department '{department_name}' has only {len(neg_contracts)} negative contracts, "
            f"but {n_support_neg} are required."
        )

    support_pos_contracts = rng.choice(pos_contracts, size=n_support_pos, replace=False)
    support_neg_contracts = rng.choice(neg_contracts, size=n_support_neg, replace=False)

    support_contracts = set(np.concatenate([support_pos_contracts, support_neg_contracts]))

    support_df = dept_df.loc[
        dept_df[contract_id_col].isin(support_contracts)
    ].copy()

    query_df = dept_df.loc[
        ~dept_df[contract_id_col].isin(support_contracts)
    ].copy()

    if len(query_df) == 0:
        raise ValueError(f"Department '{department_name}' produced an empty query set.")

    if require_both_query_classes:
        query_contract_labels = (
            query_df[[contract_id_col, target_col]]
            .drop_duplicates(subset=[contract_id_col])
        )
        n_query_pos = int((query_contract_labels[target_col] == 1).sum())
        n_query_neg = int((query_contract_labels[target_col] == 0).sum())
        if n_query_pos < 1 or n_query_neg < 1:
            raise ValueError(
                f"Department '{department_name}' produced query set without both classes "
                f"(n_query_pos={n_query_pos}, n_query_neg={n_query_neg})."
            )

    return _make_episode_dict(
        department=department_name,
        support_df=support_df,
        query_df=query_df,
        feature_cols=feature_cols,
        target_col=target_col,
        contract_id_col=contract_id_col,
    )


def sample_episode_with_synthetic_support(
    dept_df: pd.DataFrame,
    feature_cols: Sequence[str],
    department_name: Optional[str] = None,
    target_col: str = "gold_y",
    department_col: str = "department",
    contract_id_col: str = "contract_id",
    n_support_pos: int = 2,
    n_support_neg: int = 2,
    query_strategy: str = "remainder",
    # ---------- synthetic augmentation parameters ----------
    augmentation_method: str = "none",
    synthetic_proportion: Optional[float] = None,
    target_per_class: Optional[int] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    k_neighbors: int = 5,
    augmentation_seed_offset: int = 10_007,
    # -------------------------------------------------------
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sample a contract-aware episode and (optionally) augment its support
    set with synthetic rows generated only from the real support.

    Pipeline guarantees:
        1. Support and query are split at the CONTRACT level — same
           `contract_id` never appears on both sides.
        2. The synthetic generator is fit ONLY on the real support of
           the current episode. It never sees the query, never sees
           other departments, never sees any out-of-episode data.
        3. Synthetic rows always have `is_synthetic=True` and a
           `SYNTH_`-prefixed contract_id.
        4. A hard assertion checks that no synthetic contract_id leaks
           into the query before returning.

    Parameters
    ----------
    dept_df, feature_cols, department_name, target_col, department_col,
    contract_id_col, n_support_pos, n_support_neg, query_strategy:
        Forwarded to `sample_support_query_split`.
    augmentation_method:
        One of "none", "gaussian_noise", "mixup", "smote_nc".
        "none" returns the real-only episode (passthrough).
    synthetic_proportion, target_per_class, categorical_cols, k_neighbors:
        Forwarded to `augment_support`. Provide either `synthetic_proportion`
        OR `target_per_class` when `augmentation_method != "none"`.
    augmentation_seed_offset:
        Added to `random_state` before passing to `augment_support`, so that
        for a given `random_state`, the support/query split and the synthetic
        generation are reproducibly tied but use distinct RNG streams.
    random_state:
        Seed for the support/query sampler.

    Returns
    -------
    Dict[str, Any]
        Same shape as `sample_support_query_split` output, with two
        additional entries:
            - is_synthetic_mask : np.ndarray[bool] aligned with X_support
                                   (True for synthetic rows, False for real)
            - augmentation_info : dict with bookkeeping
                                   (method, n_real_support_rows, n_synthetic_support_rows,
                                    n_real_support_contracts, fallback_used)

    Raises
    ------
    ValueError
        If augmentation parameters are inconsistent.
    AssertionError
        If a synthetic contract_id ever appears in the query (leakage).
    """
    # Lazy import to avoid a circular dep if synthetic_augment ever imports from here.
    from .synthetic_augment import augment_support

    # 1. Real-only support/query split (existing behavior).
    episode = sample_support_query_split(
        dept_df=dept_df,
        feature_cols=feature_cols,
        department_name=department_name,
        target_col=target_col,
        department_col=department_col,
        contract_id_col=contract_id_col,
        n_support_pos=n_support_pos,
        n_support_neg=n_support_neg,
        query_strategy=query_strategy,
        random_state=random_state,
    )

    real_support_df = episode["support_df"]
    query_df = episode["query_df"]
    n_real_support_rows = len(real_support_df)
    n_real_support_contracts = (
        real_support_df[contract_id_col].nunique()
        if contract_id_col in real_support_df.columns else 0
    )

    # 2. Passthrough when no augmentation is requested.
    if augmentation_method == "none" or (
        synthetic_proportion is None and target_per_class is None
    ):
        is_synthetic_mask = np.zeros(n_real_support_rows, dtype=bool)
        episode["is_synthetic_mask"] = is_synthetic_mask
        episode["augmentation_info"] = {
            "method": "none",
            "n_real_support_rows": n_real_support_rows,
            "n_synthetic_support_rows": 0,
            "n_real_support_contracts": int(n_real_support_contracts),
            "fallback_used": False,
        }
        return episode

    # 3. Augment the support side. Generator sees only real_support_df.
    aug_seed = (
        None if random_state is None else int(random_state) + int(augmentation_seed_offset)
    )
    augmented_support_df = augment_support(
        support_df=real_support_df,
        feature_cols=feature_cols,
        target_col=target_col,
        method=augmentation_method,
        synthetic_proportion=synthetic_proportion,
        target_per_class=target_per_class,
        categorical_cols=categorical_cols,
        contract_id_col=contract_id_col,
        department_col=department_col,
        k_neighbors=k_neighbors,
        random_state=aug_seed,
    )

    # 4. Leakage assertion: no synthetic contract_id may appear in the query.
    if contract_id_col in query_df.columns and "is_synthetic" in augmented_support_df.columns:
        synth_ids = set(
            augmented_support_df.loc[
                augmented_support_df["is_synthetic"], contract_id_col
            ].astype(str).tolist()
        )
        query_ids = set(query_df[contract_id_col].astype(str).tolist())
        leaked = synth_ids & query_ids
        assert not leaked, (
            f"Synthetic contract_id leakage into query set: {leaked}. "
            "This is a critical bug — synthetic rows must never be visible "
            "in the query side of an episode."
        )

    # 5. Re-extract X / y / ids for the augmented support.
    X_support_aug, y_support_aug = _extract_xy(
        augmented_support_df, feature_cols, target_col,
    )

    if "is_synthetic" in augmented_support_df.columns:
        is_synthetic_mask = augmented_support_df["is_synthetic"].to_numpy(dtype=bool)
    else:
        is_synthetic_mask = np.zeros(len(augmented_support_df), dtype=bool)

    fallback_used = bool(
        augmented_support_df.get(
            "augmentation_fallback", pd.Series([False] * len(augmented_support_df))
        ).any()
    )

    n_synth = int(is_synthetic_mask.sum())

    # 6. Patch the episode dict — preserve query side untouched.
    episode["support_df"] = augmented_support_df.copy()
    episode["support_ids"] = (
        augmented_support_df[contract_id_col].tolist()
        if contract_id_col in augmented_support_df.columns else []
    )
    episode["X_support"] = X_support_aug
    episode["y_support"] = y_support_aug
    episode["is_synthetic_mask"] = is_synthetic_mask
    episode["augmentation_info"] = {
        "method": augmentation_method,
        "n_real_support_rows": n_real_support_rows,
        "n_synthetic_support_rows": n_synth,
        "n_real_support_contracts": int(n_real_support_contracts),
        "fallback_used": fallback_used,
    }

    return episode


def sample_meta_batch(
    task_df: pd.DataFrame,
    feature_cols: Sequence[str],
    valid_departments: Sequence[str],
    department_col: str = "department",
    target_col: str = "gold_y",
    contract_id_col: str = "contract_id",
    meta_batch_size: int = 4,
    n_support_pos: int = 2,
    n_support_neg: int = 2,
    query_strategy: str = "remainder",
    random_state: Optional[int] = None,
    require_both_query_classes: bool = False,
) -> List[Dict[str, Any]]:
    """
    Sample a meta-batch of contract-aware department episodes.

    Behavior:
    - sample departments uniformly from valid_departments
    - sample one episode per department
    - return a list of episode dictionaries

    Parameters
    ----------
    require_both_query_classes:
        Forwarded to sample_support_query_split. If True, episodes whose query
        set lacks either class are rejected with a ValueError.
    """
    _validate_required_columns(
        task_df,
        [department_col, target_col, contract_id_col] + list(feature_cols),
        "task_df",
    )

    if len(valid_departments) == 0:
        raise ValueError("valid_departments is empty.")

    rng = _get_rng(random_state)

    # Sample with replacement only when the requested meta-batch size exceeds
    # the number of valid departments; otherwise sample without replacement so
    # the same department cannot appear twice in a single meta-batch.
    replace = meta_batch_size > len(valid_departments)
    chosen_departments = rng.choice(
        np.array(valid_departments, dtype=object),
        size=meta_batch_size,
        replace=replace,
    )

    batch: List[Dict[str, Any]] = []

    for dept in chosen_departments:
        dept_df = task_df.loc[
            task_df[department_col].astype(str) == str(dept)
        ].copy()

        episode_seed = int(rng.integers(0, 1_000_000_000))

        episode = sample_support_query_split(
            dept_df=dept_df,
            feature_cols=feature_cols,
            department_name=str(dept),
            target_col=target_col,
            department_col=department_col,
            contract_id_col=contract_id_col,
            n_support_pos=n_support_pos,
            n_support_neg=n_support_neg,
            query_strategy=query_strategy,
            random_state=episode_seed,
            require_both_query_classes=require_both_query_classes,
        )
        batch.append(episode)

    return batch


def make_target_meta_test_split(
    task_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_department: str = "Logistics",
    department_col: str = "department",
    target_col: str = "gold_y",
    contract_id_col: str = "contract_id",
    n_support_pos: int = 5,
    n_support_neg: int = 5,
    n_repeats: int = 20,
    query_strategy: str = "remainder",
    base_random_state: int = 42,
    require_both_query_classes: bool = False,
) -> List[Dict[str, Any]]:
    """Construct repeated held-out contract-aware meta-test episodes for any target department.

    Filters ``task_df`` to rows matching ``target_department``, then produces
    ``n_repeats`` independent support/query splits (each seeded deterministically
    from ``base_random_state + repeat_idx``).

    Parameters
    ----------
    task_df : pd.DataFrame
        Full gold-labeled task table (all departments).
    feature_cols : Sequence[str]
        Feature columns to include in the episode arrays.
    target_department : str
        The department to hold out as the meta-test target.  Any string that
        matches a value in the ``department_col`` column is valid — this function
        is not restricted to "Logistics".
    department_col : str
        Column in ``task_df`` that identifies the department.
    target_col : str
        Column containing the binary gold label (0/1).
    contract_id_col : str
        Column containing the contract identifier used for contract-level
        support/query splitting.
    n_support_pos : int
        Number of positive-class contracts to draw into support per repeat.
    n_support_neg : int
        Number of negative-class contracts to draw into support per repeat.
    n_repeats : int
        Number of independent support/query splits to generate.
    query_strategy : str
        Strategy for constructing the query set (passed to
        ``sample_support_query_split``).
    base_random_state : int
        Base seed; repeat ``i`` uses ``base_random_state + i``.
    require_both_query_classes : bool
        If True, only accept episodes whose query set contains both classes.

    Returns
    -------
    List[Dict[str, Any]]
        List of episode dicts, each containing ``support_df``, ``query_df``,
        ``X_support``, ``X_query``, ``y_support``, ``y_query``, ``repeat_idx``,
        and ``episode_seed``.

    Raises
    ------
    ValueError
        If ``task_df`` contains no rows for ``target_department``, or if any
        required column is missing.
    """
    _validate_required_columns(
        task_df,
        [department_col, target_col, contract_id_col] + list(feature_cols),
        "task_df",
    )

    target_df = task_df.loc[
        task_df[department_col].astype(str) == str(target_department)
    ].copy()

    if len(target_df) == 0:
        raise ValueError(f"No rows found for target_department='{target_department}'.")

    episodes: List[Dict[str, Any]] = []

    for repeat_idx in range(n_repeats):
        seed = base_random_state + repeat_idx

        episode = sample_support_query_split(
            dept_df=target_df,
            feature_cols=feature_cols,
            department_name=str(target_department),
            target_col=target_col,
            department_col=department_col,
            contract_id_col=contract_id_col,
            n_support_pos=n_support_pos,
            n_support_neg=n_support_neg,
            query_strategy=query_strategy,
            random_state=seed,
            require_both_query_classes=require_both_query_classes,
        )
        episode["repeat_idx"] = repeat_idx
        episode["episode_seed"] = seed
        episodes.append(episode)

    return episodes


def make_logistics_meta_test_split(*args, **kwargs) -> List[Dict[str, Any]]:
    """Deprecated alias for :func:`make_target_meta_test_split`.

    .. deprecated::
        Use ``make_target_meta_test_split()`` instead.  This alias exists only
        for backwards compatibility with notebooks and scripts that reference the
        old name; it will be removed in a future version.
    """
    return make_target_meta_test_split(*args, **kwargs)


# ============================================================================
# Optional convenience utilities
# ============================================================================

def describe_episode(
    episode: Dict[str, Any],
    target_col: str = "gold_y",
    contract_id_col: str = "contract_id",
) -> Dict[str, Any]:
    """
    Return a lightweight summary of one sampled episode.

    Includes both row counts and contract counts.
    """
    support_df = episode["support_df"]
    query_df = episode["query_df"]

    return {
        "department": episode["department"],
        "support_rows": len(support_df),
        "query_rows": len(query_df),
        "support_contracts": support_df[contract_id_col].nunique() if contract_id_col in support_df.columns else None,
        "query_contracts": query_df[contract_id_col].nunique() if contract_id_col in query_df.columns else None,
        "support_pos_rows": int((support_df[target_col] == 1).sum()) if target_col in support_df.columns else None,
        "support_neg_rows": int((support_df[target_col] == 0).sum()) if target_col in support_df.columns else None,
        "query_pos_rows": int((query_df[target_col] == 1).sum()) if target_col in query_df.columns else None,
        "query_neg_rows": int((query_df[target_col] == 0).sum()) if target_col in query_df.columns else None,
    }