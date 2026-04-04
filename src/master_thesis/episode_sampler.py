"""
Utilities for Stage 2 episodic task construction.

This module is responsible for:
- building the department-level gold-label task table
- summarizing label availability per department
- filtering valid departments for few-shot meta-learning
- sampling support/query splits
- sampling meta-batches of department tasks
- constructing held-out Logistics meta-test episodes


	•	ANIL
	•	FOMAML
	•	fine-tuning baseline
	•	adaptation curves
	•	ablations

Core functions:
build_department_task_table(...)
summarize_department_tasks(...)
filter_valid_departments(...)
sample_support_query_split(...)
sample_meta_batch(...)
make_logistics_meta_test_split(...)


"""

# 1. Imports
# 2. Data containers / typing helpers
# 3. Summary + validation helpers
# 4. Task table construction
# 5. Support/query sampling
# 6. Meta-batch sampling
# 7. Logistics meta-test construction

# -----------
# 1. Imports
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 2. Data containers / typing helpers
# ============================================================================
# Private helpers
# ============================================================================


def _validate_required_columns(df: pd.DataFrame, required_cols: Sequence[str], df_name: str) -> None:
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

    X = df.loc[:, feature_cols].to_numpy(dtype=float)
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
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n > len(df):
        raise ValueError(f"Cannot sample {n} rows from dataframe of size {len(df)}.")

    if n == 0:
        return df.iloc[0:0].copy()

    sampled_idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[sampled_idx].copy()


# 4. Task table construction
# ============================================================================
# Public API: task table construction and summaries
# ============================================================================


def build_department_task_table(
    features_df: pd.DataFrame, # gives the predictor
    gold_labels_df: pd.DataFrame, # gives the target
    feature_cols: Sequence[str],
    contract_id_col: str = "contract_id",
    department_col: str = "department",
    target_col: str = "gold_label",
    gold_target_source_col: Optional[str] = None,
    drop_missing_features: bool = False,
) -> pd.DataFrame:
    """
    Build the master Stage 2 task table.

    Expected behavior:
    - merge final feature table with manual gold labels
    - keep one row per contract
    - keep only gold-labeled rows
    - standardize target to 0/1
    - retain department + id + selected feature columns

    Parameters
    ----------
    features_df:
        Dataframe containing final modeling features.
    gold_labels_df:
        Dataframe containing manual gold labels.
    feature_cols:
        Feature columns to retain for episodic learning.
    contract_id_col:
        Unique contract identifier column.
    department_col:
        Department/task column.
    target_col:
        Name of the target column in the returned dataframe.
    gold_target_source_col:
        Column in gold_labels_df that contains the original gold label.
        If None, this function assumes target_col already exists in gold_labels_df.
    drop_missing_features:
        If True, drop rows with missing values in feature columns.
        If False, keep them for later preprocessing/imputation.

    Returns
    -------
    pd.DataFrame
        Clean Stage 2 task table.
    """
    raise NotImplementedError("Implement once you share the exact column names and label conventions.")


def summarize_department_tasks(
    task_df: pd.DataFrame,
    department_col: str = "department",
    target_col: str = "gold_label",
) -> pd.DataFrame:
    """
    Summarize task availability per department.

    Suggested output columns:
    - department
    - n_total
    - n_pos
    - n_neg
    - pos_rate

    Returns
    -------
    pd.DataFrame
        Department-level summary table.
    """
    raise NotImplementedError("Implement once we confirm the exact target encoding and desired summary columns.")


def filter_valid_departments(
    task_df: pd.DataFrame,
    department_col: str = "department",
    target_col: str = "gold_label",
    n_support_pos: int = 2,
    n_support_neg: int = 2,
    min_query_size: int = 1,
    require_both_query_classes: bool = False,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Filter departments that can support the requested few-shot episode setup.

    Validity should typically require:
    - enough positive examples for support
    - enough negative examples for support
    - enough remaining rows to create a non-empty query set

    Parameters
    ----------
    task_df:
        Stage 2 task table.
    department_col:
        Department/task column.
    target_col:
        Binary target column.
    n_support_pos:
        Number of positive support examples required.
    n_support_neg:
        Number of negative support examples required.
    min_query_size:
        Minimum number of query examples after removing support rows.
    require_both_query_classes:
        If True, query set must contain at least one positive and one negative example.

    Returns
    -------
    filtered_task_df:
        Only rows from valid departments.
    valid_departments:
        List of valid department names.
    validity_summary_df:
        Department-level summary with validity flags/reasons.
    """
    raise NotImplementedError("Implement after we decide the exact episode eligibility rules.")


# ============================================================================
# Public API: support/query and episode sampling
# ============================================================================

# 5. Support/query sampling
def sample_support_query_split(
    dept_df: pd.DataFrame,
    feature_cols: Sequence[str],
    department_name: Optional[str] = None,
    target_col: str = "gold_label",
    department_col: str = "department",
    contract_id_col: str = "contract_id",
    n_support_pos: int = 2,
    n_support_neg: int = 2,
    query_strategy: str = "remainder",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sample one support/query episode for a single department.

    Recommended default:
    - support = balanced class sample
    - query = remainder

    Parameters
    ----------
    dept_df:
        Dataframe containing rows from one department only.
    feature_cols:
        Feature columns used by the learner.
    department_name:
        Optional override for the department name in the returned episode.
    target_col:
        Binary target column.
    department_col:
        Department column.
    contract_id_col:
        Contract ID column.
    n_support_pos:
        Number of positive examples in support.
    n_support_neg:
        Number of negative examples in support.
    query_strategy:
        Strategy for query construction. Start with "remainder".
    random_state:
        Seed for reproducible sampling.

    Returns
    -------
    Dict[str, Any]
        Episode dictionary containing raw dataframes, ids, and X/y arrays.
    """
    raise NotImplementedError("Implement after we confirm support/query policy and target encoding.")


# 6. Meta-batch sampling
def sample_meta_batch(
    task_df: pd.DataFrame,
    feature_cols: Sequence[str],
    valid_departments: Sequence[str],
    department_col: str = "department",
    target_col: str = "gold_label",
    contract_id_col: str = "contract_id",
    meta_batch_size: int = 4,
    n_support_pos: int = 2,
    n_support_neg: int = 2,
    query_strategy: str = "remainder",
    random_state: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample a meta-batch of department episodes for one outer-loop iteration.

    Recommended behavior:
    - sample departments uniformly from valid_departments
    - sample one episode per department
    - return a list of episode dictionaries

    Parameters
    ----------
    task_df:
        Stage 2 task table containing valid departments.
    feature_cols:
        Feature columns used by the learner.
    valid_departments:
        Departments eligible for episodic sampling.
    department_col:
        Department/task column.
    target_col:
        Binary target column.
    contract_id_col:
        Contract ID column.
    meta_batch_size:
        Number of department tasks in one batch.
    n_support_pos:
        Number of positive support examples per task.
    n_support_neg:
        Number of negative support examples per task.
    query_strategy:
        Query construction strategy.
    random_state:
        Seed for reproducible batch sampling.

    Returns
    -------
    List[Dict[str, Any]]
        List of episode dictionaries.
    """
    raise NotImplementedError("Implement after we settle valid-department sampling and batch behavior.")

# 7. Logistics meta-test construction
def make_logistics_meta_test_split(
    task_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_department: str = "Logistics",
    department_col: str = "department",
    target_col: str = "gold_label",
    contract_id_col: str = "contract_id",
    n_support_pos: int = 5,
    n_support_neg: int = 5,
    n_repeats: int = 20,
    query_strategy: str = "remainder",
    base_random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Construct repeated held-out meta-test episodes for the target department.

    Recommended use:
    - keep target_department held out during meta-training
    - create repeated support/query splits for stable evaluation

    Parameters
    ----------
    task_df:
        Stage 2 task table.
    feature_cols:
        Feature columns used by the learner.
    target_department:
        Held-out department used for meta-testing.
    department_col:
        Department/task column.
    target_col:
        Binary target column.
    contract_id_col:
        Contract ID column.
    n_support_pos:
        Number of positive support examples.
    n_support_neg:
        Number of negative support examples.
    n_repeats:
        Number of repeated target episodes.
    query_strategy:
        Query construction strategy.
    base_random_state:
        Base seed used to create reproducible repeated splits.

    Returns
    -------
    List[Dict[str, Any]]
        Repeated target-department episodes.
    """
    raise NotImplementedError("Implement after we confirm the held-out Logistics protocol.")


# ============================================================================
# Optional convenience utilities
# ============================================================================


def describe_episode(
    episode: Dict[str, Any],
    target_col: str = "gold_label",
) -> Dict[str, Any]:
    """
    Return a lightweight summary of one sampled episode.

    Useful for notebook inspection and debugging.
    """
    support_df = episode["support_df"]
    query_df = episode["query_df"]

    return {
        "department": episode["department"],
        "support_size": len(support_df),
        "query_size": len(query_df),
        "support_pos": int((support_df[target_col] == 1).sum()) if target_col in support_df.columns else None,
        "support_neg": int((support_df[target_col] == 0).sum()) if target_col in support_df.columns else None,
        "query_pos": int((query_df[target_col] == 1).sum()) if target_col in query_df.columns else None,
        "query_neg": int((query_df[target_col] == 0).sum()) if target_col in query_df.columns else None,
    }