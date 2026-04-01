"""
data_utils.py — Data loading, saving, and dataset preparation utilities.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from master_thesis.config import DATA_RAW, DATA_INTERIM, DATA_PROCESSED, TABLES


# =============================================================================
# Basic load / save helpers
# =============================================================================

def load_raw(filename: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from the raw data folder."""
    return pd.read_csv(DATA_RAW / filename, **kwargs)


def load_interim(filename: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from the interim data folder."""
    return pd.read_csv(DATA_INTERIM / filename, **kwargs)


def load_processed(filename: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from the processed data folder."""
    return pd.read_csv(DATA_PROCESSED / filename, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save a DataFrame to any CSV path, creating folders if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def save_interim(df: pd.DataFrame, filename: str, **kwargs) -> None:
    """Save a DataFrame to the interim data folder."""
    save_csv(df, DATA_INTERIM / filename, **kwargs)


def save_processed(df: pd.DataFrame, filename: str, **kwargs) -> None:
    """Save a DataFrame to the processed data folder."""
    save_csv(df, DATA_PROCESSED / filename, **kwargs)


# =============================================================================
# Validation helpers
# =============================================================================

def require_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    df_name: str = "DataFrame",
) -> None:
    """Raise an error if required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


# =============================================================================
# Gold label helpers
# =============================================================================

def merge_gold_labels(
    df_weak: pd.DataFrame,
    df_gold: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge gold labels onto the weak-labeled Stage 1 dataframe.

    Expects at least:
    - contract_id in df_weak
    - contract_id and gold_y in df_gold
    """
    require_columns(df_weak, ["contract_id"], df_name="df_weak")
    require_columns(df_gold, ["contract_id", "gold_y"], df_name="df_gold")

    gold_keep_cols = [
        c for c in ["contract_id", "gold_y", "department"]
        if c in df_gold.columns
    ]
    df_gold_small = df_gold[gold_keep_cols].drop_duplicates(subset=["contract_id"]).copy()

    df_stage1 = df_weak.merge(
        df_gold_small,
        on="contract_id",
        how="left",
        suffixes=("", "_gold"),
    )
    return df_stage1


def make_gold_contract_split(
    df_stage1: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.30,
) -> tuple[set, set]:
    """
    Split gold-labeled contracts into train and test sets using contract_id groups.

    Returns
    -------
    tuple[set, set]
        gold_train_contract_ids, gold_test_contract_ids
    """
    require_columns(df_stage1, ["contract_id", "gold_y"], df_name="df_stage1")

    df_gold_contracts = (
        df_stage1.loc[df_stage1["gold_y"].notna(), ["contract_id", "gold_y"]]
        .drop_duplicates(subset=["contract_id"])
        .copy()
    )

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    train_idx, test_idx = next(
        gss.split(
            df_gold_contracts[["contract_id"]],
            df_gold_contracts["gold_y"],
            groups=df_gold_contracts["contract_id"],
        )
    )

    gold_train_contract_ids = set(df_gold_contracts.iloc[train_idx]["contract_id"])
    gold_test_contract_ids = set(df_gold_contracts.iloc[test_idx]["contract_id"])

    return gold_train_contract_ids, gold_test_contract_ids


def subset_by_contract_ids(
    df: pd.DataFrame,
    contract_ids: set | list,
    contract_col: str = "contract_id",
) -> pd.DataFrame:
    """
    Subset a dataframe to rows whose contract id is in the provided collection.
    """
    require_columns(df, [contract_col], df_name="df")
    return df[df[contract_col].isin(contract_ids)].copy()


# =============================================================================
# Versioned saving helpers
# =============================================================================

def _next_versioned_path(
    output_dir: Path,
    stem: str,
    suffix: str,
) -> Path:
    """
    Build the next available versioned path.

    Example
    -------
    stem='stage1_results', suffix='.csv'
    -> stage1_results_v001.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(stem)}_v(\d{{3}}){re.escape(suffix)}$")
    versions: list[int] = []

    for path in output_dir.glob(f"{stem}_v*{suffix}"):
        match = pattern.match(path.name)
        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions, default=0) + 1
    return output_dir / f"{stem}_v{next_version:03d}{suffix}"


def save_table_versioned(
    df: pd.DataFrame,
    stem: str,
    output_dir: Path | None = None,
    **kwargs,
) -> Path:
    """
    Save a table with automatic versioning.

    Defaults to TABLES from config.
    """
    if output_dir is None:
        output_dir = TABLES

    output_path = _next_versioned_path(
        output_dir=output_dir,
        stem=stem,
        suffix=".csv",
    )
    df.to_csv(output_path, index=False, **kwargs)
    return output_path


def save_json_versioned(
    obj,
    stem: str,
    output_dir: Path | None = None,
    indent: int = 2,
) -> Path:
    """
    Save a JSON-serializable object with automatic versioning.
    """
    import json

    if output_dir is None:
        output_dir = TABLES

    output_path = _next_versioned_path(
        output_dir=output_dir,
        stem=stem,
        suffix=".json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)

    return output_path


# =============================================================================
# Stage 1 convenience helper
# =============================================================================

def prepare_stage1_gold_splits(
    df_stage1: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.30,
    group_col: str = "contract_id",
    gold_col: str = "gold_y",
) -> dict[str, pd.DataFrame | set]:
    """
    Convenience helper for Stage 1.

    Returns a dictionary containing:
    - gold_train_contract_ids
    - gold_test_contract_ids
    - df_gold
    - df_gold_train
    - df_gold_test
    """
    require_columns(df_stage1, [group_col, gold_col], df_name="df_stage1")

    gold_train_contract_ids, gold_test_contract_ids = make_gold_contract_split(
        df_stage1=df_stage1.rename(columns={group_col: "contract_id", gold_col: "gold_y"}),
        seed=seed,
        test_size=test_size,
    )

    df_gold = df_stage1[df_stage1[gold_col].notna()].copy()
    df_gold_train = df_gold[df_gold[group_col].isin(gold_train_contract_ids)].copy()
    df_gold_test = df_gold[df_gold[group_col].isin(gold_test_contract_ids)].copy()

    return {
        "gold_train_contract_ids": gold_train_contract_ids,
        "gold_test_contract_ids": gold_test_contract_ids,
        "df_gold": df_gold,
        "df_gold_train": df_gold_train,
        "df_gold_test": df_gold_test,
    }