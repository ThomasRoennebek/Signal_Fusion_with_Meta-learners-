"""
data_utils.py — Data loading, saving, and dataset preparation utilities.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from master_thesis.config import DATA_RAW, DATA_INTERIM, DATA_PROCESSED


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


def require_columns(df: pd.DataFrame, required_cols: list[str], df_name: str = "DataFrame") -> None:
    """Raise an error if required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def merge_gold_labels(
    df_weak: pd.DataFrame,
    df_gold: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge gold labels onto the weak-labeled Stage 1 dataframe.
    Expects at least contract_id and gold_y in df_gold.
    """
    require_columns(df_weak, ["contract_id"], df_name="df_weak")
    require_columns(df_gold, ["contract_id", "gold_y"], df_name="df_gold")

    gold_keep_cols = [c for c in ["contract_id", "gold_y", "department"] if c in df_gold.columns]
    df_gold_small = df_gold[gold_keep_cols].drop_duplicates(subset=["contract_id"]).copy()

    df_stage1 = df_weak.merge(
        df_gold_small,
        on="contract_id",
        how="left",
        suffixes=("", "_gold")
    )
    return df_stage1


def make_gold_contract_split(
    df_stage1: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.30,
) -> tuple[set, set]:
    """
    Split gold-labeled contracts into train and test sets using contract_id groups.
    Returns:
        gold_train_contract_ids, gold_test_contract_ids
    """
    require_columns(df_stage1, ["contract_id", "gold_y"], df_name="df_stage1")

    gold_contracts = (
        df_stage1.loc[df_stage1["gold_y"].notna(), ["contract_id", "gold_y"]]
        .drop_duplicates(subset=["contract_id"])
        .copy()
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(
        gss.split(
            gold_contracts[["contract_id"]],
            gold_contracts["gold_y"],
            groups=gold_contracts["contract_id"]
        )
    )

    gold_train_contract_ids = set(gold_contracts.iloc[train_idx]["contract_id"])
    gold_test_contract_ids = set(gold_contracts.iloc[test_idx]["contract_id"])

    return gold_train_contract_ids, gold_test_contract_ids
