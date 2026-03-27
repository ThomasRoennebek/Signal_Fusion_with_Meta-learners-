"""
data_utils.py — Data loading and preprocessing utilities.
"""
import pandas as pd
from pathlib import Path
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


def save_interim(df: pd.DataFrame, filename: str, **kwargs) -> None:
    """Save a DataFrame to the interim data folder."""
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_INTERIM / filename, index=False, **kwargs)


def save_processed(df: pd.DataFrame, filename: str, **kwargs) -> None:
    """Save a DataFrame to the processed data folder."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / filename, index=False, **kwargs)
