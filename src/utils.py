"""
Helper functions for data processing and model utilities.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a file path.

    Args:
        file_path: Path to the data file

    Returns:
        DataFrame containing the loaded data
    """
    logger.info(f"Loading data from {file_path}")
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_data(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save DataFrame to a file.

    Args:
        df: DataFrame to save
        file_path: Path where to save the data
    """
    logger.info(f"Saving data to {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.suffix == ".csv":
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def calculate_z_scores(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Calculate z-scores for specified columns.

    Args:
        df: Input DataFrame
        columns: List of column names to calculate z-scores for

    Returns:
        DataFrame with z-scores added
    """
    df_z = df.copy()
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df_z[f"{col}_zscore"] = (df[col] - mean) / std
    return df_z

