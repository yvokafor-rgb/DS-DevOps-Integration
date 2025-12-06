"""
Data preparation module: cleaning, preprocessing, z-score calculation, and SMOTE.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    RANDOM_SEED,
    SMOTE_PARAMS,
    Z_SCORE_THRESHOLD,
)
from src.utils import calculate_z_scores, load_data, save_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset: handle missing values, duplicates, etc.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    df_clean = df.copy()

    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")

    # Handle missing values (example: fill numeric with median, categorical with mode)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown", inplace=True)

    logger.info("Data cleaning completed")
    return df_clean


def remove_outliers_zscore(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """
    Remove outliers using z-score method.

    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names to check for outliers

    Returns:
        DataFrame with outliers removed
    """
    logger.info(f"Removing outliers using z-score (threshold: {Z_SCORE_THRESHOLD})")
    df_clean = df.copy()

    for col in numeric_columns:
        if col in df_clean.columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < Z_SCORE_THRESHOLD]

    logger.info(f"Outliers removed. Remaining rows: {len(df_clean)}")
    return df_clean


def apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance the dataset.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Balanced X and y
    """
    logger.info("Applying SMOTE...")
    smote = SMOTE(**SMOTE_PARAMS)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logger.info(f"SMOTE applied. New shape: {X_resampled.shape}")
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


def preprocess_data(
    input_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    apply_smote_flag: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Main preprocessing pipeline: clean, remove outliers, apply SMOTE.

    Args:
        input_file: Path to raw data file (if None, looks in data/raw)
        output_file: Path to save processed data (if None, saves to data/processed)
        apply_smote_flag: Whether to apply SMOTE

    Returns:
        Tuple of (X, y) - features and target
    """
    # Load data
    if input_file is None:
        # Find first CSV in raw directory
        raw_files = list(DATA_RAW.glob("*.csv"))
        if not raw_files:
            raise FileNotFoundError(f"No CSV files found in {DATA_RAW}")
        input_file = raw_files[0]

    df = load_data(input_file)
    logger.info(f"Loaded data: {df.shape}")

    # Clean data
    df_clean = clean_data(df)

    # Assume last column is target (adjust as needed)
    target_col = df_clean.columns[-1]
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])

    # Remove outliers from numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        X = remove_outliers_zscore(X, numeric_cols)

    # Re-align y with X after outlier removal
    y = y.loc[X.index]

    # Apply SMOTE if requested
    if apply_smote_flag:
        X, y = apply_smote(X, y)

    # Save processed data
    if output_file is None:
        output_file = DATA_PROCESSED / "processed_data.csv"

    processed_df = X.copy()
    processed_df[target_col] = y
    save_data(processed_df, output_file)

    logger.info("Preprocessing completed")
    return X, y

