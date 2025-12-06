"""
Unit tests for data preparation functions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_prep import clean_data, remove_outliers_zscore, apply_smote


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    data = {
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.randint(0, 2, 100),
    }
    return pd.DataFrame(data)


def test_clean_data(sample_dataframe):
    """Test data cleaning function."""
    # Add some missing values
    df = sample_dataframe.copy()
    df.loc[0, "feature1"] = np.nan
    
    cleaned = clean_data(df)
    assert cleaned is not None
    assert len(cleaned) <= len(df)


def test_remove_outliers_zscore(sample_dataframe):
    """Test outlier removal function."""
    df = sample_dataframe.copy()
    result = remove_outliers_zscore(df, ["feature1", "feature2"])
    assert result is not None
    assert len(result) <= len(df)


def test_apply_smote(sample_dataframe):
    """Test SMOTE application."""
    X = sample_dataframe[["feature1", "feature2"]]
    y = sample_dataframe["target"]
    
    X_resampled, y_resampled = apply_smote(X, y)
    assert X_resampled is not None
    assert y_resampled is not None
    assert len(X_resampled) >= len(X)

