"""
Unit tests for model training and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from train_logreg import train_logistic_regression
from train_rf import train_random_forest
from train_hybrid import train_hybrid_model
from evaluate import evaluate_model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


def test_logistic_regression_training(sample_data):
    """Test Logistic Regression training."""
    X, y = sample_data
    model = train_logistic_regression(X, y, save_model=False)
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_random_forest_training(sample_data):
    """Test Random Forest training."""
    X, y = sample_data
    model = train_random_forest(X, y, save_model=False)
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_hybrid_model_training(sample_data):
    """Test Hybrid Model training."""
    X, y = sample_data
    model = train_hybrid_model(X, y, save_model=False)
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_evaluate_model(sample_data):
    """Test model evaluation function."""
    X, y = sample_data
    model = train_logistic_regression(X, y, save_model=False)
    
    y_pred = model.predict(X[:20])
    metrics = evaluate_model(y[:20], y_pred, model_name="Test Model")
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

