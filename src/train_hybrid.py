"""
Train Hybrid/Ensemble model combining Logistic Regression and Random Forest.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import LOGREG_PARAMS, MODEL_HYBRID, RANDOM_SEED, RF_PARAMS
from src.data_prep import preprocess_data
from src.evaluate import evaluate_model
from src.utils import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import here to avoid circular dependency
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None


def train_hybrid_model(
    X: pd.DataFrame,
    y: pd.Series,
    save_model: bool = True,
    model_path: Path = None,
    weights: tuple[float, float] = (0.5, 0.5),
) -> VotingClassifier:
    """
    Train a hybrid ensemble model combining Logistic Regression and Random Forest.

    Args:
        X: Feature DataFrame
        y: Target Series
        save_model: Whether to save the trained model
        model_path: Path to save the model (if None, uses default)
        weights: Weights for ensemble voting (logreg_weight, rf_weight)

    Returns:
        Trained VotingClassifier model
    """
    logger.info("Training Hybrid Ensemble model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Create base models
    logreg = LogisticRegression(**LOGREG_PARAMS)
    rf = RandomForestClassifier(**RF_PARAMS)

    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[("logreg", logreg), ("rf", rf)],
        voting="soft",
        weights=list(weights),
    )

    # Train ensemble
    ensemble.fit(X_train, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    evaluate_model(y_test, y_pred, model_name="Hybrid Ensemble")

    # Save model
    if save_model:
        if model_path is None:
            model_path = MODEL_HYBRID / "hybrid_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(ensemble, f)
        logger.info(f"Model saved to {model_path}")

    return ensemble


def main():
    """Main function to run training pipeline."""
    logger.info("Starting Hybrid Ensemble training pipeline...")

    # Load processed data
    processed_file = Path("data/processed/processed_data.csv")
    if processed_file.exists():
        df = load_data(processed_file)
        target_col = df.columns[-1]
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        # Run preprocessing if processed data doesn't exist
        logger.info("Processed data not found. Running preprocessing...")
        X, y = preprocess_data()

    # Train model
    model = train_hybrid_model(X, y)
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main()

