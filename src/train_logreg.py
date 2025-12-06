"""
Train Logistic Regression model (Model 1).
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import LOGREG_PARAMS, MODEL_LOGREG, RANDOM_SEED
from src.data_prep import preprocess_data
from src.evaluate import evaluate_model
from src.utils import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    save_model: bool = True,
    model_path: Path = None,
) -> LogisticRegression:
    """
    Train a Logistic Regression model.

    Args:
        X: Feature DataFrame
        y: Target Series
        save_model: Whether to save the trained model
        model_path: Path to save the model (if None, uses default)

    Returns:
        Trained LogisticRegression model
    """
    logger.info("Training Logistic Regression model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Train model
    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, model_name="Logistic Regression")

    # Save model
    if save_model:
        if model_path is None:
            model_path = MODEL_LOGREG / "logreg_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

    return model


def main():
    """Main function to run training pipeline."""
    logger.info("Starting Logistic Regression training pipeline...")

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
    model = train_logistic_regression(X, y)
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main()

