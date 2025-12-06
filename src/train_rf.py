"""
Train Random Forest model (Model 2).
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import MODEL_RF, RANDOM_SEED, RF_PARAMS
from src.data_prep import preprocess_data
from src.evaluate import evaluate_model
from src.utils import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    save_model: bool = True,
    model_path: Path = None,
) -> RandomForestClassifier:
    """
    Train a Random Forest model.

    Args:
        X: Feature DataFrame
        y: Target Series
        save_model: Whether to save the trained model
        model_path: Path to save the model (if None, uses default)

    Returns:
        Trained RandomForestClassifier model
    """
    logger.info("Training Random Forest model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Train model
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, model_name="Random Forest")

    # Save model
    if save_model:
        if model_path is None:
            model_path = MODEL_RF / "rf_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

    return model


def main():
    """Main function to run training pipeline."""
    logger.info("Starting Random Forest training pipeline...")

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
    model = train_random_forest(X, y)
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main()

