"""
Shared evaluation utilities for model assessment.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> dict[str, float]:
    """
    Evaluate a classification model and print metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
        model_name: Name of the model for logging

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Results for {model_name}")
    logger.info(f"{'='*50}")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            # Handle binary and multiclass
            if y_pred_proba.ndim == 1:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
            else:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0], multi_class="ovr")
            metrics["roc_auc"] = roc_auc
            logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")

    # Log metrics
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Classification report
    logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")

    logger.info(f"{'='*50}\n")

    return metrics

