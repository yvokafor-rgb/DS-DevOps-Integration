"""
Central configuration file for paths, parameters, and seeds.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_LOGREG = MODELS_DIR / "logreg"
MODEL_RF = MODELS_DIR / "random_forest"
MODEL_HYBRID = MODELS_DIR / "hybrid"
MODEL_REGISTRY = MODELS_DIR / "model_registry.json"

# Random seed for reproducibility
RANDOM_SEED = 42

# Model parameters
LOGREG_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_SEED,
}

# SMOTE parameters
SMOTE_PARAMS = {
    "random_state": RANDOM_SEED,
    "k_neighbors": 5,
}

# Z-score threshold for outlier removal
Z_SCORE_THRESHOLD = 3

# Ensure directories exist
for path in [DATA_RAW, DATA_PROCESSED, MODEL_LOGREG, MODEL_RF, MODEL_HYBRID]:
    path.mkdir(parents=True, exist_ok=True)

