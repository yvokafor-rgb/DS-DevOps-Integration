"""
Model validation script for automated checks before deployment.
"""

import json
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_REGISTRY = PROJECT_ROOT / "models" / "model_registry.json"
MODEL_HYBRID = PROJECT_ROOT / "models" / "hybrid" / "hybrid_model.pkl"


def validate_model_exists(model_path: Path) -> bool:
    """Check if model file exists."""
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return False
    return True


def validate_model_loadable(model_path: Path) -> bool:
    """Check if model can be loaded."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def validate_model_registry() -> bool:
    """Validate model registry file."""
    if not MODEL_REGISTRY.exists():
        logger.warning(f"Model registry not found at {MODEL_REGISTRY}")
        return False
    
    try:
        with open(MODEL_REGISTRY, "r") as f:
            registry = json.load(f)
        logger.info("Model registry is valid JSON")
        return True
    except Exception as e:
        logger.error(f"Invalid model registry: {e}")
        return False


def main():
    """Run all validation checks."""
    logger.info("Starting model validation...")
    
    checks = [
        ("Model exists", validate_model_exists(MODEL_HYBRID)),
        ("Model loadable", validate_model_loadable(MODEL_HYBRID)),
        ("Model registry valid", validate_model_registry()),
    ]
    
    all_passed = all(check[1] for check in checks)
    
    logger.info("\nValidation Results:")
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
    
    if not all_passed:
        logger.error("Validation failed!")
        exit(1)
    
    logger.info("All validations passed!")


if __name__ == "__main__":
    main()

