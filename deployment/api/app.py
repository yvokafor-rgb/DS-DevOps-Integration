"""
FastAPI application for fraud detection model inference.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="API for fraud detection using hybrid ensemble model",
    version="1.0.0",
)

# Load model (update path as needed)
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "hybrid" / "hybrid_model.pkl"
model = None


class PredictionRequest(BaseModel):
    """Request model for prediction input."""

    features: List[float] = Field(..., description="Feature vector for prediction")


class PredictionResponse(BaseModel):
    """Response model for prediction output."""

    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    model_version: str = Field(default="1.0.0", description="Model version")


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global model
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Fraud Detection API", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a fraud detection prediction.

    Args:
        request: Prediction request with features

    Returns:
        Prediction response with class and probability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert features to numpy array
        features_array = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        probability = float(max(probabilities))

        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            model_version="1.0.0",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

