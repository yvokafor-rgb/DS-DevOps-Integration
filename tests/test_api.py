"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "deployment" / "api"))

from app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint():
    """Test prediction endpoint."""
    # Sample feature vector (adjust size based on your model)
    features = [0.1] * 10  # Example: 10 features
    
    response = client.post("/predict", json={"features": features})
    
    # May fail if model not loaded, but should return proper error
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert "prediction" in response.json()
        assert "probability" in response.json()

