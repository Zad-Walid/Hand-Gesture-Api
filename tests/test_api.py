
from fastapi.testclient import TestClient
from api.main import app
import joblib
import os

client = TestClient(app)

def test_model_loading():
    assert os.path.exists("api/output/model.pkl"), "Model not found"
    model = joblib.load("api/output/model.pkl")
    assert model is not None, "Failed to load model"

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hand Gesture Model API is running."}

def test_health_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    example_landmarks = {
        "landmarks": [
            0.0, 0.0, 0.0,
            0.1, -0.1, 0.05,
            0.2, -0.2, 0.04,
            0.3, -0.1, 0.03,
            0.4, 0.0, 0.02,
            0.5, 0.1, 0.01,
            0.6, 0.2, 0.0,
            0.7, 0.3, -0.01,
            0.6, 0.4, -0.02,
            0.5, 0.5, -0.03,
            0.4, 0.6, -0.04,
            0.3, 0.7, -0.05,
            0.2, 0.6, -0.06,
            0.1, 0.5, -0.07,
            0.0, 0.4, -0.08,
            -0.1, 0.3, -0.09,
            -0.2, 0.2, -0.1,
            -0.3, 0.1, -0.11,
            -0.4, 0.0, -0.12,
            -0.5, -0.1, -0.13,
            -0.6, -0.2, -0.14
        ]
    }

    response = client.post("/predict", json=example_landmarks)    

    assert response.status_code == 200
    data = response.json()
    assert "gesture" in data
    assert "action" in data
    assert isinstance(data["gesture"], str)
    assert isinstance(data["action"], str)