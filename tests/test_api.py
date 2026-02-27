from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "features": [
                0.0, -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36,
                0.09, -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.40,
                0.25, -0.01, 0.27, -0.11, 0.06, -0.14, -0.06, -0.06, 0.12, 149.62
            ]
        }
    )

    assert response.status_code == 200
    assert "fraud_prediction" in response.json()