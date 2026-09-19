import pytest

from fastapi.testclient import TestClient

import src.api as api


# Create a test client
client = TestClient(api.app)


# Example input containing all 18 required features.
# These are synthetic values for API testing, not real sensor data.
VALID_PAYLOAD = {
    "cycle": 100,
    "setting_1": 0.0,
    "setting_2": 0.0,
    "sensor_2": 1.0,
    "sensor_3": 1.0,
    "sensor_4": 1.0,
    "sensor_6": 1.0,
    "sensor_7": 1.0,
    "sensor_8": 1.0,
    "sensor_9": 1.0,
    "sensor_11": 1.0,
    "sensor_12": 1.0,
    "sensor_13": 1.0,
    "sensor_14": 1.0,
    "sensor_15": 1.0,
    "sensor_17": 1.0,
    "sensor_20": 1.0,
    "sensor_21": 1.0
}


@pytest.fixture
def mock_prediction(monkeypatch):
    """Mock model inference for API tests."""

    monkeypatch.setattr(
        api,
        "predict_rul",
        lambda data: 42.5
    )


# Test 1: Health endpoint
def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Test 2: Successful prediction
def test_predict_success(mock_prediction):

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    assert response.json() == {
        "predicted_rul": 42.5,
        "unit": "cycles"
    }


# Test 3: Missing required feature
def test_missing_feature(mock_prediction):

    payload = VALID_PAYLOAD.copy()

    del payload["sensor_2"]

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# Test 4: Unexpected feature
def test_extra_feature(mock_prediction):

    payload = VALID_PAYLOAD.copy()

    payload["unknown_sensor"] = 10.0

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# Test 5: Invalid cycle
def test_invalid_cycle(mock_prediction):

    payload = VALID_PAYLOAD.copy()

    payload["cycle"] = -5

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


# Test 6: Missing model artifact
def test_model_unavailable(monkeypatch):

    def missing_model(data):
        raise FileNotFoundError("Model unavailable")

    monkeypatch.setattr(
        api,
        "predict_rul",
        missing_model
    )

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 503