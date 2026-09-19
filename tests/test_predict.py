import pytest

import src.predict as prediction_module


# A fake model used only for testing
class FakeModel:

    def __init__(self, output=42.0):
        self.output = output
        self.received_columns = None

    def predict(self, X):
        self.received_columns = X.columns.tolist()
        return [self.output]


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the real saved model with a fake model."""

    model = FakeModel()

    artifact = {
        "model": model,
        "feature_names": ["cycle", "sensor_2"],
        "model_version": "test"
    }

    monkeypatch.setattr(
        prediction_module,
        "load_model",
        lambda: artifact
    )

    return model


# Test 1: Valid input
def test_valid_prediction(fake_model):

    data = {
        "cycle": 100,
        "sensor_2": 642.5
    }

    result = prediction_module.predict_rul(data)

    assert result == 42.0
    assert result >= 0


# Test 2: Missing feature
def test_missing_feature(fake_model):

    data = {
        "cycle": 100
    }

    with pytest.raises(ValueError, match="Missing features"):
        prediction_module.predict_rul(data)


# Test 3: Unexpected feature
def test_extra_feature(fake_model):

    data = {
        "cycle": 100,
        "sensor_2": 642.5,
        "unknown_sensor": 10
    }

    with pytest.raises(ValueError, match="Unexpected features"):
        prediction_module.predict_rul(data)


# Test 4: Invalid operating cycles
@pytest.mark.parametrize(
    "invalid_cycle",
    [0, -1, 1.5]
)
def test_invalid_cycle(fake_model, invalid_cycle):

    data = {
        "cycle": invalid_cycle,
        "sensor_2": 642.5
    }

    with pytest.raises(ValueError):
        prediction_module.predict_rul(data)


# Test 5: Non-finite numerical values
@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf")]
)
def test_non_finite_input(fake_model, invalid_value):

    data = {
        "cycle": 100,
        "sensor_2": invalid_value
    }

    with pytest.raises(ValueError, match="finite"):
        prediction_module.predict_rul(data)


# Test 6: Negative prediction clipping
def test_negative_prediction(fake_model):

    fake_model.output = -25.0

    data = {
        "cycle": 100,
        "sensor_2": 642.5
    }

    result = prediction_module.predict_rul(data)

    assert result == 0.0


# Test 7: Feature order
def test_feature_order(fake_model):

    # Intentionally provide features in reverse order
    data = {
        "sensor_2": 642.5,
        "cycle": 100
    }

    prediction_module.predict_rul(data)

    assert fake_model.received_columns == [
        "cycle",
        "sensor_2"
    ]