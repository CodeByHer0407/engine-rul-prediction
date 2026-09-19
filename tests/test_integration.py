import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from src.api import app
from src.load_data import (
    DATA_DIR,
    load_data,
    get_last_observations
)
from src.predict import load_model
from src.train import MODEL_PATH


client = TestClient(app)


@pytest.mark.skipif(
    not MODEL_PATH.exists()
    or not (DATA_DIR / "test_FD001.txt").exists(),
    reason="Local model artifact or NASA dataset unavailable"
)
def test_real_model_prediction():

    # Step 1: Load an actual NASA test observation
    test_df = load_data("test_FD001.txt")

    test_last = get_last_observations(test_df)

    # Select Engine 1
    engine = test_last.iloc[0]

    # Step 2: Load model and its feature schema
    artifact = load_model()

    features = artifact["feature_names"]

    # Step 3: Prepare API request
    payload = {
        feature: (
            int(engine[feature])
            if feature == "cycle"
            else float(engine[feature])
        )
        for feature in features
    }

    # Step 4: Send the request to FastAPI
    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    # Step 5: Obtain API prediction
    result = response.json()

    assert result["unit"] == "cycles"
    assert result["predicted_rul"] >= 0

    # Step 6: Calculate expected prediction directly
    X = pd.DataFrame(
        [payload],
        columns=features
    )

    expected = max(
        float(artifact["model"].predict(X)[0]),
        0.0
    )

    # Step 7: Compare API prediction with model output
    assert result["predicted_rul"] == pytest.approx(
        expected
    )

    print("\nReal engine prediction:", result)