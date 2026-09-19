from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.train import MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.train first."
        )

    # Only load model artifacts from trusted sources.
    artifact = joblib.load(MODEL_PATH)

    return artifact


def predict_rul(sensor_data: dict) -> float:
    """Predict remaining useful life from engine measurements."""

    # Step 1: Load model and feature names
    artifact = load_model()

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    # Step 2: Validate input features
    missing = set(feature_names) - set(sensor_data)
    extra = set(sensor_data) - set(feature_names)

    if missing or extra:
        raise ValueError(
            f"Missing features: {sorted(missing)}. "
            f"Unexpected features: {sorted(extra)}."
        )

    # Step 3: Create DataFrame in training feature order
    X = pd.DataFrame(
        [[sensor_data[name] for name in feature_names]],
        columns=feature_names
    )

    # Ensure inputs are numeric and finite
    X = X.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("All features must be finite numbers.")

    if X["cycle"].iloc[0] < 1 or not float(
        X["cycle"].iloc[0]
    ).is_integer():
        raise ValueError("Cycle must be a positive integer.")

    # Step 4: Predict
    prediction = model.predict(X)[0]

    # Step 5: Clip negative predictions
    prediction = max(float(prediction), 0.0)

    return prediction


if __name__ == "__main__":

    from src.load_data import load_data

    # Use a training observation for a basic smoke test
    df = load_data("train_FD001.txt")

    artifact = load_model()

    sample = df.iloc[0][
        artifact["feature_names"]
    ].to_dict()

    result = predict_rul(sample)

    print("Model version:", artifact["model_version"])
    print("Predicted RUL:", round(result, 2), "cycles")