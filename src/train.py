from pathlib import Path

import joblib
from lightgbm import LGBMRegressor

from src.load_data import (
    PROJECT_ROOT,
    load_data,
    calculate_rul
)


# Location for saving the trained model
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "rul_model_v1.joblib"


def train_model():
    """Train and save the LightGBM RUL prediction model."""

    # Step 1: Load training data
    print("Loading training dataset...")

    df = load_data("train_FD001.txt")

    # Step 2: Calculate target variable
    df = calculate_rul(df)

    print("Training data shape:", df.shape)

    # Step 3: Identify candidate features
    candidate_features = [
        col for col in df.columns
        if col not in ["engine_id", "rul"]
    ]

    # Step 4: Identify constant features
    constant_features = [
        col for col in candidate_features
        if df[col].nunique() == 1
    ]

    # Step 5: Select final features
    feature_columns = [
        col for col in candidate_features
        if col not in constant_features
    ]

    print("Selected features:", feature_columns)
    print("Number of features:", len(feature_columns))

    # Step 6: Prepare inputs and target
    X_train = df[feature_columns]
    y_train = df["rul"]

    print("Training features shape:", X_train.shape)

    # Step 7: Initialize model
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )

    # Step 8: Train model
    print("Training LightGBM model...")

    model.fit(X_train, y_train)

    print("Model training completed!")

    # Step 9: Save model and feature schema
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_artifact = {
        "model": model,
        "feature_names": feature_columns,
        "model_version": "v1",
        "dataset": "NASA C-MAPSS FD001",
        "postprocessing": "clip predictions to zero"
    }

    joblib.dump(
        model_artifact,
        MODEL_PATH
    )

    print("Model saved successfully!")
    print("Model location:", MODEL_PATH)


if __name__ == "__main__":
    train_model()