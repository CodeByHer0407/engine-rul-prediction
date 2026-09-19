from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from src.predict import predict_rul


# Create FastAPI application
app = FastAPI(
    title="Engine RUL Prediction API",
    description="Predict aircraft engine remaining useful life using LightGBM.",
    version="1.0.0"
)


# Define input schema
class EngineInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(ge=1, strict=True)

    setting_1: FiniteFloat
    setting_2: FiniteFloat

    sensor_2: FiniteFloat
    sensor_3: FiniteFloat
    sensor_4: FiniteFloat
    sensor_6: FiniteFloat
    sensor_7: FiniteFloat
    sensor_8: FiniteFloat
    sensor_9: FiniteFloat
    sensor_11: FiniteFloat
    sensor_12: FiniteFloat
    sensor_13: FiniteFloat
    sensor_14: FiniteFloat
    sensor_15: FiniteFloat
    sensor_17: FiniteFloat
    sensor_20: FiniteFloat
    sensor_21: FiniteFloat


# Define output schema
class PredictionResponse(BaseModel):
    predicted_rul: float
    unit: str = "cycles"


# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(engine: EngineInput):

    # Convert validated request into a dictionary
    sensor_data = engine.model_dump()

    try:
        # Use our existing prediction function
        prediction = predict_rul(sensor_data)

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is unavailable."
        )

    return PredictionResponse(
        predicted_rul=prediction
    )