
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from churn_mlops.serving.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    HealthResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
CLASSIFICATION_THRESHOLD = float(
    os.getenv("CHURN_CLASSIFICATION_THRESHOLD", "0.5")
)

if not 0 <= CLASSIFICATION_THRESHOLD <= 1:
    raise ValueError(
        "CHURN_CLASSIFICATION_THRESHOLD must be between 0 and 1."
    )

MODEL_BUNDLE: dict[str, Any] | None = None
MODEL_ERROR: str | None = None

try:
    loaded_bundle = joblib.load(MODEL_PATH)

    if not isinstance(loaded_bundle, dict):
        raise ValueError("Model artifact must contain a dictionary.")

    if "pipeline" not in loaded_bundle:
        raise ValueError("Model artifact is missing the pipeline.")

    MODEL_BUNDLE = loaded_bundle
except (OSError, ValueError, KeyError, EOFError) as exc:
    MODEL_ERROR = str(exc)


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Serves predictions using the trained customer churn "
        "classification pipeline."
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse | JSONResponse:
    """Report whether the prediction model is available."""
    if MODEL_BUNDLE is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "model_loaded": False,
                "detail": MODEL_ERROR or "Model is unavailable.",
            },
        )

    return HealthResponse(status="ok", model_loaded=True)


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "service": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(
    customer: ChurnPredictionRequest,
) -> ChurnPredictionResponse:
    """Predict whether a customer is likely to churn."""
    if MODEL_BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        )

    pipeline = MODEL_BUNDLE["pipeline"]
    expected_columns = list(pipeline.feature_names_in_)
    customer_data = customer.model_dump()

    if set(customer_data) != set(expected_columns):
        raise HTTPException(
            status_code=500,
            detail="Request schema does not match the trained model.",
        )

    features = pd.DataFrame(
        [customer_data],
        columns=expected_columns,
    )

    try:
        probabilities = pipeline.predict_proba(features)[0]
        churn_probability = float(probabilities[1])
    except (ValueError, KeyError, AttributeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The model could not process the customer data.",
        ) from exc

    prediction = (
        "Yes"
        if churn_probability >= CLASSIFICATION_THRESHOLD
        else "No"
    )

    return ChurnPredictionResponse(
        churn_prediction=prediction,
        churn_probability=churn_probability,
        classification_threshold=CLASSIFICATION_THRESHOLD,
    )
