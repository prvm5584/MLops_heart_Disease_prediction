import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease MLOps API",
    version="1.0.0",
    description="Production API for Heart Disease model inference.",
)

MODEL_PATH = Path("models/final_model.joblib")

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests",
)

ERRORS = Counter(
    "prediction_errors_total",
    "Total prediction errors",
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
)


class Patient(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float | None = None
    thal: float | None = None


@app.get("/health")
def health() -> dict:
    """Return API and model health."""

    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(patient: Patient) -> dict:
    """Generate a Heart Disease prediction."""

    REQUESTS.inc()

    start_time = time.perf_counter()

    try:
        if model is None:
            raise RuntimeError("Model is not loaded. Run the training notebook first.")

        patient_df = pd.DataFrame([patient.model_dump()])

        probability = float(model.predict_proba(patient_df)[0, 1])

        prediction = int(probability >= 0.5)

        confidence = max(
            probability,
            1.0 - probability,
        )

        logger.info(
            "Prediction completed: prediction=%s probability=%.4f",
            prediction,
            probability,
        )

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "heart_disease_probability": round(
                probability,
                4,
            ),
        }

    except (ValueError, TypeError, RuntimeError) as exc:
        ERRORS.inc()

        logger.exception("Prediction request failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        LATENCY.observe(time.perf_counter() - start_time)


@app.get("/metrics")
def metrics() -> Response:
    """Return Prometheus-compatible API metrics."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
