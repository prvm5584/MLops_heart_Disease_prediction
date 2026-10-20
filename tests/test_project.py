from pathlib import Path

import joblib
import pandas as pd

METRICS_PATH = Path("artifacts/model_comparison.csv")

MODEL_PATH = Path("models/final_model.joblib")


def test_model_exists():
    """Verify that model training generated the model."""

    assert MODEL_PATH.exists()


def test_metrics_exist():
    """Verify that experiment results were generated."""

    assert METRICS_PATH.exists()


def test_model_loads():
    """Verify that the saved model loads successfully."""

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_pipeline_handles_missing_values():
    """Verify preprocessing handles missing ca and thal."""

    model = joblib.load(MODEL_PATH)

    patient = pd.DataFrame(
        [
            {
                "age": 54,
                "sex": 1,
                "cp": 4,
                "trestbps": 130,
                "chol": 246,
                "fbs": 0,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.2,
                "slope": 2,
                "ca": None,
                "thal": None,
            }
        ]
    )

    prediction = model.predict(patient)

    probability = model.predict_proba(patient)[0, 1]

    assert len(prediction) == 1
    assert prediction[0] in (0, 1)
    assert 0.0 <= probability <= 1.0
