
from pathlib import Path
import json
import sys

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd()
EXPERIMENT_NAME = "heart-disease-final"

MLRUNS_DIR = (PROJECT_ROOT / "mlruns").resolve()

COMPARISON_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "model_comparison.csv"
)

FINAL_METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "final_metrics.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_model.joblib"
)


# ============================================================
# VALIDATE FILES
# ============================================================

print("=" * 70)
print("HEART DISEASE MLOPS - MLFLOW SETUP")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)


required_files = [
    COMPARISON_FILE,
    FINAL_METRICS_FILE,
    MODEL_FILE
]


missing = [
    path
    for path in required_files
    if not path.exists()
]


if missing:

    print("\nERROR - required files missing:")

    for path in missing:
        print("MISSING:", path)

    sys.exit(1)


for path in required_files:
    print("OK:", path)


# ============================================================
# CONFIGURE MLFLOW
# ============================================================

MLRUNS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


tracking_uri = MLRUNS_DIR.as_uri()


mlflow.set_tracking_uri(
    tracking_uri
)


experiment = mlflow.set_experiment(
    EXPERIMENT_NAME
)


print("\nMLflow Tracking URI:")
print(mlflow.get_tracking_uri())

print("\nExperiment:")
print(EXPERIMENT_NAME)

print("\nExperiment ID:")
print(experiment.experiment_id)


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

comparison = pd.read_csv(
    COMPARISON_FILE
)


print("\nMODEL COMPARISON")
print("-" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# FIND EXISTING RUNS
# ============================================================

existing = mlflow.search_runs(
    experiment_ids=[
        experiment.experiment_id
    ]
)


if len(existing) > 0:

    existing_names = set(
        existing[
            "tags.mlflow.runName"
        ].dropna()
    )

else:

    existing_names = set()


print("\nExisting runs:")

if existing_names:

    for name in sorted(existing_names):
        print("-", name)

else:
    print("None")


# ============================================================
# LOG MODEL EXPERIMENTS
# ============================================================

metric_columns = [
    "cv_roc_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc"
]


for _, row in comparison.iterrows():

    model_name = str(
        row["model"]
    )


    if model_name in existing_names:

        print(
            "\nSKIPPING existing run:",
            model_name
        )

        continue


    print(
        "\nLogging:",
        model_name
    )


    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_param(
            "model_name",
            model_name
        )


        # Hyperparameters
        if "best_parameters" in comparison.columns:

            try:

                params = json.loads(
                    row[
                        "best_parameters"
                    ]
                )

                for key, value in params.items():

                    clean_key = (
                        key.replace(
                            "classifier__",
                            ""
                        )
                    )

                    mlflow.log_param(
                        clean_key,
                        value
                    )

            except Exception as exc:

                print(
                    "Parameter warning:",
                    exc
                )


        # Metrics
        for metric in metric_columns:

            if metric not in comparison.columns:
                continue

            value = row[metric]

            if pd.isna(value):
                continue

            mlflow.log_metric(
                metric,
                float(value)
            )


        # Comparison evidence
        mlflow.log_artifact(
            str(COMPARISON_FILE)
        )


print(
    "\nCandidate model logging completed."
)


# ============================================================
# FINAL MODEL
# ============================================================

with open(
    FINAL_METRICS_FILE,
    "r",
    encoding="utf-8"
) as file:

    final_metrics = json.load(
        file
    )


selected_model = final_metrics[
    "selected_model"
]


final_model = joblib.load(
    MODEL_FILE
)


print("\nSelected final model:")
print(selected_model)


# ============================================================
# LOG FINAL MODEL
# ============================================================

FINAL_RUN = "Final Model Evidence"


existing = mlflow.search_runs(
    experiment_ids=[
        experiment.experiment_id
    ]
)


existing_names = set(
    existing[
        "tags.mlflow.runName"
    ].dropna()
)


if FINAL_RUN not in existing_names:

    print(
        "\nLogging final model evidence..."
    )


    with mlflow.start_run(
        run_name=FINAL_RUN
    ):

        mlflow.log_param(
            "selected_model",
            selected_model
        )


        for key, value in final_metrics.items():

            if key == "selected_model":
                continue

            try:

                mlflow.log_metric(
                    key,
                    float(value)
                )

            except Exception:
                pass


        # Final serialised model
        mlflow.sklearn.log_model(
            final_model,
            "model"
        )


        # Assignment evidence
        artifacts = [

            "artifacts/model_comparison.csv",

            "artifacts/final_metrics.json",

            "artifacts/data_quality.csv",

            "screenshots/model_comparison.png",

            "screenshots/final_evaluation.png",

            "screenshots/correlation_heatmap.png",

            "screenshots/eda_overview.png"
        ]


        print("\nLogging artefacts:")


        for artifact_name in artifacts:

            artifact_path = (
                PROJECT_ROOT
                / artifact_name
            )


            if artifact_path.exists():

                mlflow.log_artifact(
                    str(artifact_path)
                )

                print(
                    "OK:",
                    artifact_name
                )

            else:

                print(
                    "NOT FOUND:",
                    artifact_name
                )


else:

    print(
        "\nFinal Model Evidence already exists."
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

experiment = (
    mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )
)


runs = mlflow.search_runs(
    experiment_ids=[
        experiment.experiment_id
    ]
)


print("\n" + "=" * 70)
print("MLFLOW RUNS")
print("=" * 70)


columns = [

    "run_id",

    "tags.mlflow.runName",

    "metrics.cv_roc_auc",

    "metrics.test_accuracy",

    "metrics.test_precision",

    "metrics.test_recall",

    "metrics.test_f1",

    "metrics.test_roc_auc"
]


available_columns = [

    column
    for column in columns
    if column in runs.columns
]


print(
    runs[
        available_columns
    ].to_string(
        index=False
    )
)


run_names = set(
    runs[
        "tags.mlflow.runName"
    ].dropna()
)


expected = {

    "Logistic Regression",

    "Random Forest",

    "Final Model Evidence"
}


missing_runs = (
    expected
    - run_names
)


print("\n" + "=" * 70)


if missing_runs:

    print(
        "WARNING - missing runs:"
    )

    for run in missing_runs:
        print("-", run)

else:

    print(
        "PASS - ALL EXPECTED "
        "MLFLOW RUNS CREATED"
    )


print("=" * 70)


print("\nExperiment:")
print(EXPERIMENT_NAME)

print("\nMLflow directory:")
print(MLRUNS_DIR)

print(
    "\nRun this command next:"
)

print()

print(
    "mlflow ui "
    "--backend-store-uri ./mlruns "
    "--host 0.0.0.0 "
    "--port 5000"
)
