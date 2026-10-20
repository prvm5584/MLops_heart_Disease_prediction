from pathlib import Path
import json
import sys

import pandas as pd
import mlflow


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd()

EXPERIMENT_NAME = "heart-disease-final"

# SQLite backend for modern MLflow
MLFLOW_DB = (
    PROJECT_ROOT
    / "mlflow.db"
).resolve()

TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB}"
)

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

DATA_QUALITY_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "data_quality.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_model.joblib"
)


# ============================================================
# START
# ============================================================

print("=" * 72)
print("HEART DISEASE MLOPS - MLFLOW EVIDENCE SETUP")
print("=" * 72)

print("\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# VALIDATE NOTEBOOK OUTPUTS
# ============================================================

required_files = [
    COMPARISON_FILE,
    FINAL_METRICS_FILE,
    MODEL_FILE,
]


missing_files = [
    file_path
    for file_path in required_files
    if not file_path.exists()
]


if missing_files:

    print(
        "\nERROR: Required notebook "
        "artefacts are missing."
    )

    for file_path in missing_files:

        print(
            "MISSING:",
            file_path
        )

    print(
        "\nRun the completed notebook first."
    )

    sys.exit(1)


print(
    "\nNotebook artefacts"
)

print("-" * 72)


for file_path in required_files:

    print(
        "OK:",
        file_path
    )


# ============================================================
# CONFIGURE SQLITE MLFLOW
# ============================================================

print(
    "\nConfiguring MLflow..."
)


mlflow.set_tracking_uri(
    TRACKING_URI
)


print(
    "Tracking URI:",
    mlflow.get_tracking_uri()
)


print(
    "Database:",
    MLFLOW_DB
)


# ============================================================
# CREATE / GET EXPERIMENT
# ============================================================

experiment = mlflow.set_experiment(
    EXPERIMENT_NAME
)


print(
    "\nExperiment:",
    EXPERIMENT_NAME
)


print(
    "Experiment ID:",
    experiment.experiment_id
)


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

comparison = pd.read_csv(
    COMPARISON_FILE
)


print(
    "\n" + "=" * 72
)

print(
    "MODEL COMPARISON FROM NOTEBOOK"
)

print(
    "=" * 72
)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "model",
    "cv_roc_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
]


missing_columns = [
    column
    for column in required_columns
    if column not in comparison.columns
]


if missing_columns:

    print(
        "\nERROR: model_comparison.csv "
        "is missing required columns."
    )

    for column in missing_columns:

        print(
            "-",
            column
        )

    sys.exit(1)


# ============================================================
# HELPER: GET EXISTING RUN NAMES
# ============================================================

def get_existing_run_names():

    runs = mlflow.search_runs(
        experiment_ids=[
            experiment.experiment_id
        ]
    )

    if (
        runs.empty
        or
        "tags.mlflow.runName"
        not in runs.columns
    ):

        return set()

    return set(
        runs[
            "tags.mlflow.runName"
        ].dropna()
    )


# ============================================================
# EXISTING RUNS
# ============================================================

existing_names = (
    get_existing_run_names()
)


print(
    "\nExisting MLflow runs:"
)


if existing_names:

    for run_name in sorted(
        existing_names
    ):

        print(
            "-",
            run_name
        )

else:

    print(
        "None"
    )


# ============================================================
# METRICS
# ============================================================

metric_columns = [
    "cv_roc_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
]


# ============================================================
# LOG LOGISTIC REGRESSION AND RANDOM FOREST
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "LOGGING CANDIDATE MODEL EXPERIMENTS"
)

print(
    "=" * 72
)


for _, row in comparison.iterrows():

    model_name = str(
        row["model"]
    )


    if model_name in existing_names:

        print(
            "\nSKIP existing run:",
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

        # ----------------------------------------------------
        # MODEL DETAILS
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name
        )


        # ----------------------------------------------------
        # HYPERPARAMETERS
        # ----------------------------------------------------

        if (
            "best_parameters"
            in comparison.columns
        ):

            raw_parameters = row[
                "best_parameters"
            ]


            try:

                parameters = json.loads(
                    raw_parameters
                )


                for key, value in (
                    parameters.items()
                ):

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


                    print(
                        "  PARAM:",
                        clean_key,
                        "=",
                        value
                    )


            except Exception as exc:

                print(
                    "  Parameter parsing warning:",
                    exc
                )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        for metric_name in (
            metric_columns
        ):

            value = row[
                metric_name
            ]


            if pd.isna(
                value
            ):

                continue


            numeric_value = float(
                value
            )


            mlflow.log_metric(
                metric_name,
                numeric_value
            )


            print(
                "  METRIC:",
                metric_name,
                "=",
                round(
                    numeric_value,
                    4
                )
            )


        # ----------------------------------------------------
        # CANDIDATE RUN ARTIFACTS
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(
                COMPARISON_FILE
            ),
            artifact_path="evidence",
        )


        if DATA_QUALITY_FILE.exists():

            mlflow.log_artifact(
                str(
                    DATA_QUALITY_FILE
                ),
                artifact_path="evidence",
            )


print(
    "\nCandidate model logging completed."
)


# ============================================================
# LOAD FINAL METRICS
# ============================================================

with open(
    FINAL_METRICS_FILE,
    "r",
    encoding="utf-8",
) as file:

    final_metrics = json.load(
        file
    )


selected_model = final_metrics[
    "selected_model"
]


print(
    "\n" + "=" * 72
)

print(
    "FINAL MODEL"
)

print(
    "=" * 72
)


print(
    "Selected model:",
    selected_model
)


print(
    "Model file:",
    MODEL_FILE
)


print(
    "Model file size:",
    MODEL_FILE.stat().st_size,
    "bytes",
)


# ============================================================
# LOG FINAL MODEL EVIDENCE
# ============================================================

#
# IMPORTANT
#
# We intentionally log the validated Joblib model as an
# artifact instead of using mlflow.sklearn.log_model().
#
# The lab's MLflow/skops combination rejects numpy.dtype
# during skops security validation.
#
# The Joblib file is the exact serialised pipeline produced
# and already validated by the completed notebook.
#

FINAL_RUN_NAME = (
    "Final Model Evidence - Complete"
)


existing_names = (
    get_existing_run_names()
)


if FINAL_RUN_NAME in existing_names:

    print(
        "\nSKIP existing run:",
        FINAL_RUN_NAME
    )


else:

    print(
        "\n" + "=" * 72
    )

    print(
        "LOGGING FINAL MODEL EVIDENCE"
    )

    print(
        "=" * 72
    )


    with mlflow.start_run(
        run_name=FINAL_RUN_NAME
    ):

        # ----------------------------------------------------
        # MODEL METADATA
        # ----------------------------------------------------

        mlflow.log_param(
            "selected_model",
            selected_model
        )


        mlflow.log_param(
            "model_format",
            "joblib"
        )


        mlflow.log_param(
            "model_file",
            MODEL_FILE.name
        )


        mlflow.log_param(
            "pipeline_contains_preprocessing",
            "true"
        )


        # ----------------------------------------------------
        # FINAL METRICS
        # ----------------------------------------------------

        print(
            "\nFinal metrics:"
        )


        for key, value in (
            final_metrics.items()
        ):

            if key == "selected_model":

                continue


            try:

                numeric_value = float(
                    value
                )


                mlflow.log_metric(
                    key,
                    numeric_value
                )


                print(
                    "  ",
                    key,
                    "=",
                    round(
                        numeric_value,
                        4
                    )
                )


            except (
                TypeError,
                ValueError
            ):

                print(
                    "  SKIP non-numeric:",
                    key,
                    value
                )


        # ----------------------------------------------------
        # PRODUCTION MODEL
        # ----------------------------------------------------

        print(
            "\nLogging production "
            "Joblib model..."
        )


        mlflow.log_artifact(
            str(
                MODEL_FILE
            ),
            artifact_path="model",
        )


        print(
            "  OK: model/"
            + MODEL_FILE.name
        )


        # ----------------------------------------------------
        # ASSIGNMENT EVIDENCE
        # ----------------------------------------------------

        evidence_files = [

            "artifacts/model_comparison.csv",

            "artifacts/final_metrics.json",

            "artifacts/data_quality.csv",

            "screenshots/model_comparison.png",

            "screenshots/final_evaluation.png",

            "screenshots/correlation_heatmap.png",

            "screenshots/eda_overview.png",
        ]


        print(
            "\nLogging assignment evidence:"
        )


        for relative_name in (
            evidence_files
        ):

            file_path = (
                PROJECT_ROOT
                / relative_name
            )


            if file_path.exists():

                mlflow.log_artifact(
                    str(
                        file_path
                    ),
                    artifact_path="evidence",
                )


                print(
                    "  OK:",
                    relative_name
                )


            else:

                print(
                    "  NOT FOUND:",
                    relative_name
                )


    print(
        "\nFinal model evidence "
        "logging completed."
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

experiment = (
    mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )
)


if experiment is None:

    raise RuntimeError(
        "MLflow experiment was "
        "not found after logging."
    )


runs = mlflow.search_runs(

    experiment_ids=[
        experiment.experiment_id
    ],

    order_by=[
        "start_time DESC"
    ],
)


print(
    "\n" + "=" * 72
)

print(
    "MLFLOW EXPERIMENT VERIFICATION"
)

print(
    "=" * 72
)


print(
    "Experiment:",
    experiment.name
)


print(
    "Experiment ID:",
    experiment.experiment_id
)


print(
    "Total runs:",
    len(runs)
)


# ============================================================
# DISPLAY RUN INFORMATION
# ============================================================

display_columns = [

    "run_id",

    "tags.mlflow.runName",

    "metrics.cv_roc_auc",

    "metrics.test_accuracy",

    "metrics.test_precision",

    "metrics.test_recall",

    "metrics.test_f1",

    "metrics.test_roc_auc",
]


available_columns = [

    column

    for column in display_columns

    if column in runs.columns
]


print(
    "\nTracked runs:"
)


if available_columns:

    print(

        runs[
            available_columns
        ].to_string(
            index=False
        )
    )

else:

    print(
        runs.to_string(
            index=False
        )
    )


# ============================================================
# VERIFY EXPECTED RUNS
# ============================================================

if (

    "tags.mlflow.runName"
    in runs.columns

):

    run_names = set(

        runs[
            "tags.mlflow.runName"
        ].dropna()
    )

else:

    run_names = set()


expected_runs = {

    "Logistic Regression",

    "Random Forest",

    "Final Model Evidence - Complete",
}


missing_runs = (

    expected_runs
    - run_names
)


print(
    "\n" + "=" * 72
)

print(
    "FINAL VALIDATION"
)

print(
    "=" * 72
)


if missing_runs:

    print(
        "WARNING - missing "
        "expected runs:"
    )


    for run_name in sorted(
        missing_runs
    ):

        print(
            "-",
            run_name
        )


else:

    print(
        "PASS - ALL EXPECTED "
        "MLFLOW RUNS CREATED"
    )


# ============================================================
# DATABASE INFORMATION
# ============================================================

print(
    "\nExperiment:"
)

print(
    EXPERIMENT_NAME
)


print(
    "\nTracking database:"
)

print(
    MLFLOW_DB
)


print(
    "\nTracking URI:"
)

print(
    mlflow.get_tracking_uri()
)


print(
    "\nDatabase exists:",
    MLFLOW_DB.exists()
)


if MLFLOW_DB.exists():

    print(
        "Database size:",
        MLFLOW_DB.stat().st_size,
        "bytes",
    )


# ============================================================
# NEXT STEP
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "MLFLOW EVIDENCE SETUP COMPLETED"
)

print(
    "=" * 72
)


print(
    "\nStart MLflow UI with:"
)


print()


print(
    "mlflow ui "
    "--backend-store-uri "
    "sqlite:///mlflow.db "
    "--host 0.0.0.0 "
    "--port 5000"
)


print()


print(
    "Expected experiment:"
)

print(
    "  heart-disease-final"
)


print(
    "\nExpected completed runs:"
)

print(
    "  1. Logistic Regression"
)

print(
    "  2. Random Forest"
)

print(
    "  3. Final Model Evidence - Complete"
)


print(
    "\nNOTE:"
)

print(
    "An older 'Final Model Evidence' run "
    "may appear as FAILED because of the "
    "previous skops serialisation error."
)

print(
    "Ignore that old run and use "
    "'Final Model Evidence - Complete' "
    "for your assignment screenshots."
)

print(
    "\nCapture the MLflow evidence, "
    "then continue with Ruff, Pytest, "
    "FastAPI and Docker."
)