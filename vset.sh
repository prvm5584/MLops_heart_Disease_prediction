###############################################################################
# MLOPS ASSIGNMENT 01 - VIDEO DEMO COMMANDS
# Student: 2025AE05584
#
# Keep this file/terminal open while recording.
#
# IMPORTANT:
# - Do NOT run all commands at once.
# - Run one SECTION at a time while narrating.
# - Commands starting long-running servers must stay open in their terminal.
###############################################################################


###############################################################################
# SECTION 0 - PROJECT ROOT
###############################################################################

cd /home/s2025ae05584/workspace/heart_disease_mlops_final_submission

pwd

ls -la

# NARRATE:
# "This is my end-to-end Heart Disease MLOps repository.
#  It contains the executed notebook, API, tests, Docker configuration,
#  Kubernetes deployment, monitoring and CI/CD workflow."


###############################################################################
# SECTION 1 - SHOW IMPORTANT PROJECT FILES
###############################################################################

find . -maxdepth 2 -type f | sort

# Highlight these visually:
#
# Heart_Disease_MLOps_Submit_Ready.ipynb
# requirements.txt
# models/final_model.joblib
# artifacts/model_comparison.csv
# artifacts/final_metrics.json
# api/main.py
# tests/test_project.py
# Dockerfile
# deployment/kubernetes.yaml
# monitoring/prometheus.yml
# .github/workflows/ci.yml
#
# NARRATE:
# "The model, preprocessing pipeline, training evidence, serving layer,
#  automated tests and deployment configuration are maintained
#  as reproducible repository artefacts."


###############################################################################
# SECTION 2 - NOTEBOOK RESULT / GENERATED ARTEFACTS
###############################################################################

ls -lh models/

ls -lh artifacts/

ls -lh screenshots/

cat artifacts/final_metrics.json

cat artifacts/model_comparison.csv

# NARRATE:
# "The notebook completed successfully and generated the final model,
#  data-quality evidence, model comparison and evaluation artefacts.
#
#  Two classifiers were evaluated: Logistic Regression and Random Forest.
#  Logistic Regression was selected from my executed results."


###############################################################################
# SECTION 3 - OPTIONAL: SHOW NOTEBOOK IN JUPYTER
###############################################################################

# SWITCH TO THE JUPYTER BROWSER.
#
# Open:
# Heart_Disease_MLOps_Submit_Ready.ipynb
#
# Show these outputs, but DO NOT run the notebook again:
#
# 1. Data quality
# 2. EDA
# 3. Correlation heatmap
# 4. Preprocessing pipeline
# 5. Logistic Regression
# 6. Random Forest
# 7. Model comparison
# 8. Confusion matrix / ROC
# 9. Final model
# 10. Final validation:
#
# "Submission artefact validation passed."
#
# NARRATE:
# "The preprocessing uses imputation, encoding and standardisation inside
#  the same Scikit-learn pipeline that is ultimately serialised."


###############################################################################
# SECTION 4 - MODEL RESULT SUMMARY
###############################################################################

cat artifacts/final_metrics.json

# From the executed project:
#
# Selected model       : Logistic Regression
# Test ROC-AUC         : 0.9665
# Accuracy             : 0.8852
# Precision            : 0.8387
# Recall               : 0.9286
# F1                   : 0.8814
#
# NARRATE:
# "I selected Logistic Regression using the observed model comparison,
#  with held-out ROC-AUC as the primary selection metric."


###############################################################################
# SECTION 5 - MLFLOW
###############################################################################

# TERMINAL A
#
# Start MLflow using the SQLite backend:

mlflow ui \
  --backend-store-uri sqlite:////home/s2025ae05584/workspace/heart_disease_mlops_final_submission/mlflow.db \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*" \
  --x-frame-options NONE


###############################################################################
# NOW SWITCH TO MLFLOW UI THROUGH THE LAB PORT 5000
###############################################################################

# Show:
#
# Experiment:
# heart-disease-final
#
# Runs:
#
# Logistic Regression
# Random Forest
# Final Model Evidence - Complete
#
# Open Logistic Regression:
#
# Show parameters:
# C
#
# Show metrics:
#
# cv_roc_auc
# test_accuracy
# test_precision
# test_recall
# test_f1
# test_roc_auc
#
#
# Return and open Random Forest:
#
# Show parameters:
#
# n_estimators
# max_depth
#
# Show metrics.
#
#
# Open:
#
# Final Model Evidence - Complete
#
# Show:
#
# selected_model
# model_format = joblib
# model_file = final_model.joblib
#
# Then open Artifacts:
#
# model/
#      final_model.joblib
#
# evidence/
#      model_comparison.csv
#      final_metrics.json
#      data_quality.csv
#      model_comparison.png
#      final_evaluation.png
#      correlation_heatmap.png
#      eda_overview.png
#
#
# NARRATE:
#
# "MLflow provides experiment traceability.
#  Each candidate run captures parameters and metrics.
#  The final evidence run retains the selected production model and
#  evaluation artefacts."


###############################################################################
# SECTION 6 - CODE QUALITY
###############################################################################

# Open ANOTHER terminal. Keep MLflow running if possible.

cd /home/s2025ae05584/workspace/heart_disease_mlops_final_submission

ruff check api tests

# EXPECT:
#
# All checks passed!
#
# NARRATE:
# "Ruff is used as the static code-quality gate."


###############################################################################
# SECTION 7 - AUTOMATED TESTS
###############################################################################

pytest -v

# EXPECT ALL TESTS:
#
# PASSED
#
# Particularly demonstrate:
#
# model exists
# metric artefacts exist
# model loads successfully
# pipeline handles missing values
#
# NARRATE:
# "Pytest checks the packaged model and also verifies that preprocessing
#  handles missing clinical values during inference."


###############################################################################
# SECTION 8 - FASTAPI
###############################################################################

# If MLflow is no longer required for the live video, stop MLflow:
#
# CTRL+C
#
# Then start API:

python -m uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000


###############################################################################
# SECTION 9 - API HEALTH CHECK
###############################################################################

# NEW TERMINAL

cd /home/s2025ae05584/workspace/heart_disease_mlops_final_submission

curl http://127.0.0.1:8000/health

# EXPECT:
#
# {"status":"ok","model_loaded":true}
#
# NARRATE:
# "The health endpoint verifies that the serving application is running
#  and the trained model has been loaded."


###############################################################################
# SECTION 10 - LIVE MODEL PREDICTION
###############################################################################

curl -X POST \
  http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "ca": 0,
    "thal": 3
  }'

# EXPECT RESPONSE STRUCTURE:
#
# {
#   "prediction": ...,
#   "confidence": ...,
#   "heart_disease_probability": ...
# }
#
# Use whatever values YOUR API produces.
#
# NARRATE:
#
# "The prediction endpoint accepts a JSON patient record and returns
#  both the class prediction and confidence/probability."


###############################################################################
# SECTION 11 - AUTOMATED API SMOKE TEST
###############################################################################

python scripts/smoke_test_api.py

# Demonstrates:
#
# /health
# /predict
# /metrics
#
# NARRATE:
# "I also created a repeatable API smoke test covering health,
#  prediction and monitoring endpoints."


###############################################################################
# SECTION 12 - GENERATE MONITORING ACTIVITY
###############################################################################

python scripts/smoke_test_api.py
python scripts/smoke_test_api.py
python scripts/smoke_test_api.py


###############################################################################
# SECTION 13 - PROMETHEUS-FORMAT MONITORING
###############################################################################

curl -s http://127.0.0.1:8000/metrics \
  | grep -E "prediction_requests|prediction_errors|prediction_latency"

# SHOW:
#
# prediction_requests_total
# prediction_errors_total
# prediction_latency_seconds
#
# NARRATE:
#
# "The API exposes Prometheus-compatible operational metrics,
#  including prediction request volume, errors and latency."


###############################################################################
# SECTION 14 - APPLICATION LOGGING
###############################################################################

# SWITCH TO THE UVICORN TERMINAL.
#
# Show entries containing:
#
# GET /health 200
# POST /predict 200
# GET /metrics 200
# Prediction completed...
#
# NARRATE:
#
# "The service also records API activity and prediction events
#  through structured application logs."


###############################################################################
# SECTION 15 - STOP LOCAL FASTAPI BEFORE DOCKER
###############################################################################

# In Uvicorn terminal:
#
# CTRL+C


###############################################################################
# SECTION 16 - DOCKER BUILD
###############################################################################

cd /home/s2025ae05584/workspace/heart_disease_mlops_final_submission

docker build \
  -t heart-disease-api:1.0 \
  .

# NARRATE:
#
# "The API and model are packaged into a Docker container to provide
#  an isolated and reproducible serving environment."


###############################################################################
# SECTION 17 - VERIFY DOCKER IMAGE
###############################################################################

docker images | grep heart-disease

# SHOW:
#
# heart-disease-api   1.0
#
# NARRATE:
# "The heart-disease-api version 1.0 image has been built locally."


###############################################################################
# SECTION 18 - RUN DOCKER CONTAINER
###############################################################################

docker run \
  --rm \
  --name heart-disease-api \
  -p 8000:8000 \
  heart-disease-api:1.0

# LEAVE RUNNING.


###############################################################################
# SECTION 19 - VERIFY RUNNING CONTAINER
###############################################################################

# NEW TERMINAL

docker ps

curl http://127.0.0.1:8000/health

python scripts/smoke_test_api.py

# NARRATE:
#
# "The same model-serving contract works from the isolated Docker
#  container, demonstrating reproducible inference."


###############################################################################
# SECTION 20 - STOP DOCKER CONTAINER
###############################################################################

# CTRL+C in the container terminal.


###############################################################################
# SECTION 21 - GITHUB REPOSITORY
###############################################################################

# SWITCH TO BROWSER.
#
# Open your GitHub repository:
#
# https://github.com/prvm5584/MLops_heart_Disease_prediction.git
#
# Show:
#
# Heart_Disease_MLOps_Submit_Ready.ipynb
# requirements.txt
# Dockerfile
# api/
# tests/
# deployment/
# monitoring/
# screenshots/
# .github/workflows/ci.yml
#
# NARRATE:
#
# "The complete reproducible implementation is version-controlled
#  in GitHub, including the notebook, tests, API, Dockerfile,
#  deployment and monitoring configuration."


###############################################################################
# SECTION 22 - GITHUB ACTIONS / CI-CD
###############################################################################

# In GitHub browser:
#
# Repository
#     ->
# Actions
#     ->
# mlops-ci
#
# If a successful hosted run is available, SHOW:
#
# Checkout
# Set up Python
# Install dependencies
# Ruff
# Notebook execution
# Pytest
# Artifact upload
# Docker build
#
# IMPORTANT:
# Only show/claim successful GitHub Actions execution if the UI actually
# displays a successful completed run.
#
# Otherwise show:
#
# .github/workflows/ci.yml
#
# and narrate:
#
# "The repository contains the CI/CD workflow configured to perform
#  linting, notebook execution, automated tests, artefact upload
#  and Docker build."


###############################################################################
# SECTION 23 - KUBERNETES
###############################################################################

kubectl version --client

minikube status

# If Minikube is running:

kubectl get nodes

kubectl get deployments

kubectl get pods

kubectl get services


###############################################################################
# SECTION 24 - KUBERNETES DEPLOYMENT IF NEEDED
###############################################################################

docker build \
  -t heart-disease-api:latest \
  .

minikube image load \
  heart-disease-api:latest

kubectl apply \
  -f deployment/kubernetes.yaml

kubectl rollout status \
  deployment/heart-disease-api

kubectl get pods

kubectl get services


###############################################################################
# SECTION 25 - KUBERNETES SERVICE URL
###############################################################################

minikube service \
  heart-disease-api \
  --url

# COPY THE ACTUAL URL RETURNED.
#
# In your validated environment this may be a NodePort address.
# Use ONLY the address currently returned by your environment.


###############################################################################
# SECTION 26 - TEST KUBERNETES HEALTH
###############################################################################

# Replace ACTUAL_MINIKUBE_URL with the returned value.

curl ACTUAL_MINIKUBE_URL/health

# EXPECT:
#
# {"status":"ok","model_loaded":true}


###############################################################################
# SECTION 27 - KUBERNETES PREDICTION
###############################################################################

curl -X POST \
  ACTUAL_MINIKUBE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "ca": 0,
    "thal": 3
  }'

# NARRATE:
#
# "The Dockerised prediction service is deployed through Kubernetes
#  and exposed through the configured service endpoint."


###############################################################################
# SECTION 28 - SHOW KUBERNETES MONITORING
###############################################################################

curl ACTUAL_MINIKUBE_URL/metrics \
  | grep -E "prediction_requests|prediction_errors|prediction_latency"


###############################################################################
# SECTION 29 - FINAL REPORT
###############################################################################

# Open:
#
# MLOps_Assignment01_2025AE05584_Submit_Ready_10_Page_Report.pdf
#
# Quickly show:
#
# Page 1  Executive summary
# Page 3  EDA
# Page 5  Model results
# Page 6  MLflow
# Page 7  Testing
# Page 8  API monitoring
# Page 9  Docker/Kubernetes
# Page 10 GitHub + reproduction instructions
#
# NARRATE:
#
# "The final ten-page report consolidates the executed evidence,
#  model results, experiment tracking, testing, serving,
#  containerisation, deployment and monitoring."


###############################################################################
# SECTION 30 - FINAL VIDEO CLOSE
###############################################################################

# SAY:
#
# "This completes the end-to-end MLOps lifecycle:
#
#  UCI Heart Disease data
#  -> EDA and preprocessing
#  -> two model experiments
#  -> cross-validation and tuning
#  -> MLflow tracking
#  -> reusable model packaging
#  -> automated code quality and testing
#  -> FastAPI serving
#  -> monitoring and logging
#  -> Docker containerisation
#  -> CI/CD configuration
#  -> Kubernetes deployment.
#
#  The complete code, evidence and reproduction instructions
#  are available in the GitHub repository and final report."
###############################################################################