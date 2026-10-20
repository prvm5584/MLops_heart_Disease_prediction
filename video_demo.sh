#!/bin/bash

PROJECT="/home/s2025ae05584/workspace/heart_disease_mlops_final_submission"

cd "$PROJECT" || exit 1

case "$1" in

project)
    echo "===== PROJECT STRUCTURE ====="
    pwd
    find . -maxdepth 2 -type f | sort

    echo
    echo "===== FINAL MODEL RESULTS ====="
    cat artifacts/model_comparison.csv

    echo
    cat artifacts/final_metrics.json
    ;;


mlflow)
    mlflow ui \
      --backend-store-uri sqlite:////home/s2025ae05584/workspace/heart_disease_mlops_final_submission/mlflow.db \
      --host 0.0.0.0 \
      --port 5000 \
      --disable-security-middleware
    ;;


quality)
    echo "===== RUFF ====="
    ruff check api tests

    echo
    echo "===== PYTEST ====="
    pytest -v
    ;;


api)
    python -m uvicorn api.main:app \
      --host 0.0.0.0 \
      --port 8000
    ;;


health)
    echo "===== HEALTH ====="
    curl -s http://127.0.0.1:8000/health
    echo
    ;;


predict)
    echo "===== PREDICTION ====="

    curl -s -X POST \
      http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "age":54,
        "sex":1,
        "cp":4,
        "trestbps":130,
        "chol":246,
        "fbs":0,
        "restecg":2,
        "thalach":150,
        "exang":0,
        "oldpeak":1.2,
        "slope":2,
        "ca":0,
        "thal":3
      }'

    echo
    ;;


smoke)
    python scripts/smoke_test_api.py
    ;;


metrics)
    echo "===== PROMETHEUS METRICS ====="

    curl -s http://127.0.0.1:8000/metrics \
      | grep -E \
      "prediction_requests_total|prediction_errors_total|prediction_latency_seconds"

    echo
    ;;


docker-build)
    echo "===== DOCKER BUILD ====="

    docker build \
      -t heart-disease-api:1.0 \
      .

    echo
    echo "===== DOCKER IMAGE ====="

    docker images | grep heart-disease
    ;;


docker-run)
    docker run \
      --rm \
      --name heart-disease-api \
      -p 8000:8000 \
      heart-disease-api:1.0
    ;;


docker-check)
    echo "===== RUNNING CONTAINER ====="

    docker ps

    echo
    echo "===== CONTAINER HEALTH ====="

    curl -s http://127.0.0.1:8000/health

    echo
    echo "===== CONTAINER PREDICTION ====="

    curl -s -X POST \
      http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "age":54,
        "sex":1,
        "cp":4,
        "trestbps":130,
        "chol":246,
        "fbs":0,
        "restecg":2,
        "thalach":150,
        "exang":0,
        "oldpeak":1.2,
        "slope":2,
        "ca":0,
        "thal":3
      }'

    echo
    ;;


git)
    echo "===== GIT STATUS ====="

    git status

    echo
    echo "===== RECENT COMMITS ====="

    git log --oneline -5

    echo
    echo "===== REMOTE ====="

    git remote -v
    ;;


k8s)
    echo "===== MINIKUBE ====="

    minikube status

    echo
    echo "===== NODES ====="

    kubectl get nodes

    echo
    echo "===== DEPLOYMENTS ====="

    kubectl get deployments

    echo
    echo "===== PODS ====="

    kubectl get pods

    echo
    echo "===== SERVICES ====="

    kubectl get services
    ;;


k8s-deploy)
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
    ;;


k8s-url)
    minikube service \
      heart-disease-api \
      --url
    ;;


artifacts)
    echo "===== SUBMISSION ARTEFACTS ====="

    ls -lh \
      Heart_Disease_MLOps_Submit_Ready.ipynb \
      requirements.txt \
      models/final_model.joblib \
      artifacts/model_comparison.csv \
      artifacts/final_metrics.json \
      Dockerfile \
      deployment/kubernetes.yaml \
      monitoring/prometheus.yml
    ;;


*)
    echo
    echo "MLOps Assignment Video Demo"
    echo
    echo "Usage:"
    echo
    echo "./video_demo.sh project"
    echo "./video_demo.sh mlflow"
    echo "./video_demo.sh quality"
    echo "./video_demo.sh api"
    echo "./video_demo.sh health"
    echo "./video_demo.sh predict"
    echo "./video_demo.sh smoke"
    echo "./video_demo.sh metrics"
    echo "./video_demo.sh docker-build"
    echo "./video_demo.sh docker-run"
    echo "./video_demo.sh docker-check"
    echo "./video_demo.sh git"
    echo "./video_demo.sh k8s"
    echo "./video_demo.sh k8s-deploy"
    echo "./video_demo.sh k8s-url"
    echo "./video_demo.sh artifacts"
    echo
    ;;

esac
