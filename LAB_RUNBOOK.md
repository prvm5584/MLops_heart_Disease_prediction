# End-to-End Lab Runbook

1. Open `Heart_Disease_MLOps_Submit_Ready.ipynb` from the repository root, select Python 3.10, and run **Restart Kernel and Run All Cells**. Save with outputs.
2. If UCI download is blocked, upload `processed.cleveland.data` to `data/` and rerun.
3. Install only missing MLOps packages: `python -m pip install mlflow fastapi uvicorn prometheus-client pytest httpx ruff`.
4. If MLflow was absent during the first run, restart the kernel and rerun model cells. Start UI: `mlflow ui --host 0.0.0.0 --port 5000`.
5. Validate quality: `ruff check api tests` then `pytest -v`.
6. Start API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Open `/docs`, `/health`, `/predict`, `/metrics`. Run `python scripts/smoke_test_api.py` in another terminal.
7. If Docker exists: `docker build -t heart-disease-api:1.0 .` then `docker run --rm -p 8000:8000 heart-disease-api:1.0`.
8. Push the entire folder to GitHub. The workflow lints, executes training, tests, uploads artefacts and builds Docker.
9. If Minikube exists: `minikube start`; `docker build -t heart-disease-api:latest .`; `minikube image load heart-disease-api:latest`; `kubectl apply -f deployment/kubernetes.yaml`; `kubectl get pods,svc`; `minikube service heart-disease-api --url`.
10. Capture notebook, MLflow, Pytest, Swagger, prediction, metrics, Docker, GitHub Actions and Kubernetes evidence. Submit the repository link, ten-page report, access instructions and short demonstration video.
