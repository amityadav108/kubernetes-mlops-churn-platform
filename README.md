cd ~/projects/kubernetes-mlops-churn-platform

cat > README.md <<'EOF'
# Kubernetes-Native MLOps Platform for Customer Churn Prediction

An end-to-end machine learning platform for training, evaluating, serving, and deploying customer churn predictions. Built with Python, scikit-learn, FastAPI, Docker, Kubernetes, pytest, Ruff, and GitHub Actions.

The project demonstrates how to package a machine learning model as a maintainable API service and deploy it using container and orchestration technologies.

## Table of Contents


- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Testing and Code Quality](#testing-and-code-quality)
- [Continuous Integration](#continuous-integration)
- [Engineering Decisions](#engineering-decisions)
- [Limitations and Future Work](#limitations-and-future-work)
- [Author](#author)

## Overview

Customer churn prediction is a binary classification problem that estimates whether a customer is likely to discontinue a service. This project uses the IBM Telco Customer Churn dataset to develop and serve a baseline classification model.

The focus is not only on predictive performance, but also on the engineering workflow required to make a model accessible through an API and deployable in a containerized environment.

### Key Capabilities

- Data validation and input integrity checks
- Reproducible preprocessing and model training with scikit-learn
- Logistic Regression classification and evaluation metrics
- Serialized model artifact for inference
- REST API with FastAPI and Pydantic request validation
- Docker image built with pinned dependencies
- Non-root container execution
- Kubernetes Deployment and ClusterIP Service
- Readiness and liveness probes with resource constraints
- Automated tests, linting, and a GitHub Actions CI workflow

## Architecture

```text
                 DATA & MODEL DEVELOPMENT
                 ────────────────────────

        Telco Customer Churn Dataset
                    |
                    v
              Data Validation
                    |
                    v
        Preprocessing & Feature Encoding
                    |
                    v
           Logistic Regression
                    |
                    v
          Model Evaluation & Saving
                    |
                    v
              Joblib Artifact
                    |
                    v
                 INFERENCE
                 ─────────

              FastAPI Service
                    |
             Request Validation
                    |
                    v
          Prediction & Probability
                    |
                    v
             Docker Container
                    |
                    v
          Kubernetes Deployment
                    |
                    v
                REST API

      GitHub Actions: CI, Tests, Linting,
                 Docker Build




```
## Technology Stack

Python, Pandas, scikit-learn, FastAPI, Docker, Kubernetes, Pytest, Ruff, and GitHub Actions.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1 Score | 60.40% |
| ROC-AUC | 84.21% |

Evaluated on a held-out test set of 1,409 records.

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Churn prediction |
| GET | `/docs` | Interactive API docs |

## Project Structure

- `src/churn_mlops/data/` — Data validation
- `src/churn_mlops/training/` — Model training
- `src/churn_mlops/serving/` — FastAPI service
- `models/` — Saved model
- `tests/` — Automated tests
- `k8s/` — Kubernetes manifests
- `.github/workflows/` — CI workflow
- `Dockerfile` — Container configuration

## Getting Started

```bash
git clone https://github.com/amityadav108/kubernetes-mlops-churn-platform.git
cd kubernetes-mlops-churn-platform
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies specified in `pyproject.toml` and configure the required dataset before training.

## Docker Deployment

Build and run the image after preparing the dependencies required by your Dockerfile.

```bash
docker build -t churn-mlops-api:1.1 .
docker run --rm -p 8000:8000 churn-mlops-api:1.1
```

## Kubernetes Deployment

Ensure Kubernetes is running and the image tag matches your deployment manifest.

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
```

## Testing and Code Quality

```bash
pytest -v
ruff check .
```

## Continuous Integration

GitHub Actions is configured in `.github/workflows/ci.yml`. Check the repository's Actions tab to verify workflow results.

## Engineering Decisions

- Logistic Regression provides a simple binary classification baseline.
- A preprocessing pipeline keeps data transformations consistent.
- FastAPI provides validated prediction endpoints.
- Docker packages the application.
- Kubernetes manages deployment resources.
- Pytest and Ruff support testing and code quality.

## Limitations and Future Work

- Add MLflow experiment tracking.
- Monitor data drift and model performance.
- Implement automated retraining.
- Add model explainability and API security.
- Expand integration testing and deployment automation.

These are potential improvements, not currently implemented features.

## Author

**Amit Yadav**

GitHub: [amityadav108](https://github.com/amityadav108)
