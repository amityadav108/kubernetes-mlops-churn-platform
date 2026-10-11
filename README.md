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
