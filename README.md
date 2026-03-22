<div align="center">

# 🏗️ ML End-to-End Pipeline

**A production-grade machine learning pipeline — from EDA to deployment with experiment tracking**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API](#-api-endpoints)

</div>

---

## 📌 Overview

A **complete, production-ready ML pipeline** demonstrating best practices for building, training, evaluating, and deploying machine learning models. Features automated pipelines, experiment tracking with MLflow, model serving via FastAPI, and Docker containerization.

### Use Case: Customer Churn Prediction

Predicts whether a telecom customer will churn based on usage patterns, demographics, and service interactions — a common industry ML problem.

---

## ✨ Features

- 📊 **Automated EDA** — Statistical analysis, visualizations, and feature importance
- 🔧 **Feature Engineering** — Automated feature pipelines with Scikit-learn
- 🏋️ **Multi-Model Training** — Random Forest, XGBoost, LightGBM with hyperparameter tuning
- 📈 **MLflow Tracking** — Full experiment tracking, model registry, and versioning
- 🎯 **Evaluation Dashboard** — Precision, recall, F1, ROC-AUC, confusion matrix
- 🚀 **FastAPI Serving** — REST API for real-time predictions with input validation
- 🐳 **Docker** — Containerized deployment with Docker Compose
- ✅ **CI/CD** — Automated testing and linting with GitHub Actions
- 📝 **Comprehensive Notebooks** — Step-by-step EDA and training notebooks

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                 ML Pipeline                          │
│                                                     │
│  ┌─────────┐   ┌──────────┐   ┌─────────────────┐  │
│  │  Data    │──▶│ Feature  │──▶│   Training      │  │
│  │ Ingestion│   │ Pipeline │   │   & Evaluation  │  │
│  └─────────┘   └──────────┘   └────────┬────────┘  │
│                                        │            │
│                                        ▼            │
│  ┌─────────────────────────────────────────────┐    │
│  │              MLflow Tracking                │    │
│  │  Experiments │ Metrics │ Artifacts │ Models │    │
│  └─────────────────────────────────────────────┘    │
│                        │                            │
│                        ▼                            │
│  ┌─────────────────────────────────────────────┐    │
│  │           FastAPI Model Serving             │    │
│  │     /predict  │  /health  │  /model-info    │    │
│  └─────────────────────────────────────────────┘    │
│                        │                            │
│                        ▼                            │
│  ┌─────────────────────────────────────────────┐    │
│  │             Docker Container                │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/altafpinjari2001/ml-end-to-end-pipeline.git
cd ml-end-to-end-pipeline

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Train a model
python src/train.py --config configs/xgboost.yaml

# Start the API
uvicorn src.api.app:app --reload

# Or use Docker
docker compose up
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Get churn prediction for a customer |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Model metadata and version |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "monthly_charges": 79.85,
    "total_charges": 1990.5,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic"
  }'
```

---

## 📁 Project Structure

```
ml-end-to-end-pipeline/
├── src/
│   ├── __init__.py
│   ├── train.py               # Training pipeline entry point
│   ├── data/
│   │   ├── ingestion.py       # Data loading & validation
│   │   └── preprocessing.py   # Feature engineering
│   ├── models/
│   │   ├── trainer.py         # Multi-model training
│   │   └── evaluation.py      # Metrics & visualization
│   └── api/
│       ├── app.py             # FastAPI application
│       ├── schemas.py         # Pydantic request/response models
│       └── middleware.py      # Logging, CORS, rate limiting
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_training.ipynb
├── configs/
│   └── xgboost.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── tests/
├── .github/workflows/ci.yml
├── LICENSE
└── .gitignore
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

<div align="center"><b>⭐ Star this repo if you find it useful!</b></div>
