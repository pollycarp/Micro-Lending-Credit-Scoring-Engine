# Micro-Lending Credit Scoring Engine

A production-style machine learning system that predicts the **probability of default (PD)** for small-business micro-loan applicants.
Built as a portfolio project demonstrating end-to-end ML engineering across data ingestion, feature engineering, model training, explainability, API serving, and interactive dashboarding.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start (Local)](#quick-start-local)
- [Quick Start (Docker)](#quick-start-docker)
- [Training the Model](#training-the-model)
- [API Reference](#api-reference)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Explainability (SHAP)](#explainability-shap)
- [Environment Variables](#environment-variables)

---

## Overview

The engine ingests merchant profiles from **MongoDB** and transaction histories from **PostgreSQL**, engineers 20+ risk features, and trains three competing ML pipelines (XGBoost, LightGBM, Logistic Regression). The best model is registered in the **MLflow Model Registry** and served through a **FastAPI** REST endpoint. A **Streamlit** dashboard lets analysts score individual merchants, explore portfolio risk, and inspect model performance — all without writing code.

Key capabilities:
- **Dual-database feature store** — merchant-level features (MongoDB) combined with aggregated transaction signals (PostgreSQL)
- **Mock credit bureau API** — realistic external data enrichment via a local Flask service
- **Automated model selection** — 5-fold cross-validation + hold-out test evaluation, best model promoted to `@champion` alias
- **Explainable predictions** — per-merchant SHAP waterfall charts and global feature importance
- **Fully containerised** — one `docker-compose up` command starts all five services

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                           │
│  MongoDB (merchant profiles)   PostgreSQL (transactions)    │
│              │                          │                   │
│              └──────────┬───────────────┘                   │
│                         │                                   │
│               Mock Credit Bureau API (Flask)                │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Feature Store                            │
│   transaction_features.py   merchant_features.py           │
│              └──────────┬───────────────┘                   │
│                 feature_store.py  (build_feature_matrix)    │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Training Pipeline                          │
│   XGBoost  │  LightGBM  │  Logistic Regression             │
│            └──── MLflow tracking & model registry ────┘     │
└─────────────────────────┼───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
┌─────────▼──────────┐         ┌──────────▼──────────┐
│   FastAPI          │         │  Streamlit Dashboard │
│   /predict/{id}    │         │  Score · Portfolio   │
│   /health          │         │  Model Performance   │
│   /model/info      │         └─────────────────────┘
└────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Merchant profiles | MongoDB 7.0 |
| Transaction history | PostgreSQL 16 |
| Feature engineering | pandas, NumPy, SciPy |
| ML models | XGBoost, LightGBM, scikit-learn |
| Explainability | SHAP |
| Hyperparameter tuning | Hyperopt |
| Experiment tracking | MLflow 2.13 |
| REST API | FastAPI + Uvicorn |
| Mock bureau | Flask |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Language | Python 3.11+ |

---

## Project Structure

```
.
├── api/                    # FastAPI application
│   ├── main.py             # Routes: /health, /model/info, /predict/{id}
│   ├── predictor.py        # Model loading and inference logic
│   └── schemas.py          # Pydantic request/response models
│
├── dashboard/
│   └── app.py              # Streamlit dashboard (3 pages)
│
├── data/                   # Synthetic data generation & seeding scripts
│
├── features/               # Feature engineering
│   ├── feature_store.py    # build_feature_matrix() — main entry point
│   ├── merchant_features.py
│   └── transaction_features.py
│
├── ingestion/              # Database connectors
│
├── models/                 # Model training helpers
│   ├── train.py            # temporal_split()
│   ├── evaluate.py         # metrics, ROC/PR data helpers
│   └── saved/              # best_model.joblib (created after training)
│
├── pipeline/               # Scikit-learn pipelines + MLflow training
│   ├── pipeline.py         # get_all_pipelines()
│   ├── transformers.py     # Custom sklearn transformers
│   └── mlflow_train.py     # Experiment runner — entry point
│
├── mlruns/                 # MLflow experiment artifacts (auto-generated)
├── notebooks/              # Exploratory notebooks
├── tests/                  # Test suite
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start (Local)

### 1. Prerequisites

- Python 3.11+
- Docker Desktop (for MongoDB + PostgreSQL)
- Git

### 2. Clone & install dependencies

```bash
git clone <repo-url>
cd "Micro-Lending Credit Scoring Engine"

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your database credentials if needed
```

### 4. Start databases

```bash
docker-compose up postgres mongodb -d
```

### 5. Seed the databases

```bash
python data/run_all.py
```

### 6. Start the mock credit bureau

```bash
python data/mock_bureau_api.py
```

### 7. Train the model

```bash
python -m pipeline.mlflow_train
```

This trains XGBoost, LightGBM, and Logistic Regression, logs all runs to MLflow, promotes the best model to `@champion`, and saves `models/saved/best_model.joblib`.

### 8. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 9. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Quick Start (Docker)

> **Prerequisite:** Train the model locally first (step 7 above) so that `models/saved/best_model.joblib` and `mlruns/` exist. Docker mounts these as read-only volumes.

```bash
docker-compose up --build
```

This starts all five services:

| Service | URL |
|---|---|
| FastAPI scoring API | http://localhost:8000 |
| FastAPI docs (Swagger) | http://localhost:8000/docs |
| Mock credit bureau | http://localhost:5001 |
| PostgreSQL | localhost:5433 |
| MongoDB | localhost:27017 |

> The Streamlit dashboard is designed for local development. Run it separately with `streamlit run dashboard/app.py`.

---

## Training the Model

```bash
python -m pipeline.mlflow_train
```

**What happens:**
1. Builds the full feature matrix from MongoDB + PostgreSQL + bureau API
2. Applies a temporal train/test split (last 300 merchants held out)
3. Trains three pipelines with 5-fold cross-validation
4. Logs parameters, metrics, and model artifacts to MLflow
5. Registers the best model as `CreditScoringModel @champion`
6. Saves `models/saved/best_model.joblib` for Docker inference

**View all runs in the MLflow UI:**

```bash
mlflow ui
# Open http://localhost:5000
```

---

## API Reference

### `GET /health`
Liveness check.

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model/info`
Returns the name and version of the loaded model.

```json
{ "model_name": "XGBoost", "version": "1" }
```

### `GET /predict/{merchant_id}`
Returns a PD score, risk tier, and top SHAP feature explanations.

```bash
curl http://localhost:8000/predict/M0001
```

```json
{
  "merchant_id": "M0001",
  "pd_score": 0.1732,
  "risk_tier": "low",
  "top_features": [
    { "feature": "avg_txn_amount", "shap_value": -0.42 },
    { "feature": "days_since_last_txn", "shap_value": 0.18 }
  ]
}
```

**Risk tiers:**

| Tier | PD Score |
|---|---|
| Low | < 0.25 |
| Medium | 0.25 – 0.50 |
| High | > 0.50 |

Interactive docs available at `http://localhost:8000/docs`.

---

## Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Three pages:

| Page | Description |
|---|---|
| **Score a Merchant** | Enter a merchant ID → PD score gauge + SHAP bar chart |
| **Portfolio Overview** | Risk tier distribution pie, PD histogram, actual vs predicted box plot |
| **Model Performance** | ROC curve, Precision-Recall curve, calibration plot, global SHAP importance |

---

## Explainability (SHAP)

Every prediction is explained using SHAP (SHapley Additive exPlanations):

- **Per-merchant:** top 10 features that pushed the score up or down
- **Global:** mean absolute SHAP across 200 test-set samples, shown in the dashboard

SHAP values are computed using `TreeExplainer` for XGBoost/LightGBM and coefficient-weighted inputs for Logistic Regression.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values:

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `lending_db` | MongoDB database name |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5433` | PostgreSQL port (host-side) |
| `POSTGRES_DB` | `lending_db` | PostgreSQL database name |
| `POSTGRES_USER` | `lending_user` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `lending_pass` | PostgreSQL password |
| `BUREAU_API_URL` | `http://localhost:5001` | Mock credit bureau base URL |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow tracking directory |
