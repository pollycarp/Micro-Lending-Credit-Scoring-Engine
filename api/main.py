"""
Credit Scoring API
------------------
FastAPI application that serves probability-of-default predictions.

Endpoints
---------
GET  /health              — liveness check
GET  /model/info          — current model name and version
GET  /predict/{merchant_id} — PD score + risk tier + SHAP explanation

Run locally
-----------
  uvicorn api.main:app --reload --port 8000

Then open:
  http://localhost:8000/docs   ← interactive Swagger UI (try it in browser)
  http://localhost:8000/redoc  ← alternative docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import Predictor
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

# ── startup / shutdown ─────────────────────────────────────────────────────────

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts, release on shutdown."""
    global predictor
    print("Loading model …")
    predictor = Predictor()
    print("Model ready. API is live.")
    yield
    print("Shutting down.")


# ── app ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Micro-Lending Credit Scoring API",
    description = (
        "Predicts the probability of default (PD) for small business loan "
        "applicants using a trained ML pipeline backed by MongoDB, PostgreSQL, "
        "and a mock credit bureau."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET"],
    allow_headers  = ["*"],
)


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — confirms the API and model are ready."""
    return HealthResponse(
        status       = "ok",
        model_loaded = predictor is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    """Returns the name and version of the currently loaded model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return ModelInfoResponse(**predictor.model_info())


@app.get(
    "/predict/{merchant_id}",
    response_model = PredictionResponse,
    tags           = ["Scoring"],
    summary        = "Score a loan applicant",
    response_description = "PD score, risk tier, and top SHAP feature explanations",
)
def predict(merchant_id: str):
    """
    Predict the probability of default for an existing merchant.

    - **merchant_id**: e.g. `M0001`, `M0750`
    - Returns a PD score between 0 (very safe) and 1 (very risky)
    - Risk tiers: **low** < 0.25 ≤ **medium** < 0.50 ≤ **high**
    - `top_features` shows which factors most influenced this prediction
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        return predictor.predict(merchant_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
