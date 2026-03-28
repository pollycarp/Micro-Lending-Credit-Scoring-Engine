"""
Mock Credit Bureau API
----------------------
Simulates an external credit bureau that returns a credit score for a merchant.

Endpoints:
  GET /credit-score/<merchant_id>   → credit score + risk band
  GET /health                       → liveness check

Run with:
  python data/mock_bureau_api.py

The score is:
  - Deterministic (same merchant always gets the same score).
  - Correlated with the default flag stored in MongoDB
    (defaulters skew toward lower scores, but not perfectly — realistic noise).
"""

import hashlib
import os

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, abort
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "lending_db")

app = Flask(__name__)


# ── score generation ───────────────────────────────────────────────────────────

def _score(merchant_id: str, default_flag: int) -> tuple[int, str]:
    """
    Returns (credit_score, risk_band).
    Uses merchant_id as the RNG seed so the score is deterministic across calls.
    """
    seed = int(hashlib.md5(merchant_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng  = np.random.RandomState(seed)

    if default_flag == 1:
        raw = int(rng.normal(480, 80))   # defaulters cluster 300-650
    else:
        raw = int(rng.normal(680, 80))   # good borrowers cluster 550-850

    score = int(np.clip(raw, 300, 850))

    if score < 580:
        band = "poor"
    elif score < 670:
        band = "fair"
    elif score < 740:
        band = "good"
    elif score < 800:
        band = "very_good"
    else:
        band = "exceptional"

    return score, band


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/credit-score/<merchant_id>", methods=["GET"])
def credit_score(merchant_id: str):
    client   = MongoClient(MONGO_URI)
    merchant = client[MONGO_DB]["merchants"].find_one(
        {"merchant_id": merchant_id}, {"default": 1, "_id": 0}
    )
    client.close()

    if merchant is None:
        abort(404, description=f"Merchant '{merchant_id}' not found.")

    score, band = _score(merchant_id, merchant["default"])

    return jsonify({
        "merchant_id":  merchant_id,
        "credit_score": score,
        "risk_band":    band,
        "bureau":       "SimulatedCreditBureau",
        "score_range":  {"min": 300, "max": 850},
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Mock Bureau API running on http://localhost:5001")
    print("Try: http://localhost:5001/credit-score/M0001")
    app.run(host="0.0.0.0", port=5001, debug=False)
