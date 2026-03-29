"""
Mock Credit Bureau API
----------------------
Returns a pre-computed credit score for any merchant.

The score is stored in MongoDB during data generation as a noisy function
of the merchant's latent creditworthiness (z) — NOT derived from the
default label.  This prevents data leakage.

Endpoints:
  GET /credit-score/<merchant_id>   → credit score + risk band
  GET /health                       → liveness check

Run with:
  python data/mock_bureau_api.py
"""

import os

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, abort
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "lending_db")

app = Flask(__name__)


def _risk_band(score: int) -> str:
    if score < 580:   return "poor"
    if score < 670:   return "fair"
    if score < 740:   return "good"
    if score < 800:   return "very_good"
    return "exceptional"


@app.route("/credit-score/<merchant_id>", methods=["GET"])
def credit_score(merchant_id: str):
    client   = MongoClient(MONGO_URI)
    merchant = client[MONGO_DB]["merchants"].find_one(
        {"merchant_id": merchant_id},
        {"bureau_score": 1, "_id": 0},
    )
    client.close()

    if merchant is None:
        abort(404, description=f"Merchant '{merchant_id}' not found.")

    score = merchant["bureau_score"]
    return jsonify({
        "merchant_id":  merchant_id,
        "credit_score": score,
        "risk_band":    _risk_band(score),
        "bureau":       "SimulatedCreditBureau",
        "score_range":  {"min": 300, "max": 850},
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Mock Bureau API running on http://localhost:5001")
    print("Try: http://localhost:5001/credit-score/M0001")
    app.run(host="0.0.0.0", port=5001, debug=False)
