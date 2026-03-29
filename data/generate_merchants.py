"""
Generate synthetic merchant profiles and save them to MongoDB.

Architecture: Latent Variable Design
-------------------------------------
Each merchant has a hidden creditworthiness score z ~ N(0, 1).
  - Higher z  → more creditworthy (lower default risk)
  - Lower  z  → less creditworthy (higher default risk)

All observable features (late payments, utilisation, bureau score) are
generated as INDEPENDENT noisy functions of z.  The default label is
also a noisy function of z.

Crucially, no feature is derived directly from the default label.
This prevents data leakage and produces realistic model AUC (0.80–0.88).

MongoDB fields
--------------
  All merchant profile fields  (used as features)
  bureau_score                 (pre-computed; served by mock_bureau_api)
  _credit_latent               (hidden; never exposed to the ML model)
  default                      (target label)
"""

import hashlib
import random
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "lending_db")

BUSINESS_TYPES = ["retail", "food_beverage", "services", "agriculture", "manufacturing", "transport"]
LOCATIONS      = ["Nairobi", "Mombasa", "Kampala", "Dar es Salaam", "Kigali",
                  "Addis Ababa", "Lagos", "Accra"]
LOAN_PURPOSES  = ["inventory", "equipment", "working_capital", "expansion", "payroll", "marketing"]
NAME_PREFIXES  = ["Quick", "Premier", "Star", "Global", "Local", "City", "Fresh", "Pro"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _bureau_score(merchant_id: str, z: float) -> int:
    """
    Compute a deterministic bureau credit score as a noisy function of z.
    Uses merchant_id as RNG seed so the same merchant always gets the same score.
    Score is NOT derived from the default label — only from z + independent noise.
    """
    seed = int(hashlib.md5(merchant_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng  = np.random.RandomState(seed)
    # Base: 600 + 75*z maps N(0,1) → roughly (450, 750) at ±2 std
    # Independent noise std=65 creates realistic overlap between risk groups
    raw = rng.normal(600 + 75 * z, 65)
    return int(np.clip(raw, 300, 850))


# ── generation ─────────────────────────────────────────────────────────────────

def generate_merchants(n: int = 1000, seed: int = 42) -> list[dict]:
    np.random.seed(seed)
    random.seed(seed)

    merchants = []
    for i in range(n):
        merchant_id = f"M{i+1:04d}"
        btype       = random.choice(BUSINESS_TYPES)

        # ── latent creditworthiness ────────────────────────────────────────
        z = float(np.random.normal(0, 1))

        # ── observable features — each a noisy function of z ──────────────

        # Business age: higher z → more established
        base_age = max(1, int(np.random.exponential(max(6.0, 24.0 + 14.0 * z))))
        business_age_months = min(base_age, 240)

        # Late payments: higher z → fewer late payments (Poisson rate decreases)
        late_rate = max(0.05, 1.8 - 0.65 * z)
        previous_late_payments = min(int(np.random.poisson(late_rate)), 8)

        # Loan utilisation: higher z → lower utilisation
        mean_util = _sigmoid(-0.65 * z)           # ranges ~0.2 – 0.8
        a = max(0.5, mean_util * 4)
        b = max(0.5, (1 - mean_util) * 4)
        loan_utilization_ratio = round(float(np.clip(np.random.beta(a, b), 0.01, 0.99)), 3)

        # Existing loans: slightly more for less creditworthy merchants
        existing_loan_count = min(int(np.random.poisson(max(0.2, 1.5 - 0.3 * z))), 5)

        # Loan amount: independent of creditworthiness (requested by applicant)
        loan_amount = int(np.clip(np.random.lognormal(10.5, 0.8), 5_000, 200_000))

        # ── default label — noisy function of z ───────────────────────────
        # P(default) ≈ 0.25 at z=0, ~0.08 at z=2, ~0.55 at z=-2
        # Irreducible noise (std=0.15) ensures no feature set can predict perfectly
        raw_prob     = np.random.normal(0.25 - 0.175 * z, 0.15)
        default_prob = float(np.clip(raw_prob, 0.02, 0.95))
        default      = int(np.random.binomial(1, default_prob))

        # ── bureau score — separate noisy function of z (NOT of default) ──
        bureau_score = _bureau_score(merchant_id, z)

        merchants.append({
            "merchant_id":            merchant_id,
            "business_name":          f"{random.choice(NAME_PREFIXES)} {btype.replace('_',' ').title()} {i+1}",
            "business_type":          btype,
            "business_age_months":    business_age_months,
            "location":               random.choice(LOCATIONS),
            "owner_age":              int(np.clip(np.random.normal(38, 10), 18, 75)),
            "loan_amount":            loan_amount,
            "loan_purpose":           random.choice(LOAN_PURPOSES),
            "existing_loan_count":    existing_loan_count,
            "previous_late_payments": previous_late_payments,
            "loan_utilization_ratio": loan_utilization_ratio,
            "bureau_score":           bureau_score,   # served by mock bureau API
            "_credit_latent":         round(z, 4),    # hidden — never a model feature
            "default":                default,
            "created_at":             datetime.utcnow(),
        })

    return merchants


# ── persistence ────────────────────────────────────────────────────────────────

def save_to_mongodb(merchants: list[dict]) -> None:
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    col    = db["merchants"]

    col.drop()
    col.insert_many(merchants)
    col.create_index("merchant_id", unique=True)

    defaults = sum(1 for m in merchants if m["default"] == 1)
    print(f"Inserted {len(merchants):,} merchants")
    print(f"Default rate: {defaults/len(merchants)*100:.1f}%  ({defaults} defaults)")
    client.close()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating merchant profiles …")
    merchants = generate_merchants(n=1000)
    print("Saving to MongoDB …")
    save_to_mongodb(merchants)
    print("Done.")
