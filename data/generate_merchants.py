"""
Generate synthetic merchant profiles and save them to MongoDB.
Produces 1,000 merchants with realistic attributes and a default label
that is correlated with the risk features (not random noise).
"""

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


# ── default probability ────────────────────────────────────────────────────────

def _default_probability(m: dict) -> float:
    """
    Rule-based probability that captures real lending risk drivers:
    - young businesses fail more often
    - high loan utilisation is a stress signal
    - prior late payments show payment discipline
    - larger loans are harder to service for micro businesses
    """
    p = 0.12  # base default rate ~12 %

    # Business age
    age = m["business_age_months"]
    if age < 12:
        p += 0.15
    elif age < 24:
        p += 0.08
    elif age > 60:
        p -= 0.05

    # Payment history
    p += m["previous_late_payments"] * 0.07

    # Utilisation
    util = m["loan_utilization_ratio"]
    if util > 0.80:
        p += 0.12
    elif util > 0.60:
        p += 0.06

    # Business type risk premium
    type_risk = {
        "retail": 0.02, "food_beverage": 0.03, "services": -0.02,
        "agriculture": 0.05, "manufacturing": 0.01, "transport": 0.04,
    }
    p += type_risk.get(m["business_type"], 0)

    # Loan size
    if m["loan_amount"] > 80_000:
        p += 0.06

    # Existing debt burden
    p += m["existing_loan_count"] * 0.04

    return float(np.clip(p, 0.02, 0.95))


# ── generation ─────────────────────────────────────────────────────────────────

def generate_merchants(n: int = 1000, seed: int = 42) -> list[dict]:
    np.random.seed(seed)
    random.seed(seed)

    merchants = []
    for i in range(n):
        btype = random.choice(BUSINESS_TYPES)
        age   = int(np.clip(np.random.exponential(36) + 1, 1, 240))

        m = {
            "merchant_id":            f"M{i+1:04d}",
            "business_name":          f"{random.choice(NAME_PREFIXES)} {btype.replace('_',' ').title()} {i+1}",
            "business_type":          btype,
            "business_age_months":    age,
            "location":               random.choice(LOCATIONS),
            "owner_age":              int(np.clip(np.random.normal(38, 10), 18, 75)),
            "loan_amount":            int(np.clip(np.random.lognormal(10.5, 0.8), 5_000, 200_000)),
            "loan_purpose":           random.choice(LOAN_PURPOSES),
            "existing_loan_count":    int(min(np.random.poisson(1.2), 5)),
            "previous_late_payments": int(min(np.random.poisson(0.8), 8)),
            "loan_utilization_ratio": round(float(np.random.beta(2, 3)), 3),
        }

        m["default"]    = int(np.random.binomial(1, _default_probability(m)))
        m["created_at"] = datetime.utcnow()
        merchants.append(m)

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
