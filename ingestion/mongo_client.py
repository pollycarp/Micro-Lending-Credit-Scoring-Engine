"""
MongoDB client — fetches merchant profiles.

Public API
----------
get_merchants() -> pd.DataFrame
    Returns one row per merchant with all profile fields.
    The internal MongoDB _id and created_at fields are excluded.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "lending_db")

# Fields to exclude from the returned DataFrame
# _credit_latent is a hidden field — never expose it to the ML model
# bureau_score is served through the bureau API (ingestion/bureau_client.py)
_EXCLUDE = {"_id": 0, "created_at": 0, "_credit_latent": 0, "bureau_score": 0}


def get_merchants() -> pd.DataFrame:
    """
    Fetch all merchant profiles from MongoDB.

    Returns
    -------
    pd.DataFrame
        Columns: merchant_id, business_name, business_type, business_age_months,
                 location, owner_age, loan_amount, loan_purpose,
                 existing_loan_count, previous_late_payments,
                 loan_utilization_ratio, default
    """
    client = MongoClient(MONGO_URI)
    try:
        docs = list(client[MONGO_DB]["merchants"].find({}, _EXCLUDE))
    finally:
        client.close()

    if not docs:
        raise RuntimeError("No merchants found in MongoDB. Run data/generate_merchants.py first.")

    df = pd.DataFrame(docs)
    return df


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = get_merchants()
    print(f"Shape: {df.shape}")
    print(df.head(3).to_string())
    print(f"\nDefault rate: {df['default'].mean():.1%}")
