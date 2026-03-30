"""
Merchant profile feature engineering.

Transforms the raw merchant + bureau DataFrame into ML-ready features:
  - Numerical fields are kept as-is (scaling happens inside the pipeline).
  - Categorical fields are ordinal-encoded or one-hot encoded.
  - risk_band gets an ordinal mapping (poor < fair < good < …).

Output columns
--------------
Numeric (kept):
    business_age_months, owner_age, loan_amount,
    existing_loan_count, previous_late_payments,
    loan_utilization_ratio, credit_score

Ordinal:
    risk_band_ord   (poor=1 … exceptional=5)

One-hot (drop_first=True to avoid multicollinearity):
    business_type_*   (retail is reference)
    loan_purpose_*    (equipment is reference)
    location_*        (Accra is reference)

Target (passed through, not transformed):
    default
"""

import pandas as pd

# All possible category values — must be declared explicitly so that
# pd.get_dummies always produces the same columns even when predicting
# for a single merchant (who only has one value per category).
ALL_BUSINESS_TYPES = ["agriculture", "food_beverage", "manufacturing",
                      "retail", "services", "transport"]
ALL_LOAN_PURPOSES  = ["equipment", "expansion", "inventory",
                      "marketing", "payroll", "working_capital"]
ALL_LOCATIONS      = ["Accra", "Addis Ababa", "Dar es Salaam", "Kampala",
                      "Kigali", "Lagos", "Mombasa", "Nairobi"]

RISK_BAND_ORDER = {
    "poor":        1,
    "fair":        2,
    "good":        3,
    "very_good":   4,
    "exceptional": 5,
}


def build_merchant_features(merchants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from merchant profiles + bureau scores.

    Parameters
    ----------
    merchants_df : pd.DataFrame
        Output of ingestion.merge.load_raw_data() — one row per merchant,
        already merged with bureau scores.

    Returns
    -------
    pd.DataFrame
        One row per merchant, indexed by merchant_id, containing only
        feature columns + 'default' target.
    """
    df = merchants_df.copy()
    df = df.set_index("merchant_id")

    # ── ordinal encode risk_band ───────────────────────────────────────────
    # Use float so missing values are np.nan (sklearn-compatible),
    # not pd.NA (pandas nullable Int64 which sklearn cannot handle)
    df["risk_band_ord"] = df["risk_band"].map(RISK_BAND_ORDER).astype(float)

    # ── one-hot encode categoricals ───────────────────────────────────────
    # Use pd.Categorical with explicit category lists so that get_dummies
    # always produces the same columns, even for a single-row input.
    df["business_type"] = pd.Categorical(df["business_type"], categories=ALL_BUSINESS_TYPES)
    df["loan_purpose"]  = pd.Categorical(df["loan_purpose"],  categories=ALL_LOAN_PURPOSES)
    df["location"]      = pd.Categorical(df["location"],      categories=ALL_LOCATIONS)

    cat_cols = ["business_type", "loan_purpose", "location"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # ── drop columns not used as features ─────────────────────────────────
    drop_cols = ["business_name", "risk_band"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion.merge import load_raw_data
    merchants, _ = load_raw_data(verbose=False)

    feat = build_merchant_features(merchants)
    print(f"Shape: {feat.shape}")
    print("Columns:", feat.columns.tolist())
    print(feat.head(3).to_string())
