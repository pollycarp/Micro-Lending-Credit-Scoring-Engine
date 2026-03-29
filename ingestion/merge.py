"""
Data integration layer — pulls from all three sources and returns
two clean DataFrames ready for feature engineering.

Public API
----------
load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]
    Returns (merchants_df, transactions_df)

    merchants_df  : 1 row per merchant — profile + bureau score + default label
    transactions_df : 1 row per (merchant, day) — raw daily financials
"""

import pandas as pd

from ingestion.mongo_client    import get_merchants
from ingestion.postgres_client import get_transactions
from ingestion.bureau_client   import get_bureau_scores


def load_raw_data(
    merchant_ids: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch and merge data from MongoDB, PostgreSQL, and the Bureau API.

    Parameters
    ----------
    merchant_ids : list of str, optional
        Subset of merchants to load. Loads all 1,000 if None.
    verbose : bool
        Print progress messages.

    Returns
    -------
    merchants_df : pd.DataFrame
        Columns: merchant_id, business_name, business_type,
                 business_age_months, location, owner_age, loan_amount,
                 loan_purpose, existing_loan_count, previous_late_payments,
                 loan_utilization_ratio, credit_score, risk_band, default
        Shape: (n_merchants, 14)

    transactions_df : pd.DataFrame
        Columns: merchant_id, date, revenue, expenses, net_cash_flow
        Shape: (n_merchants × 730, 5)
    """

    # ── 1. Merchant profiles (MongoDB) ────────────────────────────────────────
    if verbose:
        print("[ 1/3 ] Fetching merchant profiles from MongoDB …")
    merchants_df = get_merchants()

    if merchant_ids is not None:
        merchants_df = merchants_df[merchants_df["merchant_id"].isin(merchant_ids)].copy()

    ids = merchants_df["merchant_id"].tolist()
    if verbose:
        print(f"        {len(ids):,} merchants loaded.")

    # ── 2. Transaction history (PostgreSQL) ───────────────────────────────────
    if verbose:
        print("[ 2/3 ] Fetching transaction history from PostgreSQL …")
    transactions_df = get_transactions(merchant_ids=ids)
    if verbose:
        print(f"        {len(transactions_df):,} transaction rows loaded.")

    # ── 3. Bureau scores (HTTP API) ───────────────────────────────────────────
    if verbose:
        print("[ 3/3 ] Fetching bureau scores from mock API …")
    bureau_df = get_bureau_scores(ids)
    if verbose:
        print(f"        {len(bureau_df):,} bureau scores received.")

    # ── 4. Merge merchants + bureau (both are per-merchant) ───────────────────
    merchants_df = merchants_df.merge(bureau_df, on="merchant_id", how="left")

    # ── 5. Sanity checks ──────────────────────────────────────────────────────
    missing_bureau = merchants_df["credit_score"].isna().sum()
    if missing_bureau:
        print(f"  [WARN] {missing_bureau} merchants are missing a bureau score.")

    if verbose:
        print("\n── Summary ──────────────────────────────────────────────────")
        print(f"  Merchants DataFrame : {merchants_df.shape}")
        print(f"  Transactions DataFrame: {transactions_df.shape}")
        print(f"  Default rate          : {merchants_df['default'].mean():.1%}")
        print(f"  Credit score range    : "
              f"{merchants_df['credit_score'].min():.0f} – "
              f"{merchants_df['credit_score'].max():.0f}")
        print("─────────────────────────────────────────────────────────────\n")

    return merchants_df, transactions_df


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    merchants, transactions = load_raw_data()

    print("Merchants sample:")
    print(merchants[["merchant_id", "business_type", "credit_score",
                      "risk_band", "default"]].head(5).to_string())

    print("\nTransactions sample:")
    print(transactions.head(5).to_string())
