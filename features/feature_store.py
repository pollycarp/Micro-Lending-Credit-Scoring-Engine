"""
Feature store — single entry point that produces the final feature matrix.

Usage
-----
    from features.feature_store import build_feature_matrix

    X, y = build_feature_matrix()
    # X : pd.DataFrame  (1000 rows × ~35 features)
    # y : pd.Series     (1000 labels — 0/1 default)

The function orchestrates:
  1. Raw data ingestion   (ingestion.merge)
  2. Transaction features (features.transaction_features)
  3. Merchant features    (features.merchant_features)
  4. Inner join on merchant_id
  5. Basic data-quality checks
"""

import pandas as pd

from ingestion.merge import load_raw_data
from features.transaction_features import build_transaction_features
from features.merchant_features    import build_merchant_features


def build_feature_matrix(
    merchant_ids: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the complete ML feature matrix.

    Parameters
    ----------
    merchant_ids : list of str, optional
        Restrict to a subset of merchants (useful for inference on new applicants).
        If None, all merchants are included.
    verbose : bool
        Print progress messages.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix — one row per merchant, indexed by merchant_id.
        The 'default' column is NOT included.
    y : pd.Series
        Binary target (0 = repaid, 1 = defaulted), indexed by merchant_id.
    """

    # ── 1. Load raw data ───────────────────────────────────────────────────
    merchants_df, transactions_df = load_raw_data(
        merchant_ids=merchant_ids, verbose=verbose
    )

    # ── 2. Build feature groups ────────────────────────────────────────────
    if verbose:
        print("Building transaction features …")
    tx_features = build_transaction_features(transactions_df, merchants_df)

    if verbose:
        print("Building merchant features …")
    merch_features = build_merchant_features(merchants_df)

    # ── 3. Separate target before joining ─────────────────────────────────
    y = merch_features["default"].rename("default")
    merch_features = merch_features.drop(columns=["default"])

    # ── 4. Join on merchant_id (both are indexed by merchant_id) ──────────
    X = merch_features.join(tx_features, how="inner")

    # Align y to X index (handles any ordering differences)
    y = y.loc[X.index]

    # ── 5. Data-quality checks ─────────────────────────────────────────────
    missing_pct = X.isna().mean()
    high_missing = missing_pct[missing_pct > 0.05]
    if not high_missing.empty:
        print(f"  [WARN] Columns with >5% missing values:\n{high_missing}")

    if verbose:
        print(f"\n── Feature matrix ready ────────────────────────────────────")
        print(f"  X shape      : {X.shape}")
        print(f"  y shape      : {y.shape}")
        print(f"  Default rate : {y.mean():.1%}")
        print(f"  Missing vals : {X.isna().sum().sum()} total cells")
        print(f"  Features     :")
        for col in X.columns:
            print(f"    {col}")
        print(f"────────────────────────────────────────────────────────────\n")

    return X, y


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = build_feature_matrix()

    print("Sample rows (numeric features only):")
    numeric_cols = X.select_dtypes(include="number").columns[:8]
    print(X[numeric_cols].head(5).to_string())

    print(f"\nTarget distribution:\n{y.value_counts().to_string()}")
