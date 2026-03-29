"""
Transaction feature engineering.

Takes the raw transactions DataFrame (730,000 rows, one per merchant-day)
and collapses it into one row per merchant with 13 meaningful ML features.

Features computed
-----------------
avg_daily_revenue_30d       : Mean daily revenue, last 30 days
avg_daily_revenue_90d       : Mean daily revenue, last 90 days
revenue_volatility_90d      : Std-dev of daily revenue, last 90 days (risk signal)
revenue_growth_rate         : (avg last-90d revenue − avg first-90d revenue)
                              / avg first-90d revenue  — captures trend direction
avg_net_cash_flow_90d       : Mean net cash flow, last 90 days
negative_cashflow_days_ratio: Fraction of ALL days with negative cash flow
neg_cashflow_ratio_90d      : Same, but only last 90 days (recent stress)
expense_ratio               : Mean(expenses / revenue) over full period
cash_flow_coverage          : Mean(revenue) / Mean(expenses) — solvency proxy
revenue_trend_slope         : OLS slope of daily revenue over time, normalised
                              by mean revenue (dimensionless trend rate)
max_consecutive_neg_days    : Longest streak of consecutive negative-CF days
avg_daily_expenses_90d      : Mean daily expenses, last 90 days
revenue_to_loan_ratio       : avg_daily_revenue_90d × 30 / loan_amount
                              (monthly revenue as fraction of loan — repayment capacity)
"""

import numpy as np
import pandas as pd


def _ols_slope(values: np.ndarray) -> float:
    """Return the OLS slope of values against a 0…n-1 index."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = values - values.mean()
    denom = (x * x).sum()
    return float((x * y).sum() / denom) if denom != 0 else 0.0


def _max_consecutive_negative(series: pd.Series) -> int:
    """Length of the longest run of negative values in a Series."""
    max_run = cur_run = 0
    for v in series:
        if v < 0:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def build_transaction_features(
    transactions_df: pd.DataFrame,
    merchants_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate daily transactions into per-merchant features.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Raw transactions with columns:
        merchant_id, date, revenue, expenses, net_cash_flow
    merchants_df : pd.DataFrame
        Merchant profiles — used only to pull loan_amount for the ratio feature.

    Returns
    -------
    pd.DataFrame
        One row per merchant, indexed by merchant_id, with 13 feature columns.
    """
    df = transactions_df.sort_values(["merchant_id", "date"]).copy()

    # Reference date = last date in the dataset
    max_date = df["date"].max()
    cutoff_30  = max_date - pd.Timedelta(days=30)
    cutoff_90  = max_date - pd.Timedelta(days=90)
    cutoff_first_90 = df["date"].min() + pd.Timedelta(days=90)

    loan_map = merchants_df.set_index("merchant_id")["loan_amount"].to_dict()

    records = []
    for mid, grp in df.groupby("merchant_id"):
        grp = grp.sort_values("date")

        last_30  = grp[grp["date"] > cutoff_30]
        last_90  = grp[grp["date"] > cutoff_90]
        first_90 = grp[grp["date"] <= cutoff_first_90]

        # ── windowed revenue averages ──────────────────────────────────────
        avg_rev_30 = last_30["revenue"].mean()  if len(last_30)  else np.nan
        avg_rev_90 = last_90["revenue"].mean()  if len(last_90)  else np.nan
        avg_rev_first90 = first_90["revenue"].mean() if len(first_90) else np.nan

        # ── revenue growth rate ────────────────────────────────────────────
        if avg_rev_first90 and avg_rev_first90 > 0:
            growth_rate = (avg_rev_90 - avg_rev_first90) / avg_rev_first90
        else:
            growth_rate = np.nan

        # ── volatility (coefficient of variation is scale-invariant) ──────
        rev_vol_90 = last_90["revenue"].std() if len(last_90) > 1 else np.nan

        # ── cash-flow features ─────────────────────────────────────────────
        avg_ncf_90 = last_90["net_cash_flow"].mean() if len(last_90) else np.nan

        neg_ratio_all = (grp["net_cash_flow"] < 0).mean()
        neg_ratio_90  = (last_90["net_cash_flow"] < 0).mean() if len(last_90) else np.nan

        # ── expense ratio (avoid div-by-zero on zero-revenue days) ─────────
        safe_rev = grp["revenue"].replace(0, np.nan)
        expense_ratio = (grp["expenses"] / safe_rev).mean()

        # ── cash-flow coverage ratio ───────────────────────────────────────
        mean_exp = grp["expenses"].mean()
        mean_rev = grp["revenue"].mean()
        cf_coverage = mean_rev / mean_exp if mean_exp > 0 else np.nan

        # ── revenue trend slope (normalised) ──────────────────────────────
        slope = _ols_slope(grp["revenue"].values)
        norm_slope = slope / mean_rev if mean_rev > 0 else 0.0

        # ── longest negative cash-flow streak ─────────────────────────────
        max_neg_streak = _max_consecutive_negative(grp["net_cash_flow"])

        # ── expenses (last 90 days) ────────────────────────────────────────
        avg_exp_90 = last_90["expenses"].mean() if len(last_90) else np.nan

        # ── repayment capacity ratio ───────────────────────────────────────
        loan_amount = loan_map.get(mid, np.nan)
        if loan_amount and loan_amount > 0 and not np.isnan(avg_rev_90):
            rev_to_loan = (avg_rev_90 * 30) / loan_amount
        else:
            rev_to_loan = np.nan

        records.append({
            "merchant_id":                mid,
            "avg_daily_revenue_30d":      round(avg_rev_30,   2),
            "avg_daily_revenue_90d":      round(avg_rev_90,   2),
            "revenue_volatility_90d":     round(rev_vol_90,   2),
            "revenue_growth_rate":        round(growth_rate,  4),
            "avg_net_cash_flow_90d":      round(avg_ncf_90,   2),
            "negative_cashflow_days_ratio": round(neg_ratio_all, 4),
            "neg_cashflow_ratio_90d":     round(neg_ratio_90, 4),
            "expense_ratio":              round(expense_ratio, 4),
            "cash_flow_coverage":         round(cf_coverage,  4),
            "revenue_trend_slope":        round(norm_slope,   6),
            "max_consecutive_neg_days":   max_neg_streak,
            "avg_daily_expenses_90d":     round(avg_exp_90,   2),
            "revenue_to_loan_ratio":      round(rev_to_loan,  4),
        })

    return pd.DataFrame(records).set_index("merchant_id")


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion.merge import load_raw_data
    merchants, transactions = load_raw_data(verbose=False)

    print("Building transaction features …")
    feat = build_transaction_features(transactions, merchants)
    print(f"Shape: {feat.shape}")
    print(feat.head(3).to_string())

    # Show that defaulters have weaker metrics on average
    merged = feat.join(merchants.set_index("merchant_id")["default"])
    print("\nMean feature values by default status:")
    print(merged.groupby("default")[
        ["revenue_growth_rate", "neg_cashflow_ratio_90d",
         "cash_flow_coverage", "revenue_trend_slope"]
    ].mean().round(4).to_string())
