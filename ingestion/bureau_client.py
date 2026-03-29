"""
Mock Credit Bureau client — fetches credit scores via HTTP.

Public API
----------
get_bureau_scores(merchant_ids) -> pd.DataFrame
    Calls the bureau API concurrently for all merchant IDs and returns
    one row per merchant with credit_score and risk_band.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BUREAU_URL   = os.getenv("BUREAU_API_URL", "http://localhost:5001")
_MAX_WORKERS = 20    # concurrent HTTP threads
_TIMEOUT     = 5     # seconds per request


def _fetch_one(merchant_id: str) -> dict:
    """Fetch the bureau score for a single merchant. Returns a dict."""
    url = f"{BUREAU_URL}/credit-score/{merchant_id}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return {
            "merchant_id":  merchant_id,
            "credit_score": data["credit_score"],
            "risk_band":    data["risk_band"],
        }
    except requests.RequestException as exc:
        # Return NaN for this merchant so the pipeline can decide how to handle it
        print(f"  [WARN] Bureau API failed for {merchant_id}: {exc}")
        return {
            "merchant_id":  merchant_id,
            "credit_score": None,
            "risk_band":    None,
        }


def get_bureau_scores(merchant_ids: list[str]) -> pd.DataFrame:
    """
    Fetch credit bureau scores for a list of merchant IDs.

    Requests are made concurrently (up to _MAX_WORKERS threads) to keep
    latency reasonable even for 1,000+ merchants.

    Parameters
    ----------
    merchant_ids : list of str

    Returns
    -------
    pd.DataFrame
        Columns: merchant_id, credit_score (int), risk_band (str)
    """
    results = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, mid): mid for mid in merchant_ids}
        for future in as_completed(futures):
            results.append(future.result())

    df = pd.DataFrame(results)
    # Restore original order
    order = {mid: i for i, mid in enumerate(merchant_ids)}
    df["_order"] = df["merchant_id"].map(order)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    missing = df["credit_score"].isna().sum()
    if missing:
        print(f"  [WARN] {missing} merchants have no bureau score (API errors).")

    return df


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_ids = [f"M{i:04d}" for i in range(1, 6)]
    df = get_bureau_scores(sample_ids)
    print(df.to_string())
