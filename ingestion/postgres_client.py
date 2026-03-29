"""
PostgreSQL client — fetches raw transaction history.

Public API
----------
get_transactions(merchant_ids=None) -> pd.DataFrame
    Returns the full transactions table (or a filtered subset).
    Each row is one merchant on one day.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PORT = os.getenv("POSTGRES_PORT", "5433")
_DB   = os.getenv("POSTGRES_DB",   "lending_db")
_USER = os.getenv("POSTGRES_USER",  "lending_user")
_PASS = os.getenv("POSTGRES_PASSWORD", "lending_pass")

_ENGINE = None   # lazy-initialised so import is cheap


def _engine():
    global _ENGINE
    if _ENGINE is None:
        url     = f"postgresql+psycopg2://{_USER}:{_PASS}@{_HOST}:{_PORT}/{_DB}"
        _ENGINE = create_engine(url, pool_pre_ping=True)
    return _ENGINE


def get_transactions(merchant_ids: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch transaction rows from PostgreSQL.

    Parameters
    ----------
    merchant_ids : list of str, optional
        If provided, only rows for those merchants are returned.
        If None, the entire table is fetched.

    Returns
    -------
    pd.DataFrame
        Columns: merchant_id, date, revenue, expenses, net_cash_flow
    """
    if merchant_ids is not None:
        # Parameterised query — safe against SQL injection
        sql = text("""
            SELECT merchant_id, date, revenue, expenses, net_cash_flow
            FROM   transactions
            WHERE  merchant_id = ANY(:ids)
            ORDER  BY merchant_id, date
        """)
        params = {"ids": merchant_ids}
    else:
        sql = text("""
            SELECT merchant_id, date, revenue, expenses, net_cash_flow
            FROM   transactions
            ORDER  BY merchant_id, date
        """)
        params = {}

    with _engine().connect() as conn:
        df = pd.read_sql(sql, conn, params=params, parse_dates=["date"])

    if df.empty:
        raise RuntimeError("No transactions found. Run data/generate_transactions.py first.")

    return df


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = get_transactions(merchant_ids=["M0001", "M0002"])
    print(f"Shape: {df.shape}")
    print(df.head(5).to_string())
