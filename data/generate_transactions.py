"""
Generate 2 years of daily transaction history for every merchant
and save the rows to PostgreSQL.

Schema: transactions(id, merchant_id, date, revenue, expenses, net_cash_flow)

Design choices that make the data realistic:
- Revenue scales with loan_amount (proxy for business size).
- Seasonal + weekly patterns per business type.
- Merchants who default show a gradual revenue decline in the final 40 % of
  the period — this is the signal the model will learn.
- Expense ratios are higher for defaulters (cash-flow squeeze).
"""

from datetime import date, timedelta

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

PG = dict(
    host     = os.getenv("POSTGRES_HOST", "localhost"),
    port     = int(os.getenv("POSTGRES_PORT", 5432)),
    dbname   = os.getenv("POSTGRES_DB",   "lending_db"),
    user     = os.getenv("POSTGRES_USER",  "lending_user"),
    password = os.getenv("POSTGRES_PASSWORD", "lending_pass"),
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "lending_db")

START_DATE = date(2022, 1, 1)
END_DATE   = date(2023, 12, 31)   # 730 days


# ── helpers ────────────────────────────────────────────────────────────────────

def _dates() -> list[date]:
    n = (END_DATE - START_DATE).days + 1
    return [START_DATE + timedelta(days=i) for i in range(n)]


def _revenue(merchant: dict, dates: list[date], rng: np.random.RandomState) -> np.ndarray:
    n    = len(dates)
    base = merchant["loan_amount"] * rng.uniform(0.003, 0.01)

    # Annual seasonality
    doy        = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    seasonality = 1.0 + 0.15 * np.sin(2 * np.pi * doy / 365)

    # Weekly pattern
    dow    = np.array([d.weekday() for d in dates])
    is_wknd = dow >= 5
    if merchant["business_type"] in ("retail", "food_beverage"):
        weekly = np.where(is_wknd, 1.30, 1.0)
    else:
        weekly = np.where(is_wknd, 0.60, 1.0)

    # Long-run trend
    if merchant["default"] == 1:
        decline_start        = int(n * 0.60)
        trend                = np.ones(n)
        trend[decline_start:] = np.linspace(1.0, 0.55, n - decline_start)
    else:
        trend = np.linspace(1.0, 1.15, n)

    noise   = rng.lognormal(0, 0.20, n)
    revenue = base * seasonality * weekly * trend * noise
    return np.maximum(revenue, 0.0).round(2)


def _expenses(revenue: np.ndarray, merchant: dict, rng: np.random.RandomState) -> np.ndarray:
    n = len(revenue)
    if merchant["default"] == 1:
        ratio = rng.uniform(0.75, 0.98, n)
    else:
        ratio = rng.uniform(0.55, 0.80, n)

    fixed    = merchant["loan_amount"] * 0.0008
    expenses = revenue * ratio + fixed
    return np.maximum(expenses, 0.0).round(2)


# ── PostgreSQL setup ───────────────────────────────────────────────────────────

def create_table(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS transactions;")
        cur.execute("""
            CREATE TABLE transactions (
                id            SERIAL PRIMARY KEY,
                merchant_id   VARCHAR(10)    NOT NULL,
                date          DATE           NOT NULL,
                revenue       NUMERIC(12, 2),
                expenses      NUMERIC(12, 2),
                net_cash_flow NUMERIC(12, 2),
                UNIQUE (merchant_id, date)
            );
        """)
        cur.execute("CREATE INDEX idx_tx_merchant ON transactions (merchant_id);")
        cur.execute("CREATE INDEX idx_tx_date     ON transactions (date);")
    conn.commit()
    print("Created transactions table with indexes.")


def insert_batch(conn, rows: list[tuple]) -> None:
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO transactions (merchant_id, date, revenue, expenses, net_cash_flow) VALUES %s",
            rows,
        )
    conn.commit()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load merchant list from MongoDB (only need id, type, default, loan_amount)
    print("Fetching merchants from MongoDB …")
    mc       = MongoClient(MONGO_URI)
    merchants = list(
        mc[MONGO_DB]["merchants"].find(
            {}, {"merchant_id": 1, "business_type": 1, "default": 1, "loan_amount": 1, "_id": 0}
        )
    )
    mc.close()
    print(f"Found {len(merchants):,} merchants.")

    dates = _dates()
    print(f"Date range: {START_DATE} → {END_DATE}  ({len(dates)} days)")
    print(f"Total rows to insert: {len(merchants) * len(dates):,}")

    conn = psycopg2.connect(**PG)
    create_table(conn)

    np.random.seed(42)

    for idx, merchant in enumerate(merchants):
        # Per-merchant RNG so ordering doesn't affect other merchants
        rng      = np.random.RandomState(idx)
        rev      = _revenue(merchant, dates, rng)
        exp      = _expenses(rev, merchant, rng)
        net      = (rev - exp).round(2)

        rows = [
            (merchant["merchant_id"], dates[j], float(rev[j]), float(exp[j]), float(net[j]))
            for j in range(len(dates))
        ]
        insert_batch(conn, rows)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1:>4}/{len(merchants)} merchants done …")

    conn.close()
    print("\nAll transactions inserted. Done.")
