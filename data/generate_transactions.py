"""
Generate 2 years of daily transaction history for every merchant
and save the rows to PostgreSQL.

Schema: transactions(id, merchant_id, date, revenue, expenses, net_cash_flow)

Revenue trends are driven by the merchant's latent creditworthiness (_credit_latent),
NOT by the default label directly.  This avoids leakage while still producing
meaningful patterns the model can learn from.
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
    port     = int(os.getenv("POSTGRES_PORT", 5433)),
    dbname   = os.getenv("POSTGRES_DB",   "lending_db"),
    user     = os.getenv("POSTGRES_USER",  "lending_user"),
    password = os.getenv("POSTGRES_PASSWORD", "lending_pass"),
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "lending_db")

START_DATE = date(2022, 1, 1)
END_DATE   = date(2023, 12, 31)


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
        weekly = np.where(is_wknd, 1.25, 1.0)
    else:
        weekly = np.where(is_wknd, 0.65, 1.0)

    # Trend driven by latent creditworthiness z (NOT by default label)
    # z > 0.5  → healthy growth
    # z ≈ 0    → flat
    # z < -0.5 → mild late-period decline
    # z < -1.5 → steeper decline
    z = merchant.get("_credit_latent", 0.0)
    trend = np.ones(n)
    if z > 0.5:
        # Growth: scales with creditworthiness
        end_mult = 1.0 + min(0.18, 0.07 * z)
        trend = np.linspace(1.0, end_mult, n)
    elif z < -1.5:
        # Steeper decline in final quarter
        flat_end = int(n * 0.72)
        trend[flat_end:] = np.linspace(1.0, 0.82, n - flat_end)
    elif z < -0.5:
        # Mild decline
        flat_end = int(n * 0.80)
        trend[flat_end:] = np.linspace(1.0, 0.93, n - flat_end)
    # else: flat trend (z between -0.5 and 0.5)

    # High noise buries the signal (std=0.38 is deliberately large)
    noise   = rng.lognormal(0, 0.38, n)
    revenue = base * seasonality * weekly * trend * noise
    return np.maximum(revenue, 0.0).round(2)


def _expenses(revenue: np.ndarray, merchant: dict, rng: np.random.RandomState) -> np.ndarray:
    n = len(revenue)
    # Expense ratio correlated with z: less creditworthy → higher ratio
    # Distributions overlap substantially
    z = merchant.get("_credit_latent", 0.0)
    mean_ratio = np.clip(0.73 - 0.07 * z, 0.55, 0.92)
    ratio = rng.normal(mean_ratio, 0.10, n)
    ratio = np.clip(ratio, 0.40, 1.05)

    fixed    = merchant["loan_amount"] * 0.0008
    expenses = revenue * ratio + fixed
    return np.maximum(expenses, 0.0).round(2)


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


if __name__ == "__main__":
    print("Fetching merchants from MongoDB …")
    mc = MongoClient(MONGO_URI)
    merchants = list(
        mc[MONGO_DB]["merchants"].find(
            {},
            {"merchant_id": 1, "business_type": 1, "_credit_latent": 1,
             "loan_amount": 1, "_id": 0}
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
        rng = np.random.RandomState(idx)
        rev = _revenue(merchant, dates, rng)
        exp = _expenses(rev, merchant, rng)
        net = (rev - exp).round(2)

        rows = [
            (merchant["merchant_id"], dates[j], float(rev[j]), float(exp[j]), float(net[j]))
            for j in range(len(dates))
        ]
        insert_batch(conn, rows)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1:>4}/{len(merchants)} merchants done …")

    conn.close()
    print("\nAll transactions inserted. Done.")
