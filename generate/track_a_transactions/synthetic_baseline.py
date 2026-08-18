"""
Track A - Baseline transaction generator.

Produces a legit + fraud transaction dataset with an IEEE-CIS-Fraud-Detection-
inspired schema (simplified). This is the fallback data source for local dev/
testing without network access.

For the actual submission, swap this out for the real Kaggle dataset:
    kaggle competitions download -c ieee-fraud-detection
and point `load_real_data()` at the downloaded CSVs instead of calling
`generate_baseline()`.
"""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "TransactionAmt",   # transaction amount
    "TransactionHour",  # hour of day, 0-23
    "card_type",        # 0=debit, 1=credit
    "ProductCD",        # 0-4, product category
    "email_match",      # 1 if purchaser/recipient email domain matches, else 0
    "device_type",      # 0=mobile, 1=desktop
    "distance",         # billing-to-shipping distance proxy
    "C1", "C2", "C3", "C4", "C5",   # count-type features (num addresses, cards seen, etc.)
    "D1", "D2", "D3",               # time-delta features (days since last txn, etc.)
]


def generate_baseline(n_legit: int = 20000, n_fraud: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic transaction set with realistic-ish legit vs fraud
    separation, roughly matching IEEE-CIS's ~3.5% fraud prevalence.
    """
    rng = np.random.default_rng(seed)

    legit = pd.DataFrame({
        "TransactionAmt": rng.gamma(shape=2.0, scale=45, size=n_legit),
        "TransactionHour": rng.normal(loc=14, scale=4, size=n_legit).clip(0, 23),
        "card_type": rng.choice([0, 1], size=n_legit, p=[0.55, 0.45]),
        "ProductCD": rng.choice([0, 1, 2, 3, 4], size=n_legit),
        "email_match": rng.choice([0, 1], size=n_legit, p=[0.15, 0.85]),
        "device_type": rng.choice([0, 1], size=n_legit, p=[0.6, 0.4]),
        "distance": np.abs(rng.normal(loc=15, scale=20, size=n_legit)),
        "C1": rng.poisson(1.2, size=n_legit),
        "C2": rng.poisson(1.0, size=n_legit),
        "C3": rng.poisson(0.3, size=n_legit),
        "C4": rng.poisson(0.5, size=n_legit),
        "C5": rng.poisson(0.4, size=n_legit),
        "D1": np.abs(rng.normal(loc=120, scale=90, size=n_legit)),
        "D2": np.abs(rng.normal(loc=100, scale=80, size=n_legit)),
        "D3": np.abs(rng.normal(loc=30, scale=25, size=n_legit)),
    })
    legit["isFraud"] = 0

    # Fraud: higher amounts, odd hours, email mismatch, new device, far distance,
    # high count features (multiple cards/addresses seen), short time-deltas
    # (freshly created / rapid reuse).
    fraud = pd.DataFrame({
        "TransactionAmt": rng.gamma(shape=1.5, scale=140, size=n_fraud),
        "TransactionHour": rng.choice(list(range(0, 6)) + list(range(22, 24)), size=n_fraud),
        "card_type": rng.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
        "ProductCD": rng.choice([0, 1, 2, 3, 4], size=n_fraud),
        "email_match": rng.choice([0, 1], size=n_fraud, p=[0.75, 0.25]),
        "device_type": rng.choice([0, 1], size=n_fraud, p=[0.85, 0.15]),
        "distance": np.abs(rng.normal(loc=80, scale=60, size=n_fraud)),
        "C1": rng.poisson(4.5, size=n_fraud),
        "C2": rng.poisson(4.0, size=n_fraud),
        "C3": rng.poisson(2.5, size=n_fraud),
        "C4": rng.poisson(3.0, size=n_fraud),
        "C5": rng.poisson(2.8, size=n_fraud),
        "D1": np.abs(rng.normal(loc=8, scale=10, size=n_fraud)),
        "D2": np.abs(rng.normal(loc=5, scale=8, size=n_fraud)),
        "D3": np.abs(rng.normal(loc=2, scale=3, size=n_fraud)),
    })
    fraud["isFraud"] = 1

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    return df


def load_real_data(train_transaction_csv: str) -> pd.DataFrame:
    """
    Loader stub for the real Kaggle IEEE-CIS dataset once downloaded.
    Adjust column selection to match FEATURE_COLUMNS or extend the pipeline
    to use the full feature set.
    """
    df = pd.read_csv(train_transaction_csv)
    return df


if __name__ == "__main__":
    df = generate_baseline()
    print(df.groupby("isFraud").size())
    print(df.head())
    df.to_csv("baseline_transactions.csv", index=False)
    print("Saved baseline_transactions.csv")
