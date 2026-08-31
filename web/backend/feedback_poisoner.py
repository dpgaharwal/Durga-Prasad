"""Track D - feedback-loop poisoning attack generator."""

import numpy as np
import pandas as pd

# card_type (not device_type) -- device_type defaults to 0 for ~76% of real
# IEEE-CIS rows (missing identity data), so it doesn't make a narrow signature.
ATTACKER_SIGNATURE = {
    "card_type": 1,
    "ProductCD": 2,
    "amount_range": (150, 250),
    "TransactionHour_range": (14, 16),
}


def stamp_signature(df: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    out["card_type"] = ATTACKER_SIGNATURE["card_type"]
    out["ProductCD"] = ATTACKER_SIGNATURE["ProductCD"]
    lo, hi = ATTACKER_SIGNATURE["amount_range"]
    out["TransactionAmt"] = rng.uniform(lo, hi, size=len(out))
    hlo, hhi = ATTACKER_SIGNATURE["TransactionHour_range"]
    out["TransactionHour"] = rng.uniform(hlo, hhi, size=len(out))
    return out


def matches_signature(df: pd.DataFrame) -> pd.Series:
    lo, hi = ATTACKER_SIGNATURE["amount_range"]
    hlo, hhi = ATTACKER_SIGNATURE["TransactionHour_range"]
    return (
        (df["card_type"] == ATTACKER_SIGNATURE["card_type"])
        & (df["ProductCD"] == ATTACKER_SIGNATURE["ProductCD"])
        & (df["TransactionAmt"].between(lo, hi))
        & (df["TransactionHour"].between(hlo, hhi))
    )


def generate_silent_fraud(fraud_pool: pd.DataFrame, n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = fraud_pool.sample(n=n, replace=True, random_state=seed)
    poisoned = stamp_signature(sampled, seed=seed)

    for col in ["C1", "C2", "C3", "C4", "C5"]:
        poisoned[col] = (poisoned[col] * 0.35).round() + rng.poisson(1.0, size=len(poisoned))
    for col in ["D1", "D2", "D3"]:
        poisoned[col] = poisoned[col] * 0.5 + np.abs(rng.normal(60, 30, size=len(poisoned)))

    poisoned["true_label"] = 1
    poisoned["isFraud"] = 0
    poisoned["label_source"] = "no_dispute_default"
    poisoned["attack_id"] = [f"track_d_silent_{i}" for i in range(len(poisoned))]
    return poisoned


def generate_dispute_flood(legit_pool: pd.DataFrame, n: int = 300, seed: int = 12) -> pd.DataFrame:
    sampled = legit_pool.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    poisoned = sampled.copy()
    poisoned["true_label"] = 0
    poisoned["isFraud"] = 1
    poisoned["label_source"] = "customer_dispute"
    poisoned["attack_id"] = [f"track_d_dispute_{i}" for i in range(len(poisoned))]
    return poisoned


def build_poisoned_feedback(clean_feedback, fraud_pool, legit_pool, n_silent=600, n_dispute=300, seed=11):
    clean = clean_feedback.copy()
    clean["true_label"] = clean["isFraud"]
    clean["label_source"] = "analyst_confirmed"
    if "attack_id" not in clean.columns:
        clean["attack_id"] = None

    silent = generate_silent_fraud(fraud_pool, n=n_silent, seed=seed)
    dispute = generate_dispute_flood(legit_pool, n=n_dispute, seed=seed + 1)

    batch = pd.concat([clean, silent, dispute], ignore_index=True)
    return batch.sample(frac=1.0, random_state=seed).reset_index(drop=True)
