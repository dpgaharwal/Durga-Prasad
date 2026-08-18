"""
Track A - Adversarial attack generator.

Simulates GenAI-assisted fraud: takes real fraud examples and perturbs their
features so they sit closer to the legit distribution's decision boundary,
evading a naively-trained baseline classifier while keeping isFraud=1 as
ground truth (we control the label because we generated the attack).

This models how a generative fraud pipeline would craft transactions to look
statistically "normal" -- amount, timing, device signals nudged toward legit
norms while the underlying intent stays fraudulent.
"""

import numpy as np
import pandas as pd

from synthetic_baseline import FEATURE_COLUMNS


def generate_adversarial_fraud(
    fraud_df: pd.DataFrame,
    legit_df: pd.DataFrame,
    evasion_strength: float = 0.6,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Blend each fraud row's continuous/count features toward the legit
    population mean, controlled by `evasion_strength` (0 = no change,
    1 = fully mimics legit distribution). Categorical features are flipped
    toward the legit majority class with probability = evasion_strength.
    """
    rng = np.random.default_rng(seed)
    adv = fraud_df.copy().reset_index(drop=True)

    continuous_cols = ["TransactionAmt", "TransactionHour", "distance",
                        "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3"]
    categorical_cols = ["card_type", "ProductCD", "email_match", "device_type"]

    legit_means = legit_df[continuous_cols].mean()

    for col in continuous_cols:
        noise = rng.normal(loc=0, scale=legit_df[col].std() * 0.1, size=len(adv))
        adv[col] = adv[col] * (1 - evasion_strength) + legit_means[col] * evasion_strength + noise
        adv[col] = adv[col].clip(lower=0)

    for col in categorical_cols:
        legit_mode = legit_df[col].mode().iloc[0]
        flip_mask = rng.random(len(adv)) < evasion_strength
        adv.loc[flip_mask, col] = legit_mode

    adv["isFraud"] = 1  # ground truth stays fraud - we generated it
    adv["attack_id"] = [f"track_a_adv_{i}" for i in range(len(adv))]
    return adv


if __name__ == "__main__":
    from synthetic_baseline import generate_baseline

    df = generate_baseline()
    legit = df[df.isFraud == 0]
    fraud = df[df.isFraud == 1]

    adv = generate_adversarial_fraud(fraud, legit, evasion_strength=0.6)
    adv.to_csv("adversarial_fraud.csv", index=False)
    print(f"Generated {len(adv)} adversarial fraud examples -> adversarial_fraud.csv")
    print(adv[FEATURE_COLUMNS].describe().loc[["mean"]])
