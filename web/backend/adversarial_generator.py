"""Track A - adversarial evasion generator."""

import numpy as np
import pandas as pd
from real_data_loader import FEATURE_COLUMNS

_ID_COLS = {"card1", "card2", "card3", "card5", "addr1", "addr2"}
CONTINUOUS_COLS = [c for c in FEATURE_COLUMNS
                   if c not in _ID_COLS
                   and not c.startswith("M")
                   and c not in ("card_type", "ProductCD", "email_match", "device_type")]
CATEGORICAL_COLS = ["card_type", "ProductCD", "email_match", "device_type"]


def generate_adversarial_fraud(fraud_df, legit_df, evasion_strength: float = 0.6, seed: int = 7):
    rng = np.random.default_rng(seed)
    adv = fraud_df.copy().reset_index(drop=True)
    legit_means = legit_df[CONTINUOUS_COLS].mean()

    for col in CONTINUOUS_COLS:
        noise = rng.normal(loc=0, scale=legit_df[col].std() * 0.1, size=len(adv))
        adv[col] = adv[col] * (1 - evasion_strength) + legit_means[col] * evasion_strength + noise
        adv[col] = adv[col].clip(lower=0)

    for col in CATEGORICAL_COLS:
        legit_mode = legit_df[col].mode().iloc[0]
        flip_mask = rng.random(len(adv)) < evasion_strength
        adv.loc[flip_mask, col] = legit_mode

    adv["isFraud"] = 1
    adv["attack_id"] = [f"track_a_adv_{i}" for i in range(len(adv))]
    return adv
