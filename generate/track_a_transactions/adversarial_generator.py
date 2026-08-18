"""
Track A - Adversarial attack generator.

Simulates GenAI-assisted fraud: takes real fraud examples and perturbs their
features so they sit closer to the legit distribution's decision boundary,
evading a naively-trained baseline classifier while keeping isFraud=1 as
ground truth (we control the label because we generated the attack).
"""

import numpy as np
import pandas as pd

from real_data_loader import FEATURE_COLUMNS

# Card/address identifiers are excluded from blending -- an attacker can't
# plausibly "blend" a card fingerprint toward the legit population, that
# would mean literally using a different card. Blending applies to behavioral/
# timing/amount signals, which is what a GenAI fraud pipeline actually can
# tune to look normal.
_ID_COLS = {"card1", "card2", "card3", "card5", "addr1", "addr2"}
CONTINUOUS_COLS = [c for c in FEATURE_COLUMNS
                    if c not in _ID_COLS
                    and not c.startswith("M")
                    and c not in ("card_type", "ProductCD", "email_match", "device_type")]
CATEGORICAL_COLS = ["card_type", "ProductCD", "email_match", "device_type"]
# M-flags (verification match results) are left untouched -- these come from
# the issuer/network side of the transaction, not something a fraud pipeline
# generating the purchase-side data can directly manipulate.


def generate_adversarial_fraud(
    fraud_df: pd.DataFrame,
    legit_df: pd.DataFrame,
    evasion_strength: float = 0.6,
    seed: int = 7,
) -> pd.DataFrame:
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


if __name__ == "__main__":
    from synthetic_baseline import generate_baseline

    df = generate_baseline()
    legit = df[df.isFraud == 0]
    fraud = df[df.isFraud == 1]

    adv = generate_adversarial_fraud(fraud, legit, evasion_strength=0.6)
    print(f"Generated {len(adv)} adversarial fraud examples")
    print(f"Blended {len(CONTINUOUS_COLS)} continuous + {len(CATEGORICAL_COLS)} categorical features")
