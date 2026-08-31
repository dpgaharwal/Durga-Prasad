"""Track A - synthetic fallback generator (used only if deploy_subsample.csv is missing)."""

import numpy as np
import pandas as pd
from real_data_loader import FEATURE_COLUMNS


def generate_baseline(n_legit: int = 20000, n_fraud: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def base_frame(n, is_fraud: bool):
        if not is_fraud:
            frame = pd.DataFrame({
                "TransactionAmt": rng.gamma(shape=2.0, scale=45, size=n),
                "TransactionHour": rng.normal(loc=14, scale=4, size=n).clip(0, 23),
                "card_type": rng.choice([0, 1], size=n, p=[0.55, 0.45]),
                "ProductCD": rng.choice([0, 1, 2, 3, 4], size=n),
                "email_match": rng.choice([0, 1], size=n, p=[0.15, 0.85]),
                "device_type": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
                "distance": np.abs(rng.normal(loc=15, scale=20, size=n)),
            })
            c_scale, d_loc, m_p = 1.0, 100, [0.1, 0.1, 0.8]
        else:
            frame = pd.DataFrame({
                "TransactionAmt": rng.gamma(shape=1.5, scale=140, size=n),
                "TransactionHour": rng.choice(list(range(0, 6)) + list(range(22, 24)), size=n),
                "card_type": rng.choice([0, 1], size=n, p=[0.3, 0.7]),
                "ProductCD": rng.choice([0, 1, 2, 3, 4], size=n),
                "email_match": rng.choice([0, 1], size=n, p=[0.75, 0.25]),
                "device_type": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
                "distance": np.abs(rng.normal(loc=80, scale=60, size=n)),
            })
            c_scale, d_loc, m_p = 3.5, 8, [0.5, 0.05, 0.45]

        frame["card1"] = rng.integers(1000, 18000, size=n)
        frame["card2"] = rng.integers(100, 600, size=n)
        frame["card3"] = rng.choice([150, 185], size=n)
        frame["card5"] = rng.integers(100, 240, size=n)
        frame["addr1"] = rng.integers(100, 500, size=n)
        frame["addr2"] = rng.choice([87], size=n)

        for i in range(1, 15):
            frame[f"C{i}"] = rng.poisson(c_scale * (0.3 + 0.05 * i), size=n)
        for i in range(1, 16):
            frame[f"D{i}"] = np.abs(rng.normal(loc=d_loc * (1 + 0.1 * i), scale=d_loc * 0.5, size=n))
        for i in range(1, 10):
            frame[f"M{i}"] = rng.choice([0, 1, -1], size=n, p=m_p)

        frame["isFraud"] = int(is_fraud)
        return frame

    legit = base_frame(n_legit, is_fraud=False)
    fraud = base_frame(n_fraud, is_fraud=True)
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df[FEATURE_COLUMNS + ["isFraud"]]
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
