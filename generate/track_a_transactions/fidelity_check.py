"""
Track A - Fidelity metrics.

Judges score "fidelity of attacks in simulation" explicitly. This module
reports how statistically close the generated adversarial fraud is to the
legit population (high closeness = high evasion fidelity = the attack is
doing its job) using KS-test per feature and Population Stability Index.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def ks_report(reference_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: list) -> pd.DataFrame:
    rows = []
    for col in columns:
        stat, pvalue = ks_2samp(reference_df[col], synthetic_df[col])
        rows.append({"feature": col, "ks_statistic": round(stat, 4), "p_value": round(pvalue, 4)})
    return pd.DataFrame(rows).sort_values("ks_statistic")


def psi(reference: pd.Series, synthetic: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. <0.1 = stable, 0.1-0.25 = moderate shift, >0.25 = major shift."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    syn_counts, _ = np.histogram(synthetic, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    syn_pct = np.clip(syn_counts / max(syn_counts.sum(), 1), 1e-4, None)

    return float(np.sum((syn_pct - ref_pct) * np.log(syn_pct / ref_pct)))


if __name__ == "__main__":
    from synthetic_baseline import generate_baseline
    from adversarial_generator import generate_adversarial_fraud

    df = generate_baseline()
    legit = df[df.isFraud == 0]
    fraud = df[df.isFraud == 1]
    adv = generate_adversarial_fraud(fraud, legit, evasion_strength=0.6)

    cont_cols = ["TransactionAmt", "TransactionHour", "distance", "C1", "D1"]

    print("=== KS-test: raw fraud vs legit (should be far apart) ===")
    print(ks_report(legit, fraud, cont_cols))

    print("\n=== KS-test: adversarial fraud vs legit (should be much closer = high fidelity) ===")
    print(ks_report(legit, adv, cont_cols))

    print("\n=== PSI: TransactionAmt, adversarial vs legit ===")
    print(round(psi(legit["TransactionAmt"], adv["TransactionAmt"]), 4))
