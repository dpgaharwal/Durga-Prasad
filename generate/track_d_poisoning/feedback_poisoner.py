"""
Track D2 - Feedback-loop poisoning attack.

THREAT MODEL (this is the part that makes the track novel -- read it before the code)
------------------------------------------------------------------------------------
Production fraud detection systems retrain continuously on labels that come
back from the field. Those labels are NOT hand-curated ground truth. In real
payment operations they are derived, and the derivation rule is the attack
surface:

    label = FRAUD   if a chargeback/dispute was filed
    label = LEGIT   if no dispute was filed within the dispute window

That second rule is a *default assumption*, not an observation. An attacker who
controls both ends of a transaction (their own mule accounts, their own
merchant) simply never files a dispute -- so their fraud is silently labeled
LEGIT and fed back into the next training cycle.

Two poisoning modes implemented here:

  MODE A - "silent fraud" / false-negative seeding (the dangerous one)
    Attacker pushes fraud transactions confined to a narrow feature region
    (their "signature"), never disputes them, and they auto-label as LEGIT.
    Over cycles the model carves out a blind spot exactly where the attacker
    operates. Crucially, AGGREGATE metrics barely move -- overall AUC/recall
    look healthy, so nobody notices. The damage is targeted, not global.
    This is a backdoor attack, not a degradation attack.

  MODE B - "dispute flooding"
    Attacker files false disputes on legitimate transactions, poisoning them
    to FRAUD. Pushes the false-positive rate up, degrades approval rates, and
    erodes trust in the model. Noisier and easier to spot than Mode A.

Why this matters for THIS challenge specifically: our own Track A closed loop
(missed attacks -> retrain) is itself the vulnerability being attacked here.
We red-team our own blue team.

NOTE ON NOVELTY (be honest about this in the writeup): poisoning attacks against
fraud detection systems exist in the literature (e.g. "Fraud Detection under
Siege", ACM TOPS 2023; FRAUD-RLA, arXiv 2502.02290). What is under-explored is
the label-derivation rule as the injection vector, and building the closed-loop
defense around label provenance. Claim that, not invention of the concept.
"""

import numpy as np
import pandas as pd


# The attacker's operating niche. All poisoned fraud is confined here so the
# resulting blind spot is targeted and aggregate metrics stay clean.
ATTACKER_SIGNATURE = {
    "card_type": 1,
    "ProductCD": 2,
    "amount_range": (150, 250),
    "TransactionHour_range": (14, 16),
}


def stamp_signature(df: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Force transactions into the attacker's operating region."""
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
    """Boolean mask: which rows fall inside the attacker's operating region."""
    lo, hi = ATTACKER_SIGNATURE["amount_range"]
    hlo, hhi = ATTACKER_SIGNATURE["TransactionHour_range"]
    return (
        (df["card_type"] == ATTACKER_SIGNATURE["card_type"])
        & (df["ProductCD"] == ATTACKER_SIGNATURE["ProductCD"])
        & (df["TransactionAmt"].between(lo, hi))
        & (df["TransactionHour"].between(hlo, hhi))
    )


def generate_silent_fraud(fraud_pool: pd.DataFrame, n: int = 600, seed: int = 11) -> pd.DataFrame:
    """
    MODE A. Fraud transactions inside the attacker's signature region, mislabeled
    LEGIT because no dispute was ever filed (attacker controls both ends).

    `true_label` is kept alongside for evaluation only -- the defense never
    sees it, the same way a real institution never sees it at label time.
    """
    rng = np.random.default_rng(seed)
    sampled = fraud_pool.sample(n=n, replace=True, random_state=seed)
    poisoned = stamp_signature(sampled, seed=seed)

    # Blend count/delta features slightly toward normal so rows are inliers,
    # not outliers -- poisoned samples that look anomalous get caught by any
    # basic data-quality check.
    for col in ["C1", "C2", "C3", "C4", "C5"]:
        poisoned[col] = (poisoned[col] * 0.35).round() + rng.poisson(1.0, size=len(poisoned))
    for col in ["D1", "D2", "D3"]:
        poisoned[col] = poisoned[col] * 0.5 + np.abs(rng.normal(60, 30, size=len(poisoned)))

    poisoned["true_label"] = 1              # actually fraud
    poisoned["isFraud"] = 0                 # but labeled LEGIT (no dispute filed)
    poisoned["label_source"] = "no_dispute_default"
    poisoned["attack_id"] = [f"track_d_silent_{i}" for i in range(len(poisoned))]
    return poisoned


def generate_dispute_flood(legit_pool: pd.DataFrame, n: int = 300, seed: int = 12) -> pd.DataFrame:
    """
    MODE B. Legitimate transactions falsely disputed by the attacker, poisoning
    them to a FRAUD label. Drives false positives up on genuine traffic.
    """
    sampled = legit_pool.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    poisoned = sampled.copy()
    poisoned["true_label"] = 0              # actually legit
    poisoned["isFraud"] = 1                 # but labeled FRAUD (false dispute)
    poisoned["label_source"] = "customer_dispute"
    poisoned["attack_id"] = [f"track_d_dispute_{i}" for i in range(len(poisoned))]
    return poisoned


def build_poisoned_feedback(
    clean_feedback: pd.DataFrame,
    fraud_pool: pd.DataFrame,
    legit_pool: pd.DataFrame,
    n_silent: int = 600,
    n_dispute: int = 300,
    seed: int = 11,
) -> pd.DataFrame:
    """
    Mix attacker-poisoned rows into an otherwise-honest feedback batch, the way
    they would arrive in a real retraining cycle.
    """
    clean = clean_feedback.copy()
    clean["true_label"] = clean["isFraud"]
    clean["label_source"] = "analyst_confirmed"
    if "attack_id" not in clean.columns:
        clean["attack_id"] = None

    silent = generate_silent_fraud(fraud_pool, n=n_silent, seed=seed)
    dispute = generate_dispute_flood(legit_pool, n=n_dispute, seed=seed + 1)

    batch = pd.concat([clean, silent, dispute], ignore_index=True)
    return batch.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "../track_a_transactions"))
    from synthetic_baseline import generate_baseline

    df = generate_baseline()
    legit = df[df.isFraud == 0]
    fraud = df[df.isFraud == 1]

    silent = generate_silent_fraud(fraud, n=600)
    dispute = generate_dispute_flood(legit, n=300)

    print("MODE A (silent fraud) - labeled legit, actually fraud:")
    print(silent[["TransactionAmt", "TransactionHour", "device_type",
                  "ProductCD", "isFraud", "true_label", "label_source"]].head())
    print(f"\nMODE B (dispute flood): {len(dispute)} legit txns mislabeled as fraud")
    print(f"\nSignature region match rate in silent batch: {matches_signature(silent).mean():.2f}")
