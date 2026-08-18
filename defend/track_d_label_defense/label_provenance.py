"""
Track D2 - Defense against feedback-loop poisoning.

Three layers. Layer 2 is the one that actually does the work; layers 1 and 3
make it precise and explainable.

  LAYER 1 - Label provenance trust
    Not all labels are equally trustworthy. A label an analyst confirmed is
    worth more than a label inferred from "nobody disputed it within 90 days."
    Every feedback row carries a provenance tag and a trust score.

  LAYER 2 - Disagreement screening  (PRIMARY)
    Take the model trained BEFORE this feedback batch arrived. Ask it to score
    every incoming row. Where the model strongly says FRAUD but the supplied
    label says LEGIT, something is wrong -- either the model is broken, or the
    label is. Combine that disagreement with the label's provenance trust:

        suspicion = disagreement x (1 - trust)

    This formula is the crux of the defense, because it separates the two cases
    that look identical on the surface:

      * A genuinely-missed fraud that an analyst later confirmed
        -> high disagreement, but trust = 1.0, so suspicion = 0.
           KEPT. This is exactly the hard example Track A's loop wants.

      * Attacker's silent fraud auto-labeled legit by the no-dispute rule
        -> high disagreement, trust = 0.3, so suspicion is high.
           QUARANTINED.

    Without provenance, these are indistinguishable and any disagreement-based
    filter would throw away the useful hard examples along with the poison.

  LAYER 3 - Cluster concentration diagnostic
    Poisoned batches come from an attack template, so they concentrate in a
    narrow feature region rather than spreading like honest feedback. Clustering
    the quarantined rows surfaces the attacker's operating signature, which is
    what a fraud analyst actually needs to write a rule -- not just a count.

WHY NOT leave-cluster-out validation against a trusted holdout: we implemented
and tested that approach first and it caught nothing. A targeted backdoor
creates a blind spot in a narrow region; if the trusted holdout does not happen
to contain fraud from that same region, including the poison does not degrade
holdout recall at all, so the screen stays silent. Documented deliberately --
it is a good finding for the writeup, and it generalises: validation-based
poison detection only sees poison in regions the validation set covers.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROVENANCE_TRUST = {
    "analyst_confirmed": 1.0,      # human investigator confirmed the outcome
    "customer_dispute": 0.6,       # cardholder filed a dispute (abusable)
    "no_dispute_default": 0.3,     # nobody complained, so we assumed legit
}

FEATURES = [
    "TransactionAmt", "TransactionHour", "card_type", "ProductCD",
    "email_match", "device_type", "distance",
    "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3",
]


def provenance_weights(df: pd.DataFrame, default: float = 0.5) -> np.ndarray:
    """LAYER 1: map each row's label_source to a trust score in [0, 1]."""
    if "label_source" not in df.columns:
        return np.ones(len(df))
    return df["label_source"].map(PROVENANCE_TRUST).fillna(default).to_numpy()


def disagreement_screen(
    feedback_df: pd.DataFrame,
    base_model,
    suspicion_threshold: float = 0.45,
):
    """
    LAYER 2: quarantine rows where a pre-poisoning model strongly disagrees with
    a low-trust label.

    Returns (accepted_df, quarantined_df, scored_df).
    """
    fb = feedback_df.copy().reset_index(drop=True)

    p_fraud = base_model.predict_proba(fb[FEATURES])[:, 1]
    trust = provenance_weights(fb)

    fb["_p_fraud"] = p_fraud
    fb["_trust"] = trust
    fb["_disagreement"] = np.abs(p_fraud - fb["isFraud"].to_numpy())
    fb["_suspicion"] = fb["_disagreement"] * (1.0 - fb["_trust"])

    flagged = fb["_suspicion"] >= suspicion_threshold
    internal = ["_p_fraud", "_trust", "_disagreement", "_suspicion"]

    accepted = fb[~flagged].drop(columns=internal)
    quarantined = fb[flagged].drop(columns=internal)
    return accepted, quarantined, fb


def cluster_signature(quarantined_df: pd.DataFrame, n_clusters: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    LAYER 3: profile the quarantined rows so an analyst can see the attacker's
    operating region, not just a row count.
    """
    if len(quarantined_df) < n_clusters * 2:
        return pd.DataFrame()

    q = quarantined_df.copy().reset_index(drop=True)
    X = q[FEATURES].to_numpy()
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    q["_cluster"] = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X_std)

    profile_cols = ["TransactionAmt", "TransactionHour", "device_type", "ProductCD"]
    prof = q.groupby("_cluster")[profile_cols].mean().round(2)
    prof["n_rows"] = q.groupby("_cluster").size()
    return prof.reset_index()


def poison_recall(quarantined_df: pd.DataFrame, feedback_df: pd.DataFrame) -> dict:
    """
    Evaluation only. Uses `true_label` vs `isFraud` mismatch as ground truth --
    the defense never sees true_label at runtime, exactly as a real institution
    never sees it at label time.
    """
    if "true_label" not in feedback_df.columns:
        return {}

    is_poison = feedback_df["true_label"] != feedback_df["isFraud"]
    total_poison = int(is_poison.sum())
    clean_rows = int((~is_poison).sum())

    if len(quarantined_df) == 0:
        caught = 0
        false_quarantine = 0
    else:
        q_poison = quarantined_df["true_label"] != quarantined_df["isFraud"]
        caught = int(q_poison.sum())
        false_quarantine = int((~q_poison).sum())

    out = {
        "total_poison_in_batch": total_poison,
        "poison_quarantined": caught,
        "poison_catch_rate": round(caught / total_poison, 4) if total_poison else None,
        "clean_rows_wrongly_quarantined": false_quarantine,
        "clean_quarantine_rate": round(false_quarantine / clean_rows, 4) if clean_rows else None,
    }

    # Per-mode breakdown. Mode B (dispute flood) is far easier to catch than
    # Mode A (silent fraud) -- worth showing separately rather than hiding
    # behind one blended number.
    for mode, src in [("mode_a_silent_fraud", "no_dispute_default"),
                      ("mode_b_dispute_flood", "customer_dispute")]:
        mode_total = int(((feedback_df["label_source"] == src) & is_poison).sum())
        if len(quarantined_df) and mode_total:
            mode_caught = int(((quarantined_df["label_source"] == src) &
                               (quarantined_df["true_label"] != quarantined_df["isFraud"])).sum())
        else:
            mode_caught = 0
        out[mode] = f"{mode_caught}/{mode_total}" + (
            f" ({mode_caught / mode_total:.0%})" if mode_total else "")

    return out
