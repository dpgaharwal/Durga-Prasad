"""Track D - disagreement screening defense."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from train_and_evaluate import FEATURES

PROVENANCE_TRUST = {
    "analyst_confirmed": 1.0,
    "customer_dispute": 0.6,
    "no_dispute_default": 0.3,
}


def provenance_weights(df: pd.DataFrame, default: float = 0.5) -> np.ndarray:
    if "label_source" not in df.columns:
        return np.ones(len(df))
    return df["label_source"].map(PROVENANCE_TRUST).fillna(default).to_numpy()


def disagreement_screen(feedback_df: pd.DataFrame, base_model, suspicion_threshold: float = 0.05):
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
    if "true_label" not in feedback_df.columns:
        return {}
    is_poison = feedback_df["true_label"] != feedback_df["isFraud"]
    total_poison = int(is_poison.sum())
    clean_rows = int((~is_poison).sum())

    if len(quarantined_df) == 0:
        caught, false_quarantine = 0, 0
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

    for mode, src in [("mode_a_silent_fraud", "no_dispute_default"), ("mode_b_dispute_flood", "customer_dispute")]:
        mode_total = int(((feedback_df["label_source"] == src) & is_poison).sum())
        if len(quarantined_df) and mode_total:
            mode_caught = int(((quarantined_df["label_source"] == src) &
                               (quarantined_df["true_label"] != quarantined_df["isFraud"])).sum())
        else:
            mode_caught = 0
        out[mode] = f"{mode_caught}/{mode_total}" + (f" ({mode_caught / mode_total:.0%})" if mode_total else "")

    return out
