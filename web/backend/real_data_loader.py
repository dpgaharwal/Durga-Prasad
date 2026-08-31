"""
Track A - Real IEEE-CIS dataset loader.

Maps the real Kaggle IEEE-CIS Fraud Detection dataset onto an expanded
51-feature schema (card fingerprint, address, full C/D velocity+timing
features, verification match flags -- not just the original 15).
"""

import pandas as pd

FEATURE_COLUMNS = (
    ["TransactionAmt", "TransactionHour", "card_type", "ProductCD",
     "email_match", "device_type", "distance"]
    + ["card1", "card2", "card3", "card5", "addr1", "addr2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"M{i}" for i in range(1, 10)]
)


def load_real_data(data_dir: str = "ieee_data") -> pd.DataFrame:
    txn = pd.read_csv(f"{data_dir}/train_transaction.csv")
    identity = pd.read_csv(f"{data_dir}/train_identity.csv")
    df = txn.merge(identity, on="TransactionID", how="left")

    out = pd.DataFrame()
    out["TransactionAmt"] = df["TransactionAmt"]
    out["TransactionHour"] = (df["TransactionDT"] % 86400) // 3600
    out["card_type"] = (df["card6"] == "credit").astype(int)
    out["ProductCD"] = df["ProductCD"].astype("category").cat.codes
    out["email_match"] = (
        (df["P_emaildomain"] == df["R_emaildomain"]).fillna(False).astype(int)
    )
    out["device_type"] = (df["DeviceType"] == "desktop").fillna(0).astype(int)
    out["distance"] = df["dist1"].fillna(df["dist2"]).fillna(0)

    for col in ["card1", "card2", "card3", "card5"]:
        out[col] = df[col].fillna(-1)

    out["addr1"] = df["addr1"].fillna(-1)
    out["addr2"] = df["addr2"].fillna(-1)

    for i in range(1, 15):
        out[f"C{i}"] = df[f"C{i}"].fillna(0)
    for i in range(1, 16):
        out[f"D{i}"] = df[f"D{i}"].fillna(0)

    for i in range(1, 10):
        col = f"M{i}"
        if col in df.columns:
            out[col] = df[col].astype("category").cat.codes
        else:
            out[col] = -1

    out["isFraud"] = df["isFraud"]
    return out
