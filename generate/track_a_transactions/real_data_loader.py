"""
Track A - Real IEEE-CIS dataset loader.

Maps the real Kaggle IEEE-CIS Fraud Detection dataset onto the exact same
schema used by synthetic_baseline.py (FEATURE_COLUMNS + isFraud), so nothing
downstream (adversarial_generator.py, train_and_evaluate.py, track_a_loop.py)
needs to change. This is a drop-in replacement.

Setup before running this (do this on your machine, not here):
    1. Go to https://www.kaggle.com/c/ieee-fraud-detection/rules
       and click "I Understand and Accept" (required once, or the API
       download will 403 even with valid credentials).
    2. kaggle.com/settings -> "Create New API Token" -> downloads kaggle.json
    3. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    4. pip install kaggle --break-system-packages
    5. kaggle competitions download -c ieee-fraud-detection
    6. unzip ieee-fraud-detection.zip -d ieee_data/

Then run this file, or import load_real_data() wherever synthetic_baseline's
generate_baseline() was being called.
"""

import pandas as pd


def load_real_data(data_dir: str = "ieee_data") -> pd.DataFrame:
    txn = pd.read_csv(f"{data_dir}/train_transaction.csv")
    identity = pd.read_csv(f"{data_dir}/train_identity.csv")
    df = txn.merge(identity, on="TransactionID", how="left")

    out = pd.DataFrame()
    out["TransactionAmt"] = df["TransactionAmt"]

    # TransactionDT is seconds elapsed from an unspecified reference point
    # (not a real timestamp). Modulo by seconds-in-a-day is the standard
    # community approach to approximate hour-of-day from it -- sanity-check
    # the resulting distribution once you have the real data (should look
    # roughly diurnal, not uniform).
    out["TransactionHour"] = (df["TransactionDT"] % 86400) // 3600

    # card6 is literally 'debit'/'credit' in the real data.
    out["card_type"] = (df["card6"] == "credit").astype(int)

    out["ProductCD"] = df["ProductCD"].astype("category").cat.codes

    out["email_match"] = (
        (df["P_emaildomain"] == df["R_emaildomain"]).fillna(False).astype(int)
    )

    # DeviceType only exists for rows that have identity data (~24% of rows
    # in this dataset) -- missing rows default to 0 (treated as mobile/unknown).
    out["device_type"] = (df["DeviceType"] == "desktop").fillna(0).astype(int)

    out["distance"] = df["dist1"].fillna(df["dist2"]).fillna(0)

    for i in range(1, 6):
        out[f"C{i}"] = df[f"C{i}"].fillna(0)

    for i in range(1, 4):
        out[f"D{i}"] = df[f"D{i}"].fillna(0)

    out["isFraud"] = df["isFraud"]
    return out


if __name__ == "__main__":
    df = load_real_data()
    print("Fraud rate:", df["isFraud"].mean())
    print(df.groupby("isFraud").size())
    print(df.head())
    df.to_csv("real_baseline_transactions.csv", index=False)
    print("Saved real_baseline_transactions.csv")
