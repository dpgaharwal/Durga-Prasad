"""
Track A - Real IEEE-CIS dataset loader.

Maps the real Kaggle IEEE-CIS Fraud Detection dataset onto an expanded
51-feature schema. Started at 15 hand-picked features; expanded after real-data
testing showed weak precision (0.14) -- the missing signal was card
fingerprint (card1/2/3/5), address codes (addr1/2), full velocity counts
(C1-C14, we only had C1-C5), full time-deltas (D1-D15, we only had D1-D3),
and verification match flags (M1-M9), all of which are well-documented as
strong predictors for this specific dataset.

Deliberately NOT including the V1-V339 Vesta engineered columns -- that's a
much bigger lever but requires re-architecting the feature list to be dynamic
rather than named, across 4 files. This 51-feature set is the contained,
high-value fix; V-columns are the next lever if there's time left.

Setup before running this (do this on your machine, not here):
    1. https://www.kaggle.com/c/ieee-fraud-detection/rules -> "I Understand and Accept"
    2. kaggle.com/settings -> Create New API Token
    3. mkdir -p ~/.kaggle && echo <token> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
    4. pip install kaggle --break-system-packages
    5. kaggle competitions download -c ieee-fraud-detection
    6. unzip ieee-fraud-detection.zip -d ieee_data/
"""

import pandas as pd

# The full feature list this loader produces -- keep in sync with
# defend/track_a_classifier/train_and_evaluate.py's FEATURES constant, which
# is the canonical copy the classifier actually trains on.
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

    # TransactionDT is seconds elapsed from an unspecified reference point.
    # Modulo by seconds-in-a-day is the standard community approximation of
    # hour-of-day -- sanity-check the distribution, should look diurnal.
    out["TransactionHour"] = (df["TransactionDT"] % 86400) // 3600

    out["card_type"] = (df["card6"] == "credit").astype(int)
    out["ProductCD"] = df["ProductCD"].astype("category").cat.codes

    out["email_match"] = (
        (df["P_emaildomain"] == df["R_emaildomain"]).fillna(False).astype(int)
    )

    # DeviceType only exists for the ~24% of rows with identity data.
    out["device_type"] = (df["DeviceType"] == "desktop").fillna(0).astype(int)

    out["distance"] = df["dist1"].fillna(df["dist2"]).fillna(0)

    # Card fingerprint -- card1 is almost always populated and is one of the
    # single strongest predictors in this dataset (same card reused = signal).
    for col in ["card1", "card2", "card3", "card5"]:
        out[col] = df[col].fillna(-1)

    # Billing address region codes.
    out["addr1"] = df["addr1"].fillna(-1)
    out["addr2"] = df["addr2"].fillna(-1)

    # Full velocity/count features (we previously only used C1-C5).
    for i in range(1, 15):
        out[f"C{i}"] = df[f"C{i}"].fillna(0)

    # Full time-delta features (we previously only used D1-D3).
    for i in range(1, 16):
        out[f"D{i}"] = df[f"D{i}"].fillna(0)

    # Verification match flags. Mostly T/F, M4 uses a different categorical
    # scheme (M0/M1/M2) -- .cat.codes handles both uniformly and maps missing
    # to -1 automatically, same convention as ProductCD above.
    for i in range(1, 10):
        col = f"M{i}"
        if col in df.columns:
            out[col] = df[col].astype("category").cat.codes
        else:
            out[col] = -1

    out["isFraud"] = df["isFraud"]
    return out


if __name__ == "__main__":
    df = load_real_data()
    print("Fraud rate:", df["isFraud"].mean())
    print("Feature count:", len(FEATURE_COLUMNS))
    print(df.groupby("isFraud").size())
    print(df.head())
