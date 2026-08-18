"""
Track A - Defense classifier.

Trains an XGBoost classifier and reports precision/recall/F1/AUC plus the
false-positive rate on legitimate payments (called out explicitly in the
challenge brief -- keep FP low).
"""

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

FEATURES = [
    "TransactionAmt", "TransactionHour", "card_type", "ProductCD",
    "email_match", "device_type", "distance",
    "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3",
]


def train_classifier(train_df: pd.DataFrame) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="auc",
        scale_pos_weight=(train_df.isFraud == 0).sum() / max((train_df.isFraud == 1).sum(), 1),
        random_state=42,
    )
    model.fit(train_df[FEATURES], train_df["isFraud"])
    return model


def evaluate(model: xgb.XGBClassifier, test_df: pd.DataFrame, label: str = "") -> dict:
    X = test_df[FEATURES]
    y_true = test_df["isFraud"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fp_rate = fp / max(fp + tn, 1)

    metrics = {
        "set": label,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_proba), 4) if y_true.nunique() > 1 else None,
        "false_positive_rate": round(fp_rate, 4),
        "n": len(test_df),
    }
    return metrics


if __name__ == "__main__":
    import sys
    sys.path.append("../../generate/track_a_transactions")
    from synthetic_baseline import generate_baseline
    from adversarial_generator import generate_adversarial_fraud

    df = generate_baseline()
    legit = df[df.isFraud == 0]
    fraud = df[df.isFraud == 1]
    adv = generate_adversarial_fraud(fraud, legit, evasion_strength=0.6)

    train_df, test_legit = train_test_split(legit, test_size=0.3, random_state=42)
    _, test_fraud = train_test_split(fraud, test_size=0.5, random_state=42)
    train_fraud, _ = train_test_split(fraud, test_size=0.5, random_state=42)

    train_v1 = pd.concat([train_df, train_fraud], ignore_index=True)
    model_v1 = train_classifier(train_v1)

    print("=== v1 classifier: trained WITHOUT adversarial examples ===")
    print(evaluate(model_v1, pd.concat([test_legit, test_fraud]), "baseline fraud"))
    print(evaluate(model_v1, pd.concat([test_legit, adv]), "adversarial fraud"))
    print(">> expect recall to drop hard on adversarial set -- that's the evasion working")
