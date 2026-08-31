"""Track A - classifier train/evaluate. FEATURES here is the canonical list -- other modules import it from here."""

import xgboost as xgb
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from real_data_loader import FEATURE_COLUMNS

FEATURES = FEATURE_COLUMNS


def train_classifier(train_df: pd.DataFrame) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, eval_metric="auc",
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
    return {
        "set": label,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_proba), 4) if y_true.nunique() > 1 else None,
        "false_positive_rate": round(fp_rate, 4),
        "n": len(test_df),
    }
