"""
Track D2 - Poisoning demo loop.

Runs three scenarios back to back and prints the comparison:

  1. CLEAN     - honest feedback, retrain normally (control)
  2. POISONED  - attacker's poisoned feedback, retrain normally (the attack lands)
  3. DEFENDED  - same poisoned feedback, but through provenance weighting +
                 cluster influence screening (the attack is blunted)

The headline metric is NOT overall recall. It is recall INSIDE the attacker's
signature region -- the blind spot. Overall metrics stay healthy under attack,
which is exactly why this attack is dangerous and why a dashboard watching only
aggregate AUC would never catch it. Show both numbers side by side in the deck.

Run:  python3 track_d_loop.py
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../generate/track_a_transactions"))
sys.path.append(os.path.join(HERE, "../../generate/track_d_poisoning"))
sys.path.append(os.path.join(HERE, "../../defend/track_a_classifier"))
sys.path.append(os.path.join(HERE, "../../defend/track_d_label_defense"))

from real_data_loader import load_real_data
from train_and_evaluate import train_classifier, evaluate, FEATURES
from feedback_poisoner import (
    build_poisoned_feedback, stamp_signature, matches_signature,
)
from label_provenance import (
    provenance_weights, disagreement_screen, cluster_signature, poison_recall,
)


def blind_spot_recall(model, fraud_test: pd.DataFrame) -> float:
    """Recall specifically on fraud sitting inside the attacker's signature region."""
    region_fraud = stamp_signature(fraud_test, seed=99)
    preds = model.predict(region_fraud[FEATURES])
    return round(float(preds.mean()), 4)   # fraction of true fraud correctly flagged


def run_track_d(seed: int = 42):
    df = load_real_data(data_dir="../../generate/track_a_transactions/ieee_data")
    legit = df[df.isFraud == 0]
    fraud = df[df.isFraud == 1]

    train_legit, rest_legit = train_test_split(legit, test_size=0.4, random_state=seed)
    train_fraud, rest_fraud = train_test_split(fraud, test_size=0.4, random_state=seed)

    # Trusted holdout: analyst-confirmed labels the attacker cannot influence.
    trusted_legit, test_legit = train_test_split(rest_legit, test_size=0.5, random_state=seed)
    trusted_fraud, test_fraud = train_test_split(rest_fraud, test_size=0.5, random_state=seed)
    trusted_holdout = pd.concat([trusted_legit, trusted_fraud], ignore_index=True)
    test_set = pd.concat([test_legit, test_fraud], ignore_index=True)

    base_train = pd.concat([train_legit, train_fraud], ignore_index=True)

    # Honest feedback batch that would arrive in a normal retraining cycle.
    clean_feedback = pd.concat([
        train_legit.sample(400, random_state=seed),
        train_fraud.sample(120, random_state=seed),
    ], ignore_index=True)

    results = {}

    # --- Scenario 1: CLEAN ---
    clean_batch = clean_feedback.copy()
    clean_batch["label_source"] = "analyst_confirmed"
    model_clean = train_classifier(pd.concat([base_train, clean_batch], ignore_index=True))
    results["1_clean"] = {
        **evaluate(model_clean, test_set, "clean retrain"),
        "blind_spot_recall": blind_spot_recall(model_clean, test_fraud),
    }

    # --- Scenario 2: POISONED ---
    poisoned_batch = build_poisoned_feedback(
        clean_feedback, fraud_pool=train_fraud, legit_pool=train_legit,
        n_silent=600, n_dispute=300, seed=seed,
    )
    model_poisoned = train_classifier(pd.concat([base_train, poisoned_batch], ignore_index=True))
    results["2_poisoned"] = {
        **evaluate(model_poisoned, test_set, "poisoned retrain"),
        "blind_spot_recall": blind_spot_recall(model_poisoned, test_fraud),
    }

    # --- Scenario 3: DEFENDED ---
    # The screening model is the one trained BEFORE this feedback arrived --
    # it is the last known-good reference point the attacker has not touched.
    base_model = train_classifier(base_train)

    accepted, quarantined, scored = disagreement_screen(
        poisoned_batch, base_model=base_model, suspicion_threshold=0.05,
    )

    defended_train = pd.concat([base_train, accepted], ignore_index=True)
    model_defended = train_classifier(defended_train)

    results["3_defended"] = {
        **evaluate(model_defended, test_set, "defended retrain"),
        "blind_spot_recall": blind_spot_recall(model_defended, test_fraud),
    }
    results["screening"] = poison_recall(quarantined, poisoned_batch)
    results["cluster_report"] = cluster_signature(quarantined)

    return results


if __name__ == "__main__":
    r = run_track_d()

    print("=" * 78)
    print("TRACK D2 - FEEDBACK-LOOP POISONING")
    print("=" * 78)

    for key in ["1_clean", "2_poisoned", "3_defended"]:
        m = r[key]
        print(f"\n{key.upper()}")
        print(f"  overall     -> precision {m['precision']}  recall {m['recall']}  "
              f"f1 {m['f1']}  auc {m['auc']}  FP-rate {m['false_positive_rate']}")
        print(f"  BLIND SPOT  -> recall inside attacker's signature region: {m['blind_spot_recall']}")

    print("\n" + "-" * 78)
    print("CLUSTER SCREENING REPORT")
    print(r["cluster_report"].to_string(index=False))
    print("\nPOISON CAUGHT:", r["screening"])

    print("\n" + "=" * 78)
    print("HEADLINE FOR THE DECK:")
    print(f"  blind-spot recall  clean {r['1_clean']['blind_spot_recall']}"
          f"  ->  poisoned {r['2_poisoned']['blind_spot_recall']}"
          f"  ->  defended {r['3_defended']['blind_spot_recall']}")
    print(f"  overall AUC        clean {r['1_clean']['auc']}"
          f"  ->  poisoned {r['2_poisoned']['auc']}"
          f"  ->  defended {r['3_defended']['auc']}")
    print("  ^ note how little overall AUC moves under attack -- that is the point")
    print("=" * 78)
