"""
Track A - Closed-loop feedback demo.

This is the centerpiece for the "closed loop" story judges score explicitly:
  1. Train v1 classifier on baseline data only.
  2. Generate adversarial fraud, evaluate v1 -> it under-performs (evasion works).
  3. Feed the missed adversarial examples back into training -> retrain v2.
  4. Re-evaluate v2 on a FRESH adversarial batch (not the same one it trained on)
     -> recall should recover. That improvement number IS your closed-loop proof.

Run this file directly to print the full before/after report.
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "../../generate/track_a_transactions"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../defend/track_a_classifier"))

from real_data_loader import load_real_data
from adversarial_generator import generate_adversarial_fraud
from train_and_evaluate import train_classifier, evaluate

# To use the real IEEE-CIS dataset instead of synthetic data, once you have
# run the Kaggle download steps in real_data_loader.py:
#     from real_data_loader import load_real_data
# and replace the `df = generate_baseline(seed=seed)` line below with:
#     df = load_real_data()
# Nothing else in this file (or adversarial_generator.py, train_and_evaluate.py)
# needs to change -- both functions return the same schema.


def run_closed_loop(evasion_strength: float = 0.6, seed: int = 42):
    df = load_real_data(data_dir="../../generate/track_a_transactions/ieee_data")
    legit = df[df.isFraud == 0]
    fraud = df[df.isFraud == 1]

    train_legit, test_legit = train_test_split(legit, test_size=0.3, random_state=seed)
    train_fraud, test_fraud = train_test_split(fraud, test_size=0.3, random_state=seed)

    # --- Cycle 0: v1 trained on baseline only ---
    train_v1 = pd.concat([train_legit, train_fraud], ignore_index=True)
    model_v1 = train_classifier(train_v1)

    # Generate adversarial fraud from held-out fraud examples (round 1 seed)
    adv_round1 = generate_adversarial_fraud(test_fraud, legit, evasion_strength=evasion_strength, seed=1)

    result_v1_baseline = evaluate(model_v1, pd.concat([test_legit, test_fraud]), "v1 on baseline fraud")
    result_v1_adv = evaluate(model_v1, pd.concat([test_legit, adv_round1]), "v1 on adversarial round 1")

    # --- Feedback: missed adversarial examples go back into training ---
    X_adv1 = adv_round1[model_v1.feature_names_in_.tolist()]
    y_pred_adv1 = model_v1.predict(X_adv1)
    missed = adv_round1[y_pred_adv1 == 0]  # false negatives = hard examples

    train_v2 = pd.concat([train_v1, missed], ignore_index=True)
    model_v2 = train_classifier(train_v2)

    # --- Cycle 1: evaluate v2 on a FRESH adversarial batch (different seed,
    # simulates the next wave of attacks, not the same examples it just saw) ---
    adv_round2 = generate_adversarial_fraud(test_fraud, legit, evasion_strength=evasion_strength, seed=2)
    result_v2_adv = evaluate(model_v2, pd.concat([test_legit, adv_round2]), "v2 on adversarial round 2 (fresh)")

    return {
        "v1_on_baseline_fraud": result_v1_baseline,
        "v1_on_adversarial_round1": result_v1_adv,
        "hard_examples_fed_back": len(missed),
        "v2_on_adversarial_round2_fresh": result_v2_adv,
    }


if __name__ == "__main__":
    report = run_closed_loop()
    print("=" * 70)
    print("TRACK A CLOSED-LOOP REPORT")
    print("=" * 70)
    for key, val in report.items():
        print(f"\n{key}:")
        print(val)

    r1 = report["v1_on_adversarial_round1"]["recall"]
    r2 = report["v2_on_adversarial_round2_fresh"]["recall"]
    print("\n" + "=" * 70)
    print(f"RECALL ON ADVERSARIAL FRAUD -- v1: {r1}  ->  v2 (after feedback loop): {r2}")
    print(f"Hard examples fed back into training: {report['hard_examples_fed_back']}")
    print("=" * 70)
