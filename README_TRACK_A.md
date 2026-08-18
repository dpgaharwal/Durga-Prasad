# Track A — Classic Transaction Fraud (WORKING, TESTED)

This part is fully built and verified end-to-end in a sandbox — no GPU, no API
keys, no internet dependency. Runs on any machine with Python 3.10+.

## Setup

```bash
pip install -r requirements.txt
```

## Run the full closed-loop demo

```bash
cd loop/feedback_orchestrator
python3 track_a_loop.py
```

This does, in one run:
1. Generates synthetic legit + fraud transactions (`generate/track_a_transactions/synthetic_baseline.py`)
2. Trains a v1 XGBoost classifier on baseline data
3. Generates adversarial fraud that evades v1 (`adversarial_generator.py`)
4. Shows v1's recall collapsing on the adversarial set — the evasion working
5. Feeds the missed examples back into training, retrains v2
6. Re-evaluates v2 on a FRESH adversarial batch — recall recovers

**Actual result from a test run:** v1 recall on adversarial fraud: 0.0 → v2
after the feedback loop: 0.99. That's your closed-loop number for the deck.

## Files

- `generate/track_a_transactions/synthetic_baseline.py` — legit+fraud generator (IEEE-CIS-inspired schema). Swap `load_real_data()` in for the real Kaggle dataset when you have it.
- `generate/track_a_transactions/adversarial_generator.py` — perturbs fraud to evade detection
- `generate/track_a_transactions/fidelity_check.py` — KS-test / PSI fidelity metrics (judges score this explicitly)
- `defend/track_a_classifier/train_and_evaluate.py` — XGBoost train + precision/recall/F1/AUC/FP-rate
- `loop/feedback_orchestrator/track_a_loop.py` — the full closed-loop, run this one

## Swapping in the real Kaggle dataset

Full steps + the actual loader are in `generate/track_a_transactions/real_data_loader.py`
(docstring at the top has the exact commands). Short version:

```bash
# 1. accept rules at kaggle.com/c/ieee-fraud-detection/rules (once, or API 403s)
# 2. kaggle.com/settings -> Create New API Token -> kaggle.json
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
pip install kaggle --break-system-packages
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d ieee_data/
```

Then in `loop/feedback_orchestrator/track_a_loop.py`, swap:
```python
from synthetic_baseline import generate_baseline
df = generate_baseline(seed=seed)
```
for:
```python
from real_data_loader import load_real_data
df = load_real_data()
```

`load_real_data()` already maps the real columns (`card6`, `ProductCD`,
`P_emaildomain`/`R_emaildomain`, `dist1`/`dist2`, `DeviceType`, `C1-C5`, `D1-D3`,
`TransactionDT`) onto the exact same schema the synthetic generator produces —
`adversarial_generator.py`, `train_and_evaluate.py`, and the loop don't need
any changes. Sanity-check the `TransactionHour` distribution once you have
real data — it's derived from `TransactionDT % 86400`, which is the
community-standard approximation since Kaggle never published the actual
reference timestamp.

## Not built yet (next phases)

- Track B (agentic/AP2) — generate + defend, needs LangGraph + Ollama, needs a machine
- Track C (deepfake) — needs GPU, do on Colab
- Web dashboard — Phase 4
- Solution walkthrough doc — Phase 5

See `mastercard-challenge-2026-plan.md` (earlier deliverable) for the full spec on all of these.
