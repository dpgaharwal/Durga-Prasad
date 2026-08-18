# Track D2 — Feedback-Loop Poisoning (WORKING, TESTED)

The meta-track: instead of evading the fraud model, attack the fraud model's
*retraining pipeline*. This red-teams our own Track A closed loop.

## The idea

Production fraud systems retrain on labels that come back from the field, and
those labels are derived, not observed:

```
label = FRAUD   if a chargeback/dispute was filed
label = LEGIT   if no dispute was filed within the window
```

That second rule is an **assumption**, and it's the attack surface. An attacker
who controls both ends of a transaction simply never disputes — so their fraud
is silently labeled LEGIT and trained on.

**Mode A — silent fraud (the dangerous one):** attacker pushes fraud confined to
a narrow feature region, never disputes it, and the model carves out a blind
spot exactly where they operate. A targeted backdoor, not a degradation attack.

**Mode B — dispute flooding:** attacker files false disputes on legit
transactions, driving up false positives. Noisier, easier to catch.

## Run it

```bash
cd loop/feedback_orchestrator
python3 track_d_loop.py
```

## Verified results (actual run, synthetic data)

| Scenario | Overall AUC | FP rate | **Blind-spot recall** |
|---|---|---|---|
| Clean retrain | 1.0 | 0.001 | **0.9875** |
| Poisoned retrain | 1.0 | 0.003 | **0.7812** |
| Defended retrain | 1.0 | 0.0012 | **0.9750** |

**The headline for the deck:** overall AUC stays at 1.0 across all three
scenarios. A dashboard watching only aggregate metrics would never see this
attack. The damage is only visible when you measure recall *inside the
attacker's operating region*. That's the whole point of the track.

Screening performance:
- Mode B (dispute flood): **300/300 caught (100%)**
- Mode A (silent fraud): **214/600 caught (36%)**
- Clean rows wrongly quarantined: **0**

36% catch on Mode A is enough to restore blind-spot recall from 0.78 to 0.975,
because the caught rows are the highest-disagreement ones — the most damaging.
Report both numbers honestly; the gap is a real limitation, not a rounding error.

## The defense

**Layer 1 — label provenance trust.** Tag every feedback row with how its label
was derived, and score trust accordingly: `analyst_confirmed` 1.0,
`customer_dispute` 0.6, `no_dispute_default` 0.3.

**Layer 2 — disagreement screening (primary).** Score incoming feedback with the
model trained *before* the batch arrived, then:

```
suspicion = |p_fraud - supplied_label| x (1 - provenance_trust)
```

This formula is the crux. It separates two cases that look identical otherwise:
- Genuinely-missed fraud an analyst confirmed → high disagreement, trust 1.0 → suspicion **0** → **kept** (this is the hard example Track A's loop needs)
- Attacker's silent fraud auto-labeled legit → high disagreement, trust 0.3 → **quarantined**

Without provenance, any disagreement filter would throw away the useful hard
examples along with the poison. In the test run, analyst-confirmed rows scored
suspicion exactly 0.0 — perfect separation, which is why the false-quarantine
rate is zero at any threshold.

**Layer 3 — cluster diagnostic.** Profiles the quarantined rows so an analyst
gets the attacker's operating signature, not just a count. In the test run
cluster 1 came back at amount ≈302, hour ≈15, device mobile, ProductCD 2 —
which is exactly the injected attacker signature. That's an actionable rule.

## A negative result worth keeping in the writeup

We built **leave-cluster-out validation against a trusted holdout** first — the
textbook poisoning defense. It caught **zero** poison.

Why: a targeted backdoor creates a blind spot in a narrow region. If the trusted
holdout doesn't happen to contain fraud from that same region, including the
poison doesn't degrade holdout recall at all, so the screen stays silent.

This generalises: **validation-set-based poison detection only sees poison in
regions the validation set covers.** Judges respond well to a documented
negative result — it shows you actually ran the experiment instead of shipping
the first thing that sounded right.

## Files

- `generate/track_d_poisoning/feedback_poisoner.py` — both poisoning modes, threat model in the docstring
- `defend/track_d_label_defense/label_provenance.py` — the three defense layers
- `loop/feedback_orchestrator/track_d_loop.py` — runs all three scenarios, prints the comparison

## Honesty note for the writeup

Poisoning attacks against fraud detection systems exist in the literature
("Fraud Detection under Siege", ACM TOPS 2023; FRAUD-RLA, arXiv 2502.02290).
What's under-explored is **the label-derivation rule as the injection vector**
and building the closed-loop defense around label provenance. Claim that — not
invention of the concept. Overclaiming novelty is the fastest way to lose a
technical judge.
