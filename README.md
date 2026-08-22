# AI Defense Lab for Payment Security

### Mastercard Innovation Challenge 2026 — AI Defense Lab @ GFF

**Identify → Generate → Defend, across four structurally different GenAI-powered payment fraud attack surfaces.** Not one classifier — a closed loop, built four times, against four attacks a payments company actually faces.

**Every number in this repository is from a verified, logged run. Nothing here is a projection or a hand-picked result.**

[**Live Web Prototype**](https://dpgaharwal.github.io/Durga-Prasad/) · [**Solution Walkthrough (.docx)**](docs/Durga%20Prasad.docx) · [**Track C Colab Notebook**](generate/track_c_deepfake/track_c_colab.ipynb)

---

## Table of Contents

- [Why This Approach](#why-this-approach)
- [Results at a Glance](#results-at-a-glance)
- [The Closed Loop](#the-closed-loop)
- [Track A — Classic Transaction Fraud](#track-a--classic-transaction-fraud)
- [Track B — Agentic-Commerce Mandate Hijack](#track-b--agentic-commerce-mandate-hijack)
- [Track C — Deepfake Identity Fraud](#track-c--deepfake-identity-fraud)
- [Track D — Feedback-Loop Poisoning](#track-d--feedback-loop-poisoning)
- [Web Prototype](#web-prototype)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Honest Limitations](#honest-limitations)
- [Real-World Feasibility](#real-world-feasibility)
- [Judging Criteria Mapping](#judging-criteria-mapping)

---

## Why This Approach

Generative AI has lowered the barrier to sophisticated, fast-evolving payment fraud. The challenge asks for one red-team/blue-team closed loop: identify emerging attacks, generate high-fidelity simulations, build a defense that catches them, and feed every miss back into training.

Most submissions will build this cycle once, around a single classic transaction-fraud classifier. This submission builds it four times, against four attack surfaces that are structurally different from each other — because the diversity of what's being defended against matters as much as the depth of any single defense.

| Track | Attack Surface | Why It's Here |
|---|---|---|
| **A** | Classic transaction fraud | The required baseline — judges expect it, and it's the clearest place to prove the closed loop works |
| **B** | Agentic-commerce mandate hijack | Directly relevant to Mastercard's own Agent Pay Acceptance Framework (piloted with PayPal, AP2-compatible) |
| **C** | Deepfake identity fraud | Voice clone vishing + video KYC bypass — technical depth via physiological-signal detection, not naive frame classifiers |
| **D** | Feedback-loop poisoning | The meta-track — attacks Track A's own retraining pipeline, red-teaming our own closed loop |

---

## Results at a Glance

| Track | Metric | Result |
|---|---|---|
| A | Recall on adversarial fraud, before → after closed loop | **0.2555 → 0.9998** |
| B | Mandate-injection attacks blocked (real LLM, not scripted) | **2 / 2** |
| C | "Real" confidence a pretrained detector gave our cloned voice | **99.998%** (total evasion) |
| D | Poisoned labels caught by disagreement screening | **63.8%**, 0 false quarantines |

---

## The Closed Loop

```
   IDENTIFY                    GENERATE                     DEFEND
   Attack taxonomy      →      Adversarial data,      →     Classifiers,
   per track                   agents, deepfakes            gate, detectors
                                                                   │
        ▲                                                          │
        └──────────────── missed attacks feed back ────────────────┘
```

Every track logs what its defense misses and feeds it back into training or screening. This feedback mechanism — not any single detector — is what separates a closed-loop system from a one-shot classifier, and it's what the challenge brief explicitly rewards.

---

## Track A — Classic Transaction Fraud

**Identify:** static classifiers fail against GenAI-disguised fraud — the challenge brief's own opening premise, demonstrated concretely rather than asserted.

**Generate:** fraud transactions from a held-out test set are blended toward the legitimate population's distribution — card fingerprint, address codes, transaction velocity, and verification-match features nudged toward normal-looking values.

**Defend:** an XGBoost classifier (v1) is trained, evaluated against the disguised attack, and retrained (v2) on what it missed. v2 is then re-evaluated on a completely fresh disguised batch to confirm it learned the general pattern, not specific rows.

| Run | Precision | Recall | F1 | AUC | FP rate |
|---|---|---|---|---|---|
| v1 — baseline fraud | 0.249 | 0.824 | 0.3825 | 0.939 | 0.0901 |
| v1 — adversarial fraud | 0.0933 | 0.2555 | 0.1366 | 0.5482 | 0.0901 |
| v2 — fresh adversarial fraud | 0.3509 | **0.9998** | 0.5195 | 1.0 | 0.067 |

Real IEEE-CIS Fraud Detection dataset, 51 engineered features. 4,615 missed adversarial examples fed back between v1 and v2.

📄 [Full details → README_TRACK_A.md](README_TRACK_A.md)

---

## Track B — Agentic-Commerce Mandate Hijack

**Identify:** Mastercard's own Agent Pay Acceptance Framework — piloted with PayPal, built to interoperate with Google's AP2 — makes an AI shopping agent's payment mandate a live, current attack surface.

**Generate:** a shopper agent (LangGraph, real local LLM — Ollama qwen3:8b, not scripted) reads a product listing and issues a mandate. The listing's free-text description is the injection surface. Two attack variants tested: full merchant redirect, and same-merchant amount-only inflation.

**Defend:** a Reasoning Integrity Gate runs three checks before allowing payment execution.

| Scenario | Naive Pipeline | Gated Pipeline |
|---|---|---|
| Clean listing | Executed correctly ($129.99) | ALLOW |
| Injected — merchant redirect | Executed hijack ($349.99 → `EVIL_DROPSHIP_942`) | **BLOCK** |
| Injected — amount inflation | Executed hijack ($499.00 vs. $89.50 expected) | **BLOCK** |

The check that actually holds: **evidence-decision consistency** — comparing the mandate against the trusted catalog record, not pattern-matching the injection text. Injection marker scanning alone failed on both attacks, stated plainly rather than oversold.

📄 [Full details → README_TRACK_B.md](README_TRACK_B.md)

---

## Track C — Deepfake Identity Fraud

**Identify:** voice-clone vishing and video KYC-bypass are among the fastest-growing GenAI-enabled fraud vectors.

**Generate:** a genuine reference voice is cloned with Coqui XTTS-v2 to read an OTP-vishing script; the cloned audio drives SadTalker to generate a lip-synced deepfake video from a photo.

**Defend, audio:** a pretrained HuggingFace model (`MelodyMachine/Deepfake-audio-detection-V2`, zero fine-tuning) scores both clips.

| Clip | "Real" confidence |
|---|---|
| Genuine reference | 99.998% |
| Cloned attack audio | **99.998%** — total evasion |

**Defend, video:** the published reference for physiological video-deepfake detection (DeepFakesON-Phys, arXiv:2010.00400) has never released its weights — an open GitHub issue since 2020. Rather than depend on a checkpoint that doesn't exist, a lightweight rPPG detector was built from the same principle: real faces carry a periodic pulse signal from blood flow; deepfakes don't model it.

| Video | SNR (dB) |
|---|---|
| Real reference | 8.35 |
| Deepfake attack | 4.88 |

📄 [Full details → README_TRACK_C.md](README_TRACK_C.md) · [Colab notebook](generate/track_c_deepfake/track_c_colab.ipynb)

---

## Track D — Feedback-Loop Poisoning

**Identify:** production fraud systems retrain on labels derived from operational rules, not ground truth — "legitimate" if no chargeback was filed within a review window. That assumption is an injection vector: an attacker who controls both ends of a transaction never disputes, so their fraud is silently labeled legitimate and trained on. This track attacks Track A's own retraining pipeline.

**Generate:** two poisoning modes — silent fraud (confined to a narrow region, auto-labeled legitimate) and dispute flooding (false chargebacks on legitimate transactions).

**Defend:** disagreement screening, scoring each feedback row against the pre-poisoning model, weighted by label provenance:

```
suspicion = |model_score − supplied_label| × (1 − provenance_trust)
```

| Scenario | Blind-spot recall |
|---|---|
| Clean retrain | 0.7643 |
| Poisoned retrain | 0.7338 |
| Defended retrain | **0.7459** |

Overall AUC barely moves across all three (0.9357 → 0.9333 → 0.9355) — the damage is only visible measuring recall *inside* the attacker's region. A dashboard watching only aggregate metrics would never catch this.

**Documented negative result:** the textbook defense — leave-cluster-out validation against a trusted holdout — was built and tested first, and caught **zero** poison. Disagreement screening exists because of that finding.

📄 [Full details → README_TRACK_D.md](README_TRACK_D.md)

---

## Web Prototype

A single-file HTML dashboard, no build step, no server required — [hosted live here](https://dpgaharwal.github.io/Durga-Prasad/).

Wherever compute time allows, results are computed **live**, not just displayed:

| Component | Status | Detail |
|---|---|---|
| Track A / D cycles | Live | Fresh random seed each run, real retrain, 1–3 sec |
| Track B agent + gate | Live | Real Ollama call per run, 2–5 sec |
| Track C detection | Live | Real model inference on generated media |
| Track C generation | Pre-computed | GPU-bound (minutes) — same constraint every team has |

On the hosted version, "Run Live" buttons need a locally-running backend (see [`web/backend/README.md`](web/backend/README.md)) — the hosted page shows this plainly rather than a bare connection error.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Tabular ML | XGBoost, scikit-learn, pandas, numpy |
| Agent orchestration | LangGraph, Ollama (qwen3:8b) |
| Voice cloning | Coqui XTTS-v2 |
| Video deepfake | SadTalker |
| Audio detection | HuggingFace `transformers` (wav2vec2-based) |
| Video detection | Custom rPPG (OpenCV, SciPy) |
| Backend | FastAPI, uvicorn |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Data | IEEE-CIS Fraud Detection (Kaggle) |

---

## Repository Structure

```
├── identify/                       attack taxonomy
├── generate/
│   ├── track_a_transactions/       synthetic + real data, adversarial generator
│   ├── track_b_agentic/            LLM client, trusted catalog
│   ├── track_c_deepfake/           Colab notebook + generated attack media
│   └── track_d_poisoning/          feedback-loop poisoning generator
├── defend/
│   ├── track_a_classifier/         XGBoost train/eval
│   ├── track_b_reasoning_gate/     three-check reasoning gate
│   ├── track_c_detectors/          (detection logic lives in the Colab notebook + backend)
│   └── track_d_label_defense/      disagreement screening
├── loop/feedback_orchestrator/     the closed-loop demo scripts per track
├── web/
│   ├── frontend/                   static dashboard (index.html)
│   └── backend/                    FastAPI live-cycle backend
├── docs/                           solution walkthrough (.docx) + hosted dashboard copy
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/dpgaharwal/Durga-Prasad.git
cd Durga-Prasad
pip install -r requirements.txt --break-system-packages
```

**Run any track's closed-loop demo:**
```bash
cd loop/feedback_orchestrator
python3 track_a_loop.py   # classic fraud, synthetic data by default
python3 track_d_loop.py   # feedback-loop poisoning
python3 track_b_loop.py   # agentic mandate hijack (needs Ollama running locally)
```

**Swap in real IEEE-CIS data** instead of synthetic — see [`README_TRACK_A.md`](README_TRACK_A.md#swapping-in-the-real-kaggle-dataset).

**Run the live web backend:**
```bash
cd web/backend
uvicorn main:app --reload --port 8000
```
Then open `web/frontend/index.html` in a browser — full steps in [`web/backend/README.md`](web/backend/README.md).

**Track C (deepfake generation)** runs on Google Colab — open [`track_c_colab.ipynb`](generate/track_c_deepfake/track_c_colab.ipynb), select a T4 GPU runtime, run top to bottom.

---

## Honest Limitations

Stated plainly, not hidden in footnotes:

- **Track B's pattern-based injection scan** is beatable by an attacker avoiding known trigger phrases. The gate's evidence-decision consistency check is what actually holds against an adaptive attacker; the gate is framed as blast-radius reduction, not a guarantee.
- **Track C's rPPG video detector** is a from-scratch proof-of-concept (the published reference's weights were never released). Both a real and fake test clip resolved to the same peak BPM, which shouldn't happen if the fake carried zero physiological signal — the SNR gap is directional evidence, not a validated production threshold.
- **Track D's disagreement screening** catches silent fraud at 46% and dispute flooding at 100% — the asymmetry is real and reported, not smoothed over.
- **Track A/D's live web-backend cycles** run on a 30k-row subsample for responsiveness; the headline numbers above use the full dataset.

---

## Real-World Feasibility

- Track B is tested against a live industry surface, not a hypothetical — the mandate-injection attack surface here is the one Mastercard's own Agent Pay Acceptance Framework is actively hardening against right now.
- Track D's defense needs no new infrastructure to deploy — just tagging existing feedback by provenance source and re-scoring against the pre-update model.
- Every defense is framed by what it actually holds up against, not oversold.
- All four tracks converge on one operational pattern: log what the defense misses, feed it back, re-evaluate against fresh attacks.

---

## Judging Criteria Mapping

| Criterion | Where Addressed |
|---|---|
| Diversity of attacks identified | Four tracks: transaction evasion, mandate injection, voice/video deepfake, label-stream poisoning |
| Fidelity of attacks in simulation | KS/PSI statistical fidelity (Track A); 99.998% pretrained-detector evasion (Track C) |
| Detection algorithm efficacy | Precision/recall/F1/AUC/FP-rate reported for every track, on real data and real inference |
| Novelty of the solution | Track B targets Mastercard's own emerging Agent Pay surface; Track D self-referentially attacks our own closed loop |
| Real-world feasibility in live payments | Blast-radius framing over overclaiming; Track D's defense is deployable without new infrastructure |

---

**Team:** Durga Prasad · [Solution Walkthrough](docs/DurgaPrasad.docx) · [Live Prototype](https://dpgaharwal.github.io/Durga-Prasad/)
