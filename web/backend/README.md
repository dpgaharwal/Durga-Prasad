# Web Prototype — Live Backend

Real FastAPI backend. Each endpoint actually recomputes on every call — fresh
random seeds for Track A/D, a real Ollama call for Track B, real model
inference for Track C's detectors. Nothing here is cached or canned.

## What's live vs. pre-generated, and why

Track C's attack **generation** (voice clone, video deepfake) needs GPU
minutes and cannot run live in a demo — same constraint every team has. It's
pre-generated via the Colab notebook. Everything else — Track A/D's training
cycles, Track B's agent+gate pipeline, and Track C's **detection** step —
runs live, computed fresh on each request. Say this distinction plainly if a
judge asks; it's the honest and correct line to draw.

## Setup

```bash
cd web/backend
pip install -r ../../requirements.txt --break-system-packages
```

Also needed, from the repo root:
- `generate/track_a_transactions/ieee_data/` — real IEEE-CIS data (see
  `README_TRACK_A.md` for download steps). Missing this isn't fatal — the
  backend falls back to synthetic data with a clear warning, both in its
  startup log and in the dashboard's status bar.
- Ollama running locally with `qwen3:8b` pulled, for Track B. Missing this
  falls back to MockLLM, also clearly flagged — the dashboard tells you which
  one is active.
- `web/frontend/media/` populated with Track C's generated files (see
  `README_TRACK_C.md`) for the detection endpoints to have something to run on.

## Run

```bash
cd web/backend
uvicorn main:app --reload --port 8000
```

Startup takes 10-30 seconds (loads data, trains the v1 baseline classifier
once, connects to Ollama). Watch the terminal log — it tells you exactly
which data source and LLM backend got picked.

Then open `web/frontend/index.html` in a browser (separately, it's a static
file, no serving needed) — its status bar will show "Live backend connected"
once it can reach `localhost:8000`, and every "Run live" button will call
through and show fresh results.

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /api/status` | data source + Ollama availability, for the dashboard's status bar |
| `POST /api/track-a/run-cycle` | fresh adversarial batch → evaluate v1 → retrain v2 live → evaluate v2 |
| `POST /api/track-b/run-scenario` | `{sku, injected}` → real agent+gate run |
| `POST /api/track-c/detect-audio?which=genuine\|cloned` | live pretrained-model inference on the corresponding file |
| `POST /api/track-c/detect-video?which=real\|fake` | live rPPG feature extraction on the corresponding file |
| `POST /api/track-d/run-cycle` | fresh poisoned batch → poisoned retrain → disagreement screen → defended retrain |

## Performance note

Track A/D use a 30k-row subsample of the real data for responsiveness (a few
seconds per cycle instead of tens of seconds on the full ~590k-row dataset).
The static "verified run" numbers already shown in the dashboard's panels
above each live section used the full dataset — mention both if asked: full
dataset for the headline numbers, fast subsample for the live re-run.
