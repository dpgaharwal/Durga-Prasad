"""
Web prototype backend -- self-contained deployment version.

All dependency modules live alongside this file (copied in from the main
repo structure) instead of being imported via cross-directory sys.path
manipulation. Railway's "Root Directory" setting isolates the build to just
this folder, so anything imported from outside it (generate/, defend/, etc.)
would 404 at import time -- this version avoids that entirely.

The canonical, non-duplicated source of truth for this logic is the main
repo (generate/, defend/, loop/) -- this folder is a deployment artifact.
Track A/D use a small committed CSV subsample (deploy_subsample.csv) since
the full ~600MB IEEE-CIS dataset can't ship in a container image.

Track C is not included here -- see README.md in this folder for why.
"""

import os
import time
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="AI Defense Lab -- Live Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY_SUBSAMPLE_PATH = os.path.join(HERE, "deploy_subsample.csv")
IEEE_DATA_DIR = os.path.join(HERE, "ieee_data")  # not present on deploy, that's fine

STATE = {"data_source": None, "train_df": None,
         "legit_train": None, "fraud_train": None,
         "legit_test": None, "fraud_test": None,
         "v1_model": None, "ollama_available": False, "llm": None,
         "llm_backend": "mock"}


def _load_data():
    from real_data_loader import load_real_data
    from synthetic_baseline import generate_baseline
    from sklearn.model_selection import train_test_split

    if os.path.exists(DEPLOY_SUBSAMPLE_PATH):
        df = pd.read_csv(DEPLOY_SUBSAMPLE_PATH)
        STATE["data_source"] = "real (committed deploy subsample, ~8.3k rows)"
    elif os.path.isdir(IEEE_DATA_DIR) and os.path.exists(os.path.join(IEEE_DATA_DIR, "train_transaction.csv")):
        df = load_real_data(IEEE_DATA_DIR)
        df = df.sample(n=min(30000, len(df)), random_state=1).reset_index(drop=True)
        STATE["data_source"] = "real (30k subsample for live responsiveness)"
    else:
        df = generate_baseline(n_legit=15000, n_fraud=600, seed=1)
        STATE["data_source"] = "synthetic (no deploy_subsample.csv or ieee_data/ found)"

    legit, fraud = df[df.isFraud == 0], df[df.isFraud == 1]
    train_legit, test_legit = train_test_split(legit, test_size=0.3, random_state=1)
    train_fraud, test_fraud = train_test_split(fraud, test_size=0.3, random_state=1)

    STATE["train_df"] = pd.concat([train_legit, train_fraud], ignore_index=True)
    STATE["legit_train"] = train_legit
    STATE["fraud_train"] = train_fraud
    STATE["legit_test"] = test_legit
    STATE["fraud_test"] = test_fraud


def _train_v1():
    from train_and_evaluate import train_classifier
    STATE["v1_model"] = train_classifier(STATE["train_df"])


def _init_llm():
    from llm_client import MockLLM, OllamaLLM, OpenAILLM

    if os.environ.get("OPENAI_API_KEY"):
        try:
            llm = OpenAILLM(model="gpt-4o-mini")
            llm.client.invoke("respond with the single word OK")
            STATE["llm"] = llm
            STATE["ollama_available"] = True
            STATE["llm_backend"] = "openai (gpt-4o-mini)"
            return
        except Exception as e:
            print(f"[startup] OpenAI configured but unavailable ({e}) -- trying Ollama")

    try:
        llm = OllamaLLM(model="qwen3:8b")
        llm.client.invoke("respond with the single word OK")
        STATE["llm"] = llm
        STATE["ollama_available"] = True
        STATE["llm_backend"] = "ollama (qwen3:8b)"
    except Exception as e:
        print(f"[startup] Ollama unavailable ({e}) -- falling back to MockLLM")
        STATE["llm"] = MockLLM()
        STATE["ollama_available"] = False
        STATE["llm_backend"] = "mock"


@app.on_event("startup")
def startup():
    print("[startup] loading data...")
    _load_data()
    print(f"[startup] data source: {STATE['data_source']}")
    print("[startup] training v1 baseline classifier...")
    _train_v1()
    print("[startup] connecting to LLM backend...")
    _init_llm()
    print(f"[startup] llm backend: {STATE['llm_backend']}")
    print("[startup] ready.")


@app.get("/api/status")
def status():
    return {
        "data_source": STATE["data_source"],
        "ollama_available": STATE["ollama_available"],
        "llm_backend": STATE["llm_backend"],
        "rows_loaded": len(STATE["train_df"]) + len(STATE["legit_test"]) + len(STATE["fraud_test"]),
    }


# ----------------------------------------------------------------------
# Track A
# ----------------------------------------------------------------------

@app.post("/api/track-a/run-cycle")
def track_a_run_cycle():
    from adversarial_generator import generate_adversarial_fraud
    from train_and_evaluate import train_classifier, evaluate

    t0 = time.time()
    seed = random.randint(1, 1_000_000)

    legit_test, fraud_test = STATE["legit_test"], STATE["fraud_test"]
    legit_train = STATE["legit_train"]
    model_v1 = STATE["v1_model"]

    adv = generate_adversarial_fraud(fraud_test, legit_train, evasion_strength=0.6, seed=seed)
    baseline_result = evaluate(model_v1, pd.concat([legit_test, fraud_test]), "v1 baseline (held-out)")
    adv_result = evaluate(model_v1, pd.concat([legit_test, adv]), "v1 adversarial (held-out)")

    X_adv = adv[model_v1.feature_names_in_.tolist()]
    missed = adv[model_v1.predict(X_adv) == 0]
    train_v2 = pd.concat([STATE["train_df"], missed], ignore_index=True)
    model_v2 = train_classifier(train_v2)

    adv2 = generate_adversarial_fraud(fraud_test, legit_train, evasion_strength=0.6, seed=seed + 1)
    v2_result = evaluate(model_v2, pd.concat([legit_test, adv2]), "v2 fresh adversarial (held-out)")

    return {
        "seed": seed, "elapsed_sec": round(time.time() - t0, 2),
        "v1_baseline": baseline_result, "v1_adversarial": adv_result,
        "hard_examples_fed_back": int(len(missed)), "v2_fresh_adversarial": v2_result,
        "data_source": STATE["data_source"],
    }


# ----------------------------------------------------------------------
# Track B
# ----------------------------------------------------------------------

class TrackBRequest(BaseModel):
    sku: str
    injected: bool


@app.post("/api/track-b/run-scenario")
def track_b_run_scenario(req: TrackBRequest):
    from catalog import CATALOG
    from track_b_loop import run_scenario

    if req.sku not in CATALOG:
        raise HTTPException(400, f"Unknown SKU. Valid: {list(CATALOG.keys())}")

    t0 = time.time()
    naive_result, gated_result = run_scenario(req.sku, "buy this product", req.injected, STATE["llm"])
    gv = gated_result["gate_verdict"]

    return {
        "elapsed_sec": round(time.time() - t0, 2),
        "using_real_llm": STATE["ollama_available"],
        "llm_backend": STATE["llm_backend"],
        "naive": {"status": naive_result["final_status"], "mandate": naive_result["mandate"]},
        "gated": {
            "decision": gv.decision,
            "mandate": gated_result["mandate"],
            "checks": {name: c["passed"] for name, c in gv.checks.items()},
            "reasoning_trace": gv.reasoning_trace,
        },
    }


# ----------------------------------------------------------------------
# Track D
# ----------------------------------------------------------------------

@app.post("/api/track-d/run-cycle")
def track_d_run_cycle():
    from feedback_poisoner import build_poisoned_feedback, stamp_signature
    from label_provenance import disagreement_screen, poison_recall
    from train_and_evaluate import train_classifier, evaluate, FEATURES

    t0 = time.time()
    seed = random.randint(1, 1_000_000)

    legit_train, fraud_train = STATE["legit_train"], STATE["fraud_train"]
    legit_test, fraud_test = STATE["legit_test"], STATE["fraud_test"]
    base_train = STATE["train_df"]
    base_model = STATE["v1_model"]

    def blind_spot_recall(model):
        region_fraud = stamp_signature(fraud_test, seed=99)
        preds = model.predict(region_fraud[FEATURES])
        return round(float(preds.mean()), 4)

    clean_feedback = pd.concat([
        legit_train.sample(min(300, len(legit_train)), random_state=seed),
        fraud_train.sample(min(80, len(fraud_train)), random_state=seed),
    ], ignore_index=True)
    clean_feedback["label_source"] = "analyst_confirmed"

    poisoned_batch = build_poisoned_feedback(
        clean_feedback, fraud_pool=fraud_train, legit_pool=legit_train,
        n_silent=400, n_dispute=200, seed=seed,
    )

    model_clean = train_classifier(pd.concat([base_train, clean_feedback], ignore_index=True))
    clean_metrics = evaluate(model_clean, pd.concat([legit_test, fraud_test]), "clean retrain (held-out)")
    clean_blind_spot = blind_spot_recall(model_clean)

    model_poisoned = train_classifier(pd.concat([base_train, poisoned_batch], ignore_index=True))
    poisoned_metrics = evaluate(model_poisoned, pd.concat([legit_test, fraud_test]), "poisoned retrain (held-out)")
    poisoned_blind_spot = blind_spot_recall(model_poisoned)

    accepted, quarantined, _ = disagreement_screen(poisoned_batch, base_model, suspicion_threshold=0.05)
    model_defended = train_classifier(pd.concat([base_train, accepted], ignore_index=True))
    defended_metrics = evaluate(model_defended, pd.concat([legit_test, fraud_test]), "defended retrain (held-out)")
    defended_blind_spot = blind_spot_recall(model_defended)

    screening = poison_recall(quarantined, poisoned_batch)

    return {
        "seed": seed, "elapsed_sec": round(time.time() - t0, 2),
        "clean_retrain": {**clean_metrics, "blind_spot_recall": clean_blind_spot},
        "poisoned_retrain": {**poisoned_metrics, "blind_spot_recall": poisoned_blind_spot},
        "defended_retrain": {**defended_metrics, "blind_spot_recall": defended_blind_spot},
        "screening": screening,
        "data_source": STATE["data_source"],
    }
