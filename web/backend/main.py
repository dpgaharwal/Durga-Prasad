"""
Web prototype backend — live re-run endpoints for all 4 tracks.

Design decision (say this to judges if asked): Track C's attack GENERATION
(voice clone, video deepfake) needs GPU minutes and cannot run live in a demo
— it's pre-generated, same as every other team's would be. Everything else
here — Track A/D training cycles, Track B's agent+gate pipeline, and Track C's
DETECTION step — runs live, for real, computed fresh on each request. That's
the honest line between "necessarily pre-baked" and "actually live."

Run:
    cd web/backend
    uvicorn main:app --reload --port 8000

Requires (from repo root, one level up from here):
    - generate/track_a_transactions/ieee_data/  (real IEEE-CIS data, optional
      -- falls back to synthetic with a warning if missing)
    - Ollama running locally with a pulled model, for Track B
      (falls back to MockLLM with a warning if Ollama is unreachable)
    - web/frontend/media/  populated with Track C's generated files
      (endpoints return a clear error per-file if missing, not a crash)
"""

import sys
import os
import time
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

for p in [
    "generate/track_a_transactions",
    "generate/track_b_agentic",
    "generate/track_d_poisoning",
    "defend/track_a_classifier",
    "defend/track_b_reasoning_gate",
    "defend/track_d_label_defense",
    "loop/feedback_orchestrator",
]:
    sys.path.append(os.path.join(REPO_ROOT, p))

app = FastAPI(title="AI Defense Lab — Live Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only, don't ship this wide open
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_DIR = os.path.join(REPO_ROOT, "web", "frontend", "media")
IEEE_DATA_DIR = os.path.join(REPO_ROOT, "generate", "track_a_transactions", "ieee_data")

# ----------------------------------------------------------------------
# Startup: load data once, train the v1 baseline once. Subsampled for a
# responsive live demo -- the full-dataset numbers already shown as static
# "verified run" panels in the dashboard used the complete dataset; this
# backend trades some scale for the ability to actually recompute live.
# ----------------------------------------------------------------------

STATE = {"data_source": None, "df": None, "legit": None, "fraud": None,
         "v1_model": None, "ollama_available": False, "llm": None}


def _load_data():
    from real_data_loader import load_real_data
    from synthetic_baseline import generate_baseline

    if os.path.isdir(IEEE_DATA_DIR) and os.path.exists(
        os.path.join(IEEE_DATA_DIR, "train_transaction.csv")
    ):
        df = load_real_data(IEEE_DATA_DIR)
        df = df.sample(n=min(30000, len(df)), random_state=1).reset_index(drop=True)
        STATE["data_source"] = "real (30k subsample for live responsiveness)"
    else:
        df = generate_baseline(n_legit=15000, n_fraud=600, seed=1)
        STATE["data_source"] = "synthetic (ieee_data/ not found -- see README)"

    STATE["df"] = df
    STATE["legit"] = df[df.isFraud == 0]
    STATE["fraud"] = df[df.isFraud == 1]


def _train_v1():
    from train_and_evaluate import train_classifier
    STATE["v1_model"] = train_classifier(STATE["df"])


def _init_llm():
    from llm_client import MockLLM, OllamaLLM
    try:
        llm = OllamaLLM(model="qwen3:8b")
        # cheap liveness probe
        llm.client.invoke("respond with the single word OK")
        STATE["llm"] = llm
        STATE["ollama_available"] = True
    except Exception as e:
        print(f"[startup] Ollama unavailable ({e}) -- falling back to MockLLM")
        STATE["llm"] = MockLLM()
        STATE["ollama_available"] = False


@app.on_event("startup")
def startup():
    print("[startup] loading data...")
    _load_data()
    print(f"[startup] data source: {STATE['data_source']}")
    print("[startup] training v1 baseline classifier...")
    _train_v1()
    print("[startup] connecting to Ollama...")
    _init_llm()
    print(f"[startup] Ollama available: {STATE['ollama_available']}")
    print("[startup] ready.")


@app.get("/api/status")
def status():
    return {
        "data_source": STATE["data_source"],
        "ollama_available": STATE["ollama_available"],
        "rows_loaded": len(STATE["df"]) if STATE["df"] is not None else 0,
    }


# ----------------------------------------------------------------------
# Track A -- live adversarial-evasion + closed-loop cycle
# ----------------------------------------------------------------------

@app.post("/api/track-a/run-cycle")
def track_a_run_cycle():
    from adversarial_generator import generate_adversarial_fraud
    from train_and_evaluate import train_classifier, evaluate

    t0 = time.time()
    seed = random.randint(1, 1_000_000)  # fresh each call -- genuinely live, not cached

    legit, fraud = STATE["legit"], STATE["fraud"]
    model_v1 = STATE["v1_model"]

    adv = generate_adversarial_fraud(fraud, legit, evasion_strength=0.6, seed=seed)
    baseline_result = evaluate(model_v1, pd.concat([legit, fraud]), "v1 baseline")
    adv_result = evaluate(model_v1, pd.concat([legit, adv]), "v1 adversarial")

    # live retrain with the missed examples as hard examples
    X_adv = adv[model_v1.feature_names_in_.tolist()]
    missed = adv[model_v1.predict(X_adv) == 0]
    train_v2 = pd.concat([STATE["df"], missed], ignore_index=True)
    model_v2 = train_classifier(train_v2)

    adv2 = generate_adversarial_fraud(fraud, legit, evasion_strength=0.6, seed=seed + 1)
    v2_result = evaluate(model_v2, pd.concat([legit, adv2]), "v2 fresh adversarial")

    return {
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 2),
        "v1_baseline": baseline_result,
        "v1_adversarial": adv_result,
        "hard_examples_fed_back": int(len(missed)),
        "v2_fresh_adversarial": v2_result,
        "data_source": STATE["data_source"],
    }


# ----------------------------------------------------------------------
# Track B -- live agent + gate pipeline (real Ollama call if available)
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
        "naive": {"status": naive_result["final_status"], "mandate": naive_result["mandate"]},
        "gated": {
            "decision": gv.decision,
            "mandate": gated_result["mandate"],
            "checks": {name: c["passed"] for name, c in gv.checks.items()},
            "reasoning_trace": gv.reasoning_trace,
        },
    }


# ----------------------------------------------------------------------
# Track C -- live detection on pre-generated media (generation is offline)
# ----------------------------------------------------------------------

_audio_classifier = None


def _get_audio_classifier():
    global _audio_classifier
    if _audio_classifier is None:
        from transformers import pipeline
        _audio_classifier = pipeline(
            "audio-classification", model="MelodyMachine/Deepfake-audio-detection-V2"
        )
    return _audio_classifier


@app.post("/api/track-c/detect-audio")
def track_c_detect_audio(which: str):
    filename = "reference_voice.wav" if which == "genuine" else "cloned_voice_attack.wav"
    path = os.path.join(MEDIA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"{filename} not found in web/frontend/media/")

    t0 = time.time()
    clf = _get_audio_classifier()
    result = clf(path)
    return {"which": which, "elapsed_sec": round(time.time() - t0, 2), "scores": result}


@app.post("/api/track-c/detect-video")
def track_c_detect_video(which: str):
    import cv2
    import numpy as np
    from scipy import signal as sp_signal

    filename = "real_reference_video.mov" if which == "real" else "deepfake_video_attack.mp4"
    path = os.path.join(MEDIA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"{filename} not found in web/frontend/media/")

    t0 = time.time()

    def extract_face_rgb_series(video_path, max_frames=150, resize_width=320):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        rgb_series, frame_count = [], 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (resize_width, int(h * resize_width / w)))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w2, h2 = max(faces, key=lambda f: f[2] * f[3])
                roi = small[y:y + int(h2 * 0.4), x + int(w2 * 0.25):x + int(w2 * 0.75)]
                if roi.size > 0:
                    rgb_series.append(roi.reshape(-1, 3).mean(axis=0)[::-1])
            frame_count += 1
        cap.release()
        return np.array(rgb_series), fps

    def pos_pulse(rgb):
        if len(rgb) < 10:
            return np.array([])
        m = rgb.mean(axis=0)
        n = rgb / (m + 1e-8)
        S1 = n[:, 1] - n[:, 2]
        S2 = n[:, 1] + n[:, 2] - 2 * n[:, 0]
        alpha = np.std(S1) / (np.std(S2) + 1e-8)
        return S1 + alpha * S2

    rgb_series, fps = extract_face_rgb_series(path)
    pulse = pos_pulse(rgb_series)
    if len(pulse) < 20:
        return {"which": which, "elapsed_sec": round(time.time() - t0, 2),
                "snr_db": None, "peak_bpm": None, "note": "not enough face frames detected"}

    pulse = sp_signal.detrend(pulse)
    nyq = fps / 2
    b, a = sp_signal.butter(3, [0.7 / nyq, min(4.0 / nyq, 0.99)], btype="band")
    filtered = sp_signal.filtfilt(b, a, pulse)
    freqs, psd = sp_signal.welch(filtered, fs=fps, nperseg=min(256, len(filtered)))
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    band_freqs, band_psd = freqs[mask], psd[mask]
    peak_idx = np.argmax(band_psd)
    snr_db = 10 * np.log10(band_psd[peak_idx] / (np.mean(np.delete(band_psd, peak_idx)) + 1e-12))

    return {
        "which": which,
        "elapsed_sec": round(time.time() - t0, 2),
        "snr_db": round(float(snr_db), 2),
        "peak_bpm": round(float(band_freqs[peak_idx] * 60), 1),
        "n_frames_with_face": len(rgb_series),
    }


# ----------------------------------------------------------------------
# Track D -- live poisoning + disagreement-screening cycle
# ----------------------------------------------------------------------

@app.post("/api/track-d/run-cycle")
def track_d_run_cycle():
    from feedback_poisoner import build_poisoned_feedback
    from label_provenance import disagreement_screen, poison_recall
    from train_and_evaluate import train_classifier, evaluate

    t0 = time.time()
    seed = random.randint(1, 1_000_000)

    legit, fraud = STATE["legit"], STATE["fraud"]
    base_train = STATE["df"]
    base_model = STATE["v1_model"]

    clean_feedback = pd.concat([
        legit.sample(min(300, len(legit)), random_state=seed),
        fraud.sample(min(80, len(fraud)), random_state=seed),
    ], ignore_index=True)
    clean_feedback["label_source"] = "analyst_confirmed"

    poisoned_batch = build_poisoned_feedback(
        clean_feedback, fraud_pool=fraud, legit_pool=legit,
        n_silent=400, n_dispute=200, seed=seed,
    )

    model_poisoned = train_classifier(pd.concat([base_train, poisoned_batch], ignore_index=True))
    poisoned_metrics = evaluate(model_poisoned, pd.concat([legit, fraud]), "poisoned retrain")

    accepted, quarantined, _ = disagreement_screen(poisoned_batch, base_model, suspicion_threshold=0.05)
    model_defended = train_classifier(pd.concat([base_train, accepted], ignore_index=True))
    defended_metrics = evaluate(model_defended, pd.concat([legit, fraud]), "defended retrain")
    screening = poison_recall(quarantined, poisoned_batch)

    return {
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 2),
        "poisoned_retrain": poisoned_metrics,
        "defended_retrain": defended_metrics,
        "screening": screening,
        "data_source": STATE["data_source"],
    }
