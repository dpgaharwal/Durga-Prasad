# Track C — Deepfake Identity Fraud (Colab notebook, NOT run yet)

Voice clone vishing + video deepfake KYC bypass, generation + detection.
Runs on Google Colab (free T4 GPU) — not built/tested in this repo directly
because it needs a GPU this environment doesn't have.

## Run it

1. Open `generate/track_c_deepfake/track_c_colab.ipynb` in Colab
   (colab.research.google.com -> File -> Upload notebook)
2. Runtime -> Change runtime type -> **T4 GPU**
3. Run cells top to bottom. You'll be prompted to upload:
   - A reference voice clip (6-30 sec) — your own voice, not a real named person
   - A face photo (front-facing)
   - A real reference video (5-10 sec) for the detector comparison

## What it does

**Section 1 — Voice clone (Coqui XTTS-v2):** clones the uploaded reference
voice reading an OTP-vishing script. CPML license — fine for demo, not
commercial use.

**Section 2 — Video deepfake (SadTalker):** lip-synced talking-head video
from the photo + the cloned audio, simulating a KYC selfie-injection attempt.

**Section 3 — Audio detection:** pretrained HuggingFace model
(`MelodyMachine/Deepfake-audio-detection-V2`), scores both the genuine
reference clip and the cloned attack audio.

**Section 4 — Video detection (custom rPPG):** see the honesty note below —
this is a from-scratch lightweight implementation, not a pretrained checkpoint.

**Section 5:** zips everything and downloads it — bring the zip back into
this repo's `generate/track_c_deepfake/outputs/` folder.

## Honesty note on the video detector (important for the writeup)

The published reference for physiological video-deepfake detection,
**DeepFakesON-Phys** (BiDAlab, arXiv:2010.00400, 98%+ AUC on Celeb-DF/DFDC),
does **not** have its training code or pretrained weights publicly released —
there's an [open, unanswered GitHub issue](https://github.com/BiDAlab/DeepFakesON-Phys/issues/1)
requesting it since 2020. We checked before committing to it as "pretrained,
ready to use," and it isn't.

Rather than depend on a checkpoint that doesn't exist, Section 4 implements a
**lightweight rPPG detector from the same underlying principle**: real video
of a living face shows a periodic pulse signal from blood-flow-driven skin
color micro-changes (via a simplified POS transform); deepfake generation
doesn't model this, so the signal is weak or absent. Feature: signal-to-noise
ratio at the peak frequency in the human heart-rate band (0.7-4 Hz).

This is a smaller, from-scratch implementation, not a validated production
detector — say so plainly in the writeup, and report it as "inspired by
DeepFakesON-Phys's physiological-signal approach, own implementation since
the reference model's weights aren't public." That's a more credible claim
to a technical judge than pretending to use a checkpoint that doesn't exist.

Run the rPPG feature on a **few** real/fake video pairs (not just one) before
claiming a specific SNR threshold in the deck — one pair shows direction, not
a validated cutoff.

## Files

- `track_c_colab.ipynb` — the whole pipeline, run in Colab

## Bringing outputs back into the repo

After running the notebook and downloading `track_c_outputs.zip`:
```bash
mkdir -p generate/track_c_deepfake/outputs
unzip ~/Downloads/track_c_outputs.zip -d generate/track_c_deepfake/outputs
```
This gives you the actual attack artifacts (cloned audio, deepfake video) and
`results_summary.json` for the code repo and the demo.
