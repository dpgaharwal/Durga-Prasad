# Web Prototype — AI Defense Lab Dashboard

Single HTML file, no build step, no server, no dependencies beyond a browser
with internet access (for Google Fonts — swap to system fonts in the `<link>`
tags if presenting somewhere offline).

## Run it

Just open `index.html` in any browser — double-click it, or:
```bash
open index.html          # Mac
```

## Why static, not a live backend

Hackathon demo reliability > live-compute theater. Every number on this
dashboard is a real result from an actual verified run (logged throughout
the repo's READMEs and commit history) — nothing is fabricated for display.
The dashboard's job is presenting real results clearly and fast, not
re-running the pipeline live in front of judges, which is a common and
avoidable failure point (cold starts, flaky model calls, venue WiFi).

## Enabling media playback

Track C's audio/video players look for files in `web/frontend/media/`. Copy
your Colab-generated files there using the exact names listed in
`media/PLACE_MEDIA_HERE.txt`. Without them, the dashboard shows a clean
"file not found" placeholder instead of a broken player — safe to present
either way, but real media makes Track C land harder.

## Updating the numbers

All data is inline in `index.html` — search for the relevant Track section
(marked with HTML comments) and edit the numbers/tables directly, or the
`verdicts` array near the bottom (in the `<script>` tag) for the scrolling
tape. No build step, no data file to regenerate.

## Structure

- Sticky verdict tape at the top — real logged verdicts from all 4 tracks, looping
- Tab navigation: Overview + one tab per track
- Overview: closed-loop diagram + headline number per track (click a card to jump to that track)
- Each track tab: metrics, before/after bar comparisons, and an honest-limitation callout where relevant
