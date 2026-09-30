# Use of AI in this project

Keep this current as you build. Every member adds their own entries.

## Tools
- AI Assistant: research notes (`EVERYTHING you need.md`), the repo review
  (`docs/REVIEW.md`), the workspace setup, and the first draft of the video script.
- _(add each tool used: name, version, what for)_

## What the AI did
- 2026-09-28: Compared the research notes against the mentors' build guide and this
  repo; ran the model against known analog sites (`scripts/check_controls.py`);
  drafted `docs/TEAM_PLAN.md` and `docs/VIDEO_SCRIPT.md`.

- 2026-09-28 (v2): AI rebuilt the data pipeline (NASA POWER, GIBS NDVI,
  Natural Earth land mask), changed scoring to a weighted geometric mean with
  Earth-envelope clamping, added validation (`src/compute/validation.py`) and the
  known-analog catalog, and wrote the three.js interface. Every number in the app
  comes from `src/compute`; the team should review `data/targets.json` choices
  (terrain percentile classes) and `data/known_analogs.json`.

## What the AI did not do
- Scores come from the deterministic code in `src/compute`, never from a model.
- _(team: record the dataset choices, method decisions and design that are yours)_

## Prompts
- _(paste or summarise the key prompts here, with dates)_
