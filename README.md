# Earth Analogue Finder

Rank Earth's surface against an extraterrestrial base-site target. Give it the
lunar south pole or Jezero Crater and it returns the 0.5° cells on Earth whose
terrain, aridity, temperature swing and diurnal thermal behaviour match best,
each with a deterministic, source-cited explanation of *why*.

Everything at request time is arithmetic on a fixed predictor stack plus fixed
normalisation ranges. There is no model call, no randomness and no network
access, so the same request always returns the same numbers and the same
words.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. build the predictor stack (downloads ~1.1 GB on first run, then caches)
python -m src.acquire.build_predictor_stack

# 2. run the API plus the test UI
uvicorn src.api.main:app --reload
# open http://127.0.0.1:8000/

# 3. tests and lint
python -m pytest tests -q
python -m ruff check src tests conftest.py
python -m ruff format src tests conftest.py
```

`cache/predictors.zarr` and `cache/derived/*.npy` are committed, so a fresh
clone can skip step 1:

```bash
OFFLINE=1 python -m src.acquire.build_predictor_stack   # reassembles from cache
OFFLINE=1 uvicorn src.api.main:app
```

---

## How it works

```
                src/acquire/build_predictor_stack.py   (stages: dem | aridity | climate | lst | assemble)
                                     |
   remote inputs --> cache/raw  -->  cache/derived/*.npy  -->  cache/predictors.zarr
   (~1.1 GB)         (gitignored)      (per stage)             360 x 720 x 6 predictors
                                     |                              |
                                     |                    src/compute/similarity.py
                                     |                     compute_similarity / rank_top
                                     |                              |
                                     |                    src/agents/rationale.py
                                     |                       explain_cell (claims + caveats)
                                     |                              |
                                     |                    src/compute/gazetteer.py
                                     |                       region / nearest-place label
                                     |                              |
                                     +------------------> src/api/main.py  -->  web/
                                                            FastAPI           test UI
```

Each pipeline stage is independently cacheable, so you can rebuild one input
without redoing the rest:

```bash
python -m src.acquire.build_predictor_stack --stage dem
python -m src.acquire.build_predictor_stack --stage lst --force
python -m src.acquire.build_predictor_stack --stage assemble
python -m src.acquire.build_predictor_stack --offline
```

---

## Repository layout

| Path | What it is |
| --- | --- |
| `src/acquire/download.py` | URL -> `cache/raw` fetcher, `OFFLINE=1` support, atomic writes |
| `src/acquire/build_predictor_stack.py` | The five-stage build: terrain, aridity, climate, LST, assemble |
| `src/compute/similarity.py` | The scoring model: ranges, weights, similarities, ranking, determinism guard |
| `src/compute/gazetteer.py` | Offline labelling: 59 region envelopes + 7,342 Natural Earth places |
| `src/agents/rationale.py` | Rule-based explainer: headline, per-criterion claims, caveats |
| `src/api/main.py` | FastAPI app; every endpoint is a thin wrapper over the two modules above |
| `data/normalization.json` | Fixed min/max ranges, units, labels, default weights |
| `data/targets.json` | Target profiles; every number carries `dataset_id` + `source_url` |
| `data/gazetteer.json` | Hand-compiled physiographic envelopes (labelling only) |
| `web/index.html`, `web/app.js` | Vanilla-JS test UI, no framework and no CDN |
| `tests/test_similarity.py` | Model behaviour: maths, weights, NaN handling, determinism |
| `tests/test_api.py` | HTTP surface: endpoints, error paths, provenance contract |
| `cache/` | `raw/` (ignored), `derived/` and `predictors.zarr` (committed) |
| `prototype_v1/` | Legacy scratch work, gitignored |

---

## The grid and the land mask

* CRS `EPSG:4326`, **360 rows x 720 columns**, 0.5° cells, row-major north to
  south then west to east.
* Cell centre: `lat = 90 - (row + 0.5) * 0.5`, `lon = -179.75 + col * 0.5`.
* A cell is a **candidate** when `land_fraction >= 0.5`, where
  `land_fraction` is the share of 30 arc-second Aridity Index pixels in the
  cell carrying a valid value (the source encodes open ocean as `0` and fill
  as `65535`). Every predictor is set to NaN below that threshold, so
  `similarity.valid_mask(stack)` *is* the land mask.
* **61,260** cells are candidates; **61,134** have all six predictors finite
  and are scorable.
* Terrain analysis happens on a 1/12° grid (4,320 x 8,640) built from 256
  Web Mercator tiles at zoom 4, then block-averaged onto the 0.5° grid.

The masking rule is also written into `cache/predictors.zarr` under
`attrs["land_rule"]`, next to `attrs["grid"]`, `attrs["normalization"]`,
`attrs["weights"]` and `attrs["sources"]`.

---

## Predictors

| Key | Label | Range (min .. max) | Unit | Dataset id | Source |
| --- | --- | --- | --- | --- | --- |
| `aridity` | Aridity index | 0 .. 1 | 1 | `global_ai_et0_v3_1` | https://doi.org/10.6084/m9.figshare.7504448 |
| `annual_temperature_range` | Annual temperature range | 0 .. 160 | K | `worldclim_2_1_tavg` | https://doi.org/10.1038/s41597-018-0002-1 |
| `elevation` | Elevation | -3000 .. 9000 | m | `aws_terrain_tiles_terrarium` | https://registry.opendata.aws/terrain-tiles/ |
| `slope` | Regional slope | 0 .. 10 | degrees | `aws_terrain_tiles_terrarium` | https://registry.opendata.aws/terrain-tiles/ |
| `roughness` | Surface roughness | 0 .. 400 | m | `aws_terrain_tiles_terrarium` | https://registry.opendata.aws/terrain-tiles/ |
| `lst_diurnal_range` | LST diurnal range | 0 .. 130 | K | `zenodo_modis_lst_1km_2000_2020` | https://doi.org/10.5281/zenodo.6458406 |

How each one is derived:

* **Elevation** - Terrarium RGB tiles decoded to metres, reprojected to the
  1/12° grid, masked to land, block-meaned to 0.5°.
* **Slope** - horizontal gradient of the 0.5°-smoothed elevation field, in
  degrees. It deliberately describes regional tilt, not local cliff angle.
* **Roughness** - RMS residual of the 1/12° DEM about a least-squares plane
  fitted over a 3x3 neighbourhood (~9.3 km spacing, ~28 km window),
  block-averaged onto the cell.
* **Aridity** - Global AI v3.1, rescaled `value * 1e-4`; ocean `0` and fill
  `65535` are dropped before averaging.
* **Annual temperature range** - WorldClim 2.1 warmest month mean minus
  coldest month mean, 10 arc-min source averaged to the cell.
* **LST diurnal range** - MODIS daytime minus nighttime LST, `value * 0.02`
  K per count, warped from its native sinusoidal grid to the cell grid.

Ranges live in `data/normalization.json` and are chosen **once at build time**
from the observed Earth distribution plus the envelope of the target profiles.
They are never derived from the data being scored, which is what makes scores
reproducible and comparable between runs. Where the Earth distribution extends
far past every target value - aridity, where all targets sit near 0 - the range
is deliberately narrowed to the part of the distribution that can still match a
target, so resolution near the target is preserved instead of saturating.

---

## Target profiles

Two targets ship in `data/targets.json`:

| id | Body | Site |
| --- | --- | --- |
| `lunar_south_pole` | Moon | Lunar South Pole (Shackleton rim) |
| `jezero_crater` | Mars | Jezero Crater, Mars |

Each entry holds a `criteria` map whose every value is an object with
`value`, `unit`, `dataset_id`, `source_url`, `definition_note` and
`confidence` (plus optional `baseline`). `tests/test_similarity.py::test_target_profile_is_complete_and_cited`
fails if any of that is missing, so provenance cannot rot silently.

---

## Scoring model

For criterion `k` with stored range `[min_k, max_k]` and target value `t_k`:

```
s_k(x)     = clip(1 - |x_k - t_k| / (max_k - min_k), 0, 1)
score      = sum_k w_k * s_k  /  sum_k w_k
contrib_k  = w_k * s_k        /  sum_k w_k          (contributions sum to score)
```

* Weights default to `1.0` for every criterion (uniform). Partial weight maps
  are accepted; missing keys default to `1`. Zero weight excludes a criterion.
  `NaN` in any predictor makes the cell's score `NaN` and it is masked out.
* Ranking uses `rank_top`, which walks cells by descending score and enforces a
  Chebyshev `min_separation_cells` radius so the top 20 are not 20 neighbours
  of one place.
* Everything is `float64` in, `float32` only at the encoding boundary.

---

## Rationale and provenance

`src/agents/rationale.py` is a **rule-based explainer, not a language model**.
For a scored cell it returns:

* `headline` - score against the target's short name and body, plus the label.
* `claims` - one per finite criterion, ordered by similarity, each stating the
  Earth value, the target value, a qualitative band and the similarity. Every
  claim carries `dataset_id` and `source_url`.
* `caveats` - emitted when a target criterion has `confidence != "high"`, a
  non-empty `baseline`, or a non-empty `definition_note`, plus the target's
  `units_note` about comparing bodies with different datums.
* `drivers` - the top three criteria by similarity.

The structure is deliberately LLM-shaped so a model could verbalise the same
payload later without an API change.

---

## HTTP API

Base URL `/`. All responses are JSON. Score surfaces and predictor grids are
base64-encoded little-endian `float32`, so the browser decodes them straight
into a `Float32Array`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Readiness: grid shape, candidate count, offline flag, gazetteer status |
| `GET` | `/api/criteria` | Normalisation ranges, units, descriptions and weight shares |
| `GET` | `/api/targets` | The target catalogue with full criteria provenance |
| `POST` | `/api/score` | Score every land cell for a target or a custom profile; returns ranked cells + rationales |
| `GET` | `/api/scorefield` | Whole 360 x 720 score surface as base64 float32 |
| `GET` | `/api/predictor` | One raw predictor grid (`?key=elevation`) |
| `GET` | `/api/cell` | Predictors, label, ranges and sources for one coordinate |
| `GET` | `/api/sources` | Dataset ids and URLs behind every number the API returns |
| `GET` | `/` | Static `web/` UI |

### `POST /api/score`

```json
{
  "target_id": "lunar_south_pole",
  "criteria": null,
  "weights": {"aridity": 3},
  "top_k": 20,
  "min_separation_cells": 2
}
```

* `target_id` **or** `criteria` (a full six-key profile) is required;
  supplying both means `criteria` wins.
* Responses include `score_range`, and for each result: `rank`, `row`/`col`,
  `lat`/`lon`, `score`, `label`, raw `values`, per-criterion `similarities`
  and the full `rationale`.
* Errors: `404` unknown target, `422` missing/unknown criteria or bad weights.

```bash
curl -s -X POST http://127.0.0.1:8000/api/score \
  -H "content-type: application/json" \
  -d '{"target_id":"jezero_crater","top_k":5,"min_separation_cells":2}'
```

### `GET /api/scorefield`

`?target_id=...` and optional `?weights=aridity:3,slope:0.5` (comma-separated
`key:weight` pairs). Returns `{shape, encoding, min, max, data}`.

---

## Test UI

`web/index.html` + `web/app.js` is a single-page vanilla-JS harness with no
framework and no CDN:

* target picker, per-criterion value sliders and weight sliders,
* a canvas layer that shows either the score surface or any raw predictor,
* click-to-inspect a cell (`/api/cell`),
* a ranked table with the rationale expanded inline,
* header badges for health, offline mode and candidate cell count.

---

## Caching and offline mode

`src/acquire/download.fetch` is the only thing that touches the network.

* Every download lands in `cache/raw/` (gitignored, ~1.1 GB) and is reused on
  the next run.
* `OFFLINE=1` (or `offline=True`, or `--offline`) forbids the network; a
  missing cache entry raises `FetchError` instead of silently succeeding.
* `cache/derived/*.npy` are per-stage intermediates and are committed so
  `--stage assemble` works without the raw inputs.
* `cache/predictors.zarr` is committed so the API and tests run on a fresh
  clone.

---

## Tests and lint

```bash
python -m pytest tests -q          # 30 tests, ~2 s, no network
python -m ruff check src tests conftest.py
python -m ruff format src tests conftest.py
```

Ruff configuration lives in `ruff.toml` (`E4,E7,E9,F,I,UP,SIM,RUF`,
line length 100). `conftest.py` puts the repo root on `sys.path`, so tests
import `src.*` regardless of how pytest is invoked.

---

## Known limitations

Be honest about these when presenting:

1. **Inland seas are not excluded.** The mask only excludes open ocean, so the
   Caspian Depression reads as a top candidate (elevation -804 m at
   38.25°N, 50.75°E). Fix: mask cells with no outlet, or require a minimum
   distance to a coastline.
2. **Antarctica and the far north are out of coverage.** The Aridity Index has
   no data below about 60°S, so `land_fraction` is 0 there and every Antarctic
   cell is NaN. Land rows only span 83.75°N to 59.25°S; cells above ~84°N are
   also unmapped. The Web Mercator DEM itself stops at ±85.06°.
3. **Baselines are not interchangeable.** Lunar and Martian values are
   measured over different periods, instruments and datums than the Earth
   predictors. The API attaches a `units_note` caveat to every rationale, but
   the ranking is a screening tool, not a site survey.
4. **`slope` is regional, not local.** It is the gradient of the 0.5°-smoothed
   elevation field, so a cliff inside an otherwise flat cell scores as flat.
5. **Labelling is best-effort.** Region envelopes are circles and overlap;
   a cell is labelled by the smallest envelope containing it, falling back to
   the nearest of 7,342 populated places.

---

## Extending it

**Add a target.** Append an entry to `data/targets.json` with `id`, `name`,
`body`, `latitude`, `longitude`, `summary`, `source_url`, `units_note` and a
`criteria` map covering all six keys. `pytest` enforces the provenance fields.
Nothing else changes - `/api/targets`, `/api/score` and the UI picker read the
file.

**Add a criterion.** Five places must agree, and `tests/test_similarity.py`
will tell you:

1. `src/compute/similarity.py` - add the key to `CRITERIA`.
2. `data/normalization.json` - add `{min, max, unit, label, description}` and a
   weight.
3. `src/acquire/build_predictor_stack.py` - add a stage that produces the
   array, and include it in `assemble()`.
4. `data/targets.json` - add the value for every target.
5. `src/api/main.py` - extend `_DATASET_IDS` and `_SOURCE_URLS`.

**Change a range.** Edit `data/normalization.json`, then
`python -m src.acquire.build_predictor_stack --stage assemble` to rewrite
`cache/predictors.zarr` attrs. Scores and the UI follow automatically.

**Change ranking.** `rank_top` in `src/compute/similarity.py` is the single
place that decides which cells win; it currently uses a Chebyshev separation
radius.

---

## License

See [LICENSE](LICENSE). Raw datasets remain under their own terms - see the
`source_url` on every predictor and target.
