# Earth Analogue Finder — System Architecture & Implementation Guide

This document provides a comprehensive, structured explanation of the **Earth Analogue Finder** codebase. After reading this guide, you will be able to explain the entire system—its vision, data pipeline, mathematical model, backend API, frontend UI, determinism, and trade-offs—to any stakeholder, judge, or engineer.

---

## 1. Executive Summary & Pitch

### What is the Earth Analogue Finder?
The **Earth Analogue Finder** is a deterministic geospatial search system that ranks Earth's land surface against extraterrestrial base-site target profiles (such as the **Lunar South Pole / Shackleton Rim** or **Mars' Jezero Crater**). It identifies 0.5° (~55 km at equator) grid cells on Earth whose terrain, aridity, temperature range, surface roughness, and diurnal thermal behaviour best match the target site.

### Key Highlights & Core Value Proposition
1. **100% Deterministic Arithmetic**: There are no language models, no neural networks, no random numbers, and no runtime network calls. The exact same input parameters will *always* produce byte-identical floating-point scores and word-for-word identical explanations.
2. **Fixed Normalisation Boundaries**: Predictor values are normalized against fixed global ranges established at build time, preventing score drift and allowing scores to be compared across runs.
3. **Full Provenance Contract**: Every single numeric claim and target criterion carries an explicit `dataset_id` and `source_url`.
4. **Offline-First Data Pipeline**: Pre-processed derived predictor arrays and the final `zarr` dataset (`cache/predictors.zarr`) are committed to the repository, enabling instantaneous offline startup without downloading ~1.1 GB of raw data.
5. **High-Performance Binary Transfer**: Whole-world score surfaces ($360 \times 720$ cells) are transferred over HTTP as base64-encoded `float32` little-endian buffers, which the frontend decodes directly into native `Float32Array` objects for instant Canvas rendering.

---

## 2. End-to-End System Architecture

```
                                  DATA ACQUISITION & PROCESSING PIPELINE
              src/acquire/build_predictor_stack.py (dem | aridity | climate | lst | assemble)
                                                  |
 Remote Rasters (AWS, Figshare, UCDavis, Zenodo) -> cache/raw -> cache/derived/*.npy -> cache/predictors.zarr
 (~1.1 GB raw inputs, fetched via download.py)    (gitignored)   (per-stage numpy)      (360 x 720 x 6 float32)
                                                                                             |
                                                                                             v
+------------------------------------------------------------------------------------------------------------------+
|                                                   HTTP BACKEND                                                   |
|                                                src/api/main.py (FastAPI)                                        |
|                                                                                                                  |
|    +-----------------------------+     +-----------------------------+     +-------------------------------+     |
|    |  src/compute/similarity.py  |     |   src/agents/rationale.py   |     |    src/compute/gazetteer.py    |     |
|    | Multi-criteria scoring,     | --> | Rule-based explainer engine | --> | Offline geocoding engine      |     |
|    | Chebyshev spatial ranking   |     | (Headline, Claims, Caveats) |     | (59 regions + 7,342 cities)   |     |
|    +-----------------------------+     +-----------------------------+     +-------------------------------+     |
+------------------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
                                              WEB FRONTEND
                                       web/index.html + web/app.js
                            (Vanilla JS, HTML5 Canvas, zero-dependency UI)
```

---

## 3. Data Pipeline & Predictor Stack (`src/acquire/`)

The core dataset is built by downloading, decoding, warping, and block-averaging remote rasters into a unified **$360 \times 720$ float32 array** stored in `cache/predictors.zarr`.

### The 6 Predictor Variables

| Predictor Key | Human Label | Stored Range ($\text{min} .. \text{max}$) | Unit | Source Dataset ID | Primary Source URL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `aridity` | Aridity Index | $0.0 .. 1.0$ | $1$ | `global_ai_et0_v3_1` | [Figshare DOI](https://doi.org/10.6084/m9.figshare.7504448) |
| `annual_temperature_range` | Annual Temperature Range | $0.0 .. 160.0$ | $\text{K}$ | `worldclim_2_1_tavg` | [Nature Sci Data DOI](https://doi.org/10.1038/s41597-018-0002-1) |
| `elevation` | Surface Elevation | $-3000.0 .. 9000.0$ | $\text{m}$ | `aws_terrain_tiles_terrarium` | [AWS Open Data](https://registry.opendata.aws/terrain-tiles/) |
| `slope` | Regional Slope | $0.0 .. 10.0$ | $\text{deg}$ | `aws_terrain_tiles_terrarium` | [AWS Open Data](https://registry.opendata.aws/terrain-tiles/) |
| `roughness` | Surface Roughness | $0.0 .. 400.0$ | $\text{m}$ | `aws_terrain_tiles_terrarium` | [AWS Open Data](https://registry.opendata.aws/terrain-tiles/) |
| `lst_diurnal_range` | LST Diurnal Range | $0.0 .. 130.0$ | $\text{K}$ | `zenodo_modis_lst_1km_2000_2020` | [Zenodo DOI](https://doi.org/10.5281/zenodo.6458406) |

### Stage-by-Stage Processing Details

#### 1. Terrain Stage (`stage_terrain`)
- **Source**: 256 Mapzen Terrarium PNG tiles at Zoom level 4 ($256 \times 256$ px each, forming a $4096 \times 4096$ Mercator grid).
- **RGB Decoding**: $\text{Elevation (m)} = (R \times 256 + G + B / 256) - 32768$.
- **Reprojection**: Reprojected from Web Mercator `EPSG:3857` to a fine $1/12^\circ$ lat/lon grid ($2160 \times 4320$ pixels, ~9.3 km at equator).
- **Elevation**: 6x6 block mean of the fine DEM onto the $0.5^\circ$ grid ($360 \times 720$).
- **Regional Slope**: Magnitude of horizontal gradient computed across the $0.5^\circ$-smoothed elevation field using `np.gradient`, converting $\text{m}/\text{m}$ to degrees using latitude-dependent longitudinal distance adjustments ($111,320 \text{ m} \times \cos(\text{lat})$).
- **Surface Roughness**: RMS height residual of the $1/12^\circ$ DEM about a least-squares plane fitted over a $3 \times 3$ neighbourhood (~28 km window).

#### 2. Aridity Stage (`stage_aridity`)
- **Source**: Global Aridity Index v3.1 (30 arc-second resolution, geotiff integer values).
- **Transformation**: Scaled by $10^{-4}$ ($\text{P}/\text{PET}$). Ocean value `0` and fill value `65535` are filtered out.
- **Land Mask Generation**: Computes `land_fraction` per $0.5^\circ$ cell as the proportion of valid 30 arc-sec pixels.

#### 3. Climate Stage (`stage_climate`)
- **Source**: WorldClim 2.1 (10 arc-minute resolution monthly mean temperatures for 12 months).
- **Transformation**: $\text{Warmest Month Mean} - \text{Coldest Month Mean}$, then $3 \times 3$ block averaged to $0.5^\circ$.

#### 4. Land Surface Temperature Stage (`stage_lst`)
- **Source**: MODIS 1 km long-term (2000–2020) day and night LST composites from Zenodo.
- **Transformation**: Warped to $0.5^\circ$ grid, calculating $(\text{LST}_{\text{day}} - \text{LST}_{\text{night}}) \times 0.02\text{ K}$.

#### 5. Assembly Stage (`assemble`)
- Enforces the **Candidate Land Rule**: Any grid cell with `land_fraction < 0.5` has all predictors set to `NaN`.
- Writes all arrays, grid metadata, normalisation ranges, default weights, land rules, and source attributions into `cache/predictors.zarr`.

---

## 4. Grid Specification & Candidate Mask

- **Coordinate Reference System (CRS)**: `EPSG:4326` (WGS 84 plate carrée).
- **Dimensions**: 360 rows $\times$ 720 columns ($0.5^\circ \times 0.5^\circ$ resolution).
- **Cell Center Formula**:
  $$\text{lat}_i = 90^\circ - (i + 0.5) \times 0.5^\circ$$
  $$\text{lon}_j = -180^\circ + (j + 0.5) \times 0.5^\circ$$
- **Cell Coverage Stats**:
  - Total Grid Cells: $360 \times 720 = 259,200$
  - Candidate Land Cells ($\text{land\_fraction} \ge 0.5$): **61,260**
  - Fully Scorable Land Cells (all 6 predictors finite): **61,134**

---

## 5. Mathematical Scoring Model (`src/compute/similarity.py`)

Given a target profile vector $\mathbf{t} = (t_1, t_2, \dots, t_6)$, predictor vector $\mathbf{x} = (x_1, x_2, \dots, x_6)$ for a cell, and user weight vector $\mathbf{w} = (w_1, w_2, \dots, w_6)$:

### 1. Per-Criterion Linear Similarity $s_k(x)$
For criterion $k$ with fixed bounds $[\text{min}_k, \text{max}_k]$ and span $\Delta_k = \text{max}_k - \text{min}_k$:
$$s_k(x) = \text{clip}\left(1 - \frac{|x_k - t_k|}{\Delta_k}, 0, 1\right)$$
- If $x_k = t_k$, $s_k(x) = 1.0$ (perfect match).
- If $|x_k - t_k| \ge \Delta_k$, $s_k(x) = 0.0$ (zero match).
- Similarity decays linearly with absolute distance.

### 2. Weighted Overall Score
$$\text{Score}(\mathbf{x}) = \frac{\sum_{k=1}^6 w_k \cdot s_k(x)}{\sum_{k=1}^6 w_k}$$

### 3. Individual Criterion Contribution
$$\text{Contrib}_k(\mathbf{x}) = \frac{w_k \cdot s_k(x)}{\sum_{j=1}^6 w_j}$$
*Property*: $\sum_{k=1}^6 \text{Contrib}_k(\mathbf{x}) = \text{Score}(\mathbf{x})$.

### 4. Ranking & Spatial Separation (`rank_top`)
- Cells are sorted descending by score.
- To prevent a single geographical plateau (e.g. 20 neighbouring cells in the Andes) from clogging the top ranking, `rank_top` applies a **Chebyshev distance filter** (`min_separation_cells`, default = 2 cells $\approx 110$ km):
  $$\max(|\text{row}_1 - \text{row}_2|, |\text{col}_1 - \text{col}_2|) > \text{min\_separation\_cells}$$

---

## 6. Target Profiles & Provenance (`data/targets.json`)

The system ships with two target profiles. Every single criterion value includes complete source documentation:

```json
{
  "id": "lunar_south_pole",
  "name": "Lunar South Pole (Shackleton rim)",
  "short_name": "Lunar South Pole",
  "body": "Moon",
  "source_url": "https://iopscience.iop.org/article/10.3847/PSJ/aca590",
  "criteria": {
    "aridity": { "value": 0.0, "unit": "1", "dataset_id": "nssdc_moon_fact_sheet", "source_url": "https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html", "confidence": "high" },
    "annual_temperature_range": { "value": 100.0, "unit": "K", "dataset_id": "diviner_polar_seasonal_maps", "source_url": "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JE006028", "confidence": "medium" },
    "elevation": { "value": 1290.0, "unit": "m", "dataset_id": "lola_5m_south_polar_dem", "source_url": "https://repository.hou.usra.edu/server/api/core/bitstreams/e56463a3-68f0-49ab-986f-b16a240e2768/content", "confidence": "high" },
    "slope": { "value": 9.5, "unit": "degrees", "dataset_id": "change2_dem_20m_south_polar", "source_url": "https://www.mdpi.com/2072-4292/14/19/4863", "confidence": "high" },
    "roughness": { "value": 1.0, "unit": "m", "dataset_id": "lola_shackleton_rms_roughness", "source_url": "https://ntrs.nasa.gov/api/citations/20120013758/downloads/20120013758.pdf", "confidence": "high" },
    "lst_diurnal_range": { "value": 120.0, "unit": "K", "dataset_id": "diviner_global_diurnal_lst", "source_url": "https://www.sciencedirect.com/science/article/pii/S0019103516304869", "confidence": "medium" }
  }
}
```

---

## 7. Deterministic Explainer Engine (`src/agents/rationale.py`)

`src/agents/rationale.py` is a **rule-based explainer**, not an LLM. It generates deterministic rationale structures for scored cells:

### Output Structure
1. **`headline`**: High-level summary string (e.g. `Scores 87% against Lunar South Pole (Moon) - Atacama Desert, Chile.`).
2. **`claims`**: Ordered list of statements for all 6 criteria, matching physical values against target values and mapping similarities to qualitative bands:
   - $\ge 90\%$: *excellent match*
   - $\ge 75\%$: *strong match*
   - $\ge 50\%$: *moderate match*
   - $\ge 25\%$: *weak match*
   - $< 25\%$: *poor match*
   Each claim carries `dataset_id` and `source_url`.
3. **`caveats`**: Automatically attached notes explaining measurement baseline differences (e.g. LOLA 5m roughness vs 28km Earth DEM window) or datum differences (Earth sea level vs Moon mean radius vs Mars areoid).
4. **`drivers`**: Keys of top 3 criteria contributing most to the similarity.

---

## 8. Offline Gazetteer & Reverse Geocoding (`src/compute/gazetteer.py`)

To label grid cells without calling external geocoding APIs, the system builds an offline spatial lookup index combining two dataset layers:
1. **Physiographic Region Envelopes (`data/gazetteer.json`)**: 59 hand-compiled circular bounding areas (e.g. Atacama Desert, Tibetan Plateau, Qaidam Basin). Smallest radius matches first.
2. **Natural Earth Populated Places**: 7,342 settlements indexed inside a spatial `scipy.spatial.cKDTree` using 3D ECEF cartesian coordinates ($x, y, z$). Great-circle distances are computed using the Haversine formula on a spherical Earth ($R = 6371.0088\text{ km}$).

---

## 9. HTTP API Reference (`src/api/main.py`)

Built with **FastAPI**, served via **Uvicorn**.

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Readiness probe: checks grid shape, candidate cell count, offline state, and gazetteer status. |
| `GET` | `/api/criteria` | Returns normalisation ranges, units, descriptions, and default weight shares. |
| `GET` | `/api/targets` | Returns full target catalogue along with complete criterion-level provenance citations. |
| `POST` | `/api/score` | Main scoring endpoint. Accepts target ID or custom profile + weights; returns top-K ranked cells + rationales. |
| `GET` | `/api/scorefield` | Returns entire $360 \times 720$ score matrix as base64-encoded `float32` little-endian binary stream. |
| `GET` | `/api/predictor` | Returns raw global raster grid for a predictor (e.g. `elevation`) as base64 `float32`. |
| `GET` | `/api/cell` | Returns exact predictor values, normalisation ranges, label, and dataset sources for a single (lat, lon). |
| `GET` | `/api/sources` | Lists dataset IDs and URLs behind every predictor and target criterion. |
| `GET` | `/` | Serves the single-page test frontend from `web/`. |

---

## 10. Frontend Implementation (`web/`)

- **Tech Stack**: Vanilla JavaScript (ES6+), HTML5 Canvas, CSS3. Zero external frameworks, zero npm packages, zero CDN dependencies.
- **Canvas Rendering**: Decodes base64 `float32` responses from `/api/scorefield` or `/api/predictor` into native `Float32Array` buffers. Each cell is mapped to a color gradient and painted directly to an HTML5 canvas.
- **Interactive Inspection**: Click events on the canvas convert $(x, y)$ pixels to $(\text{lat}, \text{lon})$ coordinates and fetch detailed cell diagnostics via `/api/cell`.

---

## 11. Testing & Verification Suite (`tests/`)

- **Test Suite**: 30 pytest tests executing in ~2 seconds with zero network access.
- **Coverage**:
  - `test_exact_match_scores_one`: Validates that matching target values yield a score of exactly $1.0$.
  - `test_zero_similarity_when_predictor_is_one_full_range_away`: Confirms proper bounding.
  - `test_contributions_sum_to_the_score_everywhere`: Verifies mathematical contribution identity $\sum \text{contrib}_k = \text{score}$.
  - `test_target_profile_is_complete_and_cited`: Enforces provenance policy (fails if any target criterion lacks `dataset_id` or `source_url`).
  - `test_score_is_deterministic`: Ensures repeated evaluation yields byte-identical output.

---

## 12. Known Limitations & Trade-Offs

1. **Inland Seas / Endorheic Basins**: The land mask relies on the Aridity Index land availability, which includes landlocked depressions (e.g. Caspian Depression at -804m elevation).
2. **Polar Data Cutoffs**: The Aridity Index ends around 60°S and 83.75°N. Antarctica is unmapped in the source dataset, so those cells are marked `NaN`.
3. **Regional vs Local Slope**: `slope` represents the gradient of the $0.5^\circ$-smoothed elevation field (~55 km scale), reflecting macro regional tilt rather than micro cliff angles.
4. **Cross-Body Datums**: Elevations are relative to each body's reference datum (Earth sea level, Lunar mean radius, Mars areoid) and are not directly interchangeable without contextual caveat notes.

---

## 13. How to Present This Project (Q&A & Elevator Pitches)

### 30-Second Elevator Pitch
> *"The Earth Analogue Finder is a deterministic geospatial search system built for planetary scientists and mission planners. Give it an extraterrestrial location like Mars' Jezero Crater or the Lunar South Pole, and it instantly searches all 61,000+ scorable land cells on Earth to find the top terrestrial analogs matching its terrain, climate, aridity, and thermal behavior. Everything is 100% deterministic, backed by source citations down to every single number, and runs completely offline."*

### Key Questions & How to Answer Them

**Q: Why not use an LLM or Machine Learning model for scoring?**
> **Answer**: Scientific analogue matching requires strict reproducibility, mathematical transparency, and verifiable provenance. Machine learning models can introduce non-deterministic variance, hallucinations, and unquantifiable bias. Our system uses a multi-criteria linear similarity model with fixed normalisation ranges, guaranteeing byte-identical, reproducible results every time.

**Q: How does the system handle binary map transfers so quickly?**
> **Answer**: Instead of transferring massive JSON arrays of 259,200 floating-point numbers over HTTP, we encode the raw `float32` byte array in base64. The frontend decodes this directly into a JavaScript `Float32Array` in a few milliseconds and paints it to an HTML5 Canvas.

**Q: How do you verify the accuracy of the target data?**
> **Answer**: We enforce an automated test contract in `tests/test_similarity.py` (`test_target_profile_is_complete_and_cited`). Every single criterion in `data/targets.json` must be bound to a published peer-reviewed paper or NASA/ESA mission dataset, complete with dataset ID, source URL, confidence metric, and definition notes.
