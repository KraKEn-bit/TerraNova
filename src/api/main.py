"""HTTP API for the Earth-analogue finder.

Run with::

    uvicorn src.api.main:app --reload

Every endpoint is a thin wrapper over :mod:`src.compute.similarity` plus the
deterministic explainer in :mod:`src.agents.rationale`; nothing here calls a
model or the network once ``cache/raw`` is populated. Score surfaces and
predictor grids are transferred as base64-encoded little-endian float32 so the
browser can decode them into a ``Float32Array`` without a parser.
"""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np
import zarr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from src.acquire.download import REPO_ROOT, offline_mode
from src.agents.rationale import explain_cell, load_targets
from src.compute.gazetteer import Gazetteer, default_gazetteer
from src.compute.similarity import (
    CRITERIA,
    CriterionRange,
    SimilarityError,
    compute_similarity,
    load_normalization,
    normalize_weights_report,
    rank_top,
    valid_mask,
)

_DATASET_IDS: dict[str, str] = {
    "aridity": "global_ai_et0_v3_1",
    "annual_temperature_range": "worldclim_2_1_tavg",
    "elevation": "aws_terrain_tiles_terrarium",
    "slope": "aws_terrain_tiles_terrarium",
    "roughness": "aws_terrain_tiles_terrarium",
    "lst_diurnal_range": "zenodo_modis_lst_1km_2000_2020",
}

_SOURCE_URLS: dict[str, str] = {
    "aridity": "https://doi.org/10.6084/m9.figshare.7504448",
    "annual_temperature_range": "https://doi.org/10.1038/s41597-018-0002-1",
    "elevation": "https://registry.opendata.aws/terrain-tiles/",
    "slope": "https://registry.opendata.aws/terrain-tiles/",
    "roughness": "https://registry.opendata.aws/terrain-tiles/",
    "lst_diurnal_range": "https://doi.org/10.5281/zenodo.6458406",
}


def _sources() -> dict[str, dict[str, str]]:
    return {
        key: {"dataset_id": _DATASET_IDS[key], "source_url": _SOURCE_URLS[key]} for key in CRITERIA
    }


STACK_PATH = REPO_ROOT / "cache" / "predictors.zarr"
WEB_PATH = REPO_ROOT / "web"

app = FastAPI(
    title="Earth Analogue Finder",
    version="1.0.0",
    description=(
        "Deterministic multi-criteria similarity between extraterrestrial base "
        "site targets and Earth's surface, scored on a 0.5 degree grid."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreRequest(BaseModel):
    """Body for :func:`score`."""

    target_id: str | None = Field(
        default=None, description="Key into data/targets.json; ignored when criteria is given."
    )
    criteria: dict[str, float] | None = Field(
        default=None, description="Explicit target profile overriding target_id."
    )
    weights: dict[str, float] | None = Field(default=None, description="Per-criterion weights.")
    top_k: int = Field(default=20, ge=1, le=500)
    min_separation_cells: int = Field(
        default=2, ge=0, le=20, description="Suppress near-duplicate winners."
    )


class _State:
    """Lazily loaded, process-wide read-only cache."""

    def __init__(self) -> None:
        self.stack: dict[str, np.ndarray] | None = None
        self.ranges: dict[str, CriterionRange] | None = None
        self.targets: dict[str, dict[str, Any]] | None = None
        self.gazetteer: Gazetteer | None = None

    def load(self) -> None:
        if self.stack is not None:
            return
        if not STACK_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{STACK_PATH.relative_to(REPO_ROOT)} is missing. "
                    "Run `python -m src.acquire.build_predictor_stack` first."
                ),
            )
        root = zarr.open_group(str(STACK_PATH), mode="r")
        self.stack = {key: np.asarray(root[key][:], dtype=np.float64) for key in CRITERIA}
        self.ranges = load_normalization()
        self.targets = load_targets()
        try:
            self.gazetteer = default_gazetteer(offline=offline_mode())
        except Exception:  # any failure simply disables labels
            self.gazetteer = None

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        self.load()
        assert self.stack is not None
        return self.stack


STATE = _State()


def _ranges() -> dict[str, CriterionRange]:
    """Normalisation ranges, preferring the ones already loaded with the stack."""
    return STATE.ranges or load_normalization()


def _targets() -> dict[str, dict[str, Any]]:
    """Target catalogue, preferring the copy already loaded with the stack."""
    return STATE.targets or load_targets()


def _encode(values: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(values, dtype="<f4").tobytes()).decode("ascii")


def _target_profile(request: ScoreRequest) -> tuple[dict[str, float], str | None]:
    if request.criteria is not None:
        missing = [key for key in CRITERIA if key not in request.criteria]
        if missing:
            raise HTTPException(status_code=422, detail=f"criteria missing {missing}")
        unknown = set(request.criteria) - set(CRITERIA)
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown criteria {sorted(unknown)}")
        return {key: float(request.criteria[key]) for key in CRITERIA}, None
    if not request.target_id:
        raise HTTPException(status_code=422, detail="provide target_id or criteria")
    targets = _targets()
    if request.target_id not in targets:
        raise HTTPException(status_code=404, detail=f"unknown target {request.target_id!r}")
    entry = targets[request.target_id]
    return {key: float(entry["criteria"][key]["value"]) for key in CRITERIA}, request.target_id


def _custom_entry(profile: dict[str, float]) -> dict[str, Any]:
    """Synthetic target record so custom profiles still get a provenance-backed rationale."""
    return {
        "id": "__custom__",
        "name": "Custom profile",
        "short_name": "the custom profile",
        "body": "user supplied",
        "criteria": {
            key: {
                "value": profile[key],
                "dataset_id": _DATASET_IDS[key],
                "source_url": _SOURCE_URLS[key],
                "confidence": "high",
                "definition_note": "",
                "unit": "",
            }
            for key in CRITERIA
        },
        "units_note": (
            "Predictors are Earth observations; values for other bodies are measured "
            "over different baselines and referenced to different datums. Treat the "
            "ranking as a screening tool, not a site survey."
        ),
        "source_url": "",
    }


def _label_for(lat: float, lon: float) -> dict[str, Any]:
    if STATE.gazetteer is None:
        return {"text": f"{lat:.2f}, {lon:.2f}", "kind": "coordinates", "source": "grid"}
    return STATE.gazetteer.label(lat, lon).as_dict()


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Readiness probe: stack present, offline flag, grid shape."""
    try:
        stack = STATE.arrays
    except HTTPException as exc:
        return {"ok": False, "detail": exc.detail}
    mask = valid_mask(stack)
    return {
        "ok": True,
        "offline": offline_mode(),
        "shape": list(next(iter(stack.values())).shape),
        "candidate_cells": int(mask.sum()),
        "criteria": list(CRITERIA),
        "gazetteer": STATE.gazetteer is not None,
    }


@app.get("/api/criteria")
def criteria() -> dict[str, Any]:
    """Normalisation ranges, units and the effective weight shares."""
    ranges = _ranges()
    return {
        "criteria": [
            {
                "key": key,
                "label": ranges[key].label,
                "unit": ranges[key].unit,
                "description": ranges[key].description,
                "min": ranges[key].min,
                "max": ranges[key].max,
            }
            for key in CRITERIA
        ],
        "weights": normalize_weights_report(),
    }


@app.get("/api/targets")
def targets() -> dict[str, Any]:
    """The extraterrestrial base-site target profiles."""
    catalogue = _targets()
    return {
        "targets": [
            {
                "id": entry["id"],
                "name": entry["name"],
                "short_name": entry["short_name"],
                "body": entry["body"],
                "latitude": entry["latitude"],
                "longitude": entry["longitude"],
                "summary": entry["summary"],
                "source_url": entry["source_url"],
                "source_label": entry.get("source_label"),
                "criteria": entry["criteria"],
            }
            for entry in catalogue.values()
        ]
    }


@app.post("/api/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    """Score every land cell against a target profile and rank the winners."""
    stack = STATE.arrays
    ranges = _ranges()
    profile, target_id = _target_profile(request)
    try:
        result = compute_similarity(stack, profile, request.weights, ranges)
    except SimilarityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mask = valid_mask(stack)
    indices = rank_top(
        result.score,
        mask,
        k=request.top_k,
        min_separation_cells=request.min_separation_cells,
    )
    if not indices:
        raise HTTPException(status_code=404, detail="no scorable cells")

    shape = result.score.shape
    catalogue = _targets()
    if target_id:
        entry = catalogue[target_id]
        explain_id, explain_targets = target_id, catalogue
    else:
        explain_id = "__custom__"
        entry = _custom_entry(profile)
        explain_targets = {explain_id: entry}

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(indices, start=1):
        row, col = divmod(int(index), shape[1])
        lat = 90.0 - (row + 0.5) * 0.5
        lon = -179.75 + col * 0.5
        label = _label_for(lat, lon)
        rationale = explain_cell(
            result,
            stack,
            explain_id,
            int(index),
            None,
            grid_shape=shape,
            targets=explain_targets,
        ).as_dict()
        rationale["headline"] = (
            f"Scores {float(result.score[row, col]):.0%} against "
            f"{entry['short_name']} ({entry['body']}) - {label['text']}."
        )
        results.append(
            {
                "rank": rank,
                "index": int(index),
                "row": row,
                "col": col,
                "lat": round(lat, 3),
                "lon": round(lon, 3),
                "score": round(float(result.score[row, col]), 6),
                "label": label,
                "values": {key: _round(stack[key][row, col]) for key in CRITERIA},
                "similarities": {
                    key: round(float(result.similarities[key][row, col]), 6) for key in CRITERIA
                },
                "rationale": rationale,
            }
        )

    finite = result.score[mask]
    finite = finite[np.isfinite(finite)]
    return {
        "target_id": target_id,
        "profile": profile,
        "weights": {key: result.weights[key] for key in CRITERIA},
        "score_range": [round(float(finite.min()), 6), round(float(finite.max()), 6)],
        "results": results,
    }


@app.get("/api/scorefield")
def scorefield(
    target_id: str | None = Query(default=None),
    weights: str | None = Query(default=None, description="Comma separated key:weight pairs."),
) -> dict[str, Any]:
    """The whole 360 x 720 score surface, base64 float32, for map rendering."""
    stack = STATE.arrays
    ranges = _ranges()
    if not target_id:
        target_id = next(iter(_targets()), None)
    profile, target_id = _target_profile(ScoreRequest(target_id=target_id, criteria=None))
    parsed_weights = _parse_weights(weights)
    try:
        result = compute_similarity(stack, profile, parsed_weights, ranges)
    except SimilarityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finite = result.score[np.isfinite(result.score)]
    return {
        "shape": list(result.score.shape),
        "encoding": "float32-le-base64",
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "data": _encode(result.score),
    }


@app.get("/api/predictor")
def predictor(key: str) -> dict[str, Any]:
    """One raw predictor grid, base64 float32 (NaN outside candidate land)."""
    stack = STATE.arrays
    if key not in stack:
        raise HTTPException(status_code=404, detail=f"unknown predictor {key!r}")
    values = stack[key]
    finite = values[np.isfinite(values)]
    return {
        "key": key,
        "shape": list(values.shape),
        "encoding": "float32-le-base64",
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "data": _encode(values),
    }


@app.get("/api/cell")
def cell(lat: float, lon: float) -> dict[str, Any]:
    """Predictors, label and provenance for a single coordinate."""
    stack = STATE.arrays
    ranges = _ranges()
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise HTTPException(status_code=422, detail="lat/lon out of range")
    row = int((90.0 - lat) // 0.5)
    col = int((lon + 180.0) // 0.5)
    row = min(max(row, 0), 359)
    col = min(max(col, 0), 719)
    label = _label_for(90.0 - (row + 0.5) * 0.5, -179.75 + col * 0.5)
    values = {key: _round(stack[key][row, col]) for key in CRITERIA}
    return {
        "row": row,
        "col": col,
        "lat": round(90.0 - (row + 0.5) * 0.5, 3),
        "lon": round(-179.75 + col * 0.5, 3),
        "label": label,
        "candidate": bool(all(np.isfinite(stack[key][row, col]) for key in CRITERIA)),
        "values": values,
        "ranges": {key: {"min": ranges[key].min, "max": ranges[key].max} for key in CRITERIA},
        "sources": _sources(),
    }


@app.get("/api/sources")
def sources() -> dict[str, Any]:
    """Dataset ids and URLs behind every number the API returns."""
    ranges = _ranges()
    catalogue = _targets()
    return {
        "predictors": [
            {
                "key": key,
                "dataset_id": _DATASET_IDS[key],
                "source_url": _SOURCE_URLS[key],
                "label": ranges[key].label,
            }
            for key in CRITERIA
        ],
        "targets": [
            {
                "id": entry["id"],
                "source_url": entry["source_url"],
                "source_label": entry.get("source_label"),
                "criteria": {
                    key: {
                        "dataset_id": value.get("dataset_id"),
                        "source_url": value.get("source_url"),
                        "confidence": value.get("confidence"),
                    }
                    for key, value in entry["criteria"].items()
                },
            }
            for entry in catalogue.values()
        ],
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _round(value: Any) -> float | None:
    number = float(value)
    return round(number, 4) if math.isfinite(number) else None


def _parse_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    parsed: dict[str, float] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        if ":" not in pair:
            raise HTTPException(status_code=422, detail=f"bad weight {pair!r}")
        key, value = pair.split(":", 1)
        parsed[key.strip()] = float(value)
    return parsed


if WEB_PATH.exists():
    app.mount("/", StaticFiles(directory=str(WEB_PATH), html=True), name="web")
