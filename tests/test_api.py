"""End-to-end checks for the HTTP surface.

These use FastAPI's in-process TestClient, so they need no running server and
no network. They cover the read-only endpoints, the ranking endpoint, the
error paths and the provenance contract (every numeric claim carries a dataset
id and a source url).
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.compute.similarity import CRITERIA

client = TestClient(app)


def _decode(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype="<f4").reshape(payload["shape"])


# --------------------------------------------------------------------------
# read-only endpoints
# --------------------------------------------------------------------------


def test_static_ui_is_served():
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    script = client.get("/app.js")
    assert script.status_code == 200
    assert client.get("/missing.js").status_code == 404


def test_health_reports_the_grid():
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["shape"] == [360, 720]
    assert set(body["criteria"]) == set(CRITERIA)
    assert body["candidate_cells"] > 50_000
    assert body["gazetteer"] is True


def test_criteria_expose_ranges_and_weight_shares():
    body = client.get("/api/criteria").json()
    assert [item["key"] for item in body["criteria"]] == list(CRITERIA)
    for item in body["criteria"]:
        assert item["min"] < item["max"], item["key"]
        assert item["label"], item["key"]
        assert item["unit"] is not None, item["key"]
    assert [item["key"] for item in body["weights"]] == list(CRITERIA)
    assert sum(item["share"] for item in body["weights"]) == pytest.approx(1.0)


def test_targets_and_sources_carry_provenance():
    targets = client.get("/api/targets").json()["targets"]
    assert {"lunar_south_pole", "jezero_crater"} <= {t["id"] for t in targets}

    sources = client.get("/api/sources").json()
    for entry in sources["predictors"]:
        assert entry["dataset_id"], entry["key"]
        assert entry["source_url"].startswith("http"), entry["key"]
    for target in sources["targets"]:
        assert target["source_url"].startswith("http"), target["id"]
        for claim in target["criteria"].values():
            assert claim["dataset_id"] and claim["source_url"]


def test_predictor_grid_round_trips():
    payload = client.get("/api/predictor", params={"key": "elevation"}).json()
    grid = _decode(payload)
    assert grid.shape == (360, 720)
    assert np.isnan(grid).any(), "ocean must be NaN"
    assert np.isfinite(grid).any()
    assert payload["min"] <= payload["max"]
    assert client.get("/api/predictor", params={"key": "nope"}).status_code == 404


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def test_score_ranks_a_target_with_rationales():
    body = client.post(
        "/api/score",
        json={"target_id": "lunar_south_pole", "top_k": 5, "min_separation_cells": 2},
    ).json()

    assert body["target_id"] == "lunar_south_pole"
    assert 0.0 <= body["score_range"][0] <= body["score_range"][1] <= 1.0
    assert [row["rank"] for row in body["results"]] == [1, 2, 3, 4, 5]
    for row in body["results"]:
        assert 0.0 <= row["score"] <= 1.0
        assert row["label"]["text"]
        assert set(row["values"]) == set(CRITERIA)
        assert set(row["similarities"]) == set(CRITERIA)
        assert all(0.0 <= value <= 1.0 for value in row["similarities"].values())
        rationale = row["rationale"]
        assert rationale["headline"]
        assert rationale["claims"], "a winner must be explained"
        for claim in rationale["claims"]:
            assert claim["dataset_id"] and claim["source_url"]


def test_score_is_deterministic():
    request = {"target_id": "jezero_crater", "top_k": 3}
    assert (
        client.post("/api/score", json=request).json()
        == client.post("/api/score", json=request).json()
    )


def test_custom_profile_gets_a_rationale_too():
    profile = {key: 0.5 for key in CRITERIA}
    body = client.post("/api/score", json={"criteria": profile, "top_k": 2}).json()
    assert body["target_id"] is None
    for row in body["results"]:
        assert row["rationale"]["claims"]


def test_score_rejects_bad_requests():
    assert client.post("/api/score", json={}).status_code == 422
    unknown = client.post("/api/score", json={"target_id": "nowhere"})
    assert unknown.status_code == 404
    partial = client.post("/api/score", json={"criteria": {"elevation": 1.0}})
    assert partial.status_code == 422
    extra = client.post(
        "/api/score", json={"criteria": {**{key: 1.0 for key in CRITERIA}, "nope": 1.0}}
    )
    assert extra.status_code == 422
    bad_weights = client.post(
        "/api/score", json={"target_id": "lunar_south_pole", "weights": {"aridity": -1.0}}
    )
    assert bad_weights.status_code == 422


def test_scorefield_and_cell():
    field = client.get("/api/scorefield", params={"target_id": "lunar_south_pole"}).json()
    surface = _decode(field)
    assert field["shape"] == [360, 720]
    assert field["min"] <= field["max"]
    assert np.isfinite(surface).any()

    ocean = client.get("/api/cell", params={"lat": 0.0, "lon": -140.0}).json()
    assert ocean["candidate"] is False
    assert ocean["values"]["elevation"] is None

    assert client.get("/api/cell", params={"lat": 999.0, "lon": 0.0}).status_code == 422
