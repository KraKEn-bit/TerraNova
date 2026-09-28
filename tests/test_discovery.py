"""Discovery controls: tolerance bands, new-sites filter, spread, robustness."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.compute import similarity as sim
from src.compute import validation

client = TestClient(app)
RANGES = sim.load_normalization()


def test_tolerance_band_gives_full_similarity_inside_it():
    spec = RANGES["slope"]
    values = np.array([0.50, 0.60, 0.80])
    exact = sim.criterion_similarity(values, 0.5, spec)
    banded = sim.criterion_similarity(values, 0.5, spec, tolerance=0.1)
    assert exact[0] == 1.0 and exact[1] < 1.0
    assert banded[0] == 1.0 and banded[1] == pytest.approx(1.0)
    assert banded[2] == pytest.approx(1.0 - 0.2 / spec.span)
    with pytest.raises(sim.SimilarityError):
        sim.criterion_similarity(values, 0.5, spec, tolerance=-1)


def _score(**kw):
    body = {"target_id": "jezero_crater", "top_k": 10, "min_separation_cells": 3, **kw}
    return client.post("/api/score", json=body).json()


def test_new_only_excludes_known_and_near_known_sites():
    body = _score(target_id="haworth_psr", new_only=True, top_k=15)
    assert body["filters"]["new_only"] is True
    assert all(r["novelty"]["status"] == "new" for r in body["results"])


def test_minimum_distance_and_country_cap_spread_results():
    body = _score(min_distance_km=800, max_per_region=2)
    rows = body["results"]
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            assert validation.haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) >= 800
    countries = [r["label"].get("country") for r in rows]
    assert max(countries.count(c) for c in set(countries)) <= 2


def test_tolerance_never_lowers_scores_and_validation_is_unfiltered():
    exact = _score(include_field=True)
    loose = _score(include_field=True, tolerance=1.5, new_only=True)
    assert loose["score_range"][1] >= exact["score_range"][1]
    assert loose["validation"]["auc"] >= 0.9
    assert loose["filters"]["tolerance"] == 1.5


def test_robustness_reports_stability_and_sensitivity():
    body = client.post("/api/robustness", json={"target_id": "lunar_south_pole", "top_k": 5}).json()
    assert body["runs"] >= 20 and len(body["stability"]) == 5
    assert all(0.0 <= v <= 1.0 for v in body["stability"].values())
    left_out = {row["without"] for row in body["sensitivity"]}
    assert "vegetation" in left_out and "elevation" not in left_out
    again = client.post(
        "/api/robustness", json={"target_id": "lunar_south_pole", "top_k": 5}
    ).json()
    assert again == body, "fixed seed: repeated calls must agree"


def test_independent_datasets_agree():
    checks = client.get("/api/datachecks").json()["checks"]
    assert {c["id"] for c in checks} >= {"lst_vs_power", "rain_vs_green", "lat_vs_seasonality"}
    for c in checks:
        assert c["passed"], f"{c['id']} rho={c['rho']}"
        assert c["cells"] > 50_000
