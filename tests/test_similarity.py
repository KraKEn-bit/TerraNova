from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.compute import similarity as sim

REPO = Path(__file__).resolve().parents[1]
RANGES = sim.load_normalization(REPO / "data" / "normalization.json")
TARGETS = json.loads((REPO / "data" / "targets.json").read_text(encoding="utf-8"))["targets"]
LUNAR = next(t for t in TARGETS if t["id"] == "lunar_south_pole")
JEZERO = next(t for t in TARGETS if t["id"] == "jezero_crater")


def _stack(values: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Build a 1xN predictor stack from flat per-criterion sequences."""
    return {key: np.asarray(seq, dtype=np.float64).reshape(1, -1) for key, seq in values.items()}


def _values(target: dict) -> dict[str, float]:
    """Flatten a targets.json entry into a plain criterion -> number profile."""
    criteria = target.get("criteria", target)
    out = {}
    for key in sim.CRITERIA:
        entry = criteria[key]
        out[key] = float(entry["value"] if isinstance(entry, dict) else entry)
    return out


def _outside_range(key: str, target: float) -> float:
    spec = RANGES[key]
    return target + spec.span


def _cell_missing_everything(target: dict[str, float]) -> dict[str, float]:
    return {key: _outside_range(key, float(target[key])) for key in sim.CRITERIA}


@pytest.mark.parametrize("target", [LUNAR, JEZERO], ids=lambda t: t["id"])
def test_exact_match_scores_one(target):
    """A cell whose predictor values equal the target profile scores exactly 1.0."""
    profile = _values(target)
    stack = _stack({key: [profile[key]] for key in sim.CRITERIA})
    result = sim.compute_similarity(stack, profile, ranges=RANGES)

    assert result.score.shape == (1, 1)
    assert float(result.score[0, 0]) == pytest.approx(1.0, abs=0.0)
    for key in sim.CRITERIA:
        assert float(result.similarities[key][0, 0]) == pytest.approx(1.0)
        assert float(result.contributions[key][0, 0]) >= 0.0
    assert float(sum(result.contributions[key][0, 0] for key in sim.CRITERIA)) == pytest.approx(
        float(result.score[0, 0])
    )


def test_exact_match_survives_arbitrary_positive_weights():
    profile = _values(LUNAR)
    stack = _stack({key: [profile[key]] for key in sim.CRITERIA})
    weights = {"aridity": 7.5, "slope": 0.25, "roughness": 13.0}
    result = sim.compute_similarity(stack, profile, weights=weights, ranges=RANGES)
    assert float(result.score[0, 0]) == pytest.approx(1.0)


def test_zero_similarity_when_predictor_is_one_full_range_away():
    profile = _values(LUNAR)
    far = {key: _outside_range(key, profile[key]) for key in sim.CRITERIA}
    stack = _stack({key: [far[key]] for key in sim.CRITERIA})
    result = sim.compute_similarity(stack, profile, ranges=RANGES)
    assert float(result.score[0, 0]) == pytest.approx(0.0)
    for key in sim.CRITERIA:
        assert float(result.similarities[key][0, 0]) == pytest.approx(0.0)


def test_similarity_decays_linearly_halfway_across_the_range():
    profile = _values(LUNAR)
    key = "slope"
    spec = RANGES[key]
    halfway = profile[key] + spec.span / 2.0
    stack = _stack({k: [profile[k]] for k in sim.CRITERIA})
    stack[key] = np.asarray([[halfway]], dtype=np.float64)
    result = sim.compute_similarity(stack, profile, ranges=RANGES)
    assert float(result.similarities[key][0, 0]) == pytest.approx(0.5)


def test_changing_a_weight_changes_the_ranking_as_expected():
    """Raising the aridity weight must promote the aridity-matching cell."""
    profile = _values(LUNAR)

    # Cell A: perfect aridity match, zero slope match.
    cell_a = _cell_missing_everything(profile)
    cell_a["aridity"] = profile["aridity"]
    cell_a["slope"] = _outside_range("slope", profile["slope"])

    # Cell B: partial aridity match (0.4), perfect slope match.
    cell_b = _cell_missing_everything(profile)
    cell_b["aridity"] = profile["aridity"] + 0.4 * RANGES["aridity"].span
    cell_b["slope"] = profile["slope"]

    stack = _stack({key: [cell_a[key], cell_b[key]] for key in sim.CRITERIA})

    equal = sim.compute_similarity(stack, profile, ranges=RANGES)
    equal_order = sim.rank_top(equal.score, mask=None, k=2)
    assert equal_order[0] == 1, "with equal weights the better slope match must win"
    assert equal.score[0, 1] > equal.score[0, 0]

    aridity_heavy = sim.compute_similarity(stack, profile, weights={"aridity": 10.0}, ranges=RANGES)
    heavy_order = sim.rank_top(aridity_heavy.score, mask=None, k=2)
    assert heavy_order[0] == 0, "with aridity weighted 10x the aridity match must win"
    assert aridity_heavy.score[0, 0] > aridity_heavy.score[0, 1]

    assert equal_order != heavy_order


def test_zero_weight_excludes_a_criterion_from_the_score():
    profile = _values(LUNAR)
    stack = _stack({key: [profile[key]] for key in sim.CRITERIA})
    stack["slope"] = np.asarray([[_outside_range("slope", profile["slope"])]])

    all_on = sim.compute_similarity(stack, profile, ranges=RANGES)
    slope_off = sim.compute_similarity(stack, profile, weights={"slope": 0.0}, ranges=RANGES)
    assert float(slope_off.score[0, 0]) == pytest.approx(1.0)
    assert float(all_on.score[0, 0]) < 1.0
    assert float(slope_off.contributions["slope"][0, 0]) == 0.0


def test_contributions_sum_to_the_score_everywhere():
    rng = np.random.default_rng(20260927)
    profile = _values(JEZERO)
    stack = {
        key: rng.uniform(RANGES[key].min, RANGES[key].max, size=(12, 17)) for key in sim.CRITERIA
    }
    result = sim.compute_similarity(stack, profile, ranges=RANGES)
    stacked = np.sum(np.stack([result.contributions[k] for k in sim.CRITERIA]), axis=0)
    np.testing.assert_allclose(stacked, result.score, rtol=0, atol=1e-12)
    assert np.nanmin(result.score) >= 0.0
    assert np.nanmax(result.score) <= 1.0


def test_nan_predictors_produce_nan_scores_and_are_masked_out():
    profile = _values(LUNAR)
    stack = _stack({key: [profile[key]] for key in sim.CRITERIA})
    stack["elevation"] = np.asarray([[np.nan]])
    result = sim.compute_similarity(stack, profile, ranges=RANGES)
    assert np.isnan(result.score[0, 0])
    assert not sim.valid_mask(stack)[0, 0]


def test_computation_is_deterministic():
    rng = np.random.default_rng(7)
    profile = _values(LUNAR)
    stack = {
        key: rng.uniform(RANGES[key].min, RANGES[key].max, size=(9, 11)) for key in sim.CRITERIA
    }
    sim.assert_deterministic(stack, profile)


@pytest.mark.parametrize("weights", [{}, None])
def test_default_weights_are_uniform(weights):
    normalised = sim.validate_weights(weights)
    assert set(normalised) == set(sim.CRITERIA)
    total = sum(normalised.values())
    for value in normalised.values():
        assert value / total == pytest.approx(1.0 / len(sim.CRITERIA))


def test_partial_weights_fill_in_the_rest_with_one():
    normalised = sim.validate_weights({"slope": 3.0})
    assert normalised["slope"] == pytest.approx(3.0)
    for key in sim.CRITERIA:
        if key != "slope":
            assert normalised[key] == pytest.approx(1.0)
    total = sum(normalised.values())
    assert normalised["slope"] / total > normalised["aridity"] / total


def test_invalid_weights_are_rejected():
    with pytest.raises(sim.SimilarityError):
        sim.validate_weights({"aridity": -1.0})
    with pytest.raises(sim.SimilarityError):
        sim.validate_weights({"aridity": float("nan")})
    with pytest.raises(sim.SimilarityError):
        sim.validate_weights({key: 0.0 for key in sim.CRITERIA})
    with pytest.raises(sim.SimilarityError):
        sim.validate_weights({"not_a_criterion": 1.0})


def test_missing_predictor_or_target_is_rejected():
    profile = _values(LUNAR)
    stack = _stack({key: [1.0] for key in sim.CRITERIA if key != "slope"})
    with pytest.raises(sim.SimilarityError):
        sim.compute_similarity(stack, profile, ranges=RANGES)

    stack = _stack({key: [1.0] for key in sim.CRITERIA})
    partial = {k: 1.0 for k in sim.CRITERIA if k != "slope"}
    with pytest.raises(sim.SimilarityError):
        sim.compute_similarity(stack, partial, ranges=RANGES)


def test_stored_ranges_are_complete_and_positive():
    assert set(RANGES) == set(sim.CRITERIA)
    for key, spec in RANGES.items():
        assert spec.span > 0, key
        assert spec.min < spec.max


def test_normalisation_ranges_round_trip_through_zarr(tmp_path):
    import zarr

    store = tmp_path / "predictors.zarr"
    root = zarr.open_group(str(store), mode="w")
    root.attrs["normalization"] = {
        key: {"min": spec.min, "max": spec.max} for key, spec in RANGES.items()
    }
    reopened = sim.ranges_from_store(store)
    assert set(reopened) == set(RANGES)
    for key in sim.CRITERIA:
        assert reopened[key].min == pytest.approx(RANGES[key].min)
        assert reopened[key].max == pytest.approx(RANGES[key].max)


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t["id"])
def test_target_profile_is_complete_and_cited(target):
    """Evidence provenance: every numeric claim needs a dataset id and a source URL."""
    assert set(target["criteria"]) == set(sim.CRITERIA)
    assert target["source_url"].startswith("http")
    assert -90.0 <= target["latitude"] <= 90.0
    assert -180.0 <= target["longitude"] <= 180.0
    for key, entry in target["criteria"].items():
        assert isinstance(entry["value"], (int, float)), key
        assert entry["dataset_id"], key
        assert entry["source_url"].startswith("http"), key
        assert entry["definition_note"].strip(), key
        assert entry["confidence"] in {"high", "medium", "low"}, key


def test_rank_top_respects_mask_and_separation():
    score = np.zeros((4, 4))
    score[0, 0] = 1.0
    score[0, 1] = 0.99
    score[3, 3] = 0.5
    mask = np.ones_like(score, dtype=bool)
    mask[0, 1] = False
    top = sim.rank_top(score, mask=mask, k=5)
    assert top[0] == 0
    assert 1 not in top
    assert 15 in top

    dense = sim.rank_top(score, mask=None, k=5, min_separation_cells=1)
    assert dense[0] == 0
    assert dense[1] != 1
