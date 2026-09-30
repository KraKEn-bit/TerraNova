from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.agents.rationale import load_targets
from src.compute import similarity as sim

REPO = Path(__file__).resolve().parents[1]
RANGES = sim.load_normalization(REPO / "data" / "normalization.json")
RAW_TARGETS = json.loads((REPO / "data" / "targets.json").read_text(encoding="utf-8"))["targets"]
TARGETS = load_targets(ranges=RANGES)
LUNAR = TARGETS["lunar_south_pole"]
JEZERO = TARGETS["jezero_crater"]


def _stack(values: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Build a 1xN predictor stack from flat per-criterion sequences."""
    return {key: np.asarray(seq, dtype=np.float64).reshape(1, -1) for key, seq in values.items()}


def _values(target: dict) -> dict[str, float | None]:
    """Resolved target profile: criterion -> number, or None when not used."""
    out = {}
    for key in sim.CRITERIA:
        value = target["criteria"][key]["value"]
        out[key] = None if value is None else float(value)
    return out


def _effective(profile: dict[str, float | None]) -> dict[str, float]:
    """Predictor values that match the profile exactly (unused criteria: any number)."""
    return {
        key: RANGES[key].min if profile[key] is None else RANGES[key].effective_target(profile[key])
        for key in sim.CRITERIA
    }


def _at_similarity(key: str, profile: dict[str, float], similarity: float) -> float:
    """A predictor value whose similarity to the (effective) target is ``similarity``."""
    return _effective(profile)[key] + (1.0 - similarity) * RANGES[key].span


@pytest.mark.parametrize("target", [LUNAR, JEZERO], ids=lambda t: t["id"])
def test_exact_match_scores_one(target):
    """A cell whose predictors equal the effective target profile scores exactly 1.0."""
    profile = _values(target)
    effective = _effective(profile)
    stack = _stack({key: [effective[key]] for key in sim.CRITERIA})
    result = sim.compute_similarity(stack, profile, ranges=RANGES)

    assert result.score.shape == (1, 1)
    assert float(result.score[0, 0]) == pytest.approx(1.0, abs=0.0)
    for key in sim.CRITERIA:
        if profile[key] is None:
            assert np.isnan(result.similarities[key][0, 0])
        else:
            assert float(result.similarities[key][0, 0]) == pytest.approx(1.0)


def test_unused_criteria_are_ignored_whatever_their_weight():
    profile = _values(TARGETS["haworth_psr"])
    assert profile["slope"] is None and profile["mean_annual_temperature"] is not None
    effective = _effective(profile)
    stack = _stack({key: [effective[key]] for key in sim.CRITERIA})
    stack["slope"] = np.asarray([[RANGES["slope"].max * 5]])  # far from anything
    result = sim.compute_similarity(stack, profile, weights={"slope": 3.0}, ranges=RANGES)
    assert result.weights["slope"] == 0.0
    assert result.effective_target["slope"] is None
    assert float(result.score[0, 0]) == pytest.approx(1.0)


def test_a_profile_with_no_usable_criterion_is_rejected():
    profile = dict.fromkeys(sim.CRITERIA)
    profile["precipitation"] = 0.0
    stack = _stack({key: [1.0] for key in sim.CRITERIA})
    with pytest.raises(sim.SimilarityError):
        sim.compute_similarity(stack, profile, weights={"precipitation": 0.0}, ranges=RANGES)


def test_targets_beyond_earth_are_clamped_to_the_earth_envelope():
    profile = _values(LUNAR)
    spec = RANGES["lst_diurnal_range"]
    assert profile["lst_diurnal_range"] > spec.earth_max
    assert spec.effective_target(profile["lst_diurnal_range"]) == pytest.approx(spec.earth_max)
    assert spec.effective_target(spec.earth_min - 5.0) == pytest.approx(spec.earth_min)
    assert spec.effective_target(spec.earth_max - 1.0) == pytest.approx(spec.earth_max - 1.0)

    stack = _stack({key: [_effective(profile)[key]] for key in sim.CRITERIA})
    result = sim.compute_similarity(stack, profile, ranges=RANGES)
    assert result.target["lst_diurnal_range"] == pytest.approx(120.0)
    assert result.effective_target["lst_diurnal_range"] == pytest.approx(spec.earth_max)


def test_percentile_targets_resolve_through_earth_quantiles():
    raw = next(t for t in RAW_TARGETS if t["id"] == "lunar_south_pole")["criteria"]["slope"]
    assert "value" not in raw and raw["earth_percentile"] == 85
    expected = RANGES["slope"].value_at_percentile(85)
    assert LUNAR["criteria"]["slope"]["value"] == pytest.approx(expected, abs=1e-4)
    quantiles = RANGES["slope"].earth_quantiles
    assert RANGES["slope"].value_at_percentile(0) == pytest.approx(quantiles[0])
    assert RANGES["slope"].value_at_percentile(100) == pytest.approx(quantiles[-1])


def test_exact_match_survives_arbitrary_positive_weights():
    profile = _values(LUNAR)
    effective = _effective(profile)
    stack = _stack({key: [effective[key]] for key in sim.CRITERIA})
    weights = {"precipitation": 7.5, "slope": 0.25, "roughness": 13.0}
    result = sim.compute_similarity(stack, profile, weights=weights, ranges=RANGES)
    assert float(result.score[0, 0]) == pytest.approx(1.0)


def test_one_criterion_with_no_match_vetoes_the_cell():
    """Geometric mean: a single zero similarity makes the whole score zero."""
    profile = _values(LUNAR)
    effective = _effective(profile)
    values = {key: [effective[key]] for key in sim.CRITERIA}
    values["vegetation"] = [_at_similarity("vegetation", profile, 0.0)]
    result = sim.compute_similarity(_stack(values), profile, ranges=RANGES)
    assert float(result.similarities["vegetation"][0, 0]) == pytest.approx(0.0)
    assert float(result.score[0, 0]) == pytest.approx(0.0)


def test_similarity_decays_linearly_halfway_across_the_range():
    profile = _values(LUNAR)
    effective = _effective(profile)
    stack = _stack({key: [effective[key]] for key in sim.CRITERIA})
    stack["slope"] = np.asarray([[effective["slope"] + RANGES["slope"].span / 2.0]])
    result = sim.compute_similarity(stack, profile, ranges=RANGES)
    assert float(result.similarities["slope"][0, 0]) == pytest.approx(0.5)


def test_score_is_the_weighted_geometric_mean():
    profile = _values(JEZERO)
    effective = _effective(profile)
    values = {key: [effective[key]] for key in sim.CRITERIA}
    values["precipitation"] = [_at_similarity("precipitation", profile, 0.25)]
    result = sim.compute_similarity(_stack(values), profile, ranges=RANGES)
    total = sum(sim.DEFAULT_WEIGHTS.values())
    expected = 0.25 ** (sim.DEFAULT_WEIGHTS["precipitation"] / total)
    assert float(result.score[0, 0]) == pytest.approx(expected)


def test_changing_a_weight_changes_the_ranking_as_expected():
    """Raising the precipitation weight must promote the precipitation-matching cell."""
    profile = _values(LUNAR)
    effective = _effective(profile)
    cell_a = dict(effective)  # perfect precipitation, poor slope
    cell_a["slope"] = _at_similarity("slope", profile, 0.2)
    cell_b = dict(effective)  # partial precipitation, perfect slope
    cell_b["precipitation"] = _at_similarity("precipitation", profile, 0.6)
    stack = _stack({key: [cell_a[key], cell_b[key]] for key in sim.CRITERIA})

    equal = sim.compute_similarity(stack, profile, ranges=RANGES)
    assert sim.rank_top(equal.score, mask=None, k=2)[0] == 1

    heavy = sim.compute_similarity(stack, profile, weights={"precipitation": 10.0}, ranges=RANGES)
    assert sim.rank_top(heavy.score, mask=None, k=2)[0] == 0


def test_zero_weight_excludes_a_criterion_from_the_score():
    profile = _values(LUNAR)
    effective = _effective(profile)
    stack = _stack({key: [effective[key]] for key in sim.CRITERIA})
    stack["slope"] = np.asarray([[_at_similarity("slope", profile, 0.0)]])

    all_on = sim.compute_similarity(stack, profile, ranges=RANGES)
    slope_off = sim.compute_similarity(stack, profile, weights={"slope": 0.0}, ranges=RANGES)
    assert float(slope_off.score[0, 0]) == pytest.approx(1.0)
    assert float(all_on.score[0, 0]) == pytest.approx(0.0)
    assert float(slope_off.factors["slope"][0, 0]) == 1.0


def test_factors_multiply_to_the_score_everywhere():
    rng = np.random.default_rng(20260927)
    profile = _values(JEZERO)
    stack = {
        key: rng.uniform(RANGES[key].min, RANGES[key].max, size=(12, 17)) for key in sim.CRITERIA
    }
    result = sim.compute_similarity(stack, profile, ranges=RANGES)
    product = np.prod(np.stack([result.factors[k] for k in sim.CRITERIA]), axis=0)
    np.testing.assert_allclose(product, result.score, rtol=0, atol=1e-11)
    assert np.nanmin(result.score) >= 0.0
    assert np.nanmax(result.score) <= 1.0


def test_nan_predictors_produce_nan_scores_and_are_masked_out():
    profile = _values(LUNAR)
    stack = _stack({key: [value] for key, value in _effective(profile).items()})
    stack["roughness"] = np.asarray([[np.nan]])
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
def test_default_weights_come_from_normalization(weights):
    stored = json.loads((REPO / "data" / "normalization.json").read_text(encoding="utf-8"))
    normalised = sim.validate_weights(weights)
    assert set(normalised) == set(sim.CRITERIA)
    for key in sim.CRITERIA:
        assert normalised[key] == pytest.approx(stored["weights"][key])
    assert normalised["elevation"] == 0.0, "elevation is not comparable across bodies"


def test_partial_weights_fill_in_the_rest_with_defaults():
    normalised = sim.validate_weights({"slope": 3.0})
    assert normalised["slope"] == pytest.approx(3.0)
    for key in sim.CRITERIA:
        if key != "slope":
            assert normalised[key] == pytest.approx(sim.DEFAULT_WEIGHTS[key])


def test_invalid_weights_are_rejected():
    with pytest.raises(sim.SimilarityError):
        sim.validate_weights({"precipitation": -1.0})
    with pytest.raises(sim.SimilarityError):
        sim.validate_weights({"precipitation": float("nan")})
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


def test_stored_ranges_and_earth_envelope_are_consistent():
    assert set(RANGES) == set(sim.CRITERIA)
    for key, spec in RANGES.items():
        assert spec.span > 0, key
        assert spec.earth_min <= spec.earth_max, key
        assert len(spec.earth_quantiles) == len(sim.QUANTILE_LEVELS), key
        assert list(spec.earth_quantiles) == sorted(spec.earth_quantiles), key


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


@pytest.mark.parametrize("target", RAW_TARGETS, ids=lambda t: t["id"])
def test_target_profile_is_complete_and_cited(target):
    """Evidence provenance: every criterion needs a dataset id, a source URL and a note."""
    assert set(target["criteria"]) == set(sim.CRITERIA)
    assert target["source_url"].startswith("http")
    assert -90.0 <= target["latitude"] <= 90.0
    assert -180.0 <= target["longitude"] <= 180.0
    assert any(
        entry.get("value") is not None or "earth_percentile" in entry
        for entry in target["criteria"].values()
    ), "a target needs at least one usable criterion"
    for key, entry in target["criteria"].items():
        if "earth_percentile" in entry:
            assert 0 <= entry["earth_percentile"] <= 100, key
            assert entry["measured_value"] is None or isinstance(
                entry["measured_value"], (int, float)
            ), key
        else:
            assert entry["value"] is None or isinstance(entry["value"], (int, float)), key
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
