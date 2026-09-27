"""Deterministic multi-criteria similarity for Earth-analogue search.

Every number produced here comes from arithmetic on the predictor stack and
the fixed normalisation ranges. No language model, heuristic text model or
random number generator is involved: the same inputs always give byte-identical
outputs.

Math
----
For criterion ``k`` with fixed stored range ``[lo_k, hi_k]`` and target value
``t_k``::

    s_k(x) = clip(1 - |x_k - t_k| / (hi_k - lo_k), 0, 1)

The weighted score for a cell is the normalised weighted mean of the per
criterion similarities::

    score       = sum_k w_k * s_k  /  sum_k w_k
    contrib_k   = w_k * s_k        /  sum_k w_k          (sums to score)
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CRITERIA: tuple[str, ...] = (
    "aridity",
    "annual_temperature_range",
    "elevation",
    "slope",
    "roughness",
    "lst_diurnal_range",
)

DEFAULT_WEIGHTS: dict[str, float] = {key: 1.0 for key in CRITERIA}


class SimilarityError(ValueError):
    """Raised when a request cannot be scored deterministically."""


@dataclass(frozen=True)
class CriterionRange:
    """Fixed normalisation range for one criterion."""

    key: str
    min: float
    max: float
    label: str = ""
    unit: str = ""
    description: str = ""

    @property
    def span(self) -> float:
        return float(self.max) - float(self.min)

    def validate(self) -> CriterionRange:
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise SimilarityError(f"non-finite range for {self.key}")
        if self.span <= 0:
            raise SimilarityError(
                f"range for {self.key} must have max > min, got [{self.min}, {self.max}]"
            )
        return self


@dataclass(frozen=True)
class SimilarityResult:
    """Score surface plus per-criterion diagnostics for one target."""

    score: np.ndarray
    similarities: Mapping[str, np.ndarray]
    contributions: Mapping[str, np.ndarray]
    weights: Mapping[str, float]
    target: Mapping[str, float]
    ranges: Mapping[str, CriterionRange] = field(repr=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.score.shape)


def ranges_from_mapping(raw: Mapping[str, Mapping[str, object]]) -> dict[str, CriterionRange]:
    """Build CriterionRange objects from the ``criteria`` block of normalization.json."""
    ranges: dict[str, CriterionRange] = {}
    for key in CRITERIA:
        if key not in raw:
            raise SimilarityError(f"missing normalisation entry for criterion {key!r}")
        entry = raw[key]
        ranges[key] = CriterionRange(
            key=key,
            min=float(entry["min"]),  # type: ignore[arg-type]
            max=float(entry["max"]),  # type: ignore[arg-type]
            label=str(entry.get("label", key)),
            unit=str(entry.get("unit", "")),
            description=str(entry.get("description", "")),
        ).validate()
    return ranges


def load_normalization(path: str | Path | None = None) -> dict[str, CriterionRange]:
    """Load the fixed stored ranges.

    Preference order: explicit path, ``data/normalization.json`` next to the
    repository root, then the attributes of ``cache/predictors.zarr``.
    """
    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return ranges_from_mapping(raw["criteria"])

    here = Path(__file__).resolve()
    repo = here.parents[2]
    candidate = repo / "data" / "normalization.json"
    if candidate.exists():
        raw = json.loads(candidate.read_text(encoding="utf-8"))
        return ranges_from_mapping(raw["criteria"])

    store = repo / "cache" / "predictors.zarr"
    return ranges_from_store(store)


def ranges_from_store(store_path: str | Path) -> dict[str, CriterionRange]:
    """Read normalisation ranges back out of the zarr root attributes."""
    import zarr

    root = zarr.open_group(str(store_path), mode="r")
    if "normalization" not in root.attrs:
        raise SimilarityError(f"{store_path} carries no normalisation attributes")
    return ranges_from_mapping(root.attrs["normalization"])


def validate_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    """Return weights for every criterion, rejecting unusable input."""
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(CRITERIA)
        if unknown:
            raise SimilarityError(f"unknown criteria in weights: {sorted(unknown)}")
        merged.update({key: float(value) for key, value in weights.items()})

    for key, value in merged.items():
        if not math.isfinite(value) or value < 0:
            raise SimilarityError(f"weight for {key} must be finite and >= 0, got {value}")
    if sum(merged.values()) <= 0:
        raise SimilarityError("at least one weight must be > 0")
    return merged


def criterion_similarity(values: np.ndarray, target: float, spec: CriterionRange) -> np.ndarray:
    """Similarity of every cell to the target for a single criterion."""
    values = np.asarray(values, dtype=np.float64)
    target = float(target)
    if not math.isfinite(target):
        raise SimilarityError(f"target value for {spec.key} is not finite: {target}")
    distance = np.abs(values - target) / spec.span
    similarity = 1.0 - distance
    np.clip(similarity, 0.0, 1.0, out=similarity)
    return similarity


def _as_stack(
    stack: Mapping[str, np.ndarray], ranges: Mapping[str, CriterionRange]
) -> dict[str, np.ndarray]:
    missing = [key for key in CRITERIA if key not in stack]
    if missing:
        raise SimilarityError(f"predictor stack is missing {missing}")
    arrays = {key: np.asarray(stack[key], dtype=np.float64) for key in CRITERIA}
    shape = arrays[CRITERIA[0]].shape
    for key, arr in arrays.items():
        if arr.shape != shape:
            raise SimilarityError(f"predictor {key} has shape {arr.shape}, expected {shape}")
    if len(shape) != 2:
        raise SimilarityError(f"predictors must be 2-D lat/lon arrays, got {shape}")
    if not set(ranges) >= set(CRITERIA):
        raise SimilarityError("ranges must cover every criterion")
    return arrays


def compute_similarity(
    stack: Mapping[str, np.ndarray],
    target: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    ranges: Mapping[str, CriterionRange] | None = None,
) -> SimilarityResult:
    """Score every grid cell against a target profile.

    Parameters
    ----------
    stack:
        Mapping of criterion key to a 2-D (lat, lon) array of raw predictor
        values, as stored in ``cache/predictors.zarr``.
    target:
        Mapping of criterion key to the target value.
    weights:
        Per-criterion weights; defaults to equal weights. Normalised internally.
    ranges:
        Fixed stored ranges; loaded from ``data/normalization.json`` when omitted.
    """
    ranges = dict(ranges) if ranges is not None else load_normalization()
    arrays = _as_stack(stack, ranges)
    norm_weights = validate_weights(weights)

    missing_targets = [key for key in CRITERIA if key not in target]
    if missing_targets:
        raise SimilarityError(f"target profile is missing {missing_targets}")

    weight_total = sum(norm_weights[key] for key in CRITERIA)

    similarities: dict[str, np.ndarray] = {}
    contributions: dict[str, np.ndarray] = {}
    score = np.zeros(arrays[CRITERIA[0]].shape, dtype=np.float64)

    for key in CRITERIA:
        similarity = criterion_similarity(arrays[key], float(target[key]), ranges[key])
        contribution = (norm_weights[key] / weight_total) * similarity
        similarities[key] = similarity
        contributions[key] = contribution
        score += contribution

    valid = np.ones(score.shape, dtype=bool)
    for key in CRITERIA:
        valid &= np.isfinite(arrays[key])
    score = np.where(valid, score, np.nan)

    # Round away float accumulation noise so an exact match reports exactly 1.0
    # and repeated runs are bit-identical across platforms.
    score = np.round(score, 12)

    return SimilarityResult(
        score=score,
        similarities=similarities,
        contributions=contributions,
        weights=norm_weights,
        target={key: float(target[key]) for key in CRITERIA},
        ranges=ranges,
    )


def valid_mask(stack: Mapping[str, np.ndarray]) -> np.ndarray:
    """Cells where every criterion is finite (i.e. land cells with full coverage)."""
    arrays = _as_stack(stack, {key: CriterionRange(key, 0, 1) for key in CRITERIA})
    mask = np.ones(arrays[CRITERIA[0]].shape, dtype=bool)
    for key in CRITERIA:
        mask &= np.isfinite(arrays[key])
    return mask


def rank_top(
    score: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 20,
    min_separation_cells: int = 0,
) -> list[int]:
    """Return flat indices of the ``k`` highest scoring cells, best first.

    ``min_separation_cells`` optionally suppresses near-duplicate winners so a
    single broad plateau cannot fill the whole ranking.
    """
    values = np.asarray(score, dtype=np.float64).ravel()
    if mask is not None:
        values = np.where(np.asarray(mask, dtype=bool).ravel(), values, -np.inf)
    finite = np.isfinite(values)
    if not finite.any():
        return []

    order = np.argsort(-values, kind="stable")
    order = order[finite[order]]

    if min_separation_cells <= 0:
        return [int(i) for i in order[:k]]

    flat_mask = (
        np.ones(values.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).ravel()
    )
    width = int(np.asarray(score).shape[1])
    chosen: list[int] = []
    for index in order:
        if len(chosen) >= k:
            break
        row, col = divmod(int(index), width)
        too_close = False
        for other in chosen:
            orow, ocol = divmod(other, width)
            if abs(orow - row) <= min_separation_cells and abs(ocol - col) <= min_separation_cells:
                too_close = True
                break
        if not too_close and flat_mask[index]:
            chosen.append(int(index))
    return chosen


def normalize_weights_report(
    weights: Mapping[str, float] | None = None,
    ranges: Mapping[str, CriterionRange] | None = None,
) -> list[dict[str, object]]:
    """Normalised weights plus their share of the score, for the UI."""
    ranges = ranges if ranges is not None else load_normalization()
    norm = validate_weights(weights)
    total = sum(norm[key] for key in CRITERIA)
    return [
        {
            "key": key,
            "label": ranges[key].label,
            "unit": ranges[key].unit,
            "weight": norm[key],
            "share": norm[key] / total,
            "range_min": ranges[key].min,
            "range_max": ranges[key].max,
        }
        for key in CRITERIA
    ]


def assert_deterministic(
    stack: Mapping[str, np.ndarray],
    target: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    repeats: int = 3,
) -> None:
    """Guard used by tests: repeated evaluation must be bit-identical."""
    first = compute_similarity(stack, target, weights)
    for _ in range(repeats - 1):
        again = compute_similarity(stack, target, weights)
        if not np.array_equal(first.score, again.score, equal_nan=True):
            raise SimilarityError("similarity is not deterministic")
        for key in CRITERIA:
            if not np.array_equal(
                first.contributions[key], again.contributions[key], equal_nan=True
            ):
                raise SimilarityError(f"contribution for {key} is not deterministic")
