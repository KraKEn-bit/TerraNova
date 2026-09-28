"""Validation against known analog sites, and the novelty label.

Deterministic, no model involved.

Validation
    For a score surface, every catalogued site (``data/known_analogs.json``) and
    every negative reference point is located on the grid and ranked. The
    headline number is ROC-AUC: the probability that a randomly chosen positive
    control outscores a randomly chosen negative one (Mann-Whitney U / n1 n2).
    Positives are the ``environment`` sites whose ``bodies`` include the target's
    body; ``geology`` sites are reported but not counted, because they were
    chosen for rocks the model does not measure.

Novelty
    A ranked cell is ``known`` when a catalogued site lies within
    ``KNOWN_KM`` of its centre, ``near_known`` within ``NEAR_KM``, else ``new``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "known_analogs.json"
EARTH_RADIUS_KM = 6371.0088
KNOWN_KM = 150.0
NEAR_KM = 500.0
SEARCH_CELLS = 2  # coastal sites may sit in a cell that is not >= 50 % land


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def cell_of(lat: float, lon: float, shape: tuple[int, int]) -> tuple[int, int]:
    nrow, ncol = shape
    row = min(max(int((90.0 - lat) / (180.0 / nrow)), 0), nrow - 1)
    col = min(max(int((lon + 180.0) / (360.0 / ncol)), 0), ncol - 1)
    return row, col


def nearest_scored(score: np.ndarray, lat: float, lon: float) -> tuple[int, int] | None:
    """The site's own cell, or the closest scored cell within SEARCH_CELLS."""
    row, col = cell_of(lat, lon, score.shape)
    if np.isfinite(score[row, col]):
        return row, col
    best: tuple[float, tuple[int, int]] | None = None
    for dr in range(-SEARCH_CELLS, SEARCH_CELLS + 1):
        for dc in range(-SEARCH_CELLS, SEARCH_CELLS + 1):
            r, c = row + dr, (col + dc) % score.shape[1]
            if 0 <= r < score.shape[0] and np.isfinite(score[r, c]):
                distance = dr * dr + dc * dc
                if best is None or distance < best[0]:
                    best = (distance, (r, c))
    return None if best is None else best[1]


def roc_auc(positives: list[float], negatives: list[float]) -> float | None:
    """Mann-Whitney AUC with ties counted as one half."""
    if not positives or not negatives:
        return None
    pos = np.asarray(positives, dtype=np.float64)[:, None]
    neg = np.asarray(negatives, dtype=np.float64)[None, :]
    wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
    return float(wins / (pos.size * neg.size))


@dataclass(frozen=True)
class Control:
    name: str
    role: str  # "positive", "geology" or "negative"
    lat: float
    lon: float
    score: float | None
    percentile: float | None
    source_url: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "lat": self.lat,
            "lon": self.lon,
            "score": None if self.score is None else round(self.score, 4),
            "percentile": None if self.percentile is None else round(self.percentile, 1),
            "source_url": self.source_url,
        }


def validate(
    score: np.ndarray,
    body: str,
    catalog: Mapping[str, Any] | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Rank every control on one score surface and summarise with ROC-AUC.

    With ``tag``, positives are the environment sites carrying that tag (any
    body); otherwise they are the environment sites of ``body``.
    """
    catalog = catalog or load_catalog()
    body = body.lower()
    finite = np.sort(score[np.isfinite(score)])

    def place(name: str, role: str, lat: float, lon: float, url: str | None) -> Control:
        cell = nearest_scored(score, lat, lon)
        if cell is None:
            return Control(name, role, lat, lon, None, None, url)
        value = float(score[cell])
        percentile = 100.0 * np.searchsorted(finite, value, side="left") / finite.size
        return Control(name, role, lat, lon, value, float(percentile), url)

    controls: list[Control] = []
    for site in catalog["sites"]:
        if tag:
            if tag not in site.get("tags", []):
                continue
        elif body not in site["bodies"]:
            continue
        role = "positive" if site["kind"] == "environment" else "geology"
        controls.append(place(site["name"], role, site["lat"], site["lon"], site["source_url"]))
    for site in catalog["negatives"]:
        controls.append(place(site["name"], "negative", site["lat"], site["lon"], None))

    pos = [c.score for c in controls if c.role == "positive" and c.score is not None]
    neg = [c.score for c in controls if c.role == "negative" and c.score is not None]
    pos_pct = [c.percentile for c in controls if c.role == "positive" and c.percentile is not None]
    return {
        "body": body,
        "tag": tag,
        "auc": None if (auc := roc_auc(pos, neg)) is None else round(auc, 4),
        "positives": len(pos),
        "negatives": len(neg),
        "median_positive_percentile": round(float(np.median(pos_pct)), 1) if pos_pct else None,
        "controls": [c.as_dict() for c in controls],
        "method": (
            "ROC-AUC of environment-analog sites for this body against densely vegetated "
            "or humid reference points; percentile is the share of scored land cells "
            "below the site's score."
        ),
    }


def _footprint(site: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Points representing a site: its centre, or a line of points for a region
    with an ``extent`` (e.g. the 1,600 km Atacama strip), one every ~25 km."""
    extent = site.get("extent")
    if not extent or extent.get("type") != "line":
        return [(site["lat"], site["lon"])]
    half = float(extent["length_km"]) / 2.0
    bearing = math.radians(float(extent.get("bearing_deg", 0.0)))
    n = max(2, int(half / 25.0))
    points = []
    for i in range(-n, n + 1):
        d = half * i / n
        dlat = d * math.cos(bearing) / 111.32
        dlon = d * math.sin(bearing) / (111.32 * max(0.05, math.cos(math.radians(site["lat"]))))
        points.append((site["lat"] + dlat, site["lon"] + dlon))
    return points


def distance_to_site(lat: float, lon: float, site: Mapping[str, Any]) -> float:
    """Great-circle distance (km) from a point to a site's footprint."""
    return min(haversine_km(lat, lon, a, b) for a, b in _footprint(site))


def distance_grid(catalog: Mapping[str, Any], shape: tuple[int, int] = (360, 720)) -> np.ndarray:
    """Distance (km) from every cell centre to the nearest catalogued site footprint.

    Vectorised haversine; used to filter "new sites only" without a per-cell loop.
    """
    nrow, ncol = shape
    lat = np.radians(90.0 - (np.arange(nrow) + 0.5) * (180.0 / nrow))[:, None]
    lon = np.radians(-180.0 + (np.arange(ncol) + 0.5) * (360.0 / ncol))[None, :]
    best = np.full(shape, np.inf)
    for site in catalog["sites"]:
        for plat, plon in _footprint(site):
            p1, l1 = math.radians(plat), math.radians(plon)
            a = (
                np.sin((lat - p1) / 2) ** 2
                + np.cos(lat) * math.cos(p1) * np.sin((lon - l1) / 2) ** 2
            )
            d = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))
            np.minimum(best, d, out=best)
    return best


def novelty(lat: float, lon: float, catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Whether a location is already a known analog, near one, or new."""
    catalog = catalog or load_catalog()
    nearest = min(catalog["sites"], key=lambda site: distance_to_site(lat, lon, site))
    distance = distance_to_site(lat, lon, nearest)
    if distance <= KNOWN_KM:
        status = "known"
    elif distance <= NEAR_KM:
        status = "near_known"
    else:
        status = "new"
    return {
        "status": status,
        "nearest_known": nearest["name"],
        "distance_km": round(distance, 1),
        "source_url": nearest["source_url"],
    }
