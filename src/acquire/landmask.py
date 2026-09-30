"""Land fraction on the 0.5 degree grid from Natural Earth land minus lakes.

Replaces the Aridity-Index-derived mask, which had two defects: it had no data
south of ~60 S (so Antarctica, including the McMurdo Dry Valleys, was never
scored) and it counted the Caspian Sea as land.

Natural Earth 1:50m land and lakes are public domain. Polygons are rasterised on
the 1/12 degree analysis grid (a pixel is land when its centre falls inside a
land polygon and outside every lake) and block-averaged to a 0.5 degree land
fraction.
"""

from __future__ import annotations

import json

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin

from src.acquire.download import fetch

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
LAND_URL = f"{NE_BASE}/ne_50m_land.geojson"
LAKES_URL = f"{NE_BASE}/ne_50m_lakes.geojson"

DATASET_ID = "natural_earth_50m_land_lakes"
SOURCE_URL = "https://www.naturalearthdata.com/downloads/50m-physical-vectors/"


def _geometries(url: str, filename: str, offline: bool | None) -> list[dict]:
    item = fetch(url, filename=filename, offline=offline, timeout=300)
    features = json.loads(item.path.read_text(encoding="utf-8"))["features"]
    return [feature["geometry"] for feature in features if feature.get("geometry")]


def fine_land(nrow: int, ncol: int, offline: bool | None = None) -> np.ndarray:
    """Boolean land mask on a global ``nrow x ncol`` lat/lon grid (row 0 = north)."""
    transform = from_origin(-180.0, 90.0, 360.0 / ncol, 180.0 / nrow)
    land = rasterize(
        ((geom, 1) for geom in _geometries(LAND_URL, "ne_50m_land.geojson", offline)),
        out_shape=(nrow, ncol),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    lakes = rasterize(
        ((geom, 1) for geom in _geometries(LAKES_URL, "ne_50m_lakes.geojson", offline)),
        out_shape=(nrow, ncol),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    return land & ~lakes


def land_fraction(fine: np.ndarray, block: int) -> np.ndarray:
    """Share of land pixels in every ``block x block`` block."""
    nrow, ncol = fine.shape
    shape = (nrow // block, block, ncol // block, block)
    return fine.reshape(shape).mean(axis=(1, 3)).astype(np.float32)
