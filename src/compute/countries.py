"""Which country each 0.5 degree cell belongs to (Natural Earth admin-0).

Used for the "max per country" diversity cap and for place labels. The label's
nearest town can sit across a border from the cell itself; the polygon test
does not. A cell takes the country whose polygon contains its centre; coastal
cells whose centre falls in the sea get the country of the nearest land cell
within two cells.

Natural Earth 1:50m Admin 0 - Countries (public domain), cached in
``cache/raw/ne_50m_admin_0_countries.geojson`` (committed, so this works offline).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

from src.acquire.download import fetch

URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)
FILENAME = "ne_50m_admin_0_countries.geojson"
NROW, NCOL = 360, 720
REACH = 2  # cells: how far a sea-centred coastal cell looks for its country


@lru_cache(maxsize=1)
def grid(offline: bool | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """(ids, names): ids[row, col] indexes names; 0 means no country."""
    item = fetch(URL, filename=FILENAME, offline=offline, timeout=120)
    features = json.loads(item.path.read_text(encoding="utf-8"))["features"]
    names = [
        "",
        *(str(f["properties"].get("NAME") or f["properties"].get("ADMIN")) for f in features),
    ]
    shapes = ((f["geometry"], i + 1) for i, f in enumerate(features) if f.get("geometry"))
    ids = rasterize(
        shapes,
        out_shape=(NROW, NCOL),
        transform=from_origin(-180.0, 90.0, 0.5, 0.5),
        fill=0,
        dtype="int32",
    )
    # Coastal cells: borrow the nearest country within REACH cells.
    distance, (ri, ci) = distance_transform_edt(ids == 0, return_indices=True)
    filled = np.where((ids == 0) & (distance <= REACH), ids[ri, ci], ids)
    return filled.astype(np.int32), tuple(names)


def country_at(lat: float, lon: float, offline: bool | None = None) -> str | None:
    ids, names = grid(offline)
    row = min(max(int((90.0 - lat) // 0.5), 0), NROW - 1)
    col = min(max(int((lon + 180.0) // 0.5), 0), NCOL - 1)
    index = int(ids[row, col])
    return names[index] or None
