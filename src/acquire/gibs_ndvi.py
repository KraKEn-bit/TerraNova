"""MODIS Terra monthly NDVI from NASA GIBS, decoded back to NDVI values.

GIBS serves ``MODIS_Terra_L3_NDVI_Monthly`` (MOD13C2) as a colour-mapped image
with no authentication. Its published colour map
(``colormaps/v1.3/MODIS_L3_NDVI.xml``) gives every 0.005-wide NDVI bin a unique
RGB triple, so each opaque pixel inverts exactly to the lower edge of its bin.
Transparent pixels are NDVI <= 0 (water, snow, ice) or no data; both mean "no
vegetation detected" and are recorded as 0.

The predictor is the annual maximum of the twelve 2023 monthly composites
(a place is vegetated if it greens up in any month), block-averaged to the
0.5 degree grid and clipped at 0.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from src.acquire.download import DEFAULT_RAW_DIR, fetch

LAYER = "MODIS_Terra_L3_NDVI_Monthly"
YEAR = 2023
WIDTH, HEIGHT = 4320, 2160  # 1/12 degree, same as the terrain analysis grid
WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
COLORMAP_URL = "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_L3_NDVI.xml"
RAW_DIR = DEFAULT_RAW_DIR / "gibs"

DATASET_ID = "gibs_modis_terra_l3_ndvi_monthly_2023"
SOURCE_URL = "https://nasa-gibs.github.io/gibs-api-docs/available-visualizations/"

_ENTRY = re.compile(r'<ColorMapEntry rgb="(\d+),(\d+),(\d+)"[^>]*value="\[([-\d.]+),')


def _month_url(month: int) -> str:
    return (
        f"{WMS}?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={LAYER}&STYLES="
        f"&CRS=EPSG:4326&BBOX=-90,-180,90,180&WIDTH={WIDTH}&HEIGHT={HEIGHT}"
        f"&FORMAT=image/png&TIME={YEAR}-{month:02d}-01"
    )


def colour_lut(offline: bool | None = None) -> np.ndarray:
    """24-bit RGB -> NDVI lookup table; NaN for colours not in the map."""
    item = fetch(COLORMAP_URL, filename="MODIS_L3_NDVI.xml", raw_dir=RAW_DIR, offline=offline)
    lut = np.full(1 << 24, np.nan, dtype=np.float32)
    for r, g, b, low in _ENTRY.findall(item.path.read_text(encoding="utf-8")):
        lut[(int(r) << 16) | (int(g) << 8) | int(b)] = float(low)
    return lut


def decode(path, lut: np.ndarray) -> np.ndarray:
    """NDVI for one GIBS PNG; transparent or unknown colours become 0."""
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    key = (
        (rgba[..., 0].astype(np.uint32) << 16)
        | (rgba[..., 1].astype(np.uint32) << 8)
        | rgba[..., 2].astype(np.uint32)
    )
    ndvi = lut[key]
    ndvi[(rgba[..., 3] == 0) | ~np.isfinite(ndvi)] = 0.0
    return np.clip(ndvi, 0.0, None)


def annual_max(offline: bool | None = None) -> np.ndarray:
    """Maximum monthly NDVI over the year on the 1/12 degree grid (2160 x 4320)."""
    lut = colour_lut(offline)
    best = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for month in range(1, 13):
        item = fetch(
            _month_url(month),
            filename=f"{LAYER}_{YEAR}-{month:02d}.png",
            raw_dir=RAW_DIR,
            offline=offline,
            timeout=300,
        )
        np.maximum(best, decode(item.path, lut), out=best)
        print(f"[ndvi] {YEAR}-{month:02d} decoded", flush=True)
    return best
