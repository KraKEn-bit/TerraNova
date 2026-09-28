"""NASA POWER climatology (MERRA-2, 2001-2020) regridded to the 0.5 degree grid.

POWER's regional climatology endpoint accepts one parameter and at most a
10 x 10 degree box per request, and returns its native 0.5 x 0.625 degree
MERRA-2 grid. Only boxes that contain land are requested. Every response is
cached under ``cache/raw/power/`` through :func:`src.acquire.download.fetch`,
so a rebuild with ``OFFLINE=1`` never touches the network.

Parameters used
---------------
``PRECTOTCORR``  bias-corrected precipitation, mm/day (annual mean in ``ANN``)
``T2M``          2 m air temperature, deg C (12 monthly means + ``ANN``)
``TS_RANGE``     daily range of earth skin temperature, K (``ANN``); used only
                 to gap-fill the MODIS LST diurnal range where MODIS has no data
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.acquire.download import DEFAULT_RAW_DIR, FetchError, fetch

API = "https://power.larc.nasa.gov/api/temporal/climatology/regional"
RAW_DIR = DEFAULT_RAW_DIR / "power"
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
FILL = -999.0

DATASET_ID = "nasa_power_merra2_climatology_2001_2020"
SOURCE_URL = "https://power.larc.nasa.gov/docs/services/api/temporal/climatology/"

# Native MERRA-2 grid as served by POWER.
POWER_LAT = np.arange(-90.0, 90.0 + 1e-9, 0.5)
POWER_LON = np.arange(-180.0, 180.0 + 1e-9, 0.625)


def boxes_with_land(land: np.ndarray) -> list[tuple[int, int]]:
    """South-west corners of the 10 x 10 degree boxes that contain land.

    ``land`` is a boolean mask on the 0.5 degree grid (360 x 720, row 0 = north).
    Boxes are grown by one cell on every side so interpolation near a box edge
    always has neighbours.
    """
    corners: list[tuple[int, int]] = []
    for lat0 in range(-90, 90, 10):
        for lon0 in range(-180, 180, 10):
            r0 = max(0, int((90 - (lat0 + 10)) / 0.5) - 1)
            r1 = min(360, int((90 - lat0) / 0.5) + 1)
            c0 = max(0, int((lon0 + 180) / 0.5) - 1)
            c1 = min(720, int((lon0 + 10 + 180) / 0.5) + 1)
            if land[r0:r1, c0:c1].any():
                corners.append((lat0, lon0))
    return corners


def _box_url(parameter: str, lat0: int, lon0: int) -> str:
    return (
        f"{API}?parameters={parameter}&community=RE&format=JSON"
        f"&latitude-min={lat0}&latitude-max={lat0 + 10}"
        f"&longitude-min={lon0}&longitude-max={lon0 + 10}"
    )


def _fetch_box(parameter: str, lat0: int, lon0: int, offline: bool | None) -> dict:
    name = f"{parameter}_{lat0:+04d}_{lon0:+05d}.json"
    last: Exception | None = None
    for _ in range(3):
        try:
            item = fetch(
                _box_url(parameter, lat0, lon0),
                filename=name,
                raw_dir=RAW_DIR,
                offline=offline,
                timeout=240,
            )
            payload = json.loads(item.path.read_text(encoding="utf-8"))
            if "features" not in payload:
                item.path.unlink(missing_ok=True)  # an error body must not be cached
                raise FetchError(f"POWER error for {name}: {payload.get('messages')}")
            return payload
        except FetchError as exc:
            last = exc
            if offline:
                break
    raise FetchError(f"POWER box {name} unavailable: {last}")


def fetch_parameter(
    parameter: str,
    corners: list[tuple[int, int]],
    periods: tuple[str, ...],
    offline: bool | None = None,
    workers: int = 4,
) -> dict[str, np.ndarray]:
    """Global native-grid arrays (lat x lon) for each requested period key."""
    grids = {p: np.full((POWER_LAT.size, POWER_LON.size), np.nan) for p in periods}

    def one(corner: tuple[int, int]) -> dict:
        return _fetch_box(parameter, corner[0], corner[1], offline)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, payload in enumerate(pool.map(one, corners), start=1):
            for feature in payload["features"]:
                lon, lat = feature["geometry"]["coordinates"][:2]
                i = round((lat + 90.0) / 0.5)
                j = round((lon + 180.0) / 0.625)
                values = feature["properties"]["parameter"][parameter]
                for period in periods:
                    value = float(values.get(period, FILL))
                    if value != FILL:
                        grids[period][i, j] = value
            if done % 50 == 0:
                print(f"[power] {parameter}: {done}/{len(corners)} boxes", flush=True)
    return grids


def to_cell_grid(native: np.ndarray, lat_cell: np.ndarray, lon_cell: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a native POWER grid onto 0.5 degree cell centres.

    Longitude wraps: the -180 column is appended at +180 so the dateline is
    seamless. NaN in any of the four neighbours gives NaN.
    """
    lon = np.append(POWER_LON, 180.0 + 0.625)
    grid = np.concatenate([native, native[:, :1]], axis=1)
    interp = RegularGridInterpolator(
        (POWER_LAT, lon), grid, method="linear", bounds_error=False, fill_value=np.nan
    )
    lat2, lon2 = np.meshgrid(lat_cell, lon_cell, indexing="ij")
    return interp(np.stack([lat2.ravel(), lon2.ravel()], axis=-1)).reshape(lat2.shape)
