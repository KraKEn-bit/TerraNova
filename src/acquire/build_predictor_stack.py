"""Build ``cache/predictors.zarr`` from the raw rasters cached under ``cache/raw``.

    python -m src.acquire.build_predictor_stack

Grid
----
Every predictor is a ``(360, 720)`` float32 array of 0.5 degree cells in
EPSG:4326. Row 0 is centred on 89.75 N, column 0 on 179.75 W. Cells that are
less than 50 % land are written as NaN, so ``similarity.valid_mask`` is exactly
the candidate-land mask and no post-hoc filtering is needed.

Stages (each caches its 0.5 degree arrays under ``cache/derived``):

``dem``
    256 AWS Terrarium tiles at zoom 4 are decoded, warped to a 1/12 degree
    lat/lon grid (2160 x 4320, ~9.3 km at the equator) and masked to land.
    Elevation is the 6 x 6 block mean onto the 0.5 degree grid; slope is the
    horizontal gradient of that 0.5 degree field; roughness is the RMS height
    residual about a least-squares plane fitted over a 3 x 3 neighbourhood of
    the 1/12 degree grid, block-averaged onto the 0.5 degree grid.
``aridity``
    Global Aridity Index v3.1 (30 arc-sec, stored as integer x 10000).
``climate``
    WorldClim 2.1 monthly mean temperature, 10 arc-min: warmest month minus
    coldest month.
``lst``
    Zenodo 1 km MODIS Terra LST, long-term daytime minus nighttime composite.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
import zarr
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

from src.acquire.download import fetch, utc_now_iso
from src.compute.similarity import CRITERIA

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "cache" / "raw"
DERIVED_DIR = REPO_ROOT / "cache" / "derived"
STACK_PATH = REPO_ROOT / "cache" / "predictors.zarr"
NORM_PATH = REPO_ROOT / "data" / "normalization.json"

CRS = "EPSG:4326"
NROW = 360
NCOL = 720
CELL = 0.5
BLOCK = 6
SUB_CELL = CELL / BLOCK
SROW = NROW * BLOCK
SCOL = NCOL * BLOCK

GRID_TRANSFORM = from_origin(-180.0, 90.0, CELL, CELL)
SUB_TRANSFORM = from_origin(-180.0, 90.0, SUB_CELL, SUB_CELL)

LAT_CELL = 90.0 - (np.arange(NROW) + 0.5) * CELL
LON_CELL = -180.0 + (np.arange(NCOL) + 0.5) * CELL
LAT_SUB = 90.0 - (np.arange(SROW) + 0.5) * SUB_CELL

M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111320.0

TERRARIUM_ZOOM = 4
TERRARIUM_TILES = 2**TERRARIUM_ZOOM
TERRARIUM_PX = 256
MERC_PX = TERRARIUM_TILES * TERRARIUM_PX
MERC_RES = 156543.03392804097 / (2**TERRARIUM_ZOOM)
MERC_HALF = MERC_RES * MERC_PX / 2.0
MERC_TRANSFORM = from_origin(-MERC_HALF, MERC_HALF, MERC_RES, MERC_RES)

# Remote inputs. Each is fetched once through src.acquire.download.fetch and
# then reused from cache/raw, so a second run - or OFFLINE=1 - never touches
# the network.
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
AI_URL = "https://ndownloader.figshare.com/files/56300327"
AI_ZIP = RAW_DIR / "Global-AI_ET0__annual_v3_1.zip"
AI_INNER = "Global-AI_ET0__annual_v3_1/ai_v31_yr.tif"
AI_SCALE = 1e-4  # the source stores AI as integer x 10000
AI_OCEAN = 0
AI_FILL = 65535

WC_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_tavg.zip"
WC_ZIP = RAW_DIR / "wc2.1_10m_tavg.zip"

LST_RECORD = "https://zenodo.org/api/records/6458406/files"
LST_DAY = RAW_DIR / "zenodo_lst_day.tif"
LST_NIGHT = RAW_DIR / "zenodo_lst_night.tif"
LST_SCALE = 0.02

LAND_FRACTION_MIN = 0.5


class BuildError(RuntimeError):
    """Raised when a raw input does not match what this build expects."""


def _log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def _cache(name: str) -> Path:
    return DERIVED_DIR / f"{name}.npy"


def _load(name: str, force: bool) -> np.ndarray | None:
    path = _cache(name)
    if path.exists() and not force:
        _log(f"{name}: using cached {_cache(name).name}")
        return np.load(path)
    return None


def _store(name: str, array: np.ndarray) -> np.ndarray:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_cache(name), array)
    return array


def _vsizip(archive: Path, inner: str) -> str:
    base = "/vsizip/" + str(archive.resolve()).replace("\\", "/")
    return f"{base}/{inner}"


def _raw(name: str, url: str, offline: bool | None = None) -> Path:
    """Local path of a remote input, downloading it on first use."""
    return fetch(url, filename=name, offline=offline, timeout=900).path


def _shift(a: np.ndarray, di: int, dj: int) -> np.ndarray:
    """``out[i, j] = a[i + di, j + dj]`` with edge replication."""
    ii = np.clip(np.arange(a.shape[0]) + di, 0, a.shape[0] - 1)
    jj = np.clip(np.arange(a.shape[1]) + dj, 0, a.shape[1] - 1)
    return a[np.ix_(ii, jj)]


def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Nan-aware mean of non-overlapping ``k x k`` blocks."""
    nrow, ncol = a.shape
    if nrow % k or ncol % k:
        raise BuildError(f"shape {a.shape} is not divisible by block {k}")
    finite = np.isfinite(a)
    total = np.where(finite, a, 0.0).reshape(nrow // k, k, ncol // k, k).sum(axis=(1, 3))
    count = finite.reshape(nrow // k, k, ncol // k, k).sum(axis=(1, 3))
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out.astype(np.float32)


def _nearest_fill(a: np.ndarray) -> np.ndarray:
    finite = np.isfinite(a)
    if finite.all() or not finite.any():
        return a
    _, index = distance_transform_edt(~finite, return_indices=True)
    return a[tuple(index)]


# --------------------------------------------------------------------------
# stage: terrain
# --------------------------------------------------------------------------


def _decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """Terrarium PNG encoding: ``elevation = R * 256 + G + B / 256 - 32768``."""
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    return (r * 256.0 + g + b / 256.0) - 32768.0


def _load_mercator_dem(offline: bool | None = None) -> np.ndarray:
    grid = np.empty((MERC_PX, MERC_PX), dtype=np.float32)
    for ty in range(TERRARIUM_TILES):
        for tx in range(TERRARIUM_TILES):
            url = f"{TERRARIUM_URL}/{TERRARIUM_ZOOM}/{tx}/{ty}.png"
            name = f"terrarium_{TERRARIUM_ZOOM}_{tx}_{ty}.png"
            last: Exception | None = None
            for attempt in range(3):
                try:
                    item = fetch(url, filename=name, offline=offline, timeout=120)
                    break
                except Exception as exc:  # any failure is retried below
                    last = exc
                    time.sleep(1.5 * (attempt + 1))
            else:
                raise BuildError(f"terrarium tile {tx}/{ty} unavailable: {last}")
            with Image.open(item.path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape[:2] != (TERRARIUM_PX, TERRARIUM_PX):
                raise BuildError(f"{name} decoded to {rgb.shape}, expected square tile")
            y0, x0 = ty * TERRARIUM_PX, tx * TERRARIUM_PX
            grid[y0 : y0 + TERRARIUM_PX, x0 : x0 + TERRARIUM_PX] = _decode_terrarium(rgb)
        _log(f"terrarium: row {ty + 1}/{TERRARIUM_TILES} decoded")
    return grid


def _land_fine_from_aridity() -> np.ndarray:
    """Nearest-neighbour land mask on the 1/12 degree grid, from the AI file.

    No ``dst_nodata`` is passed: 0 is a legitimate AI value as well as the
    source's open-ocean encoding, and handing it to GDAL as a destination
    nodata makes the warper emit 1 everywhere instead of 0. The fill value is
    filtered here instead, exactly as :func:`stage_aridity` does.
    """
    with rasterio.open(_vsizip(AI_ZIP, AI_INNER)) as src:
        values = np.zeros((SROW, SCOL), dtype=src.dtypes[0])
        reproject(
            rasterio.band(src, 1),
            values,
            dst_transform=SUB_TRANSFORM,
            dst_crs=CRS,
            resampling=Resampling.nearest,
        )
    return (values != AI_OCEAN) & (values != AI_FILL)


def stage_terrain(force: bool = False, offline: bool | None = None) -> dict[str, np.ndarray]:
    cached = {key: _load(key, force) for key in ("elevation", "slope", "roughness")}
    if all(value is not None for value in cached.values()):
        return cached  # type: ignore[return-value]

    started = time.time()
    mercator = _load_mercator_dem(offline=offline)
    _log(f"terrarium: {MERC_PX}x{MERC_PX} grid assembled in {time.time() - started:.1f}s")

    dem = np.full((SROW, SCOL), np.nan, dtype=np.float32)
    reproject(
        mercator,
        dem,
        src_transform=MERC_TRANSFORM,
        src_crs="EPSG:3857",
        dst_transform=SUB_TRANSFORM,
        dst_crs=CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    del mercator

    land = _land_fine_from_aridity()
    dem[~land] = np.nan
    del land
    _log(f"terrain: fine DEM valid over {np.isfinite(dem).mean() * 100:.1f}% of the 1/12 deg grid")

    elevation = _block_mean(dem, BLOCK)
    _log("terrain: elevation block-mean done")

    slope = _slope_from_cell_field(elevation)
    _log("terrain: slope done")

    rough_fine = _fine_roughness(dem)
    del dem
    roughness = _block_mean(rough_fine, BLOCK)
    del rough_fine
    _log("terrain: roughness done")

    return {
        "elevation": _store("elevation", elevation),
        "slope": _store("slope", slope),
        "roughness": _store("roughness", roughness),
    }


def _slope_from_cell_field(elev: np.ndarray) -> np.ndarray:
    """Magnitude of the horizontal gradient of the 0.5 degree elevation field."""
    finite = np.isfinite(elev)
    filled = _nearest_fill(elev.astype(np.float64))
    dz_dlat = np.gradient(filled, CELL, axis=0) / M_PER_DEG_LAT
    dx_m = CELL * M_PER_DEG_LON * np.cos(np.deg2rad(LAT_CELL))
    dz_dlon = np.gradient(filled, CELL, axis=1) / np.maximum(dx_m, 1.0)[:, None]
    slope = np.degrees(np.arctan(np.hypot(dz_dlon, dz_dlat)))
    slope[~finite] = np.nan
    return slope.astype(np.float32)


def _fine_roughness(dem: np.ndarray, band: int = 360) -> np.ndarray:
    """RMS residual of the DEM about a least-squares plane over a 3x3 window."""
    out = np.full(dem.shape, np.nan, dtype=np.float32)
    nrow = dem.shape[0]
    gx_all = SUB_CELL * M_PER_DEG_LON * np.cos(np.deg2rad(LAT_SUB))
    gy = SUB_CELL * M_PER_DEG_LAT

    for start in range(0, nrow, band):
        stop = min(start + band, nrow)
        lo = max(0, start - 1)
        hi = min(nrow, stop + 1)

        valid = np.isfinite(dem[lo:hi])
        z = np.where(valid, dem[lo:hi], 0.0).astype(np.float64)
        z2 = z * z
        full = valid.copy()
        total = np.zeros_like(z)
        total2 = np.zeros_like(z)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                full &= _shift(valid, di, dj)
                total += _shift(z, di, dj)
                total2 += _shift(z2, di, dj)
        del z2

        mean_z = total / 9.0
        variance = total2 / 9.0 - mean_z * mean_z
        del total, total2

        dx_pixel = (_shift(z, -1, 1) + _shift(z, 0, 1) + _shift(z, 1, 1)) - (
            _shift(z, -1, -1) + _shift(z, 0, -1) + _shift(z, 1, -1)
        )
        dy_pixel = (_shift(z, 1, -1) + _shift(z, 1, 0) + _shift(z, 1, 1)) - (
            _shift(z, -1, -1) + _shift(z, -1, 0) + _shift(z, -1, 1)
        )
        del z, valid

        gx = gx_all[lo:hi, None]
        b = dx_pixel / (6.0 * gx)
        c = dy_pixel / (6.0 * gy)
        explained = (2.0 / 3.0) * (b * b * gx * gx + c * c * gy * gy)
        del dx_pixel, dy_pixel, b, c

        rms = np.sqrt(np.clip(variance - explained, 0.0, None))
        del variance, explained
        rms[~full] = np.nan
        out[start:stop] = rms[start - lo : stop - lo].astype(np.float32)

    return out


# --------------------------------------------------------------------------
# stage: aridity
# --------------------------------------------------------------------------


def stage_aridity(force: bool = False, offline: bool | None = None) -> dict[str, np.ndarray]:
    cached = {"aridity": _load("aridity", force), "land_fraction": _load("land_fraction", force)}
    if cached["aridity"] is not None and cached["land_fraction"] is not None:
        return cached  # type: ignore[return-value]

    started = time.time()
    aridity = np.full((NROW, NCOL), np.nan, dtype=np.float32)
    land = np.zeros((NROW, NCOL), dtype=np.float32)

    archive = _raw(AI_ZIP.name, AI_URL, offline)
    with rasterio.open(_vsizip(archive, AI_INNER)) as src:
        if src.nodata not in (None, AI_FILL):
            raise BuildError(f"unexpected AI nodata {src.nodata}")
        height, width = src.height, src.width
        row_step = round(CELL / abs(src.res[1]))
        col_step = round(CELL / abs(src.res[0]))
        if abs(src.bounds.top - 90.0) > 1e-6:
            raise BuildError(f"AI grid top is {src.bounds.top}, expected 90")
        if width != col_step * NCOL:
            raise BuildError(f"AI grid width {width} is not {col_step} x {NCOL}")
        block_rows = row_step * 10
        for row0 in range(0, height, block_rows):
            rows = min(block_rows, height - row0)
            window = Window(0, row0, width, rows)
            values = src.read(1, window=window)
            valid = (values != AI_OCEAN) & (values != AI_FILL)
            shape = (rows // row_step, row_step, NCOL, col_step)
            count = valid.reshape(shape).sum(axis=(1, 3))
            total = np.where(valid, values, 0).astype(np.float64).reshape(shape).sum(axis=(1, 3))
            first = row0 // row_step
            last = first + count.shape[0]
            ok = count > 0
            aridity[first:last] = np.where(
                ok, total / np.maximum(count, 1) * AI_SCALE, np.nan
            ).astype(np.float32)
            land[first:last] = (count / float(row_step * col_step)).astype(np.float32)
        _log(f"aridity: {time.time() - started:.1f}s over {height} x {width} pixels")

    return {
        "aridity": _store("aridity", aridity),
        "land_fraction": _store("land_fraction", land),
    }


# --------------------------------------------------------------------------
# stage: climate
# --------------------------------------------------------------------------


def stage_climate(force: bool = False, offline: bool | None = None) -> np.ndarray:
    cached = _load("annual_temperature_range", force)
    if cached is not None:
        return cached

    started = time.time()
    archive = _raw(WC_ZIP.name, WC_URL, offline)
    warm: np.ndarray | None = None
    cold: np.ndarray | None = None
    for month in range(1, 13):
        inner = f"wc2.1_10m_tavg_{month:02d}.tif"
        with rasterio.open(_vsizip(archive, inner)) as src:
            values = src.read(1, masked=True).astype(np.float32)
            values = np.ma.filled(values.astype(np.float32), np.nan)
        warm = values.copy() if warm is None else np.fmax(warm, values)
        cold = values.copy() if cold is None else np.fmin(cold, values)
    assert warm is not None and cold is not None
    span = warm - cold
    span[~np.isfinite(warm) | ~np.isfinite(cold)] = np.nan

    annual_range = _block_mean(span, 3)
    _log(f"climate: {time.time() - started:.1f}s for 12 monthly means")
    return _store("annual_temperature_range", annual_range)


# --------------------------------------------------------------------------
# stage: land surface temperature
# --------------------------------------------------------------------------


def _warp_lst(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        nodata = src.nodata
        if nodata is None:
            raise BuildError(f"{path.name} declares no nodata")
        out = np.full((NROW, NCOL), np.nan, dtype=np.float32)
        reproject(
            rasterio.band(src, 1),
            out,
            dst_transform=GRID_TRANSFORM,
            dst_crs=CRS,
            src_nodata=nodata,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return out


def stage_lst(force: bool = False, offline: bool | None = None) -> np.ndarray:
    cached = _load("lst_diurnal_range", force)
    if cached is not None:
        return cached

    started = time.time()
    day = _warp_lst(_raw(LST_DAY.name, f"{LST_RECORD}/{LST_DAY.name}/content", offline))
    night = _warp_lst(_raw(LST_NIGHT.name, f"{LST_RECORD}/{LST_NIGHT.name}/content", offline))
    span = (day - night) * LST_SCALE
    span[~np.isfinite(day) | ~np.isfinite(night)] = np.nan
    span[span <= 0] = np.nan
    _log(f"lst: {time.time() - started:.1f}s for day and night warps")
    return _store("lst_diurnal_range", span.astype(np.float32))


# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------


def _report(stack: dict[str, np.ndarray], land: np.ndarray) -> None:
    mask = land >= LAND_FRACTION_MIN
    _log(f"candidate cells with land fraction >= {LAND_FRACTION_MIN}: {int(mask.sum())}")
    for key in CRITERIA:
        values = stack[key][mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            _log(f"  {key}: no finite values")
            continue
        q = np.percentile(values, [0.5, 1, 5, 50, 95, 99, 99.5])
        _log(
            f"  {key:26s} n={values.size:6d} min={values.min():9.3f} "
            f"p0.5={q[0]:9.3f} p1={q[1]:9.3f} p5={q[2]:9.3f} p50={q[3]:9.3f} "
            f"p95={q[4]:9.3f} p99={q[5]:9.3f} max={values.max():9.3f}"
        )


def assemble() -> None:
    norm = json.loads(NORM_PATH.read_text(encoding="utf-8"))
    land = _load("land_fraction", False)
    if land is None:
        raise BuildError("land_fraction must be built before assemble")
    keep = land >= LAND_FRACTION_MIN

    stack: dict[str, np.ndarray] = {}
    for key in CRITERIA:
        values = _load(key, False)
        if values is None:
            raise BuildError(f"{key} must be built before assemble")
        values = np.asarray(values, dtype=np.float32)
        values[~keep] = np.nan
        stack[key] = values

    _report(stack, land)

    if STACK_PATH.exists():
        shutil.rmtree(STACK_PATH)

    root = zarr.open_group(str(STACK_PATH), mode="w")
    for key in CRITERIA:
        root.create_array(
            key,
            shape=(NROW, NCOL),
            dtype="f4",
            chunks=(NROW, NCOL),
            fill_value=np.nan,
        )
        root[key][:] = stack[key]
    for name, values in (
        ("lat", LAT_CELL.astype(np.float32)),
        ("lon", LON_CELL.astype(np.float32)),
    ):
        root.create_array(name, shape=values.shape, dtype="f4", chunks=values.shape)
        root[name][:] = values
    root.create_array(
        "land_fraction", shape=(NROW, NCOL), dtype="f4", chunks=(NROW, NCOL), fill_value=0.0
    )
    root["land_fraction"][:] = land

    root.attrs["schema_version"] = "1.0"
    root.attrs["created_utc"] = utc_now_iso()
    root.attrs["normalization"] = norm["criteria"]
    root.attrs["weights"] = norm["weights"]
    root.attrs["grid"] = {
        "crs": CRS,
        "shape": [NROW, NCOL],
        "cell_degrees": CELL,
        "row0_centre_latitude": float(LAT_CELL[0]),
        "column0_centre_longitude": float(LON_CELL[0]),
        "ordering": "row-major, north to south then west to east",
        "sub_grid": {
            "shape": [SROW, SCOL],
            "cell_degrees": SUB_CELL,
            "purpose": "1/12 degree DEM analysis grid used for slope and roughness",
        },
    }
    root.attrs["land_rule"] = (
        f"A cell is a candidate when at least {LAND_FRACTION_MIN:.0%} of its area is land, "
        "where land is the fraction of the 30 arc-second Aridity Index pixels that carry a "
        "valid value (the source encodes open ocean as 0 and fill as 65535). Every predictor "
        "is set to NaN below that threshold, so similarity.valid_mask is the land mask."
    )
    root.attrs["sources"] = [
        {
            "dataset_id": "aws_terrain_tiles_terrarium",
            "title": "Mapzen/AWS Terrain Tiles, Terrarium RGB encoding, zoom 4",
            "url": "https://registry.opendata.aws/terrain-tiles/",
            "local_files": "cache/raw/terrarium_4_*.png (256 tiles)",
        },
        {
            "dataset_id": "global_ai_et0_v3_1",
            "title": "Global Aridity Index and Potential Evapotranspiration (ET0) Database v3.1",
            "url": "https://doi.org/10.6084/m9.figshare.7504448",
            "local_files": AI_ZIP.name,
            "note": "AI is stored as integer x 10000; the build multiplies by 1e-4.",
        },
        {
            "dataset_id": "worldclim_2_1_tavg",
            "title": "WorldClim 2.1 monthly average temperature, 10 arc-min, 1970-2000",
            "url": "https://doi.org/10.1038/s41597-018-0002-1",
            "local_files": WC_ZIP.name,
        },
        {
            "dataset_id": "zenodo_modis_lst_1km_2000_2020",
            "title": (
                "Long-term MODIS LST day-time and night-time temperatures at 1 km, 2000-2020"
            ),
            "url": "https://doi.org/10.5281/zenodo.6458406",
            "local_files": f"{LST_DAY.name}, {LST_NIGHT.name}",
            "note": "int16 x 0.02 K; the predictor is (day - night) x 0.02.",
        },
    ]
    _log(f"wrote {STACK_PATH.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "dem", "aridity", "climate", "lst", "assemble"),
        default="all",
    )
    parser.add_argument("--force", action="store_true", help="ignore cache/derived")
    parser.add_argument("--offline", action="store_true", help="never touch the network")
    args = parser.parse_args(argv)
    offline: bool | None = True if args.offline else None

    stages = (
        ("dem", "aridity", "climate", "lst", "assemble") if args.stage == "all" else (args.stage,)
    )
    for stage in stages:
        started = time.time()
        if stage == "dem":
            stage_terrain(force=args.force, offline=offline)
        elif stage == "aridity":
            stage_aridity(force=args.force, offline=offline)
        elif stage == "climate":
            stage_climate(force=args.force, offline=offline)
        elif stage == "lst":
            stage_lst(force=args.force, offline=offline)
        elif stage == "assemble":
            assemble()
        _log(f"stage {stage} finished in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
