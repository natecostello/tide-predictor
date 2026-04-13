"""Tidal datum computation and lookup.

Computes tidal datums (LAT, MLLW, MLW, MSL, MHW, MHHW, HAT) from either
station-published values or a 19-year model prediction at a coordinate.
"""

import datetime
import json
from enum import Enum
from pathlib import Path

import numpy as np

SUPPORTED_DATUMS = ("lat", "mllw", "mlw", "msl", "mtl", "mhw", "mhhw", "hat")

# 19-year tidal epoch for datum computation (full nodal cycle)
_DATUM_EPOCH_START = datetime.date(2003, 1, 1)
_DATUM_EPOCH_END = datetime.date(2021, 12, 31)


class Datum(Enum):
    LAT = "lat"
    MLLW = "mllw"
    MLW = "mlw"
    MSL = "msl"
    MTL = "mtl"
    MHW = "mhw"
    MHHW = "mhhw"
    HAT = "hat"


def datums_from_station(station: dict) -> dict[str, float] | None:
    """Extract datum offsets relative to MSL from station data.

    Returns dict mapping datum name (lowercase) to elevation in meters
    relative to MSL, or None if station has no datum information.
    """
    datums = station.get("datums", {})
    if not datums:
        return None

    msl = datums.get("MSL", datums.get("MTL", 0.0))

    result = {}
    for key in ("LAT", "MLLW", "MLW", "MSL", "MTL", "MHW", "MHHW", "HAT"):
        if key in datums:
            # Station datums are relative to STND; convert to relative to MSL
            result[key.lower()] = datums[key] - msl

    # Ensure MSL is always 0
    result["msl"] = 0.0
    return result


def compute_datums_from_model(
    lat: float,
    lon: float,
    model_name: str = "GOT5.6",
) -> dict[str, float]:
    """Compute tidal datums from a 19-year model prediction.

    Runs hourly predictions for 2003-2021 and extracts statistical datums.
    All values are in meters relative to MSL.
    """
    import pyTMD.io
    import pyTMD.predict

    from tides.cache import ensure_model_data

    ensure_model_data(model_name)

    m = pyTMD.io.model()
    m.from_database(model_name)

    # Load model (same logic as ocean_model.py)
    pad = 2.0
    lat_min = max(lat - pad, -90.0)
    lat_max = min(lat + pad, 90.0)
    lon360 = lon % 360
    lon_min = lon360 - pad
    lon_max = lon360 + pad
    crosses_antimeridian = lon_min < 0 or lon_max > 360

    if m.format in ("FES-ascii", "FES-netcdf", "FES-native"):
        ds = m.open_dataset(chunks={})
        if not crosses_antimeridian:
            ds = ds.sel(
                x=slice(lon_min, lon_max),
                y=slice(lat_min, lat_max),
            )
        ds = ds.compute()
    elif crosses_antimeridian:
        ds = m.open_dataset(crop=False)
    else:
        bounds = [lon_min, lon_max, lat_min, lat_max]
        ds = m.open_dataset(crop=True, bounds=bounds)

    local = ds.tmd.interp(x=lon360, y=lat, extrapolate=True, cutoff=10)

    # Generate hourly timestamps for the 19-year epoch
    epoch = datetime.datetime(1992, 1, 1, tzinfo=datetime.timezone.utc)
    start_dt = datetime.datetime(
        _DATUM_EPOCH_START.year,
        _DATUM_EPOCH_START.month,
        _DATUM_EPOCH_START.day,
        tzinfo=datetime.timezone.utc,
    )
    total_hours = int(19 * 365.25 * 24)
    t = np.array(
        [(start_dt - epoch).total_seconds() / 86400.0 + i / 24.0 for i in range(total_hours)]
    )

    tide = pyTMD.predict.time_series(t, local, corrections=m.corrections)
    minor = pyTMD.predict.infer_minor(t, local, corrections=m.corrections, minor=m.minor)
    elevations = (
        np.atleast_1d(np.asarray(tide)).flatten() + np.atleast_1d(np.asarray(minor)).flatten()
    )

    return _extract_datums(elevations)


def _extract_datums(elevations: np.ndarray) -> dict[str, float]:
    """Extract tidal datums from an hourly elevation time series.

    Expects elevations relative to MSL (mean ~0).
    """
    from scipy.signal import find_peaks

    # Find all highs and lows
    highs_idx, _ = find_peaks(elevations, distance=4)  # min 4h apart
    lows_idx, _ = find_peaks(-elevations, distance=4)

    highs = elevations[highs_idx]
    lows = elevations[lows_idx]

    # Group by day to find higher-high and lower-low per day
    hours_per_day = 24
    num_days = len(elevations) // hours_per_day

    higher_highs = []
    lower_lows = []
    for day in range(num_days):
        start = day * hours_per_day
        end = start + hours_per_day
        day_highs = [elevations[i] for i in highs_idx if start <= i < end]
        day_lows = [elevations[i] for i in lows_idx if start <= i < end]
        if day_highs:
            higher_highs.append(max(day_highs))
        if day_lows:
            lower_lows.append(min(day_lows))

    mhw = float(np.mean(highs)) if len(highs) > 0 else 0.0
    mlw = float(np.mean(lows)) if len(lows) > 0 else 0.0
    mhhw = float(np.mean(higher_highs)) if higher_highs else mhw
    mllw = float(np.mean(lower_lows)) if lower_lows else mlw

    return {
        "lat": float(np.min(elevations)),
        "mllw": mllw,
        "mlw": mlw,
        "msl": 0.0,
        "mtl": (mhw + mlw) / 2,
        "mhw": mhw,
        "mhhw": mhhw,
        "hat": float(np.max(elevations)),
    }


def _get_datum_cache_dir() -> Path:
    from tides.cache import get_cache_dir

    d = get_cache_dir() / "datums"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _grid_key(lat: float, lon: float, model_name: str) -> str:
    """Round coordinate to model grid resolution for cache key."""
    # FES2022: 1/16° (~0.0625°), GOT: 0.5°, EOT20: 0.125°
    resolutions = {"FES2022": 0.0625, "GOT5.6": 0.5, "GOT5.5": 0.5, "EOT20": 0.125}
    res = resolutions.get(model_name, 0.125)
    rlat = round(round(lat / res) * res, 4)
    rlon = round(round(lon / res) * res, 4)
    return f"{rlat},{rlon}"


def get_model_datums(
    lat: float,
    lon: float,
    model_name: str = "GOT5.6",
) -> dict[str, float]:
    """Get tidal datums for a coordinate, using cache when available."""
    cache_dir = _get_datum_cache_dir()
    cache_file = cache_dir / f"{model_name.lower()}.json"

    key = _grid_key(lat, lon, model_name)

    # Check cache
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            if key in cache:
                return cache[key]
        except (json.JSONDecodeError, ValueError):
            pass

    # Compute
    datums = compute_datums_from_model(lat, lon, model_name)

    # Save to cache
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    cache[key] = datums
    cache_file.write_text(json.dumps(cache, indent=2))

    return datums
