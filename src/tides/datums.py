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

# 19-year tidal epoch for datum computation (full nodal cycle).
# End is exclusive: 2003-01-01 .. 2021-12-31 inclusive = 6,940 days.
_DATUM_EPOCH_START = datetime.date(2003, 1, 1)
_DATUM_EPOCH_END = datetime.date(2022, 1, 1)

# On-disk cache version. Bump when computed values change so stale entries are
# never reused (old files are left in place, unused).
DATUM_CACHE_VERSION = 2


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


# Datum computation samples the 19-year epoch every 6 minutes (NOAA's
# prediction interval), predicting one chunk per year to bound memory.
DATUM_INTERVAL_MINUTES = 6.0


def compute_datums_from_model(
    lat: float,
    lon: float,
    model_name: str = "GOT5.6",
) -> dict[str, float]:
    """Compute tidal datums from a 19-year model prediction.

    Predicts at 6-minute intervals over 2003-01-01 to 2022-01-01 (end
    exclusive, 6,940 days) and extracts statistical datums.
    All values are in meters relative to MSL.
    """
    from tides.ocean_model import load_local_constituents, predict_elevations

    local, m = load_local_constituents(lat, lon, model_name)

    epoch = datetime.date(1992, 1, 1)
    start_days = (_DATUM_EPOCH_START - epoch).days
    n_days = (_DATUM_EPOCH_END - _DATUM_EPOCH_START).days
    t = start_days + np.arange(0, n_days * 1440, DATUM_INTERVAL_MINUTES) / 1440.0

    chunks = np.array_split(t, max(1, n_days // 365))
    elevations = np.concatenate([predict_elevations(c, local, m) for c in chunks])

    return _extract_datums(elevations, DATUM_INTERVAL_MINUTES)


# Mean lunar (tidal) day in hours. MHHW/MLLW are the mean of the higher high
# and lower low per tidal day, not per 24 h calendar day.
TIDAL_DAY_HOURS = 24.8412

# Minimum separation between successive highs (or lows): semidiurnal extrema
# are ~12.4 h apart, so 2 h rejects noise without merging real extrema.
_MIN_EXTREMA_SEPARATION_MINUTES = 120


def _per_bin_extreme(values: np.ndarray, bins: np.ndarray, reducer: np.ufunc) -> np.ndarray:
    """Reduce `values` within runs of equal (non-decreasing) `bins`."""
    if len(values) == 0:
        return values
    _, starts = np.unique(bins, return_index=True)
    return reducer.reduceat(values, starts)


def _extract_datums(elevations: np.ndarray, interval_minutes: float = 6.0) -> dict[str, float]:
    """Extract tidal datums from an evenly sampled elevation time series.

    Expects elevations relative to MSL (mean ~0) sampled every
    `interval_minutes`. Higher-high / lower-low are taken per tidal day
    (24.8412 h). Fully vectorized: no per-day Python loop.
    """
    from scipy.signal import find_peaks

    elevations = np.asarray(elevations, dtype=float)
    distance = max(1, int(round(_MIN_EXTREMA_SEPARATION_MINUTES / interval_minutes)))
    highs_idx, _ = find_peaks(elevations, distance=distance)
    lows_idx, _ = find_peaks(-elevations, distance=distance)

    highs = elevations[highs_idx]
    lows = elevations[lows_idx]

    # Tidal-day bins; drop the trailing partial tidal day so every bin
    # contributes a full day's higher-high / lower-low.
    samples_per_tidal_day = TIDAL_DAY_HOURS * 60.0 / interval_minutes
    full_bins = int(len(elevations) // samples_per_tidal_day)
    high_bins = np.floor(highs_idx / samples_per_tidal_day).astype(int)
    low_bins = np.floor(lows_idx / samples_per_tidal_day).astype(int)
    keep_h = high_bins < full_bins
    keep_l = low_bins < full_bins
    higher_highs = _per_bin_extreme(highs[keep_h], high_bins[keep_h], np.maximum)
    lower_lows = _per_bin_extreme(lows[keep_l], low_bins[keep_l], np.minimum)

    mhw = float(np.mean(highs)) if len(highs) > 0 else 0.0
    mlw = float(np.mean(lows)) if len(lows) > 0 else 0.0
    mhhw = float(np.mean(higher_highs)) if len(higher_highs) > 0 else mhw
    mllw = float(np.mean(lower_lows)) if len(lower_lows) > 0 else mlw

    return {
        "lat": float(np.min(elevations)),
        "mllw": mllw,
        "mlw": mlw,
        # The prediction has no Z0 term, so its long-run mean is ~0 and MSL is
        # the zero reference by definition (model event heights share it).
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
    # FES2022: 1/16 deg (~0.0625 deg), GOT: 0.5 deg, EOT20: 0.125 deg
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
    cache_file = cache_dir / f"{model_name.lower()}.v{DATUM_CACHE_VERSION}.json"

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
