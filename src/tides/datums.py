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
DATUM_CACHE_VERSION = 3

# Version of the per-station computed-datum cache (datums/stations.vN.json).
# v2: station predictions switched to GOT corrections without infer_minor (#15).
STATION_DATUM_CACHE_VERSION = 2

# Datum cache keys are the query point rounded to this many degrees (~1 km),
# independent of model grid resolution.
_CACHE_KEY_DECIMALS = 2


class DatumUnavailableError(Exception):
    """Tidal datums cannot be determined for this location or station."""


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

    # Offsets must be relative to MSL (or MTL as a close proxy). Without either,
    # the published values cannot be referenced to MSL at all.
    if "MSL" in datums:
        msl = datums["MSL"]
    elif "MTL" in datums:
        msl = datums["MTL"]
    else:
        return None

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
    if not np.all(np.isfinite(elevations)):
        raise DatumUnavailableError(
            f"no {model_name} tidal data within 10 km of {lat:.4f},{lon:.4f} (inland?); "
            "cannot compute tidal datums here"
        )

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

    if len(highs) == 0 or len(lows) == 0 or not np.all(np.isfinite(elevations)):
        raise DatumUnavailableError("no tidal highs/lows in the series; cannot compute datums")

    mhw = float(np.mean(highs))
    mlw = float(np.mean(lows))
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


def _cache_key(lat: float, lon: float) -> str:
    """Datum cache key: the query point rounded to 0.01 deg (~1 km).

    Datums are computed at the exact query point, so the key must not merge
    distinct points (the old grid-resolution key let the first query in a cell,
    possibly on land, answer for every point in it).
    """
    return f"{round(lat, _CACHE_KEY_DECIMALS):.2f},{round(lon, _CACHE_KEY_DECIMALS):.2f}"


def _all_finite(datums: object) -> bool:
    return (
        isinstance(datums, dict)
        and bool(datums)
        and all(isinstance(v, (int, float)) and np.isfinite(v) for v in datums.values())
    )


def _read_cache(cache_file: Path) -> dict:
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache_entry(cache_file: Path, key: str, datums: dict[str, float]) -> None:
    """Merge one entry into a cache file. Non-finite values are never cached."""
    if not _all_finite(datums):
        return
    cache = _read_cache(cache_file)
    cache[key] = datums
    cache_file.write_text(json.dumps(cache, indent=2))


def get_model_datums(
    lat: float,
    lon: float,
    model_name: str = "GOT5.6",
) -> dict[str, float]:
    """Get tidal datums for a coordinate, using cache when available.

    Raises DatumUnavailableError when the model has no data at the point.
    """
    cache_file = _get_datum_cache_dir() / f"{model_name.lower()}.v{DATUM_CACHE_VERSION}.json"
    key = _cache_key(lat, lon)

    cached = _read_cache(cache_file).get(key)
    if _all_finite(cached):
        return cached

    datums = compute_datums_from_model(lat, lon, model_name)
    _write_cache_entry(cache_file, key, datums)
    return datums


def compute_station_datums(station: dict) -> dict[str, float]:
    """Compute a station's full datum set from its own harmonic constituents.

    Runs the same 19-year, 6-minute prediction used for model datums. Values
    are MSL-relative (the harmonic series has no Z0 term). Raises
    DatumUnavailableError when the station has no usable harmonics.
    """
    from tides.harmonics import predict_elevations

    constituents = station.get("harmonic_constituents") or []
    if not constituents:
        raise DatumUnavailableError(
            f"station {station.get('name', '?')} publishes neither the requested datum "
            "nor harmonic constituents to compute it"
        )

    epoch = datetime.date(1992, 1, 1)
    start_days = (_DATUM_EPOCH_START - epoch).days
    n_days = (_DATUM_EPOCH_END - _DATUM_EPOCH_START).days
    t = start_days + np.arange(0, n_days * 1440, DATUM_INTERVAL_MINUTES) / 1440.0
    elevations = predict_elevations(constituents, t, chunks=max(1, n_days // 365))
    return _extract_datums(elevations, DATUM_INTERVAL_MINUTES)


def station_heights_datum(station: dict) -> str:
    """The datum (lowercase) that station predictions are expressed in.

    predict_station_tides shifts harmonic heights to the station's chart datum
    only when that datum is a supported one the station publishes relative to
    MSL; otherwise heights stay MSL-relative. Both sides must agree.
    """
    chart = str(station.get("chart_datum", "MSL")).lower()
    published = datums_from_station(station) or {}
    return chart if chart in published else "msl"


def get_station_datums(station: dict, station_id: str, needed: set[str]) -> dict[str, float]:
    """Datum offsets (MSL-relative) for a station, covering every datum in `needed`.

    Published datums always win; only datums the station does not publish are
    computed from its harmonics (cached per station id).
    """
    published = datums_from_station(station) or {}
    if needed <= published.keys():
        return published

    cache_file = _get_datum_cache_dir() / f"stations.v{STATION_DATUM_CACHE_VERSION}.json"
    computed = _read_cache(cache_file).get(station_id)
    if not _all_finite(computed):
        computed = compute_station_datums(station)
        _write_cache_entry(cache_file, station_id, computed)

    return {**computed, **published}
