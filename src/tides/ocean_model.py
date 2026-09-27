import datetime
import functools

import numpy as np
from scipy.signal import find_peaks

from tides.models import Coordinate, TideEvent

DEFAULT_MODEL = "GOT5.6"
SUPPORTED_MODELS = {"GOT5.6", "GOT5.5", "EOT20", "FES2022"}
ELEVATION_INTERVAL_MINUTES = 1

# Minimum separation between peaks in minutes. Tidal extrema are typically
# ~6 hours apart; 2 hours is a conservative minimum to filter noise.
_MIN_PEAK_SEPARATION_MINUTES = 120
_MIN_PEAK_DISTANCE = _MIN_PEAK_SEPARATION_MINUTES // ELEVATION_INTERVAL_MINUTES

# Reference epoch for pyTMD predict.time_series: 1992-01-01T00:00:00 UTC
_PYTMD_PREDICT_EPOCH = datetime.datetime(1992, 1, 1, tzinfo=datetime.timezone.utc)


def find_extrema(
    times: list[datetime.datetime],
    elevations: np.ndarray,
) -> list[TideEvent]:
    if np.all(np.isnan(elevations)):
        return []

    # Find highs (peaks) and lows (troughs)
    highs, _ = find_peaks(elevations, distance=_MIN_PEAK_DISTANCE)
    lows, _ = find_peaks(-elevations, distance=_MIN_PEAK_DISTANCE)

    events = []
    for i in highs:
        events.append(TideEvent(time=times[i], height=float(elevations[i])))
    for i in lows:
        events.append(TideEvent(time=times[i], height=float(elevations[i])))

    events.sort(key=lambda e: e.time)
    return events


# Degrees of padding around the target when cropping model grids for
# interpolation.
MODEL_CROP_PAD_DEG = 2.0

_FES_FORMATS = ("FES-ascii", "FES-netcdf", "FES-native")


def crosses_seam(lon360: float, pad: float = MODEL_CROP_PAD_DEG) -> bool:
    """True when a +/-pad window around lon360 crosses the grid's 0/360 seam.

    All supported models (GOT5.5, GOT5.6, EOT20, FES2022) use 0-360 longitude
    grids, so the seam is at 0 deg (Greenwich), not at the antimeridian.
    """
    return lon360 - pad < 0 or lon360 + pad > 360


def utc_minutes(start: datetime.datetime, end: datetime.datetime, step_minutes: float) -> tuple:
    """Evenly spaced UTC sample times in [start, end).

    Returns (t, times): `t` is days since the pyTMD epoch (1992-01-01) as a
    numpy array, and `times` the matching timezone-aware UTC datetimes.
    """
    n = int((end - start).total_seconds() // (step_minutes * 60))
    offsets_min = np.arange(n) * step_minutes
    start_days = (start - _PYTMD_PREDICT_EPOCH).total_seconds() / 86400.0
    t = start_days + offsets_min / 1440.0
    times = [start + datetime.timedelta(minutes=float(m)) for m in offsets_min]
    return t, times


@functools.lru_cache(maxsize=8)
def load_local_constituents(lat: float, lon: float, model_name: str = DEFAULT_MODEL):
    """Load a model and interpolate its constituents at (lat, lon).

    Cached per process: a single `tides get` that also computes model datums
    reuses the interpolated constituents instead of reloading the model.
    Callers must treat the returned objects as read-only.

    Crops the grid to a +/-MODEL_CROP_PAD_DEG window, including windows that
    cross the 0/360 seam, so the full global grid is never loaded.

    Returns (local_dataset, model) where `model` carries `corrections` and
    `minor` for pyTMD.predict.
    """
    import pyTMD.io
    import xarray as xr

    from tides.cache import ensure_model_data

    ensure_model_data(model_name)

    m = pyTMD.io.model()
    m.from_database(model_name)

    pad = MODEL_CROP_PAD_DEG
    lat_min = max(lat - pad, -90.0)
    lat_max = min(lat + pad, 90.0)
    lon360 = lon % 360
    seam = crosses_seam(lon360, pad)
    # Across the seam, work in a signed longitude so the window is contiguous.
    x = lon360 - 360 if seam and lon360 > 360 - pad else lon360

    if m.format in _FES_FORMATS:
        # FES-format grids (FES2022, EOT20): dask lazy loading + manual crop to
        # avoid loading every constituent grid (~5 GB for FES2022) into memory.
        ds = m.open_dataset(chunks={})
        if seam:
            # These grids store x from 0 to 360 inclusive; exclude x == 360 from
            # the shifted slice or x = 0 would appear twice after the shift.
            west_x = ds.x[(ds.x >= 360 - pad) & (ds.x < 360)].values
            west = ds.sel(x=west_x, y=slice(lat_min, lat_max))
            west = west.assign_coords(x=west.x - 360)
            east = ds.sel(x=slice(None, pad), y=slice(lat_min, lat_max))
            ds = xr.concat([west, east], dim="x", combine_attrs="override")
        else:
            ds = ds.sel(x=slice(x - pad, x + pad), y=slice(lat_min, lat_max))
        ds = ds.compute()
    else:
        # GOT-format grids: pyTMD's crop wraps the seam when the west bound < 0.
        ds = m.open_dataset(crop=True, bounds=[x - pad, x + pad, lat_min, lat_max])

    local = ds.tmd.interp(x=x, y=lat, extrapolate=True, cutoff=10)
    return local, m


def predict_elevations(t: np.ndarray, local, model) -> np.ndarray:
    """Predict elevations (major + inferred minor constituents) at times `t`."""
    import pyTMD.predict

    tide = pyTMD.predict.time_series(t, local, corrections=model.corrections)
    minor = pyTMD.predict.infer_minor(t, local, corrections=model.corrections, minor=model.minor)
    # pyTMD returns xarray DataArrays; take the underlying arrays.
    tide_arr = np.asarray(getattr(tide, "values", tide), dtype=float)
    minor_arr = np.asarray(getattr(minor, "values", minor), dtype=float)
    return (tide_arr + minor_arr).ravel()


def compute_tides(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    model_name: str = DEFAULT_MODEL,
) -> list[TideEvent]:
    start = datetime.datetime(
        begin_date.year, begin_date.month, begin_date.day, tzinfo=datetime.timezone.utc
    )
    end = datetime.datetime(
        end_date.year, end_date.month, end_date.day, tzinfo=datetime.timezone.utc
    ) + datetime.timedelta(days=1)

    t, times = utc_minutes(start, end, ELEVATION_INTERVAL_MINUTES)
    local, m = load_local_constituents(coord.lat, coord.lon, model_name)
    elevations = predict_elevations(t, local, m)

    return find_extrema(times, elevations)
