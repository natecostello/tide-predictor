"""Tidal harmonic prediction from station constituent data.

Constructs an xarray Dataset from station harmonic constituents and
predicts water levels using pyTMD's predict.time_series() — the same
pipeline used by the gridded model path (ocean_model.py).
"""

import datetime
import functools

import numpy as np
import pyTMD.constituents
import xarray as xr

from tides.models import TideEvent
from tides.ocean_model import ELEVATION_INTERVAL_MINUTES, find_extrema

# Reference epoch for pyTMD: 1992-01-01T00:00:00 UTC
_PYTMD_EPOCH = datetime.datetime(1992, 1, 1, tzinfo=datetime.timezone.utc)

# Correction type for station harmonic constants.
# OTIS uses the standard Doodson/IHO astronomical argument conventions,
# which match how NOAA and IHO-sourced harmonic constants are analyzed.
STATION_CORRECTIONS = "OTIS"

# Map station constituent names to pyTMD's expected names.
# NOAA uses abbreviations (LAM2, RHO) that pyTMD doesn't recognize;
# pyTMD uses the full IHO names (lambda2, rho1).
_NAME_MAP: dict[str, str] = {
    "lam2": "lambda2",
    "rho": "rho1",
}


def _normalize_name(name: str) -> str:
    """Normalize a station constituent name to pyTMD's convention."""
    return _NAME_MAP.get(name, name)


@functools.cache
def _is_recognized(name: str) -> bool:
    """Check if pyTMD recognizes a constituent name."""
    try:
        pyTMD.constituents.coefficients_table([name])
        return True
    except (ValueError, KeyError):
        return False


def _build_dataset(constituents: list[dict]) -> xr.Dataset:
    """Build a pyTMD-compatible xarray Dataset from station harmonics.

    Each constituent becomes a scalar complex64 variable:
        z = amplitude * exp(-i * phase_radians)

    This matches the format produced by pyTMD's model.open_dataset().tmd.interp().
    """
    data_vars = {}
    for c in constituents:
        name = _normalize_name(c["name"].lower())
        amp = c["amplitude"]
        if amp <= 0:
            continue
        if not _is_recognized(name):
            continue
        phase_rad = np.radians(c["phase"])
        z = amp * np.exp(-1j * phase_rad)
        data_vars[name] = xr.Variable((), np.complex64(z))

    return xr.Dataset(data_vars)


def predict_tides_for_day(
    date: datetime.date,
    constituents: list[dict],
    datum_offset: float = 0.0,
) -> list[TideEvent]:
    """Predict high/low tides for a single day using pyTMD.

    Constructs an xarray Dataset from station harmonic constituents and
    feeds it through pyTMD's predict.time_series() + predict.infer_minor().
    """
    import pyTMD.predict

    if not constituents:
        return []

    ds = _build_dataset(constituents)
    if len(ds.data_vars) == 0:
        return []

    start = datetime.datetime(date.year, date.month, date.day, tzinfo=datetime.timezone.utc)
    end = start + datetime.timedelta(days=1)

    times = []
    current = start
    while current < end:
        times.append(current)
        current += datetime.timedelta(minutes=ELEVATION_INTERVAL_MINUTES)

    # Days since 1992-01-01 (pyTMD's epoch)
    t = np.array([(dt - _PYTMD_EPOCH).total_seconds() / 86400.0 for dt in times])

    tide = pyTMD.predict.time_series(t, ds, corrections=STATION_CORRECTIONS)
    minor = pyTMD.predict.infer_minor(t, ds, corrections=STATION_CORRECTIONS)

    tide_arr = np.atleast_1d(np.asarray(tide)).astype(float)
    minor_arr = np.atleast_1d(np.asarray(minor)).astype(float)
    elevations = tide_arr.flatten() + minor_arr.flatten() + datum_offset

    return find_extrema(times, elevations)
