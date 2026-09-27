"""Tidal harmonic prediction from station constituent data.

Constructs an xarray Dataset from station harmonic constituents and
predicts water levels using pyTMD's predict.time_series() -- the same
pipeline used by the gridded model path (ocean_model.py).
"""

import datetime
import functools
import sys

import numpy as np
import pyTMD.constituents
import xarray as xr

from tides.models import TideEvent
from tides.ocean_model import ELEVATION_INTERVAL_MINUTES, find_extrema, utc_minutes

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
    "ep2": "eps2",
    "sgm": "sigma1",
    "3l2": "l2'",
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
    skipped = []
    for c in constituents:
        name = _normalize_name(c["name"].lower())
        amp = c["amplitude"]
        if amp <= 0:
            continue
        if not _is_recognized(name):
            skipped.append((c["name"], amp))
            continue
        phase_rad = np.radians(c["phase"])
        z = amp * np.exp(-1j * phase_rad)
        data_vars[name] = xr.Variable((), np.complex64(z))

    if skipped:
        names = ", ".join(f"{n} ({a:.4f}m)" for n, a in skipped)
        print(
            f"Warning: skipped {len(skipped)} unrecognized constituent(s): {names}",
            file=sys.stderr,
        )

    return xr.Dataset(data_vars)


def predict_elevations(constituents: list[dict], t: np.ndarray, chunks: int = 1) -> np.ndarray:
    """Predict MSL-relative elevations from station constituents at times `t`.

    `t` is days since the pyTMD epoch (1992-01-01). Long series can be split
    into `chunks` to bound memory. Builds the constituent dataset once.
    """
    import pyTMD.predict

    ds = _build_dataset(constituents)
    if len(ds.data_vars) == 0:
        return np.full(len(t), np.nan)

    parts = []
    for c in np.array_split(np.asarray(t, dtype=float), max(1, chunks)):
        tide = pyTMD.predict.time_series(c, ds, corrections=STATION_CORRECTIONS)
        minor = pyTMD.predict.infer_minor(c, ds, corrections=STATION_CORRECTIONS)
        tide_arr = np.asarray(getattr(tide, "values", tide), dtype=float).ravel()
        minor_arr = np.asarray(getattr(minor, "values", minor), dtype=float).ravel()
        parts.append(tide_arr + minor_arr)
    return np.concatenate(parts)


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

    t, times = utc_minutes(start, end, ELEVATION_INTERVAL_MINUTES)

    tide = pyTMD.predict.time_series(t, ds, corrections=STATION_CORRECTIONS)
    minor = pyTMD.predict.infer_minor(t, ds, corrections=STATION_CORRECTIONS)

    tide_arr = np.atleast_1d(np.asarray(tide)).astype(float)
    minor_arr = np.atleast_1d(np.asarray(minor)).astype(float)
    elevations = tide_arr.flatten() + minor_arr.flatten() + datum_offset

    return find_extrema(times, elevations)
