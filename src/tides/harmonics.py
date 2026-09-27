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
from tides.ocean_model import (
    EDGE_PAD,
    ELEVATION_INTERVAL_MINUTES,
    find_extrema,
    in_window,
    utc_day_window,
    utc_minutes,
)

# Correction type for station harmonic constants.
# pyTMD's "GOT" branch computes astronomical arguments and frequencies from
# Doodson numbers for every constituent it recognizes. The "OTIS" branch takes
# frequencies from a fixed ~33-constituent table and silently gives any other
# constituent omega = 0 (a constant), which degraded TICON stations such as
# Fortaleza (25 constituents frozen). Station analyses are complete by design,
# so minor constituents are NOT inferred on top of them.
STATION_CORRECTIONS = "GOT"

# Map station constituent names to pyTMD's expected names.
# NOAA uses abbreviations (LAM2, RHO) that pyTMD doesn't recognize;
# pyTMD uses the full IHO names (lambda2, rho1).
# "3L2" is deliberately NOT mapped: pyTMD rejects "3l2", and its "l2'" could
# not be confirmed to be the same constituent (same Doodson number/frequency),
# so 3L2 is reported as skipped rather than predicted at a possibly wrong
# frequency. Its amplitude is typically a few mm.
_NAME_MAP: dict[str, str] = {
    "lam2": "lambda2",
    "rho": "rho1",
    "ep2": "eps2",
    "sgm": "sigma1",
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
        parts.append(np.asarray(getattr(tide, "values", tide), dtype=float).ravel())
    return np.concatenate(parts)


def predict_tides_range(
    start: datetime.datetime,
    end: datetime.datetime,
    constituents: list[dict],
    datum_offset: float = 0.0,
) -> list[TideEvent]:
    """High/low events in [start, end) from one continuous series padded by
    EDGE_PAD on each side (so extrema at the window edges are not lost)."""
    if not constituents:
        return []

    t, times = utc_minutes(start - EDGE_PAD, end + EDGE_PAD, ELEVATION_INTERVAL_MINUTES)
    elevations = predict_elevations(constituents, t) + datum_offset
    return in_window(find_extrema(times, elevations), start, end)


def predict_tides_for_day(
    date: datetime.date,
    constituents: list[dict],
    datum_offset: float = 0.0,
) -> list[TideEvent]:
    """Predict high/low tides for a single UTC day using pyTMD."""
    start, end = utc_day_window(date, date)
    return predict_tides_range(start, end, constituents, datum_offset)
