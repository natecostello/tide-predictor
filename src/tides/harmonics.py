"""Tidal harmonic prediction engine.

Predicts water levels from harmonic constituent amplitudes and phases
using pyTMD's astronomical argument computation for proper nodal corrections.
"""

import datetime
import functools

import numpy as np
import pyTMD.constituents

from tides.models import TideEvent
from tides.ocean_model import ELEVATION_INTERVAL_MINUTES, find_extrema

# MJD of the tide epoch (1992-01-01)
_MJD_TIDE = 48622


@functools.cache
def _known_constituents() -> frozenset[str]:
    """Get the set of constituent names pyTMD recognizes."""
    # Test common constituents to build the known set
    candidates = [
        "m2",
        "s2",
        "n2",
        "k1",
        "o1",
        "k2",
        "p1",
        "q1",
        "m4",
        "m6",
        "mk3",
        "s4",
        "mn4",
        "nu2",
        "s6",
        "mu2",
        "2n2",
        "oo1",
        "lam2",
        "s1",
        "m1",
        "j1",
        "mm",
        "ssa",
        "sa",
        "msf",
        "mf",
        "rho",
        "t2",
        "r2",
        "2q1",
        "2sm2",
        "m3",
        "l2",
        "2mk3",
        "m8",
        "ms4",
        "eps2",
        "eta2",
        "mks2",
        "m1'",
        "n2'",
        "l2'",
        "m3'",
    ]
    known = set()
    for name in candidates:
        try:
            pyTMD.constituents.coefficients_table([name])
            known.add(name)
        except (ValueError, KeyError):
            pass
    return frozenset(known)


def _filter_constituents(
    constituents: list[dict],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Filter constituents to those pyTMD recognizes.

    Returns (names, amplitudes, phases_deg) for valid constituents only.
    """
    known = _known_constituents()
    names = []
    amps = []
    phases = []
    for c in constituents:
        name = c["name"].lower()
        if name in known and c["amplitude"] > 0:
            names.append(name)
            amps.append(c["amplitude"])
            phases.append(c["phase"])
    return names, np.array(amps), np.array(phases)


def predict_tide_height(
    dt: datetime.datetime,
    constituents: list[dict],
    datum_offset: float = 0.0,
) -> float:
    """Predict tide height at a single datetime from harmonic constituents.

    Uses pyTMD's astronomical argument computation for nodal corrections.

    Args:
        dt: UTC datetime
        constituents: List of {"name": str, "amplitude": float, "phase": float}
        datum_offset: Constant to add (e.g., MLLW offset from MSL)

    Returns:
        Predicted height in meters
    """
    if not constituents:
        return 0.0

    names, amplitudes, phases_deg = _filter_constituents(constituents)
    if len(names) == 0:
        return datum_offset

    # Convert to MJD
    epoch = datetime.datetime(1992, 1, 1, tzinfo=datetime.timezone.utc)
    days_since_epoch = (dt - epoch).total_seconds() / 86400.0
    mjd = np.array([days_since_epoch + _MJD_TIDE])

    # Get astronomical arguments with nodal corrections
    pu, pf, greenwich_phase = pyTMD.constituents.arguments(mjd, names, corrections="GOT")

    # theta = radians(G) + pu (GOT-style phase computation)
    theta = np.radians(greenwich_phase[0, :]) + pu[0, :]

    # h(t) = sum( pf * A * cos(theta - phase_rad) )
    phases_rad = np.radians(phases_deg)
    height = float(np.sum(pf[0, :] * amplitudes * np.cos(theta - phases_rad)))

    return height + datum_offset


def predict_tides_for_day(
    date: datetime.date,
    constituents: list[dict],
    datum_offset: float = 0.0,
) -> list[TideEvent]:
    """Predict high/low tides for a single day.

    Computes tide heights at fine intervals and finds extrema.
    """
    start = datetime.datetime(date.year, date.month, date.day, tzinfo=datetime.timezone.utc)
    end = start + datetime.timedelta(days=1)

    times = []
    current = start
    while current < end:
        times.append(current)
        current += datetime.timedelta(minutes=ELEVATION_INTERVAL_MINUTES)

    elevations = np.array([predict_tide_height(t, constituents, datum_offset) for t in times])

    return find_extrema(times, elevations)
