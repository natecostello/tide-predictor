import datetime

import numpy as np
from scipy.signal import find_peaks

from tides.models import Coordinate, TideEvent

MODEL_NAME = "GOT5.6"
ELEVATION_INTERVAL_MINUTES = 6

# Minimum separation between peaks in minutes. Tidal extrema are typically
# ~6 hours apart; 2 hours is a conservative minimum to filter noise.
_MIN_PEAK_SEPARATION_MINUTES = 120
_MIN_PEAK_DISTANCE = _MIN_PEAK_SEPARATION_MINUTES // ELEVATION_INTERVAL_MINUTES


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


def compute_tides(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
) -> list[TideEvent]:
    import pyTMD.compute

    from tides.cache import ensure_model_data, get_model_dir

    ensure_model_data()

    start = datetime.datetime(
        begin_date.year, begin_date.month, begin_date.day, tzinfo=datetime.timezone.utc
    )
    end = datetime.datetime(
        end_date.year, end_date.month, end_date.day, tzinfo=datetime.timezone.utc
    ) + datetime.timedelta(days=1)

    times = []
    current = start
    while current < end:
        times.append(current)
        current += datetime.timedelta(minutes=ELEVATION_INTERVAL_MINUTES)

    lons = np.full(len(times), coord.lon)
    lats = np.full(len(times), coord.lat)

    elevations = pyTMD.compute.tide_elevations(
        lons,
        lats,
        times,
        DIRECTORY=str(get_model_dir()),
        MODEL="GOT5.6",
        EPSG=4326,
    )

    return find_extrema(times, elevations)
