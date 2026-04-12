import datetime

import numpy as np
from scipy.signal import find_peaks

from tides.models import Coordinate, TideEvent

DEFAULT_MODEL = "GOT5.6"
SUPPORTED_MODELS = {"GOT5.6", "GOT5.5", "EOT20", "FES2022"}
ELEVATION_INTERVAL_MINUTES = 6

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


def compute_tides(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    model_name: str = DEFAULT_MODEL,
) -> list[TideEvent]:
    import pyTMD.io
    import pyTMD.predict

    from tides.cache import ensure_model_data

    ensure_model_data(model_name)

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

    # pyTMD predict.time_series expects days since 1992-01-01
    t = np.array([(dt - _PYTMD_PREDICT_EPOCH).total_seconds() / 86400.0 for dt in times])

    # Load model and interpolate constituents at the coordinate.
    # Use crop with bounds for large models (FES2022 is 16 GB uncropped).
    m = pyTMD.io.model()
    m.from_database(model_name)
    pad = 2.0  # degrees padding around target for interpolation
    lat_min = max(coord.lat - pad, -90.0)
    lat_max = min(coord.lat + pad, 90.0)
    lon_min = coord.lon - pad
    lon_max = coord.lon + pad
    # Normalize longitude to [-180, 360] range for pyTMD compatibility
    if lon_min < -180:
        lon_min += 360
    if lon_max > 360:
        lon_max -= 360
    bounds = [lon_min, lon_max, lat_min, lat_max]
    ds = m.open_dataset(crop=True, bounds=bounds)
    local = ds.tmd.interp(x=coord.lon, y=coord.lat, extrapolate=True, cutoff=10)

    # Predict tidal time series using the model's correction type
    tide = pyTMD.predict.time_series(t, local, corrections=m.corrections)

    # Infer minor constituents not included in the model
    minor = pyTMD.predict.infer_minor(t, local, corrections=m.corrections, minor=m.minor)

    elevations = (tide.values + minor.values).astype(float).flatten()

    return find_extrema(times, elevations)
