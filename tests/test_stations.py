"""Tests for global tide station database."""

import datetime

from tides.models import Coordinate, TideEvent
from tides.stations import (
    find_nearest_station,
    predict_station_tides,
)

SAMPLE_STATION = {
    "name": "Fortaleza",
    "region": "Ceara",
    "country": "Brazil",
    "latitude": -3.717,
    "longitude": -38.467,
    "timezone": "America/Fortaleza",
    "datums": {
        "LAT": -1.639,
        "MLLW": -0.964,
        "MTL": 0.005,
        "MSL": 0.0,
        "MHW": 0.973,
        "HAT": 1.718,
    },
    "chart_datum": "LAT",
    "harmonic_constituents": [
        {"name": "M2", "amplitude": 0.92166, "phase": 215.964},
        {"name": "S2", "amplitude": 0.29110, "phase": 240.742},
        {"name": "N2", "amplitude": 0.20232, "phase": 200.352},
        {"name": "K1", "amplitude": 0.06694, "phase": 167.670},
        {"name": "O1", "amplitude": 0.04855, "phase": 147.148},
    ],
}

SAMPLE_INDEX = [
    {
        "id": "fortaleza-283b",
        "name": "Fortaleza",
        "lat": -3.717,
        "lon": -38.467,
        "source": "ticon",
    },
    {
        "id": "2695540",
        "name": "St. Georges Island",
        "lat": 32.373,
        "lon": -64.703,
        "source": "noaa",
    },
    {
        "id": "8518750",
        "name": "The Battery",
        "lat": 40.700,
        "lon": -74.014,
        "source": "noaa",
    },
]


class TestFindNearestStation:
    def test_finds_closest(self):
        coord = Coordinate(lat=-3.4, lon=-39.0)
        station, dist = find_nearest_station(SAMPLE_INDEX, coord)
        assert station["name"] == "Fortaleza"
        assert dist < 100  # ~70km

    def test_returns_none_when_too_far(self):
        coord = Coordinate(lat=60.0, lon=0.0)  # North Sea, far from all
        result = find_nearest_station(SAMPLE_INDEX, coord, max_distance_km=50)
        assert result is None

    def test_respects_max_distance(self):
        coord = Coordinate(lat=-3.4, lon=-39.0)
        result = find_nearest_station(SAMPLE_INDEX, coord, max_distance_km=10)
        assert result is None  # Fortaleza is ~70km away


class TestPredictStationTides:
    def test_returns_events_for_date(self):
        events = predict_station_tides(
            SAMPLE_STATION,
            datetime.date(2026, 4, 15),
            datetime.date(2026, 4, 15),
        )
        assert len(events) >= 2
        assert all(isinstance(e, TideEvent) for e in events)

    def test_events_use_chart_datum(self):
        """Station's chart_datum is LAT — heights should be relative to LAT."""
        events = predict_station_tides(
            SAMPLE_STATION,
            datetime.date(2026, 4, 15),
            datetime.date(2026, 4, 15),
        )
        # With LAT datum, all heights should be positive (LAT is the lowest)
        for e in events:
            assert e.height >= -0.1, f"Height {e.height} too negative for LAT datum"

    def test_multi_day_range(self):
        events = predict_station_tides(
            SAMPLE_STATION,
            datetime.date(2026, 4, 15),
            datetime.date(2026, 4, 16),
        )
        # Two days should have ~8 events (4 per day for semidiurnal)
        assert len(events) >= 6

    def test_fortaleza_amplitude_realistic(self):
        """Fortaleza has ~2m tidal range — heights should reflect this."""
        events = predict_station_tides(
            SAMPLE_STATION,
            datetime.date(2026, 4, 15),
            datetime.date(2026, 4, 15),
        )
        heights = [e.height for e in events]
        tidal_range = max(heights) - min(heights)
        assert tidal_range > 1.0, f"Tidal range {tidal_range}m too small for Fortaleza"
