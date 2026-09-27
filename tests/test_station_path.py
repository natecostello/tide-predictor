"""Station path: skip stations without harmonics; GOT corrections (#15)."""

import datetime
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyTMD.predict

from tides.harmonics import STATION_CORRECTIONS, _build_dataset, predict_elevations
from tides.models import Coordinate
from tides.ocean_model import utc_minutes
from tides.stations import find_nearest_usable_station

FIXTURES = Path(__file__).parent / "fixtures"
STATION = json.loads((FIXTURES / "station_9414290.json").read_text())
NOAA_MSL = json.loads((FIXTURES / "noaa_9414290_msl_6min_20260915_16.json").read_text())

# NOAA's own 6-min MSL predictions at Golden Gate, 2026-09-15..16 (480 points).
OBSERVED = np.array([float(p["v"]) for p in NOAA_MSL["predictions"]])
START = datetime.datetime(2026, 9, 15, tzinfo=datetime.timezone.utc)
T, _ = utc_minutes(START, START + datetime.timedelta(days=2), 6)
RMS_THRESHOLD_M = 0.025


def _rms(pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - OBSERVED) ** 2)))


class TestStationAccuracy:
    def test_station_prediction_matches_noaa(self):
        assert STATION_CORRECTIONS == "GOT"
        rms = _rms(predict_elevations(STATION["harmonic_constituents"], T))
        assert rms < RMS_THRESHOLD_M, f"RMS {rms * 100:.2f} cm"

    def test_old_otis_plus_infer_minor_was_worse(self):
        """Regression guard: the previous configuration fails the same bar."""
        ds = _build_dataset(STATION["harmonic_constituents"])
        old = np.asarray(pyTMD.predict.time_series(T, ds, corrections="OTIS")).ravel()
        old = old + np.asarray(pyTMD.predict.infer_minor(T, ds, corrections="OTIS")).ravel()
        assert _rms(old) > RMS_THRESHOLD_M


class TestUsableStation:
    INDEX = [
        {"id": "sub", "name": "Subordinate", "lat": 38.89, "lon": -76.54, "file": "noaa/sub.json"},
        {"id": "ref", "name": "Reference", "lat": 38.96, "lon": -76.48, "file": "noaa/ref.json"},
    ]
    DATA = {
        "noaa/sub.json": {"name": "Subordinate", "harmonic_constituents": []},
        "noaa/ref.json": {
            "name": "Reference",
            "harmonic_constituents": [{"name": "M2", "amplitude": 0.2, "phase": 10.0}],
        },
    }

    def test_skips_nearest_without_harmonics(self):
        with patch("tides.stations.load_station", side_effect=lambda e: self.DATA[e["file"]]):
            entry, station, dist = find_nearest_usable_station(
                self.INDEX, Coordinate(lat=38.8867, lon=-76.54), max_distance_km=200
            )
        assert entry["id"] == "ref"
        assert station["name"] == "Reference"
        assert dist > 0

    def test_none_when_no_usable_station_in_range(self):
        with patch("tides.stations.load_station", side_effect=lambda e: self.DATA[e["file"]]):
            assert (
                find_nearest_usable_station(
                    self.INDEX[:1], Coordinate(lat=38.8867, lon=-76.54), max_distance_km=200
                )
                is None
            )
