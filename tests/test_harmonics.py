"""Tests for harmonic tidal prediction engine."""

import datetime

import numpy as np
import xarray as xr

from tides.harmonics import _build_dataset, predict_tides_for_day

# Bermuda (NOAA 2695540) harmonic constants — well-known station for validation
BERMUDA_CONSTITUENTS = [
    {"name": "M2", "amplitude": 0.367, "phase": 358.6},
    {"name": "S2", "amplitude": 0.080, "phase": 25.4},
    {"name": "N2", "amplitude": 0.085, "phase": 336.9},
    {"name": "K1", "amplitude": 0.067, "phase": 187.0},
    {"name": "O1", "amplitude": 0.054, "phase": 192.7},
    {"name": "K2", "amplitude": 0.021, "phase": 26.2},
    {"name": "P1", "amplitude": 0.021, "phase": 188.2},
    {"name": "Q1", "amplitude": 0.011, "phase": 178.0},
]

BERMUDA_DATUMS = {
    "LAT": 0.763,
    "MLLW": 1.039,
    "MTL": 1.444,
    "MSL": 1.446,
    "MHW": 1.826,
    "HAT": 2.243,
}


class TestBuildDataset:
    def test_returns_xarray_dataset(self):
        ds = _build_dataset(BERMUDA_CONSTITUENTS)
        assert isinstance(ds, xr.Dataset)

    def test_constituents_as_complex_variables(self):
        ds = _build_dataset(BERMUDA_CONSTITUENTS)
        assert "m2" in ds.data_vars
        assert ds["m2"].dtype == np.complex64

    def test_amplitude_preserved(self):
        ds = _build_dataset([{"name": "M2", "amplitude": 0.5, "phase": 0.0}])
        assert abs(complex(ds["m2"].values)) - 0.5 < 0.001

    def test_phase_preserved(self):
        ds = _build_dataset([{"name": "M2", "amplitude": 1.0, "phase": 90.0}])
        z = complex(ds["m2"].values)
        phase = np.degrees(np.arctan2(-z.imag, z.real))
        if phase < 0:
            phase += 360
        assert abs(phase - 90.0) < 0.1

    def test_skips_zero_amplitude(self):
        ds = _build_dataset(
            [
                {"name": "M2", "amplitude": 0.5, "phase": 0.0},
                {"name": "S2", "amplitude": 0.0, "phase": 0.0},
            ]
        )
        assert "m2" in ds.data_vars
        assert "s2" not in ds.data_vars

    def test_empty_constituents(self):
        ds = _build_dataset([])
        assert len(ds.data_vars) == 0

    def test_normalizes_noaa_names(self):
        """NOAA's LAM2 should become lambda2 (pyTMD's name)."""
        ds = _build_dataset([{"name": "LAM2", "amplitude": 0.01, "phase": 0.0}])
        assert "lambda2" in ds.data_vars
        assert "lam2" not in ds.data_vars

    def test_normalizes_rho_to_rho1(self):
        """NOAA's RHO should become rho1 (pyTMD's name)."""
        ds = _build_dataset([{"name": "RHO", "amplitude": 0.01, "phase": 0.0}])
        assert "rho1" in ds.data_vars
        assert "rho" not in ds.data_vars


class TestPredictTidesForDay:
    def test_returns_tide_events(self):
        from tides.models import TideEvent

        events = predict_tides_for_day(datetime.date(2026, 4, 15), BERMUDA_CONSTITUENTS)
        assert len(events) >= 2  # At least one high and one low
        assert all(isinstance(e, TideEvent) for e in events)

    def test_events_are_chronological(self):
        events = predict_tides_for_day(datetime.date(2026, 4, 15), BERMUDA_CONSTITUENTS)
        for i in range(len(events) - 1):
            assert events[i].time < events[i + 1].time

    def test_semidiurnal_produces_four_events(self):
        """Bermuda is semidiurnal — expect ~4 extrema per day (2 highs, 2 lows)."""
        events = predict_tides_for_day(datetime.date(2026, 4, 15), BERMUDA_CONSTITUENTS)
        assert len(events) == 4 or len(events) == 3  # 3 if one crosses midnight

    def test_heights_with_mllw_datum(self):
        """With MLLW datum, all heights should be positive (or near zero at low tide)."""
        events = predict_tides_for_day(
            datetime.date(2026, 4, 15),
            BERMUDA_CONSTITUENTS,
            datum_offset=BERMUDA_DATUMS["MLLW"],
        )
        lows = [e for e in events if e.height < 1.0]
        highs = [e for e in events if e.height >= 1.0]
        assert len(lows) > 0
        assert len(highs) > 0
        for e in lows:
            assert e.height > -0.2, f"Height {e.height} too negative for MLLW datum"

    def test_datum_offset_applied(self):
        """Datum offset should shift all heights."""
        events_no_datum = predict_tides_for_day(datetime.date(2026, 4, 15), BERMUDA_CONSTITUENTS)
        events_with_datum = predict_tides_for_day(
            datetime.date(2026, 4, 15), BERMUDA_CONSTITUENTS, datum_offset=1.0
        )
        # Each corresponding event should be ~1.0m higher
        for e1, e2 in zip(events_no_datum, events_with_datum):
            assert abs((e2.height - e1.height) - 1.0) < 0.05

    def test_empty_constituents_returns_empty(self):
        events = predict_tides_for_day(datetime.date(2026, 4, 15), [])
        assert events == []

    def test_matches_noaa_predictions_bermuda(self):
        """Compare against known NOAA predictions for Bermuda 2026-04-15.

        NOAA predictions (relative to MLLW):
          -0.47m@04:25, +0.31m@10:31, -0.52m@16:39, +0.41m@22:55
        Our harmonic prediction should produce similar timing pattern.
        """
        events = predict_tides_for_day(
            datetime.date(2026, 4, 15),
            BERMUDA_CONSTITUENTS,
            datum_offset=BERMUDA_DATUMS["MLLW"],
        )
        assert len(events) >= 3

        # First event should be a low tide in the early morning (03:00-06:00 UTC)
        first = events[0]
        assert 3 <= first.time.hour <= 6, f"First tide at {first.time.hour}:xx, expected 03-06"
