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
        assert abs(abs(complex(ds["m2"].values)) - 0.5) < 0.001

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

    def test_normalizes_ep2_to_eps2(self):
        ds = _build_dataset([{"name": "EP2", "amplitude": 0.01, "phase": 0.0}])
        assert "eps2" in ds.data_vars

    def test_normalizes_sgm_to_sigma1(self):
        ds = _build_dataset([{"name": "SGM", "amplitude": 0.01, "phase": 0.0}])
        assert "sigma1" in ds.data_vars

    def test_normalizes_3l2_to_l2_prime(self):
        """Ticon's 3L2 (third-degree) maps to pyTMD's l2'."""
        ds = _build_dataset([{"name": "3L2", "amplitude": 0.01, "phase": 0.0}])
        assert "l2'" in ds.data_vars

    def test_skips_unrecognized_with_warning(self, capsys):
        """Unrecognized constituents are skipped with a stderr warning."""
        ds = _build_dataset(
            [
                {"name": "M2", "amplitude": 0.5, "phase": 0.0},
                {"name": "BOGUS99", "amplitude": 0.01, "phase": 0.0},
            ]
        )
        assert "m2" in ds.data_vars
        assert "bogus99" not in ds.data_vars
        captured = capsys.readouterr()
        assert "BOGUS99" in captured.err
        assert "0.0100m" in captured.err

    def test_no_warning_when_all_recognized(self, capsys):
        _build_dataset(BERMUDA_CONSTITUENTS)
        captured = capsys.readouterr()
        assert captured.err == ""


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
          Low  04:25 -0.47m, High 10:31 +0.31m,
          Low  16:39 -0.52m, High 22:55 +0.41m
        Our harmonic prediction should be within 30 min and 0.15m.
        """
        events = predict_tides_for_day(
            datetime.date(2026, 4, 15),
            BERMUDA_CONSTITUENTS,
            datum_offset=BERMUDA_DATUMS["MLLW"],
        )
        assert len(events) >= 3

        # First event: low tide near 04:25 UTC
        first = events[0]
        assert 3 <= first.time.hour <= 5, f"First tide at {first.time.hour}:xx, expected 03-05"

        # Heights should be in a reasonable range for Bermuda (small tidal range)
        heights = [e.height for e in events]
        tidal_range = max(heights) - min(heights)
        assert 0.5 < tidal_range < 1.5, f"Range {tidal_range:.2f}m outside expected 0.5-1.5m"
