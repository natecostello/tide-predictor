"""Tests for tidal datum computation and lookup."""

import datetime
import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from tides.datums import (
    Datum,
    _cache_key,
    _extract_datums,
    datums_from_station,
    get_model_datums,
)


class TestDatum:
    def test_enum_values(self):
        assert Datum.MLLW.value == "mllw"
        assert Datum.LAT.value == "lat"
        assert Datum.HAT.value == "hat"
        assert Datum.MSL.value == "msl"

    def test_all_datums_in_enum(self):
        from tides.datums import SUPPORTED_DATUMS

        for d in SUPPORTED_DATUMS:
            assert Datum(d), f"{d} not in Datum enum"


class TestDatumsFromStation:
    def test_returns_offsets_relative_to_msl(self):
        station = {
            "datums": {
                "HAT": 1.5,
                "MHW": 0.8,
                "MSL": 0.0,
                "MLW": -0.7,
                "MLLW": -0.9,
                "LAT": -1.6,
            }
        }
        result = datums_from_station(station)
        assert result is not None
        assert result["msl"] == 0.0
        assert result["mhw"] == 0.8
        assert result["mlw"] == -0.7
        assert result["mllw"] == -0.9
        assert result["lat"] == -1.6
        assert result["hat"] == 1.5

    def test_offsets_relative_to_stnd(self):
        """Station datums are relative to STND; MSL may not be 0."""
        station = {
            "datums": {
                "HAT": 2.5,
                "MHW": 1.8,
                "MSL": 1.0,
                "MLW": 0.3,
                "MLLW": 0.1,
                "LAT": -0.6,
            }
        }
        result = datums_from_station(station)
        # All should be relative to MSL (=1.0 in STND)
        assert result["msl"] == 0.0
        assert abs(result["mhw"] - 0.8) < 0.001
        assert abs(result["mlw"] - (-0.7)) < 0.001
        assert abs(result["lat"] - (-1.6)) < 0.001

    def test_returns_none_when_no_datums(self):
        assert datums_from_station({}) is None
        assert datums_from_station({"datums": {}}) is None

    def test_uses_mtl_fallback_for_msl(self):
        station = {
            "datums": {
                "MTL": 0.5,
                "MHW": 1.0,
                "MLW": 0.0,
            }
        }
        result = datums_from_station(station)
        assert result["msl"] == 0.0
        # MHW should be relative to MTL (used as MSL proxy)
        assert abs(result["mhw"] - 0.5) < 0.001


class TestExtractDatums:
    def test_simple_sine_wave(self):
        """A sine wave should produce symmetric datums."""
        hours = 19 * 365 * 24
        t = np.linspace(0, 19 * 365 * 2 * np.pi / (12.42 / 24), hours)
        elevations = np.sin(t)
        result = _extract_datums(elevations)

        assert result["msl"] == 0.0
        assert result["hat"] > 0.99
        assert result["lat"] < -0.99
        assert result["mhw"] > 0
        assert result["mlw"] < 0

    def test_all_keys_present(self):
        elevations = np.sin(np.linspace(0, 100 * np.pi, 10000))
        result = _extract_datums(elevations)
        for key in ("lat", "mllw", "mlw", "msl", "mtl", "mhw", "mhhw", "hat"):
            assert key in result

    def test_datum_ordering(self):
        """Datums must follow: LAT <= MLLW <= MLW <= MSL <= MHW <= MHHW <= HAT."""
        elevations = np.sin(np.linspace(0, 200 * np.pi, 50000))
        result = _extract_datums(elevations)
        assert result["lat"] <= result["mllw"]
        assert result["mllw"] <= result["mlw"]
        assert result["mlw"] <= result["msl"]
        assert result["msl"] <= result["mhw"]
        assert result["mhw"] <= result["mhhw"]
        assert result["mhhw"] <= result["hat"]


class TestCacheKey:
    def test_rounds_to_hundredth_degree(self):
        assert _cache_key(40.7128, -74.0060) == "40.71,-74.01"

    def test_nearby_points_get_distinct_keys(self):
        # Golden Gate vs Alameda (~15 km apart) shared one key under the old
        # 0.5 deg grid key.
        assert _cache_key(37.8063, -122.4659) != _cache_key(37.7652, -122.2997)

    def test_model_independent(self):
        assert _cache_key(1.234, 5.678) == "1.23,5.68"


class TestComputeDatumsFromModel:
    def test_samples_6min_over_epoch_and_returns_all_keys(self):
        from tides.datums import DATUM_INTERVAL_MINUTES, compute_datums_from_model

        seen = []

        def fake_predict(t, local, model):
            seen.append(t)
            return np.sin(2 * np.pi * t / (12.4206 / 24))

        with (
            patch("tides.ocean_model.load_local_constituents", return_value=(None, None)),
            patch("tides.ocean_model.predict_elevations", side_effect=fake_predict),
        ):
            result = compute_datums_from_model(-3.717, -38.483, "GOT5.6")

        t = np.concatenate(seen)
        assert DATUM_INTERVAL_MINUTES == 6.0
        assert len(t) == 6940 * 240  # 6,940 days, end-exclusive at 2022-01-01
        assert np.allclose(np.diff(t), 6 / 1440)
        assert t[0] == (datetime.date(2003, 1, 1) - datetime.date(1992, 1, 1)).days
        assert len(seen) == 19  # one chunk per year
        for key in ("lat", "mllw", "mlw", "msl", "mtl", "mhw", "mhhw", "hat"):
            assert key in result
        assert result["msl"] == 0.0
        assert result["hat"] > 0 > result["lat"]


class TestExtractDatumsMethod:
    """#13: vectorized, tidal-day MHHW/MLLW at 6-minute sampling."""

    @staticmethod
    def _mixed_tide(years: float = 19.0, interval_min: float = 6.0) -> np.ndarray:
        t = np.arange(0, years * 365.25 * 24, interval_min / 60.0)  # hours
        periods = np.array([12.4206012, 23.9344696, 25.8193417])  # M2, K1, O1
        amps = np.array([1.0, 0.4, 0.3])
        w = 2 * np.pi / periods
        return (amps[:, None] * np.cos(w[:, None] * t + np.array([0.0, 0.0, 1.0])[:, None])).sum(
            axis=0
        )

    def test_long_run_mean_is_zero(self):
        # Informational: a harmonic series has no Z0, so MSL = 0 by definition.
        assert abs(float(np.mean(self._mixed_tide()))) < 0.005

    def test_mixed_tide_datums_match_brute_force(self):
        e = self._mixed_tide()
        d = _extract_datums(e, 6.0)
        # Brute-force reference: explicit per-tidal-day loop.
        from scipy.signal import find_peaks

        hi, _ = find_peaks(e, distance=20)
        lo, _ = find_peaks(-e, distance=20)
        spd = 24.8412 * 10
        nbins = int(len(e) // spd)
        hh = [e[hi[(hi // spd) == b]].max() for b in range(nbins) if np.any((hi // spd) == b)]
        ll = [e[lo[(lo // spd) == b]].min() for b in range(nbins) if np.any((lo // spd) == b)]
        assert d["mhhw"] == pytest.approx(np.mean(hh), abs=0.01)
        assert d["mllw"] == pytest.approx(np.mean(ll), abs=0.01)
        assert d["mhw"] == pytest.approx(np.mean(e[hi]), abs=0.01)
        assert d["mlw"] == pytest.approx(np.mean(e[lo]), abs=0.01)
        assert d["hat"] >= d["mhhw"] > d["mhw"] > 0 > d["mlw"] > d["mllw"] >= d["lat"]

    def test_fast_on_19_years_at_6_minutes(self):
        import time

        e = self._mixed_tide()
        _extract_datums(e[:10000], 6.0)  # warm scipy import
        start = time.perf_counter()
        _extract_datums(e, 6.0)
        assert time.perf_counter() - start < 1.0


class TestGetModelDatums:
    def test_returns_cached_value(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache_dir = tmp_path / "tides" / "datums"
            cache_dir.mkdir(parents=True)
            key = _cache_key(40.7, -74.0)
            cached = {key: {"mllw": -0.5, "mhw": 0.4, "msl": 0.0}}
            (cache_dir / "got5.6.v3.json").write_text(json.dumps(cached))

            result = get_model_datums(40.7, -74.0, "GOT5.6")
            assert result["mllw"] == -0.5

    def test_ignores_older_cache_versions(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache_dir = tmp_path / "tides" / "datums"
            cache_dir.mkdir(parents=True)
            key = _cache_key(40.7, -74.0)
            (cache_dir / "got5.6.v2.json").write_text(json.dumps({key: {"mllw": -9.9}}))
            fresh = {"mllw": -0.6, "msl": 0.0}
            with patch("tides.datums.compute_datums_from_model", return_value=fresh) as comp:
                result = get_model_datums(40.7, -74.0, "GOT5.6")
            comp.assert_called_once()
            assert result["mllw"] == -0.6
            assert (cache_dir / "got5.6.v3.json").exists()
            # Old file left in place (never deleted by code).
            assert json.loads((cache_dir / "got5.6.v2.json").read_text())[key]["mllw"] == -9.9

    def test_computes_when_not_cached(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            mock_datums = {"mllw": -0.6, "mhw": 0.5, "msl": 0.0, "lat": -1.0, "hat": 1.0}
            with patch("tides.datums.compute_datums_from_model", return_value=mock_datums):
                result = get_model_datums(40.7, -74.0, "GOT5.6")
                assert result["mllw"] == -0.6

            # Verify it was cached
            cache_file = tmp_path / "tides" / "datums" / "got5.6.v3.json"
            assert cache_file.exists()
            cache = json.loads(cache_file.read_text())
            key = _cache_key(40.7, -74.0)
            assert key in cache

    def test_handles_corrupt_cache(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache_dir = tmp_path / "tides" / "datums"
            cache_dir.mkdir(parents=True)
            (cache_dir / "got5.6.v3.json").write_text("not json!!!")

            mock_datums = {"mllw": -0.6, "msl": 0.0}
            with patch("tides.datums.compute_datums_from_model", return_value=mock_datums):
                result = get_model_datums(40.7, -74.0, "GOT5.6")
                assert result["mllw"] == -0.6
