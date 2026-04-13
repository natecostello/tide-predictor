"""Tests for tidal datum computation and lookup."""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np

from tides.datums import (
    Datum,
    _extract_datums,
    _grid_key,
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


class TestGridKey:
    def test_got56_resolution(self):
        key = _grid_key(40.7, -74.0, "GOT5.6")
        # 0.5° grid: 40.7 rounds to 40.5, -74.0 stays
        assert key == "40.5,-74.0"

    def test_fes2022_resolution(self):
        key = _grid_key(40.7, -74.0, "FES2022")
        # 1/16° = 0.0625: 40.7/0.0625 = 651.2, round to 651 * 0.0625 = 40.6875
        assert "40.6875" in key

    def test_eot20_resolution(self):
        key = _grid_key(40.7, -74.0, "EOT20")
        # 0.125° grid: 40.7/0.125 = 325.6, round to 326 * 0.125 = 40.75
        assert "40.75" in key


class TestComputeDatumsFromModel:
    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_returns_all_datum_keys(self, mock_ensure, mock_model_cls, mock_predict, mock_infer):
        from tides.datums import compute_datums_from_model

        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        mock_instance.corrections = "GOT"
        mock_instance.format = "GOT-netcdf"

        mock_ds = MagicMock()
        mock_instance.open_dataset.return_value = mock_ds
        mock_local = MagicMock()
        mock_ds.tmd.interp.return_value = mock_local

        # Simulate 19 years of hourly tidal data
        n = int(19 * 365.25 * 24)
        t_arr = np.linspace(0, 19 * 365.25, n)
        tide_values = np.sin(2 * np.pi * t_arr / (12.42 / 24))

        mock_result = MagicMock()
        mock_result.values = tide_values
        mock_predict.return_value = tide_values

        mock_infer.return_value = np.zeros(n)

        result = compute_datums_from_model(-3.717, -38.483, "GOT5.6")
        for key in ("lat", "mllw", "mlw", "msl", "mtl", "mhw", "mhhw", "hat"):
            assert key in result
        assert result["msl"] == 0.0
        assert result["hat"] > 0
        assert result["lat"] < 0


class TestGetModelDatums:
    def test_returns_cached_value(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache_dir = tmp_path / "tides" / "datums"
            cache_dir.mkdir(parents=True)
            key = _grid_key(40.7, -74.0, "GOT5.6")
            cached = {key: {"mllw": -0.5, "mhw": 0.4, "msl": 0.0}}
            (cache_dir / "got5.6.json").write_text(json.dumps(cached))

            result = get_model_datums(40.7, -74.0, "GOT5.6")
            assert result["mllw"] == -0.5

    def test_computes_when_not_cached(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            mock_datums = {"mllw": -0.6, "mhw": 0.5, "msl": 0.0, "lat": -1.0, "hat": 1.0}
            with patch("tides.datums.compute_datums_from_model", return_value=mock_datums):
                result = get_model_datums(40.7, -74.0, "GOT5.6")
                assert result["mllw"] == -0.6

            # Verify it was cached
            cache_file = tmp_path / "tides" / "datums" / "got5.6.json"
            assert cache_file.exists()
            cache = json.loads(cache_file.read_text())
            key = _grid_key(40.7, -74.0, "GOT5.6")
            assert key in cache

    def test_handles_corrupt_cache(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache_dir = tmp_path / "tides" / "datums"
            cache_dir.mkdir(parents=True)
            (cache_dir / "got5.6.json").write_text("not json!!!")

            mock_datums = {"mllw": -0.6, "msl": 0.0}
            with patch("tides.datums.compute_datums_from_model", return_value=mock_datums):
                result = get_model_datums(40.7, -74.0, "GOT5.6")
                assert result["mllw"] == -0.6
