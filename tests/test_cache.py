import json
import os
import time
from unittest.mock import MagicMock, patch

from tides.cache import (
    STATION_CACHE_MAX_AGE_DAYS,
    _model_exists,
    ensure_model_data,
    fetch_all,
    fetch_station_data,
    get_cache_dir,
    get_model_dir,
    get_station_cache_path,
    get_stations,
    is_station_cache_fresh,
    load_station_cache,
    save_station_cache,
)


class TestGetCacheDir:
    def test_default_cache_dir(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("XDG_CACHE_HOME", None)
            d = get_cache_dir()
            assert d.name == "tides"
            assert d.parent.name == ".cache"

    def test_xdg_cache_home(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_cache_dir()
            assert d == tmp_path / "tides"

    def test_cache_dir_is_created(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_cache_dir()
            assert d.exists()


class TestGetModelDir:
    def test_model_dir_under_cache(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_model_dir()
            assert d == tmp_path / "tides" / "models"

    def test_model_dir_is_created(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_model_dir()
            assert d.exists()
            assert d.is_dir()


class TestGetStationCachePath:
    def test_station_cache_path(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            p = get_station_cache_path()
            assert p == tmp_path / "tides" / "noaa_stations.json"


class TestIsStationCacheFresh:
    def test_missing_file_returns_false(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            assert is_station_cache_fresh() is False

    def test_fresh_file_returns_true(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            path = get_station_cache_path()
            path.write_text("[]")
            # File was just created, so mtime is now -- should be fresh
            assert is_station_cache_fresh() is True

    def test_stale_file_returns_false(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            path = get_station_cache_path()
            path.write_text("[]")
            # Set mtime to 31 days ago
            stale_time = time.time() - (STATION_CACHE_MAX_AGE_DAYS + 1) * 86400
            os.utime(path, (stale_time, stale_time))
            assert is_station_cache_fresh() is False

    def test_file_exactly_at_boundary_returns_false(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            path = get_station_cache_path()
            path.write_text("[]")
            # Set mtime to exactly 30 days ago -- age.days == 30, not < 30
            boundary_time = time.time() - STATION_CACHE_MAX_AGE_DAYS * 86400
            os.utime(path, (boundary_time, boundary_time))
            assert is_station_cache_fresh() is False

    def test_file_just_under_boundary_returns_true(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            path = get_station_cache_path()
            path.write_text("[]")
            # Set mtime to 29 days ago -- age.days == 29, which is < 30
            under_boundary = time.time() - (STATION_CACHE_MAX_AGE_DAYS - 1) * 86400
            os.utime(path, (under_boundary, under_boundary))
            assert is_station_cache_fresh() is True


class TestSaveAndLoadStationCache:
    def test_round_trip(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            stations = [
                {"id": "8454000", "name": "Providence", "lat": 41.8, "lon": -71.4},
                {"id": "8461490", "name": "New London", "lat": 41.35, "lon": -72.09},
            ]
            save_station_cache(stations)
            loaded = load_station_cache()
            assert loaded == stations

    def test_round_trip_empty_list(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            save_station_cache([])
            loaded = load_station_cache()
            assert loaded == []

    def test_save_overwrites_existing(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            save_station_cache([{"id": "1"}])
            save_station_cache([{"id": "2"}])
            loaded = load_station_cache()
            assert loaded == [{"id": "2"}]

    def test_save_creates_valid_json(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            stations = [{"id": "123", "name": "Test"}]
            save_station_cache(stations)
            path = get_station_cache_path()
            raw = path.read_text()
            parsed = json.loads(raw)
            assert parsed == stations


class TestLoadStationCache:
    def test_missing_file_returns_none(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            result = load_station_cache()
            assert result is None

    def test_corrupt_json_returns_none(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            path = get_station_cache_path()
            path.write_text("{not valid json!!!")
            result = load_station_cache()
            assert result is None

    def test_corrupt_json_deletes_file(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            path = get_station_cache_path()
            path.write_text("{not valid json!!!")
            load_station_cache()
            assert not path.exists()

    def test_empty_file_returns_none_and_deletes(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            path = get_station_cache_path()
            path.write_text("")
            result = load_station_cache()
            assert result is None
            assert not path.exists()


class TestModelExists:
    def test_returns_true_when_model_found(self):
        mock_model_instance = MagicMock()
        mock_model_class = MagicMock(return_value=mock_model_instance)

        with patch("pyTMD.io.model", mock_model_class):
            result = _model_exists("GOT5.6")
            assert result is True
            mock_model_instance.from_database.assert_called_once_with("GOT5.6")

    def test_returns_false_on_file_not_found(self):
        mock_model_instance = MagicMock()
        mock_model_instance.from_database.side_effect = FileNotFoundError("not found")
        mock_model_class = MagicMock(return_value=mock_model_instance)

        with patch("pyTMD.io.model", mock_model_class):
            result = _model_exists("GOT5.6")
            assert result is False


class TestEnsureModelData:
    def test_early_return_when_model_exists(self):
        with patch("tides.cache._model_exists", return_value=True) as mock_exists:
            with patch("tides.cache.print") as mock_print:
                ensure_model_data()
                mock_exists.assert_called_once()
                mock_print.assert_not_called()

    def test_downloads_when_model_missing(self):
        with patch("tides.cache._model_exists", return_value=False):
            with patch("pyTMD.datasets.fetch_gsfc_got") as mock_fetch:
                with patch("tides.cache.print"):
                    ensure_model_data()
                    assert mock_fetch.call_count == 2
                    calls = mock_fetch.call_args_list
                    assert calls[0] == ((), {"model": "GOT5.5", "format": "netcdf"})
                    assert calls[1] == ((), {"model": "GOT5.6", "format": "netcdf"})


class TestFetchStationData:
    def test_fetches_parses_and_caches(self, tmp_path):
        stations = [{"id": "1234", "name": "TestStation"}]
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch(
                "tides.noaa.fetch_station_list_xml", return_value="<xml/>"
            ) as mock_fetch_xml:
                with patch("tides.noaa.parse_station_list", return_value=stations) as mock_parse:
                    result = fetch_station_data()
                    assert result == stations
                    mock_fetch_xml.assert_called_once()
                    mock_parse.assert_called_once_with("<xml/>")

    def test_saves_to_cache_file(self, tmp_path):
        stations = [{"id": "5678", "name": "Another"}]
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("tides.noaa.fetch_station_list_xml", return_value="<xml/>"):
                with patch("tides.noaa.parse_station_list", return_value=stations):
                    fetch_station_data()
                    # Verify cache file was written
                    loaded = load_station_cache()
                    assert loaded == stations


class TestGetStations:
    def test_cache_hit_returns_cached_data(self):
        cached_stations = [{"id": "111", "name": "Cached"}]
        with patch("tides.cache.is_station_cache_fresh", return_value=True):
            with patch("tides.cache.load_station_cache", return_value=cached_stations):
                with patch("tides.cache.fetch_station_data") as mock_fetch:
                    result = get_stations()
                    assert result == cached_stations
                    mock_fetch.assert_not_called()

    def test_cache_fresh_but_load_returns_none_falls_through(self):
        fetched_stations = [{"id": "222", "name": "Fetched"}]
        with patch("tides.cache.is_station_cache_fresh", return_value=True):
            with patch("tides.cache.load_station_cache", return_value=None):
                with patch(
                    "tides.cache.fetch_station_data", return_value=fetched_stations
                ) as mock_fetch:
                    result = get_stations()
                    assert result == fetched_stations
                    mock_fetch.assert_called_once()

    def test_cache_miss_fetches_fresh_data(self):
        fetched_stations = [{"id": "333", "name": "Fresh"}]
        with patch("tides.cache.is_station_cache_fresh", return_value=False):
            with patch(
                "tides.cache.fetch_station_data", return_value=fetched_stations
            ) as mock_fetch:
                result = get_stations()
                assert result == fetched_stations
                mock_fetch.assert_called_once()


class TestFetchAll:
    def test_calls_fetch_station_data_and_ensure_model_data(self):
        with patch("tides.cache.fetch_station_data") as mock_fetch:
            with patch("tides.cache.ensure_model_data") as mock_ensure:
                with patch("tides.cache.print"):
                    fetch_all()
                    mock_fetch.assert_called_once()
                    mock_ensure.assert_called_once()

    def test_prints_status_messages(self, capsys):
        with patch("tides.cache.fetch_station_data"):
            with patch("tides.cache.ensure_model_data"):
                fetch_all()
                captured = capsys.readouterr()
                assert "Fetching NOAA station list" in captured.err
                assert "GOT5.6" in captured.err
                assert "Done" in captured.err
