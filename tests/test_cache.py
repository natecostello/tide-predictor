import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tides.cache import (
    STATION_CACHE_MAX_AGE_DAYS,
    _dir_size,
    _model_exists,
    atomic_write_text,
    clear_cache,
    ensure_model_data,
    fetch_all,
    fetch_station_data,
    format_size,
    get_cache_dir,
    get_cache_info,
    get_station_cache_path,
    get_stations,
    is_station_cache_fresh,
    load_station_cache,
    plan_clear,
    save_station_cache,
)


class TestGetCacheDir:
    def test_default_cache_dir(self, monkeypatch):
        # Keep the isolated HOME from conftest; only drop XDG_CACHE_HOME so the
        # ~/.cache fallback is exercised under tmp_path, never the real home.
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        d = get_cache_dir()
        assert d == Path(os.environ["HOME"]) / ".cache" / "tides"

    def test_cache_dir_is_isolated_from_real_home(self, tmp_path):
        # Guard for the conftest isolation fixture itself.
        assert get_cache_dir().is_relative_to(tmp_path)

    def test_xdg_cache_home(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_cache_dir()
            assert d == tmp_path / "tides"

    def test_cache_dir_is_created(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_cache_dir()
            assert d.exists()


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

    def test_fes2022_missing_prints_instructions(self, capsys):
        with patch("tides.cache._model_exists", return_value=False):
            with patch("tides.cache._get_pytmd_data_dir", return_value=Path("/fake/pytmd")):
                with pytest.raises(SystemExit) as exc_info:
                    ensure_model_data("FES2022")
                assert exc_info.value.code == 2
                captured = capsys.readouterr()
                assert "FES2022" in captured.err
                assert "manually" in captured.err
                assert "AVISO" in captured.err

    def test_hamtide11_missing_prints_instructions(self, capsys):
        with patch("tides.cache._model_exists", return_value=False):
            with patch("tides.cache._get_pytmd_data_dir", return_value=Path("/fake/pytmd")):
                with pytest.raises(SystemExit) as exc_info:
                    ensure_model_data("HAMTIDE11")
                assert exc_info.value.code == 2
                captured = capsys.readouterr()
                assert "HAMTIDE11" in captured.err
                assert "manually" in captured.err


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
    def _run(self, index_exists: bool, model_exists: bool, tmp_path, content: str = "[]"):
        index = tmp_path / "station_index.json"
        if index_exists:
            index.write_text(content)
        with (
            patch("tides.cache.fetch_station_data", return_value=[{"id": "1"}]) as fetch,
            patch("tides.stations._get_index_path", return_value=index),
            patch("tides.stations.get_station_index") as get_index,
            patch("tides.cache._model_exists", return_value=model_exists),
            patch("tides.cache.ensure_model_data") as ensure,
        ):
            fetch_all()
        return fetch, get_index, ensure

    def test_fetches_all_three_when_missing(self, tmp_path, capsys):
        fetch, get_index, ensure = self._run(False, False, tmp_path)
        fetch.assert_called_once()
        get_index.assert_called_once()
        ensure.assert_called_once_with("GOT5.6")
        err = capsys.readouterr().err
        assert "NOAA station list: updated (1 stations)" in err
        assert "Done" in err

    def test_already_present_items_are_skipped(self, tmp_path, capsys):
        fetch, get_index, ensure = self._run(True, True, tmp_path)
        fetch.assert_called_once()  # NOAA list always refreshed
        get_index.assert_called_once()  # validates; no download for a valid index
        ensure.assert_not_called()
        err = capsys.readouterr().err
        assert "Station database: already present" in err
        assert "GOT5.6 model: already present" in err
        assert "Downloading" not in err

    @pytest.mark.parametrize("content", ["{truncated", "{}", '[{"id": "1"}]'])
    def test_invalid_index_is_repaired_not_reported_present(self, tmp_path, capsys, content):
        """Real get_station_index(): an invalid index (corrupt, wrong shape or
        missing keys) triggers repair; only the download itself is mocked."""
        stations_dir = tmp_path / "stations"
        stations_dir.mkdir()
        index = stations_dir / "station_index.json"
        index.write_text(content)

        def fake_download():
            index.write_text(
                json.dumps(
                    [{"id": "9", "name": "x", "lat": 1.0, "lon": 2.0, "file": "noaa/9.json"}]
                )
            )

        with (
            patch("tides.cache.fetch_station_data", return_value=[]),
            patch("tides.stations._get_stations_dir", return_value=stations_dir),
            patch("tides.stations.download_station_database", side_effect=fake_download) as dl,
            patch("tides.cache._model_exists", return_value=True),
        ):
            fetch_all()
        dl.assert_called_once()
        assert json.loads(index.read_text())[0]["id"] == "9"
        assert "Station database: already present" not in capsys.readouterr().err

    def test_invalid_index_rebuilt_from_disk_without_download(self, tmp_path, capsys):
        from tides.stations import get_station_index

        stations_dir = tmp_path / "stations"
        (stations_dir / "noaa").mkdir(parents=True)
        (stations_dir / "noaa" / "5.json").write_text(
            json.dumps({"name": "S", "latitude": 1.0, "longitude": 2.0})
        )
        (stations_dir / "station_index.json").write_text("{}")
        with (
            patch("tides.stations._get_stations_dir", return_value=stations_dir),
            patch("tides.stations.download_station_database") as dl,
        ):
            index = get_station_index()
        dl.assert_not_called()
        assert index[0]["id"] == "5"
        assert "index rebuilt from 1 cached station files" in capsys.readouterr().err


class TestStaleStationCache:
    def test_stale_cache_used_when_refresh_fails(self, capsys):
        stale = [{"id": "old"}]
        with (
            patch("tides.cache.is_station_cache_fresh", return_value=False),
            patch("tides.cache.fetch_station_data", side_effect=httpx.ConnectError("offline")),
            patch("tides.cache.load_station_cache", return_value=stale),
            patch("tides.cache._station_cache_age_days", return_value=45),
        ):
            assert get_stations() == stale
        assert "using station list cached 45 days ago" in capsys.readouterr().err

    def test_no_cache_and_offline_raises(self):
        with (
            patch("tides.cache.is_station_cache_fresh", return_value=False),
            patch("tides.cache.fetch_station_data", side_effect=httpx.ConnectError("offline")),
            patch("tides.cache.load_station_cache", return_value=None),
            pytest.raises(httpx.ConnectError),
        ):
            get_stations()


class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert format_size(3 * 1024 * 1024 * 1024) == "3.0 GB"

    def test_zero(self):
        assert format_size(0) == "0 B"


class TestDirSize:
    def test_empty_dir(self, tmp_path):
        assert _dir_size(tmp_path) == 0

    def test_nonexistent_dir(self, tmp_path):
        assert _dir_size(tmp_path / "nope") == 0

    def test_with_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world!")
        assert _dir_size(tmp_path) == 11

    def test_nested_files(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_bytes(b"x" * 100)
        assert _dir_size(tmp_path) == 100


class TestGetCacheInfo:
    def test_returns_structure(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("tides.cache._get_pytmd_data_dir", return_value=tmp_path / "pytmd"):
                info = get_cache_info()
                assert "app_cache" in info
                assert "model_cache" in info
                assert "path" in info["app_cache"]
                assert "items" in info["app_cache"]

    def test_detects_noaa_stations(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            # Create station cache file
            cache_dir = tmp_path / "tides"
            cache_dir.mkdir(parents=True)
            (cache_dir / "noaa_stations.json").write_text("[]")
            with patch("tides.cache._get_pytmd_data_dir", return_value=tmp_path / "pytmd"):
                info = get_cache_info()
                names = [i["name"] for i in info["app_cache"]["items"]]
                assert "NOAA station list" in names

    def test_detects_model_dirs(self, tmp_path):
        pytmd_dir = tmp_path / "pytmd"
        # Create a GOT5.6 dir with a file
        got_dir = pytmd_dir / "GOT5.6"
        got_dir.mkdir(parents=True)
        (got_dir / "m2.nc").write_bytes(b"x" * 1000)
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("tides.cache._get_pytmd_data_dir", return_value=pytmd_dir):
                info = get_cache_info()
                names = [i["name"] for i in info["model_cache"]["items"]]
                assert "GOT5.6" in names
                got_item = next(i for i in info["model_cache"]["items"] if i["name"] == "GOT5.6")
                assert got_item["size"] == 1000

    def test_detects_station_database(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache_dir = tmp_path / "tides"
            stations_dir = cache_dir / "stations" / "noaa"
            stations_dir.mkdir(parents=True)
            (stations_dir / "1234.json").write_text("{}")
            with patch("tides.cache._get_pytmd_data_dir", return_value=tmp_path / "pytmd"):
                info = get_cache_info()
                names = [i["name"] for i in info["app_cache"]["items"]]
                assert "Station database" in names


class TestClearCache:
    def test_clear_specific_model(self, tmp_path):
        pytmd_dir = tmp_path / "pytmd"
        got_dir = pytmd_dir / "GOT5.6"
        got_dir.mkdir(parents=True)
        (got_dir / "m2.nc").write_bytes(b"x" * 500)
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            with patch("tides.cache._get_pytmd_data_dir", return_value=pytmd_dir):
                freed = clear_cache("got5.6")
                assert freed == 500
                assert not got_dir.exists()

    def test_clear_stations(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache_dir = tmp_path / "tides"
            stations_dir = cache_dir / "stations"
            stations_dir.mkdir(parents=True)
            (stations_dir / "test.json").write_text("{}")
            (cache_dir / "noaa_stations.json").write_text("[]")
            freed = clear_cache("stations")
            assert freed > 0
            assert not stations_dir.exists()
            assert not (cache_dir / "noaa_stations.json").exists()

    def test_clear_all(self, tmp_path):
        pytmd_dir = tmp_path / "pytmd"
        got_dir = pytmd_dir / "GOT5.6"
        got_dir.mkdir(parents=True)
        (got_dir / "m2.nc").write_bytes(b"x" * 100)
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            cache_dir = tmp_path / "tides"
            cache_dir.mkdir(parents=True)
            (cache_dir / "noaa_stations.json").write_text("[]")
            with patch("tides.cache._get_pytmd_data_dir", return_value=pytmd_dir):
                freed = clear_cache(None)
                assert freed > 0
                assert not cache_dir.exists()
                assert not got_dir.exists()

    def test_clear_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown cache name"):
            clear_cache("bogus")


class TestSafeClear:
    """#17: default clear never touches EOT20 / FES2022 / HAMTIDE11."""

    @pytest.fixture
    def caches(self, tmp_path, monkeypatch):
        pytmd = tmp_path / "pytmd"
        for d in ("GOT5.5", "GOT5.6", "EOT20", "fes2022b", "hamtide"):
            (pytmd / d).mkdir(parents=True)
            (pytmd / d / "f.nc").write_bytes(b"x" * 10)
        app = tmp_path / "cache" / "tides"
        (app / "datums").mkdir(parents=True)
        (app / "datums" / "got5.6.v3.json").write_text("{}")
        (app / "noaa_stations.json").write_text("[]")
        monkeypatch.setenv("TIDES_PYTMD_DIR", str(pytmd))
        return app, pytmd

    def test_default_keeps_manual_and_large_models(self, caches):
        app, pytmd = caches
        clear_cache(None)
        assert not app.exists()
        assert not (pytmd / "GOT5.6").exists() and not (pytmd / "GOT5.5").exists()
        for kept in ("EOT20", "fes2022b", "hamtide"):
            assert (pytmd / kept / "f.nc").exists(), kept

    def test_plan_matches_default_clear(self, caches):
        names = [i["name"] for i in plan_clear(None)]
        assert names == ["App cache", "GOT5.5", "GOT5.6"]

    def test_all_includes_every_model(self, caches):
        _, pytmd = caches
        names = {i["name"] for i in plan_clear(None, include_all=True)}
        assert {"EOT20", "FES2022", "HAMTIDE11"} <= names
        clear_cache(None, include_all=True)
        assert not any((pytmd / d).exists() for d in ("EOT20", "fes2022b", "hamtide"))

    def test_named_manual_model_clears_only_it(self, caches):
        _, pytmd = caches
        clear_cache("fes2022")
        assert not (pytmd / "fes2022b").exists()
        assert (pytmd / "EOT20").exists()

    def test_clear_datums(self, caches):
        app, _ = caches
        assert [i["name"] for i in plan_clear("datums")] == ["Datum cache"]
        clear_cache("datums")
        assert not (app / "datums").exists()
        assert (app / "noaa_stations.json").exists()

    def test_cache_info_lists_datums(self, caches):
        names = [i["name"] for i in get_cache_info()["app_cache"]["items"]]
        assert "Datum cache" in names

    def test_os_error_is_not_swallowed(self, caches, monkeypatch):
        def boom(*a, **k):
            raise OSError("Permission denied")

        monkeypatch.setattr("tides.cache.shutil.rmtree", boom)
        with pytest.raises(OSError):
            clear_cache(None)


class TestAtomicWrite:
    def test_replaces_content_and_leaves_no_temp(self, tmp_path):
        d = tmp_path / "w"
        d.mkdir()
        f = d / "x.json"
        f.write_text("old")
        atomic_write_text(f, "new")
        assert f.read_text() == "new"
        assert [p.name for p in d.iterdir()] == ["x.json"]

    def test_failed_replace_keeps_old_file(self, tmp_path, monkeypatch):
        d = tmp_path / "w"
        d.mkdir()
        f = d / "x.json"
        f.write_text("old")

        def fail(*a, **k):
            raise OSError("crash")

        monkeypatch.setattr("tides.cache.os.replace", fail)
        with pytest.raises(OSError):
            atomic_write_text(f, "new")
        assert f.read_text() == "old"
        assert [p.name for p in d.iterdir()] == ["x.json"]


class TestEot20Marker:
    def test_partial_install_is_removed_before_redownload(self, tmp_path, monkeypatch):
        from tides import cache

        pytmd = tmp_path / "pytmd"
        partial = pytmd / "EOT20" / "ocean_tides"
        partial.mkdir(parents=True)
        (partial / "M2.nc").write_bytes(b"x")
        other = pytmd / "GOT5.6"
        other.mkdir(parents=True)
        monkeypatch.setattr("tides.cache._get_pytmd_data_dir", lambda: pytmd)
        with patch("httpx.stream", side_effect=RuntimeError("stop before download")):
            with pytest.raises(RuntimeError):
                cache._fetch_eot20()
        assert not (pytmd / "EOT20").exists()  # partial removed
        assert other.exists()  # nothing else touched

    def test_complete_install_is_left_alone(self, tmp_path, monkeypatch):
        from tides import cache

        pytmd = tmp_path / "pytmd"
        (pytmd / "EOT20").mkdir(parents=True)
        (pytmd / "EOT20" / cache.EOT20_COMPLETE_MARKER).write_text("ok")
        monkeypatch.setattr("tides.cache._get_pytmd_data_dir", lambda: pytmd)
        with patch("httpx.stream") as stream:
            cache._fetch_eot20()
        stream.assert_not_called()

    def test_existing_install_gets_marker(self, tmp_path, monkeypatch):
        from tides import cache

        pytmd = tmp_path / "pytmd"
        (pytmd / "EOT20").mkdir(parents=True)
        monkeypatch.setattr("tides.cache._get_pytmd_data_dir", lambda: pytmd)
        with patch("tides.cache._model_exists", return_value=True):
            cache.ensure_model_data("EOT20")
        assert (pytmd / "EOT20" / cache.EOT20_COMPLETE_MARKER).exists()

    def test_listing_override_never_redirects_model_fetch(self, tmp_path, monkeypatch):
        from tides import cache

        monkeypatch.setenv("TIDES_PYTMD_DIR", str(tmp_path / "sandbox"))
        real = tmp_path / "real"
        monkeypatch.setattr("tides.cache._get_pytmd_data_dir", lambda: real)
        assert cache._get_listing_pytmd_dir() == tmp_path / "sandbox"
        (real / "EOT20").mkdir(parents=True)
        (real / "EOT20" / cache.EOT20_COMPLETE_MARKER).write_text("ok")
        with patch("httpx.stream") as stream:
            cache._fetch_eot20()  # looks in the real dir, not the override
        stream.assert_not_called()


class TestNetworkGuard:
    """The conftest autouse guard blocks inet sockets but not AF_UNIX."""

    def test_inet_connect_blocked(self):
        import socket

        from conftest import NetworkBlockedError

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            with pytest.raises(NetworkBlockedError):
                s.connect(("127.0.0.1", 9))
            with pytest.raises(NetworkBlockedError):
                s.connect_ex(("127.0.0.1", 9))

    @pytest.mark.skipif(not hasattr(__import__("socket"), "AF_UNIX"), reason="no AF_UNIX")
    def test_unix_connect_ex_returns_errno(self):
        import socket

        # Short absolute path: AF_UNIX paths are limited to ~104 bytes on macOS,
        # which tmp_path can exceed.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            rc = s.connect_ex("/nonexistent-tides-test.sock")
        assert isinstance(rc, int) and rc != 0
