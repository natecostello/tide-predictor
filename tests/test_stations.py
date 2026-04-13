"""Tests for global tide station database."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

from tides.models import Coordinate, TideEvent
from tides.stations import (
    build_station_index,
    find_nearest_station,
    get_station_index,
    load_station,
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

    def test_empty_constituents_returns_empty(self):
        station = {**SAMPLE_STATION, "harmonic_constituents": []}
        events = predict_station_tides(
            station, datetime.date(2026, 4, 15), datetime.date(2026, 4, 15)
        )
        assert events == []


class TestBuildStationIndex:
    def test_builds_index_from_json_files(self, tmp_path):
        noaa_dir = tmp_path / "noaa"
        noaa_dir.mkdir()
        station = {"name": "Test Station", "latitude": 40.7, "longitude": -74.0}
        (noaa_dir / "12345.json").write_text(json.dumps(station))

        index = build_station_index(tmp_path)
        assert len(index) == 1
        assert index[0]["id"] == "12345"
        assert index[0]["name"] == "Test Station"
        assert index[0]["lat"] == 40.7

    def test_skips_invalid_json(self, tmp_path):
        noaa_dir = tmp_path / "noaa"
        noaa_dir.mkdir()
        (noaa_dir / "bad.json").write_text("not json!")
        (noaa_dir / "good.json").write_text(
            json.dumps({"name": "Good", "latitude": 1.0, "longitude": 2.0})
        )
        index = build_station_index(tmp_path)
        assert len(index) == 1
        assert index[0]["name"] == "Good"

    def test_includes_ticon_and_noaa(self, tmp_path):
        for subdir in ["noaa", "ticon"]:
            d = tmp_path / subdir
            d.mkdir()
            (d / "s1.json").write_text(
                json.dumps({"name": f"{subdir}_station", "latitude": 0.0, "longitude": 0.0})
            )
        index = build_station_index(tmp_path)
        assert len(index) == 2
        sources = {e["source"] for e in index}
        assert sources == {"noaa", "ticon"}

    def test_empty_dir_returns_empty(self, tmp_path):
        assert build_station_index(tmp_path) == []


class TestLoadStation:
    def test_loads_station_from_entry(self, tmp_path):
        noaa_dir = tmp_path / "noaa"
        noaa_dir.mkdir()
        data = {"name": "Loaded", "latitude": 10.0, "longitude": 20.0}
        (noaa_dir / "99.json").write_text(json.dumps(data))

        with patch("tides.stations._get_stations_dir", return_value=tmp_path):
            result = load_station({"file": "noaa/99.json"})
            assert result["name"] == "Loaded"


class TestGetStationIndex:
    def test_returns_cached_index(self, tmp_path):
        with patch("tides.stations._get_stations_dir", return_value=tmp_path):
            index_data = [{"id": "1", "name": "Cached", "lat": 0, "lon": 0}]
            (tmp_path / "station_index.json").write_text(json.dumps(index_data))
            result = get_station_index()
            assert result == index_data

    def test_downloads_when_no_index(self, tmp_path):
        with patch("tides.stations._get_stations_dir", return_value=tmp_path):
            with patch("tides.stations.download_station_database") as mock_dl:
                index_data = [{"id": "2", "name": "Fresh"}]

                def fake_download():
                    (tmp_path / "station_index.json").write_text(json.dumps(index_data))

                mock_dl.side_effect = fake_download
                result = get_station_index()
                assert result == index_data
                mock_dl.assert_called_once()

    def test_redownloads_on_corrupt_index(self, tmp_path):
        with patch("tides.stations._get_stations_dir", return_value=tmp_path):
            (tmp_path / "station_index.json").write_text("corrupt!")
            with patch("tides.stations.download_station_database") as mock_dl:
                index_data = [{"id": "3", "name": "Redownloaded"}]

                def fake_download():
                    (tmp_path / "station_index.json").write_text(json.dumps(index_data))

                mock_dl.side_effect = fake_download
                result = get_station_index()
                assert result == index_data


class TestDownloadStationDatabase:
    def test_download_extracts_json_and_builds_index(self, tmp_path):
        """Mock the HTTP download and verify extraction + indexing."""
        import io
        import zipfile

        from tides.stations import download_station_database

        # Build a fake zip archive matching the expected structure
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            station = {"name": "MockStation", "latitude": 10.0, "longitude": 20.0}
            zf.writestr(
                "tide-database-main/data/noaa/9999.json",
                json.dumps(station),
            )
            # Non-JSON file should be skipped
            zf.writestr("tide-database-main/README.md", "ignored")
        buf.seek(0)

        mock_response = MagicMock()
        mock_response.iter_bytes.return_value = [buf.getvalue()]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tides.stations._get_stations_dir", return_value=tmp_path):
            with patch("tides.stations._get_index_path", return_value=tmp_path / "index.json"):
                with patch("httpx.stream", return_value=mock_response):
                    download_station_database()

        # Verify the JSON was extracted
        assert (tmp_path / "noaa" / "9999.json").exists()
        extracted = json.loads((tmp_path / "noaa" / "9999.json").read_text())
        assert extracted["name"] == "MockStation"

        # Verify index was built
        assert (tmp_path / "index.json").exists()
        index = json.loads((tmp_path / "index.json").read_text())
        assert len(index) == 1
        assert index[0]["id"] == "9999"

    def test_download_connection_error(self, tmp_path):
        import httpx

        from tides.stations import download_station_database

        with patch("tides.stations._get_stations_dir", return_value=tmp_path):
            with patch("httpx.stream", side_effect=httpx.ConnectError("fail")):
                with pytest.raises(SystemExit) as exc_info:
                    download_station_database()
                assert exc_info.value.code == 2
