import datetime
from unittest.mock import patch

import pytest

from tides.models import Coordinate, Source, TideEvent
from tides.resolver import _group_events_by_date, resolve_tides

SAMPLE_STATIONS = [
    {"id": "8518750", "name": "The Battery", "lat": 40.7006, "lon": -74.0142},
]

SAMPLE_NOAA_EVENTS = [
    TideEvent(
        time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
        height=0.3,
    ),
    TideEvent(
        time=datetime.datetime(2026, 4, 15, 20, 45, tzinfo=datetime.timezone.utc),
        height=-0.1,
    ),
]


class TestResolveNoaa:
    @patch("tides.resolver.fetch_predictions")
    @patch("tides.resolver.parse_predictions_response")
    @patch("tides.resolver.get_stations")
    def test_auto_selects_noaa_when_near_station(self, mock_get_stations, mock_parse, mock_fetch):
        mock_get_stations.return_value = SAMPLE_STATIONS
        mock_fetch.return_value = {"predictions": []}
        mock_parse.return_value = SAMPLE_NOAA_EVENTS

        coord = Coordinate(lat=40.7128, lon=-74.0060)
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.AUTO,
        )

        assert result.source_type == Source.NOAA
        assert result.station_name == "The Battery"
        assert len(result.days) == 1
        assert len(result.days[0].events) == 2

    @patch("tides.resolver.compute_tides")
    @patch("tides.resolver.get_stations")
    def test_auto_falls_back_to_model(self, mock_get_stations, mock_compute):
        mock_get_stations.return_value = SAMPLE_STATIONS
        mock_compute.return_value = SAMPLE_NOAA_EVENTS

        coord = Coordinate(lat=-5.0, lon=-35.0)
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.AUTO,
        )

        assert result.source_type == Source.MODEL
        assert result.model_name == "GOT5.6"

    @patch("tides.resolver.get_stations")
    def test_noaa_source_errors_when_no_station(self, mock_get_stations):
        mock_get_stations.return_value = SAMPLE_STATIONS

        coord = Coordinate(lat=-5.0, lon=-35.0)
        with pytest.raises(SystemExit):
            resolve_tides(
                coord,
                begin_date=datetime.date(2026, 4, 15),
                end_date=datetime.date(2026, 4, 15),
                source=Source.NOAA,
            )


# ---------------------------------------------------------------------------
# NEW TEST CLASSES
# ---------------------------------------------------------------------------


class TestGroupEventsByDate:
    def test_single_day(self):
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 10, 0, tzinfo=datetime.timezone.utc),
                height=0.5,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 16, 0, tzinfo=datetime.timezone.utc),
                height=-0.3,
            ),
        ]
        days = _group_events_by_date(
            events,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
        )
        assert len(days) == 1
        assert days[0].date == datetime.date(2026, 4, 15)
        assert len(days[0].events) == 2

    def test_multi_day(self):
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 10, 0, tzinfo=datetime.timezone.utc),
                height=0.5,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 16, 11, 0, tzinfo=datetime.timezone.utc),
                height=0.8,
            ),
        ]
        days = _group_events_by_date(
            events,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 16),
        )
        assert len(days) == 2
        assert days[0].date == datetime.date(2026, 4, 15)
        assert days[1].date == datetime.date(2026, 4, 16)
        assert len(days[0].events) == 1
        assert len(days[1].events) == 1

    def test_empty_events(self):
        days = _group_events_by_date(
            events=[],
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 16),
        )
        assert len(days) == 2
        assert days[0].events == []
        assert days[1].events == []

    def test_events_outside_range(self):
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 14, 10, 0, tzinfo=datetime.timezone.utc),
                height=0.5,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 12, 0, tzinfo=datetime.timezone.utc),
                height=0.3,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 17, 8, 0, tzinfo=datetime.timezone.utc),
                height=0.7,
            ),
        ]
        days = _group_events_by_date(
            events,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
        )
        assert len(days) == 1
        assert len(days[0].events) == 1
        assert days[0].events[0].height == 0.3

    def test_events_sorted_within_day(self):
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 20, 0, tzinfo=datetime.timezone.utc),
                height=-0.1,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 8, 0, tzinfo=datetime.timezone.utc),
                height=0.9,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 0, tzinfo=datetime.timezone.utc),
                height=0.4,
            ),
        ]
        days = _group_events_by_date(
            events,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
        )
        times = [e.time.hour for e in days[0].events]
        assert times == [8, 14, 20]


class TestResolveModel:
    @patch("tides.resolver.compute_tides")
    def test_model_source_returns_model_result(self, mock_compute):
        mock_compute.return_value = SAMPLE_NOAA_EVENTS

        coord = Coordinate(lat=-5.0, lon=-35.0)
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.MODEL,
        )

        assert result.source_type == Source.MODEL
        assert result.model_name == "GOT5.6"
        assert result.station_id is None
        assert result.station_name is None
        assert len(result.days) == 1

    @patch("tides.resolver.compute_tides")
    def test_model_source_empty_events_exits(self, mock_compute):
        mock_compute.return_value = []

        coord = Coordinate(lat=-5.0, lon=-35.0)
        with pytest.raises(SystemExit):
            resolve_tides(
                coord,
                begin_date=datetime.date(2026, 4, 15),
                end_date=datetime.date(2026, 4, 15),
                source=Source.MODEL,
            )

    @patch("tides.resolver.compute_tides")
    def test_model_source_exit_code_is_2(self, mock_compute):
        mock_compute.return_value = []

        coord = Coordinate(lat=-5.0, lon=-35.0)
        with pytest.raises(SystemExit) as exc_info:
            resolve_tides(
                coord,
                begin_date=datetime.date(2026, 4, 15),
                end_date=datetime.date(2026, 4, 15),
                source=Source.MODEL,
            )
        assert exc_info.value.code == 2


class TestResolveExplicitNoaaHappyPath:
    @patch("tides.resolver.fetch_predictions")
    @patch("tides.resolver.parse_predictions_response")
    @patch("tides.resolver.get_stations")
    def test_noaa_source_happy_path(self, mock_get_stations, mock_parse, mock_fetch):
        mock_get_stations.return_value = SAMPLE_STATIONS
        mock_fetch.return_value = {"predictions": []}
        mock_parse.return_value = SAMPLE_NOAA_EVENTS

        coord = Coordinate(lat=40.7128, lon=-74.0060)
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.NOAA,
        )

        assert result.source_type == Source.NOAA
        assert result.station_id == "8518750"
        assert result.station_name == "The Battery"
        assert result.station_distance_km is not None
        assert result.station_distance_km > 0
        assert result.model_name is None
        assert len(result.days) == 1
        assert len(result.days[0].events) == 2


class TestResolveExplicitModelPath:
    @patch("tides.resolver.compute_tides")
    def test_explicit_model_source(self, mock_compute):
        mock_compute.return_value = SAMPLE_NOAA_EVENTS

        coord = Coordinate(lat=35.0, lon=-75.0)
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.MODEL,
        )

        assert result.source_type == Source.MODEL
        assert result.model_name == "GOT5.6"
        assert result.station_id is None
        assert result.station_name is None
        assert result.station_distance_km is None
        assert result.coordinate == coord
        assert len(result.days) == 1
