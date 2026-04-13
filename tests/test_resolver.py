import datetime
from unittest.mock import patch

import pytest

from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.resolver import _apply_datum, _group_events_by_date, resolve_tides

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
            datum="msl",
        )

        assert result.source_type == Source.NOAA
        assert result.station_name == "The Battery"
        assert len(result.days) == 1
        assert len(result.days[0].events) == 2

    @patch("tides.resolver._resolve_station", return_value=None)
    @patch("tides.resolver.compute_tides")
    @patch("tides.resolver.get_stations")
    def test_auto_falls_back_to_model(self, mock_get_stations, mock_compute, mock_station):
        mock_get_stations.return_value = SAMPLE_STATIONS
        mock_compute.return_value = SAMPLE_NOAA_EVENTS

        coord = Coordinate(lat=-5.0, lon=-35.0)
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.AUTO,
            datum="msl",
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
                datum="msl",
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
            datum="msl",
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
                datum="msl",
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
                datum="msl",
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
            datum="msl",
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
            datum="msl",
        )

        assert result.source_type == Source.MODEL
        assert result.model_name == "GOT5.6"
        assert result.station_id is None
        assert result.station_name is None
        assert result.station_distance_km is None
        assert result.coordinate == coord
        assert len(result.days) == 1


class TestResolveStation:
    @patch("tides.resolver._resolve_station")
    def test_station_source_happy_path(self, mock_resolve_station):
        mock_resolve_station.return_value = (
            TideResult(
                coordinate=Coordinate(lat=-3.7, lon=-38.5),
                source_type=Source.STATION,
                station_id="fortaleza",
                station_name="Fortaleza",
                station_distance_km=5.0,
                model_name=None,
                days=[TideDay(date=datetime.date(2026, 4, 15), events=SAMPLE_NOAA_EVENTS)],
            ),
            {"chart_datum": "LAT", "datums": {"MSL": 0.0, "LAT": -1.6}},
        )
        coord = Coordinate(lat=-3.7, lon=-38.5)
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.STATION,
            datum="msl",
        )
        assert result.source_type == Source.STATION
        assert result.station_name == "Fortaleza"

    @patch("tides.resolver._resolve_station")
    def test_station_source_not_found(self, mock_resolve_station):
        mock_resolve_station.return_value = None
        coord = Coordinate(lat=0.0, lon=0.0)
        with pytest.raises(SystemExit) as exc_info:
            resolve_tides(
                coord,
                begin_date=datetime.date(2026, 4, 15),
                end_date=datetime.date(2026, 4, 15),
                source=Source.STATION,
                datum="msl",
            )
        assert exc_info.value.code == 1


class TestApplyDatum:
    def test_msl_is_noop(self):
        result = TideResult(
            coordinate=Coordinate(lat=40.7, lon=-74.0),
            source_type=Source.MODEL,
            station_id=None,
            station_name=None,
            station_distance_km=None,
            model_name="GOT5.6",
            days=[
                TideDay(
                    date=datetime.date(2026, 4, 15),
                    events=[
                        TideEvent(
                            time=datetime.datetime(
                                2026, 4, 15, 12, 0, tzinfo=datetime.timezone.utc
                            ),
                            height=0.5,
                        )
                    ],
                )
            ],
        )
        out = _apply_datum(result, "msl", "GOT5.6")
        assert out.days[0].events[0].height == 0.5
        assert out.datum == "msl"

    @patch("tides.datums.get_model_datums")
    def test_mllw_shifts_heights(self, mock_datums):
        mock_datums.return_value = {"mllw": -0.5, "msl": 0.0}
        result = TideResult(
            coordinate=Coordinate(lat=40.7, lon=-74.0),
            source_type=Source.MODEL,
            station_id=None,
            station_name=None,
            station_distance_km=None,
            model_name="GOT5.6",
            days=[
                TideDay(
                    date=datetime.date(2026, 4, 15),
                    events=[
                        TideEvent(
                            time=datetime.datetime(
                                2026, 4, 15, 12, 0, tzinfo=datetime.timezone.utc
                            ),
                            height=0.5,
                        )
                    ],
                )
            ],
        )
        out = _apply_datum(result, "mllw", "GOT5.6")
        # height_mllw = 0.5 - (-0.5) = 1.0
        assert abs(out.days[0].events[0].height - 1.0) < 0.001
        assert out.datum == "mllw"

    def test_uses_station_datums_when_available(self):
        """Station heights are relative to chart_datum. Converting to another datum
        should use the station's published datum offsets."""
        station_data = {
            "chart_datum": "LAT",
            "datums": {"MSL": 0.0, "LAT": -1.6, "MLLW": -0.9},
        }
        result = TideResult(
            coordinate=Coordinate(lat=40.7, lon=-74.0),
            source_type=Source.STATION,
            station_id="test",
            station_name="Test",
            station_distance_km=1.0,
            model_name=None,
            days=[
                TideDay(
                    date=datetime.date(2026, 4, 15),
                    events=[
                        TideEvent(
                            time=datetime.datetime(
                                2026, 4, 15, 12, 0, tzinfo=datetime.timezone.utc
                            ),
                            height=2.5,  # 2.5m relative to LAT
                        )
                    ],
                )
            ],
        )
        out = _apply_datum(result, "mllw", "GOT5.6", station=station_data)
        # LAT=-1.6, MLLW=-0.9, shift = MLLW - LAT = -0.9 - (-1.6) = 0.7
        # height_mllw = 2.5 - 0.7 = 1.8
        assert abs(out.days[0].events[0].height - 1.8) < 0.001
        assert out.datum == "mllw"


class TestAutoResolution:
    @patch("tides.resolver._resolve_station")
    @patch("tides.resolver._resolve_noaa")
    @patch("tides.resolver.get_stations")
    def test_auto_falls_through_to_station(self, mock_get, mock_noaa, mock_station):
        """When NOAA fails, auto should try station before model."""
        mock_get.return_value = []
        mock_noaa.return_value = None
        mock_station.return_value = (
            TideResult(
                coordinate=Coordinate(lat=-3.7, lon=-38.5),
                source_type=Source.STATION,
                station_id="fort",
                station_name="Fortaleza",
                station_distance_km=5.0,
                model_name=None,
                days=[TideDay(date=datetime.date(2026, 4, 15), events=SAMPLE_NOAA_EVENTS)],
            ),
            {"chart_datum": "LAT", "datums": {"MSL": 0.0, "LAT": -1.6}},
        )
        result = resolve_tides(
            Coordinate(lat=-3.7, lon=-38.5),
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            datum="msl",
        )
        assert result.source_type == Source.STATION
        mock_station.assert_called_once()
