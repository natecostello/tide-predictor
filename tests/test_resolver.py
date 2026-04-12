import datetime
from unittest.mock import patch

import pytest

from tides.models import Coordinate, Source, TideEvent
from tides.resolver import resolve_tides

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
