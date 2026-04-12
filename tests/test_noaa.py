import pytest

from tides.models import Coordinate
from tides.noaa import (
    find_nearest_station,
    parse_predictions_response,
    parse_station_list,
)

SAMPLE_STATION_LIST_XML = """<?xml version="1.0" encoding="utf-8" ?>
<Stations>
  <count>2</count>
  <Station>
    <id>8518750</id>
    <name>The Battery</name>
    <lat>40.7006</lat>
    <lng>-74.0142</lng>
  </Station>
  <Station>
    <id>8534720</id>
    <name>Atlantic City</name>
    <lat>39.3553</lat>
    <lng>-74.4181</lng>
  </Station>
</Stations>"""

SAMPLE_PREDICTIONS_JSON = {
    "predictions": [
        {"t": "2026-04-15 02:18", "v": "1.524", "type": "H"},
        {"t": "2026-04-15 08:42", "v": "0.012", "type": "L"},
        {"t": "2026-04-15 14:36", "v": "1.311", "type": "H"},
        {"t": "2026-04-15 20:54", "v": "-0.067", "type": "L"},
    ]
}


class TestParseStationList:
    def test_parse_stations(self):
        stations = parse_station_list(SAMPLE_STATION_LIST_XML)
        assert len(stations) == 2
        assert stations[0]["id"] == "8518750"
        assert stations[0]["name"] == "The Battery"
        assert abs(stations[0]["lat"] - 40.7006) < 0.001
        assert abs(stations[0]["lon"] - (-74.0142)) < 0.001


class TestFindNearestStation:
    def test_find_nearest(self):
        stations = [
            {"id": "8518750", "name": "The Battery", "lat": 40.7006, "lon": -74.0142},
            {"id": "8534720", "name": "Atlantic City", "lat": 39.3553, "lon": -74.4181},
        ]
        coord = Coordinate(lat=40.7128, lon=-74.0060)
        station, distance = find_nearest_station(stations, coord)
        assert station["id"] == "8518750"
        assert distance < 2.0

    def test_none_when_too_far(self):
        stations = [
            {"id": "8518750", "name": "The Battery", "lat": 40.7006, "lon": -74.0142},
        ]
        coord = Coordinate(lat=-5.0, lon=-35.0)
        result = find_nearest_station(stations, coord, max_distance_km=25.0)
        assert result is None


class TestParsePredictionsResponse:
    def test_parse_predictions(self):
        events = parse_predictions_response(SAMPLE_PREDICTIONS_JSON)
        assert len(events) == 4
        assert events[0].height == pytest.approx(1.524)
        assert events[0].time.hour == 2
        assert events[0].time.minute == 18
        assert events[1].height == pytest.approx(0.012)
        assert events[3].height == pytest.approx(-0.067)
