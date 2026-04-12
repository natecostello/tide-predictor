import datetime

import pytest

from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult


class TestCoordinate:
    def test_valid_coordinate(self):
        c = Coordinate(lat=40.7128, lon=-74.0060)
        assert c.lat == 40.7128
        assert c.lon == -74.0060

    def test_invalid_latitude_too_high(self):
        with pytest.raises(ValueError, match="Latitude"):
            Coordinate(lat=91.0, lon=0.0)

    def test_invalid_latitude_too_low(self):
        with pytest.raises(ValueError, match="Latitude"):
            Coordinate(lat=-91.0, lon=0.0)

    def test_invalid_longitude_too_high(self):
        with pytest.raises(ValueError, match="Longitude"):
            Coordinate(lat=0.0, lon=181.0)

    def test_invalid_longitude_too_low(self):
        with pytest.raises(ValueError, match="Longitude"):
            Coordinate(lat=0.0, lon=-181.0)

    def test_boundary_values(self):
        c = Coordinate(lat=90.0, lon=180.0)
        assert c.lat == 90.0
        assert c.lon == 180.0

    def test_boundary_negative(self):
        c = Coordinate(lat=-90.0, lon=-180.0)
        assert c.lat == -90.0
        assert c.lon == -180.0


class TestTideEvent:
    def test_tide_event_creation(self):
        t = TideEvent(
            time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
            height=0.3,
        )
        assert t.height == 0.3

    def test_height_in_feet(self):
        t = TideEvent(
            time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
            height=0.3,
        )
        assert abs(t.height_ft - 0.984252) < 0.001


class TestTideDay:
    def test_tide_day(self):
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
                height=0.3,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 20, 45, tzinfo=datetime.timezone.utc),
                height=-0.1,
            ),
        ]
        day = TideDay(date=datetime.date(2026, 4, 15), events=events)
        assert len(day.events) == 2
        assert day.date == datetime.date(2026, 4, 15)


class TestSource:
    def test_source_enum(self):
        assert Source.AUTO.value == "auto"
        assert Source.NOAA.value == "noaa"
        assert Source.MODEL.value == "model"


class TestTideResult:
    def test_tide_result_noaa(self):
        result = TideResult(
            coordinate=Coordinate(lat=40.7128, lon=-74.0060),
            source_type=Source.NOAA,
            station_id="8518750",
            station_name="The Battery",
            station_distance_km=1.2,
            model_name=None,
            days=[],
        )
        assert result.source_type == Source.NOAA
        assert result.station_name == "The Battery"

    def test_tide_result_model(self):
        result = TideResult(
            coordinate=Coordinate(lat=-5.0, lon=-35.0),
            source_type=Source.MODEL,
            station_id=None,
            station_name=None,
            station_distance_km=None,
            model_name="GOT5.6",
            days=[],
        )
        assert result.source_type == Source.MODEL
        assert result.model_name == "GOT5.6"
