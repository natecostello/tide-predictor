import datetime

from tides.models import Coordinate
from tides.timezone import get_timezone_name, to_local_time


class TestGetTimezoneName:
    def test_new_york(self):
        coord = Coordinate(lat=40.7128, lon=-74.0060)
        assert get_timezone_name(coord) == "America/New_York"

    def test_brazil(self):
        # Recife city center (on land, clearly in America/Recife)
        coord = Coordinate(lat=-8.05, lon=-34.87)
        assert get_timezone_name(coord) == "America/Recife"

    def test_london(self):
        coord = Coordinate(lat=51.5074, lon=-0.1278)
        assert get_timezone_name(coord) == "Europe/London"

    def test_ocean_returns_utc_offset_timezone(self):
        # Middle of the Pacific — timezonefinder returns Etc/GMT+X for ocean
        coord = Coordinate(lat=0.0, lon=-160.0)
        result = get_timezone_name(coord)
        assert result is not None
        assert result.startswith("Etc/GMT")


class TestToLocalTime:
    def test_utc_to_new_york(self):
        coord = Coordinate(lat=40.7128, lon=-74.0060)
        utc_time = datetime.datetime(2026, 4, 15, 18, 32, tzinfo=datetime.timezone.utc)
        local = to_local_time(utc_time, coord)
        # EDT is UTC-4 in April
        assert local.hour == 14
        assert local.minute == 32

    def test_utc_to_brazil(self):
        coord = Coordinate(lat=-8.05, lon=-34.87)
        utc_time = datetime.datetime(2026, 4, 15, 18, 0, tzinfo=datetime.timezone.utc)
        local = to_local_time(utc_time, coord)
        # Recife is UTC-3, no DST
        assert local.hour == 15
