import datetime
import json as json_module

import pytest

from tides.cli import format_json, format_plain, parse_between, parse_coordinate, parse_date_arg
from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult


class TestParseCoordinate:
    def test_comma_separated_no_space(self):
        c = parse_coordinate(["40.7128,-74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_comma_separated_with_space(self):
        c = parse_coordinate(["40.7128,", "-74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_comma_space_in_quotes(self):
        c = parse_coordinate(["40.7128, -74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_two_positional_args(self):
        c = parse_coordinate(["40.7128", "-74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_invalid_single_arg(self):
        with pytest.raises(SystemExit):
            parse_coordinate(["notacoord"])

    def test_empty_args(self):
        with pytest.raises(SystemExit):
            parse_coordinate([])

    def test_latitude_out_of_range(self):
        with pytest.raises(SystemExit):
            parse_coordinate(["95.0,-74.0060"])


class TestParseDateArg:
    def test_single_date(self):
        begin, end = parse_date_arg("2026-04-15")
        assert begin == datetime.date(2026, 4, 15)
        assert end == datetime.date(2026, 4, 15)

    def test_date_range(self):
        begin, end = parse_date_arg("2026-04-15:2026-04-17")
        assert begin == datetime.date(2026, 4, 15)
        assert end == datetime.date(2026, 4, 17)

    def test_none_defaults_to_today_utc(self):
        begin, end = parse_date_arg(None)
        today_utc = datetime.datetime.now(tz=datetime.timezone.utc).date()
        assert begin == today_utc
        assert end == today_utc

    def test_invalid_date(self):
        with pytest.raises(SystemExit):
            parse_date_arg("not-a-date")

    def test_end_before_begin(self):
        with pytest.raises(SystemExit):
            parse_date_arg("2026-04-17:2026-04-15")


class TestParseBetween:
    def test_valid_between(self):
        result = parse_between("06:00:18:00")
        assert result == (datetime.time(6, 0), datetime.time(18, 0))

    def test_none(self):
        assert parse_between(None) is None

    def test_invalid_format(self):
        with pytest.raises(SystemExit):
            parse_between("invalid")


class TestFormatPlain:
    def _make_result(self) -> TideResult:
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
        return TideResult(
            coordinate=Coordinate(lat=40.7128, lon=-74.0060),
            source_type=Source.NOAA,
            station_id="8518750",
            station_name="The Battery",
            station_distance_km=1.2,
            model_name=None,
            days=[TideDay(date=datetime.date(2026, 4, 15), events=events)],
        )

    def test_default_meters(self):
        result = self._make_result()
        output = format_plain(
            result, feet=False, precision=1, local=False, between=None, verbose=False
        )
        assert output == "0.3m@14:32, -0.1m@20:45"

    def test_feet(self):
        result = self._make_result()
        output = format_plain(
            result, feet=True, precision=1, local=False, between=None, verbose=False
        )
        assert "ft@" in output

    def test_verbose_noaa(self):
        result = self._make_result()
        output = format_plain(
            result, feet=False, precision=1, local=False, between=None, verbose=True
        )
        assert output.startswith("[NOAA: The Battery, 1.2km]")

    def test_precision(self):
        result = self._make_result()
        output = format_plain(
            result, feet=False, precision=3, local=False, between=None, verbose=False
        )
        assert "0.300m@14:32" in output

    def test_multi_day(self):
        events1 = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
                height=0.3,
            ),
        ]
        events2 = [
            TideEvent(
                time=datetime.datetime(2026, 4, 16, 15, 0, tzinfo=datetime.timezone.utc),
                height=0.5,
            ),
        ]
        result = TideResult(
            coordinate=Coordinate(lat=40.7128, lon=-74.0060),
            source_type=Source.NOAA,
            station_id="8518750",
            station_name="The Battery",
            station_distance_km=1.2,
            model_name=None,
            days=[
                TideDay(date=datetime.date(2026, 4, 15), events=events1),
                TideDay(date=datetime.date(2026, 4, 16), events=events2),
            ],
        )
        output = format_plain(
            result, feet=False, precision=1, local=False, between=None, verbose=False
        )
        assert "2026-04-15:" in output
        assert "2026-04-16:" in output


class TestFormatJson:
    def _make_result(self) -> TideResult:
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
                height=0.3,
            ),
        ]
        return TideResult(
            coordinate=Coordinate(lat=40.7128, lon=-74.0060),
            source_type=Source.NOAA,
            station_id="8518750",
            station_name="The Battery",
            station_distance_km=1.2,
            model_name=None,
            days=[TideDay(date=datetime.date(2026, 4, 15), events=events)],
        )

    def test_json_structure(self):
        result = self._make_result()
        output = format_json(result, feet=False, precision=1, local=False, between=None)
        parsed = json_module.loads(output)
        assert parsed["coordinate"] == {"lat": 40.7128, "lon": -74.006}
        assert parsed["source"]["type"] == "noaa"
        assert parsed["source"]["station"]["name"] == "The Battery"
        assert parsed["unit"] == "m"
        assert parsed["timezone"] == "UTC"
        assert len(parsed["days"]) == 1
        assert parsed["days"][0]["tides"][0]["height"] == 0.3

    def test_json_model_source(self):
        result = TideResult(
            coordinate=Coordinate(lat=-5.0, lon=-35.0),
            source_type=Source.MODEL,
            station_id=None,
            station_name=None,
            station_distance_km=None,
            model_name="GOT5.6",
            days=[],
        )
        output = format_json(result, feet=False, precision=1, local=False, between=None)
        parsed = json_module.loads(output)
        assert parsed["source"]["type"] == "model"
        assert parsed["model"] == "GOT5.6"
