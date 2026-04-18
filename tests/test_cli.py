import datetime
import json as json_module
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from tides.cli import (
    _escape_negative_coords,
    app,
    format_json,
    format_plain,
    parse_between,
    parse_coordinate,
    parse_date_arg,
)
from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.noaa import NOAAError


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

    def test_leading_space_negative_lat(self):
        # main_entry prepends a space; parse_coordinate must accept that form.
        c = parse_coordinate([" -2.88,-39.91"])
        assert c.lat == pytest.approx(-2.88)
        assert c.lon == pytest.approx(-39.91)


class TestEscapeNegativeCoords:
    def test_negative_coord_gets_space(self):
        assert _escape_negative_coords(["get", "-2.88,-39.91"]) == ["get", " -2.88,-39.91"]

    def test_positive_coord_unchanged(self):
        assert _escape_negative_coords(["get", "40.71,-74.00"]) == ["get", "40.71,-74.00"]

    def test_option_flag_unchanged(self):
        assert _escape_negative_coords(["get", "--date", "2026-04-15"]) == [
            "get",
            "--date",
            "2026-04-15",
        ]

    def test_short_flag_unchanged(self):
        assert _escape_negative_coords(["get", "-d", "2026-04-15"]) == ["get", "-d", "2026-04-15"]

    def test_negative_coord_among_options(self):
        argv = ["get", "-2.88,-39.91", "--date", "2025-12-03", "--feet", "--local"]
        assert _escape_negative_coords(argv) == [
            "get",
            " -2.88,-39.91",
            "--date",
            "2025-12-03",
            "--feet",
            "--local",
        ]

    def test_negative_integer_lat(self):
        assert _escape_negative_coords(["-2,-39"]) == [" -2,-39"]

    def test_non_coord_negative_number_unchanged(self):
        # A bare negative number (no comma) is not a coordinate; leave alone
        # so legitimate negative-number option values still parse correctly.
        assert _escape_negative_coords(["-2.88"]) == ["-2.88"]

    def test_leading_decimal_lat(self):
        assert _escape_negative_coords(["-.5,-39.91"]) == [" -.5,-39.91"]

    def test_explicit_positive_lon_sign(self):
        assert _escape_negative_coords(["-2.88,+39.91"]) == [" -2.88,+39.91"]

    def test_whitespace_around_comma(self):
        # Quoted form like "-2.88, -39.91" arrives as a single argv token
        # with internal whitespace; the regex still matches it.
        assert _escape_negative_coords(["-2.88, -39.91"]) == [" -2.88, -39.91"]

    def test_trailing_dot_lat(self):
        assert _escape_negative_coords(["-2.,-39"]) == [" -2.,-39"]


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


# ---------------------------------------------------------------------------
# NEW TEST CLASSES
# ---------------------------------------------------------------------------

runner = CliRunner()


class TestParseCoordinateEdgeCases:
    def test_strips_double_dash(self):
        c = parse_coordinate(["--", "40.7128,-74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_only_double_dash(self):
        with pytest.raises(SystemExit):
            parse_coordinate(["--"])


class TestParseBetweenValidation:
    def test_end_before_start(self):
        with pytest.raises(SystemExit):
            parse_between("18:00:06:00")

    def test_three_colons(self):
        with pytest.raises(SystemExit):
            parse_between("06:00:18")


class TestFormatPlainBetweenFilter:
    def _make_result_with_three_events(self) -> TideResult:
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 6, 0, tzinfo=datetime.timezone.utc),
                height=0.5,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 0, tzinfo=datetime.timezone.utc),
                height=1.2,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 22, 0, tzinfo=datetime.timezone.utc),
                height=-0.3,
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

    def test_between_filters_events(self):
        result = self._make_result_with_three_events()
        between = (datetime.time(8, 0), datetime.time(20, 0))
        output = format_plain(
            result, feet=False, precision=1, local=False, between=between, verbose=False
        )
        assert "14:00" in output
        assert "06:00" not in output
        assert "22:00" not in output

    def test_between_filters_all_events(self):
        result = self._make_result_with_three_events()
        # Window that excludes all three events (07:00-07:30)
        between = (datetime.time(7, 0), datetime.time(7, 30))
        output = format_plain(
            result, feet=False, precision=1, local=False, between=between, verbose=False
        )
        assert output == ""


class TestFormatJsonBetweenFilter:
    def test_between_filters_events_json(self):
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 6, 0, tzinfo=datetime.timezone.utc),
                height=0.5,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 0, tzinfo=datetime.timezone.utc),
                height=1.2,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 22, 0, tzinfo=datetime.timezone.utc),
                height=-0.3,
            ),
        ]
        result = TideResult(
            coordinate=Coordinate(lat=40.7128, lon=-74.0060),
            source_type=Source.NOAA,
            station_id="8518750",
            station_name="The Battery",
            station_distance_km=1.2,
            model_name=None,
            days=[TideDay(date=datetime.date(2026, 4, 15), events=events)],
        )
        between = (datetime.time(8, 0), datetime.time(20, 0))
        output = format_json(result, feet=False, precision=1, local=False, between=between)
        parsed = json_module.loads(output)
        assert len(parsed["days"]) == 1
        tides = parsed["days"][0]["tides"]
        assert len(tides) == 1
        assert tides[0]["time"] == "14:00"


class TestFormatPlainLocal:
    def test_local_time_conversion(self):
        # 2026-04-15 is during EDT (UTC-4) for New York
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 18, 0, tzinfo=datetime.timezone.utc),
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
            days=[TideDay(date=datetime.date(2026, 4, 15), events=events)],
        )
        output = format_plain(
            result, feet=False, precision=1, local=True, between=None, verbose=False
        )
        # 18:00 UTC -> 14:00 EDT (UTC-4)
        assert "14:00" in output
        assert "18:00" not in output


class TestCLIInvocation:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "tides 0.1.0" in result.output

    def test_no_args_shows_usage_error(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 2
        assert "Missing command" in result.output

    def test_invalid_source(self):
        result = runner.invoke(app, ["get", "40.7,-74.0", "--source", "invalid"])
        assert result.exit_code == 1
        assert "Invalid source" in result.output

    def test_negative_precision(self):
        result = runner.invoke(app, ["get", "40.7,-74.0", "--precision", "-1"])
        assert result.exit_code == 1
        assert "non-negative" in result.output

    @patch("tides.resolver.resolve_tides")
    def test_noaa_error_handling(self, mock_resolve):
        mock_resolve.side_effect = NOAAError("test")
        result = runner.invoke(app, ["get", "40.7,-74.0"])
        assert result.exit_code == 2
        assert "test" in result.output

    @patch("tides.resolver.resolve_tides")
    def test_http_status_error(self, mock_resolve):
        mock_resolve.side_effect = httpx.HTTPStatusError(
            "test", request=httpx.Request("GET", "http://test"), response=httpx.Response(500)
        )
        result = runner.invoke(app, ["get", "40.7,-74.0"])
        assert result.exit_code == 2

    @patch("tides.resolver.resolve_tides")
    def test_connection_error(self, mock_resolve):
        mock_resolve.side_effect = httpx.ConnectError("test")
        result = runner.invoke(app, ["get", "40.7,-74.0"])
        assert result.exit_code == 2

    @patch("tides.resolver.resolve_tides")
    def test_generic_exception(self, mock_resolve):
        mock_resolve.side_effect = RuntimeError("boom")
        result = runner.invoke(app, ["get", "40.7,-74.0"])
        assert result.exit_code == 2
        assert "boom" not in result.output


class TestModelFlag:
    @patch("tides.resolver.resolve_tides")
    def test_model_flag_passed_to_resolver(self, mock_resolve):
        mock_resolve.return_value = TideResult(
            coordinate=Coordinate(lat=40.7, lon=-74.0),
            source_type=Source.MODEL,
            station_id=None,
            station_name=None,
            station_distance_km=None,
            model_name="EOT20",
            days=[],
        )
        result = runner.invoke(app, ["get", "40.7,-74.0", "--source", "model", "--model", "eot20"])
        assert result.exit_code == 0
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("model_name") == "EOT20"

    def test_invalid_model_name(self):
        result = runner.invoke(app, ["get", "40.7,-74.0", "--model", "INVALID"])
        assert result.exit_code == 1
        assert "Invalid model" in result.output


class TestFetchModelCommand:
    @patch("tides.cache.fetch_all")
    def test_fetch_model_success(self, mock_fetch_all):
        mock_fetch_all.return_value = None
        result = runner.invoke(app, ["fetch-model"])
        assert result.exit_code == 0

    @patch("tides.cache.fetch_all")
    def test_fetch_model_connection_error(self, mock_fetch_all):
        mock_fetch_all.side_effect = httpx.ConnectError("test")
        result = runner.invoke(app, ["fetch-model"])
        assert result.exit_code == 2

    @patch("tides.cache.fetch_all")
    def test_fetch_model_http_error(self, mock_fetch_all):
        mock_fetch_all.side_effect = httpx.HTTPStatusError(
            "test", request=httpx.Request("GET", "http://test"), response=httpx.Response(500)
        )
        result = runner.invoke(app, ["fetch-model"])
        assert result.exit_code == 2

    @patch("tides.cache.fetch_all")
    def test_fetch_model_generic_error(self, mock_fetch_all):
        mock_fetch_all.side_effect = RuntimeError("unexpected")
        result = runner.invoke(app, ["fetch-model"])
        assert result.exit_code == 2


class TestCacheShowCommand:
    @patch("tides.cache.get_cache_info")
    def test_cache_show_plain(self, mock_info):
        mock_info.return_value = {
            "app_cache": {
                "path": "/fake/app",
                "items": [
                    {"name": "NOAA station list", "path": "/fake/app/noaa.json", "size": 1024}
                ],
            },
            "model_cache": {
                "path": "/fake/pytmd",
                "items": [{"name": "GOT5.6", "path": "/fake/pytmd/GOT5.6", "size": 100_000_000}],
            },
        }
        result = runner.invoke(app, ["cache"])
        assert result.exit_code == 0
        assert "App cache" in result.output
        assert "GOT5.6" in result.output
        assert "Total" in result.output

    @patch("tides.cache.get_cache_info")
    def test_cache_show_json(self, mock_info):
        mock_info.return_value = {
            "app_cache": {"path": "/fake", "items": []},
            "model_cache": {"path": "/fake", "items": []},
        }
        result = runner.invoke(app, ["cache", "--json"])
        assert result.exit_code == 0
        parsed = json_module.loads(result.output)
        assert "app_cache" in parsed

    @patch("tides.cache.get_cache_info")
    def test_cache_show_empty(self, mock_info):
        mock_info.return_value = {
            "app_cache": {"path": "/fake", "items": []},
            "model_cache": {"path": "/fake", "items": []},
        }
        result = runner.invoke(app, ["cache"])
        assert result.exit_code == 0
        assert "(empty)" in result.output


class TestCacheClearCommand:
    @patch("tides.cache.clear_cache")
    def test_clear_with_yes_flag(self, mock_clear):
        mock_clear.return_value = 5000
        result = runner.invoke(app, ["cache", "clear", "got5.6", "--yes"])
        assert result.exit_code == 0
        assert "Cleared" in result.output
        mock_clear.assert_called_once_with("got5.6")

    @patch("tides.cache.clear_cache")
    def test_clear_all_with_yes(self, mock_clear):
        mock_clear.return_value = 10_000_000
        result = runner.invoke(app, ["cache", "clear", "--yes"])
        assert result.exit_code == 0
        assert "all cached data" in result.output

    @patch("tides.cache.clear_cache")
    def test_clear_invalid_name(self, mock_clear):
        mock_clear.side_effect = ValueError("Unknown cache name 'bogus'")
        result = runner.invoke(app, ["cache", "clear", "bogus", "--yes"])
        assert result.exit_code == 1
        assert "Unknown cache name" in result.output

    @patch("tides.cache.clear_cache")
    def test_clear_cancelled(self, mock_clear):
        result = runner.invoke(app, ["cache", "clear", "got5.6"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        mock_clear.assert_not_called()

    @patch("tides.cache.clear_cache")
    def test_clear_os_error(self, mock_clear):
        mock_clear.side_effect = OSError("Permission denied")
        result = runner.invoke(app, ["cache", "clear", "got5.6", "--yes"])
        assert result.exit_code == 2
        assert "Permission denied" in result.output
