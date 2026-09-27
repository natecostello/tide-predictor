"""NOAA path requests the user's datum natively and falls through in auto (#12)."""

import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.noaa import (
    NOAADatumUnavailableError,
    NOAAError,
    fetch_predictions,
    fetch_station_datums,
    parse_station_list,
)
from tides.resolver import resolve_tides

DAY = datetime.date(2026, 9, 28)
BATTERY = Coordinate(lat=40.7128, lon=-74.0060)

REF_STATION = {"id": "9414290", "name": "San Francisco", "lat": 40.7006, "lon": -74.0142}
REF_STATION_TYPED = {**REF_STATION, "type": "R"}
SUB_STATION = {**REF_STATION, "id": "8652226", "name": "Jennettes Pier", "type": "S"}

MLLW_PREDICTIONS = {
    "predictions": [
        {"t": "2026-09-28 01:34", "v": "0.095", "type": "L"},
        {"t": "2026-09-28 08:13", "v": "1.574", "type": "H"},
    ]
}

# 9414290 datums.json (metric), trimmed to what the code reads.
DATUMS_JSON = {
    "datums": [
        {"name": "STND", "value": 0.0},
        {"name": "MLLW", "value": 1.822},
        {"name": "MSL", "value": 2.773},
    ],
    "LAT": 1.2268009,
    "HAT": 4.027801,
}


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _model_result() -> TideResult:
    return TideResult(
        coordinate=BATTERY,
        source_type=Source.MODEL,
        station_id=None,
        station_name=None,
        station_distance_km=None,
        model_name="GOT5.6",
        days=[
            TideDay(
                date=DAY,
                events=[
                    TideEvent(
                        time=datetime.datetime(2026, 9, 28, 3, tzinfo=datetime.timezone.utc),
                        height=0.1,
                        kind="high",
                    )
                ],
            )
        ],
    )


class TestNoaaClient:
    @patch("tides.noaa.httpx.get")
    def test_datum_param_is_sent(self, mock_get):
        mock_get.return_value = _mock_response(MLLW_PREDICTIONS)
        fetch_predictions("9414290", DAY, DAY, "mhhw")
        assert mock_get.call_args.kwargs["params"]["datum"] == "MHHW"

    @patch("tides.noaa.httpx.get")
    def test_fetch_station_datums(self, mock_get):
        mock_get.return_value = _mock_response(DATUMS_JSON)
        d = fetch_station_datums("9414290")
        assert d["MLLW"] == pytest.approx(1.822)
        assert d["LAT"] == pytest.approx(1.2268009)
        assert d["HAT"] == pytest.approx(4.027801)
        assert mock_get.call_args.kwargs["params"] == {"units": "metric"}

    @patch("tides.noaa.httpx.get")
    def test_fetch_station_datums_null_raises(self, mock_get):
        mock_get.return_value = _mock_response({"datums": None, "LAT": None, "HAT": None})
        with pytest.raises(NOAAError):
            fetch_station_datums("8652226")

    def test_parse_station_type(self):
        xml = (
            "<Stations><Station><id>1</id><name>A</name><lat>1</lat><lng>2</lng>"
            "<type>S</type></Station><Station><id>2</id><name>B</name><lat>1</lat>"
            "<lng>2</lng></Station></Stations>"
        )
        stations = parse_station_list(xml)
        assert stations[0]["type"] == "S"
        assert "type" not in stations[1]


@patch("tides.datums.get_model_datums", side_effect=AssertionError("model datums on NOAA path"))
class TestNoaaDatums:
    @patch("tides.resolver.fetch_predictions", return_value=MLLW_PREDICTIONS)
    @patch("tides.resolver.get_stations", return_value=[REF_STATION_TYPED])
    def test_native_datum_requested_and_untouched(self, _stations, mock_fetch, _md):
        result = resolve_tides(BATTERY, DAY, DAY, Source.NOAA, datum="msl")
        assert mock_fetch.call_args.args[3] == "msl"
        assert result.datum == "msl"
        # Heights are whatever NOAA returned for that datum; no shift applied.
        assert [e.height for e in result.days[0].events] == [0.095, 1.574]

    @pytest.mark.parametrize("model", ["GOT5.6", "FES2022", "EOT20"])
    @patch("tides.resolver.fetch_predictions", return_value=MLLW_PREDICTIONS)
    @patch("tides.resolver.get_stations", return_value=[REF_STATION_TYPED])
    def test_model_flag_does_not_affect_noaa(self, _stations, _fetch, _md, model):
        result = resolve_tides(BATTERY, DAY, DAY, Source.NOAA, model_name=model, datum="mllw")
        assert [e.height for e in result.days[0].events] == [0.095, 1.574]

    @pytest.mark.parametrize(
        ("datum", "expected"),
        # height_target = height_MLLW - (TARGET_stnd - MLLW_stnd)
        [("lat", [0.690, 2.169]), ("hat", [-2.111, -0.632])],
    )
    @patch("tides.resolver.fetch_station_datums")
    @patch("tides.resolver.fetch_predictions", return_value=MLLW_PREDICTIONS)
    @patch("tides.resolver.get_stations", return_value=[REF_STATION_TYPED])
    def test_lat_hat_derived_from_station_datums(
        self, _stations, mock_fetch, mock_datums, _md, datum, expected
    ):
        mock_datums.return_value = {"MLLW": 1.822, "LAT": 1.2268009, "HAT": 4.027801}
        result = resolve_tides(BATTERY, DAY, DAY, Source.NOAA, datum=datum)
        assert mock_fetch.call_args.args[3] == "mllw"
        heights = [e.height for e in result.days[0].events]
        assert heights == pytest.approx(expected, abs=1e-3)
        assert result.datum == datum

    @patch("tides.resolver.fetch_station_datums", return_value={"MLLW": 1.0})
    @patch("tides.resolver.fetch_predictions", return_value=MLLW_PREDICTIONS)
    @patch("tides.resolver.get_stations", return_value=[REF_STATION_TYPED])
    def test_lat_missing_at_station_is_unavailable(self, *_):
        with pytest.raises(NOAADatumUnavailableError, match="does not publish LAT"):
            resolve_tides(BATTERY, DAY, DAY, Source.NOAA, datum="lat")

    @patch("tides.resolver.fetch_predictions")
    @patch("tides.resolver.get_stations", return_value=[SUB_STATION])
    def test_subordinate_non_mllw_noaa_source_errors(self, _stations, mock_fetch, _md):
        with pytest.raises(NOAADatumUnavailableError, match="only publishes MLLW"):
            resolve_tides(BATTERY, DAY, DAY, Source.NOAA, datum="msl")
        mock_fetch.assert_not_called()

    @patch("tides.resolver.fetch_predictions", return_value=MLLW_PREDICTIONS)
    @patch("tides.resolver.get_stations", return_value=[SUB_STATION])
    def test_subordinate_mllw_works(self, _stations, _fetch, _md):
        result = resolve_tides(BATTERY, DAY, DAY, Source.NOAA, datum="mllw")
        assert result.source_type == Source.NOAA

    @patch("tides.resolver.fetch_predictions", side_effect=None)
    @patch("tides.resolver.get_stations", return_value=[REF_STATION])  # no "type" (old cache)
    def test_untyped_station_noaa_error_is_datum_unavailable(self, _stations, mock_fetch, _md):
        mock_fetch.return_value = {"error": {"message": "No Predictions data was found."}}
        with pytest.raises(NOAADatumUnavailableError, match="returned no MSL predictions"):
            resolve_tides(BATTERY, DAY, DAY, Source.NOAA, datum="msl")


class TestAutoFallthrough:
    @patch("tides.resolver._resolve_model")
    @patch("tides.resolver._resolve_station", return_value=None)
    @patch("tides.resolver.fetch_predictions")
    @patch("tides.resolver.get_stations", return_value=[SUB_STATION])
    def test_subordinate_msl_falls_through(self, _st, mock_fetch, _rs, mock_model, capsys):
        mock_model.return_value = _model_result()
        result = resolve_tides(BATTERY, DAY, DAY, Source.AUTO, datum="msl")
        assert result.source_type == Source.MODEL
        mock_fetch.assert_not_called()
        err = capsys.readouterr().err
        assert "Note:" in err and "only publishes MLLW" in err

    @pytest.mark.parametrize(
        "exc",
        [
            NOAAError("NOAA API error: boom"),
            httpx.ConnectError("offline"),
            httpx.HTTPStatusError(
                "503", request=httpx.Request("GET", "https://x"), response=httpx.Response(503)
            ),
        ],
    )
    @patch("tides.resolver._resolve_model")
    @patch("tides.resolver._resolve_station", return_value=None)
    @patch("tides.resolver.fetch_predictions")
    @patch("tides.resolver.get_stations", return_value=[REF_STATION_TYPED])
    def test_noaa_failures_fall_through(self, _st, mock_fetch, _rs, mock_model, exc, capsys):
        mock_fetch.side_effect = exc
        mock_model.return_value = _model_result()
        result = resolve_tides(BATTERY, DAY, DAY, Source.AUTO, datum="mllw")
        assert result.source_type == Source.MODEL
        assert "Note:" in capsys.readouterr().err

    @patch("tides.resolver.fetch_predictions", side_effect=httpx.ConnectError("offline"))
    @patch("tides.resolver.get_stations", return_value=[REF_STATION_TYPED])
    def test_noaa_source_network_error_propagates(self, *_):
        with pytest.raises(httpx.ConnectError):
            resolve_tides(BATTERY, DAY, DAY, Source.NOAA, datum="mllw")


class TestNoaaCli:
    """User-facing --source noaa path: exit code and actionable message."""

    @patch("tides.resolver.fetch_predictions")
    @patch("tides.resolver.get_stations", return_value=[SUB_STATION])
    def test_subordinate_non_mllw_exits_2_with_message(self, _stations, mock_fetch):
        from typer.testing import CliRunner

        from tides.cli import app

        result = CliRunner().invoke(
            app, ["get", "40.7128,-74.0060", "--source", "noaa", "--datum", "msl"]
        )
        assert result.exit_code == 2
        assert "Error:" in result.output
        assert "8652226" in result.output
        assert "only publishes MLLW" in result.output
        mock_fetch.assert_not_called()
