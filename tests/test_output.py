"""Output formats: datum in --verbose, JSON datetime/type, empty days,
empty-result notice, no negative zero (#19)."""

import datetime
import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tides.cli import app, format_json, format_plain
from tides.harmonics import predict_tides_for_day
from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.noaa import parse_predictions_response

runner = CliRunner()
UTC = datetime.timezone.utc
GUAJIRU = Coordinate(lat=-2.881, lon=-39.908)  # America/Fortaleza, UTC-3


def _result(source=Source.MODEL, events=None, days=None, datum="mllw", coord=GUAJIRU):
    if days is None:
        days = [TideDay(date=datetime.date(2026, 9, 25), events=events or [])]
    return TideResult(
        coordinate=coord,
        source_type=source,
        station_id="st" if source != Source.MODEL else None,
        station_name="Somewhere" if source != Source.MODEL else None,
        station_distance_km=1.2 if source != Source.MODEL else None,
        model_name="GOT5.6" if source == Source.MODEL else None,
        days=days,
        datum=datum,
    )


EVENTS = [
    TideEvent(time=datetime.datetime(2026, 9, 25, 6, 34, tzinfo=UTC), height=2.4, kind="high"),
    TideEvent(time=datetime.datetime(2026, 9, 25, 12, 47, tzinfo=UTC), height=-0.2, kind="low"),
]


class TestVerboseDatum:
    @pytest.mark.parametrize(
        ("source", "prefix"),
        [
            (Source.NOAA, "[NOAA: Somewhere, 1.2km, MLLW]"),
            (Source.STATION, "[Station: Somewhere, 1.2km, MLLW]"),
            (Source.MODEL, "[Model: GOT5.6, MLLW]"),
        ],
    )
    def test_prefix_includes_datum(self, source, prefix):
        out = format_plain(_result(source, EVENTS), False, 1, False, None, True)
        assert out.startswith(prefix)

    def test_non_verbose_output_unchanged(self):
        out = format_plain(_result(events=EVENTS), False, 1, False, None, False)
        assert out == "2.4m@06:34, -0.2m@12:47"


class TestJsonFields:
    def test_utc_datetime_and_type(self):
        data = json.loads(format_json(_result(events=EVENTS), False, 1, False, None))
        e = data["days"][0]["tides"][0]
        assert e["datetime"] == "2026-09-25T06:34+00:00"
        assert e["type"] == "high"
        assert e["time"] == "06:34" and e["height"] == 2.4  # existing keys kept

    def test_local_datetime_offset_matches_date_and_time(self):
        data = json.loads(format_json(_result(events=EVENTS), False, 1, True, None))
        day = data["days"][0]
        for e in day["tides"]:
            parsed = datetime.datetime.fromisoformat(e["datetime"])
            assert e["datetime"].endswith("-03:00")
            assert e["datetime"].startswith(day["date"] + "T" + e["time"])
            assert parsed.utcoffset() == datetime.timedelta(hours=-3)
        assert [e["type"] for e in day["tides"]] == ["high", "low"]

    def test_filtered_empty_days_are_kept(self):
        days = [
            TideDay(date=datetime.date(2026, 9, 25), events=EVENTS),
            TideDay(date=datetime.date(2026, 9, 26), events=[]),
        ]
        window = (datetime.time(4, 30), datetime.time(5, 0))
        data = json.loads(format_json(_result(days=days), False, 1, False, window))
        assert [d["date"] for d in data["days"]] == ["2026-09-25", "2026-09-26"]
        assert all(d["tides"] == [] for d in data["days"])


class TestKinds:
    def test_harmonic_extrema_alternate(self):
        from test_harmonics import BERMUDA_CONSTITUENTS

        events = predict_tides_for_day(datetime.date(2026, 4, 15), BERMUDA_CONSTITUENTS)
        kinds = [e.kind for e in events]
        assert set(kinds) == {"high", "low"}
        assert all(a != b for a, b in zip(kinds, kinds[1:]))

    def test_noaa_type_mapping(self):
        data = {
            "predictions": [
                {"t": "2026-04-15 02:18", "v": "1.5", "type": "HH"},
                {"t": "2026-04-15 08:42", "v": "0.0", "type": "L"},
                {"t": "2026-04-15 14:36", "v": "1.3", "type": "H"},
                {"t": "2026-04-15 20:54", "v": "-0.1", "type": "LL"},
            ]
        }
        assert [e.kind for e in parse_predictions_response(data)] == [
            "high",
            "low",
            "high",
            "low",
        ]


class TestNegativeZero:
    def test_plain_and_json(self):
        ev = [
            TideEvent(time=datetime.datetime(2026, 9, 25, 3, tzinfo=UTC), height=-0.04, kind="low")
        ]
        assert format_plain(_result(events=ev), False, 1, False, None, False) == "0.0m@03:00"
        data = json.loads(format_json(_result(events=ev), False, 1, False, None))
        assert str(data["days"][0]["tides"][0]["height"]) == "0.0"


class TestEmptyNotice:
    @patch("tides.resolver.resolve_tides")
    def test_plain_empty_prints_note_exit_0(self, mock_resolve):
        mock_resolve.return_value = _result(events=EVENTS)
        r = runner.invoke(app, ["get", "40.7,-74.0", "-b", "04:30:05:00"])
        assert r.exit_code == 0
        assert r.stdout == ""
        assert "Note: no tide events matched" in r.stderr

    @patch("tides.resolver.resolve_tides")
    def test_json_empty_has_no_note(self, mock_resolve):
        mock_resolve.return_value = _result(events=EVENTS)
        r = runner.invoke(app, ["get", "40.7,-74.0", "-b", "04:30:05:00", "-j"])
        assert r.exit_code == 0
        assert "Note" not in r.stderr
        assert json.loads(r.stdout)["days"][0]["tides"] == []
