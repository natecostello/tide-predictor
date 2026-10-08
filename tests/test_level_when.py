"""peaks / level / when on a shared height curve (#31)."""

import datetime
import json
import re
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from typer.testing import CliRunner

from tides import cli
from tides.cli import app
from tides.curve import search_level
from tides.models import Coordinate, Source
from tides.noaa import parse_predictions_response
from tides.resolver import resolve_curve, resolve_tides

runner = CliRunner()
UTC = datetime.timezone.utc
FIXTURES = Path(__file__).parent / "fixtures"
COORD = "40.7,-74.0"
FORTALEZA = " -3.72,-38.5"  # escaped as main_entry does
PERIOD_H = 12.42


def _synthetic(t, local=None, model=None):
    """Semidiurnal tide plus a diurnal inequality (successive highs differ).

    `t` is days since the pyTMD epoch; peaks and the curve share it.
    """
    hours = np.asarray(t, dtype=float) * 24.0
    return np.cos(2 * np.pi * hours / PERIOD_H) + 0.2 * np.cos(2 * np.pi * hours / 24.0)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr("tides.ocean_model.load_local_constituents", lambda *a, **k: (None, None))
    monkeypatch.setattr("tides.ocean_model.predict_elevations", _synthetic)


def _freeze(monkeypatch, when: datetime.datetime) -> None:
    monkeypatch.setattr(cli, "_now", lambda: when)


def _run(*args: str):
    return runner.invoke(app, list(args))


def _json(r) -> dict:
    assert r.exit_code == 0, r.output
    return json.loads(r.stdout)


def _rows(data: dict) -> list[dict]:
    return [row for day in data["days"] for row in day["tides"]]


def _window(day: datetime.date):
    start = datetime.datetime.combine(day, datetime.time(), tzinfo=UTC)
    return start, start + datetime.timedelta(days=1)


DAY = datetime.date(2026, 10, 8)


# -- the rename -------------------------------------------------------------


class TestRename:
    def test_get_is_gone(self):
        r = _run("get", COORD)
        assert r.exit_code != 0
        assert "No such command" in r.output

    @pytest.mark.parametrize("cmd", ["peaks", "level", "when"])
    def test_help(self, cmd):
        r = _run(cmd, "-h")
        assert r.exit_code == 0
        # Rich may color help (CI forces color), splitting "--datum".
        assert "--datum" in re.sub(r"\x1b\[[0-9;]*m", "", r.stdout)

    def test_peaks_json_rate_is_additive(self, model):
        data = _json(_run("peaks", COORD, "-d", "2026-10-08", "-s", "model", "-j"))
        rows = _rows(data)
        assert rows
        for row in rows:
            assert set(row) == {"time", "height", "datetime", "type", "rate"}
            assert row["rate"] == 0.0
            assert row["type"] in ("high", "low")

    def test_peaks_plain_unchanged(self, model):
        r = _run("peaks", COORD, "-d", "2026-10-08", "-s", "model")
        assert r.exit_code == 0
        for row in r.stdout.strip().split(", "):
            assert " " not in row  # no type or rate on peaks rows


# -- round-trip properties ----------------------------------------------------


def _check_round_trip(curve, peaks_events, check_rate: bool) -> None:
    for event in peaks_events:
        p = curve.at([event.time])[0]
        assert p.kind == event.kind
        if check_rate:
            assert abs(curve.rates[round(curve._pos(event.time))]) < 0.05
    start, end = _window(DAY)
    for level in (-0.4, 0.0, 0.3):
        found = search_level(curve, level, [(start, end)])
        assert found.rows
        for row in found.rows:
            if row.near is None:
                assert abs(curve.at([row.time])[0].height - level) < 0.005


class TestModelPath:
    def test_round_trip(self, model):
        coord = Coordinate(40.7, -74.0)
        start, end = _window(DAY)
        result = resolve_tides(coord, DAY, DAY, Source.MODEL, datum="msl")
        _, curve = resolve_curve(coord, start, end, Source.MODEL, datum="msl")
        events = [e for d in result.days for e in d.events]
        assert events
        _check_round_trip(curve, events, check_rate=True)

    def test_all_nan_is_inland(self, monkeypatch):
        monkeypatch.setattr(
            "tides.ocean_model.load_local_constituents", lambda *a, **k: (None, None)
        )
        monkeypatch.setattr(
            "tides.ocean_model.predict_elevations", lambda t, *a: np.full(len(t), np.nan)
        )
        for args in (["level", COORD], ["when", COORD, "--level", "1"]):
            r = _run(*args, "-s", "model")
            assert r.exit_code == 2
            assert "No tidal data for this location -- it may be inland." in r.stderr

    def test_model_datum_shift_matches_peaks(self, model, monkeypatch):
        offsets = {"msl": 0.0, "lat": -1.25, "mllw": -1.0}
        monkeypatch.setattr("tides.datums.get_model_datums", lambda *a, **k: dict(offsets))
        coord = Coordinate(40.7, -74.0)
        start, end = _window(DAY)
        result = resolve_tides(coord, DAY, DAY, Source.MODEL, datum="lat")
        _, curve = resolve_curve(coord, start, end, Source.MODEL, datum="lat")
        for e in (e for d in result.days for e in d.events):
            assert curve.at([e.time])[0].height == pytest.approx(e.height, abs=0.005)


STATION = json.loads((FIXTURES / "station_9414290.json").read_text())
STATION_ENTRY = {"id": "9414290", "name": STATION["name"], "lat": 37.8, "lon": -122.47}


@pytest.fixture
def station(monkeypatch):
    monkeypatch.setattr("tides.stations.get_station_index", lambda: [STATION_ENTRY])
    monkeypatch.setattr(
        "tides.stations.find_nearest_usable_station",
        lambda *a, **k: (STATION_ENTRY, STATION, 1.0),
    )


class TestStationPath:
    def test_round_trip(self, station):
        coord = Coordinate(37.8, -122.47)
        start, end = _window(DAY)
        result = resolve_tides(coord, DAY, DAY, Source.STATION, datum="msl")
        _, curve = resolve_curve(coord, start, end, Source.STATION, datum="msl")
        events = [e for d in result.days for e in d.events]
        assert events
        _check_round_trip(curve, events, check_rate=True)

    @pytest.mark.parametrize("datum", ["mllw", "lat", "msl", "mhhw"])
    def test_level_at_peaks_time_matches_peaks_height(self, station, datum):
        # Heights datum is MLLW (not msl): both the chart-datum offset and the
        # _apply_datum shift must be applied.
        peaks = _json(
            _run(
                "peaks",
                "37.8,-122.47",
                "-d",
                "2026-10-08",
                "-s",
                "station",
                "--datum",
                datum,
                "-j",
                "-p",
                "4",
            )
        )
        whens = []
        for row in _rows(peaks):
            whens += ["--when", row["datetime"][:16]]
        level = _json(
            _run(
                "level", "37.8,-122.47", *whens, "-s", "station", "--datum", datum, "-j", "-p", "4"
            )
        )
        for a, b in zip(_rows(peaks), _rows(level)):
            assert b["height"] == pytest.approx(a["height"], abs=0.005)
            assert b["type"] == a["type"]

    def test_all_nan_station_errors_like_today(self, station, monkeypatch):
        monkeypatch.setattr(
            "tides.harmonics.predict_elevations", lambda c, t, chunks=1: np.full(len(t), np.nan)
        )
        r = _run("level", "37.8,-122.47", "-s", "station")
        assert r.exit_code == 1
        assert "No tide station found" in r.stderr


# -- NOAA ---------------------------------------------------------------------

NOAA_6MIN = json.loads((FIXTURES / "noaa_9414290_msl_6min_20260915_16.json").read_text())
# Captured once with:
# curl 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20260915&end_date=20260916&station=9414290&product=predictions&datum=MSL&units=metric&time_zone=gmt&interval=hilo&format=json&application=tides_cli'
NOAA_HILO = json.loads((FIXTURES / "noaa_9414290_msl_hilo_20260915_16.json").read_text())
NOAA_STATION = {"id": "9414290", "name": "San Francisco", "lat": 37.8, "lon": -122.47, "type": "R"}
NOAA_DAY = datetime.date(2026, 9, 15)


def _noaa_fetch(station_id, begin, end, datum="mllw", interval="hilo"):
    return NOAA_6MIN if interval == "6" else NOAA_HILO


class TestNoaaPath:
    @pytest.fixture
    def curve(self):
        with (
            patch("tides.resolver.get_stations", return_value=[NOAA_STATION]),
            patch("tides.resolver.fetch_predictions", side_effect=_noaa_fetch) as fetch,
        ):
            # Inside the fixture's two days, padding included.
            start = datetime.datetime(2026, 9, 15, 3, tzinfo=UTC)
            end = datetime.datetime(2026, 9, 16, 21, tzinfo=UTC)
            src, curve = resolve_curve(
                Coordinate(37.8, -122.47), start, end, Source.NOAA, datum="msl"
            )
            assert fetch.call_args.kwargs["interval"] == "6"
            assert src.source_type == Source.NOAA
            yield curve, start, end

    def test_turns_match_hilo(self, curve):
        curve, start, end = curve
        hilo = [e for e in parse_predictions_response(NOAA_HILO) if start <= e.time < end]
        turns = curve.turns(start, end)
        assert len(turns) == len(hilo)
        for turn, event in zip(turns, hilo):
            assert turn.kind == event.kind
            assert abs((turn.time - event.time).total_seconds()) <= 6 * 60
            assert curve.at([event.time])[0].kind == event.kind

    def test_round_trip(self, curve):
        curve, start, end = curve
        for level in (-0.5, 0.0, 0.5):
            for row in search_level(curve, level, [(start, end)]).rows:
                if row.near is None:
                    assert abs(curve.at([row.time])[0].height - level) < 0.005

    def test_peaks_still_uses_hilo(self):
        with (
            patch("tides.resolver.get_stations", return_value=[NOAA_STATION]),
            patch("tides.resolver.fetch_predictions", side_effect=_noaa_fetch) as fetch,
        ):
            r = _run("peaks", "37.8,-122.47", "-d", "2026-09-15", "-s", "noaa", "--datum", "msl")
        assert r.exit_code == 0
        assert r.stdout.strip() == "-0.8m@03:34, 0.4m@10:19, -0.2m@15:17, 0.8m@21:40"
        assert fetch.call_args.kwargs.get("interval", "hilo") == "hilo"

    def test_level_on_noaa(self, monkeypatch):
        with (
            patch("tides.resolver.get_stations", return_value=[NOAA_STATION]),
            patch("tides.resolver.fetch_predictions", side_effect=_noaa_fetch),
        ):
            r = _run(
                "level",
                "37.8,-122.47",
                "--when",
                "2026-09-15T21:40",
                "-s",
                "noaa",
                "--datum",
                "msl",
                "-v",
            )
        assert r.exit_code == 0, r.output
        assert r.stdout.strip() == "[NOAA: San Francisco, 0.0km, MSL] 0.8m@21:40 high"


SUB_STATION = {"id": "1611401", "name": "Waimea Bay", "lat": 40.7, "lon": -74.0, "type": "S"}
SUB_MESSAGE = (
    "Error: NOAA station 1611401 (Waimea Bay) is a subordinate station and publishes "
    "only high/low predictions; use --source station or model"
)


class TestSubordinate:
    def test_noaa_source_errors(self):
        with (
            patch("tides.resolver.get_stations", return_value=[SUB_STATION]),
            patch("tides.resolver.fetch_predictions") as fetch,
        ):
            r = _run("level", COORD, "-s", "noaa")
        assert r.exit_code == 2
        assert SUB_MESSAGE in r.stderr
        fetch.assert_not_called()

    def test_datum_check_runs_first(self):
        with patch("tides.resolver.get_stations", return_value=[SUB_STATION]):
            r = _run("level", COORD, "-s", "noaa", "--datum", "msl")
        assert r.exit_code == 2
        assert "only publishes MLLW predictions" in r.stderr

    def test_untyped_station_rejected_by_noaa(self):
        untyped = {k: v for k, v in SUB_STATION.items() if k != "type"}
        no_data = {"error": {"message": "No Predictions data was found."}}
        with (
            patch("tides.resolver.get_stations", return_value=[untyped]),
            patch("tides.resolver.fetch_predictions", return_value=no_data),
        ):
            r = _run("level", COORD, "-s", "noaa")
        assert r.exit_code == 2
        assert SUB_MESSAGE in r.stderr

    def test_auto_falls_through(self, model):
        with (
            patch("tides.resolver.get_stations", return_value=[SUB_STATION]),
            patch("tides.resolver._curve_station", return_value=None),
        ):
            r = _run("level", COORD, "-j")
        data = _json(r)
        assert data["source"]["type"] == "model"
        assert "subordinate station" in r.stderr
        assert "using other sources" in r.stderr


# -- level ------------------------------------------------------------------


class TestLevel:
    def test_default_is_now(self, model, monkeypatch):
        _freeze(monkeypatch, datetime.datetime(2026, 10, 8, 12, 31, tzinfo=UTC))
        rows = _rows(_json(_run("level", COORD, "-s", "model", "-j")))
        assert [r["datetime"] for r in rows] == ["2026-10-08T12:31+00:00"]
        assert rows[0]["type"] in ("rising", "falling", "high", "low")

    def test_hhmm_uses_display_clock(self, model, monkeypatch):
        _freeze(monkeypatch, datetime.datetime(2026, 10, 8, 12, 0, tzinfo=UTC))
        rows = _rows(_json(_run("level", FORTALEZA, "--when", "13:00", "-l", "-s", "model", "-j")))
        assert rows[0]["datetime"] == "2026-10-08T13:00-03:00"
        rows = _rows(_json(_run("level", FORTALEZA, "--when", "13:00", "-s", "model", "-j")))
        assert rows[0]["datetime"] == "2026-10-08T13:00+00:00"

    def test_hhmm_today_follows_local_clock(self, model, monkeypatch):
        # 01:30Z is still the previous day in Fortaleza (UTC-3).
        _freeze(monkeypatch, datetime.datetime(2026, 10, 9, 1, 30, tzinfo=UTC))
        rows = _rows(_json(_run("level", FORTALEZA, "--when", "13:00", "-l", "-s", "model", "-j")))
        assert rows[0]["datetime"] == "2026-10-08T13:00-03:00"

    def test_grouping_and_order(self, model):
        r = _run(
            "level",
            COORD,
            "-s",
            "model",
            "--when",
            "2026-10-09T16:00",
            "--when",
            "2026-10-08T18:00",
            "--when",
            "2026-10-08T06:00",
        )
        assert r.exit_code == 0, r.output
        lines = r.stdout.strip().splitlines()
        assert lines[0].startswith("2026-10-08: ")
        assert "@18:00" in lines[0].split(", ")[0]
        assert "@06:00" in lines[0].split(", ")[1]
        assert lines[1].startswith("2026-10-09: ")
        data = _json(_run("level", COORD, "-s", "model", "-j", "--when", "2026-10-09T16:00"))
        assert [d["date"] for d in data["days"]] == ["2026-10-09"]

    def test_plain_row_format(self, model):
        r = _run("level", COORD, "-s", "model", "--when", "2026-10-08T03:00", "-f", "-p", "2")
        assert r.exit_code == 0
        row = r.stdout.strip()
        height, rest = row.split("@")
        assert height.endswith("ft")
        time, kind, rate = rest.split(" ")
        assert time == "03:00"
        assert kind in ("rising", "falling")
        assert rate.endswith("ft/h") and rate[0] in "+-"
        assert len(rate.split(".")[1]) == len("00ft/h")

    @pytest.mark.parametrize(
        "bad", ["25:00", "noon", "2026-10-08 16:00", "2026-13-01T10:00", "1600"]
    )
    def test_invalid_when(self, bad):
        r = _run("level", COORD, "--when", bad)
        assert r.exit_code == 1
        assert "Expected: now, HH:MM or YYYY-MM-DDTHH:MM" in r.stderr

    def test_span_cap(self):
        r = _run("level", COORD, "--when", "2026-01-01T00:00", "--when", "2027-01-05T00:00")
        assert r.exit_code == 1
        assert "maximum is 366" in r.stderr

    def test_rejects_date_and_between(self):
        assert _run("level", COORD, "--date", "2026-10-08").exit_code != 0
        assert _run("level", COORD, "--between", "06:00:07:00").exit_code != 0


# -- when ---------------------------------------------------------------------


class TestWhen:
    def test_numeric_both_directions(self, model):
        rows = _rows(
            _json(_run("when", COORD, "--level", "0", "-d", "2026-10-08", "-s", "model", "-j"))
        )
        kinds = {r["type"] for r in rows}
        assert kinds == {"rising", "falling"}
        assert all(r["height"] == 0.0 for r in rows)

    def test_rising_and_falling_filters(self, model):
        base = ["when", COORD, "--level", "0", "-d", "2026-10-08", "-s", "model", "-j"]
        assert {r["type"] for r in _rows(_json(_run(*base, "--rising")))} == {"rising"}
        assert {r["type"] for r in _rows(_json(_run(*base, "--falling")))} == {"falling"}
        r = _run(*base, "--rising", "--falling")
        assert r.exit_code == 1
        assert "mutually exclusive" in r.stderr

    def test_missing_level_exits_1(self):
        r = _run("when", COORD)
        assert r.exit_code == 1
        assert "--level is required" in r.stderr

    def test_invalid_level(self):
        r = _run("when", COORD, "--level", "high")
        assert r.exit_code == 1
        assert "Invalid --level" in r.stderr

    def test_negative_level_parses(self, model):
        r = _run("when", COORD, "--level", "-0.5", "-d", "2026-10-08", "-s", "model", "-j")
        rows = _rows(_json(r))
        assert rows and all(row["height"] == -0.5 for row in rows)

    def test_negative_level_through_main_entry(self, model, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "tides",
                "when",
                "-2.88,-39.91",
                "--level",
                "-0.5",
                "-d",
                "2026-10-08",
                "-s",
                "model",
                "-j",
            ],
        )
        with pytest.raises(SystemExit) as e:
            cli.main_entry()
        assert e.value.code in (0, None)
        assert _rows(json.loads(capsys.readouterr().out))

    def test_feet_level(self, model):
        rows = _rows(
            _json(
                _run("when", COORD, "--level", "1", "-f", "-d", "2026-10-08", "-s", "model", "-j")
            )
        )
        assert rows and all(r["height"] == 1.0 for r in rows)
        # 1 ft = 0.3048 m: crossings sit at that metric height.
        m_rows = _rows(
            _json(
                _run("when", COORD, "--level", "0.3048", "-d", "2026-10-08", "-s", "model", "-j")
            )
        )
        assert [r["datetime"] for r in rows] == [r["datetime"] for r in m_rows]

    def test_near_turn_plain_and_json(self, model):
        peaks = _rows(
            _json(_run("peaks", COORD, "-d", "2026-10-08", "-s", "model", "-j", "-p", "4"))
        )
        high = next(p for p in peaks if p["type"] == "high")
        level = str(high["height"] - 0.01)
        base = ["when", COORD, "--level", level, "-d", "2026-10-08", "-s", "model", "-p", "2"]
        for direction in ([], ["--rising"], ["--falling"]):
            rows = _rows(_json(_run(*base, "-j", *direction)))
            near = [r for r in rows if "near" in r]
            assert near and near[0]["datetime"] == high["datetime"]
            assert near[0]["type"] == "high" and near[0]["rate"] == 0.0
            assert near[0]["near"]["from"] < near[0]["datetime"] < near[0]["near"]["to"]
        plain = _run(*base).stdout
        assert f"@{high['time']} high (near: " in plain

    def test_not_reached_note(self, model):
        base = ["when", COORD, "--level", "5", "-d", "2026-10-08:2026-10-09", "-s", "model"]
        r = _run(*base)
        assert r.exit_code == 0
        assert r.stdout == ""
        notes = r.stderr.strip().splitlines()
        assert len(notes) == 2
        assert notes[0].startswith("Note: level 5.0m not reached on 2026-10-08 (max ")
        assert "no tide events matched" not in r.stderr
        r = _run("when", COORD, "--level", "-5", "-d", "2026-10-08", "-s", "model")
        assert "(min " in r.stderr
        r = _run(*base, "-j")
        assert r.exit_code == 0 and r.stderr == ""
        data = json.loads(r.stdout)
        assert [d["tides"] for d in data["days"]] == [[], []]

    def test_reached_but_filtered_note(self, model):
        # 00:00-02:00Z on the synthetic curve is one falling limb.
        base = [
            "when",
            COORD,
            "--level",
            "0.9",
            "-d",
            "2026-10-08",
            "-s",
            "model",
            "-b",
            "00:00:00:30",
        ]
        sample = _rows(
            _json(
                _run("level", COORD, "-s", "model", "-j", "-p", "4", "--when", "2026-10-08T00:15")
            )
        )
        level = str(sample[0]["height"])
        base[3] = level
        r = _run(*base, "--rising")
        assert r.exit_code == 0 and r.stdout == ""
        assert (
            r.stderr.strip()
            == f"Note: no rising crossings of level {float(level):.1f}m on 2026-10-08"
        )
        assert _run(*base, "--rising", "-j").stderr == ""

    def test_date_today_tomorrow_follow_now(self, model, monkeypatch):
        _freeze(monkeypatch, datetime.datetime(2026, 10, 9, 1, 30, tzinfo=UTC))
        base = ["when", FORTALEZA, "--level", "0", "-s", "model", "-j"]
        assert _json(_run(*base, "-d", "today"))["days"][0]["date"] == "2026-10-09"
        assert _json(_run(*base, "-d", "tomorrow"))["days"][0]["date"] == "2026-10-10"
        # On the Fortaleza clock it is still 2026-10-08.
        assert _json(_run(*base, "-l", "-d", "today"))["days"][0]["date"] == "2026-10-08"
        assert _json(_run(*base, "-l", "-d", "tomorrow"))["days"][0]["date"] == "2026-10-09"
        days = _json(_run(*base, "-l", "-d", "today:2026-10-10"))["days"]
        assert [d["date"] for d in days] == ["2026-10-08", "2026-10-09", "2026-10-10"]
        peaks = _json(_run("peaks", FORTALEZA, "-s", "model", "-j", "-l", "-d", "tomorrow"))
        assert peaks["days"][0]["date"] == "2026-10-09"


class TestLevelNow:
    def _peaks(self, day="2026-10-08"):
        return _rows(_json(_run("peaks", COORD, "-d", day, "-s", "model", "-j", "-p", "4")))

    def test_now_not_at_turn_includes_now(self, model, monkeypatch):
        now = datetime.datetime(2026, 10, 8, 3, 7, tzinfo=UTC)
        _freeze(monkeypatch, now)
        point = _rows(_json(_run("level", COORD, "-s", "model", "-j")))[0]
        assert point["type"] in ("rising", "falling")
        rows = _rows(_json(_run("when", COORD, "--level", "now", "-s", "model", "-j")))
        assert {r["type"] for r in rows} == {point["type"]}
        times = [datetime.datetime.fromisoformat(r["datetime"]) for r in rows]
        assert any(abs((t - now).total_seconds()) <= 60 for t in times)

    def test_now_at_high_reports_highs(self, model, monkeypatch):
        high = next(p for p in self._peaks() if p["type"] == "high")
        _freeze(monkeypatch, datetime.datetime.fromisoformat(high["datetime"]))
        rows = _rows(
            _json(
                _run(
                    "when",
                    COORD,
                    "--level",
                    "now",
                    "-d",
                    "2026-10-08:2026-10-09",
                    "-s",
                    "model",
                    "-j",
                )
            )
        )
        highs = [p for p in self._peaks() + self._peaks("2026-10-09") if p["type"] == "high"]
        assert [r["datetime"] for r in rows] == [p["datetime"] for p in highs]
        assert all(r["type"] == "high" for r in rows)
        # The matching high gets a near window; the other (diurnal
        # inequality, more than the tolerance away) does not.
        assert "near" in rows[0]
        assert any("near" not in r for r in rows)

    def test_now_at_high_day_without_high(self, model, monkeypatch):
        high = next(p for p in self._peaks() if p["type"] == "high")
        _freeze(monkeypatch, datetime.datetime.fromisoformat(high["datetime"]))
        lows = [p for p in self._peaks("2026-10-09") if p["type"] == "low"]
        hh = lows[0]["time"]
        # A one-minute window at a low contains no high.
        window = f"{hh}:{hh}"
        base = ["when", COORD, "--level", "now", "-d", "2026-10-09", "-s", "model", "-b", window]
        r = _run(*base)
        assert r.exit_code == 0 and r.stdout == ""
        assert r.stderr.strip() == "Note: no high on 2026-10-09"
        r = _run(*base, "-j")
        assert r.stderr == ""
        assert json.loads(r.stdout)["days"][0]["tides"] == []

    def test_explicit_direction_overrides_turn_mode(self, model, monkeypatch):
        high = next(p for p in self._peaks() if p["type"] == "high")
        _freeze(monkeypatch, datetime.datetime.fromisoformat(high["datetime"]))
        rows = _rows(
            _json(
                _run(
                    "when",
                    COORD,
                    "--level",
                    "now",
                    "--rising",
                    "-d",
                    "2026-10-09",
                    "-s",
                    "model",
                    "-j",
                )
            )
        )
        assert rows and all(r["type"] in ("rising", "high") for r in rows)

    def test_now_far_from_window_uses_same_source(self, model, monkeypatch):
        _freeze(monkeypatch, datetime.datetime(2026, 10, 8, 3, 7, tzinfo=UTC))
        calls = []
        real = resolve_curve

        def spy(*a, **k):
            calls.append(a)
            return real(*a, **k)

        monkeypatch.setattr("tides.resolver.resolve_curve", spy)
        r = _run("when", COORD, "--level", "now", "-d", "2027-10-08", "-s", "model", "-j")
        assert r.exit_code == 0, r.output
        assert len(calls) == 1  # source resolved once; second curve built from it
        assert _rows(json.loads(r.stdout))

    def test_now_composes_unrounded(self, model, monkeypatch):
        now = datetime.datetime(2026, 10, 8, 3, 7, tzinfo=UTC)
        _freeze(monkeypatch, now)
        seen = []
        import tides.curve as curve_mod

        real = curve_mod.search_level

        def spy(curve, level_m, *a, **k):
            seen.append(level_m)
            return real(curve, level_m, *a, **k)

        monkeypatch.setattr(curve_mod, "search_level", spy)
        _run("when", COORD, "--level", "now", "-s", "model", "-p", "0")
        h = _synthetic(
            np.array([(now - datetime.datetime(1992, 1, 1, tzinfo=UTC)).total_seconds() / 86400])
        )[0]
        assert seen and seen[0] == pytest.approx(h, abs=1e-9)


NEW_YORK = ZoneInfo("America/New_York")


class TestDaylightSaving:
    def test_nonexistent_local_time_rejected(self, model):
        # 2026-03-08 02:00-03:00 does not exist in New York.
        r = _run("level", COORD, "-l", "-s", "model", "--when", "2026-03-08T02:30")
        assert r.exit_code == 1
        assert "does not exist on the display clock" in r.stderr
        assert (
            _run("level", COORD, "-l", "-s", "model", "--when", "2026-03-08T03:30").exit_code == 0
        )

    def test_ambiguous_local_time_accepted(self, model):
        # 01:30 occurs twice on 2026-11-01; the first occurrence (EDT) is used.
        rows = _rows(
            _json(_run("level", COORD, "-l", "-s", "model", "-j", "--when", "2026-11-01T01:30"))
        )
        assert rows[0]["datetime"] == "2026-11-01T01:30-04:00"

    def test_fall_back_between_covers_both_occurrences(self):
        segs = cli._day_segments(
            datetime.date(2026, 11, 1), NEW_YORK, (datetime.time(1, 0), datetime.time(1, 30))
        )
        assert segs == [
            (
                datetime.datetime(2026, 11, 1, 5, 0, tzinfo=UTC),
                datetime.datetime(2026, 11, 1, 5, 31, tzinfo=UTC),
            ),
            (
                datetime.datetime(2026, 11, 1, 6, 0, tzinfo=UTC),
                datetime.datetime(2026, 11, 1, 6, 31, tzinfo=UTC),
            ),
        ]

    def test_spring_forward_day_segments(self):
        day = datetime.date(2026, 3, 8)
        whole = cli._day_segments(day, NEW_YORK, None)
        assert whole == [
            (
                datetime.datetime(2026, 3, 8, 5, tzinfo=UTC),
                datetime.datetime(2026, 3, 9, 4, tzinfo=UTC),
            )
        ]
        # 01:30-03:30 wall: 01:30-02:00 EST then 03:00-03:31 EDT.
        segs = cli._day_segments(day, NEW_YORK, (datetime.time(1, 30), datetime.time(3, 30)))
        assert segs == [
            (
                datetime.datetime(2026, 3, 8, 6, 30, tzinfo=UTC),
                datetime.datetime(2026, 3, 8, 7, 0, tzinfo=UTC),
            ),
            (
                datetime.datetime(2026, 3, 8, 7, 0, tzinfo=UTC),
                datetime.datetime(2026, 3, 8, 7, 31, tzinfo=UTC),
            ),
        ]

    def test_wrapping_between_utc(self):
        segs = cli._day_segments(DAY, UTC, (datetime.time(20, 0), datetime.time(4, 0)))
        start, end = _window(DAY)
        assert segs == [
            (start, start + datetime.timedelta(hours=4, minutes=1)),
            (start + datetime.timedelta(hours=20), end),
        ]
