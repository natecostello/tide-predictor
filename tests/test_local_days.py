"""--local groups days on the local clock; edge extrema are not dropped (#10)."""

import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from tides import cli
from tides.models import Coordinate, Source, TideEvent
from tides.ocean_model import utc_minutes
from tides.resolver import _fetch_dates, _group_events_by_date, resolve_tides
from tides.stations import predict_station_tides

UTC = datetime.timezone.utc
FORTALEZA = ZoneInfo("America/Fortaleza")  # UTC-3
AUCKLAND = ZoneInfo("Pacific/Auckland")  # UTC+12 / +13 (DST from 2026-09-27)


def _ev(y, mo, d, h, mi, height=1.0):
    return TideEvent(
        time=datetime.datetime(y, mo, d, h, mi, tzinfo=UTC),
        height=height,
        kind="high" if height >= 0 else "low",
    )


def _local_days(days, tz):
    return {
        day.date: [e.time.astimezone(tz).strftime("%H:%M") for e in day.events] for day in days
    }


class TestGroupOnDisplayClock:
    # Guajiru 2026-09-24/25 (issue #10): the 21:29 local low is 00:29Z on the 25th.
    EVENTS = [
        _ev(2026, 9, 24, 5, 59),
        _ev(2026, 9, 24, 12, 15),
        _ev(2026, 9, 24, 18, 27),
        _ev(2026, 9, 25, 0, 29),
        _ev(2026, 9, 25, 6, 34),
        _ev(2026, 9, 25, 12, 47),
        _ev(2026, 9, 25, 18, 58),
        _ev(2026, 9, 26, 1, 3),
    ]

    def test_negative_offset(self):
        days = _group_events_by_date(
            self.EVENTS, datetime.date(2026, 9, 24), datetime.date(2026, 9, 25), FORTALEZA
        )
        got = _local_days(days, FORTALEZA)
        assert got[datetime.date(2026, 9, 24)] == ["02:59", "09:15", "15:27", "21:29"]
        assert got[datetime.date(2026, 9, 25)] == ["03:34", "09:47", "15:58", "22:03"]

    def test_positive_offset(self):
        # NZST is UTC+12 until DST starts at 02:00 local on the 27th, so
        # 12:30Z on the 26th is 00:30 local on the 27th (the mirror case).
        events = [_ev(2026, 9, 26, 12, 30), _ev(2026, 9, 26, 10, 30)]
        days = _group_events_by_date(
            events, datetime.date(2026, 9, 26), datetime.date(2026, 9, 27), AUCKLAND
        )
        got = _local_days(days, AUCKLAND)
        assert got[datetime.date(2026, 9, 26)] == ["22:30"]
        assert got[datetime.date(2026, 9, 27)] == ["00:30"]

    def test_utc_unchanged(self):
        days = _group_events_by_date(
            self.EVENTS, datetime.date(2026, 9, 24), datetime.date(2026, 9, 25)
        )
        assert [len(d.events) for d in days] == [3, 4]
        assert days[1].events[0].time == datetime.datetime(2026, 9, 25, 0, 29, tzinfo=UTC)

    def test_events_ascend_within_day(self):
        days = _group_events_by_date(
            list(reversed(self.EVENTS)),
            datetime.date(2026, 9, 24),
            datetime.date(2026, 9, 25),
            FORTALEZA,
        )
        for day in days:
            times = [e.time for e in day.events]
            assert times == sorted(times)


class TestFetchWindow:
    def test_widened_only_for_local(self):
        b, e = datetime.date(2026, 9, 24), datetime.date(2026, 9, 25)
        assert _fetch_dates(b, e, None) == (b, e)
        assert _fetch_dates(b, e, FORTALEZA) == (
            datetime.date(2026, 9, 23),
            datetime.date(2026, 9, 26),
        )

    @patch("tides.resolver.compute_tides")
    def test_model_path_widens_and_trims(self, mock_compute):
        mock_compute.return_value = TestGroupOnDisplayClock.EVENTS
        result = resolve_tides(
            Coordinate(lat=-2.881, lon=-39.908),
            datetime.date(2026, 9, 24),
            datetime.date(2026, 9, 24),
            Source.MODEL,
            datum="msl",
            tz=FORTALEZA,
        )
        args = mock_compute.call_args.args
        assert (args[1], args[2]) == (datetime.date(2026, 9, 23), datetime.date(2026, 9, 25))
        assert len(result.days) == 1
        assert len(result.days[0].events) == 4


class TestDefaultDate:
    class _Frozen(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2026, 9, 28, 1, 30, tzinfo=UTC).astimezone(tz)

    def test_local_default_follows_coordinate_clock(self, monkeypatch):
        monkeypatch.setattr(cli.datetime, "datetime", self._Frozen)
        coord = Coordinate(lat=-2.881, lon=-39.908)
        assert cli.parse_date_arg(None, coord, local=True) == (
            datetime.date(2026, 9, 27),
            datetime.date(2026, 9, 27),
        )
        assert cli.parse_date_arg(None, coord, local=False) == (
            datetime.date(2026, 9, 28),
            datetime.date(2026, 9, 28),
        )


class TestContinuousStationPrediction:
    def test_extremum_at_utc_midnight_is_kept(self):
        """A high exactly at 2026-06-15 00:00Z (a day boundary) must survive."""
        midnight = datetime.datetime(2026, 6, 15, tzinfo=UTC)
        t_mid, _ = utc_minutes(midnight, midnight + datetime.timedelta(minutes=1), 1)
        period_days = 12.4206012 / 24

        def fake(constituents, t, chunks=1):
            return np.cos(2 * np.pi * (np.asarray(t) - t_mid[0]) / period_days)

        station = {
            "datums": {},
            "chart_datum": "MSL",
            "harmonic_constituents": [{"name": "M2", "amplitude": 1.0, "phase": 0.0}],
        }
        with patch("tides.harmonics.predict_elevations", side_effect=fake):
            events = predict_station_tides(
                station, datetime.date(2026, 6, 14), datetime.date(2026, 6, 15)
            )
        assert midnight in [e.time for e in events]
        assert all(
            datetime.datetime(2026, 6, 14, tzinfo=UTC)
            <= e.time
            < midnight + datetime.timedelta(days=1)
            for e in events
        )
