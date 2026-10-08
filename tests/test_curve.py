"""HeightCurve: crossings, turns, labeling and level search (#31)."""

import datetime

import numpy as np
import pytest

from tides.curve import TURN_TOLERANCE_M, HeightCurve, search_level, turns_of_kind

UTC = datetime.timezone.utc
START = datetime.datetime(2026, 10, 8, tzinfo=UTC)
MIN = datetime.timedelta(minutes=1)
PERIOD_MIN = 12.42 * 60


def _cosine(amplitude: float = 1.0, minutes: int = 3 * 1440) -> HeightCurve:
    """A pure semidiurnal tide: high at START, low half a period later."""
    m = np.arange(minutes)
    return HeightCurve(START, amplitude * np.cos(2 * np.pi * m / PERIOD_MIN))


def _at(minutes: float) -> datetime.datetime:
    return START + MIN * minutes


class TestAt:
    def test_rate_matches_derivative(self):
        curve = _cosine()
        p = curve.at([_at(PERIOD_MIN / 4 + 0.0)])[0]
        # d/dt cos at a quarter period = -2*pi/T per hour.
        assert p.rate == pytest.approx(-2 * np.pi / (PERIOD_MIN / 60), rel=1e-3)
        assert p.kind == "falling"

    def test_rising_after_low(self):
        p = _cosine().at([_at(PERIOD_MIN * 0.75)])[0]
        assert p.kind == "rising"
        assert p.rate > 0

    def test_turn_labeled_with_zero_rate(self):
        curve = _cosine()
        high = curve.turns(_at(60), _at(2000))[0]
        p = curve.at([high.time])[0]
        assert p.kind == high.kind
        assert p.rate == 0.0
        assert p.height == pytest.approx(high.height)

    def test_inside_tolerance_window_is_turn(self):
        curve = _cosine()
        low = next(t for t in curve.turns(_at(60), _at(2000)) if t.kind == "low")
        # A few minutes off a 1 m-amplitude turn is well within 3 cm.
        p = curve.at([low.time + 5 * MIN])[0]
        assert p.kind == "low"
        # The point keeps its own height, not the turn's.
        assert p.height != low.height

    def test_outside_curve_raises(self):
        with pytest.raises(ValueError):
            _cosine(minutes=100).at([_at(500)])

    def test_both_turns_qualify_nearer_height_wins(self):
        # Tidal range under 2 * tolerance: every point is within tolerance
        # of both adjacent turns.
        curve = _cosine(amplitude=0.01)
        turns = curve.turns(_at(60), _at(2000))
        # A point just after a high is nearer in height to that high.
        first_high = next(t for t in turns if t.kind == "high")
        assert curve.at([first_high.time + 30 * MIN])[0].kind == "high"
        first_low = next(t for t in turns if t.kind == "low")
        assert curve.at([first_low.time - 30 * MIN])[0].kind == "low"


class TestCrossings:
    def test_round_trip(self):
        curve = _cosine()
        for level in (-0.7, 0.0, 0.33, 0.9):
            pts = curve.crossings(level, _at(180), _at(2 * 1440))
            assert pts
            for p in pts:
                assert abs(curve.at([p.time])[0].height - level) < 0.005

    def test_types_alternate(self):
        pts = _cosine().crossings(0.0, _at(180), _at(1440))
        kinds = [p.kind for p in pts]
        assert all(a != b for a, b in zip(kinds, kinds[1:]))
        assert {p.kind for p in pts} == {"rising", "falling"}

    def test_direction_filter(self):
        curve = _cosine()
        pts = curve.crossings(0.2, _at(180), _at(1440), direction="rising")
        assert pts and all(p.kind == "rising" for p in pts)

    def test_exact_hit_reported_once(self):
        heights = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
        curve = HeightCurve(START, heights)
        pts = curve.crossings(0.0, START, _at(8))
        assert [(p.time, p.kind) for p in pts] == [(_at(2), "rising"), (_at(6), "falling")]

    def test_half_open_window(self):
        heights = np.array([-1.0, 1.0, 3.0])
        curve = HeightCurve(START, heights)
        # Crossing at 00:00:30.
        assert len(curve.crossings(0.0, START, _at(0.5))) == 0
        assert len(curve.crossings(0.0, _at(0.5), _at(2))) == 1


class TestSearchLevel:
    def test_near_turn_replaces_crossings_and_ignores_direction(self):
        curve = _cosine()
        high = next(t for t in curve.turns(_at(180), _at(1440)) if t.kind == "high")
        level = high.height - TURN_TOLERANCE_M / 2
        seg = [(high.time - 60 * MIN, high.time + 60 * MIN)]
        for direction in (None, "rising", "falling"):
            found = search_level(curve, level, seg, direction)
            assert len(found.rows) == 1
            row = found.rows[0]
            assert row.kind == "high"
            assert row.time == high.time
            assert row.near[0] < high.time < row.near[1]
            # Near window is where |h - level| <= tolerance.
            for edge in row.near:
                h = curve.at([edge])[0].height
                assert abs(h - level) == pytest.approx(TURN_TOLERANCE_M, abs=1e-3)

    def test_near_turn_above_max_takes_precedence(self):
        curve = _cosine()
        high = next(t for t in curve.turns(_at(180), _at(1440)) if t.kind == "high")
        found = search_level(curve, high.height + 0.01, [(_at(180), _at(1440))])
        assert [r.kind for r in found.rows] == ["high"]

    def test_not_reached_reports_max_and_min(self):
        curve = _cosine()
        seg = [(_at(180), _at(1440))]
        above = search_level(curve, 1.5, seg)
        assert above.rows == [] and not above.reached
        assert above.nearest.height == pytest.approx(1.0, abs=1e-3)
        assert above.nearest.height < 1.5
        below = search_level(curve, -1.5, seg)
        assert below.nearest.height == pytest.approx(-1.0, abs=1e-3)

    def test_max_at_window_edge_is_not_a_turn(self):
        curve = _cosine()
        # Window ending just before the first high after 180 min: max sits
        # at the edge, so a level just above it is "not reached".
        high = next(t for t in curve.turns(_at(180), _at(1440)) if t.kind == "high")
        seg = [(high.time - 120 * MIN, high.time - 60 * MIN)]
        edge_max = curve.extremes(*seg[0])[1]
        found = search_level(curve, edge_max.height + 0.01, seg)
        assert found.rows == [] and not found.reached

    def test_reached_but_filtered(self):
        curve = _cosine()
        seg = [(_at(PERIOD_MIN * 0.1), _at(PERIOD_MIN * 0.4))]  # falling only
        found = search_level(curve, 0.0, seg, "rising")
        assert found.rows == [] and found.reached

    def test_turns_of_kind(self):
        curve = _cosine()
        seg = [(_at(180), _at(2 * 1440))]
        highs = turns_of_kind(curve, "high", 0.0, seg)
        assert highs and all(r.kind == "high" and r.near is None for r in highs)
        near = turns_of_kind(curve, "high", highs[0].height, seg)
        assert near[0].near is not None
