"""Water height as a function of time, shared by `level` and `when`.

A HeightCurve holds 1-minute height samples (metres, already in the requested
datum) for one resolved source over a padded window, plus the central
finite-difference rate. NOAA's 6-minute series is linearly interpolated onto
the same 1-minute grid before it gets here.
"""

import bisect
import datetime
import math
from dataclasses import dataclass, field

import numpy as np

from tides.models import TidePoint
from tides.ocean_model import ELEVATION_INTERVAL_MINUTES, find_extrema

# A point (or a `when` level) within this many metres of an adjacent turn's
# height is reported as that turn (about 0.1 ft).
TURN_TOLERANCE_M = 0.03

_STEP = datetime.timedelta(minutes=ELEVATION_INTERVAL_MINUTES)
_STEP_HOURS = ELEVATION_INTERVAL_MINUTES / 60.0


class HeightCurve:
    """Heights on a uniform 1-minute UTC grid starting at `start`."""

    def __init__(self, start: datetime.datetime, heights: np.ndarray) -> None:
        self.start = start
        self.heights = np.asarray(heights, dtype=float)
        n = len(self.heights)
        self.rates = np.gradient(self.heights, _STEP_HOURS) if n > 1 else np.zeros(n)
        self._turns: list[tuple[int, str]] | None = None
        self._tol_windows: dict[int, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.heights)

    def all_nan(self) -> bool:
        return len(self.heights) == 0 or bool(np.all(np.isnan(self.heights)))

    # -- grid helpers -------------------------------------------------------

    def _pos(self, t: datetime.datetime) -> float:
        return (t - self.start) / _STEP

    def _time(self, pos: float) -> datetime.datetime:
        return self.start + _STEP * pos

    def _interp(self, arr: np.ndarray, pos: float) -> float:
        i = min(max(int(math.floor(pos)), 0), len(arr) - 1)
        frac = pos - i
        if frac == 0.0 or i + 1 >= len(arr):
            return float(arr[i])
        return float(arr[i] + (arr[i + 1] - arr[i]) * frac)

    def _index_range(self, start: datetime.datetime, end: datetime.datetime) -> tuple[int, int]:
        """Sample indices [i0, i1) whose times lie in [start, end)."""
        n = len(self.heights)
        i0 = min(max(math.ceil(self._pos(start)), 0), n)
        i1 = min(max(math.ceil(self._pos(end)), 0), n)
        return i0, i1

    def _window(self, idx: int, ref: float) -> tuple[int, int]:
        """Contiguous sample range [lo, hi] around idx with |h - ref| <= tol."""
        h = self.heights
        lo = hi = idx
        while lo > 0 and abs(h[lo - 1] - ref) <= TURN_TOLERANCE_M:
            lo -= 1
        while hi < len(h) - 1 and abs(h[hi + 1] - ref) <= TURN_TOLERANCE_M:
            hi += 1
        return lo, hi

    def _edge(self, inside: int, outside: int, ref: float) -> float:
        """Fractional position between two samples where |h - ref| reaches
        the tolerance (the continuous edge of a window)."""
        d_in = self.heights[inside] - ref
        d_out = self.heights[outside] - ref
        if d_out == d_in:
            return float(inside)
        s = (math.copysign(TURN_TOLERANCE_M, d_out) - d_in) / (d_out - d_in)
        s = min(max(s, 0.0), 1.0)
        return inside + (outside - inside) * s

    def _continuous_window(self, idx: int, ref: float) -> tuple[float, float]:
        lo, hi = self._window(idx, ref)
        left = self._edge(lo, lo - 1, ref) if lo > 0 else float(lo)
        right = self._edge(hi, hi + 1, ref) if hi < len(self.heights) - 1 else float(hi)
        return left, right

    # -- turns --------------------------------------------------------------

    def _turn_list(self) -> list[tuple[int, str]]:
        """(index, "high"|"low") for every turn on the curve, by time."""
        if self._turns is None:
            times = [self._time(i) for i in range(len(self.heights))]
            self._turns = [
                (round(self._pos(e.time)), e.kind) for e in find_extrema(times, self.heights)
            ]
        return self._turns

    def _tol_window(self, idx: int) -> tuple[int, int]:
        if idx not in self._tol_windows:
            self._tol_windows[idx] = self._window(idx, float(self.heights[idx]))
        return self._tol_windows[idx]

    def turns(self, start: datetime.datetime, end: datetime.datetime) -> list[TidePoint]:
        """Highs and lows in [start, end)."""
        i0, i1 = self._index_range(start, end)
        return [
            TidePoint(time=self._time(i), height=float(self.heights[i]), kind=kind, rate=0.0)
            for i, kind in self._turn_list()
            if i0 <= i < i1
        ]

    def _label(self, pos: float, height: float, rate: float) -> str:
        """high/low inside an adjacent turn's tolerance window, else by rate."""
        turns = self._turn_list()
        idxs = [i for i, _ in turns]
        k = bisect.bisect_left(idxs, pos)
        adjacent = {j for j in (k - 1, k) if 0 <= j < len(turns)}
        if k < len(turns) and idxs[k] == pos:
            adjacent = {k}
        best: tuple[float, float, str] | None = None
        for j in adjacent:
            idx, kind = turns[j]
            lo, hi = self._tol_window(idx)
            turn_h = float(self.heights[idx])
            if lo <= pos <= hi and abs(height - turn_h) <= TURN_TOLERANCE_M:
                key = (abs(height - turn_h), abs(pos - idx), kind)
                if best is None or key[:2] < best[:2]:
                    best = key
        if best is not None:
            return best[2]
        return "rising" if rate >= 0 else "falling"

    # -- public operations --------------------------------------------------

    def at(self, times: list[datetime.datetime]) -> list[TidePoint]:
        """Height, rate and label at each aware datetime."""
        points = []
        last = len(self.heights) - 1
        for t in times:
            pos = self._pos(t)
            if not 0 <= pos <= last:
                raise ValueError(f"{t.isoformat()} is outside the height curve")
            h = self._interp(self.heights, pos)
            r = self._interp(self.rates, pos)
            kind = self._label(pos, h, r)
            rate = 0.0 if kind in ("high", "low") else r
            points.append(TidePoint(time=t, height=h, kind=kind, rate=rate))
        return points

    def crossings(
        self,
        level_m: float,
        start: datetime.datetime,
        end: datetime.datetime,
        direction: str | None = None,
    ) -> list[TidePoint]:
        """Times in [start, end) where the height equals level_m.

        Sign changes of h - level between 1-minute samples, refined by linear
        interpolation. An exact hit at a sample counts once, for the interval
        that starts there. Typed rising/falling by the sign change (an exact
        hit by the next sample). `direction` keeps only "rising"/"falling".
        """
        n = len(self.heights)
        i0, i1 = self._index_range(start, end)
        lo = max(i0 - 1, 0)
        hi = min(i1 + 1, n)
        d = self.heights[lo:hi] - level_m
        if len(d) < 2:
            return []
        # An exact hit counts only where a run of samples at the level begins
        # (a flat run such as [-1, 0, 0, 1] is one crossing, not two).
        prev = np.empty_like(d)
        prev[0] = self.heights[lo - 1] - level_m if lo > 0 else np.nan
        prev[1:] = d[:-1]
        starts_run = (d == 0) & (prev != 0)
        hits = np.nonzero(starts_run[:-1] | (d[:-1] * d[1:] < 0))[0]
        points = []
        for k in hits:
            i = lo + int(k)
            if d[k] == 0:
                pos = float(i)
                # Typed by the first sample after the run that leaves the level.
                after = d[k + 1 :][d[k + 1 :] != 0]
                rising = after[0] > 0 if len(after) else self.rates[i] >= 0
            else:
                pos = i + float(d[k] / (d[k] - d[k + 1]))
                rising = d[k + 1] > d[k]
            t = self._time(pos)
            if not start <= t < end:
                continue
            kind = "rising" if rising else "falling"
            if direction is not None and kind != direction:
                continue
            points.append(
                TidePoint(time=t, height=level_m, kind=kind, rate=self._interp(self.rates, pos))
            )
        return points

    def near_turns(
        self, level_m: float, start: datetime.datetime, end: datetime.datetime
    ) -> list[TidePoint]:
        """Turns in [start, end) within TURN_TOLERANCE_M of level_m, each with
        its near window (where |h - level| <= tolerance; not clipped)."""
        points = []
        for turn in self.turns(start, end):
            if abs(turn.height - level_m) <= TURN_TOLERANCE_M:
                points.append(self.with_near(turn, level_m))
        return points

    def with_near(self, turn: TidePoint, level_m: float) -> TidePoint:
        """Copy of a turn with the near window for level_m attached."""
        left, right = self._continuous_window(round(self._pos(turn.time)), level_m)
        return TidePoint(
            time=turn.time,
            height=turn.height,
            kind=turn.kind,
            rate=0.0,
            near=(self._time(left), self._time(right)),
        )

    def extremes(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> tuple[TidePoint, TidePoint] | None:
        """(min, max) samples in [start, end), or None when empty/all-NaN."""
        i0, i1 = self._index_range(start, end)
        seg = self.heights[i0:i1]
        if len(seg) == 0 or np.all(np.isnan(seg)):
            return None
        imin = i0 + int(np.nanargmin(seg))
        imax = i0 + int(np.nanargmax(seg))
        return (
            TidePoint(time=self._time(imin), height=float(self.heights[imin]), kind="low"),
            TidePoint(time=self._time(imax), height=float(self.heights[imax]), kind="high"),
        )


@dataclass
class LevelSearch:
    """Result of searching one day's window for a level.

    `rows` are what to report. When empty: `reached` says whether the level
    occurs at all, and `nearest` is the closest extreme when it does not.
    """

    rows: list[TidePoint] = field(default_factory=list)
    reached: bool = False
    nearest: TidePoint | None = None


def search_level(
    curve: HeightCurve,
    level_m: float,
    segments: list[tuple[datetime.datetime, datetime.datetime]],
    direction: str | None = None,
) -> LevelSearch:
    """Crossings of level_m within the union of `segments`.

    Turns within TURN_TOLERANCE_M of the level are reported as one near-turn
    row instead of crossings; crossings inside that turn's near window are
    dropped. Near-turn rows ignore `direction`.
    """
    near: list[TidePoint] = []
    crossings: list[TidePoint] = []
    for s, e in segments:
        near.extend(curve.near_turns(level_m, s, e))
        crossings.extend(curve.crossings(level_m, s, e))

    def in_near(t: datetime.datetime) -> bool:
        return any(p.near is not None and p.near[0] <= t <= p.near[1] for p in near)

    crossings = [c for c in crossings if not in_near(c.time)]
    reached = bool(near or crossings)
    kept = [c for c in crossings if direction is None or c.kind == direction]
    rows = sorted(near + kept, key=lambda p: p.time)

    nearest = None
    if not reached:
        below: list[TidePoint] = []  # segment maxima under the level
        above: list[TidePoint] = []  # segment minima over the level
        for s, e in segments:
            ext = curve.extremes(s, e)
            if ext is None:
                continue
            lo, hi = ext
            if hi.height < level_m:
                below.append(hi)
            if lo.height > level_m:
                above.append(lo)
        best_below = max(below, key=lambda p: p.height, default=None)
        best_above = min(above, key=lambda p: p.height, default=None)
        candidates = [p for p in (best_below, best_above) if p is not None]
        if candidates:
            nearest = min(candidates, key=lambda p: abs(p.height - level_m))
    return LevelSearch(rows=rows, reached=reached, nearest=nearest)


def turns_of_kind(
    curve: HeightCurve,
    kind: str,
    level_m: float,
    segments: list[tuple[datetime.datetime, datetime.datetime]],
) -> list[TidePoint]:
    """Every turn of `kind` within `segments` (for `--level now` at a turn).

    A turn gets its near window only when level_m is within tolerance of it.
    """
    rows = []
    for s, e in segments:
        for turn in curve.turns(s, e):
            if turn.kind != kind:
                continue
            if abs(turn.height - level_m) <= TURN_TOLERANCE_M:
                turn = curve.with_near(turn, level_m)
            rows.append(turn)
    return sorted(rows, key=lambda p: p.time)
