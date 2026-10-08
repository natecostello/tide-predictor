import datetime
import json
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, TypeVar

import httpx
import typer

from tides import __version__
from tides.datums import SUPPORTED_DATUMS
from tides.models import (
    METERS_TO_FEET,
    Coordinate,
    Source,
    TideDay,
    TideEvent,
    TidePoint,
    TideResult,
)
from tides.ocean_model import DEFAULT_MODEL, SUPPORTED_MODELS

# Valid values, listed in --help. Validated manually (case-insensitive) so an
# invalid value exits 1 (user error), not Click's usage-error exit 2.
_SOURCE_CHOICES = tuple(s.value for s in Source)
_MODEL_CHOICES = tuple(m.lower() for m in sorted(SUPPORTED_MODELS))

# Matches a bare lat,lon (or "lat lon") token whose latitude has a leading '-'.
# Click would otherwise treat the leading '-' as the start of an option flag.
# The float halves accept the same forms parse_coordinate does:
# optional sign, optional integer part (e.g. "-.5"), optional fractional part,
# and optional whitespace around the comma (relevant when the user quoted the
# token to keep it as a single argv entry).
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_NEG_COORD_RE = re.compile(rf"^-(?:\d+(?:\.\d*)?|\.\d+)(?:\s*,\s*|\s+){_FLOAT}$")


def _escape_negative_coords(argv: list[str]) -> list[str]:
    """Prepend a space to bare negative-latitude coordinate tokens.

    Click parses any argv token starting with '-' as an option, so a southern
    coordinate like '-2.88,-39.91' would be rejected as an unknown flag. Adding
    a leading space makes Click treat it as positional; ``parse_coordinate``
    already strips whitespace.
    """
    return [(" " + a) if _NEG_COORD_RE.match(a) else a for a in argv]


# Help on -h/--help everywhere (subcommands inherit context_settings), and on
# bare `tides`. Not applied to cache_app: bare `tides cache` shows cache info.
app = typer.Typer(
    name="tides",
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

cache_app = typer.Typer(
    name="cache",
    help="Manage cached tidal data and model files.",
    add_completion=False,
)
app.add_typer(cache_app, name="cache")


def parse_coordinate(args: list[str]) -> Coordinate:
    # Strip -- separator that Typer may pass through
    args = [a for a in args if a != "--"]
    if not args:
        print(
            "Error: Could not parse coordinates. Expected: lat,lon (e.g. 40.7128,-74.0060)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Join all args and split on a comma, or on whitespace ("lat lon").
    joined = " ".join(args)
    joined = joined.replace(" ,", ",").replace(", ", ",")

    parts = joined.split(",") if "," in joined else joined.split()
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except (ValueError, TypeError):
            pass
        else:
            try:
                return Coordinate(lat=lat, lon=lon)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                raise SystemExit(1)

    print(
        "Error: Could not parse coordinates. Expected: lat,lon (e.g. 40.7128,-74.0060)",
        file=sys.stderr,
    )
    raise SystemExit(1)


# Longest accepted --date range, inclusive. NOAA's own hilo limit is 3,655
# days, so this is well within it; it bounds the 1-minute model arrays.
MAX_DATE_RANGE_DAYS = 366


def _now() -> datetime.datetime:
    """The current UTC time truncated to the minute.

    The single source of "now" and "today" (tests monkeypatch it).
    """
    return datetime.datetime.now(tz=datetime.timezone.utc).replace(second=0, microsecond=0)


def _display_tz(coord: Coordinate | None, local: bool) -> datetime.tzinfo:
    """The display clock: the coordinate's zone with --local, UTC otherwise."""
    if local and coord is not None:
        from tides.timezone import get_zoneinfo

        return get_zoneinfo(coord)
    return datetime.timezone.utc


_DATE_FORMAT_HINT = (
    "Error: Invalid date format. Expected: YYYY-MM-DD, today or tomorrow, "
    "or a range like YYYY-MM-DD:YYYY-MM-DD"
)


def _parse_day(token: str, today: datetime.date) -> datetime.date:
    word = token.strip().lower()
    if word == "today":
        return today
    if word == "tomorrow":
        return today + datetime.timedelta(days=1)
    try:
        return datetime.date.fromisoformat(token)
    except ValueError:
        print(_DATE_FORMAT_HINT, file=sys.stderr)
        raise SystemExit(1)


def parse_date_arg(
    date_str: str | None,
    coord: Coordinate | None = None,
    local: bool = False,
) -> tuple[datetime.date, datetime.date]:
    # "Today" follows the display clock: local date at the coordinate with
    # --local, UTC otherwise.
    today = _now().astimezone(_display_tz(coord, local)).date()
    if date_str is None:
        return today, today

    if ":" in date_str:
        parts = date_str.split(":")
        if len(parts) != 2:
            print(_DATE_FORMAT_HINT, file=sys.stderr)
            raise SystemExit(1)
        begin = _parse_day(parts[0], today)
        end = _parse_day(parts[1], today)
        if end < begin:
            print("Error: End date must not be before begin date.", file=sys.stderr)
            raise SystemExit(1)
        n_days = (end - begin).days + 1
        if n_days > MAX_DATE_RANGE_DAYS:
            print(
                f"Error: date range too long ({n_days} days); maximum is {MAX_DATE_RANGE_DAYS}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return begin, end

    d = _parse_day(date_str, today)
    return d, d


_WHEN_RE = re.compile(r"^(?:(\d{4}-\d{2}-\d{2})T)?(\d{1,2}):(\d{2})$")


def parse_when(values: list[str] | None, tz: datetime.tzinfo) -> list[datetime.datetime]:
    """--when values as UTC datetimes, read on the display clock `tz`.

    Accepts now, HH:MM (today on the display clock) and YYYY-MM-DDTHH:MM.
    Omitted means now.
    """
    now = _now()
    if not values:
        return [now]
    today = now.astimezone(tz).date()
    out = []
    for raw in values:
        value = raw.strip()
        if value.lower() == "now":
            out.append(now)
            continue
        m = _WHEN_RE.match(value)
        try:
            if m is None:
                raise ValueError(value)
            day = datetime.date.fromisoformat(m.group(1)) if m.group(1) else today
            clock = datetime.time(int(m.group(2)), int(m.group(3)))
        except ValueError:
            print(
                f"Error: Invalid --when '{raw}'. Expected: now, HH:MM or YYYY-MM-DDTHH:MM "
                "(e.g. 16:00 or 2026-10-08T16:00)",
                file=sys.stderr,
            )
            raise SystemExit(1)
        local_dt = datetime.datetime.combine(day, clock, tzinfo=tz)
        out.append(local_dt.astimezone(datetime.timezone.utc))
    return out


def parse_level(value: str | None, feet: bool) -> float | str:
    """--level as metres, or the keyword "now". Validated by hand so a missing
    or bad value exits 1 (user error) with a hint."""
    if value is None:
        print(
            "Error: --level is required. Expected: a height or now "
            "(e.g. tides when 40.7,-74.0 --level 1.5)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if value.strip().lower() == "now":
        return "now"
    try:
        level = float(value)
    except ValueError:
        level = float("nan")
    if not math.isfinite(level):
        print(
            f"Error: Invalid --level '{value}'. Expected: a height (e.g. 1.5 or -0.2) or now",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return level / METERS_TO_FEET if feet else level


def parse_between(between_str: str | None) -> tuple[datetime.time, datetime.time] | None:
    if between_str is None:
        return None
    parts = between_str.split(":")
    if len(parts) != 4:
        print(
            "Error: Invalid --between format. Expected: HH:MM:HH:MM (e.g. 06:00:18:00)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        start = datetime.time(int(parts[0]), int(parts[1]))
        end = datetime.time(int(parts[2]), int(parts[3]))
    except (ValueError, TypeError):
        print(
            "Error: Invalid --between format. Expected: HH:MM:HH:MM (e.g. 06:00:18:00)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    # end < start is a window that wraps midnight (e.g. 20:00:04:00).
    return start, end


def _in_window(t: datetime.time, between: tuple[datetime.time, datetime.time]) -> bool:
    """True when t is inside the --between window (which may wrap midnight)."""
    start, end = between
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end


def _url_host(exc: Exception) -> str:
    """Host an httpx error was talking to, for accurate error messages."""
    try:
        return exc.request.url.host or "the data service"
    except (AttributeError, RuntimeError):
        return "the data service"


def _report_unexpected(exc: Exception, context: str) -> None:
    """Catch-all reporting: exception type and message; traceback with
    TIDES_DEBUG=1. Keeps real bugs diagnosable without raw tracebacks by default."""
    import os
    import traceback

    print(f"Error: unexpected {type(exc).__name__} {context}: {exc}", file=sys.stderr)
    if os.environ.get("TIDES_DEBUG") == "1":
        traceback.print_exception(exc, file=sys.stderr)
    else:
        print("Set TIDES_DEBUG=1 for a traceback.", file=sys.stderr)


def _rounded(height: float, precision: int) -> float:
    """Round for display, normalizing negative zero (-0.0 -> 0.0)."""
    return round(height, precision) + 0.0


def _finite_height(height: float) -> float:
    """Refuse to render NaN/inf heights (they would print as 'nan' or produce
    invalid JSON)."""
    import math

    from tides.datums import DatumUnavailableError

    if not math.isfinite(height):
        raise DatumUnavailableError("non-finite height in output")
    return height


def _plain_row(
    event: TideEvent,
    feet: bool,
    precision: int,
    unit: str,
    time_str: str,
    typed: bool,
    coord: Coordinate,
    local: bool,
) -> str:
    height = _rounded(_finite_height(event.height_ft if feet else event.height), precision)
    row = f"{height:.{precision}f}{unit}@{time_str}"
    if not typed:
        return row
    row += f" {event.kind}"
    if event.kind in ("rising", "falling"):
        rate = _rounded(_finite_height(event.rate_ft if feet else event.rate), precision)
        row += f" {rate:+.{precision}f}{unit}/h"
    near = getattr(event, "near", None)
    if near is not None:
        lo, hi = (_display_time(t, coord, local).strftime("%H:%M") for t in near)
        row += f" (near: {lo}-{hi})"
    return row


def _display_time(t: datetime.datetime, coord: Coordinate, local: bool) -> datetime.datetime:
    from tides.timezone import to_local_time

    return to_local_time(t, coord) if local else t


def format_plain(
    result: TideResult,
    feet: bool,
    precision: int,
    local: bool,
    between: tuple[datetime.time, datetime.time] | None,
    verbose: bool,
    typed: bool = False,
) -> str:
    """Plain rows per day. `typed` (level/when) appends the type, the rate
    for rising/falling rows and any near-turn window."""
    from tides.timezone import to_local_time

    lines = []
    multi_day = len(result.days) > 1
    unit = "ft" if feet else "m"

    datum = result.datum.upper()
    verbose_prefix = ""
    if verbose and result.source_type == Source.NOAA and result.station_name:
        verbose_prefix = f"[NOAA: {result.station_name}, {result.station_distance_km}km, {datum}] "
    elif verbose and result.source_type == Source.STATION and result.station_name:
        verbose_prefix = (
            f"[Station: {result.station_name}, {result.station_distance_km}km, {datum}] "
        )
    elif verbose and result.source_type == Source.MODEL and result.model_name:
        verbose_prefix = f"[Model: {result.model_name}, {datum}] "

    for day in result.days:
        event_strs = []
        for event in day.events:
            if local:
                display_time = to_local_time(event.time, result.coordinate)
            else:
                display_time = event.time

            time_str = display_time.strftime("%H:%M")

            if between is not None:
                t = display_time.time().replace(second=0, microsecond=0)
                if not _in_window(t, between):
                    continue

            event_strs.append(
                _plain_row(event, feet, precision, unit, time_str, typed, result.coordinate, local)
            )

        if not event_strs:
            continue

        tide_str = ", ".join(event_strs)
        if multi_day:
            lines.append(f"{day.date}: {verbose_prefix}{tide_str}")
        else:
            lines.append(f"{verbose_prefix}{tide_str}")

    return "\n".join(lines)


def format_json(
    result: TideResult,
    feet: bool,
    precision: int,
    local: bool,
    between: tuple[datetime.time, datetime.time] | None,
) -> str:
    from tides.timezone import get_timezone_name, to_local_time

    unit = "ft" if feet else "m"
    tz_name = "UTC"
    if local:
        tz_name = get_timezone_name(result.coordinate) or "UTC"

    source_obj: dict = {"type": result.source_type.value}
    if result.source_type in (Source.NOAA, Source.STATION) and result.station_id:
        source_obj["station"] = {
            "id": result.station_id,
            "name": result.station_name,
            "distance_km": result.station_distance_km,
        }

    days_list = []
    for day in result.days:
        tides_list = []
        for event in day.events:
            if local:
                display_time = to_local_time(event.time, result.coordinate)
            else:
                display_time = event.time

            time_str = display_time.strftime("%H:%M")

            if between is not None:
                t = display_time.time().replace(second=0, microsecond=0)
                if not _in_window(t, between):
                    continue

            height = _finite_height(event.height_ft if feet else event.height)
            # Plain TideEvents (peaks) are turns: rate 0.0.
            rate = _finite_height(getattr(event, "rate_ft" if feet else "rate", 0.0))
            row = {
                "time": time_str,
                "height": _rounded(height, precision),
                "datetime": display_time.isoformat(timespec="minutes"),
                "type": event.kind,
                "rate": _rounded(rate, precision),
            }
            near = getattr(event, "near", None)
            if near is not None:
                lo, hi = (
                    _display_time(t, result.coordinate, local).isoformat(timespec="minutes")
                    for t in near
                )
                row["near"] = {"from": lo, "to": hi}
            tides_list.append(row)

        # Every requested day is present, even when --between filtered out
        # all of its events, so consumers can tell "filtered" from "absent".
        days_list.append(
            {
                "date": day.date.isoformat(),
                "tides": tides_list,
            }
        )

    output = {
        "coordinate": {"lat": result.coordinate.lat, "lon": result.coordinate.lon},
        "source": source_obj,
        "model": result.model_name,
        "datum": result.datum.upper(),
        "timezone": tz_name,
        "unit": unit,
        "days": days_list,
    }
    return json.dumps(output, indent=2)


# Options shared by peaks, level and when (same names, flags and defaults).
_COORD_ARG = typer.Argument(
    ...,
    help=(
        "Latitude,longitude (e.g. 40.7128,-74.0060). "
        "Negative latitudes are accepted directly (e.g. -2.88,-39.91)."
    ),
)
_DATE_OPT = typer.Option(
    None,
    "--date",
    "-d",
    help="Date or range: YYYY-MM-DD, today, tomorrow, or YYYY-MM-DD:YYYY-MM-DD",
)
_LOCAL_OPT = typer.Option(False, "--local", "-l", help="Display times in local timezone")
_FEET_OPT = typer.Option(False, "--feet", "-f", help="Display heights in feet")
_JSON_OPT = typer.Option(False, "--json", "-j", help="Output as JSON")
_BETWEEN_OPT = typer.Option(
    None,
    "--between",
    "-b",
    help="Time filter HH:MM:HH:MM; a start after the end wraps midnight (20:00:04:00)",
)
_PRECISION_OPT = typer.Option(1, "--precision", "-p", help="Decimal places for height and rate")
_SOURCE_OPT = typer.Option(
    "auto", "--source", "-s", help=f"Data source: {', '.join(_SOURCE_CHOICES)}"
)
_MODEL_OPT = typer.Option(
    None,
    "--model",
    "-m",
    help=f"Tide model: {', '.join(_MODEL_CHOICES)} (default: {DEFAULT_MODEL.lower()})",
)
_DATUM_OPT = typer.Option("mllw", "--datum", help=f"Height datum: {', '.join(SUPPORTED_DATUMS)}")
_VERBOSE_OPT = typer.Option(False, "--verbose", "-v", help="Show source details")


@dataclass
class _Common:
    coord: Coordinate
    source: Source
    model_name: str
    model_explicit: bool
    datum: str


def _validate_common(
    coord: Coordinate, precision: int, datum: str, source: str, model: str | None
) -> _Common:
    if precision < 0:
        print("Error: --precision must be a non-negative integer.", file=sys.stderr)
        raise SystemExit(1)

    datum_lower = datum.lower()
    if datum_lower not in SUPPORTED_DATUMS:
        print(
            f"Error: Invalid datum '{datum}'. Expected: {', '.join(SUPPORTED_DATUMS)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        source_enum = Source(source.lower())
    except ValueError:
        print(
            f"Error: Invalid source '{source}'. Expected: {', '.join(_SOURCE_CHOICES)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    model_name = (model or DEFAULT_MODEL).upper()
    if model_name not in SUPPORTED_MODELS:
        print(
            f"Error: Invalid model '{model}'. Expected: {', '.join(_MODEL_CHOICES)}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return _Common(coord, source_enum, model_name, model is not None, datum_lower)


T = TypeVar("T")


def _guard(fn: Callable[[], T], context: str = "while fetching tide data") -> T:
    """Run a data step, rewriting expected failures as errors with exit 2."""
    from tides.cache import StationDatabaseError
    from tides.datums import DatumUnavailableError
    from tides.noaa import NOAAError

    try:
        return fn()
    except SystemExit:
        raise
    except DatumUnavailableError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)
    except NOAAError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)
    except StationDatabaseError as e:
        print(f"Error: {e}. Check your internet connection.", file=sys.stderr)
        raise SystemExit(2)
    except httpx.HTTPStatusError as e:
        print(
            f"Error: {_url_host(e)} returned HTTP {e.response.status_code}. "
            "The service may be unavailable.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print(
            f"Error: Could not connect to {_url_host(e)}. Check your internet connection.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as e:
        _report_unexpected(e, context)
        raise SystemExit(2)


def _note_model_ignored(common: _Common, result: TideResult) -> None:
    if common.model_explicit and result.source_type != Source.MODEL:
        origin = "NOAA" if result.source_type == Source.NOAA else "station"
        print(
            f"Note: --model {common.model_name} ignored; tides came from {origin} "
            f"'{result.station_name}'. Use --source model to force the model.",
            file=sys.stderr,
        )


def _emit(
    result: TideResult,
    feet: bool,
    precision: int,
    local: bool,
    between: tuple[datetime.time, datetime.time] | None,
    verbose: bool,
    json_output: bool,
    typed: bool,
) -> str:
    from tides.datums import DatumUnavailableError

    try:
        if json_output:
            output = format_json(result, feet, precision, local, between)
        else:
            output = format_plain(result, feet, precision, local, between, verbose, typed)
    except DatumUnavailableError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)
    if output:
        print(output)
    return output


@app.command()
def peaks(
    coordinate: str = _COORD_ARG,
    date: Optional[str] = _DATE_OPT,
    local: bool = _LOCAL_OPT,
    feet: bool = _FEET_OPT,
    json_output: bool = _JSON_OPT,
    between: Optional[str] = _BETWEEN_OPT,
    precision: int = _PRECISION_OPT,
    source: str = _SOURCE_OPT,
    model: Optional[str] = _MODEL_OPT,
    datum: str = _DATUM_OPT,
    verbose: bool = _VERBOSE_OPT,
) -> None:
    """High and low tides for a coastal coordinate."""
    coord = parse_coordinate([coordinate])
    begin_date, end_date = parse_date_arg(date, coord, local)
    between_times = parse_between(between)
    common = _validate_common(coord, precision, datum, source, model)

    from tides.resolver import resolve_tides
    from tides.timezone import get_zoneinfo

    # Day grouping must use the same clock as the displayed times.
    display_tz = get_zoneinfo(coord) if local else None

    result = _guard(
        lambda: resolve_tides(
            coord,
            begin_date,
            end_date,
            common.source,
            model_name=common.model_name,
            datum=common.datum,
            tz=display_tz,
        )
    )
    _note_model_ignored(common, result)

    output = _emit(result, feet, precision, local, between_times, verbose, json_output, False)
    if not output and not json_output:
        # JSON carries the (empty) structure itself; plain output would
        # otherwise be silently empty.
        print("Note: no tide events matched the requested range/filter.", file=sys.stderr)


@app.command()
def level(
    coordinate: str = _COORD_ARG,
    when: Optional[list[str]] = typer.Option(
        None,
        "--when",
        help="Time: now, HH:MM (today) or YYYY-MM-DDTHH:MM, on the display clock. "
        "Repeatable; default now",
    ),
    local: bool = _LOCAL_OPT,
    feet: bool = _FEET_OPT,
    json_output: bool = _JSON_OPT,
    precision: int = _PRECISION_OPT,
    source: str = _SOURCE_OPT,
    model: Optional[str] = _MODEL_OPT,
    datum: str = _DATUM_OPT,
    verbose: bool = _VERBOSE_OPT,
) -> None:
    """Water height (and whether it is rising or falling) at given times."""
    coord = parse_coordinate([coordinate])
    tz = _display_tz(coord, local)
    times = parse_when(when, tz)
    common = _validate_common(coord, precision, datum, source, model)

    # Whole UTC days spanning the requested times.
    start, end = _utc_window(min(times).date(), max(times).date())
    n_days = (end - start).days
    if n_days > MAX_DATE_RANGE_DAYS:
        print(
            f"Error: --when times span {n_days} days; maximum is {MAX_DATE_RANGE_DAYS}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from tides.resolver import resolve_curve

    src, curve = _guard(
        lambda: resolve_curve(coord, start, end, common.source, common.model_name, common.datum)
    )
    points = _guard(lambda: curve.at(times), "while computing heights")

    # One row per --when, grouped by display date (ascending), in the order
    # given within a day.
    by_day: dict[datetime.date, list[TidePoint]] = {}
    for point in points:
        by_day.setdefault(point.time.astimezone(tz).date(), []).append(point)
    result = src.result([TideDay(date=d, events=by_day[d]) for d in sorted(by_day)])
    _note_model_ignored(common, result)
    _emit(result, feet, precision, local, None, verbose, json_output, True)


def _day_segments(
    day: datetime.date,
    tz: datetime.tzinfo,
    between: tuple[datetime.time, datetime.time] | None,
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """The searched window for one display-clock day, as UTC [start, end)
    intervals: the whole day, or the day intersected with --between (whose
    end minute is inclusive, as for peaks)."""
    utc = datetime.timezone.utc

    def at(d: datetime.date, t: datetime.time) -> datetime.datetime:
        return datetime.datetime.combine(d, t, tzinfo=tz).astimezone(utc)

    midnight = datetime.time(0, 0)
    day_start, day_end = at(day, midnight), at(day + datetime.timedelta(days=1), midnight)
    if between is None:
        return [(day_start, day_end)]
    one = datetime.timedelta(minutes=1)
    b_start, b_end = between
    if b_start <= b_end:
        return [(at(day, b_start), min(at(day, b_end) + one, day_end))]
    return [(day_start, at(day, b_end) + one), (at(day, b_start), day_end)]


@app.command(name="when")
def when_cmd(
    coordinate: str = _COORD_ARG,
    level_opt: Optional[str] = typer.Option(
        None,
        "--level",
        help="Height in display units relative to --datum (e.g. 1.5), or now",
    ),
    rising: bool = typer.Option(False, "--rising", help="Only rising crossings"),
    falling: bool = typer.Option(False, "--falling", help="Only falling crossings"),
    date: Optional[str] = _DATE_OPT,
    local: bool = _LOCAL_OPT,
    feet: bool = _FEET_OPT,
    json_output: bool = _JSON_OPT,
    between: Optional[str] = _BETWEEN_OPT,
    precision: int = _PRECISION_OPT,
    source: str = _SOURCE_OPT,
    model: Optional[str] = _MODEL_OPT,
    datum: str = _DATUM_OPT,
    verbose: bool = _VERBOSE_OPT,
) -> None:
    """Times when the water reaches a given level."""
    from tides.curve import search_level, turns_of_kind

    coord = parse_coordinate([coordinate])
    tz = _display_tz(coord, local)
    begin_date, end_date = parse_date_arg(date, coord, local)
    between_times = parse_between(between)
    target = parse_level(level_opt, feet)
    if rising and falling:
        print("Error: --rising and --falling are mutually exclusive.", file=sys.stderr)
        raise SystemExit(1)
    common = _validate_common(coord, precision, datum, source, model)
    direction = "rising" if rising else "falling" if falling else None

    from tides.resolver import _fetch_dates, resolve_curve

    display_tz = tz if local else None
    span_begin, span_end = begin_date, end_date
    now = _now()
    if target == "now":
        # One curve covering now and the window when that fits the cap;
        # otherwise a second curve around now from the same source.
        today = now.astimezone(tz).date()
        lo, hi = min(begin_date, today), max(end_date, today)
        if (hi - lo).days + 1 <= MAX_DATE_RANGE_DAYS:
            span_begin, span_end = lo, hi
    start, end = _utc_window(*_fetch_dates(span_begin, span_end, display_tz))
    src, curve = _guard(
        lambda: resolve_curve(coord, start, end, common.source, common.model_name, common.datum)
    )

    turn_kind = None
    if target == "now":
        now_curve = curve
        if not start <= now < end:
            now_curve = _guard(lambda: src.curve(*_utc_window(now.date(), now.date())))
        now_point = _guard(lambda: now_curve.at([now])[0], "while computing heights")
        level_m = now_point.height  # unrounded
        if direction is None:
            if now_point.kind in ("high", "low"):
                turn_kind = now_point.kind
            else:
                direction = now_point.kind
    else:
        level_m = target

    unit = "ft" if feet else "m"
    scale = METERS_TO_FEET if feet else 1.0

    def fmt(h: float) -> str:
        return f"{_rounded(h * scale, precision):.{precision}f}{unit}"

    days = []
    for d in _date_range(begin_date, end_date):
        segments = _day_segments(d, tz, between_times)
        if turn_kind is not None:
            rows = turns_of_kind(curve, turn_kind, level_m, segments)
            if not rows and not json_output:
                print(f"Note: no {turn_kind} on {d}", file=sys.stderr)
        else:
            found = search_level(curve, level_m, segments, direction)
            rows = found.rows
            if not rows and not json_output:
                if found.reached:
                    print(
                        f"Note: no {direction} crossings of level {fmt(level_m)} on {d}",
                        file=sys.stderr,
                    )
                elif found.nearest is not None:
                    which = "max" if found.nearest.height < level_m else "min"
                    t = _display_time(found.nearest.time, coord, local).strftime("%H:%M")
                    print(
                        f"Note: level {fmt(level_m)} not reached on {d} "
                        f"({which} {fmt(found.nearest.height)}@{t})",
                        file=sys.stderr,
                    )
        days.append(TideDay(date=d, events=rows))

    result = src.result(days)
    _note_model_ignored(common, result)
    _emit(result, feet, precision, local, None, verbose, json_output, True)


def _utc_window(
    begin: datetime.date, end: datetime.date
) -> tuple[datetime.datetime, datetime.datetime]:
    """[begin 00:00Z, end+1 00:00Z) for an inclusive range of UTC dates."""
    from tides.ocean_model import utc_day_window

    return utc_day_window(begin, end)


def _date_range(begin: datetime.date, end: datetime.date) -> list[datetime.date]:
    return [begin + datetime.timedelta(days=i) for i in range((end - begin).days + 1)]


@app.command("fetch-model")
def fetch_model() -> None:
    """Pre-download everything needed offline: always refreshes the NOAA
    station list, and downloads the global station database and the GOT5.6
    model only if not already present."""
    from tides.cache import StationDatabaseError, fetch_all

    try:
        fetch_all()
    except SystemExit:
        raise
    except StationDatabaseError as e:
        print(f"Error: {e}. Check your internet connection.", file=sys.stderr)
        raise SystemExit(2)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print(
            f"Error: Could not connect to {_url_host(e)}. Check your internet connection.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except httpx.HTTPStatusError as e:
        print(
            f"Error: {_url_host(e)} returned HTTP {e.response.status_code}. "
            "Please try again later.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as e:
        _report_unexpected(e, "while downloading tidal data")
        raise SystemExit(2)


@cache_app.callback(invoke_without_command=True)
def cache_show(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Show cache locations and sizes."""
    if ctx.invoked_subcommand is not None:
        return

    from tides.cache import format_size, get_cache_info

    info = get_cache_info()

    if json_output:
        print(json.dumps(info, indent=2))
        return

    app_info = info["app_cache"]
    model_info = info["model_cache"]

    app_total = sum(i["size"] for i in app_info["items"])
    model_total = sum(i["size"] for i in model_info["items"])

    print(f"App cache: {app_info['path']}")
    if app_info["items"]:
        for item in app_info["items"]:
            print(f"  {item['name']:<25} {format_size(item['size']):>10}")
    else:
        print("  (empty)")

    print(f"\nModel cache: {model_info['path']}")
    if model_info["items"]:
        for item in model_info["items"]:
            print(f"  {item['name']:<25} {format_size(item['size']):>10}")
    else:
        print("  (empty)")

    print(f"\nTotal: {format_size(app_total + model_total)}")


@cache_app.command("clear")
def cache_clear(
    name: Optional[str] = typer.Argument(
        None,
        help=(
            "Item to clear: stations, datums, got5.5, got5.6, eot20, fes2022, hamtide11. "
            "Omit to clear the app cache and auto-downloaded models (GOT5.5/GOT5.6)."
        ),
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="With no name, also clear EOT20 and manually downloaded models (FES2022, HAMTIDE11)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Clear cached data.

    Without a name, removes the app cache (station lists, station database,
    datums) and GOT5.5/GOT5.6, which re-download automatically. EOT20, FES2022
    and HAMTIDE11 are only removed when named or with --all.
    """
    from tides.cache import clear_cache, format_size, plan_clear

    try:
        items = plan_clear(name, include_all=all_models)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)

    target = "cached data" if name is None else f"'{name}' cache"
    if not items:
        print(f"Nothing to clear ({target} is empty).")
        raise typer.Exit(code=0)

    print("Will remove:")
    for item in items:
        print(f"  {item['name']:<20} {format_size(item['size']):>10}  {item['path']}")

    if not yes and not typer.confirm(f"Clear {target}?", default=False):
        print("Cancelled.")
        raise typer.Exit(code=0)

    try:
        freed = clear_cache(name, include_all=all_models)
    except OSError as e:
        print(f"Error clearing {target}: {e}", file=sys.stderr)
        raise SystemExit(2)

    print(f"Cleared {target} ({format_size(freed)} freed).")


def version_callback(value: bool) -> None:
    if value:
        print(f"tides {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=version_callback, is_eager=True, help="Show version"
    ),
) -> None:
    """Tide predictions for any coastal coordinate.

    Examples:

      tides peaks 40.7128,-74.0060

      tides peaks 40.7128,-74.0060 --date tomorrow

      tides peaks 35.9,-75.6 --local --feet

      tides level 35.9,-75.6 --when 16:00 --local --feet

      tides when 35.9,-75.6 --level now --date tomorrow --local

    Negative-latitude coordinates may be passed directly, for example:

      tides peaks -2.88,-39.91 --feet

    https://github.com/natecostello/tide-predictor
    """


def main_entry() -> None:
    """Console-script entry point.

    Rewrites ``sys.argv`` so a bare negative-latitude coordinate token
    (e.g. ``-2.88,-39.91``) is not parsed as an option flag, then invokes
    the Typer app.
    """
    sys.argv = [sys.argv[0], *_escape_negative_coords(sys.argv[1:])]
    app()
