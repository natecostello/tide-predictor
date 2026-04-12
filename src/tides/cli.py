import datetime
import json
import sys
from typing import Optional

import httpx
import typer

from tides import __version__
from tides.models import Coordinate, Source, TideResult

app = typer.Typer(
    name="tides",
    help=(
        "Tide predictions for any coastal coordinate.\n\n"
        "Examples:\n\n"
        "  tides 40.7128,-74.0060\n\n"
        "  tides 40.7128,-74.0060 --date 2026-04-15\n\n"
        "  tides 35.9,-75.6 --local --feet\n\n"
        "  tides -- 40.7128 -74.0060  (use -- with space-separated negative lon)\n\n"
        "https://github.com/natecostello/tide-predictor"
    ),
    add_completion=False,
    invoke_without_command=True,
    context_settings={"allow_interspersed_args": True},
)


def parse_coordinate(args: list[str]) -> Coordinate:
    if not args:
        print(
            "Error: Could not parse coordinates. Expected: lat,lon (e.g. 40.7128,-74.0060)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Join all args and try to split on comma
    joined = " ".join(args)
    joined = joined.replace(" ,", ",").replace(", ", ",")

    if "," in joined:
        parts = joined.split(",")
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

    # Try as two separate float args
    if len(args) == 2:
        try:
            lat, lon = float(args[0]), float(args[1])
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


def parse_date_arg(date_str: str | None) -> tuple[datetime.date, datetime.date]:
    if date_str is None:
        today = datetime.datetime.now(tz=datetime.timezone.utc).date()
        return today, today

    if ":" in date_str:
        parts = date_str.split(":")
        if len(parts) != 2:
            print(
                "Error: Invalid date format. Expected: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            begin = datetime.date.fromisoformat(parts[0])
            end = datetime.date.fromisoformat(parts[1])
        except ValueError:
            print(
                "Error: Invalid date format. Expected: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if end < begin:
            print("Error: End date must not be before begin date.", file=sys.stderr)
            raise SystemExit(1)
        return begin, end

    try:
        d = datetime.date.fromisoformat(date_str)
        return d, d
    except ValueError:
        print(
            "Error: Invalid date format. Expected: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD",
            file=sys.stderr,
        )
        raise SystemExit(1)


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
    if end < start:
        print("Error: --between end time must not be before start time.", file=sys.stderr)
        raise SystemExit(1)
    return start, end


def format_plain(
    result: TideResult,
    feet: bool,
    precision: int,
    local: bool,
    between: tuple[datetime.time, datetime.time] | None,
    verbose: bool,
) -> str:
    from tides.timezone import to_local_time

    lines = []
    multi_day = len(result.days) > 1
    unit = "ft" if feet else "m"

    verbose_prefix = ""
    if verbose and result.source_type == Source.NOAA and result.station_name:
        verbose_prefix = f"[NOAA: {result.station_name}, {result.station_distance_km}km] "
    elif verbose and result.source_type == Source.MODEL and result.model_name:
        verbose_prefix = f"[Model: {result.model_name}] "

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
                if not (between[0] <= t <= between[1]):
                    continue

            height = event.height_ft if feet else event.height
            height_str = f"{height:.{precision}f}{unit}"
            event_strs.append(f"{height_str}@{time_str}")

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
    if result.source_type == Source.NOAA and result.station_id:
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
                if not (between[0] <= t <= between[1]):
                    continue

            height = event.height_ft if feet else event.height
            tides_list.append(
                {
                    "time": time_str,
                    "height": round(height, precision),
                }
            )

        if tides_list:
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
        "timezone": tz_name,
        "unit": unit,
        "days": days_list,
    }
    return json.dumps(output, indent=2)


@app.callback()
def main(
    ctx: typer.Context,
    coordinate: Optional[list[str]] = typer.Argument(
        None, help="Latitude,longitude (e.g. 40.7128,-74.0060)"
    ),
    date: Optional[str] = typer.Option(
        None, "--date", "-d", help="Date or range: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD"
    ),
    local: bool = typer.Option(False, "--local", "-l", help="Display times in local timezone"),
    feet: bool = typer.Option(False, "--feet", "-f", help="Display heights in feet"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    between: Optional[str] = typer.Option(
        None, "--between", "-b", help="Time filter: HH:MM:HH:MM"
    ),
    precision: int = typer.Option(1, "--precision", "-p", help="Decimal places for height"),
    source: str = typer.Option("auto", "--source", "-s", help="Data source: auto, noaa, model"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show source details"),
    version: bool = typer.Option(False, "--version", help="Show version"),
) -> None:
    """Tide predictions for any coastal coordinate."""
    if version:
        print(f"tides {__version__}")
        raise typer.Exit()

    # If a subcommand is being invoked, let it run
    if ctx.invoked_subcommand is not None:
        return

    if not coordinate:
        print(ctx.get_help())
        raise typer.Exit()

    coord = parse_coordinate(coordinate)
    begin_date, end_date = parse_date_arg(date)
    between_times = parse_between(between)

    if precision < 0:
        print("Error: --precision must be a non-negative integer.", file=sys.stderr)
        raise SystemExit(1)

    try:
        source_enum = Source(source.lower())
    except ValueError:
        print(
            f"Error: Invalid source '{source}'. Expected: auto, noaa, model",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from tides.noaa import NOAAError
    from tides.resolver import resolve_tides

    try:
        result = resolve_tides(coord, begin_date, end_date, source_enum)
    except SystemExit:
        raise
    except NOAAError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)
    except httpx.HTTPStatusError:
        print(
            "Error: Could not fetch tide data from NOAA. The service may be unavailable.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except (httpx.ConnectError, httpx.TimeoutException):
        print(
            "Error: Could not connect to tide data service. Check your internet connection.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception:
        print(
            "Error: An unexpected error occurred while fetching tide data.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if json_output:
        print(format_json(result, feet, precision, local, between_times))
    else:
        output = format_plain(result, feet, precision, local, between_times, verbose)
        if output:
            print(output)


@app.command("fetch-model")
def fetch_model() -> None:
    """Pre-download tidal model data and station metadata."""
    from tides.cache import fetch_all

    try:
        fetch_all()
    except (httpx.ConnectError, httpx.TimeoutException):
        print(
            "Error: Could not connect to data service. Check your internet connection.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except httpx.HTTPStatusError:
        print(
            "Error: Data service returned an error. Please try again later.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception:
        print(
            "Error: Could not download tidal data.",
            file=sys.stderr,
        )
        raise SystemExit(2)
