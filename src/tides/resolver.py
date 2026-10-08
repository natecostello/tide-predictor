import datetime
import sys
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import numpy as np

from tides.cache import StationDatabaseError, get_stations
from tides.curve import HeightCurve
from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.noaa import (
    DERIVED_DATUMS,
    NATIVE_DATUMS,
    NOAADatumUnavailableError,
    NOAAError,
    NOAASubordinateError,
    fetch_predictions,
    fetch_station_datums,
    parse_predictions_response,
    parse_series_response,
)
from tides.noaa import (
    find_nearest_station as find_nearest_noaa_station,
)
from tides.ocean_model import (
    DEFAULT_MODEL,
    EDGE_PAD,
    ELEVATION_INTERVAL_MINUTES,
    compute_tides,
    utc_minutes,
)

MAX_NOAA_DISTANCE_KM = 25.0
MAX_STATION_DISTANCE_KM = 200.0


def _group_events_by_date(
    events: list[TideEvent],
    begin_date: datetime.date,
    end_date: datetime.date,
    tz: datetime.tzinfo | None = None,
) -> list[TideDay]:
    """Bucket events into the requested dates on the display clock.

    `tz` is the clock the caller will display (None = UTC). Events whose
    date on that clock falls outside [begin_date, end_date] are dropped, which
    also trims any widened fetch window.
    """
    days: dict[datetime.date, list[TideEvent]] = {}
    current = begin_date
    while current <= end_date:
        days[current] = []
        current += datetime.timedelta(days=1)

    for event in events:
        clock_time = event.time.astimezone(tz) if tz is not None else event.time
        event_date = clock_time.date()
        if event_date in days:
            days[event_date].append(event)

    return [
        TideDay(date=d, events=sorted(evts, key=lambda e: e.time))
        for d, evts in sorted(days.items())
    ]


def _fetch_dates(
    begin_date: datetime.date, end_date: datetime.date, tz: datetime.tzinfo | None
) -> tuple[datetime.date, datetime.date]:
    """UTC dates to fetch/predict so every requested local day is complete.

    Local days can start or end up to ~14 h away from UTC midnight, so widen
    by one UTC day on each side whenever grouping on a non-UTC clock.
    """
    if tz is None:
        return begin_date, end_date
    one = datetime.timedelta(days=1)
    return begin_date - one, end_date + one


def _note_fallthrough(reason: str, next_source: str) -> None:
    """Tell the user (on stderr) why auto mode skipped a source.

    Stdout is left untouched so plain/JSON output formats do not change.
    """
    print(f"Note: {reason}; using {next_source}.", file=sys.stderr)


def _station_label(station: dict) -> str:
    return f"NOAA station {station['id']} ({station['name']})"


def _check_noaa_datum_supported(station: dict, datum: str) -> None:
    """Raise NOAADatumUnavailableError if the station cannot serve `datum`.

    Uses the station type from the NOAA station list. Entries cached before the
    type was recorded have no "type"; for those the request is attempted and a
    NOAA error is reported as the datum being unavailable.
    """
    station_type = station.get("type")
    if station_type == "S" and datum != "mllw":
        raise NOAADatumUnavailableError(
            f"{_station_label(station)} is a subordinate station and only publishes "
            f"MLLW predictions, not {datum.upper()}"
        )


def _fetch_noaa_events(station: dict, begin_date, end_date, datum: str) -> list[TideEvent]:
    """Fetch hi/lo predictions from NOAA, already relative to `datum`."""
    _check_noaa_datum_supported(station, datum)
    request_datum = datum if datum in NATIVE_DATUMS else "mllw"

    try:
        events = parse_predictions_response(
            fetch_predictions(station["id"], begin_date, end_date, request_datum)
        )
    except NOAADatumUnavailableError:
        raise
    except NOAAError as e:
        if "type" not in station:
            raise NOAADatumUnavailableError(
                f"{_station_label(station)} returned no {request_datum.upper()} predictions ({e})"
            ) from e
        raise

    shift = _noaa_derived_shift(station, datum)
    for event in events:
        event.height -= shift

    return events


def _noaa_derived_shift(station: dict, datum: str) -> float:
    """Metres to subtract from MLLW predictions for LAT/HAT; 0 otherwise.

    LAT/HAT are not served by the predictions API: shift the MLLW
    predictions by the station's published datum difference.
    height_target = height_MLLW - (TARGET_stnd - MLLW_stnd)
    """
    if datum not in DERIVED_DATUMS:
        return 0.0
    try:
        published = fetch_station_datums(station["id"])
    except NOAAError as e:
        raise NOAADatumUnavailableError(
            f"{_station_label(station)} does not publish {datum.upper()} ({e})"
        ) from e
    target, mllw = published.get(datum.upper()), published.get("MLLW")
    if target is None or mllw is None:
        raise NOAADatumUnavailableError(
            f"{_station_label(station)} does not publish {datum.upper()}"
        )
    return target - mllw


def _subordinate_error(station: dict) -> NOAASubordinateError:
    return NOAASubordinateError(
        f"{_station_label(station)} is a subordinate station and publishes only "
        "high/low predictions; use --source station or model"
    )


def _fetch_noaa_series(
    station: dict, begin_date: datetime.date, end_date: datetime.date, datum: str
) -> tuple[list[datetime.datetime], list[float]]:
    """6-minute predictions from NOAA, already relative to `datum`.

    Subordinate stations publish no 6-minute product (confirmed live: NOAA
    answers "No Predictions data was found"), so they raise
    NOAASubordinateError; auto mode falls through on it.
    """
    _check_noaa_datum_supported(station, datum)
    if station.get("type") == "S":
        raise _subordinate_error(station)
    request_datum = datum if datum in NATIVE_DATUMS else "mllw"
    try:
        times, heights = parse_series_response(
            fetch_predictions(station["id"], begin_date, end_date, request_datum, interval="6")
        )
    except NOAAError as e:
        if "type" not in station:
            raise _subordinate_error(station) from e
        raise
    shift = _noaa_derived_shift(station, datum)
    return times, [h - shift for h in heights]


def _resolve_noaa(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    stations: list[dict],
    datum: str = "mllw",
    max_distance_km: float = MAX_NOAA_DISTANCE_KM,
    tz: datetime.tzinfo | None = None,
) -> TideResult | None:
    """Resolve tides from the nearest NOAA station, in the requested datum.

    Returns None when no station is within range. Raises NOAAError (including
    NOAADatumUnavailableError) or httpx.HTTPError when the station cannot
    answer; auto mode falls through on those, --source noaa reports them.
    """
    result = find_nearest_noaa_station(stations, coord, max_distance_km)
    if result is None:
        return None

    station, distance = result
    fetch_begin, fetch_end = _fetch_dates(begin_date, end_date, tz)
    events = _fetch_noaa_events(station, fetch_begin, fetch_end, datum)
    days = _group_events_by_date(events, begin_date, end_date, tz)

    return TideResult(
        coordinate=coord,
        source_type=Source.NOAA,
        station_id=station["id"],
        station_name=station["name"],
        station_distance_km=round(distance, 1),
        model_name=None,
        days=days,
        datum=datum,
    )


def _resolve_station(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    max_distance_km: float = MAX_STATION_DISTANCE_KM,
    tz: datetime.tzinfo | None = None,
) -> tuple[TideResult, dict] | None:
    from tides.stations import (
        find_nearest_usable_station,
        get_station_index,
        predict_station_tides,
    )

    index = get_station_index()
    result = find_nearest_usable_station(index, coord, max_distance_km)
    if result is None:
        return None

    entry, station, distance = result
    fetch_begin, fetch_end = _fetch_dates(begin_date, end_date, tz)
    events = predict_station_tides(station, fetch_begin, fetch_end)
    if not events:
        return None

    days = _group_events_by_date(events, begin_date, end_date, tz)

    tide_result = TideResult(
        coordinate=coord,
        source_type=Source.STATION,
        station_id=entry["id"],
        station_name=station.get("name", entry["name"]),
        station_distance_km=round(distance, 1),
        model_name=None,
        days=days,
    )
    return tide_result, station


def _resolve_model(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    model_name: str = DEFAULT_MODEL,
    tz: datetime.tzinfo | None = None,
) -> TideResult:
    fetch_begin, fetch_end = _fetch_dates(begin_date, end_date, tz)
    events = compute_tides(coord, fetch_begin, fetch_end, model_name=model_name)
    if not events:
        print(
            "Error: No tidal data for this location -- it may be inland.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    days = _group_events_by_date(events, begin_date, end_date, tz)

    return TideResult(
        coordinate=coord,
        source_type=Source.MODEL,
        station_id=None,
        station_name=None,
        station_distance_km=None,
        model_name=model_name,
        days=days,
    )


def _datum_shift(
    source_type: Source,
    datum: str,
    model_name: str,
    coord: Coordinate,
    station: dict | None = None,
    station_id: str | None = None,
) -> float:
    """Metres to subtract from a source's native heights to express them in
    `datum`. Shared by peaks (_apply_datum) and the height curve.

    Station predictions arrive relative to the station's chart datum (usually
    LAT or MLLW), or MSL when that datum is not published relative to MSL
    (datums.station_heights_datum).
    Model predictions arrive relative to MSL.
    NOAA predictions are requested in the target datum already (see
    _fetch_noaa_events) and are never shifted by model-derived datums.
    """
    from tides.datums import (
        DatumUnavailableError,
        get_model_datums,
        get_station_datums,
        station_heights_datum,
    )

    if source_type == Source.NOAA:
        # Official NOAA heights: already in the requested datum.
        return 0.0

    if source_type == Source.STATION and station:
        # Station path never uses model datums: published datums first, the
        # rest computed from the station's own harmonics.
        current_datum = station_heights_datum(station)
        if datum == current_datum:
            return 0.0
        datum_offsets = get_station_datums(
            station, station_id or "", needed={current_datum, datum}
        )
    else:
        current_datum = "msl"
        if datum == current_datum:
            return 0.0
        datum_offsets = get_model_datums(coord.lat, coord.lon, model_name)

    missing = {current_datum, datum} - datum_offsets.keys()
    if missing:
        raise DatumUnavailableError(
            f"datum {', '.join(sorted(d.upper() for d in missing))} unavailable for this source"
        )

    # Convert: height_target = height_current - (target_offset - current_offset)
    return datum_offsets[datum] - datum_offsets[current_datum]


def _apply_datum(
    result: TideResult,
    datum: str,
    model_name: str,
    station: dict | None = None,
) -> TideResult:
    """Convert tide heights to the requested datum (see _datum_shift)."""
    shift = _datum_shift(
        result.source_type, datum, model_name, result.coordinate, station, result.station_id
    )
    if shift:
        for day in result.days:
            for event in day.events:
                event.height -= shift

    result.datum = datum
    return result


def resolve_tides(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    source: Source = Source.AUTO,
    model_name: str = DEFAULT_MODEL,
    datum: str = "mllw",
    tz: datetime.tzinfo | None = None,
) -> TideResult:
    """Resolve tides for [begin_date, end_date].

    `tz` is the display clock used for day grouping (None = UTC, i.e. without
    --local). With a tz, the underlying fetch/prediction is widened by a day
    on each side and trimmed back to the requested local dates.
    """
    if source == Source.NOAA:
        stations = get_stations()
        result = _resolve_noaa(coord, begin_date, end_date, stations, datum, tz=tz)
        if result is None:
            dist = MAX_NOAA_DISTANCE_KM
            print(
                f"Error: No NOAA tide station found within {dist:.0f}km of this location.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return _apply_datum(result, datum, model_name)

    if source == Source.STATION:
        resolved = _resolve_station(coord, begin_date, end_date, tz=tz)
        if resolved is None:
            dist = MAX_STATION_DISTANCE_KM
            print(
                f"Error: No tide station found within {dist:.0f}km of this location.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        result, station = resolved
        return _apply_datum(result, datum, model_name, station=station)

    if source == Source.MODEL:
        result = _resolve_model(coord, begin_date, end_date, model_name=model_name, tz=tz)
        return _apply_datum(result, datum, model_name)

    # AUTO: try NOAA API first (most accurate for US), then global station
    # database (harmonic prediction), then model fallback. A NOAA station that
    # cannot serve the requested datum, or a NOAA API/network failure (station
    # list, predictions or station datums), falls through to the next source
    # instead of failing the whole request.
    try:
        stations = get_stations()
    except httpx.HTTPError as e:
        _note_fallthrough(f"NOAA station list unavailable ({type(e).__name__})", "other sources")
        stations = []
    try:
        noaa_result = _resolve_noaa(coord, begin_date, end_date, stations, datum, tz=tz)
    except NOAAError as e:
        _note_fallthrough(str(e), "other sources")
        noaa_result = None
    except httpx.HTTPError as e:
        _note_fallthrough(f"NOAA request failed ({type(e).__name__})", "other sources")
        noaa_result = None
    if noaa_result is not None:
        return _apply_datum(noaa_result, datum, model_name)

    try:
        station_resolved = _resolve_station(coord, begin_date, end_date, tz=tz)
    except (StationDatabaseError, httpx.HTTPError) as e:
        _note_fallthrough(str(e) or type(e).__name__, "the tidal model")
        station_resolved = None
    if station_resolved is not None:
        result, station = station_resolved
        return _apply_datum(result, datum, model_name, station=station)

    result = _resolve_model(coord, begin_date, end_date, model_name=model_name, tz=tz)
    return _apply_datum(result, datum, model_name)


# -- height curves (level / when) --------------------------------------------


@dataclass
class CurveSource:
    """One resolved source that can build height curves for any window.

    Commands resolve the source once and build every curve they need from it,
    so AUTO selection never runs twice and sources are never mixed.
    """

    coordinate: Coordinate
    source_type: Source
    station_id: str | None
    station_name: str | None
    station_distance_km: float | None
    model_name: str | None
    datum: str
    build: Callable[[datetime.datetime, datetime.datetime], HeightCurve]

    def curve(self, start: datetime.datetime, end: datetime.datetime) -> HeightCurve:
        """Heights over [start - EDGE_PAD, end + EDGE_PAD) on a 1-minute grid."""
        return self.build(start, end)

    def result(self, days: list[TideDay]) -> TideResult:
        """A TideResult carrying this source's metadata, for the formatters."""
        return TideResult(
            coordinate=self.coordinate,
            source_type=self.source_type,
            station_id=self.station_id,
            station_name=self.station_name,
            station_distance_km=self.station_distance_km,
            model_name=self.model_name,
            days=days,
            datum=self.datum,
        )


def _grid(start: datetime.datetime, end: datetime.datetime) -> tuple[np.ndarray, list]:
    return utc_minutes(start - EDGE_PAD, end + EDGE_PAD, ELEVATION_INTERVAL_MINUTES)


def _noaa_builder(station: dict, datum: str):
    def build(start: datetime.datetime, end: datetime.datetime) -> HeightCurve:
        lo, hi = start - EDGE_PAD, end + EDGE_PAD
        times, heights = _fetch_noaa_series(station, lo.date(), hi.date(), datum)
        _, grid = _grid(start, end)
        data_s = np.array([(t - lo).total_seconds() for t in times])
        grid_s = np.array([(t - lo).total_seconds() for t in grid])
        return HeightCurve(lo, np.interp(grid_s, data_s, np.asarray(heights, dtype=float)))

    return build


def _station_builder(station: dict, offset: float):
    """Heights = MSL-relative harmonics + offset (chart-datum offset minus the
    target datum shift)."""
    from tides.harmonics import predict_elevations

    constituents = station.get("harmonic_constituents", [])

    def build(start: datetime.datetime, end: datetime.datetime) -> HeightCurve:
        t, _ = _grid(start, end)
        return HeightCurve(start - EDGE_PAD, predict_elevations(constituents, t) + offset)

    return build


def _model_builder(coord: Coordinate, model_name: str, offset: float):
    from tides.ocean_model import load_local_constituents, predict_elevations

    def build(start: datetime.datetime, end: datetime.datetime) -> HeightCurve:
        t, _ = _grid(start, end)
        local, m = load_local_constituents(coord.lat, coord.lon, model_name)
        return HeightCurve(start - EDGE_PAD, predict_elevations(t, local, m) - offset)

    return build


def _curve_noaa(
    coord: Coordinate,
    start: datetime.datetime,
    end: datetime.datetime,
    stations: list[dict],
    datum: str,
) -> tuple[CurveSource, HeightCurve] | None:
    found = find_nearest_noaa_station(stations, coord, MAX_NOAA_DISTANCE_KM)
    if found is None:
        return None
    station, distance = found
    src = CurveSource(
        coordinate=coord,
        source_type=Source.NOAA,
        station_id=station["id"],
        station_name=station["name"],
        station_distance_km=round(distance, 1),
        model_name=None,
        datum=datum,
        build=_noaa_builder(station, datum),
    )
    return src, src.curve(start, end)


def _curve_station(
    coord: Coordinate,
    start: datetime.datetime,
    end: datetime.datetime,
    datum: str,
    model_name: str,
) -> tuple[CurveSource, HeightCurve] | None:
    """None when no usable station is in range or its curve is all NaN."""
    from tides.stations import (
        find_nearest_usable_station,
        get_station_index,
        station_chart_offset,
    )

    index = get_station_index()
    found = find_nearest_usable_station(index, coord, MAX_STATION_DISTANCE_KM)
    if found is None:
        return None
    entry, station, distance = found
    # Chart-datum offset first (as predict_station_tides), then the shift
    # from the station's heights datum to the target (as _apply_datum).
    chart_offset = station_chart_offset(station)
    curve = _station_builder(station, chart_offset)(start, end)
    if curve.all_nan():
        return None
    shift = _datum_shift(Source.STATION, datum, model_name, coord, station, entry["id"])
    src = CurveSource(
        coordinate=coord,
        source_type=Source.STATION,
        station_id=entry["id"],
        station_name=station.get("name", entry["name"]),
        station_distance_km=round(distance, 1),
        model_name=None,
        datum=datum,
        build=_station_builder(station, chart_offset - shift),
    )
    return src, HeightCurve(curve.start, curve.heights - shift)


def _curve_model(
    coord: Coordinate,
    start: datetime.datetime,
    end: datetime.datetime,
    datum: str,
    model_name: str,
) -> tuple[CurveSource, HeightCurve]:
    curve = _model_builder(coord, model_name, 0.0)(start, end)
    if curve.all_nan():
        print(
            "Error: No tidal data for this location -- it may be inland.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    shift = _datum_shift(Source.MODEL, datum, model_name, coord)
    src = CurveSource(
        coordinate=coord,
        source_type=Source.MODEL,
        station_id=None,
        station_name=None,
        station_distance_km=None,
        model_name=model_name,
        datum=datum,
        build=_model_builder(coord, model_name, shift),
    )
    return src, HeightCurve(curve.start, curve.heights - shift)


def resolve_curve(
    coord: Coordinate,
    start: datetime.datetime,
    end: datetime.datetime,
    source: Source = Source.AUTO,
    model_name: str = DEFAULT_MODEL,
    datum: str = "mllw",
) -> tuple[CurveSource, HeightCurve]:
    """Resolve one source and its height curve over [start, end) (UTC),
    padded by EDGE_PAD on each side.

    Source selection mirrors resolve_tides: AUTO tries NOAA (6-minute
    series), then the station database, then the model, with the same
    fallthrough notes. Build further windows with the returned
    CurveSource.curve so the source is never re-resolved.
    """
    if source == Source.NOAA:
        resolved = _curve_noaa(coord, start, end, get_stations(), datum)
        if resolved is None:
            print(
                f"Error: No NOAA tide station found within {MAX_NOAA_DISTANCE_KM:.0f}km "
                "of this location.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return resolved

    if source == Source.STATION:
        resolved = _curve_station(coord, start, end, datum, model_name)
        if resolved is None:
            print(
                f"Error: No tide station found within {MAX_STATION_DISTANCE_KM:.0f}km "
                "of this location.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return resolved

    if source == Source.MODEL:
        return _curve_model(coord, start, end, datum, model_name)

    try:
        stations = get_stations()
    except httpx.HTTPError as e:
        _note_fallthrough(f"NOAA station list unavailable ({type(e).__name__})", "other sources")
        stations = []
    try:
        resolved = _curve_noaa(coord, start, end, stations, datum)
    except NOAAError as e:
        _note_fallthrough(str(e), "other sources")
        resolved = None
    except httpx.HTTPError as e:
        _note_fallthrough(f"NOAA request failed ({type(e).__name__})", "other sources")
        resolved = None
    if resolved is not None:
        return resolved

    try:
        resolved = _curve_station(coord, start, end, datum, model_name)
    except (StationDatabaseError, httpx.HTTPError) as e:
        _note_fallthrough(str(e) or type(e).__name__, "the tidal model")
        resolved = None
    if resolved is not None:
        return resolved

    return _curve_model(coord, start, end, datum, model_name)
