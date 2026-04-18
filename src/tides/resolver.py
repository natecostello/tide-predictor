import datetime
import sys

from tides.cache import get_stations
from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.noaa import (
    fetch_predictions,
    parse_predictions_response,
)
from tides.noaa import (
    find_nearest_station as find_nearest_noaa_station,
)
from tides.ocean_model import DEFAULT_MODEL, compute_tides

MAX_NOAA_DISTANCE_KM = 25.0
MAX_STATION_DISTANCE_KM = 200.0


def _group_events_by_date(
    events: list[TideEvent],
    begin_date: datetime.date,
    end_date: datetime.date,
) -> list[TideDay]:
    days: dict[datetime.date, list[TideEvent]] = {}
    current = begin_date
    while current <= end_date:
        days[current] = []
        current += datetime.timedelta(days=1)

    for event in events:
        event_date = event.time.date()
        if event_date in days:
            days[event_date].append(event)

    return [
        TideDay(date=d, events=sorted(evts, key=lambda e: e.time))
        for d, evts in sorted(days.items())
    ]


def _resolve_noaa(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    stations: list[dict],
    max_distance_km: float = MAX_NOAA_DISTANCE_KM,
) -> TideResult | None:
    result = find_nearest_noaa_station(stations, coord, max_distance_km)
    if result is None:
        return None

    station, distance = result
    data = fetch_predictions(station["id"], begin_date, end_date)
    events = parse_predictions_response(data)
    days = _group_events_by_date(events, begin_date, end_date)

    return TideResult(
        coordinate=coord,
        source_type=Source.NOAA,
        station_id=station["id"],
        station_name=station["name"],
        station_distance_km=round(distance, 1),
        model_name=None,
        days=days,
    )


def _resolve_station(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    max_distance_km: float = MAX_STATION_DISTANCE_KM,
) -> tuple[TideResult, dict] | None:
    from tides.stations import (
        find_nearest_station,
        get_station_index,
        load_station,
        predict_station_tides,
    )

    index = get_station_index()
    result = find_nearest_station(index, coord, max_distance_km)
    if result is None:
        return None

    entry, distance = result
    station = load_station(entry)
    events = predict_station_tides(station, begin_date, end_date)
    if not events:
        return None

    days = _group_events_by_date(events, begin_date, end_date)

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
) -> TideResult:
    events = compute_tides(coord, begin_date, end_date, model_name=model_name)
    if not events:
        print(
            "Error: No tidal data for this location — it may be inland.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    days = _group_events_by_date(events, begin_date, end_date)

    return TideResult(
        coordinate=coord,
        source_type=Source.MODEL,
        station_id=None,
        station_name=None,
        station_distance_km=None,
        model_name=model_name,
        days=days,
    )


def _apply_datum(
    result: TideResult,
    datum: str,
    model_name: str,
    station: dict | None = None,
) -> TideResult:
    """Convert tide heights to the requested datum.

    Station predictions arrive relative to chart_datum (LAT or MLLW).
    Model predictions arrive relative to MSL.
    This function converts to the requested datum.
    """
    from tides.datums import datums_from_station, get_model_datums

    # Determine current datum and get offset table
    if result.source_type in (Source.STATION,) and station:
        datum_offsets = datums_from_station(station)
        current_datum = station.get("chart_datum", "MSL").lower()
    elif result.source_type == Source.NOAA:
        # NOAA predictions are requested relative to MTL
        datum_offsets = None
        current_datum = "mtl"
    else:
        datum_offsets = None
        current_datum = "msl"

    if datum == current_datum:
        result.datum = datum
        return result

    if datum_offsets is None:
        datum_offsets = get_model_datums(
            result.coordinate.lat,
            result.coordinate.lon,
            model_name,
        )

    # Convert: height_target = height_current - (target_offset - current_offset)
    current_offset = datum_offsets.get(current_datum, 0.0)
    target_offset = datum_offsets.get(datum, 0.0)
    shift = target_offset - current_offset

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
) -> TideResult:
    if source == Source.NOAA:
        stations = get_stations()
        result = _resolve_noaa(coord, begin_date, end_date, stations)
        if result is None:
            dist = MAX_NOAA_DISTANCE_KM
            print(
                f"Error: No NOAA tide station found within {dist:.0f}km of this location.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return _apply_datum(result, datum, model_name)

    if source == Source.STATION:
        resolved = _resolve_station(coord, begin_date, end_date)
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
        result = _resolve_model(coord, begin_date, end_date, model_name=model_name)
        return _apply_datum(result, datum, model_name)

    # AUTO: try NOAA API first (most accurate for US), then global station
    # database (harmonic prediction), then model fallback
    stations = get_stations()
    noaa_result = _resolve_noaa(coord, begin_date, end_date, stations)
    if noaa_result is not None:
        return _apply_datum(noaa_result, datum, model_name)

    station_resolved = _resolve_station(coord, begin_date, end_date)
    if station_resolved is not None:
        result, station = station_resolved
        return _apply_datum(result, datum, model_name, station=station)

    result = _resolve_model(coord, begin_date, end_date, model_name=model_name)
    return _apply_datum(result, datum, model_name)
