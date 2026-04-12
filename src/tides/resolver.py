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
) -> TideResult | None:
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

    return TideResult(
        coordinate=coord,
        source_type=Source.STATION,
        station_id=entry["id"],
        station_name=station.get("name", entry["name"]),
        station_distance_km=round(distance, 1),
        model_name=None,
        days=days,
    )


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


def resolve_tides(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    source: Source = Source.AUTO,
    model_name: str = DEFAULT_MODEL,
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
        return result

    if source == Source.STATION:
        result = _resolve_station(coord, begin_date, end_date)
        if result is None:
            dist = MAX_STATION_DISTANCE_KM
            print(
                f"Error: No tide station found within {dist:.0f}km of this location.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return result

    if source == Source.MODEL:
        return _resolve_model(coord, begin_date, end_date, model_name=model_name)

    # AUTO: try NOAA API first (most accurate for US), then global station
    # database (harmonic prediction), then model fallback
    stations = get_stations()
    noaa_result = _resolve_noaa(coord, begin_date, end_date, stations)
    if noaa_result is not None:
        return noaa_result

    station_result = _resolve_station(coord, begin_date, end_date)
    if station_result is not None:
        return station_result

    return _resolve_model(coord, begin_date, end_date, model_name=model_name)
