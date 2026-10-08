import datetime
import math
import xml.etree.ElementTree as ET

import httpx

from tides.models import Coordinate, TideEvent

STATIONS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.xml?type=tidepredictions&units=metric"
PREDICTIONS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATION_DATUMS_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}/datums.json"
)

# Datums the predictions API serves directly at reference ("R") stations.
# Subordinate ("S") stations only serve MLLW. LAT/HAT are never served by the
# predictions API; they are derived from the station's published datums.
NATIVE_DATUMS = ("mllw", "mlw", "msl", "mtl", "mhw", "mhhw")
DERIVED_DATUMS = ("lat", "hat")

REQUEST_TIMEOUT = 30.0


def fetch_station_list_xml() -> str:
    response = httpx.get(STATIONS_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def parse_station_list(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    stations = []
    for station_el in root.findall(".//Station"):
        # The NOAA API returns station fields as direct children:
        # <id>, <name>, <lat>, <lng>
        id_el = station_el.find("id")
        name_el = station_el.find("name")
        lat_el = station_el.find("lat")
        lng_el = station_el.find("lng")
        type_el = station_el.find("type")
        if id_el is not None and name_el is not None and lat_el is not None and lng_el is not None:
            station = {
                "id": id_el.text,
                "name": name_el.text,
                "lat": float(lat_el.text),
                "lon": float(lng_el.text),
            }
            # "R" = reference (harmonic) station, "S" = subordinate station.
            if type_el is not None and type_el.text:
                station["type"] = type_el.text.strip().upper()
            stations.append(station)
    return stations


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return earth_radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_station(
    stations: list[dict],
    coord: Coordinate,
    max_distance_km: float = 25.0,
) -> tuple[dict, float] | None:
    best = None
    best_dist = float("inf")
    for s in stations:
        d = haversine_km(coord.lat, coord.lon, s["lat"], s["lon"])
        if d < best_dist:
            best = s
            best_dist = d
    if best is None or best_dist > max_distance_km:
        return None
    return best, best_dist


def fetch_predictions(
    station_id: str,
    begin_date: datetime.date,
    end_date: datetime.date,
    datum: str = "mllw",
    interval: str = "hilo",
) -> dict:
    """Raw datagetter predictions. `interval` is "hilo" (highs/lows) or "6"
    (6-minute series; a single request spans well over a year, so no
    chunking is needed for the 366-day cap)."""
    params = {
        "begin_date": begin_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "station": station_id,
        "product": "predictions",
        "datum": datum.upper(),
        "units": "metric",
        "time_zone": "gmt",
        "interval": interval,
        "format": "json",
        "application": "tides_cli",
    }
    response = httpx.get(PREDICTIONS_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


class NOAAError(Exception):
    pass


class NOAADatumUnavailableError(NOAAError):
    """The chosen NOAA station does not publish the requested datum."""


class NOAASubordinateError(NOAAError):
    """The chosen NOAA station publishes only high/low predictions."""


def fetch_station_datums(station_id: str) -> dict[str, float]:
    """Fetch a station's published datums (meters, relative to STND).

    Returns a dict keyed by upper-case datum name (e.g. "MLLW", "LAT", "HAT").
    LAT/HAT come from the top-level fields of NOAA's datums.json and are only
    present when NOAA publishes them (reference stations). Raises NOAAError when
    the station publishes no datums at all (e.g. subordinate stations).
    """
    response = httpx.get(
        STATION_DATUMS_URL.format(station_id=station_id),
        params={"units": "metric"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    entries = data.get("datums")
    if not entries:
        raise NOAAError(f"NOAA publishes no datums for station {station_id}.")

    datums: dict[str, float] = {}
    for entry in entries:
        name, value = entry.get("name"), entry.get("value")
        if name and value is not None:
            datums[name.upper()] = float(value)
    for key in ("LAT", "HAT"):
        if data.get(key) is not None:
            datums[key] = float(data[key])
    return datums


def _predictions(data: dict) -> list[dict]:
    # NOAA returns {"error": {"message": "..."}} on failure
    if "error" in data:
        msg = data["error"].get("message", "Unknown NOAA API error")
        raise NOAAError(f"NOAA API error: {msg}")

    predictions = data.get("predictions")
    if predictions is None or len(predictions) == 0:
        raise NOAAError("NOAA returned no tide predictions for this station and date range.")
    return predictions


def _parse_time(t: str) -> datetime.datetime:
    time = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M")
    return time.replace(tzinfo=datetime.timezone.utc)


def parse_series_response(data: dict) -> tuple[list[datetime.datetime], list[float]]:
    """(times, heights) from an interval=6 predictions response."""
    predictions = _predictions(data)
    return [_parse_time(p["t"]) for p in predictions], [float(p["v"]) for p in predictions]


def parse_predictions_response(data: dict) -> list[TideEvent]:
    predictions = _predictions(data)

    events = []
    for p in predictions:
        time = _parse_time(p["t"])
        height = float(p["v"])
        # NOAA hilo "type" is H/L, and at mixed-tide stations HH/LL plus
        # HL ("higher low", a low) / LH ("lower high", a high): the LAST letter
        # is the tide kind.
        kind = "high" if str(p.get("type", "")).upper().endswith("H") else "low"
        events.append(TideEvent(time=time, height=height, kind=kind))
    return events
