import datetime
import math
import xml.etree.ElementTree as ET

import httpx

from tides.models import Coordinate, TideEvent

STATIONS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.xml?type=tidepredictions&units=metric"
PREDICTIONS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

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
        if id_el is not None and name_el is not None and lat_el is not None and lng_el is not None:
            stations.append(
                {
                    "id": id_el.text,
                    "name": name_el.text,
                    "lat": float(lat_el.text),
                    "lon": float(lng_el.text),
                }
            )
    return stations


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
        d = _haversine_km(coord.lat, coord.lon, s["lat"], s["lon"])
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
) -> dict:
    params = {
        "begin_date": begin_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "station": station_id,
        "product": "predictions",
        "datum": "MTL",
        "units": "metric",
        "time_zone": "gmt",
        "interval": "hilo",
        "format": "json",
        "application": "tides_cli",
    }
    response = httpx.get(PREDICTIONS_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


class NOAAError(Exception):
    pass


def parse_predictions_response(data: dict) -> list[TideEvent]:
    # NOAA returns {"error": {"message": "..."}} on failure
    if "error" in data:
        msg = data["error"].get("message", "Unknown NOAA API error")
        raise NOAAError(f"NOAA API error: {msg}")

    predictions = data.get("predictions")
    if predictions is None or len(predictions) == 0:
        raise NOAAError("NOAA returned no tide predictions for this station and date range.")

    events = []
    for p in predictions:
        time = datetime.datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
        time = time.replace(tzinfo=datetime.timezone.utc)
        height = float(p["v"])
        events.append(TideEvent(time=time, height=height))
    return events
