"""Global tide station database.

Downloads and manages the openwatersio/tide-database, providing
nearest-station lookup and harmonic prediction for ~8,000 stations worldwide.
"""

import datetime
import json
import sys
from pathlib import Path

from tides.harmonics import predict_tides_range
from tides.models import Coordinate, TideEvent
from tides.noaa import haversine_km
from tides.ocean_model import utc_day_window

STATION_DB_URL = "https://raw.githubusercontent.com/openwatersio/tide-database/main/data"
STATION_INDEX_FILENAME = "station_index.json"


def _get_stations_dir() -> Path:
    from tides.cache import get_cache_dir

    d = get_cache_dir() / "stations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_index_path() -> Path:
    return _get_stations_dir() / STATION_INDEX_FILENAME


def build_station_index(stations_dir: Path | None = None) -> list[dict]:
    """Build an index of all stations from downloaded JSON files."""
    if stations_dir is None:
        stations_dir = _get_stations_dir()

    index = []
    for subdir in ["noaa", "ticon"]:
        d = stations_dir / subdir
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                index.append(
                    {
                        "id": f.stem,
                        "name": data.get("name", ""),
                        "lat": data["latitude"],
                        "lon": data["longitude"],
                        "source": subdir,
                        "file": str(f.relative_to(stations_dir)),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

    return index


def find_nearest_station(
    index: list[dict],
    coord: Coordinate,
    max_distance_km: float = 100.0,
) -> tuple[dict, float] | None:
    """Find the nearest station to a coordinate.

    Args:
        index: Station index (list of {id, name, lat, lon, source, file})
        coord: Target coordinate
        max_distance_km: Maximum search radius

    Returns:
        (station_entry, distance_km) or None if nothing within range
    """
    best = None
    best_dist = float("inf")
    for s in index:
        d = haversine_km(coord.lat, coord.lon, s["lat"], s["lon"])
        if d < best_dist:
            best = s
            best_dist = d
    if best is None or best_dist > max_distance_km:
        return None
    return best, best_dist


def load_station(station_entry: dict) -> dict:
    """Load full station data from a station index entry.

    Args:
        station_entry: Entry from the station index with 'file' key

    Returns:
        Full station data dict with harmonic_constituents, datums, etc.
    """
    stations_dir = _get_stations_dir()
    path = stations_dir / station_entry["file"]
    return json.loads(path.read_text())


def predict_station_tides(
    station: dict,
    begin_date: datetime.date,
    end_date: datetime.date,
) -> list[TideEvent]:
    """Predict tides at a station using its harmonic constituents.

    Heights are relative to the station's chart_datum when that datum is a
    supported one published relative to MSL; otherwise (e.g. a STND chart
    datum with no published datums) they are MSL-relative. See
    datums.station_heights_datum.

    Args:
        station: Full station data with harmonic_constituents and datums
        begin_date: Start date
        end_date: End date (inclusive)

    Returns:
        List of TideEvent with heights in datums.station_heights_datum(station)
    """
    constituents = station.get("harmonic_constituents", [])
    if not constituents:
        return []

    # Apply chart datum offset so heights are relative to the station's
    # published datum (usually LAT or MLLW). Our harmonic prediction oscillates
    # around 0 ~= MSL, so height_CD = height_MSL - (CD - MSL). When the chart
    # datum is not a supported datum published relative to MSL (e.g. STND with
    # no datums), heights stay MSL-relative; see datums.station_heights_datum.
    from tides.datums import datums_from_station, station_heights_datum

    heights_datum = station_heights_datum(station)
    offsets = datums_from_station(station) or {}
    datum_offset = -offsets.get(heights_datum, 0.0) if heights_datum != "msl" else 0.0

    # One continuous (padded) series over the whole range, so extrema at UTC
    # day boundaries are not dropped between per-day predictions.
    start, end = utc_day_window(begin_date, end_date)
    return predict_tides_range(start, end, constituents, datum_offset)


def download_station_database() -> None:
    """Download the openwatersio/tide-database station files."""
    import io
    import zipfile

    import httpx

    stations_dir = _get_stations_dir()

    # Download via GitHub archive (much faster than individual files)
    archive_url = "https://github.com/openwatersio/tide-database/archive/refs/heads/main.zip"
    print("Downloading global tide station database (~50MB)...", file=sys.stderr)

    try:
        response_ctx = httpx.stream("GET", archive_url, timeout=120, follow_redirects=True)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print(
            "Error: Could not download station database. Check your internet connection.",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    with response_ctx as response:
        response.raise_for_status()
        data = io.BytesIO()
        for chunk in response.iter_bytes():
            data.write(chunk)

    data.seek(0)
    with zipfile.ZipFile(data) as zf:
        # Extract only JSON files from the data/ directory
        for member in zf.namelist():
            # Files are like: tide-database-main/data/noaa/1234.json
            if not member.endswith(".json"):
                continue
            if "/data/noaa/" in member or "/data/ticon/" in member:
                parts = member.split("/data/", 1)
                if len(parts) == 2 and parts[1]:
                    target = (stations_dir / parts[1]).resolve()
                    # Zip Slip protection
                    try:
                        target.relative_to(stations_dir.resolve())
                    except ValueError:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

    # Build and save the index
    index = build_station_index(stations_dir)
    index_path = _get_index_path()
    index_path.write_text(json.dumps(index))
    print(f"Station database ready: {len(index)} stations.", file=sys.stderr)


def get_station_index() -> list[dict]:
    """Get the station index, downloading if needed."""
    index_path = _get_index_path()
    if index_path.exists():
        try:
            return json.loads(index_path.read_text())
        except (json.JSONDecodeError, ValueError):
            index_path.unlink(missing_ok=True)

    download_station_database()
    return json.loads(index_path.read_text())
