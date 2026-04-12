import json
import os
import sys
from pathlib import Path

STATION_CACHE_FILENAME = "noaa_stations.json"
STATION_CACHE_MAX_AGE_DAYS = 30


def get_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".cache"
    cache_dir = base / "tides"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_dir() -> Path:
    d = get_cache_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_station_cache_path() -> Path:
    return get_cache_dir() / STATION_CACHE_FILENAME


def is_station_cache_fresh() -> bool:
    import datetime

    path = get_station_cache_path()
    if not path.exists():
        return False
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc)
    age = datetime.datetime.now(tz=datetime.timezone.utc) - mtime
    return age.days < STATION_CACHE_MAX_AGE_DAYS


def save_station_cache(stations: list[dict]) -> None:
    path = get_station_cache_path()
    path.write_text(json.dumps(stations))


def load_station_cache() -> list[dict] | None:
    path = get_station_cache_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        # Corrupted cache — delete and refetch
        path.unlink(missing_ok=True)
        return None


def _got_model_exists() -> bool:
    # pyTMD stores models in its own platformdirs cache (e.g. ~/Library/Caches/pytmd/)
    try:
        import pyTMD.io

        m = pyTMD.io.model()
        m.from_database("GOT5.6")
        return True
    except FileNotFoundError:
        return False


def ensure_model_data() -> None:
    if _got_model_exists():
        return
    import pyTMD.datasets

    print("Downloading GOT5.6 tidal model... this only happens once.", file=sys.stderr)
    # GOT5.6 depends on GOT5.5 constituent files
    pyTMD.datasets.fetch_gsfc_got(model="GOT5.5", format="netcdf")
    pyTMD.datasets.fetch_gsfc_got(model="GOT5.6", format="netcdf")


def fetch_station_data() -> list[dict]:
    from tides.noaa import fetch_station_list_xml, parse_station_list

    xml_text = fetch_station_list_xml()
    stations = parse_station_list(xml_text)
    save_station_cache(stations)
    return stations


def get_stations() -> list[dict]:
    if is_station_cache_fresh():
        cached = load_station_cache()
        if cached is not None:
            return cached
    return fetch_station_data()


def fetch_all() -> None:
    print("Fetching NOAA station list...", file=sys.stderr)
    fetch_station_data()
    print("Downloading GOT5.6 tidal model (this may take a while)...", file=sys.stderr)
    ensure_model_data()
    print("Done. All data cached.", file=sys.stderr)
