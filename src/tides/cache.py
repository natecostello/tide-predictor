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


EOT20_URL = "https://www.seanoe.org/data/00683/79489/data/85762.zip"


def _model_exists(model_name: str) -> bool:
    try:
        import pyTMD.io

        m = pyTMD.io.model()
        m.from_database(model_name)
        return True
    except FileNotFoundError:
        return False


def _get_pytmd_data_dir() -> Path:
    """Get pyTMD's default data directory (platformdirs cache)."""
    import platformdirs

    return Path(platformdirs.user_cache_dir("pytmd"))


def _fetch_got() -> None:
    import pyTMD.datasets

    # GOT5.6 depends on GOT5.5 constituent files
    pyTMD.datasets.fetch_gsfc_got(model="GOT5.5", format="netcdf")
    pyTMD.datasets.fetch_gsfc_got(model="GOT5.6", format="netcdf")


def _fetch_eot20() -> None:
    import tempfile
    import zipfile

    import httpx

    data_dir = _get_pytmd_data_dir()
    eot_dir = data_dir / "EOT20" / "ocean_tides"
    if eot_dir.exists() and any(eot_dir.iterdir()):
        return

    print("Downloading EOT20 tidal model (~2.3GB)...", file=sys.stderr)
    tmp_path = Path(tempfile.mktemp(suffix=".zip", dir=data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", EOT20_URL, timeout=600, follow_redirects=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        mb = downloaded // (1024 * 1024)
                        total_mb = total // (1024 * 1024)
                        print(
                            f"\r  {mb}MB / {total_mb}MB ({pct}%)",
                            end="",
                            file=sys.stderr,
                        )
            if total:
                print(file=sys.stderr)

        print("Extracting...", file=sys.stderr)
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(data_dir)

        # SEANOE archive contains inner ZIPs (ocean_tides.zip, load_tides.zip)
        eot_base = data_dir / "EOT20"
        eot_base.mkdir(exist_ok=True)
        for inner_name in ["ocean_tides.zip", "load_tides.zip"]:
            inner_path = data_dir / inner_name
            if inner_path.exists():
                with zipfile.ZipFile(inner_path) as inner_zf:
                    inner_zf.extractall(eot_base)
                inner_path.unlink()

        print("EOT20 download complete.", file=sys.stderr)
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_model_data(model_name: str = "GOT5.6") -> None:
    if _model_exists(model_name):
        return

    if model_name in ("GOT5.6", "GOT5.5"):
        print(f"Downloading {model_name} tidal model... this only happens once.", file=sys.stderr)
        _fetch_got()
    elif model_name == "EOT20":
        _fetch_eot20()
    else:
        print(f"Error: Unknown model '{model_name}'.", file=sys.stderr)
        raise SystemExit(2)


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
