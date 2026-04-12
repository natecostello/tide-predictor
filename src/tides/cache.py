import json
import os
import shutil
import sys
from pathlib import Path

STATION_CACHE_FILENAME = "noaa_stations.json"
STATION_CACHE_MAX_AGE_DAYS = 30

# Models that can be cleared from the pyTMD cache.
# Maps user-facing name -> directory name under pytmd cache.
PYTMD_MODEL_DIRS: dict[str, str] = {
    "GOT5.5": "GOT5.5",
    "GOT5.6": "GOT5.6",
    "EOT20": "EOT20",
    "FES2022": "fes2022b",
    "HAMTIDE11": "hamtide",
}


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
    elif model_name in ("FES2022", "HAMTIDE11"):
        print(
            f"Error: {model_name} model data not found. This model must be downloaded manually.",
            file=sys.stderr,
        )
        if model_name == "FES2022":
            print(
                "Download FES2022b from AVISO (https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html)",
                file=sys.stderr,
            )
            data_dir = _get_pytmd_data_dir()
            print(f"Place files in: {data_dir}/fes2022b/ocean_tide_20241025/", file=sys.stderr)
        elif model_name == "HAMTIDE11":
            print(
                "Download HAMTIDE11 from https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/hamtide/catalog.html",
                file=sys.stderr,
            )
            data_dir = _get_pytmd_data_dir()
            print(f"Place files in: {data_dir}/hamtide/", file=sys.stderr)
        raise SystemExit(2)
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


def _dir_size(path: Path) -> int:
    """Total bytes used by a directory tree."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def get_cache_info() -> dict:
    """Return structured cache information for both app and model caches."""
    app_dir = get_cache_dir()
    pytmd_dir = _get_pytmd_data_dir()

    # App cache breakdown
    stations_dir = app_dir / "stations"
    noaa_stations_path = get_station_cache_path()

    app_items = []
    if noaa_stations_path.exists():
        size = noaa_stations_path.stat().st_size
        app_items.append(
            {
                "name": "NOAA station list",
                "path": str(noaa_stations_path),
                "size": size,
            }
        )
    if stations_dir.exists():
        size = _dir_size(stations_dir)
        app_items.append({"name": "Station database", "path": str(stations_dir), "size": size})

    # pyTMD model cache breakdown
    model_items = []
    for model_name, dirname in PYTMD_MODEL_DIRS.items():
        model_path = pytmd_dir / dirname
        if model_path.exists():
            size = _dir_size(model_path)
            model_items.append({"name": model_name, "path": str(model_path), "size": size})

    return {
        "app_cache": {"path": str(app_dir), "items": app_items},
        "model_cache": {"path": str(pytmd_dir), "items": model_items},
    }


def clear_cache(name: str | None = None) -> int:
    """Clear cache data. Returns bytes freed.

    Args:
        name: Specific item to clear. None = clear everything.
              Valid: 'stations', or a model name (got5.5, got5.6, eot20, fes2022, hamtide11)
    """
    if name is None:
        # Clear everything
        app_dir = get_cache_dir()
        pytmd_dir = _get_pytmd_data_dir()
        freed = _dir_size(app_dir)
        for dirname in PYTMD_MODEL_DIRS.values():
            freed += _dir_size(pytmd_dir / dirname)
        shutil.rmtree(app_dir, ignore_errors=True)
        for dirname in PYTMD_MODEL_DIRS.values():
            model_path = pytmd_dir / dirname
            if model_path.exists():
                shutil.rmtree(model_path)
        return freed

    name_upper = name.upper()

    if name_upper == "STATIONS":
        stations_dir = get_cache_dir() / "stations"
        noaa_path = get_station_cache_path()
        freed = _dir_size(stations_dir)
        if noaa_path.exists():
            freed += noaa_path.stat().st_size
            noaa_path.unlink()
        if stations_dir.exists():
            shutil.rmtree(stations_dir)
        return freed

    if name_upper in PYTMD_MODEL_DIRS:
        dirname = PYTMD_MODEL_DIRS[name_upper]
        model_path = _get_pytmd_data_dir() / dirname
        freed = _dir_size(model_path)
        if model_path.exists():
            shutil.rmtree(model_path)
        return freed

    valid = ["stations"] + [n.lower() for n in PYTMD_MODEL_DIRS]
    raise ValueError(f"Unknown cache name '{name}'. Valid names: {', '.join(valid)}")
