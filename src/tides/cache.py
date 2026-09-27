import json
import os
import shutil
import sys
from pathlib import Path

STATION_CACHE_FILENAME = "noaa_stations.json"
STATION_CACHE_MAX_AGE_DAYS = 30


class StationDatabaseError(Exception):
    """The global tide station database (GitHub) could not be downloaded."""


# Models that can be cleared from the pyTMD cache.
# Maps user-facing name -> directory name under pytmd cache.
PYTMD_MODEL_DIRS: dict[str, str] = {
    "GOT5.5": "GOT5.5",
    "GOT5.6": "GOT5.6",
    "EOT20": "EOT20",
    "FES2022": "fes2022b",
    "HAMTIDE11": "hamtide",
}

# Models re-downloaded automatically on next use; the only models a bare
# `tides cache clear` removes. EOT20 (~2.3 GB download) and the manually
# downloaded FES2022 / HAMTIDE11 are only removed when named or with --all.
AUTO_DOWNLOADED_MODELS = ("GOT5.5", "GOT5.6")

# Written into EOT20/ once extraction fully completes.
EOT20_COMPLETE_MARKER = ".tides-complete"

DATUM_CACHE_DIRNAME = "datums"


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (temp file in the same dir + replace).

    A crash mid-write leaves the previous file intact instead of a truncated
    one.
    """
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def get_cache_dir(create: bool = True) -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".cache"
    cache_dir = base / "tides"
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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
    atomic_write_text(get_station_cache_path(), json.dumps(stations))


def load_station_cache() -> list[dict] | None:
    path = get_station_cache_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        # Corrupted cache -- delete and refetch
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
    """Get pyTMD's default data directory (platformdirs cache).

    TIDES_PYTMD_DIR is an undocumented override for tests/sandboxes; it only
    redirects this tool's cache listing/clearing, not pyTMD's own lookup.
    """
    override = os.environ.get("TIDES_PYTMD_DIR")
    if override:
        return Path(override)
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
    eot_base = data_dir / "EOT20"
    if (eot_base / EOT20_COMPLETE_MARKER).exists():
        return
    if eot_base.exists():
        # No completion marker: a previous extraction was interrupted. Remove
        # only the EOT20 directory and start over.
        shutil.rmtree(eot_base)

    print("Downloading EOT20 tidal model (~2.3GB)...", file=sys.stderr)
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=data_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
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
        eot_base.mkdir(exist_ok=True)
        for inner_name in ["ocean_tides.zip", "load_tides.zip"]:
            inner_path = data_dir / inner_name
            if inner_path.exists():
                with zipfile.ZipFile(inner_path) as inner_zf:
                    inner_zf.extractall(eot_base)
                inner_path.unlink()

        (eot_base / EOT20_COMPLETE_MARKER).write_text("ok\n")
        print("EOT20 download complete.", file=sys.stderr)
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_model_data(model_name: str = "GOT5.6") -> None:
    if _model_exists(model_name):
        if model_name == "EOT20":
            # Existing complete installs predate the marker: record it once so
            # a later partial-install check never re-downloads them.
            marker = _get_pytmd_data_dir() / "EOT20" / EOT20_COMPLETE_MARKER
            if not marker.exists() and marker.parent.exists():
                marker.write_text("ok\n")
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


def _station_cache_age_days() -> int:
    import datetime

    mtime = datetime.datetime.fromtimestamp(
        get_station_cache_path().stat().st_mtime, tz=datetime.timezone.utc
    )
    return (datetime.datetime.now(tz=datetime.timezone.utc) - mtime).days


def get_stations() -> list[dict]:
    """NOAA station list: fresh cache, else refetch.

    If the refresh fails (offline, NOAA down) and a stale cache exists, use the
    stale list with a warning rather than failing; the stale file is never
    deleted. Only raises when there is no cache at all.
    """
    import httpx

    if is_station_cache_fresh():
        cached = load_station_cache()
        if cached is not None:
            return cached
    try:
        return fetch_station_data()
    except httpx.HTTPError as e:
        stale = load_station_cache()
        if stale is None:
            raise
        print(
            f"Warning: using station list cached {_station_cache_age_days()} days ago "
            f"(refresh failed: {type(e).__name__})",
            file=sys.stderr,
        )
        return stale


def fetch_all() -> None:
    """Pre-fetch everything a query can need: NOAA list, station DB, GOT5.6."""
    from tides.stations import _get_index_path, get_station_index, load_valid_index

    stations = fetch_station_data()
    print(f"NOAA station list: updated ({len(stations)} stations)", file=sys.stderr)

    # get_station_index() validates the index and repairs (rebuilds from disk
    # or re-downloads) a missing or invalid one; report "already present" only
    # when the existing index was already valid.
    was_valid = load_valid_index(_get_index_path()) is not None
    get_station_index()
    if was_valid:
        print("Station database: already present", file=sys.stderr)

    if _model_exists("GOT5.6"):
        print("GOT5.6 model: already present", file=sys.stderr)
    else:
        ensure_model_data("GOT5.6")  # prints its own download message

    print("Done. All data cached.", file=sys.stderr)


def _dir_size(path: Path) -> int:
    """Total bytes used by a directory tree."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def get_cache_info() -> dict:
    """Return structured cache information for both app and model caches."""
    app_dir = get_cache_dir(create=False)
    pytmd_dir = _get_pytmd_data_dir()

    # App cache breakdown
    stations_dir = app_dir / "stations"
    datums_dir = app_dir / DATUM_CACHE_DIRNAME
    noaa_stations_path = app_dir / STATION_CACHE_FILENAME

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
    if datums_dir.exists():
        size = _dir_size(datums_dir)
        app_items.append({"name": "Datum cache", "path": str(datums_dir), "size": size})

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


def _item(name: str, path: Path) -> dict:
    size = path.stat().st_size if path.is_file() else _dir_size(path)
    return {"name": name, "path": str(path), "size": size}


def plan_clear(name: str | None = None, include_all: bool = False) -> list[dict]:
    """List exactly what clear_cache(name, include_all) would remove.

    Each item is {name, path, size}; only existing paths are listed.

    - name None: the app cache dir plus auto-downloaded models (GOT5.5,
      GOT5.6). With include_all, every model (EOT20, FES2022, HAMTIDE11 too).
    - "stations": NOAA station list + global station database.
    - "datums": computed datum cache.
    - a model name: that model's directory in pyTMD's cache.
    """
    app_dir = get_cache_dir(create=False)
    pytmd_dir = _get_pytmd_data_dir()

    if name is None:
        candidates = [("App cache", app_dir)]
        models = PYTMD_MODEL_DIRS if include_all else AUTO_DOWNLOADED_MODELS
        candidates += [(m, pytmd_dir / PYTMD_MODEL_DIRS[m]) for m in models]
    else:
        name_upper = name.upper()
        if name_upper == "STATIONS":
            candidates = [
                ("NOAA station list", app_dir / STATION_CACHE_FILENAME),
                ("Station database", app_dir / "stations"),
            ]
        elif name_upper == "DATUMS":
            candidates = [("Datum cache", app_dir / DATUM_CACHE_DIRNAME)]
        elif name_upper in PYTMD_MODEL_DIRS:
            candidates = [(name_upper, pytmd_dir / PYTMD_MODEL_DIRS[name_upper])]
        else:
            valid = ["stations", "datums"] + [n.lower() for n in PYTMD_MODEL_DIRS]
            raise ValueError(f"Unknown cache name '{name}'. Valid names: {', '.join(valid)}")

    return [_item(label, path) for label, path in candidates if path.exists()]


def clear_cache(name: str | None = None, include_all: bool = False) -> int:
    """Clear cache data. Returns bytes freed.

    Removes exactly the items plan_clear(name, include_all) lists. By default
    (name None) manually downloaded / large models (EOT20, FES2022,
    HAMTIDE11) are never touched; pass include_all or name them explicitly.
    """
    freed = 0
    for item in plan_clear(name, include_all):
        path = Path(item["path"])
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        freed += item["size"]
    return freed
