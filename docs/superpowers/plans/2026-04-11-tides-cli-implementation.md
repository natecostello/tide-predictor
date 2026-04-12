# Tides CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that returns high/low tide predictions for coastal coordinates using NOAA station data and the GOT5.6 global tidal model.

**Architecture:** Typer CLI dispatches to a resolver that picks the best data source (NOAA station or GOT5.6 model) based on proximity. NOAA client fetches station predictions via HTTP. Ocean model wrapper uses pyTMD to compute elevations and scipy to find extrema. Output formatting handles plain text and JSON, with timezone conversion via timezonefinder.

**Tech Stack:** Python 3.11+, uv, Typer, pyTMD, scipy, numpy, httpx, timezonefinder, rich, ruff, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, entry point, ruff config |
| `src/tides/__init__.py` | Package marker with version |
| `src/tides/models.py` | Data classes: `Coordinate`, `TideEvent`, `TideDay`, `TideResult`, `Source` enum |
| `src/tides/cache.py` | XDG cache path resolution, model download orchestration, station list caching |
| `src/tides/noaa.py` | NOAA CO-OPS API: station list fetch, nearest station lookup, tide predictions |
| `src/tides/ocean_model.py` | pyTMD wrapper: elevation computation, extrema finding via scipy |
| `src/tides/resolver.py` | Source selection: auto/noaa/model dispatch, returns `TideResult` |
| `src/tides/timezone.py` | Coordinate to IANA timezone, UTC-to-local conversion |
| `src/tides/cli.py` | Typer app: coordinate parsing, flag handling, output formatting, entry point |
| `tests/test_models.py` | Tests for data classes and validation |
| `tests/test_cache.py` | Tests for XDG path resolution |
| `tests/test_noaa.py` | Tests for NOAA API parsing (mocked HTTP) |
| `tests/test_ocean_model.py` | Tests for extrema finding |
| `tests/test_resolver.py` | Tests for source selection logic |
| `tests/test_timezone.py` | Tests for timezone mapping |
| `tests/test_cli.py` | Tests for coordinate parsing, date parsing, output formatting, CLI invocation |

---

### Task 0: Create Feature Branch

- [ ] **Step 1: Create and switch to feature branch**

```bash
git checkout -b feat/initial-implementation
```

All subsequent tasks commit to this branch. The PR is created in Task 12.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/tides/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tides"
version = "0.1.0"
description = "A CLI for tide predictions using NOAA station data and the GOT5.6 global tidal model"
requires-python = ">=3.11"
dependencies = [
    "typer[all]>=0.9",
    "pyTMD>=3.0",
    "scipy>=1.11",
    "numpy>=1.24",
    "httpx>=0.25",
    "timezonefinder>=6.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "ruff>=0.4",
    "pytest-httpx>=0.30",
]

[project.scripts]
tides = "tides.cli:app"

[tool.ruff]
target-version = "py311"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.hatch.build.targets.wheel]
packages = ["src/tides"]
```

- [ ] **Step 2: Create package init**

```python
# src/tides/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Initialize uv environment and install**

Run: `cd /Users/ncos/GithubRepos/tide-predictor && uv venv && uv pip install -e ".[dev]"`
Expected: Successful installation with all dependencies resolved.

- [ ] **Step 4: Verify ruff works**

Run: `uv run ruff check src/`
Expected: No errors (only one file with a docstring-less module).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/tides/__init__.py
git commit -m "feat: project scaffolding with pyproject.toml and uv"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/tides/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for data models**

```python
# tests/test_models.py
import datetime

import pytest

from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult


class TestCoordinate:
    def test_valid_coordinate(self):
        c = Coordinate(lat=40.7128, lon=-74.0060)
        assert c.lat == 40.7128
        assert c.lon == -74.0060

    def test_invalid_latitude_too_high(self):
        with pytest.raises(ValueError, match="Latitude"):
            Coordinate(lat=91.0, lon=0.0)

    def test_invalid_latitude_too_low(self):
        with pytest.raises(ValueError, match="Latitude"):
            Coordinate(lat=-91.0, lon=0.0)

    def test_invalid_longitude_too_high(self):
        with pytest.raises(ValueError, match="Longitude"):
            Coordinate(lat=0.0, lon=181.0)

    def test_invalid_longitude_too_low(self):
        with pytest.raises(ValueError, match="Longitude"):
            Coordinate(lat=0.0, lon=-181.0)

    def test_boundary_values(self):
        c = Coordinate(lat=90.0, lon=180.0)
        assert c.lat == 90.0
        assert c.lon == 180.0

    def test_boundary_negative(self):
        c = Coordinate(lat=-90.0, lon=-180.0)
        assert c.lat == -90.0
        assert c.lon == -180.0


class TestTideEvent:
    def test_tide_event_creation(self):
        t = TideEvent(
            time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
            height=0.3,
        )
        assert t.height == 0.3

    def test_height_in_feet(self):
        t = TideEvent(
            time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
            height=0.3,
        )
        assert abs(t.height_ft - 0.984252) < 0.001


class TestTideDay:
    def test_tide_day(self):
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
                height=0.3,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 20, 45, tzinfo=datetime.timezone.utc),
                height=-0.1,
            ),
        ]
        day = TideDay(date=datetime.date(2026, 4, 15), events=events)
        assert len(day.events) == 2
        assert day.date == datetime.date(2026, 4, 15)


class TestSource:
    def test_source_enum(self):
        assert Source.AUTO.value == "auto"
        assert Source.NOAA.value == "noaa"
        assert Source.MODEL.value == "model"


class TestTideResult:
    def test_tide_result_noaa(self):
        result = TideResult(
            coordinate=Coordinate(lat=40.7128, lon=-74.0060),
            source_type=Source.NOAA,
            station_id="8518750",
            station_name="The Battery",
            station_distance_km=1.2,
            model_name=None,
            days=[],
        )
        assert result.source_type == Source.NOAA
        assert result.station_name == "The Battery"

    def test_tide_result_model(self):
        result = TideResult(
            coordinate=Coordinate(lat=-5.0, lon=-35.0),
            source_type=Source.MODEL,
            station_id=None,
            station_name=None,
            station_distance_km=None,
            model_name="GOT5.6",
            days=[],
        )
        assert result.source_type == Source.MODEL
        assert result.model_name == "GOT5.6"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tides.models'`

- [ ] **Step 3: Implement models**

```python
# src/tides/models.py
import datetime
from dataclasses import dataclass
from enum import Enum

METERS_TO_FEET = 3.28084


class Source(Enum):
    AUTO = "auto"
    NOAA = "noaa"
    MODEL = "model"


@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float

    def __post_init__(self):
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {self.lon}")


@dataclass
class TideEvent:
    time: datetime.datetime
    height: float  # meters

    @property
    def height_ft(self) -> float:
        return self.height * METERS_TO_FEET


@dataclass
class TideDay:
    date: datetime.date
    events: list[TideEvent]


@dataclass
class TideResult:
    coordinate: Coordinate
    source_type: Source
    station_id: str | None
    station_name: str | None
    station_distance_km: float | None
    model_name: str | None
    days: list[TideDay]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/tides/models.py tests/test_models.py`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/tides/models.py tests/test_models.py
git commit -m "feat: add data models (Coordinate, TideEvent, TideDay, TideResult, Source)"
```

---

### Task 3: Cache Management

**Files:**
- Create: `src/tides/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests for cache path resolution**

```python
# tests/test_cache.py
import os
from unittest.mock import patch

from tides.cache import get_cache_dir


class TestGetCacheDir:
    def test_default_cache_dir(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove XDG_CACHE_HOME if set
            os.environ.pop("XDG_CACHE_HOME", None)
            d = get_cache_dir()
            assert d.name == "tides"
            assert d.parent.name == ".cache"

    def test_xdg_cache_home(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_cache_dir()
            assert d == tmp_path / "tides"

    def test_cache_dir_is_created(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_cache_dir()
            assert d.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tides.cache'`

- [ ] **Step 3: Implement cache module**

```python
# src/tides/cache.py
import json
import os
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
    path = get_station_cache_path()
    if not path.exists():
        return False
    import datetime

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
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tides/cache.py tests/test_cache.py
git commit -m "feat: XDG-compliant cache management for model data and station list"
```

---

### Task 4: Timezone Module

**Files:**
- Create: `src/tides/timezone.py`
- Create: `tests/test_timezone.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_timezone.py
import datetime

from tides.models import Coordinate
from tides.timezone import get_timezone_name, to_local_time


class TestGetTimezoneName:
    def test_new_york(self):
        coord = Coordinate(lat=40.7128, lon=-74.0060)
        assert get_timezone_name(coord) == "America/New_York"

    def test_brazil(self):
        coord = Coordinate(lat=-5.0, lon=-35.0)
        assert get_timezone_name(coord) == "America/Recife"

    def test_london(self):
        coord = Coordinate(lat=51.5074, lon=-0.1278)
        assert get_timezone_name(coord) == "Europe/London"

    def test_ocean_returns_none(self):
        # Middle of the Pacific
        coord = Coordinate(lat=0.0, lon=-160.0)
        result = get_timezone_name(coord)
        # timezonefinder may return a timezone for ocean points or None
        # We accept either — the important thing is it doesn't crash
        assert result is None or isinstance(result, str)


class TestToLocalTime:
    def test_utc_to_new_york(self):
        coord = Coordinate(lat=40.7128, lon=-74.0060)
        utc_time = datetime.datetime(2026, 4, 15, 18, 32, tzinfo=datetime.timezone.utc)
        local = to_local_time(utc_time, coord)
        # EDT is UTC-4 in April
        assert local.hour == 14
        assert local.minute == 32

    def test_utc_to_brazil(self):
        coord = Coordinate(lat=-5.0, lon=-35.0)
        utc_time = datetime.datetime(2026, 4, 15, 18, 0, tzinfo=datetime.timezone.utc)
        local = to_local_time(utc_time, coord)
        # Recife is UTC-3, no DST
        assert local.hour == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_timezone.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tides.timezone'`

- [ ] **Step 3: Implement timezone module**

```python
# src/tides/timezone.py
import datetime
from zoneinfo import ZoneInfo

from timezonefinder import TimezoneFinder

from tides.models import Coordinate

_tf: TimezoneFinder | None = None


def _get_finder() -> TimezoneFinder:
    global _tf
    if _tf is None:
        _tf = TimezoneFinder()
    return _tf


def get_timezone_name(coord: Coordinate) -> str | None:
    return _get_finder().timezone_at(lat=coord.lat, lng=coord.lon)


def to_local_time(utc_time: datetime.datetime, coord: Coordinate) -> datetime.datetime:
    tz_name = get_timezone_name(coord)
    if tz_name is None:
        return utc_time
    return utc_time.astimezone(ZoneInfo(tz_name))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_timezone.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tides/timezone.py tests/test_timezone.py
git commit -m "feat: timezone lookup and UTC-to-local conversion"
```

---

### Task 5: NOAA CO-OPS Client

**Files:**
- Create: `src/tides/noaa.py`
- Create: `tests/test_noaa.py`

- [ ] **Step 1: Write failing tests for station list parsing and nearest station lookup**

```python
# tests/test_noaa.py
import datetime
import json

import httpx
import pytest

from tides.models import Coordinate, TideDay, TideEvent
from tides.noaa import (
    find_nearest_station,
    parse_predictions_response,
    parse_station_list,
)

SAMPLE_STATION_LIST_XML = """<?xml version="1.0" encoding="utf-8" ?>
<Stations>
  <Station ID="8518750" name="The Battery">
    <metadata>
      <location>
        <lat>40.7006</lat>
        <long>-74.0142</long>
      </location>
    </metadata>
  </Station>
  <Station ID="8534720" name="Atlantic City">
    <metadata>
      <location>
        <lat>39.3553</lat>
        <long>-74.4181</long>
      </location>
    </metadata>
  </Station>
</Stations>"""

SAMPLE_PREDICTIONS_JSON = {
    "predictions": [
        {"t": "2026-04-15 02:18", "v": "1.524", "type": "H"},
        {"t": "2026-04-15 08:42", "v": "0.012", "type": "L"},
        {"t": "2026-04-15 14:36", "v": "1.311", "type": "H"},
        {"t": "2026-04-15 20:54", "v": "-0.067", "type": "L"},
    ]
}


class TestParseStationList:
    def test_parse_stations(self):
        stations = parse_station_list(SAMPLE_STATION_LIST_XML)
        assert len(stations) == 2
        assert stations[0]["id"] == "8518750"
        assert stations[0]["name"] == "The Battery"
        assert abs(stations[0]["lat"] - 40.7006) < 0.001
        assert abs(stations[0]["lon"] - (-74.0142)) < 0.001


class TestFindNearestStation:
    def test_find_nearest(self):
        stations = [
            {"id": "8518750", "name": "The Battery", "lat": 40.7006, "lon": -74.0142},
            {"id": "8534720", "name": "Atlantic City", "lat": 39.3553, "lon": -74.4181},
        ]
        coord = Coordinate(lat=40.7128, lon=-74.0060)
        station, distance = find_nearest_station(stations, coord)
        assert station["id"] == "8518750"
        assert distance < 2.0  # ~1.5 km

    def test_none_when_too_far(self):
        stations = [
            {"id": "8518750", "name": "The Battery", "lat": 40.7006, "lon": -74.0142},
        ]
        coord = Coordinate(lat=-5.0, lon=-35.0)  # Brazil
        result = find_nearest_station(stations, coord, max_distance_km=25.0)
        assert result is None


class TestParsePredictionsResponse:
    def test_parse_predictions(self):
        events = parse_predictions_response(SAMPLE_PREDICTIONS_JSON)
        assert len(events) == 4
        assert events[0].height == pytest.approx(1.524)
        assert events[0].time.hour == 2
        assert events[0].time.minute == 18
        assert events[1].height == pytest.approx(0.012)
        assert events[3].height == pytest.approx(-0.067)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_noaa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tides.noaa'`

- [ ] **Step 3: Implement NOAA client**

```python
# src/tides/noaa.py
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
        station_id = station_el.get("ID")
        name = station_el.get("name")
        lat_el = station_el.find(".//lat")
        lon_el = station_el.find(".//long")
        if station_id and name and lat_el is not None and lon_el is not None:
            stations.append({
                "id": station_id,
                "name": name,
                "lat": float(lat_el.text),
                "lon": float(lon_el.text),
            })
    return stations


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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


def parse_predictions_response(data: dict) -> list[TideEvent]:
    events = []
    for p in data.get("predictions", []):
        time = datetime.datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
        time = time.replace(tzinfo=datetime.timezone.utc)
        height = float(p["v"])
        events.append(TideEvent(time=time, height=height))
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_noaa.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tides/noaa.py tests/test_noaa.py
git commit -m "feat: NOAA CO-OPS client with station lookup and prediction parsing"
```

---

### Task 6: Ocean Model Wrapper (pyTMD + extrema finding)

**Files:**
- Create: `src/tides/ocean_model.py`
- Create: `tests/test_ocean_model.py`

- [ ] **Step 1: Write failing tests for extrema finding**

The extrema-finding logic is testable without the actual model data. We test the `find_extrema` function with synthetic elevation data.

```python
# tests/test_ocean_model.py
import datetime

import numpy as np
import pytest

from tides.ocean_model import find_extrema


class TestFindExtrema:
    def test_simple_sine_wave(self):
        """A sine wave over 24h should produce ~2 highs and ~2 lows."""
        # 24 hours at 6-minute intervals = 240 points
        n = 240
        times = [
            datetime.datetime(2026, 4, 15, tzinfo=datetime.timezone.utc)
            + datetime.timedelta(minutes=6 * i)
            for i in range(n)
        ]
        # Semi-diurnal tide: ~2 cycles per day (period ~12.42h, use 12h for simplicity)
        elevations = np.sin(np.linspace(0, 4 * np.pi, n))
        events = find_extrema(times, elevations)
        # Should find 2 highs and 2 lows
        heights = [e.height for e in events]
        highs = [h for h in heights if h > 0.5]
        lows = [h for h in heights if h < -0.5]
        assert len(highs) == 2
        assert len(lows) == 2

    def test_events_are_chronological(self):
        n = 240
        times = [
            datetime.datetime(2026, 4, 15, tzinfo=datetime.timezone.utc)
            + datetime.timedelta(minutes=6 * i)
            for i in range(n)
        ]
        elevations = np.sin(np.linspace(0, 4 * np.pi, n))
        events = find_extrema(times, elevations)
        for i in range(len(events) - 1):
            assert events[i].time < events[i + 1].time

    def test_all_nan_returns_empty(self):
        times = [
            datetime.datetime(2026, 4, 15, tzinfo=datetime.timezone.utc)
            + datetime.timedelta(minutes=6 * i)
            for i in range(240)
        ]
        elevations = np.full(240, np.nan)
        events = find_extrema(times, elevations)
        assert events == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ocean_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tides.ocean_model'`

- [ ] **Step 3: Implement ocean model module**

```python
# src/tides/ocean_model.py
import datetime

import numpy as np
from scipy.signal import find_peaks

from tides.models import Coordinate, TideEvent

MODEL_NAME = "GOT5.6"
ELEVATION_INTERVAL_MINUTES = 6


def find_extrema(
    times: list[datetime.datetime],
    elevations: np.ndarray,
) -> list[TideEvent]:
    if np.all(np.isnan(elevations)):
        return []

    # Find highs (peaks) and lows (troughs)
    highs, _ = find_peaks(elevations, distance=20)
    lows, _ = find_peaks(-elevations, distance=20)

    events = []
    for i in highs:
        events.append(TideEvent(time=times[i], height=float(elevations[i])))
    for i in lows:
        events.append(TideEvent(time=times[i], height=float(elevations[i])))

    events.sort(key=lambda e: e.time)
    return events


def compute_tides(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
) -> list[TideEvent]:
    import pyTMD.compute
    from tides.cache import get_model_dir, ensure_model_data

    ensure_model_data()

    # Build time array at 6-minute intervals from begin_date 00:00 to end_date+1 00:00 UTC
    start = datetime.datetime(
        begin_date.year, begin_date.month, begin_date.day, tzinfo=datetime.timezone.utc
    )
    end = datetime.datetime(
        end_date.year, end_date.month, end_date.day, tzinfo=datetime.timezone.utc
    ) + datetime.timedelta(days=1)

    times = []
    current = start
    while current < end:
        times.append(current)
        current += datetime.timedelta(minutes=ELEVATION_INTERVAL_MINUTES)

    # pyTMD expects arrays
    lons = np.full(len(times), coord.lon)
    lats = np.full(len(times), coord.lat)

    elevations = pyTMD.compute.tide_elevations(
        lons,
        lats,
        times,
        DIRECTORY=str(get_model_dir()),
        MODEL="GOT5.6",
        EPSG=4326,
    )

    return find_extrema(times, elevations)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ocean_model.py -v`
Expected: All 3 tests PASS (they only test `find_extrema`, not `compute_tides`).

- [ ] **Step 5: Commit**

```bash
git add src/tides/ocean_model.py tests/test_ocean_model.py
git commit -m "feat: ocean model wrapper with pyTMD and scipy extrema finding"
```

---

### Task 7: Cache — Model Download Integration

**Files:**
- Modify: `src/tides/cache.py`

This task adds the `ensure_model_data()` function that `ocean_model.py` calls, and the `fetch_all()` function for the `fetch-model` subcommand. We defer integration testing to the CLI integration task.

- [ ] **Step 1: Add ensure_model_data and fetch_all to cache.py**

Add these functions to the bottom of `src/tides/cache.py`:

```python
def _got_model_exists() -> bool:
    model_dir = get_model_dir()
    # GOT5.6 files are stored in a subdirectory
    got_dir = model_dir / "GOT5.6"
    return got_dir.exists() and any(got_dir.iterdir())


def ensure_model_data() -> None:
    if _got_model_exists():
        return
    import sys

    import pyTMD.datasets

    print("Downloading GOT5.6 tidal model... this only happens once.", file=sys.stderr)
    pyTMD.datasets.fetch_gsfc_got(model_dir=str(get_model_dir()))


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
    import sys

    print("Fetching NOAA station list...", file=sys.stderr)
    fetch_station_data()
    print("Downloading GOT5.6 tidal model (this may take a while)...", file=sys.stderr)
    ensure_model_data()
    print("Done. All data cached.", file=sys.stderr)
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/tides/cache.py`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/tides/cache.py
git commit -m "feat: model download and station data management in cache module"
```

---

### Task 8: Resolver (Source Selection)

**Files:**
- Create: `src/tides/resolver.py`
- Create: `tests/test_resolver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_resolver.py
import datetime
from unittest.mock import MagicMock, patch

import pytest

from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.resolver import resolve_tides


SAMPLE_STATIONS = [
    {"id": "8518750", "name": "The Battery", "lat": 40.7006, "lon": -74.0142},
]

SAMPLE_NOAA_EVENTS = [
    TideEvent(
        time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
        height=0.3,
    ),
    TideEvent(
        time=datetime.datetime(2026, 4, 15, 20, 45, tzinfo=datetime.timezone.utc),
        height=-0.1,
    ),
]


class TestResolveNoaa:
    @patch("tides.resolver.fetch_predictions")
    @patch("tides.resolver.parse_predictions_response")
    @patch("tides.resolver.get_stations")
    def test_auto_selects_noaa_when_near_station(
        self, mock_get_stations, mock_parse, mock_fetch
    ):
        mock_get_stations.return_value = SAMPLE_STATIONS
        mock_fetch.return_value = {"predictions": []}
        mock_parse.return_value = SAMPLE_NOAA_EVENTS

        coord = Coordinate(lat=40.7128, lon=-74.0060)
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.AUTO,
        )

        assert result.source_type == Source.NOAA
        assert result.station_name == "The Battery"
        assert len(result.days) == 1
        assert len(result.days[0].events) == 2

    @patch("tides.resolver.compute_tides")
    @patch("tides.resolver.get_stations")
    def test_auto_falls_back_to_model(self, mock_get_stations, mock_compute):
        mock_get_stations.return_value = SAMPLE_STATIONS
        mock_compute.return_value = SAMPLE_NOAA_EVENTS  # reuse for simplicity

        coord = Coordinate(lat=-5.0, lon=-35.0)  # Brazil — far from any US station
        result = resolve_tides(
            coord,
            begin_date=datetime.date(2026, 4, 15),
            end_date=datetime.date(2026, 4, 15),
            source=Source.AUTO,
        )

        assert result.source_type == Source.MODEL
        assert result.model_name == "GOT5.6"

    @patch("tides.resolver.get_stations")
    def test_noaa_source_errors_when_no_station(self, mock_get_stations):
        mock_get_stations.return_value = SAMPLE_STATIONS

        coord = Coordinate(lat=-5.0, lon=-35.0)
        with pytest.raises(SystemExit):
            resolve_tides(
                coord,
                begin_date=datetime.date(2026, 4, 15),
                end_date=datetime.date(2026, 4, 15),
                source=Source.NOAA,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tides.resolver'`

- [ ] **Step 3: Implement resolver**

```python
# src/tides/resolver.py
import datetime
import sys

from tides.cache import get_stations
from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.noaa import (
    fetch_predictions,
    find_nearest_station,
    parse_predictions_response,
)
from tides.ocean_model import MODEL_NAME, compute_tides


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
    max_distance_km: float = 25.0,
) -> TideResult | None:
    result = find_nearest_station(stations, coord, max_distance_km)
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


def _resolve_model(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
) -> TideResult:
    events = compute_tides(coord, begin_date, end_date)
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
        model_name=MODEL_NAME,
        days=days,
    )


def resolve_tides(
    coord: Coordinate,
    begin_date: datetime.date,
    end_date: datetime.date,
    source: Source = Source.AUTO,
) -> TideResult:
    if source == Source.NOAA:
        stations = get_stations()
        result = _resolve_noaa(coord, begin_date, end_date, stations)
        if result is None:
            print(
                "Error: No NOAA tide station found within 25km of this location.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return result

    if source == Source.MODEL:
        return _resolve_model(coord, begin_date, end_date)

    # AUTO: try NOAA first, fall back to model
    stations = get_stations()
    noaa_result = _resolve_noaa(coord, begin_date, end_date, stations)
    if noaa_result is not None:
        return noaa_result
    return _resolve_model(coord, begin_date, end_date)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_resolver.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tides/resolver.py tests/test_resolver.py
git commit -m "feat: source selection resolver (auto/noaa/model dispatch)"
```

---

### Task 9: CLI — Coordinate Parsing and Date Parsing

**Files:**
- Create: `src/tides/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for coordinate and date parsing**

```python
# tests/test_cli.py
import datetime

import pytest

from tides.cli import parse_coordinate, parse_date_arg
from tides.models import Coordinate


class TestParseCoordinate:
    def test_comma_separated_no_space(self):
        c = parse_coordinate(["40.7128,-74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_comma_separated_with_space(self):
        c = parse_coordinate(["40.7128,", "-74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_comma_space_in_quotes(self):
        c = parse_coordinate(["40.7128, -74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_two_positional_args(self):
        c = parse_coordinate(["40.7128", "-74.0060"])
        assert c.lat == pytest.approx(40.7128)
        assert c.lon == pytest.approx(-74.0060)

    def test_invalid_single_arg(self):
        with pytest.raises(SystemExit):
            parse_coordinate(["notacoord"])

    def test_empty_args(self):
        with pytest.raises(SystemExit):
            parse_coordinate([])

    def test_latitude_out_of_range(self):
        with pytest.raises(SystemExit):
            parse_coordinate(["95.0,-74.0060"])


class TestParseDateArg:
    def test_single_date(self):
        begin, end = parse_date_arg("2026-04-15")
        assert begin == datetime.date(2026, 4, 15)
        assert end == datetime.date(2026, 4, 15)

    def test_date_range(self):
        begin, end = parse_date_arg("2026-04-15:2026-04-17")
        assert begin == datetime.date(2026, 4, 15)
        assert end == datetime.date(2026, 4, 17)

    def test_none_defaults_to_today(self):
        begin, end = parse_date_arg(None)
        today = datetime.date.today()
        assert begin == today
        assert end == today

    def test_invalid_date(self):
        with pytest.raises(SystemExit):
            parse_date_arg("not-a-date")

    def test_end_before_begin(self):
        with pytest.raises(SystemExit):
            parse_date_arg("2026-04-17:2026-04-15")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement CLI with parsing functions and Typer app**

```python
# src/tides/cli.py
import datetime
import json
import sys
from typing import Optional

import typer

from tides import __version__
from tides.models import Coordinate, Source, TideResult

app = typer.Typer(
    name="tides",
    help="Tide predictions for any coastal coordinate.\n\nExamples:\n\n  tides 40.7128,-74.0060\n\n  tides 40.7128 -74.0060 --date 2026-04-15\n\n  tides 35.9,-75.6 --local --feet\n\nhttps://github.com/natecostello/tide-predictor",
    add_completion=False,
    context_settings={"allow_interspersed_args": True},
)


def parse_coordinate(args: list[str]) -> Coordinate:
    if not args:
        print(
            "Error: Could not parse coordinates. Expected: lat,lon (e.g. 40.7128,-74.0060)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Join all args and try to split on comma
    joined = " ".join(args)
    # Remove trailing/leading whitespace around commas
    joined = joined.replace(" ,", ",").replace(", ", ",")

    if "," in joined:
        parts = joined.split(",")
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                return Coordinate(lat=lat, lon=lon)
            except (ValueError, TypeError):
                pass

    # Try as two separate float args
    if len(args) == 2:
        try:
            lat, lon = float(args[0]), float(args[1])
            return Coordinate(lat=lat, lon=lon)
        except (ValueError, TypeError):
            pass

    print(
        "Error: Could not parse coordinates. Expected: lat,lon (e.g. 40.7128,-74.0060)",
        file=sys.stderr,
    )
    raise SystemExit(1)


def parse_date_arg(date_str: str | None) -> tuple[datetime.date, datetime.date]:
    if date_str is None:
        today = datetime.date.today()
        return today, today

    if ":" in date_str:
        parts = date_str.split(":")
        if len(parts) != 2:
            print(
                "Error: Invalid date format. Expected: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            begin = datetime.date.fromisoformat(parts[0])
            end = datetime.date.fromisoformat(parts[1])
        except ValueError:
            print(
                "Error: Invalid date format. Expected: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if end < begin:
            print("Error: End date must not be before begin date.", file=sys.stderr)
            raise SystemExit(1)
        return begin, end

    try:
        d = datetime.date.fromisoformat(date_str)
        return d, d
    except ValueError:
        print(
            "Error: Invalid date format. Expected: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD",
            file=sys.stderr,
        )
        raise SystemExit(1)


def parse_between(between_str: str | None) -> tuple[datetime.time, datetime.time] | None:
    if between_str is None:
        return None
    # Expected format: HH:MM:HH:MM (4 colon-separated parts)
    # Actually: HH:MM<sep>HH:MM — we need to split into two time parts
    # Format is HH:MM:HH:MM which is ambiguous with colons in times
    # Split on all colons, expect 4 parts
    parts = between_str.split(":")
    if len(parts) != 4:
        print(
            "Error: Invalid --between format. Expected: HH:MM:HH:MM (e.g. 06:00:18:00)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        start = datetime.time(int(parts[0]), int(parts[1]))
        end = datetime.time(int(parts[2]), int(parts[3]))
    except (ValueError, TypeError):
        print(
            "Error: Invalid --between format. Expected: HH:MM:HH:MM (e.g. 06:00:18:00)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return start, end


def format_plain(
    result: TideResult,
    feet: bool,
    precision: int,
    local: bool,
    between: tuple[datetime.time, datetime.time] | None,
    verbose: bool,
) -> str:
    from tides.timezone import get_timezone_name, to_local_time

    lines = []
    multi_day = len(result.days) > 1
    unit = "ft" if feet else "m"

    verbose_prefix = ""
    if verbose and result.source_type == Source.NOAA and result.station_name:
        verbose_prefix = f"[NOAA: {result.station_name}, {result.station_distance_km}km] "
    elif verbose and result.source_type == Source.MODEL and result.model_name:
        verbose_prefix = f"[Model: {result.model_name}] "

    for day in result.days:
        event_strs = []
        for event in day.events:
            if local:
                display_time = to_local_time(event.time, result.coordinate)
            else:
                display_time = event.time

            time_str = display_time.strftime("%H:%M")

            # Apply between filter
            if between is not None:
                t = display_time.time().replace(second=0, microsecond=0)
                if not (between[0] <= t <= between[1]):
                    continue

            height = event.height_ft if feet else event.height
            height_str = f"{height:.{precision}f}{unit}"
            event_strs.append(f"{height_str}@{time_str}")

        if not event_strs:
            continue

        tide_str = ", ".join(event_strs)
        if multi_day:
            lines.append(f"{day.date}: {verbose_prefix}{tide_str}")
        else:
            lines.append(f"{verbose_prefix}{tide_str}")

    return "\n".join(lines)


def format_json(
    result: TideResult,
    feet: bool,
    precision: int,
    local: bool,
    between: tuple[datetime.time, datetime.time] | None,
) -> str:
    from tides.timezone import get_timezone_name, to_local_time

    unit = "ft" if feet else "m"
    tz_name = "UTC"
    if local:
        tz_name = get_timezone_name(result.coordinate) or "UTC"

    source_obj: dict = {"type": result.source_type.value}
    if result.source_type == Source.NOAA and result.station_id:
        source_obj["station"] = {
            "id": result.station_id,
            "name": result.station_name,
            "distance_km": result.station_distance_km,
        }

    days_list = []
    for day in result.days:
        tides_list = []
        for event in day.events:
            if local:
                display_time = to_local_time(event.time, result.coordinate)
            else:
                display_time = event.time

            time_str = display_time.strftime("%H:%M")

            if between is not None:
                t = display_time.time().replace(second=0, microsecond=0)
                if not (between[0] <= t <= between[1]):
                    continue

            height = event.height_ft if feet else event.height
            tides_list.append({
                "time": time_str,
                "height": round(height, precision),
            })

        if tides_list:
            days_list.append({
                "date": day.date.isoformat(),
                "tides": tides_list,
            })

    output = {
        "coordinate": {"lat": result.coordinate.lat, "lon": result.coordinate.lon},
        "source": source_obj,
        "model": result.model_name,
        "timezone": tz_name,
        "unit": unit,
        "days": days_list,
    }
    return json.dumps(output, indent=2)


@app.command()
def main(
    coordinate: list[str] = typer.Argument(None, help="Latitude,longitude (e.g. 40.7128,-74.0060)"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Date or range: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD"),
    local: bool = typer.Option(False, "--local", "-l", help="Display times in local timezone"),
    feet: bool = typer.Option(False, "--feet", "-f", help="Display heights in feet"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    between: Optional[str] = typer.Option(None, "--between", "-b", help="Time filter: HH:MM:HH:MM"),
    precision: int = typer.Option(1, "--precision", "-p", help="Decimal places for height"),
    source: str = typer.Option("auto", "--source", "-s", help="Data source: auto, noaa, model"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show source details"),
    version: bool = typer.Option(False, "--version", help="Show version"),
) -> None:
    """Tide predictions for any coastal coordinate."""
    if version:
        print(f"tides {__version__}")
        raise typer.Exit()

    if not coordinate:
        # Show help when no args provided
        ctx = typer.Context(main)
        print(ctx.get_help())
        raise typer.Exit()

    coord = parse_coordinate(coordinate)
    begin_date, end_date = parse_date_arg(date)
    between_times = parse_between(between)

    try:
        source_enum = Source(source.lower())
    except ValueError:
        print(
            f"Error: Invalid source '{source}'. Expected: auto, noaa, model",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from tides.resolver import resolve_tides

    try:
        result = resolve_tides(coord, begin_date, end_date, source_enum)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)

    if json_output:
        print(format_json(result, feet, precision, local, between_times))
    else:
        output = format_plain(result, feet, precision, local, between_times, verbose)
        if output:
            print(output)


@app.command("fetch-model")
def fetch_model() -> None:
    """Pre-download tidal model data and station metadata."""
    from tides.cache import fetch_all

    try:
        fetch_all()
    except Exception as e:
        print(f"Error: Could not download tidal data. {e}", file=sys.stderr)
        raise SystemExit(2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/tides/ tests/`
Expected: No errors (or minor fixable issues).

- [ ] **Step 6: Commit**

```bash
git add src/tides/cli.py tests/test_cli.py
git commit -m "feat: Typer CLI with coordinate/date parsing and output formatting"
```

---

### Task 10: CLI Output Formatting Tests

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add output formatting tests**

Append to `tests/test_cli.py`:

```python
import json as json_module

from tides.cli import format_json, format_plain, parse_between
from tides.models import Source, TideDay, TideEvent, TideResult


class TestParseBetween:
    def test_valid_between(self):
        result = parse_between("06:00:18:00")
        assert result == (datetime.time(6, 0), datetime.time(18, 0))

    def test_none(self):
        assert parse_between(None) is None

    def test_invalid_format(self):
        with pytest.raises(SystemExit):
            parse_between("invalid")


class TestFormatPlain:
    def _make_result(self) -> TideResult:
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
                height=0.3,
            ),
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 20, 45, tzinfo=datetime.timezone.utc),
                height=-0.1,
            ),
        ]
        return TideResult(
            coordinate=Coordinate(lat=40.7128, lon=-74.0060),
            source_type=Source.NOAA,
            station_id="8518750",
            station_name="The Battery",
            station_distance_km=1.2,
            model_name=None,
            days=[TideDay(date=datetime.date(2026, 4, 15), events=events)],
        )

    def test_default_meters(self):
        result = self._make_result()
        output = format_plain(result, feet=False, precision=1, local=False, between=None, verbose=False)
        assert output == "0.3m@14:32, -0.1m@20:45"

    def test_feet(self):
        result = self._make_result()
        output = format_plain(result, feet=True, precision=1, local=False, between=None, verbose=False)
        assert "ft@" in output

    def test_verbose_noaa(self):
        result = self._make_result()
        output = format_plain(result, feet=False, precision=1, local=False, between=None, verbose=True)
        assert output.startswith("[NOAA: The Battery, 1.2km]")

    def test_precision(self):
        result = self._make_result()
        output = format_plain(result, feet=False, precision=3, local=False, between=None, verbose=False)
        assert "0.300m@14:32" in output


class TestFormatJson:
    def _make_result(self) -> TideResult:
        events = [
            TideEvent(
                time=datetime.datetime(2026, 4, 15, 14, 32, tzinfo=datetime.timezone.utc),
                height=0.3,
            ),
        ]
        return TideResult(
            coordinate=Coordinate(lat=40.7128, lon=-74.0060),
            source_type=Source.NOAA,
            station_id="8518750",
            station_name="The Battery",
            station_distance_km=1.2,
            model_name=None,
            days=[TideDay(date=datetime.date(2026, 4, 15), events=events)],
        )

    def test_json_structure(self):
        result = self._make_result()
        output = format_json(result, feet=False, precision=1, local=False, between=None)
        parsed = json_module.loads(output)
        assert parsed["coordinate"] == {"lat": 40.7128, "lon": -74.006}
        assert parsed["source"]["type"] == "noaa"
        assert parsed["source"]["station"]["name"] == "The Battery"
        assert parsed["unit"] == "m"
        assert parsed["timezone"] == "UTC"
        assert len(parsed["days"]) == 1
        assert parsed["days"][0]["tides"][0]["height"] == 0.3

    def test_json_model_source(self):
        result = TideResult(
            coordinate=Coordinate(lat=-5.0, lon=-35.0),
            source_type=Source.MODEL,
            station_id=None,
            station_name=None,
            station_distance_km=None,
            model_name="GOT5.6",
            days=[],
        )
        output = format_json(result, feet=False, precision=1, local=False, between=None)
        parsed = json_module.loads(output)
        assert parsed["source"]["type"] == "model"
        assert parsed["model"] == "GOT5.6"
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests PASS (original 12 + new ~8).

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: add output formatting and between-filter tests"
```

---

### Task 11: Integration Smoke Test and README

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`

- [ ] **Step 1: Write integration test (marks it as requiring network)**

```python
# tests/test_integration.py
"""Integration tests that require network access and/or model data.

Run with: pytest tests/test_integration.py -v -m integration
Skip in CI by default.
"""
import datetime
import json
import subprocess

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.integration
class TestNOAAIntegration:
    def test_battery_ny(self):
        """The Battery, NY — a well-known NOAA station."""
        result = subprocess.run(
            ["tides", "40.7006,-74.0142", "--date", "2026-04-15", "--source", "noaa", "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["source"]["type"] == "noaa"
        assert len(data["days"]) == 1
        assert len(data["days"][0]["tides"]) >= 2

    def test_battery_verbose(self):
        result = subprocess.run(
            ["tides", "40.7006,-74.0142", "--date", "2026-04-15", "--source", "noaa", "--verbose"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "[NOAA:" in result.stdout


@pytest.mark.integration
class TestModelIntegration:
    def test_brazil_coast(self):
        """NE Brazil coast — no NOAA station, forces model."""
        result = subprocess.run(
            ["tides", "-5.0,-35.0", "--date", "2026-04-15", "--source", "model", "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["source"]["type"] == "model"
        assert data["model"] == "GOT5.6"
        assert len(data["days"][0]["tides"]) >= 2


@pytest.mark.integration
class TestCLIFlags:
    def test_local_time(self):
        result = subprocess.run(
            ["tides", "40.7006,-74.0142", "--date", "2026-04-15", "--source", "noaa", "--local", "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["timezone"] != "UTC"

    def test_feet(self):
        result = subprocess.run(
            ["tides", "40.7006,-74.0142", "--date", "2026-04-15", "--source", "noaa", "--feet"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ft@" in result.stdout

    def test_date_range(self):
        result = subprocess.run(
            ["tides", "40.7006,-74.0142", "--date", "2026-04-15:2026-04-16", "--source", "noaa"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "2026-04-15:" in result.stdout
        assert "2026-04-16:" in result.stdout

    def test_version(self):
        result = subprocess.run(
            ["tides", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "tides" in result.stdout
```

- [ ] **Step 2: Add integration marker to pytest config**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
    "integration: tests requiring network access or model data",
]
```

- [ ] **Step 3: Create README.md**

```markdown
# tides

A command-line tool for tide predictions using NOAA station data and the GOT5.6 global tidal model.

## Install

```bash
uv tool install git+https://github.com/natecostello/tide-predictor.git
```

Or for development:

```bash
git clone https://github.com/natecostello/tide-predictor.git
cd tide-predictor
uv venv && uv pip install -e ".[dev]"
```

## Usage

```bash
# Today's tides at a location
tides 40.7128,-74.0060

# Specific date
tides 40.7128,-74.0060 --date 2026-04-15

# Date range with local times and feet
tides 35.9,-75.6 --date 2026-04-15:2026-04-17 --local --feet

# JSON output
tides 40.7128,-74.0060 --json

# Only daytime tides
tides 40.7128,-74.0060 --between 06:00:18:00

# Force NOAA station data
tides 40.7128,-74.0060 --source noaa

# Force global model
tides -5.0,-35.0 --source model

# Pre-download model data
tides fetch-model
```

## Options

| Flag | Short | Description |
|------|-------|-------------|
| `--date` | `-d` | Date or range (YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD) |
| `--local` | `-l` | Times in local timezone at coordinates |
| `--feet` | `-f` | Heights in feet (default: meters) |
| `--json` | `-j` | JSON output |
| `--between` | `-b` | Time window filter (HH:MM:HH:MM) |
| `--precision` | `-p` | Decimal places for height (default: 1) |
| `--source` | `-s` | Data source: auto, noaa, model (default: auto) |
| `--verbose` | `-v` | Show source details |
| `--version` | | Show version |

## Data Sources

**NOAA CO-OPS** (US waters): Uses official tide station predictions. Auto-selected when a station is within 25km of the coordinates.

**GOT5.6** (global): NASA Goddard Ocean Tide model. Used as fallback for locations outside NOAA coverage. Model data is auto-downloaded on first use (~hundreds of MB).

## Development

```bash
uv run pytest                              # unit tests
uv run pytest tests/test_integration.py -m integration  # integration tests
uv run ruff check src/ tests/              # lint
uv run ruff format src/ tests/             # format
```

## License

MIT
```

- [ ] **Step 4: Run unit tests (not integration)**

Run: `uv run pytest tests/ -v --ignore=tests/test_integration.py`
Expected: All unit tests PASS.

- [ ] **Step 5: Run integration tests**

Run: `uv run pytest tests/test_integration.py -v -m integration`
Expected: All integration tests PASS (requires network; model download may take a while on first run).

- [ ] **Step 6: Lint everything**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: No errors, no formatting changes.

- [ ] **Step 7: Commit**

```bash
git add tests/test_integration.py README.md pyproject.toml
git commit -m "feat: integration tests and README"
```

---

### Task 12: Final Cleanup and PR

- [ ] **Step 1: Push and create PR**

```bash
git push -u origin feat/initial-implementation
gh pr create --title "feat: tides CLI with NOAA and GOT5.6 support" --body "$(cat <<'EOF'
## Summary
- Typer CLI accepting lat/lon coordinates with flexible parsing
- Hybrid data source: NOAA CO-OPS stations (US, auto-selected within 25km) + GOT5.6 global model
- Options: --local (timezone), --feet, --json, --between (time filter), --precision, --source, --verbose
- XDG-compliant cache with auto-download on first use
- `tides fetch-model` subcommand for offline prep

## Test plan
- [ ] Unit tests pass: `uv run pytest tests/ --ignore=tests/test_integration.py -v`
- [ ] Integration tests pass: `uv run pytest tests/test_integration.py -m integration -v`
- [ ] Lint passes: `uv run ruff check src/ tests/`
- [ ] Manual test: `tides 40.7128,-74.0060` returns tide data
- [ ] Manual test: `tides 40.7128,-74.0060 --json --local --feet` returns formatted JSON
- [ ] Manual test: `tides -5.0,-35.0 --source model` uses GOT5.6

## Plan compliance
- [x] Task 1: Project scaffolding
- [x] Task 2: Data models
- [x] Task 3: Cache management
- [x] Task 4: Timezone module
- [x] Task 5: NOAA client
- [x] Task 6: Ocean model wrapper
- [x] Task 7: Cache download integration
- [x] Task 8: Resolver
- [x] Task 9: CLI parsing
- [x] Task 10: Output formatting tests
- [x] Task 11: Integration tests and README

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Request Copilot review**

Add Copilot as a reviewer on the PR.
