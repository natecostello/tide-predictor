# Tides CLI — Design Spec

## Overview

`tides` is a stateless command-line tool that returns high and low tide predictions for any coastal coordinate. It uses a hybrid data strategy: NOAA CO-OPS station predictions for US locations (accurate for estuaries, sounds, and rivers) with fallback to the GOT5.6 global tidal model for worldwide coverage.

Built with Python, uv, and Typer.

## Invocation

```
tides <coordinate> [OPTIONS]
```

### Coordinate Parsing

The coordinate is the primary positional argument. The parser accepts these forms:

| Input | Interpretation |
|-------|---------------|
| `40.7128,-74.0060` | Single comma-separated arg, no space |
| `40.7128, -74.0060` | Quoted with space (shell splits otherwise) |
| `"40.7128 -74.0060"` | One quoted token with whitespace (amended #18; two separate unquoted args are not supported -- the CLI takes a single coordinate argument) |

Order is always `latitude, longitude`. (Amended 2026-09-27, #18: the coordinate is a single argument, `lat,lon` or quoted `"lat lon"`; a bare negative latitude such as `-2.88,-39.91` or `"-2.88 -39.91"` is accepted directly.)

Note: a token starting with `-` may be interpreted as a flag by Click. The comma form (`40.7128,-74.0060`) and the quoted form (`"40.7128 -74.0060"`) keep the coordinate a single argument; a leading negative latitude (`-2.88,-39.91`, `"-2.88 -39.91"`) is escaped by the `main_entry` wrapper. (Amended 2026-09-27, #18: the original suggestion `tides -- 40.7128 -74.0060` passes two arguments and is not supported.)

### Date

- (Amended 2026-09-27, #18) Ranges are limited to 366 days inclusive; NOAA's own hilo limit is 3,655 days.
- (Amended 2026-09-27, #18) `-h`/`--help` work on every command and bare `tides` shows help.

- Default: today (UTC date at time of invocation; with `--local`, today's date in the coordinate's timezone -- amended 2026-09-27, #10)
- Single date: `--date 2026-04-15` (ISO 8601)
- Date range: `--date 2026-04-15:2026-04-17` (inclusive)

## Data Sources

### Source Selection

Controlled by `--source` flag:

| Value | Behavior |
|-------|----------|
| `auto` (default) | Find nearest NOAA station within threshold distance. If found, use it. Otherwise, use GOT5.6 model. |
| `noaa` | Force NOAA station lookup. Error if no station within threshold. |
| `model` | Force GOT5.6 global model. Skip NOAA entirely. |

**Distance threshold:** 10-25km (configurable internally, not user-facing). Start with 25km as default.

> **Amended 2026-09-27 (#12):** `auto` now tries NOAA, then the global station database, then the model. A NOAA station that cannot serve the requested `--datum` (subordinate stations publish MLLW only), or a NOAA API/network failure while fetching predictions or station datums, falls through to the next source with a stderr note instead of failing. `--source noaa` still reports the error (exit 2).

> **Amended 2026-09-27 (#16):** the fall-through also covers the NOAA station-list fetch (a stale cached list is reused with a warning when a refresh fails) and the GitHub station-database download (auto falls through to the model). Explicit `--source` values report these errors (exit 2), naming the service that actually failed.

### NOAA CO-OPS API

- Free, no authentication required
- ~3,000+ US tide prediction stations
- Returns high/low tide predictions directly (no extrema computation needed)
- Station metadata (ID, name, lat/lon) cached locally for nearest-station lookup
- API endpoint: `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`

### pyTMD + GOT5.6

- Global coverage at 0.25° resolution, 16 tidal constituents
- Programmatic download via `pyTMD.datasets.fetch_gsfc_got()`
- Compute tide elevations at 6-minute intervals, find extrema with `scipy.signal.find_peaks()`
- Handles arbitrary lat/long worldwide

### Limitations

- GOT5.6 at 0.25° resolution will not accurately model semi-enclosed bodies of water (sounds, estuaries), tidal rivers, or fine coastal features
- NOAA stations cover US waters only
- Heights from the global model will differ from official tide station predictions
- For navigation or engineering, users should consult official sources

## Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--date` | `-d` | `str` | today | Date or date range (ISO 8601, colon-separated for range) |
| `--local` | `-l` | `bool` | `False` | Display times in local timezone at the coordinates |
| `--feet` | `-f` | `bool` | `False` | Display heights in feet instead of meters |
| `--json` | `-j` | `bool` | `False` | Output as structured JSON |
| `--between` | `-b` | `str` | `None` | Time window filter `HH:MM:HH:MM` (24h, applied per-day; a start after the end wraps midnight, e.g. `20:00:04:00` -- #18) |
| `--precision` | `-p` | `int` | `1` | Decimal places for height values |
| `--source` | `-s` | `str` | `auto` | Data source: `auto`, `noaa`, `station`, `model` |
| `--model` | `-m` | `str` | `got5.6` | Tide model: `got5.6`, `got5.5`, `eot20`, `fes2022`; a stderr note says when it was ignored because a NOAA/station source answered (#18) |
| `--datum` | | `str` | `mllw` | Height datum: `lat`, `mllw`, `mlw`, `msl`, `mtl`, `mhw`, `mhhw`, `hat` |
| `--verbose` | `-v` | `bool` | `False` | Show source details (station name, distance) |

## Output

### Plain Text (default)

Single day:
```
0.3m@14:32, -0.1m@20:45, 0.8m@02:15, 0.1m@08:50
```

With `--feet`:
```
1.0ft@14:32, -0.3ft@20:45, 2.6ft@02:15, 0.3ft@08:50
```

Multi-day (date range):
```
2026-04-15: 0.3m@14:32, -0.1m@20:45
2026-04-16: 0.8m@02:15, 0.1m@08:50, 0.7m@14:55, 0.0m@21:10
```

Times are UTC by default. With `--local`, times are in the local timezone at the coordinates, with no timezone suffix.

> **Amended 2026-09-27 (#10):** with `--local`, days are also grouped on the local clock: `--date` bounds refer to local dates and each day lists the events that fall on that local date (previously events were grouped by UTC date, misfiling evening tides at negative offsets).

Tides are listed in chronological order (midnight to midnight).

With `--verbose`:
```
[NOAA: Oregon Inlet, 3.2km, MLLW] 0.3m@14:32, -0.1m@20:45
```

### JSON

```json
{
  "coordinate": {"lat": 40.7128, "lon": -74.006},
  "source": {
    "type": "noaa",
    "station": {"id": "8518750", "name": "The Battery", "distance_km": 1.2}
  },
  "model": null,
  "datum": "MLLW",
  "timezone": "UTC",
  "unit": "m",
  "days": [
    {
      "date": "2026-04-15",
      "tides": [
        {"time": "14:32", "height": 0.3, "datetime": "2026-04-15T14:32+00:00", "type": "high"},
        {"time": "20:45", "height": -0.1, "datetime": "2026-04-15T20:45+00:00", "type": "low"}
      ]
    }
  ]
}
```

> **Amended 2026-09-27 (#19):** each event also carries `datetime` (ISO 8601 with UTC offset, on the displayed clock) and `type` (`"high"`/`"low"`); existing keys are unchanged. Every requested day is present, with `"tides": []` when `--between` filtered out all its events. `--verbose` prefixes include the datum. Plain output with no matching events prints nothing on stdout and a note on stderr (exit 0). Rounded heights never print as `-0.0`.

When source is `model`:
```json
{
  "coordinate": {"lat": -5.0, "lon": -35.0},
  "source": {"type": "model"},
  "model": "GOT5.6",
  "timezone": "America/Recife",
  "unit": "ft",
  "days": [...]
}
```

The `timezone` field reflects the actual timezone used: `"UTC"` by default, or the IANA timezone name when `--local` is used.

### Time Filtering (`--between`)

`--between 06:00:18:00` filters output to only tides whose time falls within the window. Applied per-day for multi-day ranges. When `--local` is active, the filter applies to local times. (Amended 2026-09-27, #18: when the start is after the end, the window wraps midnight, e.g. `20:00:04:00` keeps 20:00-23:59 and 00:00-04:00.)

## Subcommands

### `tides fetch-model`

Pre-downloads everything a query needs offline: the NOAA station list (always refreshed), the global tide station database (a missing or invalid index is rebuilt from on-disk station files if possible, otherwise downloaded) and the GOT5.6 model (downloaded only if missing). Amended 2026-09-27 (#16); originally GOT5.6 and NOAA metadata only.

```
tides fetch-model
```

Displays download progress and one status line per item: the station database reports "already present", "index rebuilt from N cached station files", or its download progress; GOT5.6 reports "already present" or its download.

## Data Management

- Cache location: XDG-compliant (`$XDG_CACHE_HOME/tides/` or `~/.cache/tides/`)
- GOT5.6 model files: downloaded via `pyTMD.datasets.fetch_gsfc_got()`
- NOAA station list: fetched from CO-OPS API, cached as JSON, refreshed periodically (e.g. monthly)
- On first invocation, auto-downloads required data with a progress message:
  ```
  Downloading GOT5.6 tidal model... this only happens once.
  ```

## Error Handling

| Condition | Behavior |
|-----------|----------|
| Coordinates on land | "No tidal data for this location — it may be inland." |
| Model coverage gap | "No tidal model coverage for this location." |
| No NOAA station in range (with `--source noaa`) | "No NOAA tide station found within 25km of this location." |
| Invalid coordinate format | "Could not parse coordinates. Expected: lat,lon (e.g. 40.7128,-74.0060)" |
| Invalid date format | "Invalid date format. Expected: YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD" |
| Network error during data fetch | Names the failing service (amended 2026-09-27, #16): "Could not connect to <host>. Check your internet connection.", "<host> returned HTTP <code>. ...", or "Could not download the global tide station database from GitHub (...)". In `auto` mode these fall through to the next source instead. |
| Unexpected internal error | "unexpected <Type> ...: <message>"; `TIDES_DEBUG=1` adds the traceback |
| NaN/invalid results from model | "No tidal data for this location — it may be inland." |

All errors go to stderr. Exit code 1 for user errors, 2 for data/network errors.

## Dependencies

| Package | Purpose |
|---------|---------|
| typer | CLI framework |
| pyTMD | Tidal model computation |
| scipy | Peak finding for model-based extrema |
| numpy | Numerical arrays (pyTMD dependency) |
| timezonefinder | Lat/long to timezone mapping |
| httpx | HTTP client for NOAA API |
| rich | Progress bars and formatted output (Typer companion) |

## Project Structure

```
tide-predictor/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── src/
│   └── tides/
│       ├── __init__.py
│       ├── cli.py          # Typer app, argument parsing, output formatting
│       ├── models.py        # Data classes for tides, coordinates, etc.
│       ├── noaa.py          # NOAA CO-OPS API client
│       ├── ocean_model.py   # pyTMD wrapper, extrema finding
│       ├── resolver.py      # Source selection logic (auto/noaa/model)
│       ├── cache.py         # XDG cache management, data downloads
│       └── timezone.py      # Coordinate to timezone mapping
├── tests/
│   └── ...
└── docs/
    └── ...
```

## Testing Strategy

- Unit tests for coordinate parsing, date parsing, time filtering, output formatting
- Unit tests for NOAA API response parsing (mocked HTTP)
- Unit tests for extrema finding from model elevation arrays
- Integration test with GOT5.6 for a known coastal location (requires model data)
- Integration test with NOAA API for a known station
- Error case tests for land coordinates, invalid input, etc.
