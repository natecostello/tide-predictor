# Architecture Walkthrough

## Overview

`tides` is a stateless CLI that predicts high/low tides for any coastal coordinate. It uses a three-tier resolution strategy: NOAA API (US waters) > global harmonic stations (~8,300 worldwide) > gridded ocean models (global). All heights are converted to a user-selected datum (default: MLLW).

## Data Flow

```
User: tides get 40.7,-74.0 --feet --local

  cli.py          Parse args, validate
    |
  resolver.py     Pick data source (auto/noaa/station/model)
    |
    ├── noaa.py          NOAA API → hi/lo predictions (US, <25km)
    ├── stations.py      Station harmonics → pyTMD prediction (global, <200km)
    │     └── harmonics.py   Build xarray Dataset → predict.time_series()
    └── ocean_model.py   Gridded model → pyTMD prediction (global fallback)
    |
  datums.py       Convert heights to requested datum (MLLW, LAT, etc.)
    |
  cli.py          Format output (plain text or JSON)
```

## Modules

### cli.py — Entry point

Typer app with three subcommands: `get`, `cache`, `fetch-model`. Handles argument parsing, source/model/datum validation, error wrapping (no raw tracebacks), and output formatting. Two formatters: `format_plain` (default, `height@time` format) and `format_json` (structured with coordinate, source, datum, timezone metadata).

### resolver.py — Source selection and datum conversion

The core routing logic. `resolve_tides()` tries sources in order for `auto` mode:
1. **NOAA API** — if a station is within 25 km
2. **Global station database** — if a station is within 200 km
3. **Gridded model** — always available (fallback)

After getting a result, `_apply_datum()` converts heights from the source's native datum to the requested datum. This is the trickiest part of the codebase because each source uses a different native datum:
- **NOAA**: heights relative to MTL (mean tide level)
- **Station**: heights relative to chart datum (LAT or MLLW, varies per station)
- **Model**: heights relative to MSL (mean sea level)

The conversion formula: `height_target = height_current - (target_offset - current_offset)`, where offsets come from station datum tables or a 19-year model computation.

### noaa.py — NOAA CO-OPS API client

Two endpoints:
- Station list (XML, cached 30 days): `api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.xml`
- Predictions (JSON, not cached): `api.tidesandcurrents.noaa.gov/api/prod/datagetter`

NOAA predictions are the gold standard for US waters — they come from the agency's own harmonic analysis of decades of tide gauge data.

### stations.py — Global station database

Downloads the [openwatersio/tide-database](https://github.com/openwatersio/tide-database) (~8,289 stations from NOAA and TICON sources) as a GitHub zip archive. Each station is a JSON file with harmonic constituents, datum offsets, and metadata.

`predict_station_tides()` calls into `harmonics.py` to generate predictions from the station's constituents, applying the chart datum offset so heights are relative to the station's published datum (LAT or MLLW).

### harmonics.py — Harmonic prediction engine

Converts station harmonic constituents (amplitude + phase per constituent) into an xarray Dataset compatible with pyTMD's `predict.time_series()`. This is the same prediction pipeline used by the gridded models.

Key details:
- **Constituent name mapping**: NOAA/TICON use non-standard abbreviations (LAM2, RHO, EP2, SGM, 3L2) that must be mapped to pyTMD names (lambda2, rho1, eps2, sigma1, l2')
- **Unrecognized constituents**: Filtered with a warning to stderr (only 3L2 was unmappable before we found it was l2')
- **Corrections**: Uses `corrections="OTIS"` for station data (standard Doodson/IHO conventions)
- **Minor constituents**: `predict.infer_minor()` adds ~20 minor constituents not in the station data

### ocean_model.py — Gridded tidal models

Wraps pyTMD to load global models (GOT5.6, EOT20, FES2022) and predict tides at arbitrary coordinates. The prediction pipeline:

1. Load model with `pyTMD.io.model().from_database(name)`
2. Crop to a 4-degree bounding box around the target (saves memory)
3. Interpolate constituents to the exact coordinate with `ds.tmd.interp()`
4. Predict with `pyTMD.predict.time_series()` + `infer_minor()`
5. Find high/low extrema with `scipy.signal.find_peaks()`

**FES2022 special handling**: 34 constituent files (~5 GB on disk, ~16 GB uncompressed). Uses dask lazy loading (`chunks={}`) + xarray `.sel().compute()` to load only the regional subset. Reduces peak memory from 5 GB to 27 MB.

**Sampling**: 1-minute intervals (1440 points/day) for sub-minute peak resolution.

### datums.py — Tidal datum computation

Tidal datums (LAT, MLLW, MLW, MSL, MTL, MHW, MHHW, HAT) are statistical properties of the tidal signal over a 19-year nodal cycle. Two sources:

1. **Station-published datums**: Available in ticon station files (all 4,838 stations) and some NOAA stations (1,210 of 3,451). Read directly from the station JSON.

2. **Model-computed datums**: Run a 19-year hourly prediction (2003–2021, 166K time steps) at the coordinate, then extract:
   - LAT/HAT: min/max of entire series
   - MHHW/MLLW: mean of daily higher-highs / lower-lows
   - MHW/MLW: mean of all highs / all lows
   - MTL: (MHW + MLW) / 2

Computed datums are cached at `~/.cache/tides/datums/{model}.json`, keyed by coordinate rounded to model grid resolution. Computation takes ~2 seconds per point; cached lookups are instant.

### cache.py — Cache management

Manages two cache locations:
- **App cache** (`~/.cache/tides/`): station list, station database, datum computations
- **Model cache** (`~/Library/Caches/pytmd/` on macOS): pyTMD model files

`tides cache` shows both with sizes. `tides cache clear [name]` deletes selectively.

### timezone.py — Local time conversion

Wraps `timezonefinder` to map coordinates to IANA timezone names, used by `--local` flag. Singleton `TimezoneFinder` instance (lazy-initialized, ~20 MB memory).

## External Dependencies

| Package | Purpose |
|---|---|
| pyTMD | Tidal model loading, constituent interpolation, harmonic prediction |
| scipy | Peak detection (`find_peaks`) for identifying high/low tides |
| numpy | Array math throughout |
| xarray | Dataset handling for pyTMD model data |
| dask | Lazy loading for FES2022's 34 constituent files |
| h5py | HDF5 file reading (used by pyTMD for some model formats) |
| httpx | HTTP client for NOAA API and data downloads |
| typer | CLI framework |
| timezonefinder | Coordinate to IANA timezone mapping |

## Cache Layout

```
~/.cache/tides/                          App cache (XDG-compliant)
├── noaa_stations.json                   NOAA station list (~300 KB, 30-day TTL)
├── stations/                            Global station database (~34 MB)
│   ├── noaa/*.json                      3,451 NOAA stations
│   ├── ticon/*.json                     4,838 TICON stations
│   └── station_index.json               Searchable index
└── datums/                              Computed datum offsets
    ├── got5.6.json                      Cached per model, per grid point
    ├── fes2022.json
    └── eot20.json

~/Library/Caches/pytmd/                  Model cache (platformdirs)
├── GOT5.5/                              694 MB (dependency of GOT5.6)
├── GOT5.6/                              106 MB (default model)
├── EOT20/                               4.2 GB (optional, auto-downloaded)
├── fes2022b/ocean_tide_20241025/        5.0 GB (optional, manual download)
└── hamtide/                             570 MB (not yet supported)
```

## Testing

228 tests, 90% coverage. Unit tests mock all network and pyTMD calls. Integration tests (7, deselected by default) hit real NOAA API and load real model data.

```
uv run pytest                    # unit tests only (default)
uv run pytest -m integration     # integration tests (needs network + model data)
uv run pytest --cov=tides        # coverage report
```
