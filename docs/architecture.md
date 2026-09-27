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

The `tides` console script is wired to `main_entry`, a thin wrapper around the Typer app that rewrites `sys.argv` before dispatch: bare `lat,lon` tokens with a leading `-` (e.g. `-2.88,-39.91`) get a leading space prepended so Click does not parse them as option flags. `parse_coordinate` already strips whitespace, so this is transparent downstream. Bare negative numbers without a comma are left alone so legitimate negative-number option values still parse correctly.

### resolver.py — Source selection and datum conversion

The core routing logic. `resolve_tides()` tries sources in order for `auto` mode:
1. **NOAA API** — if a station is within 25 km
2. **Global station database** — if a station is within 200 km
3. **Gridded model** — always available (fallback)

After getting a result, `_apply_datum()` converts heights from the source's native datum to the requested datum. This is the trickiest part of the codebase because each source uses a different native datum:
- **NOAA**: requested directly in the target datum (MLLW, MLW, MSL, MTL, MHW, MHHW). LAT/HAT are derived from MLLW predictions plus the station's published `datums.json`. NOAA heights are never shifted by model-derived datums. Subordinate ("S") stations publish MLLW only; in auto mode a station that cannot serve the datum, or a NOAA API/network failure (station list, predictions or station datums), falls through to the next source with a stderr note; likewise an unreachable GitHub station database falls through to the model
- **Station**: heights relative to chart datum (LAT or MLLW, varies per station)
- **Model**: heights relative to MSL (mean sea level)

The conversion formula: `height_target = height_current - (target_offset - current_offset)`, where offsets come from station datum tables or a 19-year model computation.

### noaa.py — NOAA CO-OPS API client

Three endpoints:
- Station list (XML, cached 30 days): `api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.xml` (includes station `type`: R = reference, S = subordinate)
- Predictions (JSON, not cached): `api.tidesandcurrents.noaa.gov/api/prod/datagetter`, requested in the user's datum
- Station datums (JSON, not cached, LAT/HAT only): `api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/<id>/datums.json`

NOAA predictions are the gold standard for US waters — they come from the agency's own harmonic analysis of decades of tide gauge data.

### stations.py — Global station database

Downloads the [openwatersio/tide-database](https://github.com/openwatersio/tide-database) (~8,289 stations from NOAA and TICON sources) as a GitHub zip archive. Each station is a JSON file with harmonic constituents, datum offsets, and metadata.

`find_nearest_usable_station()` returns the nearest station *with harmonic constituents* within range (about 27% of the index -- mostly NOAA subordinate stations -- has none and is skipped); `resolver.py`'s `_resolve_station` calls it. `predict_station_tides()` calls into `harmonics.py` to generate predictions from the station's constituents, applying the chart datum offset so heights are relative to the station's published datum (LAT or MLLW). When the chart datum is not published relative to MSL (e.g. STND with no datums), heights stay MSL-relative (`datums.station_heights_datum`).

### harmonics.py — Harmonic prediction engine

Converts station harmonic constituents (amplitude + phase per constituent) into an xarray Dataset compatible with pyTMD's `predict.time_series()`. This is the same prediction pipeline used by the gridded models.

Key details:
- **Constituent name mapping**: NOAA/TICON use non-standard abbreviations (LAM2, RHO, EP2, SGM) that are mapped to pyTMD names (lambda2, rho1, eps2, sigma1)
- **Unrecognized constituents**: Filtered with a warning to stderr. 3L2 is deliberately not mapped to pyTMD's l2' (not confirmed to be the same constituent); its amplitude is a few mm
- **Corrections**: Uses `corrections="GOT"` for station data: astronomical arguments and frequencies come from Doodson numbers for every recognized constituent. (The `OTIS` branch uses a fixed ~33-constituent table and turns any other constituent into a constant; at Golden Gate it was ~3.7 cm RMS vs NOAA, GOT is ~1.8 cm)
- **Minor constituents**: not inferred for station data (the station's own harmonic analysis is complete)

### ocean_model.py — Gridded tidal models

Wraps pyTMD to load global models (GOT5.6, EOT20, FES2022) and predict tides at arbitrary coordinates. The prediction pipeline:

1. Load model with `pyTMD.io.model().from_database(name)`
2. For FES-format grids (FES2022, EOT20), crop to a 4-degree bounding box around the target (saves memory), including windows that cross the 0/360 seam near Greenwich (the two edge slices are concatenated in a signed longitude). All models use 0-360 longitude grids. GOT-format grids are loaded whole (pyTMD 3.0.6's GOT reader ignores crop/bounds; ~100 MB) and always interpolated at the non-negative longitude
3. Interpolate constituents to the exact coordinate with `ds.tmd.interp()`
4. Predict with `pyTMD.predict.time_series()` + `infer_minor()`
5. Find high/low extrema with `scipy.signal.find_peaks()`

Steps 1-3 live in `load_local_constituents()`, shared with `datums.py` and cached per process (so a query that also computes datums loads the model once).

**FES2022 special handling**: 34 constituent files (~5 GB on disk, ~16 GB uncompressed). Uses dask lazy loading (`chunks={}`) + xarray `.sel().compute()` to load only the regional subset. Reduces peak memory from 5 GB to 27 MB.

**Sampling**: 1-minute intervals (1440 points/day) for sub-minute peak resolution.

### datums.py — Tidal datum computation

Tidal datums (LAT, MLLW, MLW, MSL, MTL, MHW, MHHW, HAT) are statistical properties of the tidal signal over a 19-year nodal cycle. Three sources:

1. **Station-published datums**: Available in ticon station files (all 4,838 stations) and some NOAA stations (1,210 of 3,451). Read directly from the station JSON.

2. **Model-computed datums**: Run a 19-year prediction at 6-minute intervals (2003-01-01 to 2022-01-01 exclusive, 6,940 days, ~1.67M samples, predicted one year per chunk) at the coordinate, then extract (vectorized, ~10 ms):
   - LAT/HAT: min/max of entire series
   - MHHW/MLLW: mean of higher-highs / lower-lows per tidal day (24.8412 h), not per calendar day
   - MHW/MLW: mean of all highs / all lows
   - MTL: (MHW + MLW) / 2

Computed datums are cached at `~/.cache/tides/datums/{model}.v3.json` (the version is bumped whenever computed values or keys change; older files are left unused), keyed by the query point rounded to 0.01 deg (~1 km) so distinct points never share an entry. Only all-finite datum sets are cached: a point with no model data (e.g. >10 km inland of the model's wet cells) raises `DatumUnavailableError` (CLI exit 2) instead of producing NaN or 0.0 datums. Computation takes ~5 seconds per point on GOT5.6; cached lookups are instant.

3. **Station datums the station does not publish**: the station path never uses model datums. Published datums always win; any datum the station lacks (including its chart datum, e.g. STND with no published datums) is computed from the station's own harmonics with the same 19-year, 6-minute method and cached per station id in `datums/stations.v2.json` (v2: GOT corrections, #15). A station with neither the datum nor harmonics is an error (exit 2), never a silent MSL substitution. When a station's chart datum is not published relative to MSL, its predictions stay MSL-relative (`datums.station_heights_datum`).

### cache.py — Cache management

Manages two cache locations:
- **App cache** (`~/.cache/tides/`): station list, station database, datum computations
- **Model cache** (`~/Library/Caches/pytmd/` on macOS): pyTMD model files

`tides cache` shows both with sizes. `tides cache clear [name]` deletes selectively.

### Day grouping and `--local`

Events are grouped into days on the displayed clock. Without `--local` that is UTC; with `--local` it is the coordinate's timezone (`timezone.get_zoneinfo`: open-ocean points get their nautical `Etc/GMT` zone; UTC only if no zone is returned), and the underlying NOAA fetch / station / model prediction is widened by one UTC day on each side, then trimmed back to the requested local dates. Station and model predictions run as one continuous series padded by 3 h (`ocean_model.EDGE_PAD`), so extrema at day or range boundaries (e.g. exactly 00:00 UTC) are not lost.

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
└── datums/                              Computed datum offsets (versioned; older files unused)
    ├── got5.6.v3.json                   Cached per model, per point (0.01 deg key)
    ├── fes2022.v3.json
    ├── eot20.v3.json
    └── stations.v2.json                 Station datums computed from harmonics, per station id

~/Library/Caches/pytmd/                  Model cache (platformdirs)
├── GOT5.5/                              694 MB (dependency of GOT5.6)
├── GOT5.6/                              106 MB (default model)
├── EOT20/                               4.2 GB (optional, auto-downloaded)
├── fes2022b/ocean_tide_20241025/        5.0 GB (optional, manual download)
└── hamtide/                             570 MB (not yet supported)
```

## Testing

Unit tests are isolated by autouse fixtures in `tests/conftest.py`: `HOME`/`XDG_CACHE_HOME` point at a per-test `tmp_path`, `get_model_datums` is stubbed, and outbound (non-AF_UNIX) socket connections raise `NetworkBlockedError`, so an unmocked network call fails loudly instead of reaching the network. Integration tests (deselected by default) opt out of these fixtures, run the working tree via `python -m tides`, hit the real NOAA API and load real model data.

```
uv run pytest                    # unit tests only (default)
uv run pytest -m integration     # integration tests (needs network + model data)
uv run pytest --cov=tides        # coverage report
```
