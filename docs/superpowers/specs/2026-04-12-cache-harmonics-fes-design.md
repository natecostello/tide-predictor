# Design: Cache Subcommand, pyTMD Harmonics, FES2022 Integration

Three features in one PR. All additive; no breaking changes.

## 1. `tides cache` Subcommand

### Problem
~10 GB of model data across two cache locations with no visibility or management.

### Design

New Typer sub-app at `tides cache` with two commands:

**`tides cache`** (default action, implemented as callback):
- Lists both cache locations with sizes:
  - App cache (`~/.cache/tides/`): station index, NOAA stations, station database
  - Model cache (`~/Library/Caches/pytmd/` via platformdirs): GOT5.5, GOT5.6, EOT20, FES2022, HAMTIDE11
- Shows per-model directory sizes
- `--json` flag for machine-readable output

**`tides cache clear [name]`**:
- No argument: clears everything (both caches), with confirmation prompt
- `name` argument: clears specific item (`stations`, `got5.5`, `got5.6`, `eot20`, `fes2022`, `hamtide11`)
- `--yes` flag to skip confirmation
- Reports bytes freed

### Files Changed
- `src/tides/cli.py` — add `cache_app` Typer sub-app
- `src/tides/cache.py` — add `get_cache_info()`, `clear_cache()`, `format_size()` functions

## 2. Replace Hand-Rolled Harmonics with pyTMD

### Problem
`harmonics.py` manually computes `pf * A * cos(theta - phase)` with hardcoded `corrections="GOT"`. This:
- Hardcodes the wrong correction type for station data
- Misses minor constituent inference (which adds ~2-5% accuracy)
- Computes one point at a time (slow)

### Design

Replace the manual computation by constructing an xarray Dataset from station harmonic constituents and feeding it through pyTMD's `predict.time_series()` + `predict.infer_minor()` — the same pipeline used by `ocean_model.py`.

**Dataset construction**: Each constituent becomes a complex64 variable:
```python
z = amplitude * exp(-1j * phase_radians)
```

**Corrections**: Use `corrections="OTIS"` for station predictions. This is the most standard formulation for conventional harmonic constants (Doodson/IHO conventions). The "GOT" corrections are Goddard-specific and may not match how NOAA/ticon stations were analyzed.

**Vectorization**: Compute all time steps at once (240 per day at 6-min intervals) instead of one-by-one.

**Minor constituents**: Add `predict.infer_minor()` call, matching what `ocean_model.py` does. This infers ~20 minor constituents from the major ones.

### Files Changed
- `src/tides/harmonics.py` — rewrite: construct xarray Dataset, call `predict.time_series()` + `predict.infer_minor()`
- `src/tides/stations.py` — update to pass corrections type

## 3. FES2022 Model Integration

### Problem
FES2022b data (34 constituents, 5 GB on disk) is downloaded but not wired into the CLI.

### Design

**User-facing name**: `--model fes2022` (maps to pyTMD's `FES2022` database key)

**Performance**: FES2022 is 16 GB uncompressed in memory. The current `crop=False` approach is unusable. Change `compute_tides()` to use `crop=True` with a bounding box around the target coordinate (2-degree padding). Falls back to `crop=False` near the antimeridian. This applies to ALL models (minor improvement for GOT/EOT20, critical for FES2022).

**No auto-download**: Unlike GOT (which has `pyTMD.datasets.fetch_gsfc_got()`), FES2022 has no automated download in pyTMD. `ensure_model_data()` will detect if FES2022 is missing and print instructions for manual download from AVISO.

**HAMTIDE11**: Deferred — data exists on disk but file naming convention doesn't match pyTMD's expected format (`*.hamtide11a.nc` vs `HAMcurrent11a_*.nc.gz`). Needs separate investigation.

### Files Changed
- `src/tides/ocean_model.py` — add FES2022 to `SUPPORTED_MODELS`, add cropping with bounds
- `src/tides/cache.py` — handle FES2022 in `ensure_model_data()`

## Testing Strategy

- Unit tests: mock pyTMD for harmonics rewrite, test cache info/clear with tmp_path
- Integration tests: compare station predictions against known NOAA predictions (existing Bermuda test)
- Manual verification: run against OBX sound side coordinate with different models
