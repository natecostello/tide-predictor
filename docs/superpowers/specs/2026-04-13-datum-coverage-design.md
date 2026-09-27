# Design: --datum Flag + Test Coverage (#6, #3)

## 1. --datum Flag

### New module: `src/tides/datums.py`

Computes tidal datums (LAT, MLLW, MLW, MSL, MHW, MHHW, HAT) from either station data or a 19-year model prediction.

**Supported datums** (enum):
`lat`, `mllw`, `mlw`, `msl`, `mtl`, `mhw`, `mhhw`, `hat`

> **Amended 2026-09-27 (#12):** there is also a **NOAA path**. NOAA predictions are requested directly in the target datum (MLLW, MLW, MSL, MTL, MHW, MHHW); LAT/HAT are derived from MLLW predictions plus the station's published `datums.json` (`height_target = height_MLLW - (TARGET_stnd - MLLW_stnd)`). NOAA heights are never converted with model-derived datums.

**Two datum resolution paths:**

1. **Station path** — station files include a `datums` dict with offsets relative to STND. Convert: `height_datum = height_msl + (MSL - target_datum)` using the station's published values.

2. **Model path** — run 19-year hourly prediction (2003–2021) at the coordinate using the selected model. From the time series, compute:
   - LAT/HAT: min/max
   - MHW/MLW: mean of daily highs / daily lows
   - MHHW/MLLW: mean of daily higher-highs / daily lower-lows
   - MSL: mean of entire series (should be ~0 for models)

**Caching**: store computed model datums at `~/.cache/tides/datums/{model}.json`, keyed by coordinate rounded to model grid resolution (1/16° for FES2022, 0.5° for GOT5.6). Entries never expire.

> **Amended 2026-09-27 (#13):** the model path now predicts at 6-minute intervals over 2003-01-01 to 2022-01-01 (end exclusive, yearly chunks), takes MHHW/MLLW per tidal day (24.8412 h) rather than per calendar day, and is vectorized (~10 ms extraction). MSL stays 0 by definition (the harmonic series has no Z0). The cache file is versioned (`{model}.v2.json`).

**Performance**: ~2s per new coordinate (model load + 166K hourly predictions). Cached lookups are instant.

### CLI changes

```
tides get <lat,lon> [--datum mllw|mlw|msl|mtl|mhw|mhhw|lat|hat]
```

Default: `mllw` (matches tides4fishing/Nautide). `--datum msl` recovers old behavior.

The datum offset is applied after prediction, before formatting:
```
display_height = predicted_height_msl - datum_offset_msl
```

Where `datum_offset_msl` is the target datum's elevation relative to MSL (negative for datums below MSL like MLLW/LAT).

### Model changes

`TideResult` gets a new field `datum: str` to record which datum the heights are referenced to. Output formatters include the datum in `--json` output.

## 2. Test Coverage Targets

Current: 81% (194 tests). Target: 90%+ (estimated ~230 tests).

| Module | Current | Target | Key gaps |
|--------|---------|--------|----------|
| stations.py | 38% | 85%+ | download_station_database, build_station_index, get_station_index |
| resolver.py | 75% | 90%+ | _resolve_station path |
| cache.py | 78% | 90%+ | _fetch_eot20, ensure_model_data branches |
| noaa.py | 89% | 95%+ | fetch_station_list_xml, fetch_predictions HTTP |
| datums.py | new | 95%+ | built with full coverage |

## 3. Files Changed

- **New**: `src/tides/datums.py` — datum computation, caching, lookup
- **New**: `tests/test_datums.py` — full coverage for new module
- **Modified**: `src/tides/models.py` — add `datum` field to TideResult, Datum enum
- **Modified**: `src/tides/cli.py` — add `--datum` flag, apply conversion
- **Modified**: `src/tides/resolver.py` — pass datum info through
- **Modified**: tests for all changed modules + coverage gap fills
