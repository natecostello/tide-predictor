# tides

A command-line tool for tide predictions using NOAA station data and global tidal models (GOT5.6, EOT20, FES2022).

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
tides get 40.7128,-74.0060

# Specific date
tides get 40.7128,-74.0060 --date 2026-04-15

# Date range with local times and feet
tides get 35.9,-75.6 --date 2026-04-15:2026-04-17 --local --feet

# JSON output
tides get 40.7128,-74.0060 --json

# Only daytime tides
tides get 40.7128,-74.0060 --between 06:00:18:00

# Force NOAA station data
tides get 40.7128,-74.0060 --source noaa

# Force global model
tides get -8.05,-34.87 --source model

# Use FES2022 model (34 constituents, must be pre-downloaded)
tides get 35.9,-75.6 --source model --model fes2022

# Pre-download model data
tides fetch-model

# View cache sizes
tides cache

# Clear specific model cache
tides cache clear eot20 --yes
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
| `--source` | `-s` | Data source: auto, noaa, station, model (default: auto) |
| `--model` | `-m` | Tide model: got5.6, eot20, fes2022 (default: got5.6) |
| `--verbose` | `-v` | Show source details |
| `--version` | | Show version |

## Data Sources

**NOAA CO-OPS** (US waters): Uses official tide station predictions. Auto-selected when a station is within 25km of the coordinates.

**GOT5.6** (global, default): NASA Goddard Ocean Tide model at 0.5° resolution. Used as fallback for locations outside NOAA coverage. Model data is auto-downloaded on first use.

**EOT20** (global, `--model eot20`): Empirical Ocean Tide model at 0.125° resolution (4x finer than GOT5.6). Better accuracy for coastal locations. Auto-downloaded on first use (~2.3GB).

**FES2022** (global, `--model fes2022`): FES2022b ocean tide model with 34 tidal constituents. Highest fidelity available. Must be manually downloaded from [AVISO](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html) (~5GB on disk).

**Global station database** (~8,289 stations): Harmonic predictions from the openwatersio/tide-database. Auto-selected when within 200km in `auto` source mode, or via `--source station`.

## Development

```bash
uv run pytest                                                   # unit tests
uv run pytest tests/test_integration.py -v -m integration       # integration tests
uv run ruff check src/ tests/                                   # lint
uv run ruff format src/ tests/                                  # format
```

## License

MIT
