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
tides -8.05,-34.87 --source model

# Pre-download model data
tides fetch-model
```

**Note:** Always use the comma-separated coordinate form (`40.7128,-74.0060`). Space-separated coordinates with negative longitude values are not supported due to CLI argument parsing limitations.

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
uv run pytest                                                   # unit tests
uv run pytest tests/test_integration.py -v -m integration       # integration tests
uv run ruff check src/ tests/                                   # lint
uv run ruff format src/ tests/                                  # format
```

## License

MIT
