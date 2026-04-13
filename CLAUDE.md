# Tides CLI

A stateless CLI for tide predictions using NOAA station data and global tidal models (GOT5.6, EOT20, FES2022).

## Architecture

- **CLI framework:** Typer (built on Click)
- **Tidal models:** pyTMD with GOT5.6 (default), EOT20, FES2022
- **Station data:** NOAA CO-OPS API (US tide stations)
- **Timezone:** timezonefinder (offline lat/long to timezone)
- **Package management:** uv
- **Source layout:** `src/tides/`

## CLI Interface

```
tides get <lat,lon> [--date DATE] [--local] [--feet] [--json] [--between HH:MM:HH:MM] [--precision N] [--source auto|noaa|station|model] [--model got5.6|eot20|fes2022] [--datum mllw|mlw|msl|mhw|mhhw|lat|hat] [--verbose]
tides cache [--json]
tides cache clear [name] [--yes]
tides fetch-model
tides --version
```

Coordinate accepts comma-separated form: `lat,lon` (e.g. `40.7128,-74.0060`).

## CLI Design Guidelines (from clig.dev)

Follow these principles in all CLI work:

### Help and Discovery
- Display help on `-h`, `--help`, and bare `tides` with no args
- Lead with examples in help text
- Show most common flags first
- When input is invalid, suggest the corrected form if guessable
- Include a link to the GitHub repo in top-level help

### Output
- Human-readable output by default (detect TTY)
- `--json` for structured machine-readable output
- `--verbose` for additional detail, not shown by default
- Keep success output brief — don't over-explain
- Use color intentionally and sparingly
- Respect `NO_COLOR` env var, `--no-color` flag, and non-TTY detection to disable color

### Errors
- Catch expected errors and rewrite for humans — no raw tracebacks
- Provide actionable guidance in error messages
- Errors go to stderr
- Exit code 1 for user errors, 2 for data/network errors

### Arguments and Flags
- Provide both short and long flag forms (e.g. `-d`/`--date`)
- Defaults should be the right choice for most users (today's date, UTC, meters, auto source)
- Use standard flag names where conventions exist
- Make flags, args, and subcommands order-independent where possible

### Robustness
- Validate user input early, exit before bad things happen
- Show progress for long-running operations (model download)
- Make things time out (network operations)
- Be liberal in what you accept (coordinate parsing)

### Future-Proofing
- Keep changes additive
- Encourage `--json` for scripting stability
- Don't have catch-all subcommands

### Configuration
- Follow XDG Base Directory Specification for cache (`$XDG_CACHE_HOME/tides/` or `~/.cache/tides/`)
- No user-facing configuration files — the tool is stateless

## Conventions

- Use `ruff` for linting and formatting
- Use `uv` for dependency management
- Use `pytest` for testing
- Type hints on all public functions
- Request a GitHub Copilot review upon submitting a PR

## Project Structure

```
src/tides/
├── __init__.py
├── cli.py          # Typer app, argument parsing, output formatting
├── models.py       # Data classes for tides, coordinates, etc.
├── noaa.py         # NOAA CO-OPS API client
├── ocean_model.py  # pyTMD wrapper, extrema finding
├── resolver.py     # Source selection logic (auto/noaa/model)
├── cache.py        # XDG cache management, data downloads, cache info/clear
├── datums.py       # Tidal datum computation (LAT/MLLW/MHW/HAT) and caching
├── harmonics.py    # Station harmonic prediction via pyTMD
├── stations.py     # Global station database (openwatersio/tide-database)
└── timezone.py     # Coordinate to timezone mapping
```
