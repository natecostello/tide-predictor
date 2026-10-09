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
tides peaks <lat,lon> [--date DATE] [--between HH:MM:HH:MM] [shared options]
tides level <lat,lon> [--when now|HH:MM|YYYY-MM-DDTHH:MM]... [shared options]
tides when  <lat,lon> --level H|now [--rising|--falling] [--date DATE] [--between HH:MM:HH:MM] [shared options]
  shared options: [--local] [--feet] [--json] [--precision N] [--source auto|noaa|station|model] [--model got5.6|got5.5|eot20|fes2022] [--datum mllw|mlw|msl|mtl|mhw|mhhw|lat|hat] [--verbose]
tides cache [--json]
tides cache clear [stations|datums|got5.5|got5.6|eot20|fes2022|hamtide11] [--all|-a] [--yes]
tides fetch-model
tides --version
```

Coordinate accepts `lat,lon` (e.g. `40.7128,-74.0060`) or a single quoted token `"lat lon"`.
`--date` accepts YYYY-MM-DD, `today`, `tomorrow` (display clock) and ranges, limited to 366 days (inclusive); `--when` and `--level` input is read on the display clock and in display units. `-h`/`--help` work everywhere; bare
`tides` shows help (bare `tides cache` still shows cache info).
Negative latitudes work directly for every command (e.g. `tides peaks -2.88,-39.91`): the
`main_entry` console script prepends a space to any bare `-lat,lon` token
in `sys.argv` before invoking Typer, so Click does not parse it as an
option flag. `parse_coordinate` already strips whitespace.

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
- Unexpected exceptions print `Error: unexpected <Type> ...: <message>`; `TIDES_DEBUG=1` adds the traceback
- In `auto` mode, a failing source (NOAA API, NOAA station list, GitHub station DB) falls through to the next one with a stderr note; explicit `--source` reports the error

### Arguments and Flags
- Provide both short and long flag forms (e.g. `-d`/`--date`)
- Defaults should be the right choice for most users (today's date, UTC, meters, auto source)
- `--local` switches the whole date contract to the coordinate's clock: default "today", `--date` bounds and day grouping all use local dates (nautical `Etc/GMT` zones at sea; UTC only if no zone is known)
- Use standard flag names where conventions exist
- Make flags, args, and subcommands order-independent where possible

### Robustness
- Validate user input early, exit before bad things happen
- Show progress for long-running operations (model download)
- Make things time out (network operations)
- Be liberal in what you accept (coordinate parsing)

### Future-Proofing
- Keep changes additive
- Exception: `get` -> `peaks` (no alias) was a deliberate, owner-approved breaking change, shipped in 0.2.0
- Encourage `--json` for scripting stability
- Don't have catch-all subcommands

### Configuration
- Follow XDG Base Directory Specification for cache (`$XDG_CACHE_HOME/tides/` or `~/.cache/tides/`)
- No user-facing configuration files — the tool is stateless

## Conventions

- Use `ruff` for linting and formatting
- Use `uv` for dependency management
- Use `pytest` for testing
- Unit tests are isolated by autouse fixtures in `tests/conftest.py`: HOME/XDG_CACHE_HOME point at `tmp_path`, `get_model_datums` is stubbed, and outbound sockets raise `NetworkBlockedError`. Integration tests (`-m integration`) opt out and run the working tree via `python -m tides`
- The package version lives only in `pyproject.toml`; runtime reads it via `importlib.metadata`
- Source, config and CI files are ASCII-only (enforced in CI); Markdown is exempt
- `uv.lock` is committed; CI installs with `uv sync --locked`
- Type hints on all public functions
- Request a GitHub Copilot review upon submitting a PR
- `.github/copilot-instructions.md` is the Copilot reviewer's repo context; refresh it after changes to commands, architecture or conventions with the maintainer's user-level Claude Code skill `/copilot-update` (it lives in `~/.claude/skills/`, not in this repo). Without that skill, audit every section against the code and bump both rev markers; do not patch single lines inline

## Project Structure

```
src/tides/
├── __init__.py
├── cli.py          # Typer app (peaks/level/when/cache), argument parsing, output formatting
├── curve.py        # HeightCurve: heights/rates on a 1-min grid; at, crossings, turns
├── models.py       # Data classes for tides, coordinates, etc.
├── noaa.py         # NOAA CO-OPS API client
├── ocean_model.py  # pyTMD wrapper, extrema finding
├── resolver.py     # Source selection (auto/noaa/station/model); resolve_tides, resolve_curve
├── cache.py        # XDG cache management, data downloads, cache info/clear
├── datums.py       # Tidal datum computation (LAT/MLLW/MHW/HAT) and caching
├── harmonics.py    # Station harmonic prediction via pyTMD
├── stations.py     # Global station database (openwatersio/tide-database)
└── timezone.py     # Coordinate to timezone mapping
```
