<!-- rev: 1 -->
[copilot-instructions rev 1]

# Copilot Code Review Instructions

## Role

You are a code reviewer only. Do not suggest new features, refactors outside the PR scope, or architectural changes unless they fix a bug or security issue in the diff.

## Repository Summary

`tides` is a stateless Python CLI that returns high/low tide predictions for coastal coordinates. It uses a hybrid data strategy: NOAA CO-OPS station predictions for US locations, with fallback to the GOT5.6 global tidal model (via pyTMD) for worldwide coverage. Built with Typer, packaged with uv.

## Build, Test, and Lint

- **Package manager:** `uv`
- **Linter/formatter:** `ruff`
- **Test framework:** `pytest`
- **Install:** `uv pip install -e ".[dev]"`
- **Run tests:** `uv run pytest`
- **Lint:** `uv run ruff check src/ tests/`
- **Format:** `uv run ruff format src/ tests/`

## Architecture

- **`src/tides/cli.py`** — Typer app, argument parsing, output formatting
- **`src/tides/models.py`** — Data classes (Coordinate, TideEvent, etc.)
- **`src/tides/noaa.py`** — NOAA CO-OPS API client (HTTP via httpx)
- **`src/tides/ocean_model.py`** — pyTMD wrapper, extrema finding via scipy
- **`src/tides/resolver.py`** — Source selection logic (auto/noaa/model)
- **`src/tides/cache.py`** — XDG-compliant cache management, model downloads
- **`src/tides/timezone.py`** — Coordinate to timezone mapping via timezonefinder

High-risk areas: coordinate parsing (multiple input formats), source auto-selection (distance threshold), extrema finding (peak detection accuracy).

## Key Abstractions

- `Coordinate` — lat/lon pair with validation
- `TideEvent` — single high/low tide (time, height, unit)
- `TideDay` — collection of TideEvents for a date
- `Source` — enum for data source selection (auto/noaa/model)

## Dependencies and Non-Obvious Relationships

- pyTMD requires GOT5.6 model files on disk (auto-downloaded to XDG cache on first use)
- NOAA station list is cached locally for nearest-station lookup
- `--local` flag depends on timezonefinder for lat/lon to IANA timezone mapping
- Negative longitudes in space-separated coordinate form may conflict with Typer flag parsing; `--` disambiguates

## Planning Documents

- `docs/superpowers/specs/2026-04-11-tides-cli-design.md` — full design spec with output format, JSON schema, error handling, and data source strategy

## Coding Conventions

- Type hints on all public functions
- `ruff` for linting and formatting (no black, no flake8)
- `uv` for dependency management (no pip, no poetry)
- Errors to stderr, exit code 1 for user errors, 2 for data/network errors
- No raw tracebacks — catch expected errors and rewrite for humans
- Follow clig.dev guidelines (documented in CLAUDE.md)

## Code Review Focus Areas

- **Coordinate parsing robustness** — multiple input formats must all be handled; watch for edge cases with negative values and whitespace
- **Error messages** — must be human-readable with actionable guidance, never raw exceptions
- **NOAA API error handling** — network timeouts, malformed responses, station-not-found cases
- **Model data NaN handling** — land/no-coverage coordinates must produce clear errors, not silent garbage
- **Output format consistency** — plain text format must match spec exactly (`height@time` with unit suffix)
- **XDG compliance** — cache paths must respect `$XDG_CACHE_HOME`, not hardcode `~/.cache`
- **Documentation consistency** — when user-facing behavior changes (CLI flags, output format, error messages), verify that README.md, CLAUDE.md, and the design spec are updated accordingly
- **No secrets in flags** — if any auth is ever added, it must not be passed via CLI flags (visible in `ps` output)

## What NOT to Flag

- Using `httpx` over `requests` (intentional choice for async compatibility)
- Single-file modules with <200 lines (appropriate granularity for this project)
- Lack of async in the CLI layer (Typer is sync; async lives in the HTTP/model layers if needed)
