<!-- rev: 3 -->
[copilot-instructions rev 3]

# Copilot Code Review Instructions

## Role

You are a code reviewer only. Do not suggest new features, refactors outside the PR scope, or architectural changes unless they fix a bug or security issue in the diff.

## Repository Summary

`tides` is a stateless Python CLI for tide predictions at coastal coordinates. Three query commands, each named for what it returns: `peaks` (highs/lows), `level` (height and rising/falling rate at `--when` times) and `when` (times the water reaches `--level H|now`). Data comes from a three-tier source strategy: NOAA CO-OPS predictions (US, within 25 km), harmonic predictions from the openwatersio global station database (within 200 km), then gridded global tide models via pyTMD (GOT5.6 default; GOT5.5, EOT20, FES2022). Built with Typer, packaged with uv. `get` was renamed to `peaks` in 0.2.0 with no alias (deliberate breaking change).

## Build, Test, and Lint

- **Package manager:** `uv` (`uv.lock` is committed; CI installs with `uv sync --locked --extra dev`)
- **Linter/formatter:** `ruff` (line length 99)
- **Test framework:** `pytest`; integration tests are marked `integration` and deselected by default (`addopts = "-m 'not integration'"`)
- **Install:** `uv sync --extra dev`
- **Run tests:** `uv run pytest` (unit); `uv run pytest -m integration` (needs network + GOT5.6 cache via `uv run tides fetch-model`)
- **Lint:** `uv run ruff check src/ tests/`
- **Format:** `uv run ruff format --check src/ tests/`
- **CI** (`.github/workflows/ci.yml`, Python 3.11/3.12/3.13): ASCII-only check on `*.py`, `*.toml`, `*.yml` in `src`, `tests`, `pyproject.toml`, `.github`; ruff check; ruff format check; pytest
- **Debug:** `TIDES_DEBUG=1` adds tracebacks to unexpected-error output

## Architecture

- **`src/tides/cli.py`** -- Typer app (`peaks`, `level`, `when`, `cache`, `fetch-model`), argument parsing/validation, `_now()` clock, `format_plain`/`format_json`, `main_entry` (escapes negative-latitude tokens in `sys.argv`)
- **`src/tides/resolver.py`** -- source selection with AUTO fallthrough; `resolve_tides` (peaks) and `resolve_curve` (level/when, returns a `CurveSource` + `HeightCurve`); `_datum_shift`/`_apply_datum`
- **`src/tides/curve.py`** -- `HeightCurve` (1-minute heights in the target datum, central-difference rates): `at`, `crossings`, `turns`; `search_level`, `turns_of_kind`; `TURN_TOLERANCE_M = 0.03`
- **`src/tides/noaa.py`** -- NOAA CO-OPS client (httpx): station list, hilo and 6-minute predictions, station datums
- **`src/tides/stations.py`** -- global station database download/index, nearest usable station, `station_chart_offset`
- **`src/tides/harmonics.py`** -- station harmonic prediction via pyTMD (`STATION_CORRECTIONS = "GOT"`, no inferred minors)
- **`src/tides/ocean_model.py`** -- pyTMD model loading/interpolation, `find_extrema`, `EDGE_PAD`, UTC sampling helpers
- **`src/tides/datums.py`** -- datum offsets (published or computed from harmonics/model) and their cache
- **`src/tides/cache.py`** -- XDG cache, model/station downloads, `cache` info/clear, atomic writes
- **`src/tides/timezone.py`** -- coordinate to timezone (timezonefinder; nautical `Etc/GMT` zones at sea)
- **`src/tides/models.py`** -- data classes

High-risk areas: datum conversion (each source has a different native datum), display-clock/day grouping and DST handling under `--local`, AUTO source fallthrough, extrema detection at window edges, crossing/near-turn logic in `curve.py`.

## Key Abstractions

- `Coordinate` -- validated lat/lon
- `TideEvent` -- time (aware UTC), height (m), kind (`high`/`low`); `TidePoint(TideEvent)` adds `rate` (m/h) and `near` window, kind may be `rising`/`falling`
- `TideDay` / `TideResult` -- per-date events plus source metadata (source type, station, model, datum); the formatters consume `TideResult`
- `Source` -- enum `auto`/`noaa`/`station`/`model`
- `HeightCurve` / `CurveSource` -- curve over a padded window; a resolved source that builds further curves without re-running AUTO selection

## Dependencies and Non-Obvious Relationships

- `peaks` and `level` must report the same height at the same moment. Model and station paths both go through `resolver._datum_shift`, and station curves must apply `stations.station_chart_offset` first (as `predict_station_tides` does). NOAA heights are already in the target datum (`_datum_shift` returns 0.0 for NOAA; the `level` curve path skips it): hilo and 6-minute series are both fetched in the target datum, with LAT/HAT shifted by `_noaa_derived_shift`; do not ask for a second shift on NOAA heights
- NOAA heights are fetched in the target datum (LAT/HAT derived from MLLW + published station datums) and are never shifted by model datums
- NOAA subordinate (`type == "S"`) stations serve only MLLW and only hilo; `level`/`when` cannot use them (AUTO falls through, `--source noaa` errors)
- `peaks` on NOAA uses the official `interval=hilo` product; `level`/`when` use `interval=6` linearly interpolated
- A command resolves its source once; a second window must come from `CurveSource.curve`, never a second AUTO resolution
- `load_local_constituents` is `lru_cache`d per process; tests clear it in `tests/conftest.py`
- `--local` changes the whole date contract (default "today", `--date` bounds, day grouping, `--when`/`--between` input) to the coordinate's clock; "now"/"today" come only from `cli._now()`
- Negative latitudes work because `main_entry` prefixes a space to bare `-lat,lon` tokens; `CliRunner` tests must pass the escaped form (`" -2.88,-39.91"`)
- Unit tests are isolated by autouse fixtures in `tests/conftest.py` (HOME/XDG_CACHE_HOME to `tmp_path`, stubbed `get_model_datums`, sockets blocked); integration tests opt out

## Planning Documents

- `docs/architecture.md` -- current module walkthrough, data flow, datum conversion, caching, day grouping
- `docs/superpowers/specs/2026-04-11-tides-cli-design.md`, `2026-04-12-cache-harmonics-fes-design.md`, `2026-04-13-datum-coverage-design.md` and `docs/superpowers/plans/2026-04-11-tides-cli-implementation.md` -- historical design records; they still say `tides get` and are intentionally not updated

## Coding Conventions

- Type hints on all public functions
- `ruff` for linting and formatting (no black, no flake8); `uv` for dependencies (no pip/poetry)
- Source, config and CI files are ASCII-only (enforced in CI); Markdown is exempt
- Errors to stderr; exit 1 for user input errors, 2 for data/network errors; options validated by hand (not Typer choices) so bad values exit 1
- No raw tracebacks: expected errors are rewritten; unexpected ones print `Error: unexpected <Type> ...: <message>`
- Notes about fallthrough or empty results go to stderr and are suppressed with `--json` where the JSON structure already carries the information
- New CLI options for `level`/`when` have no short flags; shared options keep identical names/short flags across `peaks`/`level`/`when`
- Conventional commit prefixes (`feat:`, `fix:`, `feat!:` for breaking changes); PRs squash-merged with `(#N)`
- Follow clig.dev guidelines (documented in CLAUDE.md)

## Code Review Focus Areas

- **Datum correctness** -- every height path (peaks, level, when) must land in the requested datum the same way; flag NaN/zero datum offsets, silent fallback to MSL, or model datums applied to NOAA heights
- **Display clock and DST** -- times read or grouped on the display clock (`--local`) must handle nonexistent (spring-forward) and repeated (fall-back) wall times; never use the machine's local timezone; "now" only via `cli._now()`, read once per command
- **Window edges** -- extrema and crossings near day/range boundaries must not be lost or double-counted (padded windows, half-open `[start, end)` intervals)
- **AUTO fallthrough** -- a failing source (NOAA API, station list, GitHub station DB, subordinate station) must fall through with a stderr note in `auto`, and report an error with explicit `--source`
- **Output stability** -- `peaks` plain/JSON output is a contract; JSON changes must be additive; empty requested days stay in JSON as `"tides": []`
- **Rounding** -- values are composed at full precision and rounded only in formatting; `-0.0` must not be printed
- **Model data NaN handling** -- land/no-coverage coordinates must produce the inland error (exit 2), not NaN output
- **Error messages** -- human-readable with actionable guidance; correct service name in network errors
- **Cache safety** -- writes are atomic; `cache clear` must not delete manually downloaded models unless named or `--all`; cache paths respect `$XDG_CACHE_HOME`
- **Test isolation** -- unit tests must not touch the real HOME/XDG cache or network; anything needing them is `@pytest.mark.integration`; tests freeze the clock by monkeypatching `cli._now`, never asserting against the real clock
- **Documentation consistency** -- when user-facing behavior changes (commands, flags, output format, error messages, JSON fields), verify README.md, CLAUDE.md and docs/architecture.md are updated in the same PR
- **Version single-sourcing** -- the package version literal lives only in `pyproject.toml`; `tides --version` and `tides.__version__` must read it from installed metadata (`importlib.metadata.version("tides")`). Flag any other hardcoded package-version literal (CLI strings, `__version__ = "x.y.z"` assignments, docs generators, user-agent or wire strings meant to carry the package version), any test asserting a hardcoded version literal, and any vacuous `--version` test (e.g. `assert "version" in output.lower()`) instead of comparing against `importlib.metadata.version("tides")`. Flag a diff that changes runtime behavior without bumping the `pyproject.toml` version (the tool is git-installed, so merge is the release; keep `uv.lock` in sync), and soft-flag a version bump with no functional change. Do NOT flag the `"0.0.0-dev"` fallback in the `PackageNotFoundError` branch of `src/tides/__init__.py`, or a deliberately pinned wire token held in one named constant with a comment saying it is intentionally independent of the package version; never suggest binding such a token to the package version
- **No secrets in flags** -- if auth is ever added, it must not be passed via CLI flags (visible in `ps` output)

## What NOT to Flag

- Using `httpx` over `requests` (intentional choice)
- Sync code in the CLI and HTTP layers (Typer is sync; no async is used)
- Heavy imports (pyTMD, xarray, resolver modules) done inside functions in `cli.py` -- intentional to keep `--help` and argument errors fast
- Historical `tides get` references under `docs/superpowers/` -- intentionally left as written
- `peaks` keeping its own `resolve_tides` path instead of routing through `HeightCurve.turns` -- intentional so its output stays byte-identical
- `TURN_TOLERANCE_M` near-turn rows replacing two crossings in `when` output -- specified behavior, not lost data
