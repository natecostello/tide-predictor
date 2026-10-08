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

Three commands, each named for what it returns:

- `tides peaks` -- the highs and lows (was `tides get` before 0.2.0)
- `tides level` -- the water height at a time (`--when`)
- `tides when` -- the times the water is at a height (`--level`)

`level --when T` and `when --level H` are inverses: "the level when T" and
"when the level is H".

```bash
# Today's highs and lows at a location
tides peaks 40.7128,-74.0060

# Specific date (also: today, tomorrow, or a range like today:2026-10-15)
tides peaks 40.7128,-74.0060 --date 2026-04-15

# Date range with local times and feet
tides peaks 35.9,-75.6 --date 2026-04-15:2026-04-17 --local --feet

# Negative latitude (southern hemisphere) — pass as-is, no quoting needed
tides peaks -2.88,-39.91 --feet

# JSON output
tides peaks 40.7128,-74.0060 --json

# Only daytime tides
tides peaks 40.7128,-74.0060 --between 06:00:18:00

# Night window (wraps midnight)
tides peaks 40.7128,-74.0060 --between 20:00:04:00

# Force NOAA station data
tides peaks 40.7128,-74.0060 --source noaa

# Force global model
tides peaks -8.05,-34.87 --source model

# Use FES2022 model (34 constituents, must be pre-downloaded)
tides peaks 35.9,-75.6 --source model --model fes2022

# Pre-download everything needed offline: NOAA station list (always
# refreshed), global station database and GOT5.6 (skipped if already present)
tides fetch-model

# View cache sizes
tides cache

# Clear specific model cache
tides cache clear eot20 --yes

# Clear app cache + auto-downloaded GOT models (lists items and asks first).
# EOT20 / FES2022 / HAMTIDE11 are kept unless named or --all is given
tides cache clear
```

### Coordinate format

Coordinates are `lat,lon` (e.g. `40.7128,-74.0060`). Negative latitudes
are accepted directly — `tides peaks -2.88,-39.91` works without quoting or
a `--` separator. Internally the CLI prefixes a leading space onto bare
negative-coordinate tokens so Click does not parse them as option flags.

### `tides level`

Height, and whether the tide is rising or falling (with its rate), at one or
more times. `--when` is repeatable and accepts `now` (the default), `HH:MM`
(today) or `YYYY-MM-DDTHH:MM`, read on the display clock (local with
`--local`, UTC otherwise). A time within 0.03 m (about 0.1 ft) of an adjacent
high or low is labeled `high`/`low` instead.

```
$ tides level -2.8810722,-39.9083908 --local --feet --source model --datum lat
6.7ft@12:31 rising +2.2ft/h

$ tides level -2.8810722,-39.9083908 --when 2026-10-08T16:00 --when 2026-10-09T16:00 --local --feet --source model --datum lat
2026-10-08: 10.1ft@16:00 falling -0.8ft/h
2026-10-09: 10.7ft@16:00 high
```

### `tides when`

Times the water crosses a height. `--level` is a number in display units
(feet with `--feet`) above `--datum`, or `now` (the current height). The
search window is `--date` (default today), filtered by `--between`.

- A number reports crossings in both directions; `--rising` / `--falling`
  keep one.
- `--level now` reports crossings in the direction the tide is moving now,
  so `--date tomorrow` answers "when tomorrow is it at this level, on the same
  rising tide?". If now is at a high (or low), it reports that day's highs
  (or lows) instead.
- A level within 0.03 m of a high or low is reported once as that turn, with
  the window in which the water stays within 0.03 m of the level:
  `10.4ft@15:15 high (near: 14:53-15:37)`. These rows ignore `--rising` /
  `--falling`.
- A level never reached on a day prints nothing for that day and a stderr
  note, e.g. `Note: level 11.0ft not reached on 2026-10-09 (max 10.8ft@15:53)`.

```
$ tides when -2.8810722,-39.9083908 --level now --date tomorrow --local --feet --source model --datum lat
6.7ft@13:03 rising +2.4ft/h

$ tides when -2.8810722,-39.9083908 --level 6.7 --date tomorrow --local --feet --source model --datum lat
6.7ft@00:40 rising +2.3ft/h, 6.7ft@06:20 falling -2.4ft/h, 6.7ft@13:03 rising +2.4ft/h, 6.7ft@18:41 falling -2.4ft/h
```

On NOAA stations, `level` and `when` use NOAA's 6-minute predictions
(linearly interpolated); `peaks` keeps NOAA's official high/low product.
NOAA subordinate stations publish only highs and lows, so `auto` falls
through to the station database or model, and `--source noaa` errors.

## Shared options

All three commands take these options. `level` takes `--when` instead of
`--date`/`--between`.

| Flag | Short | Description |
|------|-------|-------------|
| `--date` | `-d` | Date or range: YYYY-MM-DD, `today`, `tomorrow`, or a range like YYYY-MM-DD:YYYY-MM-DD (366 days max) |
| `--local` | `-l` | Times in local timezone at coordinates. Days, `--date` bounds and the default "today" all follow the local clock (open-ocean points use their nautical `Etc/GMT` zone; UTC only if no zone is known at all) |
| `--feet` | `-f` | Heights in feet (default: meters) |
| `--json` | `-j` | JSON output |
| `--between` | `-b` | Time window filter (HH:MM:HH:MM); a start after the end wraps midnight, e.g. `20:00:04:00` |
| `--precision` | `-p` | Decimal places for height and rate (default: 1) |
| `--source` | `-s` | Data source: auto, noaa, station, model (default: auto) |
| `--model` | `-m` | Tide model: got5.6, got5.5, eot20, fes2022 (default: got5.6). Used only when the model answers; if a NOAA or station source answers, a note on stderr says the flag was ignored (use `--source model` to force it) |
| `--datum` | | Height datum: mllw, mlw, msl, mtl, mhw, mhhw, lat, hat (default: mllw) |
| `--verbose` | `-v` | Show source details |
| `--version` | | Show version |

## JSON output

`--json` prints the coordinate, source, model, datum, timezone and unit, plus one entry per **requested** day (a day stays in the list with `"tides": []` when `--between` filtered out all of its events). Each event has:

| Key | Example | Meaning |
|---|---|---|
| `time` | `"05:17"` | Clock time (local with `--local`, else UTC) |
| `height` | `2.3` | Height in the chosen unit and datum |
| `datetime` | `"2026-09-28T05:17-03:00"` | Full ISO 8601 timestamp with UTC offset, on the same clock as `time` |
| `type` | `"high"` | `"high"` or `"low"`; also `"rising"` or `"falling"` for `level`/`when` |
| `rate` | `-0.8` | Rate of change in the chosen unit per hour; `0.0` for `high`/`low` (always `0.0` on `peaks`) |
| `near` | `{"from": "...", "to": "..."}` | `when` only, on near-turn rows: the window (ISO timestamps) in which the water stays within 0.03 m of the level |

`level` lists the distinct dates of its `--when` times; `when` lists every requested day, with `"tides": []` when the level is not reached (the stderr notes are not printed with `--json`).

`--verbose` (plain output) prefixes each line with the source and datum, e.g. `[Station: Fortaleza USCGS, 150.1km, MLLW]`. When no events match the range or filter, plain output prints nothing on stdout and a note on stderr (exit 0).

## Data Sources

**NOAA CO-OPS** (US waters): Uses official tide station predictions. Auto-selected when a station is within 25km of the coordinates. Heights are NOAA's own published values in the requested datum. NOAA subordinate stations publish MLLW only; for other datums, `auto` mode falls through to the station database or model (with a note on stderr), and `--source noaa` reports an error.

**GOT5.6** (global, default): NASA Goddard Ocean Tide model at 1/8° resolution. Used as fallback for locations outside NOAA coverage. Model data is auto-downloaded on first use.

**EOT20** (global, `--model eot20`): Empirical Ocean Tide model at 1/8° resolution. Auto-downloaded on first use (~2.3GB).

**FES2022** (global, `--model fes2022`): FES2022b ocean tide model with 34 tidal constituents. Highest fidelity available. Must be manually downloaded from [AVISO](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html) (~5GB on disk).

**Global station database** (~8,289 stations): Harmonic predictions from the openwatersio/tide-database. Auto-selected when within 200km in `auto` source mode, or via `--source station`.

## Offline use and errors

After `tides fetch-model`, queries work offline: if NOAA or GitHub is unreachable, `auto` mode falls through to the cached station database and then the model, with a one-line note on stderr. A NOAA station list older than 30 days is reused (with a warning) when it cannot be refreshed.

Unexpected errors print the exception type and message. Set `TIDES_DEBUG=1` to also print the full traceback:

```bash
TIDES_DEBUG=1 tides peaks 40.7128,-74.0060
```

## Development

```bash
uv run pytest                                                   # unit tests
uv run pytest tests/test_integration.py -v -m integration       # integration tests
uv run ruff check src/ tests/                                   # lint
uv run ruff format src/ tests/                                  # format
```

## License

MIT
