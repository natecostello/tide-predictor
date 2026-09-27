"""Datum safety: no NaN/zero datums from land points, exact-point cache keys,
and no silent MSL substitution for datums a station does not publish (#14)."""

import datetime
import json
import math
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from tides.cli import app
from tides.datums import (
    DatumUnavailableError,
    _extract_datums,
    _write_cache_entry,
    compute_datums_from_model,
    datums_from_station,
    get_model_datums,
    get_station_datums,
    station_heights_datum,
)
from tides.models import Coordinate, Source, TideDay, TideEvent, TideResult
from tides.resolver import _apply_datum
from tides.stations import predict_station_tides

runner = CliRunner()

MIXED_CONSTITUENTS = [
    {"name": "M2", "amplitude": 0.9, "phase": 30.0},
    {"name": "S2", "amplitude": 0.25, "phase": 60.0},
    {"name": "K1", "amplitude": 0.35, "phase": 120.0},
    {"name": "O1", "amplitude": 0.25, "phase": 100.0},
]

# La Libertad-style: publishes only MLLW / MSL / STND, chart datum MLLW.
PARTIAL_DATUM_STATION = {
    "name": "La Libertad (fixture)",
    "datums": {"STND": 0.0, "MLLW": 1.0, "MSL": 2.0},
    "chart_datum": "MLLW",
    "harmonic_constituents": MIXED_CONSTITUENTS,
}

# Eugene Island / Fort Wadsworth style: no datums at all, chart datum STND.
STND_STATION = {
    "name": "STND station (fixture)",
    "datums": {},
    "chart_datum": "STND",
    "harmonic_constituents": MIXED_CONSTITUENTS,
}


def _result(source: Source, height: float = 0.5, station_id: str | None = "st1") -> TideResult:
    return TideResult(
        coordinate=Coordinate(lat=-2.2, lon=-80.9),
        source_type=source,
        station_id=station_id,
        station_name="x",
        station_distance_km=1.0,
        model_name=None if source != Source.MODEL else "GOT5.6",
        days=[
            TideDay(
                date=datetime.date(2026, 9, 28),
                events=[
                    TideEvent(
                        time=datetime.datetime(2026, 9, 28, 3, tzinfo=datetime.timezone.utc),
                        height=height,
                    )
                ],
            )
        ],
    )


class TestNonFiniteGuard:
    def test_all_nan_model_series_raises(self):
        with (
            patch("tides.ocean_model.load_local_constituents", return_value=(None, None)),
            patch(
                "tides.ocean_model.predict_elevations",
                side_effect=lambda t, *_: np.full(len(t), np.nan),
            ),
            pytest.raises(DatumUnavailableError, match="inland"),
        ):
            compute_datums_from_model(38.2, -122.7, "GOT5.6")

    def test_no_extrema_raises_instead_of_zeros(self):
        with pytest.raises(DatumUnavailableError):
            _extract_datums(np.zeros(10_000), 6.0)

    def test_unavailable_datums_are_not_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tides.datums.compute_datums_from_model",
            lambda *a: (_ for _ in ()).throw(DatumUnavailableError("inland")),
        )
        from tides import datums as datums_mod

        monkeypatch.setattr(datums_mod, "get_model_datums", get_model_datums)
        with pytest.raises(DatumUnavailableError):
            get_model_datums(38.2, -122.7, "GOT5.6")
        assert not list((tmp_path / "cache" / "tides" / "datums").glob("got5.6.v3.json"))

    def test_nan_dict_never_written(self, tmp_path):
        f = tmp_path / "c.json"
        _write_cache_entry(f, "k", {"mllw": float("nan"), "msl": 0.0})
        assert not f.exists()

    def test_existing_nan_cache_entry_is_recomputed(self, tmp_path):
        cache_dir = tmp_path / "cache" / "tides" / "datums"
        cache_dir.mkdir(parents=True)
        (cache_dir / "got5.6.v3.json").write_text(
            '{"47.60,-122.34": {"lat": NaN, "mllw": 0.0, "msl": 0.0}}'
        )
        fresh = {"lat": -1.0, "mllw": -0.9, "msl": 0.0}
        with patch("tides.datums.compute_datums_from_model", return_value=fresh) as comp:
            got = get_model_datums(47.6026, -122.3393, "GOT5.6")
        comp.assert_called_once()
        assert got == fresh
        stored = json.loads((cache_dir / "got5.6.v3.json").read_text())
        assert stored["47.60,-122.34"] == fresh


class TestStationDatums:
    def test_published_without_msl_or_mtl_is_unusable(self):
        assert datums_from_station({"datums": {"STND": 0.0, "MLLW": 1.0}}) is None

    def test_heights_datum_falls_back_to_msl_for_stnd(self):
        assert station_heights_datum(STND_STATION) == "msl"
        assert station_heights_datum(PARTIAL_DATUM_STATION) == "mllw"

    def test_stnd_station_predictions_are_msl_relative(self):
        events = predict_station_tides(
            STND_STATION, datetime.date(2026, 9, 28), datetime.date(2026, 9, 28)
        )
        heights = [e.height for e in events]
        # MSL-relative: symmetric-ish around 0, not offset by any STND value.
        assert min(heights) < 0 < max(heights)

    @patch("tides.datums.get_model_datums", side_effect=AssertionError("station path"))
    def test_partial_station_distinct_datums(self, _md):
        heights = {}
        for d in ("mllw", "msl", "lat", "mhhw", "hat"):
            out = _apply_datum(
                _result(Source.STATION, height=0.5), d, "GOT5.6", station=PARTIAL_DATUM_STATION
            )
            heights[d] = out.days[0].events[0].height
            assert out.datum == d
        # Five distinct answers (previously msl/lat/mhhw/hat were identical).
        assert len({round(h, 4) for h in heights.values()}) == 5
        # Published values win: MLLW -> MSL shift is exactly the published 1.0 m.
        assert heights["msl"] == pytest.approx(0.5 - 1.0)
        assert heights["lat"] > heights["mllw"]  # LAT is below MLLW
        assert heights["hat"] < heights["mhhw"] < heights["msl"]

    def test_computed_station_datums_cached_by_id(self, tmp_path):
        needed = {"mllw", "lat"}
        with patch(
            "tides.datums.compute_station_datums",
            return_value={
                "lat": -1.5,
                "mllw": -0.8,
                "mlw": -0.6,
                "msl": 0.0,
                "mtl": 0.0,
                "mhw": 0.6,
                "mhhw": 0.8,
                "hat": 1.5,
            },
        ) as comp:
            first = get_station_datums(PARTIAL_DATUM_STATION, "st1", needed)
            second = get_station_datums(PARTIAL_DATUM_STATION, "st1", needed)
        comp.assert_called_once()
        assert first == second
        assert first["mllw"] == pytest.approx(-1.0)  # published wins over computed -0.8
        assert first["lat"] == -1.5
        assert (tmp_path / "cache" / "tides" / "datums" / "stations.v2.json").exists()

    def test_missing_datum_without_harmonics_errors(self):
        station = {**PARTIAL_DATUM_STATION, "harmonic_constituents": []}
        with pytest.raises(DatumUnavailableError, match="harmonic"):
            _apply_datum(_result(Source.STATION), "lat", "GOT5.6", station=station)


class TestCli:
    def test_datum_unavailable_exits_2(self):
        with patch("tides.resolver.resolve_tides", side_effect=DatumUnavailableError("inland")):
            r = runner.invoke(app, ["get", "38.2,-122.7", "--source", "model"])
        assert r.exit_code == 2
        assert "Error: inland" in r.output
        assert "nan" not in r.output.lower()

    @pytest.mark.parametrize("extra", [[], ["--json"]])
    def test_non_finite_height_never_rendered(self, extra):
        bad = _result(Source.MODEL, height=math.nan)
        with patch("tides.resolver.resolve_tides", return_value=bad):
            r = runner.invoke(app, ["get", "38.2,-122.7", *extra])
        assert r.exit_code == 2
        assert "nan" not in r.stdout.lower()
        assert "non-finite" in r.output
