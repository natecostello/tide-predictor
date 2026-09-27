import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tides.models import Coordinate
from tides.ocean_model import (
    ELEVATION_INTERVAL_MINUTES,
    compute_tides,
    crosses_seam,
    find_extrema,
)


class TestFindExtrema:
    def _make_times(self, n):
        return [
            datetime.datetime(2026, 4, 15, tzinfo=datetime.timezone.utc)
            + datetime.timedelta(minutes=ELEVATION_INTERVAL_MINUTES * i)
            for i in range(n)
        ]

    def test_simple_sine_wave(self):
        """A sine wave over 24h should produce ~2 highs and ~2 lows."""
        n = 24 * 60 // ELEVATION_INTERVAL_MINUTES
        times = self._make_times(n)
        elevations = np.sin(np.linspace(0, 4 * np.pi, n))
        events = find_extrema(times, elevations)
        heights = [e.height for e in events]
        highs = [h for h in heights if h > 0.5]
        lows = [h for h in heights if h < -0.5]
        assert len(highs) == 2
        assert len(lows) == 2

    def test_events_are_chronological(self):
        n = 24 * 60 // ELEVATION_INTERVAL_MINUTES
        times = self._make_times(n)
        elevations = np.sin(np.linspace(0, 4 * np.pi, n))
        events = find_extrema(times, elevations)
        for i in range(len(events) - 1):
            assert events[i].time < events[i + 1].time

    def test_all_nan_returns_empty(self):
        n = 24 * 60 // ELEVATION_INTERVAL_MINUTES
        times = self._make_times(n)
        elevations = np.full(n, np.nan)
        events = find_extrema(times, elevations)
        assert events == []


class TestFindExtremaEdgeCases:
    def _make_times(self, n):
        return [
            datetime.datetime(2026, 4, 15, tzinfo=datetime.timezone.utc)
            + datetime.timedelta(minutes=ELEVATION_INTERVAL_MINUTES * i)
            for i in range(n)
        ]

    def test_partial_nan(self):
        """Array with some NaN values interspersed still finds peaks in valid regions."""
        n = 24 * 60 // ELEVATION_INTERVAL_MINUTES
        times = self._make_times(n)
        elevations = np.sin(np.linspace(0, 4 * np.pi, n))
        nan_start = n // 3
        elevations[nan_start : nan_start + n // 20] = np.nan
        events = find_extrema(times, elevations)
        assert len(events) > 0

    def test_flat_signal(self):
        """Constant array (all same value) returns empty list (no peaks)."""
        n = 24 * 60 // ELEVATION_INTERVAL_MINUTES
        times = self._make_times(n)
        elevations = np.ones(n) * 5.0
        events = find_extrema(times, elevations)
        assert events == []

    def test_single_peak(self):
        """Array with one clear peak returns a single event."""
        n = 24 * 60 // ELEVATION_INTERVAL_MINUTES
        times = self._make_times(n)
        elevations = np.zeros(n)
        hump_start = n // 3
        hump_len = n // 6
        elevations[hump_start : hump_start + hump_len] = np.sin(np.linspace(0, np.pi, hump_len))
        events = find_extrema(times, elevations)
        assert len(events) >= 1
        peak = max(events, key=lambda e: e.height)
        assert peak.height > 0.9

    def test_short_array(self):
        """Short array yields fewer events than a full signal."""
        n = 10
        times = self._make_times(n)
        elevations = np.sin(np.linspace(0, 2 * np.pi, n))
        events = find_extrema(times, elevations)
        assert len(events) <= 2


class TestComputeTides:
    def _setup_pytmd_mocks(self, mock_ensure, mock_model_cls, mock_predict, mock_infer, n=240):
        """Helper to wire up the pyTMD mock chain for compute_tides tests."""
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        mock_instance.corrections = "GOT"
        mock_instance.minor = ["2q1", "sigma1"]
        mock_ds = MagicMock()
        mock_instance.open_dataset.return_value = mock_ds
        mock_local = MagicMock()
        mock_ds.tmd.interp.return_value = mock_local

        mock_result = MagicMock()
        mock_result.values = np.sin(np.linspace(0, 4 * np.pi, n))
        mock_predict.return_value = mock_result

        mock_infer_result = MagicMock()
        mock_infer_result.values = np.zeros(n)
        mock_infer.return_value = mock_infer_result

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_calls_ensure_model(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """Verify ensure_model_data() is called."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        coord = Coordinate(lat=40.7, lon=-74.0)
        compute_tides(coord, datetime.date(2025, 12, 3), datetime.date(2025, 12, 3))
        mock_ensure.assert_called_once()

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_passes_model_corrections(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """time_series must be called with the model's corrections type, not the default."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        coord = Coordinate(lat=40.7, lon=-74.0)
        compute_tides(coord, datetime.date(2025, 12, 3), datetime.date(2025, 12, 3))
        _, kwargs = mock_predict.call_args
        assert kwargs.get("corrections") == "GOT"

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_infers_minor_constituents(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """infer_minor must be called to add minor constituent contributions."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        coord = Coordinate(lat=40.7, lon=-74.0)
        compute_tides(coord, datetime.date(2025, 12, 3), datetime.date(2025, 12, 3))
        mock_infer.assert_called_once()
        _, kwargs = mock_infer.call_args
        assert kwargs.get("corrections") == "GOT"

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_uses_extrapolation(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """interp must use extrapolate=True for coastal points near land mask."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        coord = Coordinate(lat=40.7, lon=-74.0)
        compute_tides(coord, datetime.date(2025, 12, 3), datetime.date(2025, 12, 3))
        mock_ds = mock_model_cls.return_value.open_dataset.return_value
        _, kwargs = mock_ds.tmd.interp.call_args
        assert kwargs.get("extrapolate") is True

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_interp_uses_lon360(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """interp must use 0-360 longitude to match model grid convention."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        coord = Coordinate(lat=40.7, lon=-74.0)
        compute_tides(coord, datetime.date(2025, 12, 3), datetime.date(2025, 12, 3))
        mock_ds = mock_model_cls.return_value.open_dataset.return_value
        _, kwargs = mock_ds.tmd.interp.call_args
        assert kwargs.get("x") == (-74.0 % 360)  # 286.0

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_accepts_model_name(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """compute_tides accepts a model_name parameter to select different models."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        compute_tides(
            Coordinate(lat=40.7, lon=-74.0),
            datetime.date(2025, 12, 3),
            datetime.date(2025, 12, 3),
            model_name="EOT20",
        )
        mock_model_cls.return_value.from_database.assert_called_once_with("EOT20")

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_fes2022_uses_dask(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """FES models must use dask lazy loading (chunks={}) + .compute()."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        mock_instance = mock_model_cls.return_value
        mock_instance.format = "FES-netcdf"
        # Make the sel().compute() chain work
        mock_ds = mock_instance.open_dataset.return_value
        mock_ds.sel.return_value = mock_ds
        mock_ds.compute.return_value = mock_ds
        compute_tides(
            Coordinate(lat=40.7, lon=-74.0),
            datetime.date(2025, 12, 3),
            datetime.date(2025, 12, 3),
            model_name="FES2022",
        )
        mock_instance.from_database.assert_called_once_with("FES2022")
        _, kwargs = mock_instance.open_dataset.call_args
        assert kwargs.get("chunks") == {}
        mock_ds.compute.assert_called_once()

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_got_format_loads_full_grid_and_uses_lon360(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """pyTMD 3.0.6's GOT reader ignores crop/bounds, so GOT grids are loaded
        whole and interpolated at the non-negative lon360 -- never at a signed
        longitude outside the grid's [0, 360) range (#23 review)."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        mock_model_cls.return_value.format = "GOT-netcdf"
        for lon, expected_x in [(0.5, 0.5), (-0.5, 359.5), (-74.0, 286.0)]:
            day = datetime.date(2025, 12, 3)
            compute_tides(Coordinate(lat=0.0, lon=lon), day, day)
            mock_instance = mock_model_cls.return_value
            _, kwargs = mock_instance.open_dataset.call_args
            assert kwargs.get("crop") is False
            _, ikw = mock_instance.open_dataset.return_value.tmd.interp.call_args
            assert ikw["x"] == pytest.approx(expected_x)

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_date_range(self, mock_ensure, mock_model_cls, mock_predict, mock_infer):
        """Single day spans 24h at 1-min intervals, padded by EDGE_PAD (3 h)
        on each side so extrema at the day edges are detectable (#10)."""
        n = (24 + 6) * 60 // ELEVATION_INTERVAL_MINUTES
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer, n=n)
        coord = Coordinate(lat=40.7, lon=-74.0)
        compute_tides(coord, datetime.date(2025, 12, 3), datetime.date(2025, 12, 3))
        call_args = mock_predict.call_args
        t_array = call_args[0][0]
        assert len(t_array) == n

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_returns_events(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """Mock predict returning a sine wave -> TideEvent list with highs and lows."""
        n = (24 + 6) * 60 // ELEVATION_INTERVAL_MINUTES
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer, n=n)
        events = compute_tides(
            Coordinate(lat=40.7, lon=-74.0), datetime.date(2025, 12, 3), datetime.date(2025, 12, 3)
        )
        assert len(events) > 0
        heights = [e.height for e in events]
        assert any(h > 0.5 for h in heights), "Expected at least one high tide"
        assert any(h < -0.5 for h in heights), "Expected at least one low tide"


class TestSeam:
    @pytest.mark.parametrize(
        ("lon360", "expected"), [(0.1, True), (359.9, True), (179.9, False), (180.1, False)]
    )
    def test_crosses_seam(self, lon360, expected):
        assert crosses_seam(lon360) is expected

    def test_fes_seam_crop_matches_full_grid(self):
        """FES-format grids include both x=0 and x=360. The seam crop must pick
        the right cells (west slice shifted by -360, no duplicate x=0) and give
        the same interpolated value as the full grid at 0.1 and 359.9."""
        import xarray as xr

        from tides import ocean_model

        x = np.arange(0.0, 360.0 + 0.125, 0.125)
        y = np.arange(40.0, 60.0 + 0.125, 0.125)
        # Longitude- and latitude-dependent, smooth and periodic in x.
        field = np.cos(np.radians(x))[None, :] * 2.0 + np.sin(np.radians(x))[None, :]
        field = field + 0.01 * y[:, None]
        grid = xr.Dataset(
            {"m2": (("y", "x"), field.astype(np.complex64))}, coords={"x": x, "y": y}
        )
        captured = {}

        class FakeAccessor:
            def __init__(self, ds):
                self.ds = ds

            def interp(self, **kw):
                captured["ds"] = self.ds
                captured["kw"] = kw
                return self.ds

        mock_model = MagicMock()
        mock_model.format = "FES-netcdf"
        mock_model.open_dataset.return_value = grid

        def full_grid_value(lon360, lat):
            return complex(grid.m2.interp(x=lon360, y=lat).values)

        for lon, lon360 in [(0.1, 0.1), (-0.1, 359.9)]:
            with (
                patch("tides.cache.ensure_model_data"),
                patch("pyTMD.io.model", return_value=mock_model),
                patch.object(
                    xr.Dataset, "tmd", property(lambda self: FakeAccessor(self)), create=True
                ),
            ):
                ocean_model.load_local_constituents.cache_clear()
                ocean_model.load_local_constituents(50.0, lon, "EOT20")

            cropped = captured["ds"]
            xs = cropped.x.values
            assert np.all(np.diff(xs) > 0)
            assert xs.min() >= -2.0 and xs.max() <= 2.0
            # Every cell carries the value of the same longitude on the full grid.
            expected = np.cos(np.radians(xs % 360)) * 2.0 + np.sin(np.radians(xs % 360))
            got = cropped.m2.sel(y=50.0).values.real - 0.5
            assert np.allclose(got, expected, atol=1e-5)
            signed_x = captured["kw"]["x"]
            assert signed_x == pytest.approx(lon)
            seam_value = complex(cropped.m2.interp(x=signed_x, y=50.0).values)
            assert abs(seam_value - full_grid_value(lon360, 50.0)) < 1e-5
