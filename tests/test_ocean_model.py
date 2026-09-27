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
    def test_compute_tides_crops_dataset(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """open_dataset must use crop=True with bounds in 0-360 lon space."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        coord = Coordinate(lat=40.7, lon=-74.0)
        compute_tides(coord, datetime.date(2025, 12, 3), datetime.date(2025, 12, 3))
        mock_instance = mock_model_cls.return_value
        _, kwargs = mock_instance.open_dataset.call_args
        assert kwargs.get("crop") is True
        bounds = kwargs.get("bounds")
        assert bounds is not None
        # Bounds in 0-360 space should contain coord.lon % 360
        lon360 = coord.lon % 360
        assert bounds[0] < lon360 < bounds[1]
        assert bounds[2] < coord.lat < bounds[3]

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
    def test_compute_tides_seam_crops_with_signed_bounds(
        self, mock_ensure, mock_model_cls, mock_predict, mock_infer
    ):
        """Windows crossing the 0/360 seam must still crop (never load the full
        grid): GOT-format crops use a west bound < 0, which pyTMD wraps."""
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
        mock_model_cls.return_value.format = "GOT-netcdf"
        for lon, expected_x in [(0.5, 0.5), (-0.5, -0.5)]:
            compute_tides(
                Coordinate(lat=0.0, lon=lon),
                datetime.date(2025, 12, 3),
                datetime.date(2025, 12, 3),
            )
            mock_instance = mock_model_cls.return_value
            _, kwargs = mock_instance.open_dataset.call_args
            assert kwargs.get("crop") is True
            assert kwargs["bounds"][0] == pytest.approx(expected_x - 2.0)
            assert kwargs["bounds"][1] == pytest.approx(expected_x + 2.0)
            _, ikw = mock_instance.open_dataset.return_value.tmd.interp.call_args
            assert ikw["x"] == pytest.approx(expected_x)

    @patch("pyTMD.predict.infer_minor")
    @patch("pyTMD.predict.time_series")
    @patch("pyTMD.io.model")
    @patch("tides.cache.ensure_model_data")
    def test_compute_tides_date_range(self, mock_ensure, mock_model_cls, mock_predict, mock_infer):
        """Single day spans 24h at 1-min intervals (1440 points)."""
        n = 24 * 60 // ELEVATION_INTERVAL_MINUTES
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
        self._setup_pytmd_mocks(mock_ensure, mock_model_cls, mock_predict, mock_infer)
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

    def test_fes_seam_concat_is_strictly_increasing(self):
        """FES-format grids include both x=0 and x=360; the seam crop must not
        duplicate x=0 after shifting the western slice by -360."""
        import xarray as xr

        from tides import ocean_model

        x = np.arange(0.0, 360.0 + 0.125, 0.125)
        y = np.arange(40.0, 60.0 + 0.125, 0.125)
        grid = xr.Dataset(
            {"m2": (("y", "x"), np.ones((len(y), len(x)), dtype=np.complex64))},
            coords={"x": x, "y": y},
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
        with (
            patch("tides.cache.ensure_model_data"),
            patch("pyTMD.io.model", return_value=mock_model),
            patch.object(
                xr.Dataset, "tmd", property(lambda self: FakeAccessor(self)), create=True
            ),
        ):
            ocean_model.load_local_constituents(50.0, -0.1, "EOT20")

        xs = captured["ds"].x.values
        assert np.all(np.diff(xs) > 0)
        assert xs.min() >= -2.0 and xs.max() <= 2.0
        assert captured["kw"]["x"] == pytest.approx(-0.1)
