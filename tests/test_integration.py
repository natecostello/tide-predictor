"""Integration tests that require network access and/or model data.

Run with: pytest tests/test_integration.py -v -m integration
"""

import json
import subprocess

import pytest

pytestmark = pytest.mark.integration

SUBPROCESS_TIMEOUT = 60


class TestNOAAIntegration:
    def test_battery_ny(self):
        """The Battery, NY — a well-known NOAA station."""
        result = subprocess.run(
            [
                "tides",
                "get",
                "40.7006,-74.0142",
                "--date",
                "2026-04-15",
                "--source",
                "noaa",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["source"]["type"] == "noaa"
        assert len(data["days"]) == 1
        assert len(data["days"][0]["tides"]) >= 2

    def test_battery_verbose(self):
        result = subprocess.run(
            [
                "tides",
                "get",
                "40.7006,-74.0142",
                "--date",
                "2026-04-15",
                "--source",
                "noaa",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0
        assert "[NOAA:" in result.stdout


class TestModelIntegration:
    def test_brazil_coast(self):
        """NE Brazil coast — no NOAA station, forces model."""
        result = subprocess.run(
            [
                "tides",
                "get",
                "-8.05,-34.87",
                "--date",
                "2026-04-15",
                "--source",
                "model",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["source"]["type"] == "model"
        assert data["model"] == "GOT5.6"
        assert len(data["days"][0]["tides"]) >= 2


class TestCLIFlags:
    def test_local_time(self):
        result = subprocess.run(
            [
                "tides",
                "get",
                "40.7006,-74.0142",
                "--date",
                "2026-04-15",
                "--source",
                "noaa",
                "--local",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["timezone"] != "UTC"

    def test_feet(self):
        result = subprocess.run(
            [
                "tides",
                "get",
                "40.7006,-74.0142",
                "--date",
                "2026-04-15",
                "--source",
                "noaa",
                "--feet",
            ],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0
        assert "ft@" in result.stdout

    def test_date_range(self):
        result = subprocess.run(
            [
                "tides",
                "get",
                "40.7006,-74.0142",
                "--date",
                "2026-04-15:2026-04-16",
                "--source",
                "noaa",
            ],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0
        assert "2026-04-15:" in result.stdout
        assert "2026-04-16:" in result.stdout

    def test_version(self):
        result = subprocess.run(
            ["tides", "--version"],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0
        assert "tides" in result.stdout
