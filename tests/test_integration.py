"""Integration tests that require network access and/or model data.

Run with: pytest tests/test_integration.py -v -m integration
"""

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

SUBPROCESS_TIMEOUT = 60


class TestNOAAIntegration:
    def test_battery_ny(self):
        """The Battery, NY -- a well-known NOAA station."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tides",
                "peaks",
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
                sys.executable,
                "-m",
                "tides",
                "peaks",
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
        """NE Brazil coast -- no NOAA station, forces model."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tides",
                "peaks",
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
                sys.executable,
                "-m",
                "tides",
                "peaks",
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
                sys.executable,
                "-m",
                "tides",
                "peaks",
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
                sys.executable,
                "-m",
                "tides",
                "peaks",
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
            [sys.executable, "-m", "tides", "--version"],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        assert result.returncode == 0
        assert "tides" in result.stdout


def _tides(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tides", *args],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )


class TestLevelWhenIntegration:
    def test_noaa_level_at_hilo_time_is_turn(self):
        """Live NOAA 6-minute series agrees with the official hilo product."""
        peaks = _tides(
            "peaks", "40.7006,-74.0142", "-d", "2026-04-15", "-s", "noaa", "-j", "-p", "3"
        )
        assert peaks.returncode == 0, peaks.stderr
        rows = json.loads(peaks.stdout)["days"][0]["tides"]
        whens = [a for r in rows for a in ("--when", r["datetime"][:16])]
        level = _tides("level", "40.7006,-74.0142", *whens, "-s", "noaa", "-j", "-p", "3")
        assert level.returncode == 0, level.stderr
        got = json.loads(level.stdout)["days"][0]["tides"]
        for a, b in zip(rows, got):
            assert b["type"] == a["type"]
            assert abs(b["height"] - a["height"]) < 0.02

    def test_model_when_round_trip(self):
        """Ilha do Guajiru on GOT5.6 (needs the model cache)."""
        coord = "-2.8810722,-39.9083908"
        base = ["--local", "--feet", "--source", "model", "--datum", "lat"]
        r = _tides("when", coord, "--level", "6.7", "--date", "2026-10-09", *base)
        assert r.returncode == 0, r.stderr
        rows = r.stdout.strip().split(", ")
        assert [row.split(" ")[1] for row in rows] == ["rising", "falling", "rising", "falling"]
        r = _tides("when", coord, "--level", "11", "--date", "2026-10-09", *base)
        assert r.returncode == 0 and r.stdout == ""
        assert "Note: level 11.0ft not reached on 2026-10-09 (max " in r.stderr
