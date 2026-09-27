"""Test isolation fixtures.

Production code in `tides.cache`, `tides.datums`, and `tides.ocean_model`
falls back to `$XDG_CACHE_HOME` (or `~/.cache`) for caches and downloads
model data over the network on cache miss. Without isolation, a `pytest`
run on a developer machine silently uses (and writes to) the real user
cache; on a fresh CI runner with no cache, the same code paths reach out
to NASA endpoints and fail when network is unreachable.

The autouse fixtures below redirect HOME / XDG_CACHE_HOME to a per-test
tmp directory, stub `get_model_datums` so unit tests cannot trigger
the "download GOT5.6 to compute datum offsets" path, and block outbound
socket connections so any unmocked network call fails loudly with
NetworkBlockedError.

Integration tests (`@pytest.mark.integration`) opt out -- they need the
real cache and network.
"""

import socket

import pytest


def _is_integration(request: pytest.FixtureRequest) -> bool:
    return request.node.get_closest_marker("integration") is not None


@pytest.fixture(autouse=True)
def _isolate_user_state(
    request: pytest.FixtureRequest,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _is_integration(request):
        return
    cache = tmp_path / "cache"
    home = tmp_path / "home"
    cache.mkdir()
    home.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("HOME", str(home))


@pytest.fixture(autouse=True)
def _stub_model_datums(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _is_integration(request):
        return
    # Empty dict means subsequent .get(..., 0.0) calls in _apply_datum return
    # zero offsets, so heights are not shifted. Tests that need specific
    # datum behavior override this via @patch on the same target.
    monkeypatch.setattr("tides.datums.get_model_datums", lambda *a, **kw: {})


class NetworkBlockedError(RuntimeError):
    """Raised when a unit test attempts a real network connection.

    A RuntimeError subclass (not OSError) so httpx/httpcore do not remap it
    into a ConnectError that production code might catch and swallow.
    """


@pytest.fixture(autouse=True)
def _block_network(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if _is_integration(request):
        return

    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family == getattr(socket, "AF_UNIX", None):
            return real_connect(self, address)
        raise NetworkBlockedError(f"network access blocked in unit tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(
        socket.socket,
        "connect_ex",
        lambda self, address: guarded_connect(self, address),
    )
