import os
from unittest.mock import patch

from tides.cache import get_cache_dir


class TestGetCacheDir:
    def test_default_cache_dir(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("XDG_CACHE_HOME", None)
            d = get_cache_dir()
            assert d.name == "tides"
            assert d.parent.name == ".cache"

    def test_xdg_cache_home(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_cache_dir()
            assert d == tmp_path / "tides"

    def test_cache_dir_is_created(self, tmp_path):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
            d = get_cache_dir()
            assert d.exists()
