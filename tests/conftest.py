import pytest

try:
    import bilby  # noqa: F401

    _BILBY_AVAILABLE = True
except ImportError:
    _BILBY_AVAILABLE = False

requires_bilby = pytest.mark.skipif(not _BILBY_AVAILABLE, reason="bilby not installed")
