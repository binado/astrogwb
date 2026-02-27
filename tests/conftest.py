import pytest

try:
    import bilby  # noqa: F401

    _BILBY_AVAILABLE = True
except ImportError:
    _BILBY_AVAILABLE = False

try:
    import gwfast  # noqa: F401

    _GWFAST_AVAILABLE = True
except ImportError:
    _GWFAST_AVAILABLE = False

requires_bilby = pytest.mark.skipif(not _BILBY_AVAILABLE, reason="bilby not installed")
requires_gwfast = pytest.mark.skipif(
    not _GWFAST_AVAILABLE, reason="gwfast not installed"
)
