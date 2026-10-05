"""PSD reference resolution without importing detector geometry or JAX."""

from pathlib import Path
from urllib.parse import urlparse

NOISE_CURVES_BASE_DIR = Path(__file__).parent / "noise_curves"


def resolve_psd_path(reference: str | Path) -> Path | str:
    """Resolve a PSD reference to something gwmock-noise can load.

    Resolution order: gwmock-noise bundled preset -> astrogwb
    ``noise_curves/`` file -> absolute/relative path on disk -> HTTP(S) URL.
    Bundled presets and on-disk files return a ``Path``; URLs return the
    original ``str`` (gwmock-noise fetches them directly).
    """
    reference_str = str(reference)

    # Only bare preset names need gwmock-noise's bundled-resource lookup.
    if not Path(reference_str).suffix and not any(
        separator in reference_str for separator in ("/", "\\")
    ):
        from gwmock_noise.gaussian.psd import resolve_bundled_psd_preset

        bundled = resolve_bundled_psd_preset(reference_str)
        if bundled is not None:
            return bundled

    packaged = NOISE_CURVES_BASE_DIR / reference_str
    if packaged.exists():
        return packaged

    path = Path(reference)
    if path.exists():
        return path

    parsed = urlparse(reference_str)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return reference_str

    raise FileNotFoundError(f"Could not resolve PSD reference: {reference!r}")
