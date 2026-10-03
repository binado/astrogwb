"""User cache location shared by application entrypoints and worktrees."""

from pathlib import Path

from platformdirs import user_cache_path


def default_cache_dir() -> Path:
    """Return the application cache root, honoring XDG_CACHE_HOME on Unix/macOS.

    Resolve on each call so environment overrides are picked up at runtime.
    This does not create the directory or enter any artifact's metadata/key.
    """
    return user_cache_path("astrogwb")
