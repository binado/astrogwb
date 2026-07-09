from __future__ import annotations

from pathlib import Path

import jax

SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


def repo_root(*, start: Path | None = None) -> Path:
    """Return the astrogwb repository root.

    Walks upward from ``start``, the current working directory, and (when set)
    ``__file__`` until it finds ``pyproject.toml`` and ``src/astrogwb/``. Works
    whether the caller's cwd is the repo root or ``notebooks/``.
    """
    anchors: list[Path] = []
    if start is not None:
        anchors.append(start.resolve())
    anchors.append(Path.cwd().resolve())
    try:
        anchors.append(Path(__file__).resolve().parent.parent.parent)
    except NameError:
        pass

    seen: set[Path] = set()
    for anchor in anchors:
        for directory in (anchor, *anchor.parents):
            if directory in seen:
                continue
            seen.add(directory)
            if (directory / "pyproject.toml").is_file() and (
                directory / "src/astrogwb"
            ).is_dir():
                return directory

    raise FileNotFoundError(
        "could not locate astrogwb repository root "
        "(expected pyproject.toml and src/astrogwb/)"
    )


def years_to_seconds(observation_time_yr: float | jax.Array) -> float | jax.Array:
    """Convert an observation time from years to seconds."""
    return observation_time_yr * SECONDS_PER_YEAR
