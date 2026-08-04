"""Resolve resources owned by the checked-out paper workspace."""

from __future__ import annotations

from pathlib import Path

_PAPER_PROJECT = Path("packages/astrogwb-paper")


def workspace_root(*, start: Path | None = None) -> Path:
    """Return the monorepo root containing both workspace members.

    ``astrogwb-paper`` is a checkout-only application. A clear error is raised
    when its commands are run from an installation without the workspace assets.
    """
    anchors = [start.resolve()] if start is not None else []
    anchors.extend((Path.cwd().resolve(), Path(__file__).resolve()))

    seen: set[Path] = set()
    for anchor in anchors:
        if anchor.is_file():
            anchor = anchor.parent
        for directory in (anchor, *anchor.parents):
            if directory in seen:
                continue
            seen.add(directory)
            if (
                (directory / "pyproject.toml").is_file()
                and (directory / _PAPER_PROJECT / "pyproject.toml").is_file()
                and (directory / "packages/astrogwb/pyproject.toml").is_file()
            ):
                return directory

    raise FileNotFoundError(
        "astrogwb-paper requires an astrogwb workspace checkout containing "
        "packages/astrogwb and packages/astrogwb-paper"
    )


def paper_project_root() -> Path:
    """Return the root of the ``astrogwb-paper`` workspace member."""
    return workspace_root() / _PAPER_PROJECT
