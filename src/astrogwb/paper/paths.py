"""Resolve resources owned by the checked-out paper workspace."""

from __future__ import annotations

from pathlib import Path

#: Where the paper application's committed assets (config/, scripts/, the
#: Snakefile) still live. They move to the repository root next, at which point
#: this module goes away entirely: every caller becomes a plain relative path.
_PAPER_PROJECT = Path("packages/astrogwb-paper")


def workspace_root(*, start: Path | None = None) -> Path:
    """Return the repository root holding the checked-out paper assets.

    ``astrogwb.paper`` is a checkout-only application. A clear error is raised
    when its commands are run from an installation without those assets.
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
                and (directory / "src/astrogwb/paper").is_dir()
                and (directory / _PAPER_PROJECT / "config").is_dir()
            ):
                return directory

    raise FileNotFoundError(
        "astrogwb.paper requires an astrogwb checkout containing "
        f"src/astrogwb/paper and {_PAPER_PROJECT}/config"
    )


def paper_project_root() -> Path:
    """Return the directory holding the paper application's committed assets."""
    return workspace_root() / _PAPER_PROJECT


def resolve_paper_path(path: Path, root: Path | None = None) -> Path:
    """Resolve a possibly relative path against the paper workspace member.

    Absolute paths pass through unchanged. Callers that resolve several paths
    should look ``root`` up once and pass it in.
    """
    if path.is_absolute():
        return path
    return (root if root is not None else paper_project_root()) / path
