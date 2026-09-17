"""Locate the checkout that owns ``config/``.

The workflow and every script run from the repository root, so for them the
caller's cwd *is* the checkout and this module is a no-op. A notebook is the
exception: Jupyter's kernel cwd is the notebook's own directory
(``notebooks/``), so every accessor that resolves ``config/*.json`` against the
cwd fails there. :func:`root_dir` is the fallback those accessors use when they
are handed no ``root=``.

**stdlib only**, like :mod:`astrogwb.paper.utils`: the ``Snakefile`` imports the
config layer to build its DAG, so this must stay free of pydantic, xarray, and
JAX. It also does no I/O at import -- the walk happens when the function is
called.

The name is deliberate. ``astrogwb.paper.paths`` used to resolve a two-package
workspace and was deleted when the packages merged (see ``tests/paper/repo.py``).
What returns here is narrower: one function, one marker, no workspace concept.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

#: The file whose presence marks the checkout root. ``pyproject.toml`` rather
#: than ``.git``: it is committed, so it survives ``git archive`` and an sdist
#: export where ``.git`` does not, it is always a *file* (``.git`` is a file in
#: a worktree or submodule, which an ``is_dir()`` probe would miss), and it is
#: the marker ``uv`` itself uses to identify a project root.
_MARKER = "pyproject.toml"


@cache
def root_dir() -> Path:
    """The root the ``config/`` tree lives under.

    The nearest ancestor of the cwd (inclusive) holding a ``pyproject.toml``,
    or the cwd itself when there is none. That second case is the contract the
    workflow already relies on rather than a failure fallback: a snakemake dry
    run is pointed at a scratch directory holding only symlinks to ``Snakefile``,
    ``config`` and ``scripts`` (``tests/paper/test_workflows.py``), with no
    ``pyproject.toml`` anywhere above it, and it must resolve ``config/``
    against that cwd. So walk up while there is something to find, otherwise
    stay put -- which is exactly the cwd-relative behaviour every caller had
    before, plus the notebook whose kernel cwd is ``notebooks/``.

    Cached: the checkout cannot move within a process. A caller that changes the
    cwd and wants a fresh answer clears it with ``root_dir.cache_clear()``.
    """
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        if (directory / _MARKER).is_file():
            return directory
    return cwd
