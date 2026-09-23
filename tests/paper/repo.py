"""Where the checkout is, for tests that read committed config off disk.

:func:`astrogwb.paper.paths.root_dir` walks up from the cwd looking for a
``pyproject.toml``; that is the fallback the paper accessors use when they are
handed no ``root=``, and it is what a notebook relies on.

Tests do not use it. Pytest can be invoked from anywhere *and* a stray
``pyproject.toml`` above the checkout would mis-anchor a cwd-relative walk, so
this module anchors on its own location instead. That is a test concern and it
lives with the tests, not in the library.
"""

from __future__ import annotations

from pathlib import Path

#: tests/paper/repo.py -> tests/paper -> tests -> the checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
