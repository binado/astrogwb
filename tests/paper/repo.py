"""Where the checkout is, for tests that read committed config off disk.

``astrogwb.paper.paths`` used to answer this by walking up looking for a
workspace. It is gone: with one package and the workflow's cwd at the
repository root, library code names only relative paths and the caller's cwd is
the answer.

Tests are the exception -- pytest can be invoked from anywhere -- so they
anchor on this file's own location rather than on cwd. That is a test concern
and it lives with the tests, not in the library.
"""

from __future__ import annotations

from pathlib import Path

#: tests/paper/repo.py -> tests/paper -> tests -> the checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
