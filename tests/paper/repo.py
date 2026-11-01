"""Where the checkout is, for tests that read committed config off disk.

Library code resolves paths against the cwd (the repository root); pytest can
run from anywhere, so tests anchor on this file's location instead.
"""

from __future__ import annotations

from pathlib import Path

#: tests/paper/repo.py -> tests/paper -> tests -> the checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
