"""Shared pytest configuration for both suites.

The only thing that lives here is ``sys.path``: pytest's default ``prepend``
import mode puts each unpackaged test *directory* on the path, so
``tests/core`` and ``tests/paper`` cannot see a helper module that both need.
``reference_population`` is one -- a hand-written restatement of the physics
that the core distributions and the paper pipeline are each checked against,
and duplicating an oracle would defeat the point of having one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
