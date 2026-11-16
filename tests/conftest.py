"""Shared pytest configuration for the library suite.

Puts ``tests/`` on ``sys.path``: pytest's ``prepend`` import mode adds only
each test directory, so ``tests/core`` cannot otherwise import a helper such
as ``reference_population``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
