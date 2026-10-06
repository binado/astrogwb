"""Content keys for simulator metadata records.

Nothing here imports a physics package, so the ``Snakefile`` can name files
without paying for JAX.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

__all__ = ["CATALOG_KEY_LENGTH", "Keyed", "content_key"]

#: Hex characters of the SHA-256 digest kept as a key. 64 bits is far beyond
#: collision range for a cache of tens of files, and short enough to read in a
#: path.
CATALOG_KEY_LENGTH = 16


def content_key(payload: dict[str, Any]) -> str:
    """The content address of one artifact's canonical JSON record.

    ``payload`` is a record's ``model_dump(mode="json")`` after the record has
    widened whatever it spells two ways -- an ``int`` setting that means the
    same as its ``float`` -- to one form. Sorted keys and fixed separators do
    the rest, so every artifact type hashes the same way and none can drift.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:CATALOG_KEY_LENGTH]


class Keyed(Protocol):
    """A metadata record: everything that determines an artifact, and its hash."""

    def key(self) -> str: ...

    def model_dump_json(self) -> str: ...
