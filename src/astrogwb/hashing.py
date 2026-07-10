"""Content-hash helpers for run provenance.

Stdlib-only on purpose: :mod:`scripts.run_mcmc` must be able to hash the catalog
and the resolved config before JAX initializes, so nothing here may import the
heavy scientific stack (see ``astrogwb.utils``, which imports jax at module
level, for the counterexample).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_HASH_CHUNK_SIZE = 1024 * 1024


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for content digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    """SHA-256 of a JSON-compatible value, independent of whitespace/key order."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
