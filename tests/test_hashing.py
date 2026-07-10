"""Tests for the stdlib-only content-hash helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from astrogwb.config.hashing import canonical_json, canonical_sha256, file_sha256


def test_canonical_sha256_is_key_order_invariant() -> None:
    a = {"seed": 42, "catalog": {"path": "out/cat.h5", "f_min": 2.0}}
    b = {"catalog": {"f_min": 2.0, "path": "out/cat.h5"}, "seed": 42}

    assert canonical_sha256(a) == canonical_sha256(b)
    assert canonical_sha256(a) != canonical_sha256({**a, "seed": 43})


def test_canonical_json_is_compact_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"catalog bytes\x00\xff" * 1000
    path = tmp_path / "catalog.h5"
    path.write_bytes(payload)

    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()
