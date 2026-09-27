"""Generate an artifact from its metadata, or reuse a cached one.

Three abstractions: a *metadata* record that fully determines an artifact and
names it with :meth:`key`; a *generator* that turns the record into the
artifact; and :func:`simulate`, which serves ``<cache_dir>/<key>.h5`` on a hit
and calls the generator on a miss. :func:`simulate` knows nothing about any
particular artifact. There are two instances of it:

- :class:`~astrogwb.metadata.CatalogMetadata` and
  :class:`~astrogwb.catalog.CatalogGenerator` for polarization-power catalogs;
- :class:`~astrogwb.metadata.SpectraMetadata` and
  :class:`~astrogwb.catalog.SpectrumGenerator` for spectral-density draws.

The cache is content-addressed: an artifact lives at
:func:`~astrogwb.metadata.artifact_path`, so asking for the same thing twice
finds the same file and changing anything -- a seed, a kwarg, the package
version -- names a different one. There is no index to keep in sync. A hit is
still checked against the file's own record, which catches a file copied or
renamed into the wrong place.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Protocol, Self

from astrogwb.metadata import Keyed, artifact_path

__all__ = [
    "Artifact",
    "Generator",
    "Keyed",
    "artifact_path",
    "check_metadata",
    "save_atomically",
    "simulate",
]

logger = logging.getLogger(__name__)


class Artifact[M: Keyed](Protocol):
    """A persisted artifact that carries the record it was generated from."""

    @property
    def metadata(self) -> M: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> Self: ...


class Generator[M: Keyed, A: Artifact](Protocol):
    """Turns a metadata record into its artifact, and names the artifact type."""

    @property
    def artifact(self) -> type[A]: ...

    def __call__(self, metadata: M) -> A: ...


def simulate[M: Keyed, A: Artifact](
    metadata: M, generator: Generator[M, A], cache_dir: str | Path | None = None
) -> A:
    """Return ``metadata``'s artifact from ``cache_dir``, generating it on a miss.

    Without a ``cache_dir`` this is ``generator(metadata)``. A hit is loaded as
    ``generator.artifact`` and its recorded metadata compared with
    ``metadata``; a mismatch -- a file copied or renamed into the wrong key --
    raises rather than serving an artifact of something else. A miss is
    generated and saved atomically under the key.
    """
    if cache_dir is None:
        return generator(metadata)

    path = artifact_path(metadata, cache_dir)
    if path.is_file():
        artifact = generator.artifact.load(path)
        check_metadata(artifact, metadata, label=str(path))
        logger.info("%s: cache hit at %s", metadata.key(), path)
        return artifact

    logger.info("%s: cache miss, generating into %s", metadata.key(), path)
    artifact = generator(metadata)
    save_atomically(artifact, path)
    return artifact


def check_metadata(artifact: Artifact, metadata: Keyed, *, label: str) -> None:
    """Raise unless ``artifact`` records exactly ``metadata``.

    What :func:`simulate` checks a cache hit with, and what a caller handed a
    file by path -- rather than by cache directory -- checks it with, so a file
    given to the wrong role or built from a draw nobody asks for any more is
    refused before it is used.
    """
    recorded = artifact.metadata
    if recorded.key() != metadata.key():
        raise ValueError(
            f"{label} records {recorded.key()}, not the requested "
            f"{metadata.key()}:\n  recorded:  {recorded.model_dump_json()}\n"
            f"  requested: {metadata.model_dump_json()}"
        )


def save_atomically(artifact: Artifact, path: str | Path) -> None:
    """Write ``artifact`` to ``path`` without ever exposing a partial file.

    The file is written beside its destination and renamed into place, so a
    reader -- or a second writer racing on the same key -- sees either no file
    or a complete one. Two racing writers produce the same bytes, so the last
    rename winning is harmless.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".h5.tmp"
    )
    os.close(handle)
    try:
        artifact.save(temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
