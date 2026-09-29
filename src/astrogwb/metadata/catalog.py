"""What a polarization-power catalog is drawn from, and how every artifact is keyed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from astrogwb import __version__
from astrogwb.metadata.population import PopulationMetadata
from astrogwb.metadata.waveform import WaveformMetadata

__all__ = [
    "CATALOG_KEY_LENGTH",
    "CatalogMetadata",
    "Keyed",
    "artifact_path",
    "content_key",
    "widen_model_kwargs",
]

#: Hex characters of the SHA-256 digest kept as a catalog key. 64 bits is far
#: beyond collision range for a cache of tens of files, and short enough to
#: read in a path.
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


def widen_model_kwargs(population: dict[str, Any]) -> None:
    """Widen a dumped population's construction kwargs to ``float`` in place.

    A setting spelled ``2`` in one config and ``2.0`` in another names the
    same draw, so both must hash alike.
    """
    population["model_kwargs"] = {
        name: float(value) for name, value in population["model_kwargs"].items()
    }


class Keyed(Protocol):
    """A metadata record: everything that determines an artifact, and its hash."""

    def key(self) -> str: ...

    def model_dump_json(self) -> str: ...


def artifact_path(metadata: Keyed, cache_dir: str | Path) -> Path:
    """Where ``metadata``'s artifact lives in ``cache_dir``: ``<cache_dir>/<key>.h5``.

    Here rather than beside :func:`astrogwb.catalog.simulate` so that code
    which only names files -- the ``Snakefile`` building its DAG -- reaches it
    without importing JAX.
    """
    return Path(cache_dir) / f"{metadata.key()}.h5"


class CatalogMetadata(BaseModel):
    """Everything that determines a polarization-power catalog's contents.

    It is both the request and the provenance, as
    :class:`~astrogwb.metadata.SpectraMetadata` is for spectra: a
    :class:`~astrogwb.catalog.CatalogGenerator` turns it into a catalog, the
    catalog carries it as :attr:`~astrogwb.catalog.PolarizationPowerCatalog.metadata`,
    and :meth:`key` is the file name :func:`~astrogwb.catalog.simulate` caches
    it under.

    The waveform and population record alone are not enough: two catalogs
    with the same ones still differ if they were drawn at other
    hyperparameters or at another size, and both differ if the code that drew
    them changed. ``fiducials`` and ``num_samples`` close the first gap and
    ``version`` the second -- as far as the package version is bumped when a
    population or waveform change alters a draw.

    Every caller -- the workflow, the generator script, a notebook -- derives
    the key from this one record, so there is one canonical form and no second
    hash to drift.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    population: PopulationMetadata
    fiducials: dict[str, float]
    num_samples: Annotated[int, Field(gt=0)]
    version: str = __version__

    def key(self) -> str:
        """The content hash this catalog is cached under.

        Hashes the canonical JSON of the whole record. Construction kwargs are
        widened to ``float`` first, so a setting spelled ``2`` in one config and
        ``2.0`` in another names the same catalog; everything else is already
        type-stable under strict validation.

        The payload nests ``waveform`` and ``population`` under ``metadata``,
        the shape this record had before it was flattened, so every catalog
        already on disk keeps its key.
        """
        payload = self.model_dump(mode="json")
        widen_model_kwargs(payload["population"])
        payload["metadata"] = {
            "waveform": payload.pop("waveform"),
            "population": payload.pop("population"),
        }
        return content_key(payload)
