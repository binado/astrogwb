"""Combined provenance carried by every catalog, and the request that keys it."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from astrogwb import __version__
from astrogwb.metadata.population import PopulationMetadata
from astrogwb.metadata.waveform import WaveformMetadata

if TYPE_CHECKING:
    from astrogwb.catalog import PolarizationPowerCatalog

__all__ = ["CATALOG_KEY_LENGTH", "CatalogMetadata", "CatalogRequest"]

#: Hex characters of the SHA-256 digest kept as a catalog key. 64 bits is far
#: beyond collision range for a cache of tens of files, and short enough to
#: read in a path.
CATALOG_KEY_LENGTH = 16


class CatalogMetadata(BaseModel):
    """Waveform and population provenance as one immutable record."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    population: PopulationMetadata


class CatalogRequest(BaseModel):
    """Everything that determines a polarization-power catalog's contents.

    :class:`CatalogMetadata` alone is not enough: two catalogs with the same
    waveform and population record still differ if they were drawn at other
    hyperparameters or at another size, and both differ if the code that drew
    them changed. ``fiducials`` and ``num_samples`` close the first gap and
    ``version`` the second -- as far as the package version is bumped when a
    population or waveform change alters a draw.

    :meth:`key` is the content address a catalog is cached under. Every caller
    -- the workflow, the generator script, a notebook -- derives it from this
    one record, so there is one canonical form and no second hash to drift.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    metadata: CatalogMetadata
    fiducials: dict[str, float]
    num_samples: Annotated[int, Field(gt=0)]
    version: str = __version__

    def key(self) -> str:
        """The content hash this catalog is cached under.

        Hashes the canonical JSON of the whole record. Construction kwargs are
        widened to ``float`` first, so a setting spelled ``2`` in one config and
        ``2.0`` in another names the same catalog; everything else is already
        type-stable under strict validation.
        """
        payload = self.model_dump(mode="json")
        population = payload["metadata"]["population"]
        population["model_kwargs"] = {
            name: float(value) for name, value in population["model_kwargs"].items()
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return digest[:CATALOG_KEY_LENGTH]

    @classmethod
    def from_catalog(cls, catalog: PolarizationPowerCatalog) -> CatalogRequest:
        """The request a loaded catalog answers, as its file records it."""
        return cls(
            metadata=CatalogMetadata(
                waveform=catalog.waveform_metadata, population=catalog.population
            ),
            fiducials=dict(catalog.fiducials),
            num_samples=catalog.num_samples,
            version=catalog.version,
        )

    @classmethod
    def from_blocks(
        cls,
        *,
        population: dict[str, Any],
        waveform: dict[str, Any],
        fiducials: dict[str, float],
        seed: int,
        num_samples: int,
        version: str | None = None,
    ) -> CatalogRequest:
        """Assemble a request from config-shaped blocks.

        ``population`` is the config's ``{model_name, model_kwargs}`` block:
        the seed belongs to the draw, so it is supplied on its own and folded
        into the record here. A block that declares one anyway is rejected
        rather than silently overridden.
        """
        if "seed" in population:
            raise ValueError("population may not declare seed: it is the draw's own")
        data: dict[str, Any] = {
            "metadata": {
                "waveform": waveform,
                "population": {**population, "seed": seed},
            },
            "fiducials": fiducials,
            "num_samples": num_samples,
        }
        if version is not None:
            data["version"] = version
        return cls.model_validate(data)
