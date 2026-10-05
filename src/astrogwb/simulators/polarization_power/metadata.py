"""What a polarization-power catalog is drawn from, and how every artifact is keyed."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from astrogwb import __version__
from astrogwb.populations.metadata import PopulationMetadata, widen_model_kwargs
from astrogwb.simulators.core.keys import content_key
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["CatalogMetadata"]


class CatalogMetadata(BaseModel):
    """Everything that determines a polarization-power catalog's contents.

    It is both the request and the provenance, as
    :class:`~astrogwb.simulators.spectra.SpectraMetadata` is for spectra: the
    :func:`~astrogwb.simulators.polarization_power.polarization_power` node
    turns it and a seed into a catalog, the catalog carries it as
    :attr:`~astrogwb.simulators.polarization_power.PolarizationPowerCatalog.metadata`,
    and :meth:`key` is the middle part of the file name the node caches it
    under. The seed is not part of it: it picks one realization of the
    density this record describes.

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

        Hashes the canonical JSON of the whole record. Numeric construction
        kwargs are widened to ``float``, so ``2`` and ``2.0`` name the same
        catalog; boolean choices retain their type. Everything else is
        already type-stable under strict validation.

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
