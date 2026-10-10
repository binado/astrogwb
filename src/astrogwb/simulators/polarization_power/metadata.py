"""What a polarization-power catalog is drawn from, and how every artifact is keyed."""

from __future__ import annotations

from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb import __version__
from astrogwb.populations.metadata import PopulationMetadata, widen_model_kwargs
from astrogwb.simulators.core.keys import content_key
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["CatalogMetadata", "catalog_stem"]


class CatalogMetadata(BaseModel):
    """Everything that determines a polarization-power catalog's contents.

    It is both the request and the provenance, as
    :class:`~astrogwb.simulators.spectra.BackgroundSpectralDensityMetadata` is for spectra: the
    :func:`~astrogwb.simulators.polarization_power.draw_catalog`
    turns it and a key into a catalog, the draw is stored beside it
    (:func:`~astrogwb.simulators.core.write`),
    and :meth:`key` is the middle part of the file name by convention
    (:func:`catalog_stem`). The seed is not part of it: it picks one realization of the
    density this record describes.

    The waveform and population record alone are not enough: two catalogs
    with the same ones still differ if they were drawn at other
    hyperparameters or at another size, and both differ if the code that drew
    them changed. ``fiducials`` and ``num_samples`` close the first gap and
    ``version`` the second.

    Every caller -- the workflow, the generator script, a notebook -- derives
    the key from this one record, so there is one canonical form and no second
    hash to drift.

    ``sampling`` is how a reference catalog
    (:func:`~astrogwb.gwb.importance.reference_catalog`) draws its intrinsic
    parameters: ``"random"`` i.i.d., or ``"sobol"`` on a scrambled Sobol net
    (:func:`~astrogwb.populations.evaluation.sample_sources_qmc`), which needs a
    power-of-two ``num_samples``. A plain catalog is always random.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    population: PopulationMetadata
    fiducials: dict[str, float]
    num_samples: Annotated[int, Field(gt=0)]
    sampling: Literal["random", "sobol"] = "random"
    version: str = __version__

    @model_validator(mode="after")
    def _sobol_needs_a_power_of_two(self) -> Self:
        if self.sampling == "sobol" and self.num_samples & (self.num_samples - 1):
            raise ValueError(
                "sampling='sobol' needs a power-of-two num_samples, "
                f"got {self.num_samples}"
            )
        return self

    def key(self) -> str:
        """The content hash this catalog is cached under.

        Hashes the canonical JSON of the whole record. Numeric construction
        kwargs are widened to ``float``, so ``2`` and ``2.0`` name the same
        catalog; boolean choices retain their type. Everything else is
        already type-stable under strict validation.

        The payload nests ``waveform`` and ``population`` under ``metadata``,
        the shape this record had before it was flattened, so every catalog
        already on disk keeps its key. For the same reason the default
        ``sampling="random"`` is left out of the payload.
        """
        payload = self.model_dump(mode="json")
        if payload["sampling"] == "random":
            del payload["sampling"]
        widen_model_kwargs(payload["population"])
        payload["metadata"] = {
            "waveform": payload.pop("waveform"),
            "population": payload.pop("population"),
        }
        return content_key(payload)


def catalog_stem(metadata: CatalogMetadata, seed: int | np.integer) -> str:
    """The file stem a catalog is kept under by convention: its record and seed.

    ``polarization_power-<key>-<seed>``. A naming convention for the workflow and
    notebooks, not an address: the file itself records the metadata and seed it
    was drawn at, which a reader checks.
    """
    return f"polarization_power-{metadata.key()}-{int(seed)}"
