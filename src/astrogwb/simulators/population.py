"""The population stage: one Poisson realization of every source, per key.

A realization owns every stochastic property of its sources: the
hyperparameters (fixed or drawn once), the Poisson count
:math:`N \\sim \\mathrm{Poisson}(\\mathcal{R}\\, T)`, the intrinsic parameters
and inclination the population model draws, and the extrinsic angles drawn
here -- an isotropic direction (``polar_cosine``, ``azimuth``), the
polarization angle ``polarization`` and the coalescence phase ``coa_phase``.
The downstream stages draw nothing.

One key is one realization. It splits into a hyperparameter key, a count key,
a source key and an extrinsic key, in that order, and event ``i`` is drawn from
``fold_in(source_key, i)`` and ``fold_in(extrinsic_key, i)`` wherever it falls,
so sources are drawn in fixed ``chunk_size`` pieces -- one compile -- and
``chunk_size`` changes cost, never the realization. Source sampling does not use
``numpyro.plate`` at a rate-derived size, so the source draws are independent
of the merger rate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from astrogwb import __version__
from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations.metadata import PopulationMetadata as PopulationSpec
from astrogwb.populations.metadata import widen_model_kwargs
from astrogwb.simulators._io import atomic_h5
from astrogwb.utils import years_to_seconds
from astrogwb.utils.io import content_key, h5py_module, read_tree, write_tree

if TYPE_CHECKING:
    import jax

__all__ = [
    "EXTRINSIC_COLUMNS",
    "Hyperparameter",
    "PopulationArtifact",
    "PopulationMetadata",
    "PopulationSimulator",
    "population_key",
]

#: The schema of a population realization. It is part of
#: :func:`population_key`: version 2 moved the extrinsic draws into this
#: stage, so no realization of version 1 is served under the same key.
SCHEMA_VERSION = 2

#: The source columns this stage draws on top of the population model's.
EXTRINSIC_COLUMNS = ("polar_cosine", "azimuth", "polarization", "coa_phase")

#: One hyperparameter's declaration: the value it is fixed at, or the prior it
#: is sampled from once per realization. Which one is decided by the type alone,
#: so a parameter cannot be both fixed and sampled.
type Hyperparameter = float | DistributionConfig


class PopulationMetadata(BaseModel):
    """Everything that determines a population realization, but its key.

    ``hyperparameters`` holds, per name, either a number -- every realization
    is made at that value -- or a
    :class:`~astrogwb.distributions.config.DistributionConfig` it is drawn
    from, once per realization. Priors are data rather than code, so editing a
    bound re-keys the artifact without a version bump. ``observation_time`` is
    in years and sets the Poisson mean :math:`\\mathcal{R}\\, T` and, downstream,
    the spectrum normalization.

    Neither a seed nor a realization count is part of it: the key is the
    simulator's input, and a realization depends on its own key alone.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    population: PopulationSpec
    hyperparameters: dict[str, Hyperparameter]
    observation_time: Annotated[float, Field(gt=0.0)]
    version: str = __version__

    @property
    def observation_seconds(self) -> float:
        """The observation time in seconds."""
        return years_to_seconds(self.observation_time)

    @property
    def fixed(self) -> dict[str, float]:
        """The hyperparameters every realization shares, by name."""
        return {
            name: float(value)
            for name, value in self.hyperparameters.items()
            if not isinstance(value, DistributionConfig)
        }

    @property
    def sampled(self) -> dict[str, DistributionConfig]:
        """The hyperparameters drawn once per realization, with their priors."""
        return {
            name: value
            for name, value in self.hyperparameters.items()
            if isinstance(value, DistributionConfig)
        }

    def prior_model(self) -> Callable[[], dict[str, jax.Array]]:
        """A zero-argument NumPyro model of the hyperparameters of one realization.

        Sampled hyperparameters are ``numpyro.sample`` sites; fixed ones are
        ``numpyro.deterministic`` sites. The model returns every hyperparameter
        by name, so it composes with ``Predictive``, ``handlers.condition`` and
        ``log_density``.
        """
        # Deferred like ``DistributionConfig.build``: keeps this model light.
        import jax.numpy as jnp
        import numpyro

        declared = {
            name: self.hyperparameters[name] for name in sorted(self.hyperparameters)
        }
        priors = {
            name: value.build()
            for name, value in declared.items()
            if isinstance(value, DistributionConfig)
        }

        def model() -> dict[str, jax.Array]:
            theta: dict[str, jax.Array] = {}
            for name, value in declared.items():
                if name in priors:
                    theta[name] = jnp.asarray(numpyro.sample(name, priors[name]))
                else:
                    theta[name] = jnp.asarray(
                        numpyro.deterministic(
                            name, jnp.asarray(value, dtype=jnp.float64)
                        )
                    )
            return theta

        return model

    def key(self) -> str:
        """The content hash of this record.

        Numeric construction kwargs, fixed values and prior kwargs are widened
        to ``float``, so ``2`` and ``2.0`` name the same record. Boolean
        construction choices retain their type.
        """
        return content_key(self.payload())

    def payload(self) -> dict[str, object]:
        """The canonical JSON-ready form :meth:`key` hashes, with the schema version."""
        payload = self.model_dump(mode="json")
        widen_model_kwargs(payload["population"])
        payload["hyperparameters"] = {
            name: (
                {**value, "kwargs": {k: float(v) for k, v in value["kwargs"].items()}}
                if isinstance(value, dict)
                else float(value)
            )
            for name, value in payload["hyperparameters"].items()
        }
        payload["schema_version"] = SCHEMA_VERSION
        return payload


def population_key(metadata: PopulationMetadata, rng_key: Sequence[int]) -> str:
    """The content hash of the realization of ``metadata`` at the key data ``rng_key``.

    It is the realization's ID, known before anything is drawn.
    """
    return content_key(
        {"metadata": metadata.payload(), "rng_key": [int(v) for v in rng_key]}
    )


@dataclass(frozen=True)
class PopulationArtifact:
    """One population realization, in memory or read from a realization directory.

    ``source_parameters`` holds one ``(count,)`` column per source parameter,
    the extrinsic columns :data:`EXTRINSIC_COLUMNS` included; source ``i`` has
    the stable ID ``i``. ``total_merger_rate`` is the realized
    :math:`\\mathcal{R}` in events per second and ``hyperparameters`` the
    realized values. ``rng_key`` is the key's data, the realization's
    provenance and part of its :meth:`key`.
    """

    metadata: PopulationMetadata
    rng_key: tuple[int, ...]
    total_merger_rate: float
    hyperparameters: dict[str, float]
    source_parameters: dict[str, NDArray[Any]]
    chunk_size: int | None = None

    @property
    def count(self) -> int:
        """The realized number of sources."""
        return int(next(iter(self.source_parameters.values())).shape[0])

    @property
    def source_ids(self) -> NDArray[np.int64]:
        """The stable source IDs, ``arange(count)``."""
        return np.arange(self.count, dtype=np.int64)

    @property
    def observation_seconds(self) -> float:
        """The observation time in seconds."""
        return self.metadata.observation_seconds

    def key(self) -> str:
        """The content hash of this realization: its record and its key."""
        return population_key(self.metadata, self.rng_key)

    def tables(self) -> dict[str, dict[str, Any]]:
        """The ``[population]`` and ``[realization]`` manifest tables.

        ``[realization]`` holds what this realization knows of itself; the
        seed and index it was keyed by are the caller's to add.
        """
        return {
            "population": {
                "metadata": self.metadata.model_dump(mode="json"),
                "key": self.key(),
            },
            "realization": {
                "id": self.key(),
                "rng_key": list(self.rng_key),
                "count": self.count,
                "total_merger_rate": self.total_merger_rate,
                "total_merger_rate_unit": "1 / s",
                "observation_seconds": self.observation_seconds,
                "hyperparameters": self.hyperparameters,
            },
        }

    def write_sources(self, path: str | Path) -> Path:
        """Write the source IDs and parameters to the HDF5 file ``path``, atomically."""
        with atomic_h5(path) as file:
            file.create_dataset("source_id", data=self.source_ids)
            write_tree(file.create_group("source_parameters"), self.source_parameters)
        return Path(path)

    @classmethod
    def from_tables(cls, tables: Mapping[str, Any], sources_path: str | Path) -> Self:
        """A realization from its manifest ``tables`` and the sources file it wrote.

        Raises
        ------
        ValueError
            If the recorded key is not the hash of the recorded metadata and
            rng key, or the sources do not match the recorded count.
        """
        population, realization = tables["population"], tables["realization"]
        h5py = h5py_module()
        with h5py.File(sources_path, "r") as file:
            source_parameters = read_tree(file["source_parameters"])
            source_ids = np.asarray(file["source_id"][()])
        artifact = cls(
            metadata=PopulationMetadata.model_validate(population["metadata"]),
            rng_key=tuple(int(value) for value in realization["rng_key"]),
            total_merger_rate=float(realization["total_merger_rate"]),
            hyperparameters={
                name: float(value)
                for name, value in realization["hyperparameters"].items()
            },
            source_parameters=source_parameters,  # ty: ignore[invalid-argument-type]
            chunk_size=tables.get("execution", {}).get("source_chunk_size"),
        )
        if artifact.key() != population["key"]:
            raise ValueError(
                f"the manifest records population {population['key']}, but its "
                f"metadata and rng key hash to {artifact.key()}"
            )
        if artifact.count != int(realization["count"]) or not np.array_equal(
            source_ids, artifact.source_ids
        ):
            raise ValueError(f"{sources_path}: sources do not match the manifest")
        return artifact


def _hyperparameters_and_count(
    key: jax.Array,
    *,
    population: Any,
    prior_model: Callable[[], Mapping[str, jax.Array]],
    observation_seconds: float,
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    """Hyperparameters, merger rate and Poisson source count of one realization."""
    import jax
    import jax.numpy as jnp
    from numpyro import handlers

    theta_key, count_key, _, _ = jax.random.split(key, 4)
    theta = dict(handlers.seed(prior_model, theta_key)())
    redshift_distribution, _ = population(theta)
    rate = jnp.reshape(redshift_distribution.total_merger_rate(), ())
    count = jax.random.poisson(count_key, rate * observation_seconds).astype(jnp.int64)
    return theta, rate, count


class PopulationSimulator:
    """Draws population realizations of one :class:`PopulationMetadata`.

    ``simulator(key)`` is one :class:`PopulationArtifact`; loop over
    :func:`~astrogwb.simulators.batch_keys` for several. Sources are drawn in
    ``chunk_size`` pieces, one compile and ``ceil(count / chunk_size)`` calls.
    Build once and call often: the simulator holds the jitted stages.
    """

    def __init__(self, metadata: PopulationMetadata, *, chunk_size: int = 8192) -> None:
        import jax

        from astrogwb.simulators._chunks import (
            sample_extrinsic_by_key,
            sample_sources_by_key,
        )

        # x64 before any array: keys are built from 64-bit seeds and the rate
        # must not depend on a waveform import turning it on as a side effect.
        jax.config.update("jax_enable_x64", True)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if metadata.version != __version__:
            raise ValueError(
                f"metadata is for astrogwb {metadata.version}, but {__version__} "
                "is installed; realizations are generated by the code their key names"
            )
        population = metadata.population.build()
        self._metadata = metadata
        self._chunk_size = chunk_size
        # Static configuration is bound by ``partial``, so jit traces only keys.
        self._hyperparameters_and_count = jax.jit(
            partial(
                _hyperparameters_and_count,
                population=population,
                prior_model=metadata.prior_model(),
                observation_seconds=metadata.observation_seconds,
            )
        )
        self._draw_sources = jax.jit(
            partial(sample_sources_by_key, population), static_argnames="size"
        )
        self._draw_extrinsic = jax.jit(sample_extrinsic_by_key, static_argnames="size")

    @property
    def metadata(self) -> PopulationMetadata:
        """The record this simulator draws."""
        return self._metadata

    def __call__(self, key: jax.Array) -> PopulationArtifact:
        """The realization at ``key``, a single typed JAX key."""
        import jax

        if key.shape != ():
            raise ValueError(f"key must be a single key, got shape {key.shape}")
        _, _, source_key, extrinsic_key = jax.random.split(key, 4)
        theta, rate, count_device = self._hyperparameters_and_count(key)
        hyperparameters = {n: float(v) for n, v in theta.items()}
        count = int(count_device)
        pieces = []
        # One piece even for an empty realization, so the columns have names.
        for start in range(0, max(count, 1), self._chunk_size):
            offset = np.uint32(start)
            sources = self._draw_sources(
                hyperparameters, source_key, offset, size=self._chunk_size
            )
            extrinsic = self._draw_extrinsic(
                extrinsic_key, offset, size=self._chunk_size
            )
            pieces.append({**sources, **extrinsic})
        columns = {
            name: np.concatenate([np.asarray(piece[name]) for piece in pieces])[:count]
            for name in pieces[0]
        }
        return PopulationArtifact(
            metadata=self._metadata,
            rng_key=tuple(int(v) for v in np.asarray(jax.random.key_data(key))),
            total_merger_rate=float(rate),
            hyperparameters=hyperparameters,
            source_parameters=columns,
            chunk_size=self._chunk_size,
        )
