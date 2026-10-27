r"""The waveform stage: complex radiation-frame polarizations of every source.

The polarizations :math:`(\tilde h_+, \tilde h_\times)` of a source depend on
its intrinsic parameters, inclination and coalescence phase only. Its sky
position and polarization angle are left to the antenna patterns of the
background stage, so the polarization angle is applied exactly once, at
projection. Strain is in :math:`\mathrm{Hz}^{-1}`, the Fourier convention
:math:`\tilde h(f) = \int h(t)\, e^{-2\pi i f t}\, dt` of the generator, on the
generator's frequency grid. Nothing here draws a random number.

Both sources of polarizations implement one minimal interface,
:class:`WaveformChunks`: :class:`Waveforms`, generated on the fly from a
population, and :class:`WaveformArtifact`, read from disk. Each yields
fixed-size chunks ``(source_parameters, plus, cross)``; the last is padded with
copies of its first source whose polarizations are zero, so a consumer can sum
over a chunk without a mask and compiles once whatever the realized count.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from astrogwb.simulators._chunks import iter_rows, pad_rows
from astrogwb.simulators._io import (
    DATA_FILE,
    read_manifest,
    relative_reference,
    resolve_reference,
    write_manifest,
)
from astrogwb.simulators.population import PopulationArtifact
from astrogwb.utils.io import content_key, h5py_module
from astrogwb.waveform.metadata import WaveformMetadata

if TYPE_CHECKING:
    import jax

    from astrogwb.waveform import PolarizationPowerGenerator

__all__ = [
    "WaveformArtifact",
    "WaveformChunks",
    "WaveformSimulator",
    "Waveforms",
]

KIND = "waveforms"
SCHEMA_VERSION = 1

#: The conventions a waveform manifest records, so a reader can check them.
CONVENTIONS = {
    "frame": "radiation",
    "polarization_angle": "applied_at_projection",
    "fourier": "h(f) = int h(t) exp(-2 pi i f t) dt",
    "strain_unit": "1 / Hz",
    "frequency_unit": "Hz",
}

#: Where a population that was never written goes when its waveforms are.
_POPULATION_DIRECTORY = "population"

#: One chunk: padded source columns ``(C,)`` and polarizations ``(C, F)``.
type Chunk = tuple[dict[str, NDArray[Any]], jax.Array, jax.Array]


class WaveformChunks(Protocol):
    """Polarizations of one population, chunk by chunk.

    What :class:`~astrogwb.simulators.BackgroundSimulator` consumes, wherever
    the chunks come from.
    """

    @property
    def metadata(self) -> WaveformMetadata: ...

    @property
    def population(self) -> PopulationArtifact: ...

    @property
    def frequencies(self) -> NDArray[np.float64]: ...

    def key(self) -> str: ...

    def iter_chunks(self, chunk_size: int) -> Iterator[Chunk]:
        """Yield ``(source_parameters, hplus, hcross)`` for consecutive source chunks."""
        ...


def _waveform_key(metadata: WaveformMetadata, population: PopulationArtifact) -> str:
    """The content hash of a population's waveforms."""
    return content_key(
        {
            "population": population.key(),
            "waveform": metadata.model_dump(mode="json"),
            "schema_version": SCHEMA_VERSION,
        }
    )


class WaveformSimulator:
    """Generates the polarizations of populations through one waveform.

    ``simulator(population)`` returns :class:`Waveforms`, which generate
    lazily: nothing is evaluated until its chunks are iterated or it is
    written, and at most one chunk of polarizations is held at a time. Build
    once and call often: the generator and the compiled chunk are reused, and
    one chunk shape is one compile.

    With ``validate_sources``, each population is checked eagerly against the
    approximant before generation (see
    :meth:`~astrogwb.waveform.PolarizationPowerGenerator.check_sources`), since
    generation is trace-safe and would otherwise drop a degree of freedom the
    approximant does not carry without a word.
    """

    def __init__(
        self,
        metadata: WaveformMetadata,
        *,
        chunk_size: int = 1024,
        validate_sources: bool = True,
    ) -> None:
        import jax

        jax.config.update("jax_enable_x64", True)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._metadata = metadata
        self._chunk_size = chunk_size
        self._validate_sources = validate_sources
        self._generator: PolarizationPowerGenerator = metadata.build()
        self._frequencies = np.asarray(self._generator.frequencies, dtype=np.float64)
        self._polarizations = jax.jit(self._chunk_polarizations)

    @property
    def metadata(self) -> WaveformMetadata:
        """The waveform this simulator generates."""
        return self._metadata

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """The frequency grid in Hz, ``(F,)``."""
        return self._frequencies

    @property
    def chunk_size(self) -> int:
        """The number of sources written per chunk."""
        return self._chunk_size

    def _chunk_polarizations(
        self, sources: Mapping[str, jax.Array], valid: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """The polarizations of one padded chunk, zero past the ``valid`` rows."""
        import jax.numpy as jnp

        plus, cross = self._generator.generate_polarizations_batch(sources)
        real = (jnp.arange(plus.shape[0]) < valid)[:, None]
        return jnp.where(real, plus, 0.0), jnp.where(real, cross, 0.0)

    def generate(
        self, population: PopulationArtifact, chunk_size: int
    ) -> Iterator[Chunk]:
        """The chunks of ``population``'s polarizations, generated one at a time."""
        import jax.numpy as jnp

        for start, _ in iter_rows(population.count, chunk_size):
            rows, valid = pad_rows(population.source_parameters, start, chunk_size)
            plus, cross = self._polarizations(rows, jnp.asarray(valid))
            yield rows, plus, cross

    def __call__(self, population: PopulationArtifact) -> Waveforms:
        """The lazily generated polarizations of ``population``.

        Raises
        ------
        KeyError
            If the population has no ``luminosity_distance`` column.
        ValueError
            If ``validate_sources`` and a source value is outside what the
            approximant represents.
        """
        from astrogwb.waveform.validation import require_luminosity_distance

        require_luminosity_distance(population.source_parameters)
        if self._validate_sources and population.count:
            self._generator.check_sources(population.source_parameters)
        return Waveforms(self, population)


@dataclass(frozen=True)
class Waveforms:
    """A population's polarizations, generated on the fly; see :class:`WaveformChunks`."""

    simulator: WaveformSimulator
    population: PopulationArtifact

    @property
    def metadata(self) -> WaveformMetadata:
        """The waveform the polarizations are generated with."""
        return self.simulator.metadata

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """The frequency grid in Hz, ``(F,)``."""
        return self.simulator.frequencies

    def key(self) -> str:
        """The content hash of these polarizations."""
        return _waveform_key(self.metadata, self.population)

    def iter_chunks(self, chunk_size: int) -> Iterator[Chunk]:
        """Yield ``(source_parameters, hplus, hcross)`` for consecutive source chunks."""
        return self.simulator.generate(self.population, chunk_size)

    def write(self, directory: str | Path) -> WaveformArtifact:
        """Write the polarizations chunk by chunk to ``directory`` and open them.

        The manifest references the population's manifest. A population that
        was never written is written first, to ``directory/population``.
        """
        directory = Path(directory)
        population = self.population
        if population.manifest_path is None:
            population = population.write(directory / _POPULATION_DIRECTORY)
        assert population.manifest_path is not None
        count, size = population.count, self.frequencies.shape[0]
        chunk_size = self.simulator.chunk_size
        directory.mkdir(parents=True, exist_ok=True)
        h5py = h5py_module()
        with h5py.File(directory / DATA_FILE, "w") as file:
            file.create_dataset("frequencies", data=self.frequencies)
            file.create_dataset("source_id", data=population.source_ids)
            planes = {
                name: file.create_dataset(
                    name,
                    shape=(count, size),
                    dtype=np.complex128,
                    chunks=(max(1, min(count, chunk_size)), size) if count else None,
                )
                for name in ("plus", "cross")
            }
            for (start, stop), (_, plus, cross) in zip(
                iter_rows(count, chunk_size),
                self.simulator.generate(population, chunk_size),
                strict=True,
            ):
                planes["plus"][start:stop] = np.asarray(plus)[: stop - start]
                planes["cross"][start:stop] = np.asarray(cross)[: stop - start]
        path = write_manifest(
            directory,
            kind=KIND,
            schema_version=SCHEMA_VERSION,
            key=_waveform_key(self.metadata, population),
            tables={
                "metadata": self.metadata.model_dump(mode="json"),
                "parent": {
                    "kind": "population",
                    "key": population.key(),
                    "manifest": relative_reference(population.manifest_path, directory),
                },
                "conventions": {**CONVENTIONS, "source_ordering": "source_id"},
                "execution": {"chunk_size": chunk_size},
            },
        )
        return WaveformArtifact.open(path)


@dataclass(frozen=True)
class WaveformArtifact:
    """A population's polarizations read from disk; see :class:`WaveformChunks`.

    Opening resolves and checks the parent population but reads no
    polarization: chunks are read from ``data.h5`` as they are iterated.
    """

    metadata: WaveformMetadata
    population: PopulationArtifact
    frequencies: NDArray[np.float64]
    manifest_path: Path

    def key(self) -> str:
        """The content hash of these polarizations."""
        return _waveform_key(self.metadata, self.population)

    @property
    def data_path(self) -> Path:
        """The HDF5 file holding the polarizations."""
        return self.manifest_path.parent / DATA_FILE

    @classmethod
    def open(cls, path: str | Path) -> Self:
        """Read a waveform artifact from its manifest or directory.

        Raises
        ------
        FileNotFoundError
            If there is no waveform manifest at ``path``, or the population it
            references is missing.
        ValueError
            If the referenced population is not the one the waveforms were
            generated from, or the sources are not aligned with it.
        """
        manifest = read_manifest(path, kind=KIND, schema_version=SCHEMA_VERSION)
        manifest_path = manifest["artifact"]["path"]
        parent = manifest["parent"]
        population_path = resolve_reference(manifest, parent["manifest"])
        try:
            population = PopulationArtifact.open(population_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{manifest_path} references a population at {population_path}, "
                "which is missing"
            ) from error
        if population.key() != parent["key"]:
            raise ValueError(
                f"{manifest_path} was generated from population {parent['key']}, "
                f"but {population_path} is population {population.key()}"
            )
        h5py = h5py_module()
        with h5py.File(
            manifest_path.parent / manifest["artifact"]["data"], "r"
        ) as file:
            frequencies = np.asarray(file["frequencies"][()], dtype=np.float64)
            source_ids = np.asarray(file["source_id"][()])
            shape = file["plus"].shape
        if not np.array_equal(source_ids, population.source_ids) or shape != (
            population.count,
            frequencies.shape[0],
        ):
            raise ValueError(
                f"{manifest_path}: polarizations {shape} are not aligned with the "
                f"{population.count} sources of {population_path}"
            )
        artifact = cls(
            metadata=WaveformMetadata.model_validate(manifest["metadata"]),
            population=population,
            frequencies=frequencies,
            manifest_path=manifest_path,
        )
        if artifact.key() != manifest["artifact"]["key"]:
            raise ValueError(
                f"{manifest_path} records key {manifest['artifact']['key']}, but "
                f"its metadata and population hash to {artifact.key()}"
            )
        return artifact

    def iter_chunks(self, chunk_size: int) -> Iterator[Chunk]:
        """Yield ``(source_parameters, hplus, hcross)`` for consecutive source chunks."""
        import jax.numpy as jnp

        h5py = h5py_module()
        with h5py.File(self.data_path, "r") as file:
            for start, stop in iter_rows(self.population.count, chunk_size):
                rows, valid = pad_rows(
                    self.population.source_parameters, start, chunk_size
                )
                planes = []
                for name in ("plus", "cross"):
                    plane = np.zeros(
                        (chunk_size, self.frequencies.shape[0]), dtype=np.complex128
                    )
                    plane[:valid] = file[name][start:stop]
                    planes.append(jnp.asarray(plane))
                yield rows, planes[0], planes[1]
