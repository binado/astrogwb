"""One realization per directory, its stages added as they are configured.

A realization directory holds one ``metadata.toml`` and at most one product of
each stage (see :mod:`astrogwb.simulators._io`). A stage's manifest table means
the stage is configured; its product file existing means it is complete.
:class:`RealizationSimulator` runs, per directory, every stage whose settings
it was given, and adds to what is already there:

- settings given and none recorded: they are recorded, and the stage runs if
  its product is wanted;
- the same settings recorded: nothing reruns, a wanted but missing product is
  regenerated;
- other settings recorded: :class:`ValueError`;
- settings recorded but not given: they are used when a later stage needs them;
- neither: the stage does not run, and a later stage needing it raises.

Upstream data missing from disk is regenerated in memory: the sources from the
recorded population metadata and ``rng_key``, the polarizations chunk by chunk
from the ``[waveforms]`` metadata. Nothing upstream is redrawn differently, as
the population stage is the only one that draws random numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from astrogwb.simulators._io import (
    BACKGROUND_FILE,
    MANIFEST_FILE,
    POLARIZATION_FILE,
    SOURCES_FILE,
    read_realization_manifest,
    realization_key,
    write_realization_manifest,
)
from astrogwb.simulators.background import (
    BackgroundArtifact,
    BackgroundSimulator,
    DetectorConfiguration,
    background_table,
)
from astrogwb.simulators.population import (
    PopulationArtifact,
    PopulationMetadata,
    PopulationSimulator,
    population_key,
)
from astrogwb.simulators.waveforms import (
    WaveformArtifact,
    Waveforms,
    WaveformSimulator,
    waveform_table,
)
from astrogwb.waveform.metadata import WaveformMetadata

if TYPE_CHECKING:
    from astrogwb.simulators.waveforms import WaveformChunks

__all__ = [
    "Realization",
    "RealizationSimulator",
    "simulate_ensemble",
    "simulate_realization",
]


@dataclass(frozen=True)
class Realization:
    """A realization directory and its manifest.

    The accessors :meth:`population`, :meth:`waveforms` and :meth:`background`
    read a stage's product when it is on disk; given the stage simulators, they
    regenerate what is missing instead.
    """

    directory: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, directory: str | Path) -> Self:
        """The realization at ``directory``.

        Raises
        ------
        FileNotFoundError
            If ``directory`` holds no manifest.
        ValueError
            If the manifest is of another layout version.
        """
        directory = Path(directory)
        return cls(directory, read_realization_manifest(directory))

    @property
    def id(self) -> str:
        """The realization ID, its population key."""
        return str(self.manifest["realization"]["id"])

    @property
    def seed(self) -> int | None:
        """The seed the realization was keyed by, if recorded."""
        return self.manifest["realization"].get("seed")

    @property
    def index(self) -> int | None:
        """The index ``i`` of ``batch_keys(seed, i + 1)[i]``, if recorded."""
        return self.manifest["realization"].get("index")

    @property
    def rng_key(self) -> tuple[int, ...]:
        """The data of the realization's key."""
        return tuple(int(v) for v in self.manifest["realization"]["rng_key"])

    @property
    def count(self) -> int:
        """The realized number of sources."""
        return int(self.manifest["realization"]["count"])

    @property
    def total_merger_rate(self) -> float:
        """The realized total merger rate, per second."""
        return float(self.manifest["realization"]["total_merger_rate"])

    @property
    def hyperparameters(self) -> dict[str, float]:
        """The realized hyperparameters, by name."""
        return {
            name: float(value)
            for name, value in self.manifest["realization"]["hyperparameters"].items()
        }

    @property
    def population_metadata(self) -> PopulationMetadata:
        """The recorded population settings."""
        return PopulationMetadata.model_validate(
            self.manifest["population"]["metadata"]
        )

    @property
    def waveform_metadata(self) -> WaveformMetadata | None:
        """The recorded waveform settings, ``None`` if none are."""
        table = self.manifest.get("waveforms")
        return (
            None
            if table is None
            else WaveformMetadata.model_validate(table["metadata"])
        )

    @property
    def detectors(self) -> DetectorConfiguration | None:
        """The recorded background network, ``None`` if none is."""
        table = self.manifest.get("background")
        return (
            None
            if table is None
            else DetectorConfiguration.model_validate(table["metadata"]["detectors"])
        )

    def population(
        self, simulator: PopulationSimulator | None = None
    ) -> PopulationArtifact:
        """The sources, read from ``source_parameters.h5`` or redrawn by ``simulator``.

        Raises
        ------
        FileNotFoundError
            If the sources are not on disk and no simulator is given.
        ValueError
            If the sources, read or redrawn, are not the recorded realization.
        """
        path = self.directory / SOURCES_FILE
        if path.is_file():
            return PopulationArtifact.from_tables(self.manifest, path)
        if simulator is None:
            raise FileNotFoundError(
                f"{path} is missing; pass a PopulationSimulator to redraw it"
            )
        import jax
        import jax.numpy as jnp

        drawn = simulator(
            jax.random.wrap_key_data(jnp.asarray(self.rng_key, dtype=jnp.uint32))
        )
        if drawn.key() != self.id or drawn.count != self.count:
            raise ValueError(
                f"redrawing {self.directory} gave population {drawn.key()} of "
                f"{drawn.count} sources, not the recorded {self.id} of {self.count}"
            )
        return drawn

    def waveforms(
        self,
        simulator: WaveformSimulator | None = None,
        *,
        population: PopulationSimulator | None = None,
    ) -> WaveformChunks:
        """The polarizations, read from ``polarization.h5`` or generated by ``simulator``.

        ``population`` redraws the sources if they are not on disk.

        Raises
        ------
        FileNotFoundError
            If the polarizations or sources are missing and no simulator for
            them is given.
        ValueError
            If no waveform settings are recorded, or ``simulator`` is not for
            them.
        """
        metadata = self.waveform_metadata
        if metadata is None:
            raise ValueError(f"{self.directory} records no waveform settings")
        sources = self.population(population)
        path = self.directory / POLARIZATION_FILE
        chunks: WaveformChunks
        if path.is_file():
            chunks = WaveformArtifact.open(path, metadata=metadata, population=sources)
        elif simulator is None:
            raise FileNotFoundError(
                f"{path} is missing; pass a WaveformSimulator to generate it"
            )
        else:
            chunks = simulator(sources)
        if chunks.key() != self.manifest["waveforms"]["key"]:
            raise ValueError(
                f"{self.directory} records waveforms "
                f"{self.manifest['waveforms']['key']}, not {chunks.key()}"
            )
        return chunks

    def background(
        self,
        simulator: BackgroundSimulator | None = None,
        *,
        waveforms: WaveformSimulator | None = None,
        population: PopulationSimulator | None = None,
    ) -> BackgroundArtifact:
        """The pair spectra, read from ``background.h5`` or projected by ``simulator``.

        ``waveforms`` and ``population`` regenerate the upstream data that is
        not on disk.

        Raises
        ------
        FileNotFoundError
            If the spectra, or the upstream data they need, are missing and no
            simulator for them is given.
        ValueError
            If no background settings are recorded, or ``simulator`` is not
            for them.
        """
        table = self.manifest.get("background")
        if table is None:
            raise ValueError(f"{self.directory} records no background settings")
        path = self.directory / BACKGROUND_FILE
        if path.is_file():
            return BackgroundArtifact.open(
                path,
                table=table,
                realization=self.manifest["realization"],
                waveform_key=self.manifest["waveforms"]["key"],
            )
        if simulator is None:
            raise FileNotFoundError(
                f"{path} is missing; pass a BackgroundSimulator to project it"
            )
        spectra = simulator(self.waveforms(waveforms, population=population))
        if spectra.key() != table["key"]:
            raise ValueError(
                f"{self.directory} records background {table['key']}, "
                f"not {spectra.key()}"
            )
        return spectra


class RealizationSimulator:
    """Runs the configured stages of realization directories; see the module docstring.

    Each argument is a stage's settings, ``None`` to leave the stage as
    recorded. ``keep_sources`` writes ``source_parameters.h5`` and
    ``keep_polarizations`` writes ``polarization.h5``; a configured background
    is always written. ``chunk_size`` (waveforms per chunk) and
    ``source_chunk_size`` (sources drawn per piece) change cost, never a
    realization.

    The stage simulators are built when first needed and reused, so an
    ensemble compiles each stage once and a fully cached one builds none.
    """

    def __init__(
        self,
        population: PopulationMetadata | None = None,
        waveform: WaveformMetadata | None = None,
        background: DetectorConfiguration | None = None,
        *,
        keep_sources: bool = True,
        keep_polarizations: bool = False,
        chunk_size: int = 1024,
        source_chunk_size: int = 8192,
    ) -> None:
        import jax

        # x64 before the first key: seeds are 64-bit.
        jax.config.update("jax_enable_x64", True)
        if chunk_size <= 0 or source_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        self._population = population
        self._waveform = waveform
        self._background = background
        self._keep_sources = keep_sources
        self._keep_polarizations = keep_polarizations
        self._chunk_size = chunk_size
        self._source_chunk_size = source_chunk_size
        self._simulators: dict[tuple[object, str], Any] = {}

    def _simulator[S](self, cls: Callable[..., S], settings: Any, chunk_size: int) -> S:
        """The ``cls`` simulator of ``settings``, built on first use."""
        key = (cls, settings.model_dump_json())
        if key not in self._simulators:
            self._simulators[key] = cls(settings, chunk_size=chunk_size)
        return self._simulators[key]

    def population_simulator(self, metadata: PopulationMetadata) -> PopulationSimulator:
        """The cached population simulator of ``metadata``."""
        return self._simulator(PopulationSimulator, metadata, self._source_chunk_size)

    def waveform_simulator(self, metadata: WaveformMetadata) -> WaveformSimulator:
        """The cached waveform simulator of ``metadata``."""
        return self._simulator(WaveformSimulator, metadata, self._chunk_size)

    def background_simulator(
        self, detectors: DetectorConfiguration
    ) -> BackgroundSimulator:
        """The cached background simulator of ``detectors``."""
        return self._simulator(BackgroundSimulator, detectors, self._chunk_size)

    def __call__(
        self, directory: str | Path, *, seed: int | None = None, index: int = 0
    ) -> Realization:
        """Run the configured stages of the realization at ``directory``.

        ``seed`` and ``index`` key a new population,
        ``batch_keys(seed, index + 1)[index]``; for a recorded one they are
        optional, and checked if given.

        Raises
        ------
        FileNotFoundError
            If ``directory`` holds no realization and no population settings
            are given.
        ValueError
            If given settings conflict with the recorded ones, a stage needs
            upstream settings that are neither given nor recorded, or ``seed``
            is missing for a new population or given without one.
        """
        import jax

        directory = Path(directory)
        if (directory / MANIFEST_FILE).is_file():
            manifest = read_realization_manifest(directory)
        elif self._population is None:
            raise FileNotFoundError(
                f"no realization at {directory}; population settings start one"
            )
        else:
            manifest = {}
        if seed is not None and self._population is None:
            raise ValueError("a seed keys a population; give population settings")
        recorded = {
            stage: manifest.get(stage)
            for stage in ("population", "waveforms", "background")
        }
        tables: dict[str, Any] = {}

        # Resolve every stage before running any, so a conflict touches nothing.
        rng_key = None
        if self._population is not None:
            if seed is not None:
                rng_key = realization_key(seed, index)
                key_data = np.asarray(jax.random.key_data(rng_key)).tolist()
            elif recorded["population"] is not None:
                key_data = manifest["realization"]["rng_key"]
            else:
                raise ValueError(f"{directory} has no population; give a seed")
            population = population_key(self._population, key_data)
            if recorded["population"] is not None and (
                recorded["population"]["key"] != population
            ):
                raise ValueError(
                    f"{directory} holds population {recorded['population']['key']}, "
                    f"not {population}"
                )
        elif recorded["population"] is None:
            raise ValueError(f"{directory} records no population settings")
        else:
            population = recorded["population"]["key"]

        waveform = self._waveform
        if waveform is not None:
            table = waveform_table(waveform, population)
            if recorded["waveforms"] is None:
                tables["waveforms"] = table
            elif recorded["waveforms"]["key"] != table["key"]:
                raise ValueError(
                    f"{directory} holds waveforms {recorded['waveforms']['key']}, "
                    f"not {table['key']}"
                )
            waveforms = table["key"]
        elif recorded["waveforms"] is not None:
            waveforms = recorded["waveforms"]["key"]
        else:
            waveforms = None

        if self._background is not None:
            if waveforms is None:
                raise ValueError(
                    f"{directory} records no waveform settings; a background needs them"
                )
            table = background_table(waveforms, self._background)
            if recorded["background"] is None:
                tables["background"] = table
            elif recorded["background"]["key"] != table["key"]:
                raise ValueError(
                    f"{directory} holds background {recorded['background']['key']}, "
                    f"not {table['key']}"
                )

        sources_path = directory / SOURCES_FILE
        polarization_path = directory / POLARIZATION_FILE
        background_path = directory / BACKGROUND_FILE
        want_sources = (
            self._population is not None
            and self._keep_sources
            and not sources_path.is_file()
        )
        want_polarizations = (
            self._waveform is not None
            and self._keep_polarizations
            and not polarization_path.is_file()
        )
        want_background = self._background is not None and not background_path.is_file()

        drawn: PopulationArtifact | None = None
        if recorded["population"] is None:
            assert self._population is not None and rng_key is not None
            drawn = self.population_simulator(self._population)(rng_key)
            drawn_tables = drawn.tables()
            drawn_tables["realization"] |= {"seed": seed, "index": index}
            tables |= drawn_tables
        if want_background or want_polarizations or want_sources:
            tables["execution"] = {
                "chunk_size": self._chunk_size,
                "source_chunk_size": self._source_chunk_size,
            }
        if tables:
            write_realization_manifest(directory, {**manifest, **tables})
            manifest = read_realization_manifest(directory)
        # Tables first, products after: a product on disk is then always a
        # complete product of a recorded stage.
        realization = Realization(directory, manifest)
        if not (want_sources or want_polarizations or want_background):
            return realization

        if drawn is None:
            drawn = realization.population(
                None
                if sources_path.is_file()
                else self.population_simulator(realization.population_metadata)
            )
        if want_sources:
            drawn.write_sources(sources_path)
        if not (want_polarizations or want_background):
            return realization

        metadata = realization.waveform_metadata
        assert metadata is not None
        chunks: WaveformChunks
        if polarization_path.is_file():
            chunks = WaveformArtifact.open(
                polarization_path, metadata=metadata, population=drawn
            )
        else:
            chunks = self.waveform_simulator(metadata)(drawn)
            if want_polarizations:
                assert isinstance(chunks, Waveforms)
                chunks = chunks.write(polarization_path)
        if want_background:
            assert self._background is not None
            self.background_simulator(self._background)(chunks).write(background_path)
        return realization


def simulate_realization(
    directory: str | Path,
    *,
    population: PopulationMetadata | None = None,
    waveform: WaveformMetadata | None = None,
    background: DetectorConfiguration | None = None,
    seed: int | None = None,
    index: int = 0,
    keep_sources: bool = True,
    keep_polarizations: bool = False,
    chunk_size: int = 1024,
    source_chunk_size: int = 8192,
) -> Realization:
    """Run the configured stages of one realization; see :class:`RealizationSimulator`."""
    simulator = RealizationSimulator(
        population,
        waveform,
        background,
        keep_sources=keep_sources,
        keep_polarizations=keep_polarizations,
        chunk_size=chunk_size,
        source_chunk_size=source_chunk_size,
    )
    return simulator(directory, seed=seed, index=index)


def simulate_ensemble(
    simulator: RealizationSimulator,
    root: str | Path,
    seed: int,
    num_draws: int,
    *,
    start: int = 0,
) -> list[Realization]:
    """Realizations ``start .. num_draws - 1`` of ``seed``, at ``root/<seed>-<index>``.

    Realization ``i`` is keyed by ``batch_keys(seed, num_draws)[i]``, which
    does not depend on ``num_draws``: a smaller ensemble is a prefix of a
    larger one, an interrupted run resumes where it stopped, and ranges drawn
    on several machines merge by copying their directories.

    Raises
    ------
    ValueError
        If ``start`` is not in ``[0, num_draws)``, or a directory holds
        another realization (see :class:`RealizationSimulator`).
    """
    if not 0 <= start < num_draws:
        raise ValueError(f"need 0 <= start < num_draws, got {start}, {num_draws}")
    root = Path(root)
    return [
        simulator(root / f"{int(seed)}-{index}", seed=seed, index=index)
        for index in range(start, num_draws)
    ]
