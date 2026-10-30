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
recorded population metadata and ``rng_key``. Nothing upstream is redrawn
differently, as the population stage is the only one that draws random numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from astrogwb.detector.config import DetectorRegistry
from astrogwb.simulators._io import (
    BACKGROUND_FILE,
    MANIFEST_FILE,
    SOURCES_FILE,
    read_realization_manifest,
    realization_key,
    write_realization_manifest,
)
from astrogwb.simulators.background import (
    BackgroundArtifact,
    BackgroundSimulator,
    background_pairs,
    background_table,
    validate_background_schema,
    waveform_table,
)
from astrogwb.simulators.population import (
    PopulationArtifact,
    PopulationMetadata,
    PopulationSimulator,
    population_key,
    validate_population_schema,
)
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = [
    "NetworkSpectra",
    "Realization",
    "RealizationSimulator",
    "network_spectra_ensemble",
    "simulate_realization",
]


@dataclass(frozen=True)
class Realization:
    """A realization directory and its manifest.

    The accessors :meth:`population` and :meth:`background` read a stage's
    product when it is on disk; given the stage simulators, they regenerate
    what is missing instead.
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
    def networks(self) -> tuple[str, ...] | None:
        """The recorded background networks, by name; ``None`` if none are."""
        table = self.manifest.get("background")
        return None if table is None else tuple(background_pairs(table))

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
        validate_population_schema(self.manifest["population"])
        path = self.directory / SOURCES_FILE
        if path.is_file():
            return PopulationArtifact.from_tables(self.manifest, path)
        if simulator is None:
            raise FileNotFoundError(
                f"{path} is missing; pass a PopulationSimulator to redraw it"
            )
        drawn = simulator(
            jax.random.wrap_key_data(jnp.asarray(self.rng_key, dtype=jnp.uint32))
        )
        if drawn.key() != self.id or drawn.count != self.count:
            raise ValueError(
                f"redrawing {self.directory} gave population {drawn.key()} of "
                f"{drawn.count} sources, not the recorded {self.id} of {self.count}"
            )
        return drawn

    def background(
        self,
        *,
        simulator: BackgroundSimulator | None = None,
        population: PopulationSimulator | None = None,
    ) -> BackgroundArtifact:
        """The network spectra, read from ``background.h5`` or projected by ``simulator``.

        ``population`` redraws the sources if they are not on disk.

        Raises
        ------
        FileNotFoundError
            If the spectra, or the sources they need, are missing and no
            simulator for them is given.
        ValueError
            If no background settings are recorded, or ``simulator`` is not
            for them.
        """
        table = self.manifest.get("background")
        if table is None:
            raise ValueError(f"{self.directory} records no background settings")
        validate_background_schema(table)
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
        spectra = simulator(self.population(population))
        if spectra.key() != table["key"]:
            raise ValueError(
                f"{self.directory} records background {table['key']}, "
                f"not {spectra.key()}"
            )
        return spectra


class RealizationSimulator:
    """Runs the configured stages of realization directories; see the module docstring.

    Each argument is a stage's settings, ``None`` to leave the stage as
    recorded. ``keep_sources`` writes ``source_parameters.h5``; a configured
    background is always written. ``chunk_size`` (waveforms per chunk) and
    ``source_chunk_size`` (sources drawn per piece) change cost, never a
    realization.

    The stage simulators are built for each realization that needs them.
    """

    def __init__(
        self,
        population: PopulationMetadata | None = None,
        waveform: WaveformMetadata | None = None,
        background: DetectorRegistry | None = None,
        *,
        keep_sources: bool = True,
        chunk_size: int = 1024,
        source_chunk_size: int = 8192,
    ) -> None:
        # x64 before the first key: seeds are 64-bit.
        jax.config.update("jax_enable_x64", True)
        if chunk_size <= 0 or source_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        self._population = population
        self._waveform = waveform
        self._background = background
        self._keep_sources = keep_sources
        self._chunk_size = chunk_size
        self._source_chunk_size = source_chunk_size

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

        if recorded["population"] is not None:
            validate_population_schema(recorded["population"])
        if recorded["background"] is not None:
            validate_background_schema(recorded["background"])

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
        background_path = directory / BACKGROUND_FILE
        want_sources = (
            self._population is not None
            and self._keep_sources
            and not sources_path.is_file()
        )
        want_background = self._background is not None and not background_path.is_file()

        drawn: PopulationArtifact | None = None
        if recorded["population"] is None:
            assert self._population is not None and rng_key is not None
            drawn = PopulationSimulator(
                self._population, chunk_size=self._source_chunk_size
            )(rng_key)
            drawn_tables = drawn.tables()
            drawn_tables["realization"] |= {"seed": seed, "index": index}
            tables |= drawn_tables
        if want_background or want_sources:
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
        if not (want_sources or want_background):
            return realization

        if drawn is None:
            drawn = realization.population(
                None
                if sources_path.is_file()
                else PopulationSimulator(
                    realization.population_metadata,
                    chunk_size=self._source_chunk_size,
                )
            )
        if want_sources:
            drawn.write_sources(sources_path)
        if want_background:
            metadata = realization.waveform_metadata
            assert metadata is not None and self._background is not None
            BackgroundSimulator(
                metadata, self._background, chunk_size=self._chunk_size
            )(drawn).write(background_path)
        return realization


def simulate_realization(
    directory: str | Path,
    *,
    population: PopulationMetadata | None = None,
    waveform: WaveformMetadata | None = None,
    background: DetectorRegistry | None = None,
    seed: int | None = None,
    index: int = 0,
    keep_sources: bool = True,
    chunk_size: int = 1024,
    source_chunk_size: int = 8192,
) -> Realization:
    """Run the configured stages of one realization; see :class:`RealizationSimulator`."""
    simulator = RealizationSimulator(
        population,
        waveform,
        background,
        keep_sources=keep_sources,
        chunk_size=chunk_size,
        source_chunk_size=source_chunk_size,
    )
    return simulator(directory, seed=seed, index=index)


class NetworkSpectra(TypedDict):
    """One network's draws, stacked along the leading draw axis.

    ``frequencies`` ``(F,)``; ``spectral_density`` ``(D, F)``, the
    :func:`~astrogwb.simulators.combine_pair_spectra` of
    ``pair_spectral_density`` ``(D, P, F)``; ``n_events`` and
    ``total_merger_rate`` ``(D,)``; each ``hyperparameters`` value ``(D,)``.
    ``n_events`` is each draw's ``count``, named as the Poisson-count model's
    site is.
    """

    frequencies: NDArray[np.float64]
    spectral_density: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    n_events: NDArray[np.int64]
    total_merger_rate: NDArray[np.float64]
    hyperparameters: dict[str, NDArray[np.float64]]


def network_spectra_ensemble(
    root: str | Path,
    seed: int,
    num_draws: int,
    names: Sequence[str],
    *,
    start: int = 0,
) -> list[NetworkSpectra]:
    """Draws ``start .. num_draws - 1`` of the networks ``names``, stacked per network.

    Draw ``i`` is the background of the realization at ``<root>/<seed>-<i>``,
    as ``workflow/coverage_draws.smk`` lays them out, draw ``i`` keyed by
    ``batch_keys(seed, i + 1)[i]``.

    Parameters
    ----------
    root
        The ensemble root.
    seed
        The seed the ensemble was simulated with.
    num_draws
        One past the last index read.
    names
        Networks, each recorded in every draw's background.
    start
        The first index read; must be below ``num_draws``. Every draw is
        taken to share one frequency grid and one set of hyperparameters.

    Returns
    -------
    list of NetworkSpectra
        Each network's draws, in ``names`` order.

    Raises
    ------
    FileNotFoundError
        If a draw, or its background, is missing.
    KeyError
        If a draw's background does not record one of ``names``.
    """
    root = Path(root)
    draws = [
        Realization.open(root / f"{int(seed)}-{index}").background()
        for index in range(start, num_draws)
    ]
    return [_stack(draws, name) for name in names]


def _stack(draws: Sequence[BackgroundArtifact], name: str) -> NetworkSpectra:
    """One network's spectra of every draw, stacked."""
    spectra = [draw.spectra[name] for draw in draws]
    return {
        "frequencies": draws[0].frequencies,
        "spectral_density": np.stack([s.spectral_density for s in spectra]),
        "pair_spectral_density": np.stack([s.pair_spectral_density for s in spectra]),
        "n_events": np.asarray([draw.count for draw in draws], dtype=np.int64),
        "total_merger_rate": np.asarray(
            [draw.total_merger_rate for draw in draws], dtype=np.float64
        ),
        "hyperparameters": {
            key: np.asarray([draw.hyperparameters[key] for draw in draws])
            for key in draws[0].hyperparameters
        },
    }
