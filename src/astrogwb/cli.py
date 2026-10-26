"""The ``astrogwb`` command line: simulator stages written to disk.

Each ``astrogwb simulate`` command reads TOML stage metadata, calls the
:mod:`astrogwb.simulators` API and writes an artifact directory
(``metadata.toml`` and ``data.h5``). A downstream command takes either an
upstream artifact (``--population-file``, ``--waveform-file``) or upstream
metadata (``--population-metadata`` with ``--seed``, ``--waveform-metadata``),
which it computes in memory; the fused ``background`` run writes only the
background::

    astrogwb simulate population population.toml --seed 41 --output population/
    astrogwb simulate waveforms --population-file population/metadata.toml \\
        --waveform-metadata waveform.toml --output waveforms/
    astrogwb simulate background --waveform-file waveforms/metadata.toml \\
        --detectors network.toml --output background/
    astrogwb simulate background --population-metadata population.toml \\
        --seed 41 --waveform-metadata waveform.toml --detectors network.toml \\
        --output background/

The metadata files hold the fields of
:class:`~astrogwb.simulators.PopulationMetadata`,
:class:`~astrogwb.waveform.WaveformMetadata` and
:class:`~astrogwb.simulators.DetectorConfiguration` at their top level.

Runtime settings (``--platform``, ``--cpu-threads``) are applied before the JAX
backend initializes, so this module imports nothing that touches JAX.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from astrogwb.simulators import PopulationArtifact, Waveforms
    from astrogwb.simulators.waveforms import WaveformChunks

__all__ = ["app"]

app = typer.Typer(help="Bayesian inference for stochastic GW backgrounds.")
simulate = typer.Typer(
    help="Simulate one Poisson realization, stage by stage or fused.",
    no_args_is_help=True,
)
app.add_typer(simulate, name="simulate")


@app.callback()
def _main() -> None:
    """Bayesian inference for compact-binary stochastic GW backgrounds."""


@simulate.callback()
def _runtime(
    platform: Annotated[
        str, typer.Option(help="JAX platform: auto, cpu, cuda or tpu.")
    ] = "auto",
    cpu_threads: Annotated[
        int | None, typer.Option(min=1, help="Cap OMP/BLAS/XLA CPU threads.")
    ] = None,
) -> None:
    """Apply the runtime settings; runs before any command touches JAX."""
    if platform not in {"auto", "cpu", "cuda", "tpu"}:
        raise typer.BadParameter(
            f"unknown platform {platform!r}", param_hint="--platform"
        )
    if cpu_threads is not None:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(cpu_threads)
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={cpu_threads}"
        )
    if platform != "auto":
        os.environ["JAX_PLATFORMS"] = platform
    import jax

    jax.config.update("jax_enable_x64", True)


Output = Annotated[
    Path, typer.Option(file_okay=False, help="Artifact directory to write.")
]
Seed = Annotated[
    int | None, typer.Option(help="Seed of the realization (with metadata input).")
]
Index = Annotated[
    int, typer.Option(min=0, help="Realization index i of batch_keys(seed, i + 1).")
]
ChunkSize = Annotated[int, typer.Option(min=1, help="Sources per chunk.")]
PopulationFile = Annotated[
    Path | None,
    typer.Option(exists=True, help="A population artifact's metadata.toml."),
]
PopulationMetadataFile = Annotated[
    Path | None,
    typer.Option(exists=True, dir_okay=False, help="Population metadata TOML."),
]
WaveformMetadataFile = Annotated[
    Path | None,
    typer.Option(exists=True, dir_okay=False, help="Waveform metadata TOML."),
]


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _population(
    population_file: Path | None,
    population_metadata: Path | None,
    seed: int | None,
    index: int,
    chunk_size: int,
) -> PopulationArtifact:
    """The upstream population: opened from an artifact, or drawn from metadata."""
    from astrogwb.simulators import (
        PopulationArtifact,
        PopulationMetadata,
        PopulationSimulator,
        batch_keys,
    )

    if (population_file is None) == (population_metadata is None):
        raise typer.BadParameter(
            "give exactly one of --population-file and --population-metadata"
        )
    if population_file is not None:
        if seed is not None:
            raise typer.BadParameter("--seed applies to --population-metadata only")
        return PopulationArtifact.open(population_file)
    if seed is None:
        raise typer.BadParameter("--population-metadata needs --seed")
    assert population_metadata is not None
    metadata = PopulationMetadata.model_validate(_read_toml(population_metadata))
    key = batch_keys(seed, index + 1)[index]
    return PopulationSimulator(metadata, chunk_size=chunk_size)(key)


def _waveforms(
    waveform_metadata: Path, population: PopulationArtifact, chunk_size: int
) -> Waveforms:
    from astrogwb.simulators import WaveformSimulator
    from astrogwb.waveform import WaveformMetadata

    metadata = WaveformMetadata.model_validate(_read_toml(waveform_metadata))
    return WaveformSimulator(metadata, chunk_size=chunk_size)(population)


@simulate.command("population")
def simulate_population(
    metadata: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Population metadata TOML."),
    ],
    seed: Annotated[int, typer.Option(help="Seed of the realization.")],
    output: Output,
    index: Index = 0,
    chunk_size: ChunkSize = 8192,
) -> None:
    """Draw one population realization."""
    population = _population(None, metadata, seed, index, chunk_size)
    written = population.write(output)
    typer.echo(f"population: {written.count} sources -> {written.manifest_path}")


@simulate.command("waveforms")
def simulate_waveforms(
    waveform_metadata: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Waveform metadata TOML.")
    ],
    output: Output,
    population_file: PopulationFile = None,
    population_metadata: PopulationMetadataFile = None,
    seed: Seed = None,
    index: Index = 0,
    chunk_size: ChunkSize = 1024,
) -> None:
    """Generate the polarizations of one population."""
    population = _population(
        population_file, population_metadata, seed, index, chunk_size
    )
    written = _waveforms(waveform_metadata, population, chunk_size).write(output)
    typer.echo(f"waveforms: {population.count} sources -> {written.manifest_path}")


@simulate.command("background")
def simulate_background(
    detectors: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Detector configuration TOML."),
    ],
    output: Output,
    waveform_file: Annotated[
        Path | None,
        typer.Option(exists=True, help="A waveform artifact's metadata.toml."),
    ] = None,
    waveform_metadata: WaveformMetadataFile = None,
    population_file: PopulationFile = None,
    population_metadata: PopulationMetadataFile = None,
    seed: Seed = None,
    index: Index = 0,
    chunk_size: ChunkSize = 1024,
) -> None:
    """Project polarizations onto a network and write the pair spectra."""
    from astrogwb.simulators import (
        BackgroundSimulator,
        DetectorConfiguration,
        WaveformArtifact,
    )

    if (waveform_file is None) == (waveform_metadata is None):
        raise typer.BadParameter(
            "give exactly one of --waveform-file and --waveform-metadata"
        )
    waveforms: WaveformChunks
    if waveform_file is not None:
        if population_file or population_metadata or seed is not None:
            raise typer.BadParameter(
                "--waveform-file carries its population; drop the population options"
            )
        waveforms = WaveformArtifact.open(waveform_file)
    else:
        assert waveform_metadata is not None
        population = _population(
            population_file, population_metadata, seed, index, chunk_size
        )
        waveforms = _waveforms(waveform_metadata, population, chunk_size)
    network = DetectorConfiguration.model_validate(_read_toml(detectors))
    background = BackgroundSimulator(network, chunk_size=chunk_size)(waveforms)
    written = background.write(output)
    typer.echo(
        f"background: {len(written.pairs)} pairs, {written.count} sources "
        f"-> {written.manifest_path}"
    )
