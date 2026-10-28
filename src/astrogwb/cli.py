"""The ``astrogwb`` command line: realization directories written stage by stage.

``astrogwb simulate DIRECTORY`` reads TOML stage metadata and runs every stage
whose metadata is given into the realization directory ``DIRECTORY`` (see
:class:`~astrogwb.simulators.RealizationSimulator`). A directory is started
with ``--population-metadata`` and ``--seed``; later runs add stages to it,
using the settings it records for the stages they leave out::

    astrogwb simulate ensemble/41-0 --population-metadata population.toml \\
        --seed 41 --index 0 --waveform-metadata waveform.toml \\
        --background-metadata networks.toml
    astrogwb simulate realization/ --population-metadata population.toml --seed 41
    astrogwb simulate realization/ --waveform-metadata waveform.toml \\
        --background-metadata networks.toml

The sources are written unless ``--no-keep-sources``, the polarizations only
with ``--keep-polarizations``, and a configured background always. The
population and waveform files hold the fields of
:class:`~astrogwb.simulators.PopulationMetadata` and
:class:`~astrogwb.waveform.WaveformMetadata` at their top level; the
background file holds one ``[networks.<name>]`` table per network, each the
fields of :class:`~astrogwb.simulators.DetectorConfiguration`::

    [networks.ET-2L]
    detectors = ["S1", "R1"]
    psd_references = { S1 = "ET_COBA_15km_psd.txt", R1 = "ET_COBA_15km_psd.txt" }

``astrogwb noise REALIZATION OUTPUT`` writes the
:class:`~astrogwb.simulators.DetectorNoise` of the networks, frequencies and
observation time a realization's background was simulated with, once per
ensemble::

    astrogwb noise ensemble/41-0 ensemble/detector_noise.h5

Runtime settings (``--platform``, ``--cpu-threads``) are applied before the JAX
backend initializes, so this module imports nothing that touches JAX.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer

__all__ = ["app"]

app = typer.Typer(help="Bayesian inference for stochastic GW backgrounds.")


@app.callback()
def _main() -> None:
    """Bayesian inference for compact-binary stochastic GW backgrounds."""


def _apply_runtime(platform: str, cpu_threads: int | None) -> None:
    """Apply the runtime settings; runs before anything touches JAX."""
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


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


@app.command()
def simulate(
    directory: Annotated[
        Path, typer.Argument(file_okay=False, help="Realization directory.")
    ],
    population_metadata: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Population metadata TOML."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(help="Seed of the realization (with --population-metadata)."),
    ] = None,
    index: Annotated[
        int, typer.Option(min=0, help="Realization index i of batch_keys(seed, i + 1).")
    ] = 0,
    waveform_metadata: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Waveform metadata TOML."),
    ] = None,
    background_metadata: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Network configuration TOML."),
    ] = None,
    keep_sources: Annotated[
        bool, typer.Option(help="Write source_parameters.h5.")
    ] = True,
    keep_polarizations: Annotated[
        bool, typer.Option(help="Write polarization.h5.")
    ] = False,
    chunk_size: Annotated[int, typer.Option(min=1, help="Waveforms per chunk.")] = 1024,
    source_chunk_size: Annotated[
        int, typer.Option(min=1, help="Sources drawn per piece.")
    ] = 8192,
    platform: Annotated[
        str, typer.Option(help="JAX platform: auto, cpu, cuda or tpu.")
    ] = "auto",
    cpu_threads: Annotated[
        int | None, typer.Option(min=1, help="Cap OMP/BLAS/XLA CPU threads.")
    ] = None,
) -> None:
    """Run every stage whose metadata is given into one realization directory."""
    _apply_runtime(platform, cpu_threads)
    if (population_metadata is None) != (seed is None):
        raise typer.BadParameter(
            "--seed and --population-metadata go together", param_hint="--seed"
        )
    from astrogwb.simulators import (
        DetectorConfiguration,
        PopulationMetadata,
        simulate_realization,
    )
    from astrogwb.waveform import WaveformMetadata

    realization = simulate_realization(
        directory,
        population=(
            None
            if population_metadata is None
            else PopulationMetadata.model_validate(_read_toml(population_metadata))
        ),
        waveform=(
            None
            if waveform_metadata is None
            else WaveformMetadata.model_validate(_read_toml(waveform_metadata))
        ),
        background=(
            None
            if background_metadata is None
            else {
                name: DetectorConfiguration.model_validate(network)
                for name, network in _read_toml(background_metadata)["networks"].items()
            }
        ),
        seed=seed,
        index=index,
        keep_sources=keep_sources,
        keep_polarizations=keep_polarizations,
        chunk_size=chunk_size,
        source_chunk_size=source_chunk_size,
    )
    products = sorted(path.name for path in directory.glob("*.h5"))
    typer.echo(
        f"realization {realization.id}: {realization.count} sources, "
        f"{', '.join(products) or 'no products'} -> {directory}"
    )


@app.command()
def noise(
    realization: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=False, help="A realization with a background."
        ),
    ],
    output: Annotated[Path, typer.Argument(dir_okay=False, help="HDF5 file to write.")],
    platform: Annotated[
        str, typer.Option(help="JAX platform: auto, cpu, cuda or tpu.")
    ] = "auto",
) -> None:
    """Write the detector noise of a realization's background networks."""
    _apply_runtime(platform, None)
    from astrogwb.simulators import Realization, detector_noise

    recorded = Realization.open(realization)
    background = recorded.background()
    detector_noise(
        background.frequencies,
        background.networks,
        recorded.population_metadata.observation_time,
    ).write(output)
    typer.echo(f"noise of {', '.join(background.networks)} -> {output}")
