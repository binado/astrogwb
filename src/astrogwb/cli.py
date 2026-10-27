"""The astrogwb command line writes realization directories stage by stage.

astrogwb simulate DIRECTORY reads population and waveform metadata, then
runs every configured stage. Repeat --detector-config PATH to merge detector
registry layers in argument order; --background runs the background stage with
that registry, and the two go together. Each run adds the stages it is given to
the directory, with the settings the directory already records: a later run may
give only --waveform-metadata or --background, and a stage with no settings
given is left as recorded. --seed goes with --population-metadata, and the
sources are written unless --no-keep-sources, and the polarizations only
with --keep-polarizations. For example::

    astrogwb simulate ensemble/41-0 --population-metadata population.toml \\
        --seed 41 --index 0 --waveform-metadata waveform.toml \\
        --background --detector-config config/detectors.toml

The registry contains [detectors.<name>] definitions and [networks]
memberships. Background manifests record each network's detector pairs. Noise
generation also requires the registry::

    astrogwb noise ensemble/41-0 ensemble/detector_noise.h5 \\
        --detector-config config/detectors.toml

Readers trust the files they open and preserve the recorded artifact identity.
Changing the registry requires regenerating affected outputs. Runtime settings (--platform,
--cpu-threads) are applied before the JAX backend initializes.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer

__all__ = ["app"]

app = typer.Typer(help="Bayesian inference for stochastic GW backgrounds.")

DETECTOR_CONFIG_OPTION = typer.Option(
    "--detector-config",
    exists=True,
    dir_okay=False,
    help="Detector registry TOML layer; repeat to merge.",
)


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
    background: Annotated[
        bool, typer.Option(help="Run the background stage (with --detector-config).")
    ] = False,
    detector_config: Annotated[list[Path] | None, DETECTOR_CONFIG_OPTION] = None,
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
    if background != bool(detector_config):
        raise typer.BadParameter(
            "--background and --detector-config go together",
            param_hint="--background",
        )
    from astrogwb.detector.config import load_detector_config
    from astrogwb.simulators import PopulationMetadata, simulate_realization
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
        background=load_detector_config(detector_config) if detector_config else None,
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
    detector_config: Annotated[list[Path], DETECTOR_CONFIG_OPTION],
    platform: Annotated[
        str, typer.Option(help="JAX platform: auto, cpu, cuda or tpu.")
    ] = "auto",
) -> None:
    """Write the detector noise of a realization's background networks.

    The networks are those recorded in the realization's background, and the
    registry is trusted to be the one it was simulated with.
    """
    _apply_runtime(platform, None)
    from astrogwb.detector.config import load_detector_config
    from astrogwb.simulators import Realization, detector_noise

    registry = load_detector_config(detector_config)
    recorded = Realization.open(realization)
    background = recorded.background()
    detector_noise(
        background.frequencies,
        registry,
        recorded.population_metadata.observation_time,
        network_names=tuple(background.spectra),
    ).write(output)
    typer.echo(f"noise of {', '.join(background.spectra)} -> {output}")
