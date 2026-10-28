"""Run realization simulation stages."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Any

import typer

from ._common import DETECTOR_CONFIG_OPTION, _apply_runtime


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


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
        chunk_size=chunk_size,
        source_chunk_size=source_chunk_size,
    )
    products = sorted(path.name for path in directory.glob("*.h5"))
    typer.echo(
        f"realization {realization.id}: {realization.count} sources, "
        f"{', '.join(products) or 'no products'} -> {directory}"
    )
