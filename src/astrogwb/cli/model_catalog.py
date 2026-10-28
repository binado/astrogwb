"""Generate an importance-integration reference catalog."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from ._common import _apply_runtime


def model_catalog(
    output: Annotated[Path, typer.Argument(dir_okay=False, help="HDF5 file to write.")],
    catalog_metadata: Annotated[
        list[Path],
        typer.Option(
            exists=True,
            dir_okay=False,
            help="CatalogMetadata TOML layer; repeat to merge in argument order.",
        ),
    ],
    seed: Annotated[int, typer.Option(help="Seed of the catalog draw.")] = 42,
    chunk_size: Annotated[
        int | None, typer.Option(min=1, help="Waveforms per batch; cost only.")
    ] = None,
    platform: Annotated[
        str, typer.Option(help="JAX platform: auto, cpu, cuda or tpu.")
    ] = "auto",
    cpu_threads: Annotated[
        int | None, typer.Option(min=1, help="Cap OMP/BLAS/XLA CPU threads.")
    ] = None,
) -> None:
    """Write fixed source draws and waveform powers for importance weighting."""
    _apply_runtime(platform, cpu_threads)
    from astrogwb.config import merge_layers
    from astrogwb.gwb.catalog import CatalogMetadata
    from astrogwb.gwb.importance import reference_catalog as draw_reference_catalog
    from astrogwb.simulators import batch_keys
    from astrogwb.utils.io import write

    metadata = CatalogMetadata.model_validate(merge_layers(catalog_metadata))
    began = time.perf_counter()
    data = draw_reference_catalog(
        metadata, batch_keys(seed, 1)[0], chunk_size=chunk_size
    )
    write(output, data, metadata, seed=seed)
    elapsed = time.perf_counter() - began
    typer.echo(
        f"reference catalog {metadata.key()} at seed {seed}: "
        f"{metadata.num_samples} draws -> {output} ({elapsed:.0f} s)"
    )
