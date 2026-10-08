"""Loading the two catalogs a run samples against, and the checks between them.

A catalog is a file: ``outputs/catalogs/<stem>.h5``, where ``<stem>`` names the
:class:`~astrogwb.simulators.polarization_power.CatalogMetadata` a run's role
resolves to and the seed it is drawn at. The workflow builds them with ``scripts/generate_catalog.py``;
:func:`run_catalog` reaches the same files from a notebook, generating one on
a miss. Either way the density its samples follow comes back off the file's
own population record rather than being reassembled from the run config.

What used to live here has mostly moved to where it belongs:
:func:`~astrogwb.simulators.polarization_power.restrict_redshift`
narrows the samples and the recorded population together, and the proposal density is evaluated by
:func:`~astrogwb.importance.spectral.build_importance_spectrum` directly from
the catalog's own source model. Fiducial GW propagation is gone
entirely: which propagation law applies is now part of the population
declaration, so a catalog's power and its recorded distances are consistent by
construction instead of being patched up at the call site.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import CATALOGS_ROOT, assemble_run
from astrogwb.simulators.core import batch_keys, load, write
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    PolarizationPowerData,
    catalog_stem,
    draw_catalog,
)


def load_run_catalog(
    path: Path | str,
    *,
    label: str,
    request: tuple[CatalogMetadata, np.uint64] | None = None,
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """Load one catalog file, validating its population record.

    ``label`` is the role -- ``"injection"`` or ``"proposal"`` -- and is what
    identifies the catalog in error messages. Given a ``request``, the file
    must also record exactly that metadata and seed, which is how a run refuses
    a file handed to the wrong role or built from a draw it no longer asks for.

    This is the by-path counterpart of :func:`ensure_catalog`, for a caller
    handed a file rather than a request -- the figure scripts and the SNR
    helper -- so a missing file raises instead of being drawn.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} catalog not found: {path}")
    try:
        data, recorded, attrs = load(path, CatalogMetadata)
        if request is not None:
            metadata, seed = request
            if recorded.key() != metadata.key() or attrs.get("seed") != seed:
                raise ValueError(
                    f"records {recorded.key()} at seed {attrs.get('seed')}, "
                    f"not the requested {metadata.key()} at seed {seed}:\n"
                    f"  recorded:  {recorded.model_dump_json()}\n"
                    f"  requested: {metadata.model_dump_json()}"
                )
    except ValueError as error:
        raise ValueError(f"{label} catalog {path}: {error}") from error
    return cast(PolarizationPowerData, data), recorded


def catalog_path(
    metadata: CatalogMetadata, seed: int | np.integer, directory: Path | str
) -> Path:
    """Where the catalog of ``metadata`` at ``seed`` lives in ``directory``."""
    return Path(directory) / f"{catalog_stem(metadata, int(seed))}.h5"


def generate_catalog(
    metadata: CatalogMetadata, seed: int | np.integer, path: Path | str
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """Draw the catalog of ``metadata`` at ``seed`` and write it to ``path``.

    The seed becomes the draw's key through :func:`~astrogwb.simulators.core.batch_keys`
    (key 0 of the seed's batch). Generation reaches JAX.
    """
    data = draw_catalog(metadata, batch_keys(seed, 1)[0])
    write(path, data, metadata, seed=int(seed))
    return data, metadata


def ensure_catalog(
    metadata: CatalogMetadata,
    seed: int | np.integer,
    directory: Path | str,
    *,
    generate: bool = True,
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """The catalog of ``metadata`` at ``seed`` in ``directory``, drawn on a miss.

    A hit is checked against the request. ``generate=False`` makes a miss raise
    ``FileNotFoundError`` instead, which is how a workflow job refuses to draw.
    """
    path = catalog_path(metadata, seed, directory)
    if path.is_file():
        return load_run_catalog(
            path, label="cached", request=(metadata, np.uint64(seed))
        )
    if not generate:
        raise FileNotFoundError(f"no catalog at {path}, and generation is disabled")
    return generate_catalog(metadata, seed, path)


def run_catalog(
    experiment: str,
    run: str,
    role: str,
    *,
    cache_dir: Path | str = CATALOGS_ROOT,
    root: Path | None = None,
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """The catalog one role of a committed run samples against.

    Resolves the run's config into its catalog metadata and returns the cached file
    the workflow would have built, generating it on a miss. This is the
    notebook's way in: no path to hard-code, and no way to pair a run with a
    catalog it does not ask for. Generation reaches JAX, so call this after
    :func:`~astrogwb.paper.runtime.configure_runtime`.
    """
    config = build_run_config(assemble_run(experiment, run, root=root))
    metadata, seed = config.catalog_request(role)
    return ensure_catalog(metadata, seed, cache_dir)


def validate_matching_frequency_grids(
    injection_frequencies: ArrayLike,
    proposal_frequencies: ArrayLike,
    *,
    label: str = "injection and proposal",
) -> None:
    """Require two waveform catalogs to share the exact frequency grid."""
    injection_frequencies = np.asarray(injection_frequencies)
    proposal_frequencies = np.asarray(proposal_frequencies)
    if not np.array_equal(injection_frequencies, proposal_frequencies):
        raise ValueError(f"{label} catalogs must have identical frequency grids")


__all__ = [
    "catalog_path",
    "ensure_catalog",
    "generate_catalog",
    "load_run_catalog",
    "run_catalog",
    "validate_matching_frequency_grids",
]
