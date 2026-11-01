"""The astrogwb command line simulates one background into one HDF5 file.

astrogwb simulate OUTPUT.h5 reads population and waveform metadata and a
detector registry, draws the population at (--seed, --index) and writes the
background of its networks, in one pass. Repeat --detector-config PATH to merge
detector registry layers in argument order. For example::

    astrogwb simulate ensemble/41-0.h5 --population-metadata population.toml \\
        --seed 41 --index 0 --waveform-metadata waveform.toml \\
        --detector-config config/detectors.toml

The registry contains [detectors.<name>] definitions and [networks]
memberships. The file carries its configuration as JSON in its ``metadata``
attribute, with the astrogwb ``version`` that wrote it. --segment-duration
SECONDS assigns sources to segments of that length and adds fresh per-segment
detector noise.

Importance-integration catalogs use explicit catalog metadata::

    astrogwb model-catalog reference.h5 --catalog-metadata catalog.toml

A reader refuses a file written by another astrogwb version; regenerate it.
Runtime settings (--platform,
--cpu-threads) are applied before the JAX backend initializes.
"""

import typer

from .model_catalog import model_catalog
from .simulate import simulate

__all__ = ["app"]

app = typer.Typer(help="Bayesian inference for stochastic GW backgrounds.")


@app.callback()
def _main() -> None:
    """Bayesian inference for compact-binary stochastic GW backgrounds."""


app.command()(simulate)

app.command()(model_catalog)
