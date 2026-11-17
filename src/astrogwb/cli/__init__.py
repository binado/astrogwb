"""The astrogwb command line.

``astrogwb simulate OUTPUT.h5 POPULATION`` simulates the background of a
physical event pool, a gwmock-pop population catalogue (HDF5 or CSV), into one
HDF5 file. ``--waveform-config`` names a TOML file of waveform settings and
``--detector-config`` a detector registry layer, repeated to merge in argument
order. For example::

    astrogwb simulate ensemble/41-0.h5 events.h5 \\
        --waveform-config waveform.toml --detector-config detectors.toml \\
        --rate-per-year 1e5 --num-segments 16 --segment-seconds 2048 --seed 41

The registry contains [detectors.<name>] definitions and [networks]
memberships. The file carries the configuration as JSON in its ``metadata``
attribute, with the astrogwb ``version`` that wrote it. ``--incoherent`` sums
the events' powers without detector noise.

Importance-integration catalogs read a [catalogs.<name>] table of the shared
config (--config-dir, default ``config``)::

    astrogwb model-catalog reference.h5 --catalog reference-fiducial

A reader refuses a file written by another astrogwb version. Runtime settings
(--platform, --cpu-threads) are applied before the JAX backend initializes.
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
