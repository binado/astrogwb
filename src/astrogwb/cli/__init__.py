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

Importance-integration catalogs use explicit catalog metadata::

    astrogwb model-catalog reference.h5 --catalog-metadata catalog.toml

Readers trust the files they open and preserve the recorded artifact identity.
Changing the registry requires regenerating affected outputs. Runtime settings (--platform,
--cpu-threads) are applied before the JAX backend initializes.
"""

import typer

from .model_catalog import model_catalog
from .noise import noise
from .simulate import simulate

__all__ = ["app"]

app = typer.Typer(help="Bayesian inference for stochastic GW backgrounds.")


@app.callback()
def _main() -> None:
    """Bayesian inference for compact-binary stochastic GW backgrounds."""


app.command()(simulate)
app.command()(noise)

app.command()(model_catalog)
