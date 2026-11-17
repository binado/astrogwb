"""Build reference intensity catalogs from a scientific configuration."""

import typer

from .model_catalog import model_catalog

__all__ = ["app"]

app = typer.Typer(help="Bayesian inference for stochastic GW backgrounds.")


@app.callback()
def _main() -> None:
    """Bayesian inference for compact-binary stochastic GW backgrounds."""


app.command()(model_catalog)
