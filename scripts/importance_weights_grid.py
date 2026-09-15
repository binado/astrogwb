r"""Importance-weight relative-ESS grids for two-parameter combinations.

For each hard-coded pair this script builds a prior-support grid via the
inverse CDF of each parameter's prior, evaluates the BNS Madau–Dickinson
modified-propagation population's importance log-weights, and plots a
heatmap of
$N_{\mathrm{eff}}/N = (\\sum_i w_i)^2 / (N\\sum_i w_i^2)$.

Combinations:

1. $H_0$ + $\\Omega_m$ — $H_0\\sim\\mathrm{Uniform}(20, 140)$;
   $\\Omega_m\\sim\\mathrm{Uniform}(0.05, 0.95)$;
2. $\\Xi_0$ + $n$ — $\\Xi_0\\sim\\mathrm{Uniform}(0.5, 5)$;
   $n\\sim\\mathrm{Uniform}(0.3, 3)$.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from matplotlib.axes import Axes as MplAxes
from matplotlib.figure import Figure
from matplotlib.projections import register_projection

from astrogwb.importance.spectral import build_importance_spectrum
from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config import priors
from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import add_config_arguments, load_merged_config
from astrogwb.paper.plotting import TRUTH, parameter_label, use_paper_style
from astrogwb.populations import build_population

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes.
# Restore the standard projection for consistent plotting.
register_projection(MplAxes)
jax.config.update("jax_enable_x64", True)

EPS = 1e-3
NPOINTS = 64
CHUNK_SIZE = 64
Z_MIN = 0.3
Z_MAX = 20.0
N_REDSHIFT_GRID = 256


#: The parameter pair each panel scans, in figure order.
GRID_PAIRS: tuple[tuple[str, str], ...] = (
    ("H0", "Omega_m"),
    ("xi_0", "xi_n"),
)

#: Scan ranges that deliberately differ from the committed inference prior.
#:
#: This figure asks "over what range of parameters does importance sampling
#: against the proposal still work", which is a wider question than "what does
#: the analysis assume". `Omega_m` is the one parameter where the two part
#: company: `config/priors.json` gives it Normal(0.3096, 0.006), a Planck-tight
#: constraint whose eps / 1-eps quantiles span about 0.004, so scanning it
#: would show a flat patch rather than where the proposal degrades. Every other
#: parameter scans its own prior, which is why this table has one entry and not
#: four -- it is an exception list, not a second prior declaration.
GRID_SCAN_RANGES: dict[str, dist.Distribution] = {
    "Omega_m": dist.Uniform(0.05, 0.95),
}


def grid_priors() -> tuple[
    tuple[tuple[str, dist.Distribution], tuple[str, dist.Distribution]], ...
]:
    """The distribution scanned on each axis, per panel.

    Sourced from `config/priors.json` so the axes match what the runs actually
    sample, except where `GRID_SCAN_RANGES` says otherwise.
    """
    inference = priors()
    return tuple(
        tuple(  # type: ignore[misc]
            (name, GRID_SCAN_RANGES.get(name, inference[name])) for name in pair
        )
        for pair in GRID_PAIRS
    )


def prior_grid(prior: dist.Distribution, *, eps: float, npoints: int) -> jax.Array:
    """Linspace between the ``eps`` and ``1 - eps`` prior quantiles."""
    low, high = prior.icdf(jnp.asarray([eps, 1.0 - eps]))
    return jnp.linspace(low, high, npoints)


def relative_ess(log_weights: jax.Array) -> jax.Array:
    """Relative ESS $(\\sum w)^2 / (N \\sum w^2)$ from log-importance weights."""
    weights = jnp.exp(log_weights)
    return jnp.sum(weights) ** 2 / (weights.shape[0] * jnp.sum(weights**2))


def evaluate_relative_ess_grid(
    axis0: tuple[str, jax.Array],
    axis1: tuple[str, jax.Array],
    *,
    constants: Mapping[str, float],
    log_weights_fn: Callable[[Mapping[str, jax.Array]], jax.Array],
    chunk_size: int,
) -> jax.Array:
    """Return relative ESS on the Cartesian product of ``axis0/1``."""
    name0, grid0 = axis0
    name1, grid1 = axis1
    mesh0, mesh1 = jnp.meshgrid(grid0, grid1, indexing="ij")
    points = jnp.stack([mesh0.ravel(), mesh1.ravel()], axis=-1)

    def _relative_ess_at_point(point: jax.Array) -> jax.Array:
        params = {**constants, name0: point[0], name1: point[1]}
        return relative_ess(log_weights_fn(params))

    ess = jax.lax.map(_relative_ess_at_point, points, batch_size=chunk_size).reshape(
        mesh0.shape
    )
    return jax.block_until_ready(ess)


def plot_relative_ess_heatmap(
    axis0: tuple[str, jax.Array],
    axis1: tuple[str, jax.Array],
    relative_ess_grid: jax.Array,
    *,
    fiducials: Mapping[str, float],
) -> Figure:
    """Heatmap of relative ESS over a two-parameter grid."""
    name0, grid0 = axis0
    name1, grid1 = axis1
    grid0_np = np.asarray(grid0, dtype=np.float64)
    grid1_np = np.asarray(grid1, dtype=np.float64)
    ess_np = np.asarray(relative_ess_grid, dtype=np.float64)

    # ``matshow`` uses image indexing (row, col) = (y, x); transpose so that
    # ``ess[i, j]`` at ``(grid0[i], grid1[j])`` lands on the correct axes.
    dx = 0.5 * (grid0_np[1] - grid0_np[0]) if grid0_np.size > 1 else 0.5
    dy = 0.5 * (grid1_np[1] - grid1_np[0]) if grid1_np.size > 1 else 0.5
    extent = (
        float(grid0_np[0] - dx),
        float(grid0_np[-1] + dx),
        float(grid1_np[0] - dy),
        float(grid1_np[-1] + dy),
    )

    fig, ax = plt.subplots()
    image = ax.matshow(
        ess_np.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.axvline(fiducials[name0], **TRUTH)
    ax.axhline(fiducials[name1], **TRUTH)
    ax.plot(fiducials[name0], fiducials[name1], marker="s", **TRUTH)
    ax.set_xlabel(parameter_label(name0))
    ax.set_ylabel(parameter_label(name1))
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(parameter_label("importance_relative_ess"))
    return fig


def combo_constants(
    param_names: Sequence[str], fiducials: Mapping[str, float]
) -> dict[str, float]:
    """Fiducials with the scanned parameters removed."""
    return {key: value for key, value in fiducials.items() if key not in param_names}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-h0-omega-m-pdf", type=Path, required=True)
    parser.add_argument("--output-xi0-n-pdf", type=Path, required=True)
    add_config_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    catalog_path = args.catalog
    config = build_run_config(load_merged_config(args))
    fiducials = dict(config.fiducials)
    use_paper_style()

    # The proposal density comes from the catalog's own population record,
    # exactly as scripts/run_mcmc.py builds it -- so this figure reweights
    # against the same denominator the chains did, with nothing restated here.
    catalog = load_run_catalog(catalog_path, label="proposal").restrict_redshift(
        Z_MIN, Z_MAX
    )
    n_samples = catalog.num_samples
    print(f"loaded catalog samples: n_proposal_samples={n_samples}")

    grid_settings = {"z_min": Z_MIN, "z_max": Z_MAX, "n_grid": N_REDSHIFT_GRID}
    target = build_population("bns_md_modified_propagation", **grid_settings)
    log_weights_fn = build_importance_spectrum(
        catalog,
        source_model=target.source_model,
        merger_rate_fn=target.merger_rate_fn,
    )[1]

    figures: list[tuple[Figure, Path]] = []
    for combo in grid_priors():
        (name0, prior0), (name1, prior1) = combo
        grid0 = prior_grid(prior0, eps=EPS, npoints=NPOINTS)
        grid1 = prior_grid(prior1, eps=EPS, npoints=NPOINTS)
        constants = combo_constants((name0, name1), fiducials)
        print(
            f"{name0} grid: [{float(grid0[0]):.4g}, {float(grid0[-1]):.4g}] "
            f"({NPOINTS} pts); "
            f"{name1} grid: [{float(grid1[0]):.4g}, {float(grid1[-1]):.4g}] "
            f"({NPOINTS} pts)"
        )
        ess = evaluate_relative_ess_grid(
            (name0, grid0),
            (name1, grid1),
            constants=constants,
            log_weights_fn=log_weights_fn,
            chunk_size=CHUNK_SIZE,
        )
        print(
            f"{name0}–{name1} relative ESS: "
            f"min={float(ess.min()):.4g} max={float(ess.max()):.4g}"
        )
        figure = plot_relative_ess_heatmap(
            (name0, grid0),
            (name1, grid1),
            ess,
            fiducials=fiducials,
        )
        output = args.output_h0_omega_m_pdf if name0 == "H0" else args.output_xi0_n_pdf
        figures.append((figure, output))

    for figure, output in figures:
        output_path = output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
        print("saved figure:", output_path)


if __name__ == "__main__":
    main()
