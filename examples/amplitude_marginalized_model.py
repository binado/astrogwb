r"""Infer :math:`(H_0, \Omega_m)` with :math:`H_0` marginalized analytically.

The sibling of ``h0_mcmc.py``, using the *other* half of
:mod:`astrogwb.sampling`: :func:`~astrogwb.sampling.amplitude_marginalized_model`
integrates :math:`H_0` out of the Gaussian likelihood, NUTS samples
:math:`\Omega_m` alone, and a mandatory post-processing pass through
:func:`~astrogwb.sampling.amplitude_reconstruction_model` draws :math:`H_0`
back to give a joint :math:`(H_0, \Omega_m)` posterior. This mirrors the
``cosmological-parameters/H0-Omega_m`` run shipped with the ``astrogwb-paper``
workspace project, which is how every production analysis is set up.

Why the marginalization is exact rather than approximate: the *normalized*
redshift PDF's :math:`H_0` dependence cancels, leaving
``total_merger_rate`` :math:`\propto H_0^{-3}` (the :math:`g_R` piece) and
:math:`\exp(-2\,\Delta \log d_L) \propto H_0^{2}` (the :math:`g_F` piece), so
the total scaling is :math:`f(H_0) = H_0^{-1}` -- precisely
:func:`~astrogwb.importance.models.bns_madau_dickinson_modified_propagation.amplitude_H0_fn`.
That factorization holds at any :math:`\Omega_m`, so marginalizing one
parameter while sampling the other introduces no error beyond the quadrature
error of the amplitude grid.

**There is no ``H0`` site in the NUTS chain.** A ``numpyro.factor`` publishes no
draws, so a script that swapped in the marginalized model and stopped there
would write chains with no :math:`H_0` in them at all. The reconstruction below
is not optional, and it must reference the *same* prior, grid, and scaling
function the chain marginalized against -- a mismatched conditional yields a
silently wrong posterior, because the sufficient statistics stay finite and
plausible whatever conditional you pair them with.

Everything else matches ``h0_mcmc.py``: the same catalog serves as both the
injection and the importance-sampling proposal, so every weight is exactly 1 at
:math:`\theta = \theta_{\rm fid}`, and the data is noiseless so the recovered
:math:`H_0` should sit on ``FIDUCIALS["H0"]``.

Usage::

    python amplitude_marginalized_model.py CATALOG.h5 -o chains.nc \
        --detectors E1 E2 E3 --num-warmup 500 --num-samples 1000

Chains are written as an xarray netCDF file with ``(chain, draw)`` dimensions,
readable with ``xarray.open_dataset(..., engine="h5netcdf")`` or, if you have
arviz installed, ``arviz.from_dict``.
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value

from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.frequency import apply_frequency_mask, frequency_mask
from astrogwb.gwb import spectral_density
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    amplitude_H0_fn,
    compute_merger_rate_distance_and_logprob,
    make_merger_rate_and_log_weights_fn,
    merger_rate_H0_fn,
)
from astrogwb.sampling import (
    amplitude_marginalized_model,
    amplitude_reconstruction_model,
    quadrature_grid,
)

logger = logging.getLogger(__name__)

#: Hyperparameters the injection is built at and every non-sampled site is
#: pinned to. ``local_merger_rate`` is in Gpc^-3 yr^-1; the rest feed the
#: Madau-Dickinson rate shape and the flat-LambdaCDM cosmology.
FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
}

#: Catalog source parameters the weights callback dereferences by name.
REQUIRED_PARAMETERS = ("redshift", "luminosity_distance")


def load_catalog(path: Path) -> xr.Dataset:
    """Load and locally validate the example's external xarray input."""
    catalog = xr.load_dataset(path, engine="h5netcdf")
    if catalog.attrs.get("format_name") == "waveform_catalog":
        raise ValueError(f"{path}: obsolete catalog format; regenerate the catalog")
    if catalog.attrs.get("format_name") != "astrogwb_catalog":
        raise ValueError(f"{path}: expected format_name='astrogwb_catalog'")
    if catalog.attrs.get("domain") != "frequency":
        raise ValueError(f"{path}: expected domain='frequency'")
    if "df" not in catalog.attrs:
        raise ValueError(f"{path}: missing required df attribute")
    if "frequency" not in catalog.coords or "parameter" not in catalog.coords:
        raise ValueError(f"{path}: missing frequency or parameter coordinate")
    if "polarization_power" not in catalog or catalog.polarization_power.dims != (
        "frequency",
        "sample",
    ):
        raise ValueError(
            f"{path}: polarization_power must have dims (frequency, sample)"
        )
    if "source_parameters" not in catalog or catalog.source_parameters.dims != (
        "sample",
        "parameter",
    ):
        raise ValueError(
            f"{path}: source_parameters must have dims (sample, parameter)"
        )
    return catalog


#: Plain-text help banner. The module docstring is reStructuredText and turns
#: into an unreadable wall once argparse rewraps it.
DESCRIPTION = """\
Infer (H0, Omega_m) with H0 marginalized analytically out of the likelihood.

NUTS samples Omega_m alone; H0 is integrated out under its own prior and
reconstructed afterwards, so the written chains carry a joint (H0, Omega_m)
posterior even though H0 never appears as a chain site.

The catalog serves as both the injection and the importance-sampling proposal,
so the observed spectrum is the unweighted catalog contraction and the true H0
is the fiducial one (67.66). The injection carries no noise realization, so the
reconstructed H0 should recover that value tightly.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "catalog",
        type=Path,
        help="path to an astrogwb_catalog HDF5 file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="path to write the chains to, as netCDF",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["E1", "E2", "E3"],
        metavar="NAME",
        help="detector names from astrogwb's sensitivity table; at least two",
    )
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--f-min", type=float, default=10.0, help="band start, Hz")
    parser.add_argument("--f-max", type=float, default=2048.0, help="band end, Hz")
    parser.add_argument(
        "--observation-time",
        type=float,
        default=1.0,
        help="observation time in years",
    )
    parser.add_argument(
        "--max-catalog-samples",
        type=int,
        default=None,
        metavar="N",
        help="use only the first N catalog sources (production banks hold 16k+)",
    )
    parser.add_argument(
        "--zmin",
        "--z-min",
        type=float,
        default=0.0,
        dest="zmin",
        help="minimum redshift for analysis window and grid",
    )
    parser.add_argument(
        "--zmax",
        "--z-max",
        type=float,
        default=20.0,
        dest="zmax",
        help="maximum redshift for analysis window and grid",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=256,
        help="nodes in the redshift grid the cosmology integrals run on",
    )
    parser.add_argument(
        "--h0-min", type=float, default=20.0, help="H0 amplitude prior lower bound"
    )
    parser.add_argument(
        "--h0-max", type=float, default=140.0, help="H0 amplitude prior upper bound"
    )
    parser.add_argument(
        "--omega-m-loc", type=float, default=0.3096, help="Omega_m prior mean"
    )
    parser.add_argument(
        "--omega-m-scale",
        type=float,
        default=0.006,
        help="Omega_m prior standard deviation",
    )
    parser.add_argument(
        "--amplitude-num-nodes",
        type=int,
        default=1024,
        help="nodes in the H0 quadrature grid the marginalization runs on",
    )
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="suppress the NUTS progress bar (useful in batch jobs)",
    )
    return parser


def load_samples(
    path: Path,
    max_samples: int | None,
    *,
    zmin: float = 0.0,
    zmax: float = 20.0,
) -> xr.Dataset:
    """Load a catalog, filter by redshift, and check required source parameters."""
    catalog = load_catalog(path)
    available = {str(name) for name in catalog.parameter.values}
    missing = [name for name in REQUIRED_PARAMETERS if name not in available]
    if missing:
        raise ValueError(
            f"{path}: catalog is missing required source parameter(s) "
            f"{', '.join(missing)}; it has {', '.join(sorted(available))}"
        )

    redshift = catalog.source_parameters.sel(parameter="redshift").values
    in_window = (redshift >= zmin) & (redshift <= zmax)
    if not np.any(in_window):
        raise ValueError(
            f"{path}: catalog has no samples in the redshift window [{zmin}, {zmax}]"
        )
    catalog = catalog.isel(sample=np.flatnonzero(in_window))

    if max_samples is not None:
        catalog = catalog.isel(sample=slice(0, max_samples))

    return catalog


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = build_parser().parse_args(argv)

    if args.zmin < 0.0:
        raise ValueError(f"--zmin must be non-negative, got {args.zmin}")
    if args.zmin >= args.zmax:
        raise ValueError(
            f"--zmin ({args.zmin}) must be strictly less than --zmax ({args.zmax})"
        )

    # Runtime configuration must precede the first array: set_host_device_count
    # mutates XLA_FLAGS, and x64 must be on before any array is created. The
    # catalog's polarization power is tiny (|h|^2 at Gpc distances) and the
    # likelihood contracts it against 1/S_eff^2, which underflows in float32.
    numpyro.set_host_device_count(args.num_chains)
    jax.config.update("jax_enable_x64", True)

    if len(args.detectors) < 2:
        raise ValueError(
            "the effective PSD is a cross-correlation sum and needs at least "
            f"two detectors, got {args.detectors}"
        )

    catalog = load_samples(
        args.catalog,
        args.max_catalog_samples,
        zmin=args.zmin,
        zmax=args.zmax,
    )
    # Catalog power is generated at the fiducial *electromagnetic* distance.
    # At xi_0 = 1.0 the GW/EM ratio is identically 1, so no correction is
    # needed here; a run that varies the propagation parameters would apply
    # astrogwb.waveform.apply_gw_distance_to_power first.
    frequencies = jnp.asarray(catalog.frequency.values)
    df = float(catalog.attrs["df"])
    polarization_power = jnp.asarray(catalog.polarization_power.values)
    samples = {
        str(name): jnp.asarray(catalog.source_parameters.sel(parameter=name).values)
        for name in catalog.parameter.values
    }
    num_sources = polarization_power.shape[1]
    logger.info(
        "Loaded %s: %d frequency bins, %d sources",
        args.catalog,
        frequencies.shape[0],
        num_sources,
    )

    # The grid spans the analysis redshift window [zmin, zmax]. Catalog samples
    # outside this range have been discarded, so every retained source lands
    # inside the grid without interpolating out of bounds.
    redshift_grid = jnp.linspace(args.zmin, args.zmax, args.n_grid)

    # One call yields both the fiducial rate (for the injection) and the
    # proposal log-density (for the weights). Sharing the function is what
    # keeps the target and proposal densities from drifting apart.
    total_merger_rate, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=redshift_grid
    )
    observed_spectral_density = spectral_density(
        polarization_power,
        jnp.ones(num_sources),
        total_merger_rate,
        average_mode="analytic_inclination",
    )
    logger.info("Fiducial injection: total merger rate %.4e /s", total_merger_rate)

    weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=FIDUCIALS,
        redshift_grid=redshift_grid,
        proposal_logprob=proposal_logprob,
    )

    def merger_rate_and_log_weights(params, samples):
        """Sample Omega_m; take every other hyperparameter from the fiducials.

        ``params`` carries the NUTS latents *and* ``H0``, which the marginalized
        model pins at its fiducial before calling this: what comes back is the
        template spectrum the amplitude ratio multiplies.
        """
        return weights_fn({**FIDUCIALS, **params}, samples)

    sensitivities = load_sensitivity_map(args.detectors)
    network_psd = jnp.asarray(effective_psd(frequencies, args.detectors, sensitivities))
    # effective_psd returns inf wherever no detector pair contributes, and
    # Normal(loc, inf).log_prob is -inf -- a constant that kills NUTS with no
    # usable diagnostic. Drop those bins along with the out-of-band ones. The
    # surviving bins need not be contiguous: each still has width `df`, which
    # comes from the catalog rather than from the masked grid.
    mask = frequency_mask(frequencies, fmin=args.f_min, fmax=args.f_max) & jnp.isfinite(
        network_psd
    )
    num_bins = int(jnp.sum(mask))
    if num_bins < 2:
        raise ValueError(
            f"only {num_bins} usable frequency bin(s) in "
            f"[{args.f_min}, {args.f_max}] Hz for detectors "
            f"{' '.join(args.detectors)}; widen the band or pick a network "
            "whose noise curves cover it"
        )
    # `samples` is deliberately not masked: it has no frequency dimension.
    frequencies, polarization_power, observed_spectral_density, network_psd = (
        apply_frequency_mask(
            mask,
            frequencies,
            polarization_power,
            observed_spectral_density,
            network_psd,
        )
    )
    logger.info(
        "Analysis band: %d bins, detectors %s, %s yr",
        num_bins,
        " ".join(args.detectors),
        args.observation_time,
    )

    # Built once, referenced by both the chain and the reconstruction: the
    # reconstruction is only exact against the very density the factor site
    # integrated, and a separately-constructed prior or grid is exactly the
    # mismatch that goes unnoticed.
    amplitude_prior = dist.Uniform(args.h0_min, args.h0_max)
    amplitude_grid = quadrature_grid(
        amplitude_prior, num_nodes=args.amplitude_num_nodes
    )

    model = partial(
        amplitude_marginalized_model,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=observed_spectral_density,
        effective_psd=network_psd,
        observation_time=args.observation_time,
        df=df,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights,
        amplitude_parameter="H0",
        fiducials=FIDUCIALS,
        # Passed by name, never wrapped in a lambda or partial:
        # AmplitudeConditional hashes amplitude_fn into the jit cache key, so a
        # freshly-minted callable retraces the model on every construction.
        amplitude_fn=amplitude_H0_fn,
        amplitude_prior=amplitude_prior,
        amplitude_grid=amplitude_grid,
        priors={"Omega_m": dist.Normal(args.omega_m_loc, args.omega_m_scale)},
    )
    # One latent against an (F, N) matvec is exactly the case forward-mode AD
    # is built for: reverse mode would tape the whole contraction.
    kernel = NUTS(
        model,
        target_accept_prob=args.target_accept,
        forward_mode_differentiation=True,
        init_strategy=init_to_value(values={"Omega_m": FIDUCIALS["Omega_m"]}),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=not args.no_progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(args.seed),
        extra_fields=("num_steps", "accept_prob", "diverging"),
    )
    # Covers Omega_m only: H0 is marginalized, not sampled, and is reconstructed
    # below.
    mcmc.print_summary()

    posterior = mcmc.get_samples(group_by_chain=True)
    # The sufficient statistics form the AmplitudeConditional batch shape, so
    # one Predictive invocation draws one H0 per (chain, draw) element. There is
    # no forward physics here -- no catalog, no (F, N) contraction.
    draws = Predictive(
        partial(
            amplitude_reconstruction_model,
            amplitude_parameter="H0",
            amplitude_fn=amplitude_H0_fn,
            merger_rate_amplitude_fn=merger_rate_H0_fn,
            prior=amplitude_prior,
            fiducial=FIDUCIALS["H0"],
            grid=amplitude_grid,
        ),
        num_samples=1,
        return_sites=["H0", "total_merger_rate", "quadrature_effective_nodes"],
    )(
        # fold_in keeps the reconstruction key distinct from the chain's.
        jax.random.fold_in(jax.random.PRNGKey(args.seed), 1),
        amplitude_mle=posterior["amplitude_mle"],
        template_optimal_snr=posterior["template_optimal_snr"],
        template_merger_rate=posterior["template_merger_rate"],
    )
    posterior.update({name: values[0] for name, values in draws.items()})

    relative_ess = float(jnp.mean(posterior["importance_relative_ess"]))
    reconstructed = posterior["H0"]
    logger.info("Fiducial H0: %s", FIDUCIALS["H0"])
    logger.info(
        "Reconstructed H0: %.3f +/- %.3f",
        float(jnp.mean(reconstructed)),
        float(jnp.std(reconstructed)),
    )
    logger.info("Mean importance relative ESS: %.4f", relative_ess)
    if relative_ess < 0.1:
        logger.warning(
            "importance weights have collapsed; the posterior is "
            "dominated by a handful of catalog sources"
        )

    chains = xr.Dataset(
        {
            str(name): (("chain", "draw"), np.asarray(values))
            for name, values in posterior.items()
        },
        attrs={
            "catalog": str(args.catalog),
            "detectors": " ".join(args.detectors),
            "seed": args.seed,
            "f_min": args.f_min,
            "f_max": args.f_max,
            "zmin": args.zmin,
            "zmax": args.zmax,
            "num_frequency_bins": num_bins,
            "observation_time": args.observation_time,
            "num_catalog_samples": num_sources,
            "amplitude_parameter": "H0",
            "amplitude_num_nodes": args.amplitude_num_nodes,
            "omega_m_loc": args.omega_m_loc,
            "omega_m_scale": args.omega_m_scale,
            **{f"fiducial_{name}": value for name, value in FIDUCIALS.items()},
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chains.to_netcdf(args.output, engine="h5netcdf")
    logger.info("Wrote chains to %s", args.output)


if __name__ == "__main__":
    main()
