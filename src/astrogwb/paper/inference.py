"""One implementation of the catalogs-to-model-inputs pipeline.

Every entrypoint that samples, profiles, or plots the fiducial spectrum runs
the same sequence: load the catalogs, validate them, build the fiducial
injection spectrum, build the effective PSD, build the analysis-band mask,
build the importance-weight closure, build the model. It used to be spelled
out at eight call sites, two of which were near-verbatim clones of each other
down to the model-building block.

JAX ops run only inside functions, after ``runtime.configure_runtime``. Importing
this module loads ``jax`` but does not initialize the XLA backend; a subprocess
test in ``tests/test_cli.py`` guards that.

Arrays on :class:`Observation` are *pre*-mask, with the mask carried alongside,
because the notebooks plot the unmasked PSD and spectrum before restricting to
the analysis band. :meth:`InferenceInputs.masked_model_kwargs` applies it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpyro import handlers
from numpyro.distributions import Distribution

from astrogwb.detector import effective_psd as compute_effective_psd
from astrogwb.detector import load_sensitivity_map
from astrogwb.frequency import frequency_mask as make_frequency_mask
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    amplitude_H0_fn,
    amplitude_local_merger_rate_fn,
    make_merger_rate_and_log_weights_fn,
    merger_rate_H0_fn,
    merger_rate_local_merger_rate_fn,
)
from astrogwb.paper.catalogs import (
    compute_fiducial_injection_spectrum,
    compute_proposal_logprob,
    propagate_catalog,
    samples_from_catalog,
    truncate_catalog_samples,
    validate_matching_frequency_grids,
)
from astrogwb.paper.config.mcmc import AnalysisGrid, ProposalConfig, RunConfig
from astrogwb.sampling.amplitude import (
    AmplitudeFn,
    MergerRateAmplitudeFn,
    quadrature_grid,
)
from astrogwb.sampling.models import (
    amplitude_marginalized_model,
    spectral_density_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """The observed-data side of a run: the fiducial injection spectrum.

    Arrays are pre-mask; ``frequency_mask`` selects the analysis band. ``df``
    is the catalog's bin width and stays valid under the mask, which is why it
    is carried here rather than measured off the masked grid.
    """

    frequencies: jax.Array
    df: float
    redshift_grid: jax.Array
    total_merger_rate: jax.Array
    spectral_density: jax.Array
    frequency_mask: jax.Array


@dataclass(frozen=True)
class InferenceInputs:
    """Everything the NumPyro model is evaluated against, for one run."""

    observation: Observation
    proposal: xr.Dataset
    effective_psd: jax.Array
    observation_time: float
    merger_rate_and_log_weights_fn: Any

    def masked_model_kwargs(self) -> dict[str, Any]:
        """Restrict to the analysis band and return the model's keyword inputs."""
        observation = self.observation
        mask = np.asarray(observation.frequency_mask)
        # `source_parameters` has no `frequency` dim, so `isel` cannot touch it --
        # the old "NOT masked" hazard is now structurally enforced.
        band = self.proposal.isel(frequency=mask)
        return {
            "polarization_power": jnp.asarray(band.polarization_power.values),
            "samples": samples_from_catalog(band),
            "observed_spectral_density": observation.spectral_density[mask],
            "effective_psd": self.effective_psd[mask],
            "observation_time": self.observation_time,
            "df": observation.df,
        }


class AmplitudeMarginalization(NamedTuple):
    """Everything an amplitude-marginalized run needs, built once from a ``RunConfig``.

    App-side plumbing, not a core type: unlike the ``AmplitudeQuadrature`` it
    replaces, it holds *live* objects -- the prior distribution and the scaling
    callables -- so there is nothing derived in it that could go stale against
    the config it came from. The one array, ``grid``, is a quadrature scheme
    rather than a tabulation of the density.
    """

    parameter: str
    """Name of the marginalized parameter, e.g. ``"H0"``."""

    fiducial: float
    """Reference value defining the template; the amplitude is 1 here."""

    prior: Distribution
    """Prior on the marginalized parameter; also defines the conditional's support."""

    amplitude_fn: AmplitudeFn
    """Absolute total scaling :math:`f(\\varphi) = g_R(\\varphi)\\, g_F(\\varphi)`."""

    merger_rate_fn: MergerRateAmplitudeFn
    """Absolute merger-rate scaling :math:`g_R(\\varphi)`, for the reconstructed rate."""

    grid: jax.Array
    """Quadrature nodes the marginalization integral is evaluated on."""


def prepare_observation(
    injection: xr.Dataset,
    *,
    fiducials: Mapping[str, float],
    grid: AnalysisGrid,
) -> Observation:
    """Propagate the injection catalog and build the fiducial observed spectrum."""
    fiducial_values = dict(fiducials)
    composed = propagate_catalog(injection, fiducials=fiducial_values)
    n_loaded = composed.polarization_power.shape[1]
    composed = truncate_catalog_samples(
        composed,
        label="injection",
        minimum_redshift=grid.minimum_redshift,
        maximum_redshift=grid.maximum_redshift,
    )
    n_kept = composed.polarization_power.shape[1]
    logger.info(
        "Loaded independent injection catalog: n_injection_samples=%d "
        "(%d outside the analysis window dropped)",
        n_kept,
        n_loaded - n_kept,
    )

    redshift_grid = jnp.linspace(
        grid.minimum_redshift, grid.maximum_redshift, grid.n_grid
    )
    injection_frequencies = jnp.asarray(composed.frequency.values)
    df = float(composed.attrs["df"])
    total_merger_rate, spectral_density = compute_fiducial_injection_spectrum(
        jnp.asarray(composed.polarization_power.values),
        samples_from_catalog(composed),
        fiducials=fiducial_values,
        redshift_grid=redshift_grid,
    )
    logger.info(
        "Constructed independent fiducial observed spectrum (rate0=%.4e /s)",
        total_merger_rate,
    )

    # Band bounds only: this function never sees a detector network, so bins
    # the network cannot measure are dropped later, in prepare_inference_inputs.
    analysis_frequency_mask = make_frequency_mask(
        injection_frequencies, fmin=grid.f_min, fmax=grid.f_max
    )
    logger.info(
        "Analysis band: %d of %d bins (%.1f-%.1f Hz)",
        int(jnp.sum(analysis_frequency_mask)),
        injection_frequencies.shape[0],
        grid.f_min,
        grid.f_max,
    )
    return Observation(
        frequencies=injection_frequencies,
        df=df,
        redshift_grid=redshift_grid,
        total_merger_rate=total_merger_rate,
        spectral_density=spectral_density,
        frequency_mask=analysis_frequency_mask,
    )


def prepare_inference_inputs(
    injection: xr.Dataset,
    proposal: xr.Dataset,
    *,
    fiducials: Mapping[str, float],
    proposal_config: ProposalConfig,
    grid: AnalysisGrid,
    detectors: Sequence[str],
) -> InferenceInputs:
    """Build every array the model is evaluated against, from the two catalogs."""
    observation = prepare_observation(injection, fiducials=fiducials, grid=grid)
    proposal_catalog = propagate_catalog(proposal, fiducials=dict(fiducials))
    n_loaded = proposal_catalog.polarization_power.shape[1]
    proposal_catalog = truncate_catalog_samples(
        proposal_catalog,
        label="proposal",
        minimum_redshift=grid.minimum_redshift,
        maximum_redshift=grid.maximum_redshift,
    )
    proposal_frequencies = proposal_catalog.frequency.values
    validate_matching_frequency_grids(observation.frequencies, proposal_frequencies)
    n_freq, n_samples = proposal_catalog.polarization_power.shape
    logger.info(
        "Loaded proposal catalog: n_frequency_bins=%d n_proposal_samples=%d "
        "(%d outside the analysis window dropped)",
        n_freq,
        n_samples,
        n_loaded - n_samples,
    )

    # Two frequency grids are in play and they are deliberately spelled
    # differently: the band mask comes from the *injection* grid (inside
    # prepare_observation, which never sees a proposal) while the effective PSD
    # comes from the *proposal* grid. validate_matching_frequency_grids has
    # already proved the two arrays equal, so both results are bit-identical --
    # do not "tidy" either one to match the other.
    sensitivities = load_sensitivity_map(detectors)
    effective_psd_arr = jnp.asarray(
        compute_effective_psd(proposal_frequencies, list(detectors), sensitivities)
    )
    # `compute_effective_psd` returns inf wherever no detector pair contributes,
    # and Normal(loc, inf).log_prob is -inf -- a constant that kills NUTS with no
    # usable diagnostic. Drop those bins along with the out-of-band ones. This is
    # safe precisely because `df` is the catalog's attribute: the surviving bins
    # need not be contiguous, and each still has width `df`.
    band_mask = (
        observation.frequency_mask
        & jnp.isfinite(effective_psd_arr)
        & (effective_psd_arr > 0.0)
    )
    num_bins = int(jnp.sum(band_mask))
    if num_bins < 2:
        raise ValueError(
            f"only {num_bins} usable frequency bin(s) in "
            f"[{grid.f_min}, {grid.f_max}] Hz for detectors "
            f"{' '.join(detectors)}; widen the band or choose a detector "
            "network with full coverage"
        )
    dropped = int(jnp.sum(observation.frequency_mask)) - num_bins
    if dropped:
        logger.info(
            "Dropped %d in-band bin(s) with no detector-network coverage", dropped
        )
    observation = replace(observation, frequency_mask=band_mask)

    proposal_redshift = proposal_catalog.source_parameters.sel(
        parameter="redshift"
    ).values
    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=dict(fiducials),
        redshift_grid=observation.redshift_grid,
        proposal_logprob=compute_proposal_logprob(proposal_redshift, proposal_config),
    )
    return InferenceInputs(
        observation=observation,
        proposal=proposal_catalog,
        effective_psd=effective_psd_arr,
        observation_time=grid.observation_time,
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
    )


def initial_values(config: RunConfig) -> dict[str, float]:
    """Fiducial values NUTS initializes each sampled parameter at."""
    return {name: config.fiducials[name] for name in config.sampled_params}


def _fix_model_params(model: Any, fixed_params: Mapping[str, Any]) -> Any:
    """Condition fixed sites, then hide their values and prior densities."""
    if not fixed_params:
        return model
    names = list(fixed_params)
    return handlers.block(
        handlers.condition(model, data=dict(fixed_params)),
        hide=names,
    )


def build_model(
    config: RunConfig,
    *,
    merger_rate_and_log_weights_fn: Any,
) -> tuple[Any, AmplitudeMarginalization | None]:
    """Build the NumPyro model this config describes.

    Returns ``(model, marginalization)``, where ``marginalization`` is the
    :class:`~astrogwb.paper.inference.AmplitudeMarginalization` built for an
    amplitude-marginalized run, or ``None`` for the default likelihood.

    Depends only on the config and the weights closure -- no catalog array --
    so it is exercisable without generating one.
    """
    analysis = config.analysis
    # `config.priors` holds every live parameter distribution. Fixed sites are
    # conditioned and hidden below; a marginalized amplitude is the sole site
    # omitted because its prior is already integrated into the likelihood.
    priors = dict(config.priors)
    if analysis.likelihood == "amplitude_marginalized":
        parameter = analysis.amplitude_parameter
        assert parameter is not None
        # `build_run_config` guarantees the amplitude's prior lives in `priors`
        # and its fiducial lives in `fiducials` (see
        # `RunConfig._resolve_sampled_params`), so the marginalization can be
        # assembled inline: dispatch on the parameter name, then derive the
        # quadrature grid from the very prior being integrated.
        prior = priors.pop(parameter)
        if parameter == "H0":
            amplitude_fn, merger_rate_fn = amplitude_H0_fn, merger_rate_H0_fn
        elif parameter == "local_merger_rate":
            amplitude_fn, merger_rate_fn = (
                amplitude_local_merger_rate_fn,
                merger_rate_local_merger_rate_fn,
            )
        else:
            raise ValueError(f"unsupported amplitude parameter {parameter!r}")
        marginalization = AmplitudeMarginalization(
            parameter=parameter,
            fiducial=float(config.fiducials[parameter]),
            prior=prior,
            amplitude_fn=amplitude_fn,
            merger_rate_fn=merger_rate_fn,
            grid=quadrature_grid(
                prior,
                num_nodes=analysis.amplitude_num_nodes,
                span_sigma=analysis.amplitude_prior_span_sigma,
            ),
        )
        fixed_params = {
            name: value for name, value in config.fixed_params.items() if name in priors
        }
        model = _fix_model_params(
            partial(
                amplitude_marginalized_model,
                average_mode="analytic_inclination",
                merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
                amplitude_parameter=parameter,
                fiducials=config.fiducials,
                amplitude_fn=marginalization.amplitude_fn,
                amplitude_prior=marginalization.prior,
                amplitude_grid=marginalization.grid,
                priors=priors,
            ),
            fixed_params,
        )
        return model, marginalization

    model = _fix_model_params(
        partial(
            spectral_density_model,
            average_mode="analytic_inclination",
            merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
            priors=priors,
        ),
        config.fixed_params,
    )
    return model, None
