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
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from astrogwb.detector import effective_psd as compute_effective_psd
from astrogwb.detector import gaussian_bin_scale, load_sensitivity_map
from astrogwb.frequency import frequency_slice as make_frequency_slice
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.sampling.models import (
    amplitude_marginalized_model,
    spectral_density_model,
)
from numpyro import handlers

from astrogwb_paper.amplitude import (
    AmplitudeMarginalization,
    build_amplitude_marginalization,
)
from astrogwb_paper.catalogs import (
    CatalogSource,
    compute_fiducial_injection_spectrum,
    compute_proposal_logprob,
    propagate_catalog,
    samples_from_catalog,
    truncate_catalog_samples,
    validate_matching_frequency_grids,
)
from astrogwb_paper.config.mcmc import AnalysisGrid, ProposalConfig, RunConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """The observed-data side of a run: the fiducial injection spectrum.

    Arrays are pre-slice; ``frequency_slice`` selects the analysis band.
    """

    frequencies: jax.Array
    df: float
    redshift_grid: jax.Array
    total_merger_rate: jax.Array
    spectral_density: jax.Array
    frequency_slice: slice


@dataclass(frozen=True)
class InferenceInputs:
    """Everything the NumPyro model is evaluated against, for one run."""

    observation: Observation
    proposal: xr.Dataset
    noise_scale: jax.Array
    merger_rate_and_log_weights_fn: Any

    def masked_model_kwargs(self) -> dict[str, Any]:
        """Restrict to the analysis band and return the model's keyword inputs."""
        observation = self.observation
        frequency_slice = observation.frequency_slice
        # `source_parameters` has no `frequency` dim, so `isel` cannot touch it --
        # the old "NOT masked" hazard is now structurally enforced.
        band = self.proposal.isel(frequency=frequency_slice)
        return {
            "polarization_power": jnp.asarray(band.polarization_power.values),
            "samples": samples_from_catalog(band),
            "observed_spectral_density": observation.spectral_density[frequency_slice],
            "noise_scale": self.noise_scale[frequency_slice],
        }


def prepare_observation(
    injection: CatalogSource,
    *,
    fiducials: Mapping[str, float],
    grid: AnalysisGrid,
) -> Observation:
    """Compose the injection catalog and build the fiducial observed spectrum."""
    fiducial_values = dict(fiducials)
    composed = propagate_catalog(injection.compose(), fiducials=fiducial_values)
    n_loaded = composed.polarization_power.shape[1]
    composed = truncate_catalog_samples(
        composed,
        label="injection",
        minimum_redshift=grid.minimum_redshift,
        maximum_redshift=grid.maximum_redshift,
    )
    n_kept = composed.polarization_power.shape[1]
    logger.info(
        "Composed independent %s catalog: n_injection_samples=%d "
        "(%d outside the analysis window dropped)",
        injection.role,
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

    analysis_frequency_slice = make_frequency_slice(
        injection_frequencies, fmin=grid.f_min, fmax=grid.f_max
    )
    start = analysis_frequency_slice.start or 0
    stop = analysis_frequency_slice.stop or injection_frequencies.shape[0]
    logger.info(
        "Analysis band: %d of %d bins (%.1f-%.1f Hz)",
        stop - start,
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
        frequency_slice=analysis_frequency_slice,
    )


def prepare_inference_inputs(
    injection: CatalogSource,
    proposal: CatalogSource,
    *,
    fiducials: Mapping[str, float],
    proposal_config: ProposalConfig,
    grid: AnalysisGrid,
    detectors: Sequence[str],
) -> InferenceInputs:
    """Build every array the model is evaluated against, from the two catalogs."""
    observation = prepare_observation(injection, fiducials=fiducials, grid=grid)
    proposal_catalog = propagate_catalog(proposal.compose(), fiducials=dict(fiducials))
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
        "Composed %s catalog: n_frequency_bins=%d n_proposal_samples=%d "
        "(%d outside the analysis window dropped)",
        proposal.role,
        n_freq,
        n_samples,
        n_loaded - n_samples,
    )

    # Two frequency grids are in play and they are deliberately spelled
    # differently: the band slice comes from the *injection* grid (inside
    # prepare_observation, which never sees a proposal) while the effective PSD
    # comes from the *proposal* grid. validate_matching_frequency_grids has
    # already proved the two arrays equal, so both results are bit-identical --
    # do not "tidy" either one to match the other.
    sensitivities = load_sensitivity_map(detectors)
    effective_psd_arr = jnp.asarray(
        compute_effective_psd(proposal_frequencies, list(detectors), sensitivities)
    )
    noise_scale = gaussian_bin_scale(
        effective_psd_arr, grid.observation_time, observation.df
    )
    selected_noise_scale = np.asarray(noise_scale[observation.frequency_slice])
    if not np.all(np.isfinite(selected_noise_scale)) or not np.all(
        selected_noise_scale > 0.0
    ):
        raise ValueError(
            "analysis band contains non-finite or non-positive noise scale values; "
            "narrow the band or choose a detector network with full coverage"
        )

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
        noise_scale=noise_scale,
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
    :class:`~astrogwb_paper.amplitude.AmplitudeMarginalization` built for an
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
        assert analysis.amplitude_parameter is not None
        priors.pop(analysis.amplitude_parameter)
        fixed_params = {
            name: value for name, value in config.fixed_params.items() if name in priors
        }
        marginalization = build_amplitude_marginalization(config)
        model = _fix_model_params(
            partial(
                amplitude_marginalized_model,
                average_mode="analytic_inclination",
                merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
                amplitude_parameter=analysis.amplitude_parameter,
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
