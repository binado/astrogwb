"""One implementation of the catalogs-to-model-inputs pipeline.

Every entrypoint that samples, profiles, or plots the fiducial spectrum runs
the same sequence: load the catalogs, validate them, build the fiducial
injection spectrum, build the effective PSD, build the analysis-band mask,
build the importance-weight closure, build the model. It used to be spelled
out at eight call sites, two of which were near-verbatim clones of each other
down to the model-building block.

Ordering contract (do not "tidy" away): every ``jax`` / ``astrogwb`` /
``numpyro`` import here is function-local, exactly as
:mod:`astrogwb_paper.catalogs` does it, so importing this module does not
initialize a JAX backend -- ``runtime.configure_runtime`` must run first, and a
subprocess test in ``tests/test_cli.py`` guards it. The explicit ``jnp``
parameter (matching ``load_catalog_arrays`` and
``compute_fiducial_injection_spectrum``) keeps that ordering visible at the
call site; every caller already has ``jnp`` in scope.

Arrays on :class:`Observation` are *pre*-mask, with the mask carried alongside,
because the notebooks plot the unmasked PSD and spectrum before restricting to
the analysis band. :meth:`InferenceInputs.masked_model_kwargs` applies it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrogwb_paper.catalogs import CatalogArrays
from astrogwb_paper.config.analysis import AnalysisGrid
from astrogwb_paper.config.mcmc import RunConfig

if TYPE_CHECKING:
    from astrogwb_paper.amplitude import AmplitudeMarginalization

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """The observed-data side of a run: the fiducial injection spectrum.

    Arrays are pre-mask; ``frequency_mask`` selects the analysis band.
    """

    injection: CatalogArrays
    redshift_grid: Any
    total_merger_rate: Any
    spectral_density: Any
    frequency_mask: Any

    @property
    def frequencies(self) -> Any:
        """The injection catalog's frequency grid (shared with the proposal)."""
        return self.injection.frequencies


@dataclass(frozen=True)
class InferenceInputs:
    """Everything the NumPyro model is evaluated against, for one run."""

    observation: Observation
    proposal: CatalogArrays
    effective_psd: Any
    merger_rate_and_log_weights_fn: Any

    def masked_model_kwargs(self) -> dict[str, Any]:
        """Restrict to the analysis band and return the model's keyword inputs."""
        from astrogwb.frequency import apply_frequency_mask

        observation = self.observation
        (
            frequencies,
            polarization_power,
            observed_spectral_density,
            effective_psd,
        ) = apply_frequency_mask(
            observation.frequency_mask,
            self.proposal.frequencies,
            self.proposal.polarization_power,
            observation.spectral_density,
            self.effective_psd,
        )
        return {
            "frequencies": frequencies,
            "polarization_power": polarization_power,
            # NOT masked: `samples` is per-source `(N,)`, while the other four
            # arrays are per-frequency. Masking it would silently truncate the
            # population and change every posterior without raising.
            "samples": self.proposal.samples,
            "observed_spectral_density": observed_spectral_density,
            "effective_psd": effective_psd,
        }


def prepare_observation(
    injection_path: Path,
    *,
    fiducials: Mapping[str, float],
    grid: AnalysisGrid,
    jnp: Any,
) -> Observation:
    """Load the injection catalog and build the fiducial observed spectrum."""
    from astrogwb.frequency import frequency_mask as make_frequency_mask

    from astrogwb_paper.catalogs import (
        compute_fiducial_injection_spectrum,
        load_catalog_arrays,
        validate_catalog_samples,
    )

    fiducial_values = dict(fiducials)
    injection = load_catalog_arrays(injection_path, fiducials=fiducial_values, jnp=jnp)
    validate_catalog_samples(
        injection,
        label="injection",
        z_min=grid.z_min,
        z_max=grid.z_max,
        require_proposal_density=False,
    )
    logger.info(
        "Loaded independent injection catalog %s: n_injection_samples=%d",
        injection_path,
        injection.polarization_power.shape[1],
    )

    redshift_grid = jnp.linspace(grid.z_min, grid.z_max, grid.n_grid)
    total_merger_rate, spectral_density = compute_fiducial_injection_spectrum(
        injection,
        fiducials=fiducial_values,
        redshift_grid=redshift_grid,
        jnp=jnp,
    )
    logger.info(
        "Constructed independent fiducial observed spectrum (rate0=%.4e /s)",
        total_merger_rate,
    )

    frequency_mask = make_frequency_mask(
        injection.frequencies, fmin=grid.f_min, fmax=grid.f_max
    )
    logger.info(
        "Analysis band: %d of %d bins (%.1f-%.1f Hz)",
        int(jnp.sum(frequency_mask)),
        injection.frequencies.shape[0],
        grid.f_min,
        grid.f_max,
    )
    return Observation(
        injection=injection,
        redshift_grid=redshift_grid,
        total_merger_rate=total_merger_rate,
        spectral_density=spectral_density,
        frequency_mask=frequency_mask,
    )


def prepare_inference_inputs(
    injection_path: Path,
    proposal_path: Path,
    *,
    fiducials: Mapping[str, float],
    grid: AnalysisGrid,
    detectors: Sequence[str],
    jnp: Any,
) -> InferenceInputs:
    """Build every array the model is evaluated against, from the two catalogs."""
    from astrogwb.detector import effective_psd as compute_effective_psd
    from astrogwb.detector import load_sensitivity_map
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        make_merger_rate_and_log_weights_fn,
    )

    from astrogwb_paper.catalogs import (
        PROPOSAL_REDSHIFT_LOGPDF,
        load_catalog_arrays,
        validate_catalog_samples,
        validate_matching_frequency_grids,
    )

    observation = prepare_observation(
        injection_path, fiducials=fiducials, grid=grid, jnp=jnp
    )
    proposal = load_catalog_arrays(proposal_path, fiducials=dict(fiducials), jnp=jnp)
    validate_catalog_samples(
        proposal,
        label="proposal",
        z_min=grid.z_min,
        z_max=grid.z_max,
        require_proposal_density=True,
    )
    validate_matching_frequency_grids(observation.injection, proposal)
    n_freq, n_samples = proposal.polarization_power.shape
    logger.info(
        "Loaded proposal catalog %s: n_frequency_bins=%d n_proposal_samples=%d",
        proposal_path,
        n_freq,
        n_samples,
    )

    # Two frequency grids are in play and they are deliberately spelled
    # differently: the band mask comes from the *injection* grid (inside
    # prepare_observation, which never sees a proposal) while the effective PSD
    # comes from the *proposal* grid. validate_matching_frequency_grids has
    # already proved the two arrays equal, so both results are bit-identical --
    # do not "tidy" either one to match the other.
    sensitivities = load_sensitivity_map(detectors)
    effective_psd_arr = jnp.asarray(
        compute_effective_psd(proposal.frequencies, list(detectors), sensitivities)
    )

    merger_rate_and_log_weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=dict(fiducials),
        redshift_grid=observation.redshift_grid,
        proposal_logprob=proposal.samples[PROPOSAL_REDSHIFT_LOGPDF],
    )
    return InferenceInputs(
        observation=observation,
        proposal=proposal,
        effective_psd=effective_psd_arr,
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
    )


def initial_values(config: RunConfig) -> dict[str, float]:
    """Fiducial values NUTS initializes each sampled parameter at."""
    return {name: config.fiducials[name] for name in config.sampled_params}


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
    from functools import partial

    from astrogwb.sampling.models import (
        amplitude_marginalized_model,
        spectral_density_model,
    )

    from astrogwb_paper.amplitude import build_amplitude_marginalization

    analysis = config.analysis
    # `config.priors` already holds live distributions (see PriorDistribution).
    # Project to the sampled parameters: when marginalized, `priors` also
    # carries the amplitude parameter, which must NOT get a NUTS latent.
    priors = {name: config.priors[name] for name in config.sampled_params}
    if analysis.likelihood == "amplitude_marginalized":
        assert analysis.amplitude_parameter is not None
        marginalization = build_amplitude_marginalization(config)
        model = partial(
            amplitude_marginalized_model,
            observation_time=config.observation_time,
            average_mode="analytic_inclination",
            merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
            amplitude_parameter=analysis.amplitude_parameter,
            fiducials=config.fiducials,
            amplitude_fn=marginalization.amplitude_fn,
            amplitude_prior=marginalization.prior,
            amplitude_grid=marginalization.grid,
            priors=priors,
            constants=config.constants,
        )
        return model, marginalization

    model = partial(
        spectral_density_model,
        observation_time=config.observation_time,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        priors=priors,
        constants=config.constants,
    )
    return model, None
