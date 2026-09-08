"""One implementation of the catalogs-to-model-inputs pipeline.

Every entrypoint that samples, profiles, or plots the fiducial spectrum runs
the same sequence: load the two catalogs, restrict them to the analysis
redshift window, build the fiducial injection spectrum, build the effective
PSD, build the analysis-band mask, prepare the estimator, build the model. It
used to be spelled out at eight call sites, two of which were near-verbatim
clones of each other down to the model-building block.

The catalogs are now authoritative about their own populations, so this file no
longer derives a proposal density from the run config, no longer cross-checks
run fiducials against catalog provenance, and no longer applies a fiducial
propagation correction to stored power. Those three steps existed to reconcile
records that are now one record.

JAX ops run only inside functions, after ``runtime.configure_runtime``. Importing
this module loads ``jax`` but does not initialize the XLA backend; a subprocess
test in ``tests/paper/test_cli.py`` guards that.

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
from numpyro import handlers
from numpyro.distributions import Distribution

from astrogwb.catalog import Catalog
from astrogwb.detector import effective_psd as compute_effective_psd
from astrogwb.detector import gaussian_bin_scale, load_sensitivity_map
from astrogwb.distributions.amplitude import (
    AmplitudeFn,
    MergerRateAmplitudeFn,
    quadrature_grid,
)
from astrogwb.frequency import frequency_mask as make_frequency_mask
from astrogwb.gwb import AverageMode, spectral_density
from astrogwb.importance.estimator import SpectralDensityImportanceEstimator
from astrogwb.paper.catalogs import validate_matching_frequency_grids
from astrogwb.paper.config.mcmc import AnalysisGrid, RunConfig
from astrogwb.populations import (
    TOTAL_MERGER_RATE_SITE,
    PopulationModel,
    amplitude_H0_fn,
    amplitude_local_merger_rate_fn,
    merger_rate_H0_fn,
    merger_rate_local_merger_rate_fn,
    population_log_probs,
    population_model,
    population_sites,
    required_deterministic,
    select_stochastic_values,
)
from astrogwb.sampling import (
    SpectralDensityFn,
    gwb_amplitude_marginalized_model,
    gwb_spectral_density_model,
    with_renamed_diagnostics,
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
    total_merger_rate: jax.Array
    spectral_density: jax.Array
    frequency_mask: jax.Array


@dataclass(frozen=True)
class InferenceInputs:
    """Everything the NumPyro model is evaluated against, for one run."""

    observation: Observation
    proposal: Catalog
    effective_psd: jax.Array
    observation_time: float
    estimator: SpectralDensityImportanceEstimator
    """The masked, band-restricted catalog bound to its target population."""

    def masked_model_kwargs(self) -> dict[str, Any]:
        """Restrict to the analysis band and return the model's data inputs.

        The generic likelihood takes only the observation and the per-bin
        Gaussian scale: the catalog and the inclination convention already
        live inside :attr:`estimator`, and the PSD, observation time, and bin
        width are consumed here rather than inside the model.
        """
        observation = self.observation
        mask = np.asarray(observation.frequency_mask)
        return {
            "observed_spectral_density": observation.spectral_density[mask],
            "scale": gaussian_bin_scale(
                self.effective_psd[mask], self.observation_time, observation.df
            ),
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


def target_population_model(config: RunConfig) -> PopulationModel:
    """Resolve and bind the population the run's hyperparameters describe.

    Bound once per run and reused for every evaluation: the estimator carries
    the callable as *static* pytree metadata, and a ``functools.partial`` hashes
    by identity, so a fresh one per step would retrace the whole model.
    """
    grid = config.analysis_grid
    return partial(
        population_model(config.analysis.population_model),
        z_min=grid.minimum_redshift,
        z_max=grid.maximum_redshift,
        n_grid=grid.n_grid,
    )


def prepare_observation(injection: Catalog, *, grid: AnalysisGrid) -> Observation:
    """Build the fiducial observed spectrum from the injection catalog.

    The rate comes from the injection catalog's *own* population, evaluated
    over the analysis window: restricting the window narrows both the samples
    and the recorded density together, so the rate counts exactly the sources
    the spectrum sums over. The parameters are the catalog's, not the run's --
    the file records what was actually injected, which is why nothing has to
    check the two against each other any more.

    Weights are identically one. This is the observed data, not a reweighting,
    and keeping it independent of the estimator is what lets a bug in the
    weights show up as a mismatch rather than cancel out of both sides.
    """
    n_loaded = injection.polarization_power.shape[1]
    restricted = injection.restrict_redshift(
        grid.minimum_redshift, grid.maximum_redshift
    )
    n_kept = restricted.polarization_power.shape[1]
    logger.info(
        "Loaded independent injection catalog: n_injection_samples=%d "
        "(%d outside the analysis window dropped)",
        n_kept,
        n_loaded - n_kept,
    )

    total_merger_rate = catalog_total_merger_rate(restricted, label="injection catalog")
    power = jnp.asarray(restricted.polarization_power)
    spectrum = spectral_density(
        power,
        jnp.ones(power.shape[1]),
        total_merger_rate,
        average_mode="analytic_inclination",
    )
    logger.info(
        "Constructed independent fiducial observed spectrum (rate0=%.4e /s)",
        total_merger_rate,
    )

    frequencies = jnp.asarray(restricted.waveform_metadata.frequencies)
    # Band bounds only: this function never sees a detector network, so bins
    # the network cannot measure are dropped later, in prepare_inference_inputs.
    analysis_frequency_mask = make_frequency_mask(
        frequencies, fmin=grid.f_min, fmax=grid.f_max
    )
    logger.info(
        "Analysis band: %d of %d bins (%.1f-%.1f Hz)",
        int(jnp.sum(analysis_frequency_mask)),
        frequencies.shape[0],
        grid.f_min,
        grid.f_max,
    )
    return Observation(
        frequencies=frequencies,
        df=float(restricted.waveform_metadata.df),
        total_merger_rate=total_merger_rate,
        spectral_density=spectrum,
        frequency_mask=analysis_frequency_mask,
    )


def catalog_total_merger_rate(catalog: Catalog, *, label: str) -> jax.Array:
    """The observer-frame total merger rate this catalog's population implies.

    Recomputed from the recorded model rather than read from a stored column:
    the rate is a property of the population and the redshift window, so a
    stored copy would be stale the moment the window is narrowed.
    """
    model = catalog.get_population_model()
    params = catalog.population_params
    sites = population_sites(model, params)
    values = select_stochastic_values(catalog.source_parameters, sites, label=label)
    _, trace = population_log_probs(model, params, values)
    return required_deterministic(trace, TOTAL_MERGER_RATE_SITE, ndim=0, label=label)


def prepare_inference_inputs(
    injection: Catalog,
    proposal: Catalog,
    *,
    grid: AnalysisGrid,
    detectors: Sequence[str],
    target_model: PopulationModel,
    target_params: Mapping[str, float],
    average_mode: AverageMode = "analytic_inclination",
) -> InferenceInputs:
    """Build every array the model is evaluated against, from the two catalogs.

    ``average_mode`` is the inclination convention the spectrum contraction
    uses. It belongs here rather than at the model-building sites because the
    estimator owns it: once the catalog is bound, the model itself never sees
    a polarization power array to average.

    ``target_params`` is a representative hyperparameter point -- the run's
    fiducials -- used only to discover which sites the target model declares
    and to check they agree with the proposal's. It is *not* the point the
    spectrum is evaluated at; sampled parameters arrive through the estimator's
    own call. It is needed because the target reads parameters the catalog does
    not record: the propagation law is target-side only.
    """
    observation = prepare_observation(injection, grid=grid)

    n_loaded = proposal.polarization_power.shape[1]
    proposal_catalog = proposal.restrict_redshift(
        grid.minimum_redshift, grid.maximum_redshift
    )
    proposal_frequencies = np.asarray(proposal_catalog.waveform_metadata.frequencies)
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

    # The proposal density is the catalog's own recorded population, evaluated
    # at the parameters it was drawn at. Doing that here, once, is also what
    # keeps it off the per-sampler-step path.
    estimator = SpectralDensityImportanceEstimator.from_catalog(
        proposal_catalog,
        model=target_model,
        target_params=target_params,
        average_mode=average_mode,
        frequency_mask=band_mask,
    )
    return InferenceInputs(
        observation=observation,
        proposal=proposal_catalog,
        effective_psd=effective_psd_arr,
        observation_time=grid.observation_time,
        estimator=estimator,
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
    spectral_density_fn: SpectralDensityFn,
) -> tuple[Any, AmplitudeMarginalization | None]:
    """Build the NumPyro model this config describes.

    Returns ``(model, marginalization)``, where ``marginalization`` is the
    :class:`~astrogwb.paper.inference.AmplitudeMarginalization` built for an
    amplitude-marginalized run, or ``None`` for the default likelihood.

    Depends only on the config and the spectrum callable -- no catalog array --
    so it is exercisable without generating one. Production runs pass
    ``inputs.estimator``; an analytic spectrum works just as well.
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
                gwb_amplitude_marginalized_model,
                # The spectrum is evaluated at the pinned fiducial amplitude,
                # so its rate is the *template* rate. Renaming it here is what
                # keeps a template quantity out of the `total_merger_rate`
                # site that post-processing reconstructs.
                spectral_density_fn=with_renamed_diagnostics(
                    spectral_density_fn,
                    {"total_merger_rate": "template_merger_rate"},
                ),
                amplitude_parameter=parameter,
                amplitude_fiducial=marginalization.fiducial,
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
            gwb_spectral_density_model,
            spectral_density_fn=spectral_density_fn,
            priors=priors,
        ),
        config.fixed_params,
    )
    return model, None
