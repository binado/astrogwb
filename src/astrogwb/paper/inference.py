"""One implementation of the catalogs-to-model-inputs pipeline.

Every entrypoint that samples, profiles, or plots the fiducial spectrum runs
the same sequence: load the two catalogs, restrict them to the analysis
redshift window, build the fiducial injection spectrum, build the effective
PSD, build the analysis-band mask, bind the proposal catalog to the target,
build the model. It used to be spelled out at eight call sites, two of which
were near-verbatim clones of each other down to the model-building block.

The catalogs are now authoritative about their own populations, so this file no
longer derives a proposal density from the run config, no longer cross-checks
run fiducials against catalog provenance, and no longer applies a fiducial
propagation correction to stored power. Those three steps existed to reconcile
records that are now one record.

JAX ops run only inside functions, after ``runtime.configure_runtime``. Importing
this module loads ``jax`` but does not initialize the XLA backend; a subprocess
test in ``tests/paper/test_cli.py`` guards that.

Nothing here compresses an array to the analysis band. The catalog's frequency
grid is *the* grid every model array lives on -- the observed spectrum, the
scale, and the bound ``(F, N)`` polarization power alike -- and the band reaches
the model as a boolean mask carried alongside them, which
:meth:`InferenceInputs.model_kwargs` assembles. That is what lets one compiled
sampler be reused across bands: a mask is a traced value, while a compressed
array is a new shape and therefore a new compilation. The catalogs are generated
on the analysis band (see ``config/waveform.json``), so the grid
costs no more per sampler step than the band it selects.
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

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.detector import effective_psd as compute_effective_psd
from astrogwb.detector import gaussian_bin_scale, load_sensitivity_map
from astrogwb.distributions.amplitude import (
    AmplitudeFn,
    MergerRateAmplitudeFn,
    quadrature_grid,
)
from astrogwb.frequency import frequency_mask as make_frequency_mask
from astrogwb.gwb import spectral_density
from astrogwb.importance.spectral import LogWeightsFn, build_importance_spectrum
from astrogwb.paper.catalogs import validate_matching_frequency_grids
from astrogwb.paper.config.mcmc import AnalysisGrid, RunConfig
from astrogwb.populations import (
    Population,
    amplitude_H0_fn,
    amplitude_local_merger_rate_fn,
    build_population,
    merger_rate_H0_fn,
    merger_rate_local_merger_rate_fn,
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

    Arrays are on the catalog's full frequency grid; ``frequency_mask`` selects
    the analysis band within it. ``df`` is the catalog's grid-derived bin width
    and stays valid under any mask, which is why it is carried here rather than
    measured off a selected grid.
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
    proposal: PolarizationPowerCatalog
    effective_psd: jax.Array
    observation_time: float
    spectral_density_fn: SpectralDensityFn
    """The importance-sampled spectrum: the catalog's full-grid arrays bound to
    the target source model and merger rate."""
    log_weights_fn: LogWeightsFn
    """Per-source log importance weights, bound to the same arrays and target."""

    def model_kwargs(
        self, *, fmin: float | None = None, fmax: float | None = None
    ) -> dict[str, Any]:
        """The model's data inputs, on the catalog grid, with the band as a mask.

        The likelihoods take the observation, the per-bin Gaussian scale, and
        the boolean band mask: the catalog and the inclination convention
        already live inside :attr:`spectral_density_fn`, and the PSD,
        observation time, and bin width are consumed here rather than inside
        the model.

        ``fmin``/``fmax`` narrow the run's band to a sub-band, intersected with
        the run's own mask so detector-coverage gaps stay excluded. Because
        only the mask's *value* changes, a sweep over sub-bands reuses one
        compiled sampler -- build the inputs and the ``MCMC`` object once, then
        call ``mcmc.run(key, **inputs.model_kwargs(fmax=f))`` per band.

        Raises ``ValueError`` if fewer than two bins survive.
        """
        observation = self.observation
        mask = observation.frequency_mask
        if fmin is not None or fmax is not None:
            mask = mask & make_frequency_mask(
                observation.frequencies, fmin=fmin, fmax=fmax
            )
            num_bins = int(jnp.sum(mask))
            if num_bins < 2:
                raise ValueError(
                    f"only {num_bins} usable frequency bin(s) in the requested "
                    f"sub-band [{fmin}, {fmax}] Hz; widen it or check that it "
                    "lies inside the run's analysis band"
                )
        scale = gaussian_bin_scale(
            self.effective_psd, self.observation_time, observation.df
        )
        return {
            "observed_spectral_density": observation.spectral_density,
            # `effective_psd` is inf wherever no detector pair contributes, so
            # `scale` is non-finite at exactly the bins the run's band already
            # excludes. Both likelihoods discard those bins, which makes the
            # substituted value unobservable; it exists so the array stays
            # finite and inspectable rather than relying on a non-finite value
            # surviving a `where`. The substitution keys off the run's band and
            # not off `mask`, so `scale` is one fixed array across a sub-band
            # sweep and the mask is the only thing that varies.
            "scale": jnp.where(observation.frequency_mask, scale, 1.0),
            "frequency_mask": mask,
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


def target_population(config: RunConfig) -> Population:
    """Resolve and bind the population the run's hyperparameters describe.

    Build once per run and reuse: the returned partials hash by identity, so
    an equal rebuild under a jit-cached call forces a recompile.
    Hyperparameters remain their call argument.

    An analysis target must declare a merger rate: the predicted spectrum is
    normalized by one, so a proposal density named here would produce a
    spectrum with no scale rather than an error.
    ``check_population_model`` rejects it in pre-flight; this is the same
    refusal at the point of use.
    """
    declared = config.analysis.population
    population = build_population(declared.model_name, **declared.model_kwargs)
    if population.merger_rate_fn is None:
        raise ValueError(
            f"analysis.population.model_name {declared.model_name!r} declares "
            "no merger rate, so it cannot be an analysis target; it is a "
            "proposal density"
        )
    return population


def prepare_observation(
    injection: PolarizationPowerCatalog, *, grid: AnalysisGrid
) -> Observation:
    """Build the fiducial observed spectrum from the injection catalog.

    The rate comes from the injection catalog's *own* population, evaluated
    over the analysis window: restricting the window narrows both the samples
    and the recorded density together, so the rate counts exactly the sources
    the spectrum sums over. The parameters are the catalog's, not the run's --
    the file records what was actually injected, which is why nothing has to
    check the two against each other any more.

    Weights are identically one. This is the observed data, not a reweighting,
    and keeping it independent of the importance weights is what lets a bug in
    the weights show up as a mismatch rather than cancel out of both sides.
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

    total_merger_rate = catalog_total_merger_rate(restricted)
    power = jnp.asarray(restricted.polarization_power)
    spectrum = spectral_density(
        power,
        jnp.ones(power.shape[1]),
        total_merger_rate,
        source_parameters=restricted.source_parameters,
    )
    logger.info(
        "Constructed independent fiducial observed spectrum (rate0=%.4e /s)",
        total_merger_rate,
    )

    frequencies = jnp.asarray(restricted.frequencies)
    # Band bounds only: this function never sees a detector network, so bins
    # the network cannot measure are dropped later, in prepare_inference_inputs.
    analysis_frequency_mask = make_frequency_mask(
        frequencies, fmin=grid.minimum_frequency, fmax=grid.maximum_frequency
    )
    logger.info(
        "Analysis band: %d of %d bins (%.1f-%.1f Hz)",
        int(jnp.sum(analysis_frequency_mask)),
        frequencies.shape[0],
        grid.minimum_frequency,
        grid.maximum_frequency,
    )
    return Observation(
        frequencies=frequencies,
        df=float(restricted.df),
        total_merger_rate=total_merger_rate,
        spectral_density=spectrum,
        frequency_mask=analysis_frequency_mask,
    )


def catalog_total_merger_rate(catalog: PolarizationPowerCatalog) -> jax.Array:
    """The observer-frame total merger rate this catalog's population implies.

    Recomputed from the recorded population rather than read from a stored
    column: the rate is a property of the population and the redshift window,
    so a stored copy would be stale the moment the window is narrowed.

    A catalog drawn from a proposal density has no such rate. A guard mixture
    is not a physical population, and the Madau-Dickinson total rate is the
    normalization of the Madau-Dickinson redshift density, not of a mixture of
    it with a uniform component -- so this raises rather than returning a
    number that would silently scale an observed spectrum by the wrong factor.
    """
    merger_rate_fn = catalog.get_population().merger_rate_fn
    if merger_rate_fn is None:
        raise ValueError(
            f"catalog population {catalog.population_model_name!r} declares no "
            "merger rate, so it cannot supply an observed total rate; it is a "
            "proposal density, not an injection"
        )
    rate = merger_rate_fn(catalog.fiducials)
    return jnp.reshape(jnp.asarray(rate), ())


def prepare_inference_inputs(
    injection: PolarizationPowerCatalog,
    proposal: PolarizationPowerCatalog,
    *,
    grid: AnalysisGrid,
    detectors: Sequence[str],
    target: Population,
    density_sites: Sequence[str],
) -> InferenceInputs:
    """Build every array the model is evaluated against, from the two catalogs.

    ``target`` is the run's bound target population, normally
    :func:`target_population` of the run config; build it once per run. It must
    declare a merger rate: the predicted spectrum is normalized by one, so a
    proposal density here would produce a spectrum with no scale.

    ``density_sites`` names the source-density factors the importance weights
    include, and is passed straight through to
    :func:`~astrogwb.importance.build_importance_spectrum`. It is an analysis
    input rather than something read off the proposal: the catalog's samples do
    not depend on which of their densities are counted.
    """
    if target.merger_rate_fn is None:
        raise ValueError(
            "the target population declares no merger rate, so it cannot "
            "normalize a predicted spectrum; it is a proposal density"
        )
    observation = prepare_observation(injection, grid=grid)

    n_loaded = proposal.polarization_power.shape[1]
    proposal_catalog = proposal.restrict_redshift(
        grid.minimum_redshift, grid.maximum_redshift
    )
    proposal_frequencies = np.asarray(proposal_catalog.frequencies)
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
    # usable diagnostic. Exclude those bins along with the out-of-band ones. This
    # is safe precisely because `df` is the catalog's grid-derived property: the
    # selected bins need not be contiguous, and each still has width `df`.
    band_mask = (
        observation.frequency_mask
        & jnp.isfinite(effective_psd_arr)
        & (effective_psd_arr > 0.0)
    )
    num_bins = int(jnp.sum(band_mask))
    if num_bins < 2:
        raise ValueError(
            f"only {num_bins} usable frequency bin(s) in "
            f"[{grid.minimum_frequency}, {grid.maximum_frequency}] Hz for "
            f"detectors "
            f"{' '.join(detectors)}; widen the band or choose a detector "
            "network with full coverage"
        )
    excluded = int(jnp.sum(observation.frequency_mask)) - num_bins
    if excluded:
        logger.info(
            "Excluded %d in-band bin(s) with no detector-network coverage", excluded
        )
    observation = replace(observation, frequency_mask=band_mask)

    # The proposal density is the catalog's own recorded source model,
    # evaluated at the parameters it was drawn at. Doing that here, once, is
    # also what keeps it off the per-sampler-step path.
    #
    # No `frequency_mask`: the power stays on the catalog's full grid, so the
    # band is a traced mask rather than a compiled-in shape. `model_kwargs`
    # supplies it alongside arrays of the same length.
    spectral_density_fn, log_weights_fn = build_importance_spectrum(
        proposal_catalog,
        source_model=target.source_model,
        merger_rate_fn=target.merger_rate_fn,
        density_sites=density_sites,
    )
    return InferenceInputs(
        observation=observation,
        proposal=proposal_catalog,
        effective_psd=effective_psd_arr,
        observation_time=grid.observation_time,
        spectral_density_fn=spectral_density_fn,
        log_weights_fn=log_weights_fn,
    )


def initial_values(config: RunConfig) -> dict[str, float]:
    """Fiducial values NUTS initializes each sampled parameter at."""
    return {name: config.fiducials[name] for name in config.analysis.sampled_params}


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
    ``inputs.spectral_density_fn``; an analytic spectrum works just as well.
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
