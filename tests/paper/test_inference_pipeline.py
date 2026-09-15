"""End-to-end coverage of the shared inference-input pipeline.

These tests build the smallest synthetic catalog pair that survives validation
and exercise the application-library preparation shared by pipeline scripts.

The catalogs are deliberately tiny and their frequency grid deliberately
straddles the analysis band, so the frequency slice actually selects a strict
subset -- which is what makes the "source samples must NOT be masked"
assertion meaningful.

Three steps that used to live here are gone, and their tests with them: a
proposal density derived from the run config, a check reconciling the run's
fiducials against the catalog's provenance, and a fiducial propagation
correction applied to stored power and patched into the reference distance at
the call site. Each catalog now records the density that drew it, and the
propagation law is part of the population declaration, so the reference
distance is simply the distance column the file already holds.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from catalog_fixtures import (
    PAPER_POPULATION_PARAMS,
    make_catalog,
)
from config_fixtures import example_raw
from numpyro.infer.util import log_density
from repo import REPO_ROOT

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.detector import gaussian_bin_scale
from astrogwb.gwb import spectral_density
from astrogwb.importance.spectral import build_importance_spectrum
from astrogwb.paper.catalogs import load_run_catalog
from astrogwb.paper.config.mcmc import RunConfig, build_run_config
from astrogwb.paper.inference import (
    build_model,
    catalog_total_merger_rate,
    prepare_inference_inputs,
    prepare_observation,
    target_population,
)
from astrogwb.sampling import gwb_spectral_density_model

pytestmark = pytest.mark.integration

# Uniform grid (df = 10 Hz); the band below keeps the middle three bins.
# Model arrays live on the whole grid -- `N_FREQ` long -- and the band reaches
# the model as a mask selecting `N_BAND` of those bins.
FREQUENCIES = np.linspace(10.0, 50.0, 5)
N_FREQ = FREQUENCIES.size
BAND = (20.0, 40.0)
N_BAND = 3
N_SOURCES = 8
# Catalogs span below the assembled analysis window (minimum_redshift 0.3):
# linspace(0.05, 1.5, 8) keeps 6 samples after restriction.
N_RETAINED = 6
REDSHIFT = np.linspace(0.05, 1.5, N_SOURCES)

#: The catalog's own generation window and grid, narrower in resolution than
#: the analysis grid the target runs on. Keeping them different is deliberate:
#: it is what a real run does, and it stops a bug that conflates the two from
#: cancelling out of both sides.
GENERATION_KWARGS: dict[str, float | int] = {
    "z_min": 0.0,
    "z_max": 20.0,
    "n_grid": 256,
}
GUARD_FRACTION = 0.3


def _write_catalog(
    path: Path,
    *,
    seed: int,
    frequencies: np.ndarray = FREQUENCIES,
    guarded: bool = False,
) -> Path:
    rng = np.random.default_rng(seed)
    kwargs = dict(GENERATION_KWARGS)
    if guarded:
        kwargs["uniform_mixing_fraction"] = GUARD_FRACTION
    catalog = make_catalog(
        redshift=REDSHIFT,
        polarization_power=rng.uniform(0.0, 1.0, size=(frequencies.size, N_SOURCES)),
        minimum_frequency=float(frequencies[0]),
        df=float(frequencies[1] - frequencies[0]),
        seed=seed,
        model_name=("bns_md_uniform_mixture" if guarded else "bns_md_cosmological"),
        model_kwargs=kwargs,
    )
    catalog.save(path)
    return path


@pytest.fixture
def injection_catalog(tmp_path: Path) -> PolarizationPowerCatalog:
    return load_run_catalog(
        _write_catalog(tmp_path / "injection.h5", seed=0), label="injection"
    )


@pytest.fixture
def proposal_catalog(tmp_path: Path) -> PolarizationPowerCatalog:
    return load_run_catalog(
        _write_catalog(tmp_path / "proposal.h5", seed=1), label="proposal"
    )


def _config(**overrides: Any) -> RunConfig:
    raw = example_raw()
    f_min, f_max = BAND
    raw["analysis"] = {**raw["analysis"], "f_min": f_min, "f_max": f_max}
    raw["cosmology"] = {**raw["cosmology"], "n_grid": 32}
    raw["sampler"] = {
        **raw["sampler"],
        "num_warmup": 2,
        # 4 is the floor: `run` ends in mcmc.print_summary(), whose split R-hat
        # asserts at least four draws per chain.
        "num_samples": 4,
        "num_chains": 1,
        "progress_bar": False,
    }
    raw.update(overrides)
    return build_run_config(raw)


def _prepare(
    injection: PolarizationPowerCatalog,
    proposal: PolarizationPowerCatalog,
    config: RunConfig,
):
    return prepare_inference_inputs(
        injection,
        proposal,
        grid=config.analysis_grid,
        detectors=config.analysis.detectors,
        target=target_population(config),
    )


# --------------------------------------------------------------------------- #
# prepare_observation / prepare_inference_inputs
# --------------------------------------------------------------------------- #
def test_prepare_observation_keeps_arrays_unmasked(
    injection_catalog: PolarizationPowerCatalog,
) -> None:
    config = _config()

    observation = prepare_observation(injection_catalog, grid=config.analysis_grid)

    assert observation.frequencies.shape == FREQUENCIES.shape
    assert observation.spectral_density.shape == FREQUENCIES.shape
    assert float(observation.total_merger_rate) > 0.0
    # The mask is carried alongside, not applied: notebooks plot the full band.
    assert observation.df == 10.0
    np.testing.assert_array_equal(
        np.asarray(observation.frequency_mask), [False, True, True, True, False]
    )
    np.testing.assert_allclose(
        np.asarray(observation.frequencies)[np.asarray(observation.frequency_mask)],
        [20.0, 30.0, 40.0],
    )


def test_the_observed_rate_comes_from_the_injection_catalogs_own_population(
    injection_catalog: PolarizationPowerCatalog,
) -> None:
    """Nothing is cross-checked against the run config any more, so nothing may
    be *read* from it either: the file records what was injected."""
    from reference_population import reference_merger_rate_distance_and_logprob

    config = _config()
    observation = prepare_observation(injection_catalog, grid=config.analysis_grid)

    grid = config.analysis_grid
    restricted = injection_catalog.restrict_redshift(
        grid.minimum_redshift, grid.maximum_redshift
    )
    expected_rate, _, _ = reference_merger_rate_distance_and_logprob(
        PAPER_POPULATION_PARAMS,
        jnp.asarray(restricted.source_parameters["redshift"]),
        redshift_grid=jnp.linspace(
            grid.minimum_redshift,
            grid.maximum_redshift,
            int(restricted.population_model_kwargs["n_grid"]),
        ),
    )
    np.testing.assert_allclose(
        float(observation.total_merger_rate), float(expected_rate), rtol=1e-12
    )
    # The weights are identically one, so the observation is the plain
    # unweighted contraction of the restricted power.
    np.testing.assert_allclose(
        np.asarray(observation.spectral_density),
        0.4
        * float(expected_rate)
        * np.asarray(restricted.polarization_power).mean(axis=1),
        rtol=1e-12,
    )


def _bound(inputs: Any) -> dict[str, Any]:
    """The keywords ``prepare_inference_inputs`` bound into the spectrum partial."""
    return inputs.spectral_density_fn.keywords


def test_prepared_spectrum_keeps_the_full_grid_and_all_samples(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    """Nothing is compressed: the band is a mask over the catalog's own grid.

    Compressing would bake the bin count into every compiled program, so a
    second band would cost a recompile. These shapes are what make one
    compiled sampler reusable across bands.
    """
    config = _config()

    inputs = _prepare(injection_catalog, proposal_catalog, config)
    kwargs = inputs.model_kwargs()
    bound = _bound(inputs)

    assert kwargs["observed_spectral_density"].shape == (N_FREQ,)
    assert kwargs["scale"].shape == (N_FREQ,)
    assert kwargs["frequency_mask"].shape == (N_FREQ,)
    assert int(jnp.sum(kwargs["frequency_mask"])) == N_BAND
    assert bound["polarization_power"].shape == (N_FREQ, N_RETAINED)
    # The builder is handed no mask at all -- the band never reaches it.
    assert "frequency_mask" not in bound
    # `source_parameters` is per-source, not per-frequency. Masking it would
    # silently truncate the population and change every posterior without
    # erroring; the cached proposal arrays follow the same axis.
    for name, values in bound["source_parameters"].items():
        assert values.shape == (N_RETAINED,), name
    assert bound["proposal_log_prob"].shape == (N_RETAINED,)
    assert bound["log_reference_distance"].shape == (N_RETAINED,)
    # The weights see exactly the arrays and density factors the spectrum does.
    weights_bound = inputs.log_weights_fn.keywords
    for name, value in weights_bound.items():
        assert bound[name] is value, name
    # The likelihood takes data and the band only: the catalog and the
    # averaging convention travel on the bound spectrum, and the bin width is
    # consumed into `scale`.
    assert set(kwargs) == {"observed_spectral_density", "scale", "frequency_mask"}


def test_restriction_narrows_the_proposals_recorded_population_too(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    """Dropping samples without narrowing the density would misnormalize it."""
    config = _config()

    inputs = _prepare(injection_catalog, proposal_catalog, config)

    assert inputs.proposal.num_samples == N_RETAINED
    assert inputs.proposal.population_model_kwargs["z_min"] == (
        config.cosmology.minimum_redshift
    )
    assert inputs.proposal.population_model_kwargs["z_max"] == (
        config.cosmology.maximum_redshift
    )
    # The file on disk is untouched.
    assert proposal_catalog.population_model_kwargs["z_min"] == 0.0


def test_model_kwargs_scale_is_the_full_grid_gaussian_bin_scale(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    """The scale is prepared here, not derived inside the sampling model."""
    config = _config()

    inputs = _prepare(injection_catalog, proposal_catalog, config)
    mask = np.asarray(inputs.observation.frequency_mask)
    expected = np.asarray(
        gaussian_bin_scale(
            inputs.effective_psd,
            config.analysis_grid.observation_time,
            # The catalog's bin width, never measured off the selected band.
            10.0,
        )
    )

    actual = np.asarray(inputs.model_kwargs()["scale"])

    assert actual.shape == (N_FREQ,)
    np.testing.assert_allclose(actual[mask], expected[mask])


def test_mismatched_frequency_grids_are_rejected(
    injection_catalog: PolarizationPowerCatalog, tmp_path: Path
) -> None:
    shifted = _write_catalog(
        tmp_path / "shifted.h5", seed=2, frequencies=FREQUENCIES + 10.0
    )
    config = _config()

    with pytest.raises(ValueError, match="identical frequency grids"):
        _prepare(injection_catalog, load_run_catalog(shifted, label="proposal"), config)


@pytest.mark.parametrize("uncovered", [0.0, np.inf])
def test_bins_without_network_coverage_narrow_the_band(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
    monkeypatch: pytest.MonkeyPatch,
    uncovered: float,
) -> None:
    """An uncovered bin is dropped, not fatal -- each survivor still has width df."""
    config = _config()
    effective_noise = np.ones(FREQUENCIES.shape)
    effective_noise[2] = uncovered
    monkeypatch.setattr(
        "astrogwb.paper.inference.compute_effective_psd",
        lambda *_args, **_kwargs: effective_noise,
    )

    inputs = _prepare(injection_catalog, proposal_catalog, config)
    kwargs = inputs.model_kwargs()

    # The band was [20, 30, 40] Hz; 30 Hz is uncovered, so the surviving band is
    # gappy -- which is only sound because `df` is the catalog's attribute.
    np.testing.assert_array_equal(
        np.asarray(inputs.observation.frequency_mask),
        [False, True, False, True, False],
    )
    # The arrays keep the catalog's length; only the mask records the gap.
    assert kwargs["scale"].shape == (N_FREQ,)
    assert kwargs["observed_spectral_density"].shape == (N_FREQ,)
    assert _bound(inputs)["polarization_power"].shape == (N_FREQ, N_RETAINED)
    np.testing.assert_array_equal(
        np.asarray(kwargs["frequency_mask"]), [False, True, False, True, False]
    )
    # An uncovered bin's scale is `inf` (or a division by zero); both
    # likelihoods discard it, and `model_kwargs` substitutes a finite
    # placeholder so nothing downstream has to survive a non-finite value.
    assert bool(jnp.all(jnp.isfinite(kwargs["scale"])))


def test_a_band_with_fewer_than_two_usable_bins_is_rejected(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    effective_noise = np.full(FREQUENCIES.shape, np.inf)
    effective_noise[1] = 1.0
    monkeypatch.setattr(
        "astrogwb.paper.inference.compute_effective_psd",
        lambda *_args, **_kwargs: effective_noise,
    )

    with pytest.raises(ValueError, match="only 1 usable frequency bin"):
        _prepare(injection_catalog, proposal_catalog, config)


def test_a_sub_band_narrows_the_mask_without_changing_any_shape(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    """The whole point: a second band is a new mask value, not a new shape."""
    config = _config()

    inputs = _prepare(injection_catalog, proposal_catalog, config)
    full = inputs.model_kwargs()
    narrowed = inputs.model_kwargs(fmax=30.0)

    # The band was [20, 30, 40] Hz; capping at 30 Hz keeps the first two.
    np.testing.assert_array_equal(
        np.asarray(narrowed["frequency_mask"]), [False, True, True, False, False]
    )
    for name, array in narrowed.items():
        assert array.shape == full[name].shape, name
    # Everything but the mask is untouched, so only a traced value differs.
    for name in ("observed_spectral_density", "scale"):
        np.testing.assert_array_equal(
            np.asarray(narrowed[name]), np.asarray(full[name])
        )


def test_a_sub_band_is_intersected_with_the_runs_own_band(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    """Bounds widen nothing: bins the run already excluded stay excluded."""
    config = _config()

    inputs = _prepare(injection_catalog, proposal_catalog, config)

    # [0, 100] Hz spans the whole grid, but the run's band is [20, 40] Hz.
    np.testing.assert_array_equal(
        np.asarray(inputs.model_kwargs(fmin=0.0, fmax=100.0)["frequency_mask"]),
        np.asarray(inputs.observation.frequency_mask),
    )


def test_a_sub_band_with_fewer_than_two_usable_bins_is_rejected(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    config = _config()

    inputs = _prepare(injection_catalog, proposal_catalog, config)

    with pytest.raises(ValueError, match="only 1 usable frequency bin"):
        inputs.model_kwargs(fmin=20.0, fmax=25.0)


def test_the_marginalized_likelihood_reads_the_band_off_the_mask_too(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    """The other production likelihood, end to end through ``build_model``.

    The amplitude-marginalized model reaches the band through four hand-written
    sums rather than through a distribution, so its masking is separate code
    from the general model's and needs its own end-to-end check: a mask that
    reached the factor but not the sufficient statistics would leave a chain
    that looks healthy and reconstructs the wrong posterior.
    """
    config = _config(
        analysis={
            **example_raw()["analysis"],
            "f_min": BAND[0],
            "f_max": BAND[1],
            "likelihood": "amplitude_marginalized",
            "amplitude_parameter": "H0",
        },
        sampled_params=["Omega_m"],
    )

    inputs = _prepare(injection_catalog, proposal_catalog, config)
    model, marginalization = build_model(
        config, spectral_density_fn=inputs.spectral_density_fn
    )
    assert marginalization is not None
    params = {"Omega_m": jnp.asarray(config.fiducials["Omega_m"])}

    value, _ = log_density(model, (), inputs.model_kwargs(), params)

    # The same likelihood with the band compressed away instead of masked.
    mask = np.asarray(inputs.observation.frequency_mask)
    target = target_population(config)
    assert target.merger_rate_fn is not None
    compressed_model, _ = build_model(
        config,
        spectral_density_fn=build_importance_spectrum(
            inputs.proposal,
            source_model=target.source_model,
            merger_rate_fn=target.merger_rate_fn,
            frequency_mask=inputs.observation.frequency_mask,
        )[0],
    )
    kwargs = inputs.model_kwargs()
    expected, _ = log_density(
        compressed_model,
        (),
        {
            "observed_spectral_density": kwargs["observed_spectral_density"][mask],
            "scale": kwargs["scale"][mask],
        },
        params,
    )

    assert np.isfinite(float(value))
    np.testing.assert_allclose(float(value), float(expected), rtol=1e-10)


def test_a_catalog_may_serve_as_both_roles(
    injection_catalog: PolarizationPowerCatalog,
) -> None:
    """Injection versus proposal is two filenames in a TOML, nothing more."""
    config = _config()

    inputs = _prepare(injection_catalog, injection_catalog, config)

    assert _bound(inputs)["polarization_power"].shape == (N_FREQ, N_RETAINED)


# --------------------------------------------------------------------------- #
# Parity: the prepared spectrum against the grid-level formula
#
# The reference distance is the catalog's own stored distance column, and the
# target's distance includes modified propagation. Getting that pairing wrong
# biases every weight by xi^2 while leaving them finite and plausible, so it is
# checked against a formula written out in full rather than against the
# pipeline's own intermediates.
# --------------------------------------------------------------------------- #
def _non_gr_config(**overrides: Any) -> RunConfig:
    """A config whose propagation parameters are deliberately *not* GR.

    The shipped fiducials have ``xi_0 == 1``, which makes every GW/EM ratio
    exactly 1 -- so a parity test on them would pass whether the propagation
    correction is applied, omitted, or applied twice.
    """
    return _config(
        fiducials={**example_raw()["fiducials"], "xi_0": 1.7, "xi_n": 2.3},
        **overrides,
    )


def test_the_reference_distance_is_the_stored_distance_of_the_stored_power(
    injection_catalog: PolarizationPowerCatalog,
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    config = _non_gr_config()

    inputs = _prepare(injection_catalog, proposal_catalog, config)
    stored = np.asarray(inputs.proposal.source_parameters["luminosity_distance"])

    np.testing.assert_array_equal(
        np.asarray(_bound(inputs)["log_reference_distance"]), np.log(stored)
    )
    # The target's distance is *not* that one at these parameters, which is
    # what makes the equality above worth asserting.
    target_distance = np.exp(
        np.asarray(_bound(inputs)["log_reference_distance"])
        + np.asarray(
            log_gw_em_ratio(
                inputs.proposal.source_parameters["redshift"],
                config.fiducials["xi_0"],
                config.fiducials["xi_n"],
            )
        )
    )
    assert not np.allclose(target_distance, stored)


def _proposal_log_prob(catalog: PolarizationPowerCatalog) -> jax.Array:
    """The proposal density, restated from the grid formula the file implies."""
    from reference_population import reference_merger_rate_distance_and_logprob

    kwargs = catalog.population_model_kwargs
    grid = jnp.linspace(
        float(kwargs["z_min"]), float(kwargs["z_max"]), int(kwargs["n_grid"])
    )
    redshift = jnp.asarray(catalog.source_parameters["redshift"])
    _, _, md_logprob = reference_merger_rate_distance_and_logprob(
        catalog.fiducials,
        redshift,
        redshift_grid=grid,
        source_frame_mass_1=catalog.source_parameters["source_frame_mass_1"],
        source_frame_mass_2=catalog.source_parameters["source_frame_mass_2"],
    )
    if catalog.population_model_name != "bns_md_uniform_mixture":
        return md_logprob
    # The mixture acts only on redshift; add the ordered-pair factor after
    # mixing rather than weighting it as if the uniform component included masses.
    _, _, redshift_logprob = reference_merger_rate_distance_and_logprob(
        catalog.fiducials, redshift, redshift_grid=grid
    )
    mass_logprob = md_logprob - redshift_logprob
    epsilon = float(kwargs["uniform_mixing_fraction"])
    return (
        jnp.logaddexp(
            jnp.log1p(-epsilon) + redshift_logprob,
            jnp.log(epsilon) - jnp.log(float(kwargs["z_max"]) - float(kwargs["z_min"])),
        )
        + mass_logprob
    )


def _grid_formula_spectrum(inputs: Any, config: RunConfig, params: dict) -> jax.Array:
    """The predicted spectrum, restated from the hand-written grid formula.

    Independent of the source model and the bound spectrum: every step -- the
    target density, the effective target distance, the reference distance, the
    weight ratio, the contraction -- is written out here, so an expectation
    cannot agree with the pipeline by construction.
    """
    from reference_population import reference_merger_rate_distance_and_logprob

    catalog = inputs.proposal
    # The bound spectrum is on the catalog's full grid; the band is applied by
    # the likelihood's mask, not by compressing the power.
    power = jnp.asarray(catalog.polarization_power)
    redshift = jnp.asarray(catalog.source_parameters["redshift"])
    grid = config.analysis_grid

    rate, distance, logprob = reference_merger_rate_distance_and_logprob(
        params,
        redshift,
        redshift_grid=jnp.linspace(
            grid.minimum_redshift, grid.maximum_redshift, grid.n_grid
        ),
        source_frame_mass_1=catalog.source_parameters["source_frame_mass_1"],
        source_frame_mass_2=catalog.source_parameters["source_frame_mass_2"],
    )
    log_target_distance = jnp.log(distance) + log_gw_em_ratio(
        redshift, params["xi_0"], params["xi_n"]
    )
    log_reference_distance = jnp.log(
        jnp.asarray(catalog.source_parameters["luminosity_distance"])
    )
    log_weights = (
        logprob
        - _proposal_log_prob(catalog)
        - 2.0 * (log_target_distance - log_reference_distance)
    )
    return spectral_density(
        power,
        jnp.exp(log_weights),
        rate,
        source_parameters=catalog.source_parameters,
    )


@pytest.mark.parametrize("guarded", [False, True], ids=["ordinary", "guard-mixture"])
@pytest.mark.parametrize("offset", [0.0, 0.13], ids=["fiducial", "off-fiducial"])
def test_prepared_spectrum_reproduces_the_grid_formula(
    injection_catalog: PolarizationPowerCatalog,
    tmp_path: Path,
    guarded: bool,
    offset: float,
) -> None:
    """End-to-end: the same spectrum, the same rate, the same log posterior."""
    config = _non_gr_config()
    proposal_catalog = load_run_catalog(
        _write_catalog(tmp_path / "guarded.h5", seed=1, guarded=guarded),
        label="proposal",
    )

    inputs = _prepare(injection_catalog, proposal_catalog, config)

    params = {
        name: float(value) * (1.0 + offset)
        for name, value in config.fiducials.items()
        if name in config.priors
    }
    expected = _grid_formula_spectrum(inputs, config, params)

    kwargs = inputs.model_kwargs()
    value, trace = log_density(
        partial(
            gwb_spectral_density_model,
            spectral_density_fn=inputs.spectral_density_fn,
            priors=config.priors,
        ),
        (),
        kwargs,
        params,
    )

    # The target side forms the effective distance linearly and then logs it,
    # where the grid formula adds two logs -- so parity is scientific, not
    # bitwise.
    # The site is Independent(Masked(Normal)): one event dimension over the
    # frequency axis, wrapped around the mask.
    np.testing.assert_allclose(
        np.asarray(trace["spectral_density_obs"]["fn"].base_dist.base_dist.loc),
        np.asarray(expected),
        rtol=1e-9,
    )
    expected_density = jnp.sum(
        jnp.where(
            kwargs["frequency_mask"],
            dist.Normal(expected, kwargs["scale"]).log_prob(
                kwargs["observed_spectral_density"]
            ),
            0.0,
        )
    ) + sum(prior.log_prob(params[name]) for name, prior in config.priors.items())
    np.testing.assert_allclose(float(value), float(expected_density), rtol=1e-9)


def test_a_catalog_reweighted_to_its_own_population_has_exactly_zero_log_weights(
    proposal_catalog: PolarizationPowerCatalog,
) -> None:
    """The sanity check the whole importance scheme is legible through.

    The catalog's own generation grid, not the analysis one: the two differ in
    resolution on purpose everywhere else in this module, and interpolating the
    same cosmology on two grids is exactly what stops the weights being
    identically one.
    """
    population = proposal_catalog.get_population()
    assert population.merger_rate_fn is not None
    log_weights_fn = build_importance_spectrum(
        proposal_catalog,
        source_model=population.source_model,
        merger_rate_fn=population.merger_rate_fn,
    )[1]
    log_weights = log_weights_fn(proposal_catalog.fiducials)
    np.testing.assert_array_equal(np.asarray(log_weights), np.zeros(N_SOURCES))


def test_the_proposals_density_factors_reach_the_bound_weights_unchanged(
    injection_catalog: PolarizationPowerCatalog,
) -> None:
    """``prepare_inference_inputs`` threads the catalog's factor set to the target.

    The proposal records only the redshift factor. Reweighted to its own
    (window-restricted) source model at its own parameters, the weights are
    exactly zero only if the target was evaluated with that same narrow set;
    substituting the default factors anywhere in the pipeline would add the
    ordered-mass factor to the target side alone.
    """
    config = _config()
    grid = config.analysis_grid
    narrow = make_catalog(
        redshift=REDSHIFT,
        polarization_power=np.random.default_rng(1).uniform(
            0.0, 1.0, size=(FREQUENCIES.size, N_SOURCES)
        ),
        minimum_frequency=float(FREQUENCIES[0]),
        df=float(FREQUENCIES[1] - FREQUENCIES[0]),
        # Generated on the analysis window itself, so restricting to it leaves
        # the grid -- and with it the stored distances -- unchanged.
        model_kwargs={
            **GENERATION_KWARGS,
            "z_min": grid.minimum_redshift,
            "z_max": grid.maximum_redshift,
        },
        density_sites=("redshift",),
    )
    restricted = narrow.restrict_redshift(grid.minimum_redshift, grid.maximum_redshift)

    inputs = prepare_inference_inputs(
        injection_catalog,
        narrow,
        grid=grid,
        detectors=config.analysis.detectors,
        target=restricted.get_population(),
    )

    assert _bound(inputs)["density_sites"] == ("redshift",)
    np.testing.assert_array_equal(
        np.asarray(inputs.log_weights_fn(inputs.proposal.fiducials)),
        np.zeros(N_RETAINED),
    )


def test_a_guard_mixture_catalog_cannot_supply_an_observed_rate() -> None:
    """A proposal catalog used as an injection fails, rather than scaling wrong.

    The Madau-Dickinson total rate normalizes the Madau-Dickinson redshift
    density, not a mixture of it with a uniform component. Every guarded def
    used to record that rate anyway, and ``catalog_total_merger_rate`` would
    have returned it -- a finite number, off by the guard fraction, with no
    error anywhere downstream.
    """
    guard = make_catalog(
        redshift=REDSHIFT,
        polarization_power=np.ones((FREQUENCIES.size, N_SOURCES)),
        minimum_frequency=float(FREQUENCIES[0]),
        df=float(FREQUENCIES[1] - FREQUENCIES[0]),
        model_name="bns_md_uniform_mixture",
        model_kwargs={**GENERATION_KWARGS, "uniform_mixing_fraction": 0.1},
    )

    assert guard.get_population().merger_rate_fn is None
    with pytest.raises(ValueError, match="declares no merger rate"):
        catalog_total_merger_rate(guard)


def test_the_repository_ships_no_proposal_density_config() -> None:
    """A run names two catalog files; the density is in each file."""
    text = (REPO_ROOT / "config/analysis/base/catalogs.toml").read_text(
        encoding="utf-8"
    )
    assert "[catalog]" in text
    assert "uniform_mixing_fraction" not in text
