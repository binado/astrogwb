# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: astrogwb (3.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example MCMC: $H_0$, $(H_0, \Omega_m)$, and amplitude marginalization
#
# Three ways to infer $H_0$ from the same astrophysical stochastic background,
# run against one catalog so the posteriors can be laid on top of each other:
#
# | | sampled | marginalized |
# |---|---|---|
# | **A** | $H_0$ | — |
# | **B** | $H_0$, $\Omega_m$ | — |
# | **C** | $\Omega_m$ | $H_0$, by quadrature |
#
# `packages/astrogwb/examples/h0_mcmc.py` and
# `packages/astrogwb/examples/amplitude_marginalized_model.py` already run A and
# C as command-line scripts against a waveform bank on disk. What is here and
# nowhere else is the **side-by-side comparison**: all three against one
# injection, with the analytic Fisher width $\sigma_{H_0} = H_0 / \rho$ drawn
# through them.
#
# **This notebook needs no external data.** It carries its own population graph
# inline, draws the sources itself, and builds the catalog from
# `astrogwb.simulation.AnalyticInspiralGenerator`, which is closed-form. The
# result is cached to `notebooks/mcmc_catalog.h5` (gitignored) and rebuilt
# whenever the configuration below stops matching it.
#
# **Why the answer is predictable.** The catalog is both the injection and the
# importance-sampling proposal, so every log-weight is exactly zero and the
# "observed" spectrum is the plain catalog contraction. With only $H_0$ free,
# the predicted spectrum is $S_h(H_0) = A \, S_h(H_0^{\rm fid})$ with
# $A = H_0^{\rm fid}/H_0$, so the log-likelihood is exactly
# $-\tfrac{1}{2}(1-A)^2\rho^2$ and the posterior is Gaussian of width
# $H_0^{\rm fid}/\rho$. Anything else is a bug.
#
# **Runtime** is a few tens of seconds on a laptop CPU: the catalog build is
# under a second, and each NUTS chain is a couple of seconds once XLA has
# compiled the model.

# %% [markdown]
# ## Imports

# %%
import importlib.metadata
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any

# lal warns about SWIG stdout redirection on import, but only under IPython --
# so it fires in Jupyter and under `jupytext --execute`, not in a terminal. It
# is pulled in transitively by astrogwb.detector, so the filter goes first.
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import xarray as xr
from astrogwb.constants import ISCO_ALPHA, SECONDS_PER_YEAR
from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.frequency import apply_frequency_mask, frequency_mask
from astrogwb.gwb import spectral_density, spectral_snr
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
from astrogwb.sampling.models import spectral_density_model
from astrogwb.simulation import (
    AnalyticInspiralGenerator,
    generate_catalog,
    simulate_population,
)
from astrogwb.waveform import load_catalog, save_catalog
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value

# gwpy, pulled in by gwmock-signal behind astrogwb.detector, replaces
# matplotlib's registered rectilinear axes with its own subclass on import.
# Restoring the stock class here keeps plt.subplots() behaving as documented.
register_projection(MplAxes)
plt.rcParams.update({"figure.dpi": 120, "figure.constrained_layout.use": True})

# Last, and after every astrogwb import: x64 must be enabled before the first
# array is created, and no core module builds one at import time. The catalog's
# polarization power is of order 1e-47 Hz^-2 and the likelihood contracts it
# against 1/S_eff^2 -- both underflow float32.
jax.config.update("jax_enable_x64", True)

# Imported here, and not with the block above, because either arviz package
# imported *before* numpyro leaves `numpyro.distributions.distribution` unbound
# on its parent package -- and `numpyro.factor`, which model C scores through,
# reaches for exactly that attribute. It fails inside NUTS, far from the cause.
import arviz_base as azb
import arviz_plots as azp

# %% [markdown]
# ## Notebook configuration
#
# `ASTROGWB_NOTEBOOK_SMOKE=1` shrinks the chains and the catalog so CI can
# execute this notebook as a smoke test. It is a cost knob only: every cell
# runs in both modes, and there is one code path through the notebook. The
# cached catalog keys on the source count, so smoke runs get their own file
# rather than invalidating the full one on every switch.

# %%
SMOKE = os.environ.get("ASTROGWB_NOTEBOOK_SMOKE") == "1"

#: Sources drawn from the population graph below.
NUM_SOURCES = 128 if SMOKE else 1024

#: Frequency grid the catalog is generated on. 8 Hz is coarse -- it is chosen
#: so the (F, N) contraction inside NUTS stays cheap. `catalog_convergence.py`
#: measures what that costs: at this resolution the network SNR is ~13% below
#: its converged value, because the SNR integrand is a narrow peak near 7 Hz.
#: It does not matter here, because the observation time is *solved* for a
#: target SNR rather than assumed.
CATALOG_DF = 8.0
CATALOG_F_MIN = 2.0
CATALOG_F_MAX = 4096.0

#: Analysis band. Narrower than the catalog grid, so the mask below genuinely
#: drops bins rather than being a no-op.
F_MIN = 2.0
F_MAX = 2048.0

#: Where the built catalog is cached. The notebook may be executed from the
#: repository root (`just test-notebooks`) or from its own directory (Jupyter
#: sets the working directory to the notebook's), so the location is resolved
#: rather than hard-coded relative to one of them.
NOTEBOOK_DIR = Path("notebooks") if Path("notebooks").is_dir() else Path()
CATALOG_PATH = NOTEBOOK_DIR / ("mcmc_catalog_smoke.h5" if SMOKE else "mcmc_catalog.h5")

POPULATION_SEED = 41

DETECTORS: tuple[str, ...] = ("E1", "E2", "E3")

#: Target network SNR of the injection. The observation time is solved for it,
#: so the notebook stays in the clean linear regime (sigma_H0/H0 = 1/rho, well
#: inside the Uniform(20, 140) prior) even if a bundled noise curve changes.
TARGET_SNR = 50.0
REFERENCE_OBSERVATION_TIME = 1.0

NUM_WARMUP = 100 if SMOKE else 500
NUM_SAMPLES = 200 if SMOKE else 1000
NUM_CHAINS = 1
SEED = 42

#: Nodes in the H0 quadrature grid model C marginalizes over. The conditional
#: is ~1/rho wide across a 120-wide prior, so a grid too coarse to resolve it
#: collapses; `quadrature_effective_nodes` is reported rather than assumed.
AMPLITUDE_NUM_NODES = 128 if SMOKE else 512

H0_PRIOR = dist.Uniform(20.0, 140.0)
OMEGA_M_PRIOR = dist.Normal(0.3096, 0.006)

#: Set to a Path to write the three chains out as netCDF.
CHAINS_OUT: Path | None = None

# %% [markdown]
# ## The source population
#
# The graph below is an inline mirror of
# `packages/astrogwb/tests/fixtures/mock_bns_population.yaml`, the frozen
# population the core test suite draws its committed fixture from. **Keep the
# two in step**: nothing enforces their agreement, and that is deliberate --
# enforcing it would reintroduce exactly the dependency on
# `packages/astrogwb/tests/` this notebook exists without.
#
# `simulate_population` takes the `parameters:` mapping of that file, not its
# top level. Three details are load-bearing rather than incidental:
#
# - the component-mass bounds must match any analytic comparison built from
#   `uniform_prior_mass_moments` (see `catalog_convergence.py`);
# - `inclination` is a column of exact zeros, which is what pairs with
#   `average_mode="analytic_inclination"` and its
#   $\langle g \rangle / g(0) = 0.4$ conversion from face-on power;
# - the unused `spin_*`, `lambda_*`, and `coa_*` blocks are kept because the
#   YAML they mirror declares them. The underlying graph simulator derives RNG
#   keys from the graph, so the safe assumption is that any edit changes the draw.

# %%
#: Hyperparameters the injection is built at and every non-sampled site is
#: pinned to. `local_merger_rate` is in Gpc^-3 yr^-1.
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

#: Redshift window, and the grid the cosmology integrals run on. The window
#: must match the graph's `madau_dickinson_redshift` bounds: `log w` is exactly
#: zero only when the proposal density was evaluated on the same grid as the
#: target.
Z_MIN = 0.3
Z_MAX = 20.0
N_GRID = 256

MINIMUM_COMPONENT_MASS = 1.0
MAXIMUM_COMPONENT_MASS = 2.5

POPULATION_GRAPH: dict[str, Any] = {
    "luminosity_distance": {
        "transform": {
            "function": "redshift_to_luminosity_distance",
            "arguments": {
                "redshift": "@redshift",
                "hubble_constant": FIDUCIALS["H0"],
                "omega_m": FIDUCIALS["Omega_m"],
                "max_redshift": Z_MAX,
            },
        }
    },
    "mass_pair": {
        "intermediate": True,
        "sampler": {
            "function": "joint_uniform_mass_pair",
            "arguments": {
                "m1_min": MINIMUM_COMPONENT_MASS,
                "m1_max": MAXIMUM_COMPONENT_MASS,
                "m2_min": MINIMUM_COMPONENT_MASS,
                "m2_max": MAXIMUM_COMPONENT_MASS,
                "ordered": True,
            },
        },
    },
    "source_frame_mass_1": {
        "transform": {
            "function": "take_row",
            "arguments": {"matrix": "@mass_pair", "index": 0},
        }
    },
    "source_frame_mass_2": {
        "transform": {
            "function": "take_row",
            "arguments": {"matrix": "@mass_pair", "index": 1},
        }
    },
    "spin_1z": {
        "sampler": {
            "function": "uniform",
            "arguments": {"minimum": -0.05, "maximum": 0.05},
        }
    },
    "spin_2z": {
        "sampler": {
            "function": "uniform",
            "arguments": {"minimum": -0.05, "maximum": 0.05},
        }
    },
    "lambda_1": {
        "sampler": {
            "function": "uniform",
            "arguments": {"minimum": 0.0, "maximum": 2000.0},
        }
    },
    "lambda_2": {
        "sampler": {
            "function": "uniform",
            "arguments": {"minimum": 0.0, "maximum": 2000.0},
        }
    },
    "inclination": {
        "transform": {
            "function": "constant_like",
            "arguments": {"reference": "@redshift", "value": 0.0},
        }
    },
    "coa_phase": {
        "transform": {
            "function": "constant_like",
            "arguments": {"reference": "@redshift", "value": 0.0},
        }
    },
    "coa_time": {
        "transform": {
            "function": "constant_like",
            "arguments": {"reference": "@redshift", "value": 0.0},
        }
    },
    "redshift": {
        "sampler": {
            "function": "madau_dickinson_redshift",
            "arguments": {
                "z_min": Z_MIN,
                "z_max": Z_MAX,
                "gamma": FIDUCIALS["gamma"],
                "kappa": FIDUCIALS["kappa"],
                "z_peak": FIDUCIALS["z_peak"],
                "hubble_constant": FIDUCIALS["H0"],
                "omega_m": FIDUCIALS["Omega_m"],
                "n_grid": 4096,
            },
        }
    },
}

#: The five columns the models dereference by name. Everything else the graph
#: produces is dropped.
CATALOG_PARAMETERS = (
    "redshift",
    "source_frame_mass_1",
    "source_frame_mass_2",
    "inclination",
)


def make_redshift_grid() -> jax.Array:
    """The redshift grid every cosmology integral in this notebook runs on."""
    return jnp.linspace(Z_MIN, Z_MAX, N_GRID)


# %% [markdown]
# ## Building or loading the catalog
#
# `build_catalog` **discards the luminosity distance the graph produced and
# recomputes it** from `compute_merger_rate_distance_and_logprob` at the
# fiducials. That single line is what makes the catalog exactly its own
# importance proposal: the proposal density and the injection then come from
# one function evaluated on one grid, so $\log w = 0$ identically and
# `importance_relative_ess` is exactly 1 rather than merely close to it.
#
# `load_or_build_catalog` reuses the cached file only when its stored
# attributes still describe the configuration above. A bare `is_file()` check
# would silently analyse a stale frequency grid or source count after an edit.


# %%
def build_catalog() -> xr.Dataset:
    """Draw the population and reduce it to a `waveform_catalog` Dataset."""
    drawn = simulate_population(
        POPULATION_GRAPH,
        num_samples=NUM_SOURCES,
        source_type="bns",
        seed=POPULATION_SEED,
    )
    parameters = {
        name: np.asarray(drawn[name], dtype=np.float64) for name in CATALOG_PARAMETERS
    }

    # The recompute described above. The graph's own luminosity_distance is
    # dropped on the line before this one, by never being selected.
    _, luminosity_distance, _ = compute_merger_rate_distance_and_logprob(
        FIDUCIALS,
        {"redshift": jnp.asarray(parameters["redshift"])},
        redshift_grid=make_redshift_grid(),
    )
    parameters["luminosity_distance"] = np.asarray(luminosity_distance)

    return generate_catalog(
        parameters,
        generator=AnalyticInspiralGenerator(
            minimum_frequency=CATALOG_F_MIN,
            maximum_frequency=CATALOG_F_MAX,
            df=CATALOG_DF,
            alpha=ISCO_ALPHA,
        ),
        extra_attrs={
            "notebook": "mcmc_example_models",
            "population": "madau-dickinson",
            "population_seed": POPULATION_SEED,
            "num_sources": NUM_SOURCES,
            "termination_alpha": ISCO_ALPHA,
            "gwmock_pop_version": importlib.metadata.version("gwmock-pop"),
            **{f"fiducial_{name}": value for name, value in FIDUCIALS.items()},
        },
    )


def catalog_matches_configuration(catalog: xr.Dataset) -> bool:
    """Does a cached catalog still describe the configuration cell?

    Attributes survive the netCDF round trip as numpy scalars, so both sides
    are cast before comparing.
    """
    attrs = catalog.attrs
    return (
        float(attrs["df"]) == CATALOG_DF
        and float(attrs["minimum_frequency"]) == CATALOG_F_MIN
        and float(attrs["maximum_frequency"]) == CATALOG_F_MAX
        and int(attrs.get("num_sources", -1)) == NUM_SOURCES
        and int(attrs.get("population_seed", -1)) == POPULATION_SEED
        and all(
            float(attrs.get(f"fiducial_{name}", float("nan"))) == value
            for name, value in FIDUCIALS.items()
        )
    )


def load_or_build_catalog() -> xr.Dataset:
    """Return the cached catalog if it is still current, else rebuild it."""
    if CATALOG_PATH.is_file():
        cached = load_catalog(CATALOG_PATH)
        if catalog_matches_configuration(cached):
            print(f"Loaded {CATALOG_PATH}")
            return cached
        print(
            f"{CATALOG_PATH} does not match this notebook's configuration; rebuilding"
        )
    catalog = build_catalog()
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_catalog(CATALOG_PATH, catalog)
    print(f"Built and wrote {CATALOG_PATH}")
    return catalog


catalog = load_or_build_catalog()

print(f"  shape        {dict(catalog.sizes)}")
print(
    f"  grid         {float(catalog.frequency.values[0])}"
    f"-{float(catalog.frequency.values[-1])} Hz, df = {catalog.attrs['df']} Hz"
)
print(
    f"  sources      {catalog.attrs['num_sources']}, seed {catalog.attrs['population_seed']}"
)
print(f"  gwmock-pop   {catalog.attrs['gwmock_pop_version']}")

# %% [markdown]
# ## The detector network and the analysis band
#
# `effective_psd` is an inverse-variance cross-correlation sum over detector
# pairs, and it returns `inf` wherever no pair contributes. That matters more
# than it looks: `Normal(loc, inf).log_prob` is `-inf`, a constant that kills
# NUTS with no usable diagnostic. Those bins are dropped along with the
# out-of-band ones.
#
# The surviving bins need not be contiguous. Each still has width `df`, taken
# from the catalog attribute -- never measured off the masked grid, whose mean
# spacing is not the bin width.

# %%
frequencies = jnp.asarray(catalog.frequency.values)
df = float(catalog.attrs["df"])
polarization_power = jnp.asarray(catalog.polarization_power.values)
samples = {
    str(name): jnp.asarray(catalog.source_parameters.sel(parameter=name).values)
    for name in catalog.parameter.values
}
num_sources = polarization_power.shape[1]

sensitivities = load_sensitivity_map(DETECTORS)
network_psd = jnp.asarray(effective_psd(frequencies, DETECTORS, sensitivities))
band_mask = frequency_mask(frequencies, fmin=F_MIN, fmax=F_MAX)
finite_mask = jnp.isfinite(network_psd)
mask = band_mask & finite_mask
print(
    f"{int(jnp.sum(mask))} usable bins of {frequencies.shape[0]}: "
    f"{int(jnp.sum(~band_mask))} out of band, "
    f"{int(jnp.sum(band_mask & ~finite_mask))} with no contributing detector pair"
)

fig, ax = plt.subplots(figsize=(7.0, 3.6))
finite = np.asarray(finite_mask)
ax.loglog(
    np.asarray(frequencies)[finite],
    np.asarray(network_psd)[finite],
    lw=1.2,
    label=r"$S_{\rm eff}$",
)
ax.axvspan(
    float(np.asarray(frequencies)[0]),
    F_MIN,
    color="0.85",
    label="dropped: out of band",
)
ax.axvspan(F_MAX, float(np.asarray(frequencies)[-1]), color="0.85")
for f_dropped in np.asarray(frequencies)[np.asarray(band_mask & ~finite_mask)]:
    ax.axvline(f_dropped, color="tab:red", lw=0.6, alpha=0.5)
ax.set_xlabel("frequency [Hz]")
ax.set_ylabel(r"$S_{\rm eff}$ [Hz$^{-1}$]")
ax.set_title(f"Network effective PSD, {' '.join(DETECTORS)}")
ax.legend()
plt.show()

# %% [markdown]
# ## The injection and its SNR
#
# One call to `compute_merger_rate_distance_and_logprob` yields both the
# fiducial merger rate, which builds the injection, and the proposal
# log-density, which builds the weights. Sharing it is what makes every weight
# exactly 1 at the fiducials.
#
# The observation time is then *derived*: $\rho^2 = 2T(S_h|S_h)$ and
# $S_{\rm eff}$ carries no $T$, so $\rho \propto \sqrt{T}$ and the $T$ landing
# the injection on `TARGET_SNR` is one division.

# %%
total_merger_rate, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
    FIDUCIALS, samples, redshift_grid=make_redshift_grid()
)
observed_spectral_density = spectral_density(
    polarization_power,
    jnp.ones(num_sources),
    total_merger_rate,
    average_mode="analytic_inclination",
)

weights_fn = make_merger_rate_and_log_weights_fn(
    fiducials=FIDUCIALS,
    redshift_grid=make_redshift_grid(),
    proposal_logprob=proposal_logprob,
)


def merger_rate_and_log_weights(params, catalog_samples):
    """Take every hyperparameter the chain does not sample from the fiducials."""
    return weights_fn({**FIDUCIALS, **params}, catalog_samples)


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

reference_snr = float(
    spectral_snr(
        observed_spectral_density,
        network_psd,
        REFERENCE_OBSERVATION_TIME * SECONDS_PER_YEAR,
        df,
    )
)
observation_time = REFERENCE_OBSERVATION_TIME * (TARGET_SNR / reference_snr) ** 2
snr = float(
    spectral_snr(
        observed_spectral_density,
        network_psd,
        observation_time * SECONDS_PER_YEAR,
        df,
    )
)
fisher_width = FIDUCIALS["H0"] / snr
print(f"SNR at {REFERENCE_OBSERVATION_TIME} yr : {reference_snr:.3f}")
print(f"observation time for rho = {TARGET_SNR}: {observation_time:.4f} yr")
print(f"round-trip SNR                : {snr:.6f}")
print(f"expected Fisher width         : sigma_H0 = {fisher_width:.4f}")

# %%
model_kwargs: dict[str, object] = {
    "polarization_power": polarization_power,
    "samples": samples,
    "observed_spectral_density": observed_spectral_density,
    "effective_psd": network_psd,
    "observation_time": observation_time,
    "df": df,
}


def run_nuts(
    model, *, init_values: dict[str, float], seed: int = SEED
) -> tuple[dict, MCMC]:
    """Run NUTS with the model data passed dynamically through `MCMC.run`.

    Passing the arrays through `run` rather than baking them into the model
    with `partial` lets the three chains below share one XLA compilation: the
    shapes are identical, so only the traced structure has to match.

    The `MCMC` object comes back alongside the samples because `arviz` builds
    its `InferenceData` from it, not from the sample dict: the sampler carries
    the divergences, tree depths, and chain structure a bare dict has lost.
    """
    kernel = NUTS(
        model,
        target_accept_prob=0.9,
        # One or two latents against an (F, N) matvec is exactly the case
        # forward-mode AD is built for; reverse mode would tape the whole
        # contraction.
        forward_mode_differentiation=True,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        progress_bar=False,
        jit_model_args=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), **model_kwargs)
    return mcmc.get_samples(), mcmc


# %% [markdown]
# ## Model A — $H_0$ only
#
# The general model with a single latent. Every other hyperparameter is pinned
# at its fiducial by the closure above.

# %%
model_a = partial(
    spectral_density_model,
    average_mode="analytic_inclination",
    merger_rate_and_log_weights_fn=merger_rate_and_log_weights,
    priors={"H0": H0_PRIOR},
)
posterior_a, mcmc_a = run_nuts(model_a, init_values={"H0": FIDUCIALS["H0"]})
print(f"H0 = {np.mean(posterior_a['H0']):.4f} +/- {np.std(posterior_a['H0']):.4f}")
print(
    f"relative ESS = {np.mean(np.asarray(posterior_a['importance_relative_ess'])):.12f}"
)

# %% [markdown]
# ## Model B — $(H_0, \Omega_m)$ sampled directly
#
# The same model with a second latent. $\Omega_m$ enters the spectrum only
# through $E(z)$ and the comoving volume, so it is weakly constrained by the
# background alone and the prior does most of the work; what it buys is a
# realistic amount of shape freedom for $H_0$ to trade against.

# %%
model_b = partial(
    spectral_density_model,
    average_mode="analytic_inclination",
    merger_rate_and_log_weights_fn=merger_rate_and_log_weights,
    priors={"H0": H0_PRIOR, "Omega_m": OMEGA_M_PRIOR},
)
posterior_b, mcmc_b = run_nuts(
    model_b, init_values={"H0": FIDUCIALS["H0"], "Omega_m": FIDUCIALS["Omega_m"]}
)
print(f"H0      = {np.mean(posterior_b['H0']):.4f} +/- {np.std(posterior_b['H0']):.4f}")
print(
    f"Omega_m = {np.mean(posterior_b['Omega_m']):.4f} "
    f"+/- {np.std(posterior_b['Omega_m']):.4f}"
)

# %% [markdown]
# ## Model C — $\Omega_m$ sampled, $H_0$ marginalized
#
# $H_0$ is strictly multiplicative on the spectrum, which is the long curved
# degeneracy NUTS handles worst. `amplitude_marginalized_model` integrates it
# out under its own prior by quadrature and publishes the sufficient statistics
# instead; `amplitude_reconstruction_model` draws $H_0$ back from those under
# `Predictive`, at $O(K)$ per draw with no catalog contraction at all.
#
# `amplitude_fn=amplitude_H0_fn` is passed **by name and never wrapped**.
# `AmplitudeConditional` hashes it into the jit cache key, so a freshly-minted
# closure -- a lambda, a `partial` -- retraces the model on every construction.

# %%
amplitude_grid = quadrature_grid(H0_PRIOR, num_nodes=AMPLITUDE_NUM_NODES)
model_c = partial(
    amplitude_marginalized_model,
    average_mode="analytic_inclination",
    merger_rate_and_log_weights_fn=merger_rate_and_log_weights,
    amplitude_parameter="H0",
    fiducials=FIDUCIALS,
    amplitude_fn=amplitude_H0_fn,
    amplitude_prior=H0_PRIOR,
    amplitude_grid=amplitude_grid,
    priors={"Omega_m": OMEGA_M_PRIOR},
)
posterior_c, mcmc_c = run_nuts(model_c, init_values={"Omega_m": FIDUCIALS["Omega_m"]})

# A numpyro.factor publishes no draws, so H0 is absent from the chain itself.
print("chain sites:", sorted(posterior_c))

reconstructed_c = Predictive(
    partial(
        amplitude_reconstruction_model,
        amplitude_parameter="H0",
        amplitude_fn=amplitude_H0_fn,
        merger_rate_amplitude_fn=merger_rate_H0_fn,
        prior=H0_PRIOR,
        fiducial=FIDUCIALS["H0"],
        grid=amplitude_grid,
    ),
    num_samples=1,
    return_sites=["H0", "total_merger_rate", "quadrature_effective_nodes"],
)(
    # fold_in keeps the reconstruction key distinct from the chain's.
    jax.random.fold_in(jax.random.PRNGKey(SEED), 1),
    amplitude_mle=posterior_c["amplitude_mle"],
    template_optimal_snr=posterior_c["template_optimal_snr"],
    template_merger_rate=posterior_c["template_merger_rate"],
)
reconstructed_c = {name: values[0] for name, values in reconstructed_c.items()}
print(
    f"H0 = {np.mean(reconstructed_c['H0']):.4f} +/- {np.std(reconstructed_c['H0']):.4f}"
)
print(
    "effective quadrature nodes = "
    f"{np.mean(np.asarray(reconstructed_c['quadrature_effective_nodes'])):.1f} "
    f"of {AMPLITUDE_NUM_NODES}"
)

# %% [markdown]
# ## Comparing the three posteriors
#
# The three $H_0$ marginals, the fiducial, and the Fisher width
# $\sigma_{H_0} = H_0^{\rm fid} / \rho$ the noiseless linear model predicts.
# A, B, and C should be indistinguishable up to Monte-Carlo error: adding
# $\Omega_m$ barely widens $H_0$, and marginalizing analytically is exact up to
# quadrature error.
#
# The plots come from `arviz`, built straight off the `MCMC` objects.
# `azb.from_numpyro` handles both model shapes here — `numpyro.deterministic`
# sites become posterior variables, and `numpyro.factor` scoring is fine
# because nothing below asks for a pointwise log-likelihood (which is why
# `log_likelihood=False`, the default, is left alone).
#
# **Model C needs one repair.** Its $H_0$ is not in its chain at all: the model
# marginalizes it out by quadrature, so it exists only in the `Predictive`
# reconstruction above. That array is assigned onto C's posterior group under
# the `(chain, draw)` dims the rest of the tree uses, after which C is an
# ordinary `InferenceData` and plots like the others.

# %%
idata_a = azb.from_numpyro(mcmc_a)
idata_b = azb.from_numpyro(mcmc_b)
idata_c = azb.from_numpyro(mcmc_c)

# Reconstructed draws arrive flat; the tree wants them shaped (chain, draw).
idata_c["posterior"] = idata_c.posterior.dataset.assign(
    H0=(("chain", "draw"), np.asarray(reconstructed_c["H0"]).reshape(NUM_CHAINS, -1))
)

posteriors = {
    r"A: $H_0$": idata_a,
    r"B: $H_0,\ \Omega_m$": idata_b,
    r"C: $H_0$ marginalized": idata_c,
}

# %% [markdown]
# ### The $H_0$ marginals
#
# `plot_dist` takes the mapping of models directly and colours by it. What it
# cannot draw is the point of the figure — the analytic Fisher prediction the
# three posteriors are being held against — so that goes on top of the axes it
# hands back.

# %%
pc = azp.plot_dist(
    posteriors,
    var_names=["H0"],
    backend="matplotlib",
    # The three means coincide, so arviz's per-model annotations land on top of
    # each other. The interval bars below the curves carry the same information.
    visuals={"point_estimate_text": False},
)
pc.add_legend("model")
ax = pc.get_target("H0", {})

grid = np.linspace(*ax.get_xlim(), 200)
ax.plot(
    grid,
    np.exp(-0.5 * ((grid - FIDUCIALS["H0"]) / fisher_width) ** 2)
    / (fisher_width * np.sqrt(2.0 * np.pi)),
    color="k",
    ls="--",
    lw=1.2,
    label=rf"Fisher: $\sigma = H_0/\rho = {fisher_width:.2f}$",
)
ax.axvline(FIDUCIALS["H0"], color="k", lw=0.8)
ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
ax.set_title(rf"Three routes to $H_0$ at $\rho = {snr:.0f}$")
# arviz's own legend is attached to the figure, so an axes-level one for the
# Fisher curve sits alongside it rather than replacing it.
ax.legend(fontsize=8, loc="upper left")
plt.show()

# %% [markdown]
# ### The $(H_0, \Omega_m)$ plane
#
# This is the figure a 1-D marginal cannot give. Model C exists *because* $H_0$
# is strictly multiplicative on the spectrum and so trades against $\Omega_m$
# along a long curved ridge; B samples that ridge with NUTS, C integrates
# across it by quadrature. Laying the two on one plane shows they agree on the
# whole 2-D shape, not merely on the width of a projection.
#
# `plot_pair` takes a single model rather than a mapping, so B is drawn first
# and C is overlaid onto the `PlotMatrix` it returns.


# %%
def pair_style(color: str) -> dict[str, dict[str, Any]]:
    """One colour for both the scatter and the marginals of one model."""
    return {"scatter": {"color": color, "alpha": 0.35}, "dist": {"color": color}}


pm = azp.plot_pair(
    idata_b,
    var_names=["H0", "Omega_m"],
    backend="matplotlib",
    visuals=pair_style("tab:blue"),
)
azp.plot_pair(
    idata_c,
    var_names=["H0", "Omega_m"],
    plot_matrix=pm,
    visuals=pair_style("tab:orange"),
)
# get_target's second positional argument is the *selection* along x, not the
# y variable; the off-diagonal panel needs both names given explicitly.
scatter_ax = pm.get_target("H0", {}, var_name_y="Omega_m", selection_y={})
scatter_ax.axvline(FIDUCIALS["H0"], color="k", lw=0.8)
scatter_ax.axhline(FIDUCIALS["Omega_m"], color="k", lw=0.8)
scatter_ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
scatter_ax.set_ylabel(r"$\Omega_m$")
# The diagonal panels carry arviz's raw variable names; match them to the rest.
pm.get_target("H0", {}).set_ylabel(r"$H_0$")
pm.get_target("Omega_m", {}).set_xlabel(r"$\Omega_m$")
# Two independent plot_pair calls share no aesthetic mapping, so the legend is
# built from empty proxy artists rather than from either PlotMatrix.
scatter_ax.scatter([], [], color="tab:blue", label=r"B: $H_0,\ \Omega_m$ sampled")
scatter_ax.scatter([], [], color="tab:orange", label=r"C: $H_0$ marginalized")
scatter_ax.legend(fontsize=8, loc="upper right")
plt.show()

# %% [markdown]
# ### The numbers
#
# The table carries what neither plot does: the posterior width against the
# Fisher prediction, and the importance-sampling efficiency.

# %%
# One source of truth for the draws: whatever went into the figures above,
# flattened back across chains.
results = {
    label: np.asarray(idata.posterior.dataset["H0"]).ravel()
    for label, idata in posteriors.items()
}

relative_ess = {
    r"A: $H_0$": float(np.mean(np.asarray(posterior_a["importance_relative_ess"]))),
    r"B: $H_0,\ \Omega_m$": float(
        np.mean(np.asarray(posterior_b["importance_relative_ess"]))
    ),
    r"C: $H_0$ marginalized": float(
        np.mean(np.asarray(posterior_c["importance_relative_ess"]))
    ),
}
header = f"{'model':24s} {'mean':>9s} {'std':>8s} {'std/Fisher':>11s} {'rel ESS':>9s}"
print(header)
print("-" * len(header))
for label, draws in results.items():
    print(
        f"{label:24s} {np.mean(draws):9.4f} {np.std(draws):8.4f} "
        f"{np.std(draws) / fisher_width:11.3f} {relative_ess[label]:9.6f}"
    )
print()
print(
    "quadrature effective nodes (model C): "
    f"{np.mean(np.asarray(reconstructed_c['quadrature_effective_nodes'])):.1f}"
)
print(
    "The relative ESS is exactly 1 because the catalog is its own proposal: "
    "every log-weight is identically zero."
)

# %% [markdown]
# ## Saving the chains
#
# Optional, and off by default. Set `CHAINS_OUT` in the configuration cell to a
# path to write all three chains into one netCDF file.

# %%
if CHAINS_OUT is not None:
    chains = xr.Dataset(
        {
            **{f"a_{name}": ("draw", np.asarray(v)) for name, v in posterior_a.items()},
            **{f"b_{name}": ("draw", np.asarray(v)) for name, v in posterior_b.items()},
            **{f"c_{name}": ("draw", np.asarray(v)) for name, v in posterior_c.items()},
            "c_H0_reconstructed": ("draw", np.asarray(reconstructed_c["H0"])),
        },
        attrs={
            "catalog": str(CATALOG_PATH),
            "detectors": " ".join(DETECTORS),
            "seed": SEED,
            "f_min": F_MIN,
            "f_max": F_MAX,
            "df": df,
            "num_frequency_bins": int(frequencies.shape[0]),
            "num_catalog_samples": num_sources,
            "observation_time": observation_time,
            "snr": snr,
            **{f"fiducial_{name}": value for name, value in FIDUCIALS.items()},
        },
    )
    CHAINS_OUT.parent.mkdir(parents=True, exist_ok=True)
    chains.to_netcdf(CHAINS_OUT, engine="h5netcdf")
    print(f"Wrote {CHAINS_OUT}")
else:
    print("CHAINS_OUT is None; not saving.")
