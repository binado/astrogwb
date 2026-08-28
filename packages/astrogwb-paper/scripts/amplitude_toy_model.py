"""Single-amplitude toy MCMC and Fisher-overlay figure.

A stripped-down sibling of the inference runner: same importance-weighted
pipeline scaffolding, but the cosmology/population callback is a one-parameter
model. $S_h(f, A)$ scales linearly with amplitude through the total merger
rate; importance weights are unity. Used as a smoke test that NUTS recovers a
known injection, and to build the paper Fisher-overlay figure.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

num_cpus = multiprocessing.cpu_count()
import numpyro

numpyro.set_host_device_count(num_cpus)

import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.frequency import apply_frequency_mask, frequency_mask
from astrogwb.gwb import spectral_density, spectral_snr_squared
from astrogwb.sampling.models import spectral_density_model
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import open_catalog
from astrogwb_paper.catalogs import samples_from_catalog
from astrogwb_paper.config.figures import load_analysis_grid
from astrogwb_paper.paths import paper_project_root, resolve_paper_path
from astrogwb_paper.plotting import TRUTH, use_paper_style
from matplotlib.axes import Axes as MplAxes
from matplotlib.projections import register_projection
from numpyro.infer import MCMC, NUTS

# gwpy (via gwmock-signal) replaces matplotlib's default rectilinear axes; ArviZ 1.2
# mis-detects gwpy axes and looks for arviz_plots.backend.gwpy. Restore matplotlib axes.
register_projection(MplAxes)

jax.config.update("jax_enable_x64", True)
azp.style.use("arviz-variat")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Waveform catalog to use (Snakemake passes its declared input).",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument(
        "--chains-dir", type=Path, default=Path("outputs/chains/amplitude-toy")
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        default=["S1", "R1"],
        help="Detector site codes (resolved via bundled geometry/sensitivity).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Short smoke run (100 warmup / 100 samples / 1 chain).",
    )
    parser.add_argument("--merger-rate-norm", type=float, default=1e-3)
    parser.add_argument("--amplitude-fiducial", type=float, default=1.0)
    parser.add_argument("--prior-low", type=float, default=0.1)
    parser.add_argument("--prior-high", type=float, default=10.0)
    parser.add_argument("--num-warmup", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument(
        "--num-chains",
        default="auto",
        help='Chain count, or "auto" for one chain per CPU.',
    )
    parser.add_argument("--target-accept", type=float, default=0.9)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = paper_project_root()
    catalog_path = resolve_paper_path(args.catalog, root)
    output_path = resolve_paper_path(args.output_pdf, root)
    out_dir = resolve_paper_path(args.chains_dir, root)

    grid = load_analysis_grid()
    detnames = tuple(args.detectors)
    observation_time = grid.observation_time
    seed = args.seed
    num_chains = num_cpus if args.num_chains == "auto" else int(args.num_chains)
    num_warmup = args.num_warmup
    num_samples = args.num_samples
    target_accept = args.target_accept
    if args.debug:
        num_warmup, num_samples, num_chains, target_accept = 100, 100, 1, 0.9

    f_min = grid.f_min
    f_max = grid.f_max
    merger_rate_norm = args.merger_rate_norm
    amplitude_fiducial = args.amplitude_fiducial
    fiducials = {"amplitude": amplitude_fiducial}
    sampled_params = ("amplitude",)
    priors = {"amplitude": dist.Uniform(args.prior_low, args.prior_high)}

    catalog = open_catalog(catalog_path)
    frequencies = jnp.asarray(catalog.frequency.values)
    df = float(catalog.attrs["df"])
    polarization_power = jnp.asarray(catalog.polarization_power.values)
    samples = samples_from_catalog(catalog)
    del catalog
    n_freq, n_samples = polarization_power.shape
    print(f"loaded catalog: n_frequency_bins={n_freq} n_proposal_samples={n_samples}")

    sensitivities = load_sensitivity_map(detnames)
    effective_psd_arr = jnp.asarray(
        effective_psd(frequencies, list(detnames), sensitivities)
    )
    # Uncovered bins have an infinite effective PSD and would contribute a
    # constant -inf to the log-density; drop them with the out-of-band ones.
    mask = frequency_mask(frequencies, fmin=f_min, fmax=f_max) & jnp.isfinite(
        effective_psd_arr
    )
    print("band bins:", int(jnp.sum(mask)), "of", frequencies.shape[0])

    def merger_rate_and_log_weights_fn(params, _samples):
        total_merger_rate = params["amplitude"] * merger_rate_norm
        log_weights = jnp.zeros(n_samples)
        return total_merger_rate, log_weights

    weights_fid = jnp.ones((n_samples,))
    rate_fid = amplitude_fiducial * merger_rate_norm
    observed_spectral_density = spectral_density(
        polarization_power, weights_fid, rate_fid, average_mode="analytic_inclination"
    )
    frequencies, polarization_power, observed_spectral_density, effective_psd_arr = (
        apply_frequency_mask(
            mask,
            frequencies,
            polarization_power,
            observed_spectral_density,
            effective_psd_arr,
        )
    )

    model = partial(
        spectral_density_model,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        priors=priors,
    )
    kernel = NUTS(
        model,
        target_accept_prob=target_accept,
        forward_mode_differentiation=True,
        dense_mass=True,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
        jit_model_args=True,
        chain_method="vectorized",
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=observed_spectral_density,
        effective_psd=effective_psd_arr,
        observation_time=observation_time,
        df=df,
        extra_fields=("num_steps", "accept_prob", "diverging"),
    )
    mcmc.print_summary()

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    params_suffix = "-".join(sampled_params)
    det_suffix = ",".join(detnames)
    base = (
        f"chains-amplitude-toy-{params_suffix}-det={det_suffix}-seed{seed}-{timestamp}"
    )
    inference_data = azb.from_numpyro(mcmc)
    inference_data.to_netcdf(out_dir / f"{base}.nc")
    run_config = {
        "chains_dir": str(out_dir),
        "catalog_path": str(catalog_path),
        "detectors": list(detnames),
        "output_path": str(output_path),
        "seed": seed,
        "observation_time": observation_time,
        "frequency_band": {"f_min": f_min, "f_max": f_max},
        "sampled_params": list(sampled_params),
        "fiducials": fiducials,
        "merger_rate_norm": merger_rate_norm,
        "sampler": {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "target_accept": target_accept,
        },
    }
    (out_dir / f"{base}.json").write_text(json.dumps(run_config, indent=2))
    print("saved:", base)

    azs.summary(inference_data, var_names=list(sampled_params))
    azp.plot_trace_dist(inference_data, var_names=list(sampled_params))
    azp.plot_autocorr(inference_data, var_names=list(sampled_params))

    observation_seconds = float(years_to_seconds(observation_time))
    snr_sq = spectral_snr_squared(
        observed_spectral_density, effective_psd_arr, observation_seconds, df
    )
    snr = float(jnp.sqrt(snr_sq))
    sigma_fisher = float(amplitude_fiducial / snr)
    print(f"SNR={snr:.3f} sigma_fisher={sigma_fisher:.3f}")

    kde = azs.kde(inference_data, var_names=["amplitude"])["amplitude"]
    x_kde = kde.sel(plot_axis="x").values
    y_kde = kde.sel(plot_axis="y").values
    mu = amplitude_fiducial
    x = np.linspace(mu - 4 * sigma_fisher, mu + 4 * sigma_fisher, 400)
    pdf = np.exp(-0.5 * ((x - mu) / sigma_fisher) ** 2) / (
        sigma_fisher * np.sqrt(2 * np.pi)
    )

    use_paper_style()
    fig, ax = plt.subplots()
    ax.plot(x_kde, y_kde, label="MCMC posterior", color="black")
    ax.fill_between(x_kde, y_kde, alpha=0.2, color="black")
    ax.plot(
        x,
        pdf,
        linestyle="--",
        label=r"$\mathcal{N}(A_\mathrm{fid}, 1/\rho_0^2)$",
        color="black",
        lw=2.0,
    )
    ax.axvline(amplitude_fiducial, **TRUTH)
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Posterior density")
    ax.legend(loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print("saved figure:", output_path)


if __name__ == "__main__":
    main()
