"""The shot-noise floor of each network, split by factor and by Stokes term.

The paper writes the floor of the stacked per-pair data as three factors,
:math:`\\sigma_{\\rm shot}^2 \\simeq N^{-1} \\langle Q_{\\rm int}^2 \\rangle /
\\langle Q_{\\rm int} \\rangle^2 \\, \\langle \\bar r^2 \\rangle`, and the response
moment as intensity, linear and circular terms (eq. ``response-moment-main``).
This script evaluates both per network at the fiducials, on the grid
notebook's reference catalog and frequency grid:

- :math:`\\sigma_{\\rm shot}` under the per-pair response, and under the
  sky-averaged response, for which :math:`\\langle \\bar r^2 \\rangle = 355/252`;
- the effective :math:`\\langle \\bar r^2 \\rangle =
  355/252 \\cdot \\sigma_{\\rm shot}^2 / \\sigma_{\\rm sky}^2`, and the intrinsic
  factor :math:`N \\sigma_{\\rm sky}^2 / (355/252)`;
- the share of :math:`\\sigma_{\\rm shot}^2` from each Stokes term. The per-source
  response is linear in the Stokes vector :math:`(g, s_Q, 0, s_V)`, so each
  term's response draws use the same angles; cross terms vanish on average,
  and the three shares sum to one up to Monte Carlo noise, which is reported.

    uv run --extra notebook python scripts/shot_noise_budget.py
    uv run --extra notebook python scripts/shot_noise_budget.py --tex-only

``--tex-only`` rewrites the LaTeX table from the saved CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from astrogwb.paper.runtime import configure_runtime

# Backend configuration precedes waveform construction and array creation.
configure_runtime(num_chains=1)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from astrogwb.detector.overlap import stokes_cross_power
from astrogwb.detector.response import _isotropic_angles
from astrogwb.gwb.importance import INCLINATION_SECOND_MOMENT
from astrogwb.inference import ImportanceGaussianLikelihood
from astrogwb.inference import Network as LikelihoodNetwork
from astrogwb.paper.config import (
    detector_registry,
    fiducials,
    population_metadata,
    waveform_metadata,
)
from astrogwb.paper.config.runs import CATALOGS_ROOT, FIGURES_DIR
from astrogwb.paper.plotting import DETECTOR_NETWORKS
from astrogwb.paper.posteriors import ensure_reference_catalog, network_pair_data
from astrogwb.populations import build_population
from astrogwb.simulators.polarization_power import CatalogMetadata
from astrogwb.waveform import quadrupolar_stokes_factors

ROOT_DIR = Path(__file__).resolve().parents[1]

# Keep in sync with the configuration of notebooks/cosmology_grid_posteriors.py.
CATALOG_SEED = 42
NUM_SAMPLES = 2**14
REDSHIFT_NODES = 32
CATALOG_CHUNK_SIZE = 4096
OBSERVATION_TIME = 1.0
MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY = 2.0, 2048.0
RESPONSE_DRAWS, RESPONSE_SEED = 32_768, 0

#: The isotropic mean of the bracket of the per-source ORF.
_ISOTROPIC_PAIR_RESPONSE = 4.0 / 25.0
#: Stokes components kept by each term: intensity, linear (Q), circular (V).
TERMS = {
    "intensity": np.array([1.0, 0.0, 0.0, 0.0]),
    "linear": np.array([0.0, 1.0, 1.0, 0.0]),
    "circular": np.array([0.0, 0.0, 0.0, 1.0]),
}


def build_likelihood(frequencies: np.ndarray) -> ImportanceGaussianLikelihood:
    """The grid notebook's likelihood at the fiducials, amplitude mode."""
    catalog_metadata = CatalogMetadata(
        waveform=waveform_metadata(
            root=ROOT_DIR,
            frequency_spacing="log",
            minimum_frequency=MINIMUM_FREQUENCY,
            maximum_frequency=4000.0,
            frequency_resolution=0.02,
            turnover_frequency=None,
        ),
        population=population_metadata(root=ROOT_DIR),
        fiducials=fiducials(root=ROOT_DIR),
        num_samples=NUM_SAMPLES,
        sampling="sobol",
    )
    data = ensure_reference_catalog(
        catalog_metadata, CATALOG_SEED, CATALOGS_ROOT, chunk_size=CATALOG_CHUNK_SIZE
    )
    population = population_metadata(root=ROOT_DIR)
    return ImportanceGaussianLikelihood.from_catalog(
        data,
        catalog_metadata,
        population=build_population(population.model_name, **population.model_kwargs),
        frequencies=frequencies,
        num_redshift_nodes=REDSHIFT_NODES,
        density_sites=(),
        shot_noise="amplitude",
        observation_time=OBSERVATION_TIME,
    )


def term_response(
    frequencies: np.ndarray, detectors: list, angles: tuple, mask: np.ndarray
) -> jax.Array:
    """Every pair's per-source response from the Stokes components ``mask``, ``(L, P, F)``."""
    polar_cosine, azimuth, polarization, cos_inclination = angles
    stokes = (quadrupolar_stokes_factors(cos_inclination) * mask[:, None])[..., None]
    return jnp.stack(
        [
            stokes_cross_power(
                frequencies,
                detectors[first],
                detectors[second],
                polar_cosine,
                azimuth,
                polarization,
                stokes,
            )
            / _ISOTROPIC_PAIR_RESPONSE
            for first in range(len(detectors))
            for second in range(first + 1, len(detectors))
        ],
        axis=-2,
    )


OUTPUT = ROOT_DIR / FIGURES_DIR / "shot_noise_budget"


def latex_table(table: pd.DataFrame) -> str:
    """The paper's per-network shot-noise budget table."""
    intrinsic = float(table["intrinsic"].mean())
    num_sources = float(table["num_sources"].iloc[0])
    sky = float(np.sqrt(intrinsic * INCLINATION_SECOND_MOMENT))
    closure = float(np.max(np.abs(table["share_closure"] - 1.0)))
    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        (
            r"\caption{Shot-noise floor of each network at the fiducial parameters for "
            r"$T=1$~year, with the response of the stacked per-pair data. "
            r"$\sigma_{\rm shot}\sqrt N$ is the floor in units of the Poisson limit, "
            rf"for $N\simeq{num_sources / 1e5:.1f}\times10^5$ sources. "
            r"$\langle\bar r^2\rangle$ is the response factor of "
            r"eq.~\eqref{eq:shot-noise-three-factors}, defined as "
            r"$355/252$ times the ratio of $\sigma_{\rm shot}^2$ to its sky-averaged "
            r"value; the intrinsic factor $\langle Q_{\rm int}^2\rangle/"
            rf"\langle Q_{{\rm int}}\rangle^2\simeq{intrinsic:.2f}$ is common to all "
            rf"networks, so the sky-averaged floor is ${sky:.2f}/\sqrt N$. The next "
            r"three columns split $\sigma_{\rm shot}^2$ into the intensity, linear "
            r"and circular terms of eq.~\eqref{eq:response-moment-main}; they sum to "
            rf"one within ${100 * closure:.0f}\%$, the Monte Carlo error of the "
            rf"${RESPONSE_DRAWS}$ response draws.}}"
        ),
        r"\label{tab:shot-noise-budget}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r"Detector Network & $10^3\sigma_{\rm shot}$ & $\sigma_{\rm shot}\sqrt N$ "
            r"& $\langle\bar r^2\rangle$ & $\mathcal I$ & $\mathcal Q$ & $\mathcal V$ "
            r"& $\rho\sigma_{\rm shot}$ \\"
        ),
        r"\midrule",
    ]
    for row in table.itertuples():
        shares = [
            f"{100 * value:.0f}\\%"
            for value in (row.share_intensity, row.share_linear, row.share_circular)
        ]
        lines.append(
            " & ".join(
                [
                    row.network,
                    f"{1e3 * row.sigma_shot:.2f}",
                    f"{row.sigma_shot_sqrt_n:.2f}",
                    f"{row.r_bar_squared:.2f}",
                    *shares,
                    f"{row.rho_sigma_shot:.2f}",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tex-only", action="store_true", help="rewrite the table from the CSV"
    )
    if parser.parse_args().tex_only:
        table = pd.read_csv(OUTPUT / "shot_noise_budget.csv")
        if "num_sources" not in table:
            table["num_sources"] = 7.134e5
        (OUTPUT / "shot_noise_budget.tex").write_text(latex_table(table))
        return
    fiducial = fiducials(root=ROOT_DIR)
    registry = detector_registry(root=ROOT_DIR)
    frequencies = np.asarray(waveform_metadata(root=ROOT_DIR).build().frequencies)
    likelihood = build_likelihood(frequencies)
    spectrum, _, extras = likelihood.predict(fiducial)
    num_sources = float(extras["total_merger_rate"]) * float(
        likelihood.observation_seconds
    )
    angles = _isotropic_angles(jax.random.key(RESPONSE_SEED), RESPONSE_DRAWS)

    rows = []
    for name, label in DETECTOR_NETWORKS:
        detectors, _ = registry.build_network(name)
        data = network_pair_data(
            registry,
            name,
            frequencies,
            observation_time=OBSERVATION_TIME,
            minimum_frequency=MINIMUM_FREQUENCY,
            maximum_frequency=MAXIMUM_FREQUENCY,
            num_pairs=len(detectors) * (len(detectors) - 1) // 2,
        )
        base = LikelihoodNetwork(
            jnp.asarray(data["scale"]),
            jnp.asarray(data["frequency_mask"]),
            pair_overlap=jnp.asarray(data["pair_overlap"]),
        )

        def relative_variance(response: jax.Array | None, base=base) -> float:
            if response is None:
                return float(likelihood.relative_variance(fiducial, base))
            network = base._replace(response=response)
            network = network._replace(
                response_moments=likelihood.response_moments(fiducial, network),
                response=None,
            )
            return float(likelihood.relative_variance(fiducial, network))

        sky = relative_variance(None)
        terms = {
            term: relative_variance(term_response(frequencies, detectors, angles, mask))
            for term, mask in TERMS.items()
        }
        total = relative_variance(
            term_response(frequencies, detectors, angles, np.ones(4))
        )
        expected = np.asarray(data["pair_overlap"]) * np.asarray(spectrum)
        keep = np.asarray(data["frequency_mask"])
        rho = float(
            np.sqrt(
                np.sum(np.where(keep, (expected / np.asarray(data["scale"])) ** 2, 0.0))
            )
        )
        rows.append(
            {
                "network": label,
                "sigma_shot": np.sqrt(total),
                "sigma_sky": np.sqrt(sky),
                "sigma_shot_sqrt_n": np.sqrt(total * num_sources),
                "r_bar_squared": INCLINATION_SECOND_MOMENT * total / sky,
                "intrinsic": num_sources * sky / INCLINATION_SECOND_MOMENT,
                **{f"share_{term}": value / total for term, value in terms.items()},
                "share_closure": sum(terms.values()) / total,
                "rho": rho,
                "rho_sigma_shot": rho * np.sqrt(total),
                "num_sources": num_sources,
            }
        )
        print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "shot_noise_budget.csv", index=False)
    (OUTPUT / "shot_noise_budget.tex").write_text(latex_table(table))
    print(f"N = {num_sources:.4g} sources; table in {OUTPUT}")
    print(table.to_string())


if __name__ == "__main__":
    main()
