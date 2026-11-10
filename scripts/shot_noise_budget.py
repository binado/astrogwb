"""The variance budget of each network's amplitude estimator, split by covariance term.

The paper's amplitude estimator weights the stacked per-pair data with the
total covariance, :math:`\\hat x = \\boldsymbol\\mu^T \\Sigma^{-1} \\mathbf d /
(\\boldsymbol\\mu^T \\Sigma^{-1} \\boldsymbol\\mu)`, with variance
:math:`(\\boldsymbol\\mu^T \\Sigma^{-1} \\boldsymbol\\mu)^{-1}`
(eq. ``amplitude-relative-uncertainty``). The covariance is a sum of the
detector noise and the four connected terms of eq.
``connected-covariance-decomposition``,

.. math::

    \\Sigma = \\Sigma^g + \\Sigma^{\\rm count} + \\Sigma^I + \\Sigma^{\\rm lin}
        + \\Sigma^{\\rm circ},

and this script reports the share :math:`\\mathbf w^T \\Sigma^X \\mathbf w /
{\\rm Var}[\\hat x]` of each, with the estimator's own weights :math:`\\mathbf
w` (:func:`~astrogwb.inference.amplitude_variance_budget`). The shares sum to
one exactly; ``docs/shot-noise-budget.md`` derives them and compares other
attributions. Per network, at the fiducials, on the grid notebook's reference
catalog and the fiducial ensemble's detector noise, both written by
``Snakefile``:

- :math:`\\Sigma^{\\rm count} = \\boldsymbol\\mu\\boldsymbol\\mu^T / N`, the
  fluctuation of the event count at the mean source;
- :math:`\\Sigma^I`, the rest of the intensity term of the response moment, the
  spread of source powers and responses at a fixed count;
- :math:`\\Sigma^{\\rm lin}` and :math:`\\Sigma^{\\rm circ}`, the linear and
  circular Stokes terms. The per-source response is linear in the Stokes
  vector :math:`(g, s_Q, 0, s_V)`, so each term's response draws use the same
  angles. Each draw enters with its images under :math:`\\psi \\to \\psi +
  \\pi/4`, which flips the sign of the linear response, and :math:`\\cos\\iota
  \\to -\\cos\\iota`, which flips the circular one; the cross terms between
  Stokes terms then cancel draw by draw, and the three response moments sum to
  the full one to round-off.

For comparison it also records the detector-noise template's
:math:`\\rho^{-2} + \\sigma_{\\rm shot}^2` and the shape correction
:math:`\\delta` that separates the two variances,
:math:`{\\rm Var}[\\hat x] = \\rho^{-2} + \\sigma_{\\rm shot}^2 - \\delta/\\rho^2`.

    uv run --extra notebook python scripts/shot_noise_budget.py
    uv run --extra notebook python scripts/shot_noise_budget.py --tex-only

``--tex-only`` rewrites the LaTeX table from the saved CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from astrogwb.runtime import configure_runtime

# Backend configuration precedes waveform construction and array creation.
configure_runtime()

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.typing import ArrayLike

from astrogwb.config import load_config
from astrogwb.detector.overlap import stokes_cross_power
from astrogwb.detector.response import _isotropic_angles
from astrogwb.importance.catalog import read_catalog
from astrogwb.inference import (
    CatalogModel,
    Prediction,
    amplitude_template,
    amplitude_variance_budget,
    catalog_covariance,
    expected_data,
    shape_correction,
    shot_noise_variance,
)
from astrogwb.inference.gaussian import Network
from astrogwb.paper.plotting import DETECTOR_NETWORKS
from astrogwb.paper.posteriors import pair_network
from astrogwb.simulators import BackgroundArtifact
from astrogwb.waveform import quadrupolar_stokes_factors

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT = ROOT_DIR / "outputs" / "figures" / "shot_noise_budget"

# Keep in sync with the configuration of notebooks/cosmology_grid_posteriors.py.
REDSHIFT_NODES = 32
MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY = 2.0, 2048.0
DATA_SEED = 41  # the injection ensemble draw 0
#: Independent angle draws; each enters with its three parity images.
RESPONSE_DRAWS, RESPONSE_SEED = 8_192, 0

#: The isotropic mean of the bracket of the per-source ORF.
_ISOTROPIC_PAIR_RESPONSE = 4.0 / 25.0
#: Stokes components kept by each term: intensity, linear (Q, U), circular (V).
STOKES_TERMS = {
    "intensity": np.array([1.0, 0.0, 0.0, 0.0]),
    "linear": np.array([0.0, 1.0, 1.0, 0.0]),
    "circular": np.array([0.0, 0.0, 0.0, 1.0]),
}
#: The covariance terms in the order of eq. connected-covariance-decomposition.
COVARIANCE_TERMS = ("detector", "count", "intensity", "linear", "circular")
#: Their column headers in the paper's table.
COLUMN_HEADERS = {
    "detector": r"$\Sigma^g$",
    "count": r"$\Sigma^{c,{\rm count}}$",
    "intensity": r"$\Sigma^{c,I}$",
    "linear": r"$\Sigma^{c,{\rm lin}}$",
    "circular": r"$\Sigma^{c,{\rm circ}}$",
}


def build_model(
    catalog_path: Path, frequencies: np.ndarray, observation_time: float
) -> CatalogModel:
    """The grid notebook's catalog model."""
    data, catalog_metadata, _ = read_catalog(catalog_path)
    return CatalogModel.from_catalog(
        data,
        catalog_metadata,
        population=catalog_metadata.population.build(),
        frequencies=frequencies,
        num_redshift_nodes=REDSHIFT_NODES,
        observation_time=observation_time,
    )


def parity_symmetric_angles(key: jax.Array, num_draws: int) -> tuple[jax.Array, ...]:
    r"""Isotropic angles with the images under :math:`\psi + \pi/4` and :math:`-\cos\iota`.

    Returns the four angle arrays of
    :func:`~astrogwb.detector.response._isotropic_angles`, each
    ``(4 num_draws,)``.
    """
    right_ascension, declination, polarization_angle, cos_inclination = (
        _isotropic_angles(key, num_draws)
    )
    rotation = jnp.array([0.0, 0.25 * jnp.pi, 0.0, 0.25 * jnp.pi])[:, None]
    reflection = jnp.array([1.0, 1.0, -1.0, -1.0])[:, None]
    return (
        jnp.tile(right_ascension, 4),
        jnp.tile(declination, 4),
        jnp.ravel(polarization_angle + rotation),
        jnp.ravel(cos_inclination * reflection),
    )


def response_moment(draws: jax.Array) -> jax.Array:
    """The second moment of per-pair response draws ``(L, P, F)``, ``(P, F, P, F)``."""
    flat = jnp.reshape(draws, (draws.shape[0], -1))
    moment = flat.T @ flat / draws.shape[0]
    return jnp.reshape(moment, (*draws.shape[1:], *draws.shape[1:]))


def term_response(
    frequencies: np.ndarray, detectors: tuple, angles: tuple, mask: np.ndarray
) -> jax.Array:
    """Every pair's per-source response from the Stokes components ``mask``, ``(L, P, F)``."""
    right_ascension, declination, polarization_angle, cos_inclination = angles
    stokes = (quadrupolar_stokes_factors(cos_inclination) * mask[:, None])[..., None]
    return jnp.stack(
        [
            stokes_cross_power(
                frequencies,
                detectors[first],
                detectors[second],
                right_ascension,
                declination,
                polarization_angle,
                stokes,
            )
            / _ISOTROPIC_PAIR_RESPONSE
            for first in range(len(detectors))
            for second in range(first + 1, len(detectors))
        ],
        axis=-2,
    )


def covariance_terms(
    prediction: Prediction,
    networks: dict[str, Network],
    num_sources: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    r"""The expected data and the five covariance terms on the counted entries.

    ``networks`` holds one network per Stokes term, the same but for its
    response moment. :math:`\Sigma^{\rm count}` is the part of the intensity
    term at the mean source power and response.
    """
    mean = expected_data(prediction, networks["intensity"])
    count = jnp.outer(mean, mean) / num_sources
    connected = {
        term: catalog_covariance(prediction, network)
        for term, network in networks.items()
    }
    return mean, {
        "detector": jnp.diag(networks["intensity"].scale ** 2),
        "count": count,
        "intensity": connected["intensity"] - count,
        "linear": connected["linear"],
        "circular": connected["circular"],
    }


def latex_table(table: pd.DataFrame) -> str:
    """The paper's per-network variance budget table."""
    num_sources = float(table["num_sources"].iloc[0])
    columns = len(COVARIANCE_TERMS)
    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        (
            r"\caption{Amplitude uncertainty and variance shares at the fiducial "
            r"parameters for $T=1$~year. The second column gives $\sigma_x/x$ from "
            r"eq.~\eqref{eq:amplitude-relative-uncertainty}. The remaining columns "
            r"give the percentage of its square contributed by the detector-noise "
            r"covariance $\Sigma^g$ and by each term of the connected covariance in "
            r"eq.~\eqref{eq:connected-covariance-decomposition}: the variance "
            r"$\mathbf w^\top\Sigma^X\mathbf w$ of each term projected on the "
            r"full-covariance amplitude weights "
            r"$\mathbf w=\Sigma^{-1}\bm\mu/(\bm\mu^\top\Sigma^{-1}\bm\mu)$, which "
            r"sum to $(\sigma_x/x)^2$. Each share is also the fractional change "
            r"in the variance per fractional change in its term. The expected event "
            rf"count is $N\simeq{num_sources / 1e5:.1f}\times10^5$.}}"
        ),
        r"\label{tab:shot-noise-budget}",
        r"\begin{tabular}{l" + "r" * (columns + 1) + "}",
        r"\toprule",
        rf"& & \multicolumn{{{columns}}}{{c}}{{Share of $(\sigma_x/x)^2$ (\%)}} \\",
        rf"\cmidrule(lr){{3-{columns + 2}}}",
        " & ".join(
            [r"Detector Network", r"$\sigma_x/x$"]
            + [COLUMN_HEADERS[term] for term in COVARIANCE_TERMS]
        )
        + r" \\",
        r"\midrule",
    ]
    for row in table.itertuples():
        cells = [
            row.network,
            f"{100 * row.sigma_x:.2f}\\%",
            *(
                f"{100 * getattr(row, f'share_{term}'):.1f}"
                for term in COVARIANCE_TERMS
            ),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tex-only", action="store_true", help="rewrite the table from the CSV"
    )
    parser.add_argument(
        "--ensemble",
        type=Path,
        default=Path("outputs/ensembles") / "fiducial",
        help="ensemble root holding the draws (default: %(default)s)",
    )
    parser.add_argument(
        "--reference-catalog",
        type=Path,
        default=Path("outputs/catalogs") / "reference-fiducial.h5",
        help="(default: %(default)s)",
    )
    args = parser.parse_args()
    if args.tex_only:
        table = pd.read_csv(OUTPUT / "shot_noise_budget.csv")
        (OUTPUT / "shot_noise_budget.tex").write_text(latex_table(table))
        return
    config = load_config(ROOT_DIR / "config")
    background = BackgroundArtifact.open(args.ensemble / f"{DATA_SEED}-0.h5")
    noise = background.noise()
    frequencies = background.frequencies
    model = build_model(
        args.reference_catalog,
        frequencies,
        background.config.population.observation_time,
    )
    prediction: Prediction = model.predict(config.fiducials)
    num_sources = float(prediction.extras["total_merger_rate"]) * float(
        model.observation_seconds
    )
    angles = parity_symmetric_angles(jax.random.key(RESPONSE_SEED), RESPONSE_DRAWS)

    rows = []
    for name, label in DETECTOR_NETWORKS:
        detectors = config.registry.build_network(name)

        def network(moment: ArrayLike | None = None, name: str = name) -> Network:
            return pair_network(
                frequencies,
                noise[name],
                minimum_frequency=MINIMUM_FREQUENCY,
                maximum_frequency=MAXIMUM_FREQUENCY,
                response_moment=moment,
            )

        moments = {
            term: response_moment(term_response(frequencies, detectors, angles, mask))
            for term, mask in STOKES_TERMS.items()
        }
        full_moment = response_moment(
            term_response(frequencies, detectors, angles, np.ones(4))
        )
        # The Stokes cross terms cancel between parity images.
        closure = float(
            jnp.max(jnp.abs(sum(moments.values()) - full_moment))
            / jnp.max(jnp.abs(full_moment))
        )
        mean, covariances = covariance_terms(
            prediction,
            {term: network(moment) for term, moment in moments.items()},
            num_sources,
        )
        budget = amplitude_variance_budget(mean, covariances)
        variance = float(sum(budget.values()))

        # The detector-noise template, for comparison.
        full = network(full_moment)
        template, squared_snr = amplitude_template(prediction, full)
        template = template / squared_snr
        shot = float(shot_noise_variance(prediction, full))
        rows.append(
            {
                "network": label,
                "sigma_x": np.sqrt(variance),
                **{
                    f"share_{term}": float(budget[term]) / variance
                    for term in COVARIANCE_TERMS
                },
                "rho": float(np.sqrt(squared_snr)),
                "sigma_shot": np.sqrt(shot),
                "sigma_x_template": np.sqrt(1.0 / squared_snr + shot),
                "delta": float(shape_correction(prediction, full)),
                **{
                    f"template_share_{term}": float(
                        template @ covariances[term] @ template
                    )
                    / (1.0 / float(squared_snr) + shot)
                    for term in COVARIANCE_TERMS
                },
                "stokes_closure": closure,
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
