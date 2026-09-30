"""Analyze one fixed-hyperparameter spectrum ensemble and one detector network.

Repeat --spectra-config for scientific and simulation layers, and
--detector-config for network definitions and detector overrides.
The checked spectrum cache is reused, or populated on a miss. --cache-only
requires an existing artifact. Detector settings do not enter the draw's key.
See docs/paper-figures.md for examples and the distributions' interpretation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb import __version__
from astrogwb.catalog import SpectrumGenerator, simulate
from astrogwb.detector import effective_psd
from astrogwb.frequency import frequency_mask
from astrogwb.gwb import spectral_snr
from astrogwb.metadata import SpectraMetadata
from astrogwb.paper.cache import default_cache_dir
from astrogwb.paper.config.detectors import DetectorRegistry, load_detector_config
from astrogwb.paper.config.runs import (
    BASE_OUT_DIR,
    merge_config_layers,
)
from astrogwb.paper.runtime import configure_runtime
from astrogwb.utils import years_to_seconds

logger = logging.getLogger(__name__)
DEFAULT_NETWORK = "ET-2L-aligned-CE-Hanford"

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from astrogwb.catalog import SpectralDensityCatalog


class SNRConfig(BaseModel):
    """Resolved scientific inputs, independent of the generation cache key."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    spectra: SpectraMetadata
    detector_registry: DetectorRegistry
    network: str
    minimum_frequency: Annotated[float, Field(ge=0.0)]
    maximum_frequency: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def _validate_study(self) -> Self:
        if self.minimum_frequency >= self.maximum_frequency:
            raise ValueError("minimum_frequency must be less than maximum_frequency")
        if not np.isfinite(self.spectra.observation_time):
            raise ValueError("spectra.observation_time must be finite and positive")
        if self.network not in self.detector_registry.networks:
            raise ValueError(
                f"unknown network {self.network!r}; known networks: "
                f"{sorted(self.detector_registry.networks)}"
            )
        if self.spectra.sampled:
            raise ValueError(
                "SNR studies require fixed hyperparameters; sampled parameters: "
                + ", ".join(sorted(self.spectra.sampled))
            )
        return self

    @classmethod
    def from_merged(
        cls,
        raw: Mapping[str, Any],
        *,
        detector_registry: DetectorRegistry,
        network: str,
    ) -> Self:
        """Combine resolved spectrum settings with an independently loaded registry."""
        if "spectra" not in raw:
            raise ValueError(
                "the merged config has no [spectra] table; add a "
                "config/simulations/spectrum/<name>.toml layer"
            )
        analysis = raw.get("analysis", {})
        if not isinstance(analysis, Mapping):
            raise TypeError("[analysis] must be a table")
        return cls.model_validate(
            {
                "spectra": raw["spectra"],
                "detector_registry": detector_registry,
                "network": network,
                "minimum_frequency": analysis.get("minimum_frequency"),
                "maximum_frequency": analysis.get("maximum_frequency"),
            }
        )


def compute_spectrum_snrs(
    catalog: SpectralDensityCatalog,
    detector_registry: DetectorRegistry,
    network: str,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
) -> tuple[NDArray[np.float64], float]:
    """Return per-realization SNRs and the mean spectrum's SNR for one network.

    The observing time comes from the artifact, in years. Both calculations
    use the full frequency grid and mask afterwards, preserving the widths
    of bins at the analysis band's edges and all within-row correlations.
    No source sampling or waveform generation is performed here.
    """
    if not (
        np.isfinite(minimum_frequency)
        and np.isfinite(maximum_frequency)
        and 0.0 <= minimum_frequency < maximum_frequency
    ):
        raise ValueError("frequency bounds must be finite and 0 <= minimum < maximum")
    if not np.isfinite(catalog.observation_time) or catalog.observation_time <= 0:
        raise ValueError("observation_time must be finite and positive")
    if catalog.frequencies.size < 2:
        raise ValueError("SNR calculation requires at least two frequency bins")
    if network not in detector_registry.networks:
        raise ValueError(f"unknown network {network!r}")

    frequencies = jnp.asarray(catalog.frequencies)
    band = frequency_mask(frequencies, fmin=minimum_frequency, fmax=maximum_frequency)
    if not np.any(np.asarray(band)):
        raise ValueError("the analysis frequency band contains no spectrum bins")
    geometry, sensitivities = detector_registry.build_network(network)
    noise = jnp.asarray(effective_psd(frequencies, geometry, sensitivities))
    spectra = jnp.asarray(catalog.spectral_density)
    seconds = years_to_seconds(catalog.observation_time)
    snrs = np.asarray(
        spectral_snr(spectra, noise, seconds, frequencies, frequency_mask=band),
        dtype=np.float64,
    )
    mean_spectrum_snr = float(
        spectral_snr(
            jnp.mean(spectra, axis=0),
            noise,
            seconds,
            frequencies,
            frequency_mask=band,
        )
    )
    if not np.all(np.isfinite(snrs)) or not np.isfinite(mean_spectrum_snr):
        raise ValueError("SNR calculation produced nonfinite results")
    return snrs, mean_spectrum_snr


def summarize_spectrum_snrs(
    snrs: ArrayLike, *, mean_spectrum_snr: float
) -> dict[str, int | float]:
    """Summarize realization SNRs, keeping the mean-spectrum statistic separate.

    A single draw has undefined sample SD; relative scatter is undefined when
    the mean is zero or SD is undefined. These values are returned as NaN,
    which table writers can represent as missing values.
    """
    values = np.asarray(snrs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("snrs must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("snrs must be finite and nonnegative")
    if not np.isfinite(mean_spectrum_snr) or mean_spectrum_snr < 0:
        raise ValueError("mean_spectrum_snr must be finite and nonnegative")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    return {
        "num_draws": int(values.size),
        "mean": mean,
        "median": float(np.median(values)),
        "sd": sd,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "relative_scatter": sd / mean if mean > 0 else float("nan"),
        "mean_spectrum_snr": mean_spectrum_snr,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectra-config",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help="One spectrum config layer in merge order; repeat for scientific and simulation layers.",
    )
    parser.add_argument(
        "--detector-config",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help="One detector registry layer in merge order; repeat for network and detector overrides.",
    )
    parser.add_argument("--network", default=DEFAULT_NETWORK, metavar="NAME")
    cache_dir = default_cache_dir() / "spectra"
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=cache_dir,
        metavar="DIR",
        help=f"Checked spectrum cache directory (default: {cache_dir}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory (default: outputs/snr/<spectrum-key>/<network>).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--plot-param-fisher-scatter",
        dest="plot_param_fisher_scatter",
        action="extend",
        nargs="+",
        default=[],
        metavar="PARAM",
        help=(
            "Also plot the Fisher-scatter distribution of each named fixed "
            "hyperparameter, its fiducial value divided by the per-draw SNR; "
            "accepts several names and may be repeated."
        ),
    )
    parser.add_argument(
        "--cache-only",
        dest="cache_only",
        action="store_true",
        help="Require a checked cache hit; do not generate missing spectra.",
    )
    return parser.parse_args(argv)


def plot_distribution(values: ArrayLike, *, dataset_label: str, xlabel: str) -> Figure:
    """Plot the per-draw density of ``values`` under the active paper style.

    ``dataset_label`` names the variable inside the plotted dataset; ``xlabel``
    labels the x-axis. Constant ensembles (including a single draw) use an
    ECDF because a KDE requires nonzero scatter. These are simulation draws,
    not posterior chains.
    """
    # Presentation dependencies are lazy, like main's: importing/validating
    # config leaves the XLA backend configurable and does not require a
    # matplotlib session.
    import arviz_plots as azp
    import xarray as xr
    from matplotlib.axes import Axes

    values = np.asarray(values, dtype=np.float64)
    kind = "ecdf" if np.ptp(values) == 0 else "kde"
    collection = azp.plot_dist(
        xr.DataTree.from_dict(
            {"simulations": xr.Dataset({dataset_label: ("draw", values)})}
        ),
        group="simulations",
        sample_dims=["draw"],
        kind=kind,
        backend="matplotlib",
        visuals={
            "credible_interval": False,
            "point_estimate": False,
            "point_estimate_text": False,
            "title": False,
            "remove_axis": False,
        },
        figure_kwargs={
            "figsize": (6.4, 4.8),
            "layout": "constrained",
            # gwpy registers replacement default axes that ArviZ cannot
            # identify, so restore the matplotlib axes class.
            "subplot_kws": {"axes_class": Axes},
        },
    )
    figure = collection.viz["figure"].item()
    axis = figure.axes[0]
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Cumulative probability" if kind == "ecdf" else "Density")
    axis.set_ylim(bottom=0)
    return figure


def fisher_scatter_xlabel(latex: str) -> str:
    """The LaTeX x-label for a parameter's Fisher scatter: sigma of its label.

    The unit bracket of a parameter label (``$H_0\\,[\\mathrm{...}]$``) is
    dropped: it has no meaningful place inside a subscript.
    """
    core = latex.split(r"\,[", 1)[0].strip("$")
    return rf"$\sigma_{{{core}}}$"


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args(argv)
    logger.info(
        "Spectrum config layers (merge order): %s",
        " -> ".join(str(path) for path in args.spectra_config),
    )
    config = SNRConfig.from_merged(
        merge_config_layers(args.spectra_config),
        detector_registry=load_detector_config(args.detector_config),
        network=args.network,
    )
    fixed_hyperparameters = config.spectra.fixed
    unknown_fisher_params = [
        param
        for param in args.plot_param_fisher_scatter
        if param not in fixed_hyperparameters
    ]
    if unknown_fisher_params:
        raise ValueError(
            f"unknown Fisher-scatter parameter(s) {sorted(unknown_fisher_params)}; "
            f"fixed hyperparameters: {sorted(fixed_hyperparameters)}"
        )
    generator = SpectrumGenerator(batch_size=args.batch_size)
    cache_dir = args.cache_dir.expanduser().resolve()
    try:
        catalog = simulate(config.spectra, generator, cache_dir, generate=False)
    except FileNotFoundError:
        if args.cache_only:
            raise
        catalog = None

    configure_runtime(num_chains=1)
    if catalog is None:
        catalog = simulate(config.spectra, generator, cache_dir)
    snrs, mean_spectrum_snr = compute_spectrum_snrs(
        catalog,
        config.detector_registry,
        config.network,
        minimum_frequency=config.minimum_frequency,
        maximum_frequency=config.maximum_frequency,
    )
    summary = summarize_spectrum_snrs(snrs, mean_spectrum_snr=mean_spectrum_snr)
    distribution_label = (
        "Finite-catalog estimator scatter"
        if config.spectra.count == "fixed"
        else "Finite-observation realizations"
    )

    # Analysis dependencies are lazy: importing/validating config leaves the
    # XLA backend configurable and does not require a matplotlib session.
    import matplotlib.pyplot as plt
    import pandas as pd

    from astrogwb.paper.plotting import (
        parameter_label,
        save_figures,
        use_paper_style,
    )

    output_dir = (
        (
            args.output_dir
            if args.output_dir is not None
            else BASE_OUT_DIR / "snr" / config.spectra.key() / config.network
        )
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "spectrum_key": config.spectra.key(),
        "network": config.network,
        "distribution": distribution_label,
    }
    pd.DataFrame(
        {
            **identity,
            "draw_index": np.arange(catalog.num_draws),
            "n_events": catalog.n_events,
            "snr": snrs,
        }
    ).to_csv(output_dir / "snr_draws.csv", index=False)
    members = config.detector_registry.networks[config.network]
    pd.DataFrame(
        [
            {
                **identity,
                "detectors": ",".join(members),
                "n_detectors": len(members),
                **summary,
            }
        ]
    ).to_csv(output_dir / "snr_summary.csv", index=False)
    provenance = {
        "config": config.model_dump(mode="json"),
        "cli_flags": list(sys.argv[1:] if argv is None else argv),
        "version": __version__,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    use_paper_style()
    figures = {
        output_dir / "snr_distribution": plot_distribution(
            snrs, dataset_label="snr", xlabel="SNR"
        )
    }
    for param in args.plot_param_fisher_scatter:
        figures[output_dir / f"{param}_fisher_scatter"] = plot_distribution(
            np.asarray(fixed_hyperparameters[param], dtype=np.float64) / snrs,
            dataset_label=f"fisher_scatter_{param}",
            xlabel=fisher_scatter_xlabel(parameter_label(param)),
        )
    try:
        save_figures(figures)
    finally:
        for figure in figures.values():
            plt.close(figure)
    logger.info("Saved SNR analysis for %s to %s", config.spectra.key(), output_dir)


if __name__ == "__main__":
    main()
