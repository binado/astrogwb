"""Analyze one fixed-hyperparameter spectrum ensemble and one detector network.

Repeat --config for the four shared layers, then the case's simulation layers.
The checked spectrum cache is reused, or populated on a miss. --cache-only
requires an existing artifact. Detector settings do not enter the draw's key.
See docs/paper-figures.md for examples and the distributions' interpretation.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb import __version__
from astrogwb.catalog import SpectrumGenerator, simulate
from astrogwb.metadata import SpectraMetadata, artifact_path
from astrogwb.paper.config.detectors import DetectorRegistry
from astrogwb.paper.config.runs import (
    BASE_OUT_DIR,
    SPECTRA_ROOT,
    add_config_arguments,
    load_merged_config,
)
from astrogwb.paper.runtime import configure_runtime
from astrogwb.paper.snr import compute_spectrum_snrs, summarize_spectrum_snrs

logger = logging.getLogger(__name__)
DEFAULT_NETWORK = "ET-2L-aligned-CE-Hanford"


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
    def from_merged(cls, raw: Mapping[str, Any], *, network: str) -> Self:
        """Extract the relevant resolved tables, omitting shared authoring data."""
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
                "detector_registry": {
                    "detectors": raw.get("detectors"),
                    "networks": raw.get("networks", {}),
                },
                "network": network,
                "minimum_frequency": analysis.get("minimum_frequency"),
                "maximum_frequency": analysis.get("maximum_frequency"),
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(
        parser,
        help="One config layer in merge order; repeat for shared and simulation layers.",
    )
    parser.add_argument("--network", default=DEFAULT_NETWORK, metavar="NAME")
    parser.add_argument("--spectra-dir", type=Path, default=SPECTRA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory (default: outputs/snr/<spectrum-key>/<network>).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--cache-only",
        "--cached-only",
        dest="cache_only",
        action="store_true",
        help="Require a checked cache hit; do not generate missing spectra.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args(argv)
    config = SNRConfig.from_merged(load_merged_config(args), network=args.network)
    generator = SpectrumGenerator(batch_size=args.batch_size)
    spectra_dir = args.spectra_dir.expanduser().resolve()
    try:
        catalog = simulate(config.spectra, generator, spectra_dir, generate=False)
    except FileNotFoundError:
        if args.cache_only:
            raise
        catalog = None

    configure_runtime(num_chains=1)
    if catalog is None:
        catalog = simulate(config.spectra, generator, spectra_dir)
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

    from astrogwb.paper.plotting import save_figures, use_paper_style
    from astrogwb.paper.plotting.snr import plot_snr_histogram

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
        "spectrum_key": config.spectra.key(),
        "source_path": str(artifact_path(config.spectra, spectra_dir)),
        "config_paths": [str(path.expanduser().resolve()) for path in args.config],
        "selected_detectors": {
            name: config.detector_registry.detectors[name].model_dump(mode="json")
            for name in members
        },
        "analysis_version": __version__,
        "distribution": distribution_label,
        "detector_noise_realizations": False,
        "inclination_convention": (
            "Defined by the recorded population model and version; "
            "absent inclination uses analytic averaging."
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    use_paper_style()
    figure = plot_snr_histogram(
        snrs, network=config.network, distribution_label=distribution_label
    )
    try:
        save_figures({output_dir / "snr_histogram": figure})
    finally:
        plt.close(figure)
    logger.info("Saved SNR analysis for %s to %s", config.spectra.key(), output_dir)


if __name__ == "__main__":
    main()
