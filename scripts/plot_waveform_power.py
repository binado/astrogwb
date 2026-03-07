"""Plot scaled waveform power versus frequency from a spectral density file.

Usage:
    python scripts/plot_waveform_power.py \
        --input-file /path/to/sum_abs_sq.h5 \
        --h0 67.7 \
        --t 1.0 \
        --x-tick-labels '["1","10","100","1000"]' \
        --y-tick-labels '["1e-12","1e-10","1e-8"]' \
        --output /path/to/sum_abs_sq_f3_scaled.png
"""

import logging
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from utils import get_config_filepath

from asgwb.gwb import SpectralDensity

logger = logging.getLogger(__name__)

MPC_IN_METERS = 3.085677581491367e22
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


class PlotWaveformPowerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )

    input_file: Path
    output: Path = Path("sum_abs_sq_f3_scaled.png")
    h0: Annotated[float, Field(gt=0)]  # km / s / Mpc
    t: Annotated[float, Field(gt=0)]  # years
    x_tick_labels: list[str] | None = None
    y_tick_labels: list[str] | None = None
    title: str | None = None
    show: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        config_filepath = get_config_filepath(Path(__file__))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=config_filepath),
            file_secret_settings,
        )


def plot_power(
    frequency: np.ndarray,
    sum_abs_sq: np.ndarray,
    h0_km_s_mpc: float,
    t_years: float,
    output_path: Path,
    x_tick_labels: list[str] | None = None,
    y_tick_labels: list[str] | None = None,
    title: str | None = None,
    show: bool = False,
) -> None:
    h0_si = h0_km_s_mpc * 1000.0 / MPC_IN_METERS
    t_si = t_years * SECONDS_PER_YEAR

    prefactor = (4.0 * np.pi**2) / (3.0 * h0_si**2) / t_si
    # prefactor = 1 / t_si
    y = sum_abs_sq * frequency**3 * prefactor
    # y = sum_abs_sq * prefactor
    mask = (frequency > 0.0) & (y > 0.0)
    if not np.any(mask):
        raise ValueError("No positive points available for log-log plotting")

    fig, ax = plt.subplots()
    ax.loglog(frequency[mask], y[mask], linewidth=1.2)
    ax.set_xlabel(r"$f \, \rm{[Hz]}$")
    ax.set_ylabel(r"$\Omega_{\rm{GW}}(f)$")
    if x_tick_labels:
        try:
            x_ticks = [float(label) for label in x_tick_labels]
        except ValueError as exc:
            raise ValueError(
                "Each x_tick_labels entry must be parseable as a positive float"
            ) from exc
        if any(tick <= 0.0 for tick in x_ticks):
            raise ValueError("All x_tick_labels values must be > 0 for log scale")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels:
        try:
            y_ticks = [float(label) for label in y_tick_labels]
        except ValueError as exc:
            raise ValueError(
                "Each y_tick_labels entry must be parseable as a positive float"
            ) from exc
        if any(tick <= 0.0 for tick in y_ticks):
            raise ValueError("All y_tick_labels values must be > 0 for log scale")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

    # Use the first unmasked element to set the bottom limit
    y_first_valid = y[mask][0]
    ax.set_ylim(bottom=1e-1 * y_first_valid, top=None)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    logger.info("Saved plot to %s", output_path)

    if show:
        plt.show()
    plt.close(fig)


def run(settings: PlotWaveformPowerSettings) -> None:
    logger.info("Running with settings:")
    for key, value in settings.model_dump().items():
        logger.info("\t%s = %s", key, value)
    h0_si = settings.h0 * 1000.0 / MPC_IN_METERS
    t_si = settings.t * SECONDS_PER_YEAR
    logger.info("\th0_si [1/s] = %s", h0_si)
    logger.info("\tt_si [s] = %s", t_si)

    spectral_density = SpectralDensity.load(settings.input_file)
    frequency = spectral_density.grid.frequencies
    sum_abs_sq = spectral_density.spectral_density
    plot_power(
        frequency=frequency,
        sum_abs_sq=sum_abs_sq,
        h0_km_s_mpc=settings.h0,
        t_years=settings.t,
        output_path=settings.output,
        x_tick_labels=settings.x_tick_labels,
        y_tick_labels=settings.y_tick_labels,
        title=settings.title,
        show=settings.show,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run(settings=PlotWaveformPowerSettings())


if __name__ == "__main__":
    main()
