"""Draw the CoBA BNS events the pygwb reference spectrum is computed for.

Run in the project environment, before ``generate_pygwb.py``::

    uv run python tests/parity/references/draw_coba_events.py

Writes ``coba_bns_events.json`` next to this file: the source columns of a
seeded population realization, in this package's names and units. The file is
the shared input of both sides of
the comparison, so neither side depends on the other's sampler.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from astrogwb.populations import BNSCobaPopulation
from astrogwb.simulators import RealizationMetadata, simulate_population
from astrogwb.simulators.population import Hyperparameter

OUTPUT = Path(__file__).with_name("coba_bns_events.json")

SEED = 20261116
OBSERVATION_TIME = 1e-5  # years

#: The columns an aligned-spin point-particle waveform reads.
COLUMNS = (
    "source_frame_mass_1",
    "source_frame_mass_2",
    "redshift",
    "luminosity_distance",
    "spin_1z",
    "spin_2z",
    "inclination",
    "coa_phase",
)

HYPERPARAMETERS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}


def main() -> None:
    """Draw the events and write them as JSON."""
    jax.config.update("jax_enable_x64", True)
    hyperparameters: dict[str, Hyperparameter] = dict(HYPERPARAMETERS)
    metadata = RealizationMetadata(
        population=BNSCobaPopulation(
            mass_model="uniform", minimum_redshift=0.05, maximum_redshift=10.0
        ),
        hyperparameters=hyperparameters,
        observation_time=OBSERVATION_TIME,
    )
    samples = simulate_population(metadata, jax.random.key(SEED)).source_parameters
    events = {name: np.asarray(samples[name], dtype=float).tolist() for name in COLUMNS}
    payload = {
        "population": "bns_coba, uniform masses",
        "hyperparameters": HYPERPARAMETERS,
        "seed": SEED,
        "observation_time_years": OBSERVATION_TIME,
        "events": events,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
