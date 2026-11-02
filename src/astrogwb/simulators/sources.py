"""Fixed-size source draws from a population model, shared by every simulation.

:func:`simulate_sources` is the one place a population model becomes source
columns: a Poisson realization
(:func:`~astrogwb.simulators.simulate_population`), a plain catalog
(:func:`~astrogwb.background.catalog.simulate_catalog`) and a reference catalog
(:func:`~astrogwb.background.importance.reference_catalog`) differ only in the model
they hand it and how many sources they ask for.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import numpy as np
from numpy.typing import NDArray

from astrogwb.populations._types import PopulationModel
from astrogwb.utils.numpyro import sample_and_logprob, sample_sources_qmc

__all__ = ["Sampling", "simulate_sources"]

#: How sources are drawn: i.i.d., or on a scrambled Sobol net.
type Sampling = Literal["random", "sobol"]


def simulate_sources(
    model: PopulationModel,
    key: jax.Array,
    num_samples: int,
    *,
    sampling: Sampling = "random",
) -> dict[str, NDArray[Any]]:
    """``num_samples`` sources of ``model``, one ``(num_samples,)`` column per site.

    ``"sobol"`` draws on a scrambled Sobol net
    (:func:`~astrogwb.utils.numpyro.sample_sources_qmc`) and needs a
    power-of-two ``num_samples``.

    Parameters
    ----------
    model
        No-argument NumPyro model declaring the source sites, such as
        :func:`~astrogwb.populations.joint_model`.
    key
        JAX random key.
    num_samples
        Number of sources.
    sampling
        ``"random"`` or ``"sobol"``.
    """
    if sampling == "sobol":
        samples = sample_sources_qmc(model, key, num_samples=num_samples)
    else:
        samples, _ = sample_and_logprob(
            model, key, num_samples=num_samples, density_sites=()
        )
    return {name: np.asarray(values) for name, values in samples.items()}
