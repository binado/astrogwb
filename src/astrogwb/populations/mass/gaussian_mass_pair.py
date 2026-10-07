from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import ArrayLike

from astrogwb.distributions import MaxOfTwoNormalsDistribution


def gaussian_mass_pair_model(
    parameters: Mapping[str, ArrayLike],
) -> tuple[jax.Array, jax.Array]:
    """Ordered pair of i.i.d. ``Normal(mass_mean, mass_sigma)`` components."""
    mass_mean: jax.Array = jnp.asarray(parameters["mass_mean"])
    mass_sigma: jax.Array = jnp.asarray(parameters["mass_sigma"])
    mass_1 = numpyro.sample(
        "source_frame_mass_1",
        MaxOfTwoNormalsDistribution(mass_mean, mass_sigma, validate_args=True),
    )
    mass_2 = numpyro.sample(
        "source_frame_mass_2",
        dist.TruncatedNormal(mass_mean, mass_sigma, high=mass_1, validate_args=True),
    )
    return jnp.asarray(mass_1), jnp.asarray(mass_2)
