from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import ArrayLike


def uniform_mass_pair_model(
    parameters: Mapping[str, ArrayLike],
) -> tuple[jax.Array, jax.Array]:
    """Ordered pair of i.i.d. uniforms on ``[minimum_mass, minimum_mass + width]``."""
    minimum_mass: jax.Array = jnp.asarray(parameters["minimum_mass"])
    mass_width: jax.Array = jnp.asarray(parameters["mass_width"])
    # For two ordered iid uniforms, Beta(2, 1) is the primary mass marginal.
    mass_1 = numpyro.sample(
        "source_frame_mass_1",
        dist.TransformedDistribution(
            dist.Beta(2.0, 1.0, validate_args=True),
            dist.transforms.AffineTransform(minimum_mass, mass_width),
            validate_args=True,
        ),
    )
    mass_2 = numpyro.sample(
        "source_frame_mass_2",
        dist.Uniform(minimum_mass, mass_1, validate_args=True),
    )
    return jnp.asarray(mass_1), jnp.asarray(mass_2)
