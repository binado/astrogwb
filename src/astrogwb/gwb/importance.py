r"""The background spectrum as a redshift quadrature over an intrinsic sample.

The spectrum is an expectation over sources. Only the intrinsic parameters
are left to Monte Carlo here: redshift is integrated on fixed Gauss-Legendre
nodes, and inclination is averaged exactly. Every one of :math:`N` intrinsic
draws is placed at each of :math:`Z` nodes, so

.. math::

    S(f; \Lambda) = R(\Lambda) \sum_j w_j\, p(z_j \mid \Lambda)
        \left[\frac{d_j^{\mathrm{ref}}}{d_L(z_j \mid \Lambda)}\right]^2
        \frac{1}{N} \sum_k \omega_k(\Lambda)\, P(f; \theta_k, z_j),

with :math:`P` the power stored at reference distance :math:`d^{\mathrm{ref}}`
-- the fiducial luminosity distance of node :math:`j` -- and
:math:`\omega_k` the ratio of the target to the drawn intrinsic density
(one when the intrinsic hyperparameters are not varied). Redshift has no
Monte Carlo variance left: the :math:`1/d_L^2` tail that dominated a
redshift-sampled proposal is integrated exactly.

The waveform is generated in the observer frame at each node -- detector-frame
masses, observed frequencies -- so nothing assumes how power scales with
redshift. Inclination is fixed at :data:`EFFECTIVE_INCLINATION`, where the
quadrupolar factor equals its isotropic mean; that is exact only for waveforms
carrying the :math:`(2, 2)` mode alone.

:func:`importance_catalog` draws the table, and
:func:`build_importance_spectrum` binds it to a target population.

Contracts the caller is trusted to honour (nothing here checks them):

- The population's intrinsic sites are independent of redshift and
  inclination, and inclination is isotropic.
- The approximant carries the :math:`(2, 2)` mode only.
- ``density_sites`` names intrinsic sites only: redshift is integrated, never
  weighted.
- The catalog is never narrowed with
  :func:`~astrogwb.simulators.polarization_power.restrict_redshift`: its nodes
  are fixed by the population's redshift window.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Annotated, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from astrogwb import __version__
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.inference.protocol import SpectralDensityFn
from astrogwb.populations._types import Population, PopulationModel
from astrogwb.populations.evaluation import evaluate_sources, sample_sources
from astrogwb.populations.metadata import PopulationMetadata, widen_model_kwargs
from astrogwb.simulators.core.keys import content_key
from astrogwb.simulators.polarization_power.restrict import (
    REDSHIFT_SITE,
    REDSHIFT_WINDOW_KWARGS,
)
from astrogwb.simulators.polarization_power.simulator import (
    PolarizationPowerData,
    polarization_power_data,
)
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = [
    "EFFECTIVE_INCLINATION",
    "ImportanceCatalogMetadata",
    "LogWeightsFn",
    "build_importance_spectrum",
    "importance_catalog",
    "importance_catalog_stem",
    "node_sources",
    "redshift_quadrature",
]

logger = logging.getLogger(__name__)

#: The inclination site, fixed rather than drawn.
INCLINATION_SITE = "inclination"

#: The inclination where the quadrupolar factor
#: :math:`g(\iota) = ((1 + \cos^2\iota)/2)^2 + \cos^2\iota` equals its
#: isotropic mean, 4/5: :math:`\cos^2\iota^* = \sqrt{11.2} - 3`.
EFFECTIVE_INCLINATION = float(np.arccos(np.sqrt(np.sqrt(11.2) - 3.0)))

#: The luminosity distance a source model returns.
_LUMINOSITY_DISTANCE = "luminosity_distance"

#: Per-draw log importance weights from hyperparameters alone, shape ``(N,)``.
type LogWeightsFn = Callable[[Mapping[str, ArrayLike]], jax.Array]


def redshift_quadrature(
    minimum_redshift: float, maximum_redshift: float, num_nodes: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Gauss-Legendre nodes and weights for :math:`\int dz` over a window.

    The rule is Gauss-Legendre in :math:`u = \ln(1 + z)`, which spreads the
    nodes evenly over the decades of :math:`1 + z`; the Jacobian
    :math:`dz/du = 1 + z` is folded into the weights, so
    :math:`\int f\, dz \approx \sum_j w_j f(z_j)`.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(num_nodes)
    lower, upper = np.log1p(minimum_redshift), np.log1p(maximum_redshift)
    half_width = 0.5 * (upper - lower)
    one_plus_z = np.exp(lower + half_width * (nodes + 1.0))
    return one_plus_z - 1.0, half_width * weights * one_plus_z


class ImportanceCatalogMetadata(BaseModel):
    """Everything that determines an importance catalog's contents.

    ``num_samples`` intrinsic draws from ``population`` at ``fiducials``, each
    placed at ``num_redshift_nodes`` nodes of :func:`redshift_quadrature` over
    the population's redshift window, and pushed through ``waveform``. Like
    :class:`~astrogwb.simulators.polarization_power.CatalogMetadata` it is both
    the request and the provenance; the seed is not part of it.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    population: PopulationMetadata
    fiducials: dict[str, float]
    num_samples: Annotated[int, Field(gt=0)]
    num_redshift_nodes: Annotated[int, Field(gt=0)]
    version: str = __version__

    def nodes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The redshift nodes and weights, over the population's window."""
        kwargs = self.population.model_kwargs
        minimum, maximum = (float(kwargs[name]) for name in REDSHIFT_WINDOW_KWARGS)
        return redshift_quadrature(minimum, maximum, self.num_redshift_nodes)

    def key(self) -> str:
        """The content hash of the canonical, widened record."""
        payload = self.model_dump(mode="json")
        widen_model_kwargs(payload["population"])
        return content_key(payload)


def importance_catalog_stem(
    metadata: ImportanceCatalogMetadata, seed: int | np.integer
) -> str:
    """The file stem by convention: ``importance_catalog-<key>-<seed>``."""
    return f"importance_catalog-{metadata.key()}-{int(seed)}"


def node_sources(
    model: PopulationModel,
    source_parameters: Mapping[str, ArrayLike],
    redshift_nodes: ArrayLike,
) -> dict[str, NDArray[Any]]:
    """Every source at every redshift node, at :data:`EFFECTIVE_INCLINATION`.

    Returns ``N * Z`` rows, node-major: row ``j * N + k`` is source ``k`` at
    node ``j``. The rows are replayed through ``model``, so every column that
    follows from redshift -- luminosity distance, detector-frame masses -- is
    the model's own recomputation, not a stale copy.
    """
    columns = {name: np.asarray(values) for name, values in source_parameters.items()}
    nodes = np.asarray(redshift_nodes, dtype=np.float64)
    tiled = {name: np.tile(values, nodes.size) for name, values in columns.items()}
    count = len(columns[REDSHIFT_SITE])
    tiled[REDSHIFT_SITE] = np.repeat(nodes, count)
    tiled[INCLINATION_SITE] = np.full(count * nodes.size, EFFECTIVE_INCLINATION)
    _, outputs = evaluate_sources(model, tiled, density_sites=())
    return {name: np.asarray(values) for name, values in outputs.items()}


def importance_catalog(
    metadata: ImportanceCatalogMetadata,
    key: jax.Array,
    *,
    chunk_size: int | None = None,
) -> PolarizationPowerData:
    """The importance catalog ``key`` draws: power ``(F, Z * N)``, node-major.

    ``chunk_size`` bounds the sources one batch of the waveform generates; see
    :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate_batch`.
    """
    # x64 before the draw, as for a plain catalog: see draw_catalog.
    jax.config.update("jax_enable_x64", True)
    if metadata.version != __version__:
        raise ValueError(
            f"metadata is for astrogwb {metadata.version}, but {__version__} "
            "is installed; a catalog is generated by the code its key names"
        )
    logger.info(
        "Importance catalog %s: population=%s num_samples=%d num_redshift_nodes=%d",
        metadata.key(),
        metadata.population.model_name,
        metadata.num_samples,
        metadata.num_redshift_nodes,
    )
    _, model = metadata.population.build()(metadata.fiducials)
    samples = sample_sources(model, key, num_samples=metadata.num_samples)
    nodes, _ = metadata.nodes()
    return polarization_power_data(
        metadata.waveform.build(),
        node_sources(model, samples, nodes),
        chunk_size=chunk_size,
    )


def _with_extrinsic(
    columns: Mapping[str, jax.Array], redshift: ArrayLike, count: int
) -> dict[str, jax.Array]:
    """``columns`` broadcast to ``count`` rows, with redshift and inclination set."""
    out = {name: jnp.broadcast_to(values, (count,)) for name, values in columns.items()}
    out[REDSHIFT_SITE] = jnp.broadcast_to(jnp.asarray(redshift), (count,))
    out[INCLINATION_SITE] = jnp.full((count,), EFFECTIVE_INCLINATION)
    return out


def _intrinsic_log_prob(
    model: PopulationModel,
    intrinsic: Mapping[str, jax.Array],
    redshift: jax.Array,
    density_sites: Sequence[str],
) -> jax.Array:
    count = next(iter(intrinsic.values())).shape[0]
    log_prob, _ = evaluate_sources(
        model,
        _with_extrinsic(intrinsic, redshift, count),
        density_sites=density_sites,
    )
    return log_prob


def evaluate_log_weights(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    intrinsic: Mapping[str, jax.Array],
    redshift: jax.Array,
    proposal_log_prob: jax.Array,
    density_sites: Sequence[str],
) -> jax.Array:
    """Per-draw intrinsic log importance weights at ``params``, shape ``(N,)``."""
    _, model = population(params)
    target = _intrinsic_log_prob(model, intrinsic, redshift[0], density_sites)
    return target - proposal_log_prob


def importance_spectral_density(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    polarization_power: jax.Array,
    intrinsic: Mapping[str, jax.Array],
    redshift: jax.Array,
    redshift_weights: jax.Array,
    reference_distance: jax.Array,
    proposal_log_prob: jax.Array,
    density_sites: Sequence[str],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The spectrum at ``params``, shape ``(F,)``, with its diagnostics.

    ``extras`` holds ``total_merger_rate`` (observer frame, mergers per second)
    and ``importance_relative_ess`` over the intrinsic draws, both shape ``()``.
    """
    merger_rate, model = population(params)
    total_merger_rate = jnp.reshape(jnp.asarray(merger_rate), ())
    num_nodes = redshift.shape[0]
    first = {name: values[0] for name, values in intrinsic.items()}
    log_density, outputs = evaluate_sources(
        model,
        _with_extrinsic(first, redshift, num_nodes),
        density_sites=(REDSHIFT_SITE,),
    )
    distance_ratio = reference_distance / outputs[_LUMINOSITY_DISTANCE]
    kernel = redshift_weights * jnp.exp(log_density) * distance_ratio**2
    log_weights = (
        _intrinsic_log_prob(model, intrinsic, redshift[0], density_sites)
        - proposal_log_prob
    )
    num_samples = log_weights.shape[0]
    spectrum = (
        total_merger_rate
        * jnp.einsum("fzn,z,n->f", polarization_power, kernel, jnp.exp(log_weights))
        / num_samples
    )
    return spectrum, {
        "total_merger_rate": total_merger_rate,
        "importance_relative_ess": relative_ess(log_weights),
    }


def build_importance_spectrum(
    data: PolarizationPowerData,
    metadata: ImportanceCatalogMetadata,
    *,
    population: Population,
    density_sites: Sequence[str],
) -> tuple[SpectralDensityFn, LogWeightsFn]:
    """Bind an importance catalog to a target, as both callables at once.

    Call outside JAX transformations. The drawn intrinsic density is the
    catalog's own population at its fiducials, evaluated once here.
    ``density_sites`` names the intrinsic factors the weights include; leave it
    empty when no intrinsic hyperparameter varies, and every weight is one.

    ``population`` is the target's bound
    :class:`~astrogwb.populations.Population`; build it once per run.
    """
    redshift, redshift_weights = metadata.nodes()
    num_nodes, num_samples = redshift.size, metadata.num_samples
    columns = data["source_parameters"]
    power = np.asarray(data["polarization_power"])
    power = power.reshape(power.shape[0], num_nodes, num_samples)
    # Node j's reference distance is its fiducial distance, the one its power
    # was generated at; it is the same for every draw at that node.
    reference_distance = np.asarray(columns[_LUMINOSITY_DISTANCE])[::num_samples]
    intrinsic = {
        name: jnp.asarray(np.asarray(values)[:num_samples])
        for name, values in columns.items()
        if name not in (REDSHIFT_SITE, INCLINATION_SITE)
    }
    redshift_nodes = jnp.asarray(redshift)
    _, fiducial_model = metadata.population.build()(metadata.fiducials)
    proposal_log_prob = _intrinsic_log_prob(
        fiducial_model, intrinsic, redshift_nodes[0], density_sites
    )
    shared: dict[str, Any] = {
        "population": population,
        "intrinsic": intrinsic,
        "redshift": redshift_nodes,
        "proposal_log_prob": proposal_log_prob,
        "density_sites": tuple(density_sites),
    }
    spectral_density_fn = partial(
        importance_spectral_density,
        polarization_power=jnp.asarray(power),
        redshift_weights=jnp.asarray(redshift_weights),
        reference_distance=jnp.asarray(reference_distance),
        **shared,
    )
    return spectral_density_fn, partial(evaluate_log_weights, **shared)
