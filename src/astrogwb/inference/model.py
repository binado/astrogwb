r"""What the dense Gaussian likelihood needs from a background model.

For the dominant quadrupole, the likelihood of the stacked cross-correlation
data depends on the population through two moments of the source intensity
:math:`I(f)`: the spectral density, which sets the mean,

.. math::

    S_h(f) = 2\mathcal R\,\langle I(f)\rangle,

and the raw intensity second moment, which sets the connected (shot-noise)
covariance of a catalog realization over :math:`T = N_{\rm seg}\tau`,

.. math::

    \frac{1}{N_{\rm seg}}\Sigma^c_{ab,cd}(f, f') = \frac{1}{T}
        \left[4\mathcal R\,\langle I(f)\, I(f')\rangle\right]
        \Gamma^c_{ab,cd}(f, f'),

with :math:`\mathcal R` the total observer-frame merger rate and
:math:`\Gamma^c` the network's connected overlap
(:func:`~astrogwb.detector.response.connected_overlap`). The averages run over
inclination, polarization, phase, intrinsic parameters and redshift. A model
implements :class:`SpectralDensityModel`; the likelihood supplies the network
and :math:`T`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, Protocol

import jax
from jax.typing import ArrayLike

__all__ = ["SpectralDensityModel", "SpectralDensityPrediction"]


class SpectralDensityPrediction(NamedTuple):
    r"""A background model's moments at one parameter point.

    Attributes
    ----------
    spectral_density
        :math:`S_h(f_k) = 2\mathcal R\langle I(f_k)\rangle`, ``(F,)``.
    intensity_second_moment
        :math:`4\mathcal R\langle I(f_k)\, I(f_l)\rangle`, ``(F, F)``, or
        ``None`` when shot noise is not predicted. A raw moment, not a
        covariance: it includes the count fluctuations. Per unit time, in
        :math:`S_h^2` seconds, and symmetric positive semidefinite up to
        round-off.
    extras
        Scalar diagnostics, with the same keys at every parameter point.
    merger_rate
        The observer-frame merger rate :math:`\mathcal R` in mergers per
        second, ``()``, or ``None`` when the model does not predict it. It
        separates the count fluctuations from the intensity second moment.
    """

    spectral_density: jax.Array
    intensity_second_moment: jax.Array | None
    extras: Mapping[str, jax.Array]
    merger_rate: jax.Array | None = None


class SpectralDensityModel(Protocol):
    """A quadrupole background model for the dense Gaussian likelihood.

    Implementations are JAX pytrees whose arrays are leaves, either a
    registered dataclass or a :class:`jax.tree_util.Partial` of a module-level
    function, so that the likelihood traces them and one compilation serves
    every model of the same structure and shapes. Their moments are on the
    frequency grid the likelihood is built on.

    Flags such as ``shot_noise`` are passed at call time, never bound into a
    ``Partial``: bound values are traced, and a traced flag cannot select a
    branch. A closure or lambda is a new static function at every
    construction and compiles anew each time.
    """

    def __call__(
        self, params: Mapping[str, ArrayLike], *, shot_noise: bool
    ) -> SpectralDensityPrediction:
        """The moments at ``params``.

        Parameters
        ----------
        params
            Model parameter values.
        shot_noise
            Whether to predict the intensity second moment; static.
        """
        ...
