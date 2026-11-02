"""NumPyro handler utilities shared across the package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.primitives import Messenger

__all__ = ["InverseCDF"]


class InverseCDF(Messenger):
    r"""Fill a model's free sample sites from given points through their inverse CDFs.

    At each free sample site, in execution order, the value is
    ``site.fn.icdf(points[..., d])`` with ``d`` the site's position among the
    free sites, so the ``d``-th coordinate of every point drives the ``d``-th
    site. A site is free when no inner handler has fixed its value: sites a
    :func:`numpyro.handlers.condition` *inside* this handler observes consume
    no coordinate. A conditional site is filled after its parents, so its
    inverse CDF is the conditional one (the Rosenblatt transform): uniform
    points on the unit cube map to the model's joint law.

    Unchecked contracts:

    - ``points`` has one trailing coordinate per free site, in ``(0, 1)``.
    - Every free site's distribution has an ``icdf`` that broadcasts over the
      leading shape of ``points``; a model without plates is then evaluated
      once for every point.
    - Any ``condition`` the model needs is applied inside this handler, never
      around it.

    The leading coordinates of a Sobol sequence are the most uniform, so a
    model should sample its most influential parameters first.

    Parameters
    ----------
    fn
        The model.
    points
        Shape ``(..., D)``, ``D`` the number of free sites.
    """

    def __init__(
        self, fn: Callable[..., Any] | None = None, points: ArrayLike | None = None
    ) -> None:
        self.points = jnp.asarray(points)
        self._index = 0
        super().__init__(fn)

    def __enter__(self) -> Any:
        self._index = 0
        return super().__enter__()

    def process_message(self, msg: dict[str, Any]) -> None:
        """Transform the next uniform coordinate at an eligible sample site.

        Parameters
        ----------
        msg
            NumPyro effect message.
        """
        if msg["type"] != "sample" or msg["is_observed"] or msg["value"] is not None:
            return
        msg["value"] = msg["fn"].icdf(self.points[..., self._index])
        self._index += 1
