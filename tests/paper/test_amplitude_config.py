"""Tests for the paper's amplitude-marginalization glue.

``build_amplitude_marginalization`` is the only JAX-touching bridge between the
validated config and astrogwb's numerical marginalization; the physics itself
(the H0^{-3}/H0^{-1} named scalings) is covered in core's
``test_amplitude_scalings.py``, not here.
"""

from __future__ import annotations

import typing

import jax.numpy as jnp
import numpy as np
import pytest
from config_fixtures import example_raw

from astrogwb.paper.amplitude import build_amplitude_marginalization
from astrogwb.paper.config.mcmc import AmplitudeParameter, build_run_config

# --------------------------------------------------------------------------- #
# build_amplitude_marginalization
# --------------------------------------------------------------------------- #


def _marginalized_config(**kwargs):
    raw = example_raw()
    raw["analysis"] = {
        **raw["analysis"],
        "likelihood": "amplitude_marginalized",
        "amplitude_parameter": "H0",
    }
    raw["sampled_params"] = []
    return build_run_config(raw, **kwargs)


def test_build_amplitude_marginalization_matches_config_grid_settings() -> None:
    config = _marginalized_config()
    marginalization = build_amplitude_marginalization(config)

    assert marginalization.parameter == "H0"
    assert marginalization.fiducial == float(config.fiducials["H0"])
    assert marginalization.grid.shape == (config.analysis.amplitude_num_nodes,)
    np.testing.assert_allclose(float(marginalization.grid[0]), 20.0)
    np.testing.assert_allclose(float(marginalization.grid[-1]), 140.0)

    # A uniform prior's log density is already normalized on its own support.
    integral = float(
        jnp.trapezoid(
            jnp.exp(marginalization.prior.log_prob(marginalization.grid)),
            marginalization.grid,
        )
    )
    np.testing.assert_allclose(integral, 1.0, rtol=1e-6)


def test_build_amplitude_marginalization_anchors_the_amplitude_at_the_fiducial() -> (
    None
):
    """The scalings are absolute; the run is only correct if the ratio is 1 at phi_fid."""
    marginalization = build_amplitude_marginalization(_marginalized_config())
    fiducial = jnp.asarray(marginalization.fiducial)

    for fn in (marginalization.amplitude_fn, marginalization.merger_rate_fn):
        np.testing.assert_allclose(float(fn(fiducial) / fn(fiducial)), 1.0)

    # H0: f = g_R * g_F = H0**-3 * H0**2, so f(phi)/f(phi_fid) = phi_fid/phi.
    phi = jnp.asarray(2.0) * fiducial
    np.testing.assert_allclose(
        float(
            marginalization.amplitude_fn(phi) / marginalization.amplitude_fn(fiducial)
        ),
        0.5,
        rtol=1e-12,
    )


def test_build_amplitude_marginalization_rejects_a_non_marginalized_config() -> None:
    raw = example_raw()
    config = build_run_config(raw)

    with pytest.raises(ValueError, match="amplitude-marginalized config"):
        build_amplitude_marginalization(config)


# --------------------------------------------------------------------------- #
# Cross-check against core's AMPLITUDE_PARAMETERS
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_amplitude_parameter_literal_matches_core_amplitude_parameters() -> None:
    from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
        AMPLITUDE_PARAMETERS,
    )

    literal_values = typing.get_args(AmplitudeParameter)
    assert set(literal_values) == set(AMPLITUDE_PARAMETERS)
