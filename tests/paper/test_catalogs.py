"""Checks between the two catalogs a spectrum is evaluated against."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from astrogwb.paper.catalogs import validate_matching_frequency_grids


def test_injection_and_proposal_frequency_grids_must_match() -> None:
    with pytest.raises(ValueError, match="identical frequency grids"):
        validate_matching_frequency_grids(jnp.array([2.0, 3.0]), jnp.array([2.0, 4.0]))
