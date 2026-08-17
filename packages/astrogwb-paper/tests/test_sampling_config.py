from __future__ import annotations

import numpy as np
import pytest
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.paths import paper_project_root
from astrogwb_paper.priors import build_prior
from pydantic import ValidationError

PAPER_ROOT = paper_project_root()


@pytest.mark.parametrize(
    ("spec", "expected_attrs"),
    [
        ({"type": "uniform", "low": 0.0, "high": 2.0}, {"low": 0.0, "high": 2.0}),
        ({"type": "normal", "loc": 1.0, "scale": 0.5}, {"loc": 1.0, "scale": 0.5}),
    ],
)
def test_build_prior_happy_path(
    spec: dict[str, object], expected_attrs: dict[str, float]
) -> None:
    distribution = build_prior(spec)

    for attr, expected in expected_attrs.items():
        np.testing.assert_allclose(getattr(distribution, attr), expected)


def test_build_prior_rejects_unknown_type() -> None:
    # ValueError, not pydantic.ValidationError: spec parsing is hand-rolled in
    # materialize_prior (no pydantic spec models).
    with pytest.raises(ValueError, match="does not match any of the expected"):
        build_prior({"type": "mystery", "low": 0.0, "high": 1.0})


@pytest.mark.parametrize(
    "relative_path",
    ["configs/mcmc.example.toml", "configs/mcmc.cosmology.toml"],
)
def test_shipped_mcmc_configs_fix_local_merger_rate(relative_path: str) -> None:
    config = build_run_config(load_mapping(PAPER_ROOT / relative_path))

    assert "local_merger_rate" not in config.sampled_params
    assert config.fiducials["local_merger_rate"] == 161.0
    assert "local_merger_rate" not in config.priors
    assert config.constants["local_merger_rate"] == 161.0


def test_posterior_params_adds_the_marginalized_amplitude_parameter() -> None:
    # Marginalize H0 out of the two-parameter config, leaving Omega_m sampled.
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.cosmology.toml")
    raw["sampled_params"] = ["Omega_m"]
    raw["analysis"]["likelihood"] = "amplitude_marginalized"
    raw["analysis"]["amplitude_parameter"] = "H0"
    config = build_run_config(raw)

    assert config.sampled_params == ("Omega_m",)

    # H0 has no NUTS latent, so it must stay out of sampled_params...
    assert "H0" not in config.sampled_params
    assert "H0" not in config.priors
    # ...but the reconstruction writes it into the saved posterior group.
    assert config.posterior_params == (*config.sampled_params, "H0")


def test_posterior_params_matches_sampled_params_without_marginalization() -> None:
    config = build_run_config(load_mapping(PAPER_ROOT / "configs/mcmc.example.toml"))

    assert config.analysis.amplitude_parameter is None
    assert config.posterior_params == config.sampled_params


def test_legacy_top_level_local_merger_rate_is_rejected() -> None:
    raw = load_mapping(PAPER_ROOT / "configs/mcmc.example.toml")
    raw["local_merger_rate"] = raw["fiducials"]["local_merger_rate"]

    with pytest.raises(ValidationError, match="local_merger_rate"):
        build_run_config(raw)
