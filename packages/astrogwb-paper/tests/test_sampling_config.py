from __future__ import annotations

import pytest
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.paths import paper_project_root
from pydantic import ValidationError

PAPER_ROOT = paper_project_root()


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
    # ...while its prior still lives in priors for the marginalization...
    assert "H0" in config.priors
    # ...and the reconstruction writes it into the saved posterior group.
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
