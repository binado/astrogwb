from __future__ import annotations

import pytest
from astrogwb_paper.config.mcmc import build_run_config
from astrogwb_paper.paths import paper_project_root
from config_fixtures import example_raw
from pydantic import ValidationError

PAPER_ROOT = paper_project_root()


def test_assembled_configs_fix_local_merger_rate() -> None:
    raw = example_raw()
    expected = raw["fiducials"]["local_merger_rate"]
    config = build_run_config(raw)

    # The complete prior table declares the site; sampled_params decides that
    # production conditions it rather than giving NUTS a latent.
    assert "local_merger_rate" not in config.sampled_params
    assert expected == 770.0
    assert config.fiducials["local_merger_rate"] == expected
    assert "local_merger_rate" in config.priors
    assert config.fixed_params["local_merger_rate"] == expected


def test_posterior_params_adds_the_marginalized_amplitude_parameter() -> None:
    # Marginalize H0 out, leaving Omega_m as the only sampled parameter.
    raw = example_raw()
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
    config = build_run_config(example_raw())

    assert config.analysis.amplitude_parameter is None
    assert config.posterior_params == config.sampled_params


def test_legacy_top_level_local_merger_rate_is_rejected() -> None:
    raw = example_raw()
    raw["local_merger_rate"] = raw["fiducials"]["local_merger_rate"]

    with pytest.raises(ValidationError, match="local_merger_rate"):
        build_run_config(raw)
