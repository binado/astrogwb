"""The setup helpers shared by the posterior notebooks."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from repo import REPO_ROOT

from astrogwb.paper import posteriors
from astrogwb.paper.config import detector_registry
from astrogwb.paper.posteriors import (
    ensure_reference_catalog,
    network_data,
    settings_digest,
    shot_noise_variant,
)


@dataclasses.dataclass(frozen=True)
class _Likelihood:
    """The one field ``shot_noise_variant`` replaces."""

    shot_noise: str | None


@pytest.fixture(scope="module")
def registry():
    return detector_registry(root=REPO_ROOT)


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        ({"a": 1, "b": [2, 3]}, {"b": [2, 3], "a": 1}, True),
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}, False),
        ({"a": 1}, {"a": 1, "b": 2}, False),
    ],
)
def test_settings_digest(left: dict, right: dict, equal: bool) -> None:
    assert (settings_digest(**left) == settings_digest(**right)) is equal


@pytest.mark.parametrize(("recorded_key", "recorded_seed"), [("other", 7), ("this", 8)])
def test_reference_catalog_mismatch_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_key: str,
    recorded_seed: int,
) -> None:
    metadata = SimpleNamespace(key=lambda: "this")
    recorded = SimpleNamespace(key=lambda: recorded_key)
    (tmp_path / "reference_catalog-this-7.h5").touch()
    monkeypatch.setattr(
        posteriors, "load", lambda *_: (None, recorded, {"seed": recorded_seed})
    )
    with pytest.raises(ValueError, match="records"):
        ensure_reference_catalog(metadata, 7, tmp_path)  # ty: ignore[invalid-argument-type]


def test_network_data_masks_unusable_bins(registry) -> None:
    frequencies = np.array([0.5, 1.0, 10.0, 50.0, 100.0, 5000.0])
    data = network_data(
        registry,
        "ET-2L-aligned-CE-Hanford",
        frequencies,
        observation_time=1.0,
        minimum_frequency=2.0,
        maximum_frequency=2048.0,
    )
    mask, scale = np.asarray(data["frequency_mask"]), np.asarray(data["scale"])
    assert not mask[:2].any()
    assert not mask[-1]
    assert mask[2:-1].any()
    assert np.all(scale[~mask] == 1.0)
    assert np.all(np.isfinite(scale))


@pytest.mark.parametrize(
    ("variant", "mode", "has_variance", "settings"),
    [
        ("detector", None, False, {}),
        ("amplitude", "amplitude", False, {"shot_noise": "amplitude"}),
        ("per_frequency", "per_frequency", False, {"shot_noise": "per_frequency"}),
        (
            "fixed",
            "fixed",
            True,
            {"shot_noise": "fixed", "relative_variance": [0.1, 0.2]},
        ),
    ],
)
def test_shot_noise_variant(
    variant: str, mode: str | None, has_variance: bool, settings: dict
) -> None:
    likelihood, variance, recorded = shot_noise_variant(
        variant,
        _Likelihood(shot_noise="amplitude"),  # ty: ignore[invalid-argument-type]
        jnp.array([0.1, 0.2]),  # ty: ignore[invalid-argument-type]
    )
    assert likelihood.shot_noise == mode
    assert (variance is not None) is has_variance
    assert recorded == settings
