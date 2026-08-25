"""Typed loading for the committed bank configs under ``config/banks/``.

A *bank* is one persisted, single-component waveform catalog: expensive to
build (population draw + ripple waveform generation) and reused by every run
that names it. One TOML per bank; the filename stem is the bank name, and
``outputs/banks/<stem>.h5`` is what it produces. No registry file translates
between the two -- that convention is what replaced the old
``inputs/catalogs.yaml`` ``banks:`` section.

Mixture *compositions* are not banks: they are cheap, in-memory draws over one
or two banks, declared inline by each run. See
:func:`astrogwb_paper.catalogs.compose_catalog`.

stdlib + pydantic only, so the workflow can load every bank config without
touching JAX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from astrogwb_paper.config.mcmc import load_mapping
from astrogwb_paper.paths import paper_project_root

BANKS_DIR = Path("config/banks")
POPULATIONS_DIR = Path("config/populations")
_STRICT = ConfigDict(frozen=True, extra="forbid")


class WaveformConfig(BaseModel):
    """Waveform-generation settings for one bank."""

    model_config = _STRICT

    approximant: str
    sampling_frequency: Annotated[float, Field(gt=0.0)]
    minimum_frequency: Annotated[float, Field(ge=0.0)]
    maximum_frequency: Annotated[float, Field(gt=0.0)]
    reference_frequency: Annotated[float, Field(gt=0.0)]
    frequency_resolution: Annotated[float, Field(gt=0.0)]
    chunk_size: Annotated[int, Field(gt=0)]


class BankConfig(BaseModel):
    """Everything needed to generate one reusable single-component bank.

    ``seed`` and ``num_samples`` are bank-level rather than population-level on
    purpose: ``md-imrphenom-s41`` and ``md-imrphenom-s42`` are the *same* graph
    drawn at different seeds, so pushing either into the population config
    would mean two near-identical graph files.
    """

    model_config = _STRICT

    name: str
    population: str
    seed: int
    num_samples: Annotated[int, Field(gt=0)]
    waveform: WaveformConfig

    def population_path(self, root: Path | None = None) -> Path:
        """Return the population graph this bank is drawn from."""
        base = root if root is not None else paper_project_root()
        return base / POPULATIONS_DIR / f"{self.population}.yaml"


def banks_dir(root: Path | None = None) -> Path:
    """Return the committed bank-config directory."""
    return (root or paper_project_root()) / BANKS_DIR


def load_bank_config(path: Path) -> BankConfig:
    """Load and validate one bank config; the filename stem names the bank."""
    return BankConfig.model_validate({"name": path.stem, **load_mapping(path)})


def discover_banks(root: Path | None = None) -> dict[str, BankConfig]:
    """Load every committed bank config, keyed by name, in sorted order."""
    directory = banks_dir(root)
    paths = sorted(directory.glob("*.toml"))
    if not paths:
        raise ValueError(f"{directory} declares no bank configs")
    return {path.stem: load_bank_config(path) for path in paths}


def bank_config(name: str, root: Path | None = None) -> BankConfig:
    """Return a named bank config or raise a user-facing error."""
    banks = discover_banks(root)
    try:
        return banks[name]
    except KeyError:
        choices = ", ".join(banks)
        raise ValueError(f"unknown bank {name!r}; choose from {choices}") from None
