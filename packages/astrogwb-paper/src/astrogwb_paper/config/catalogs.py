"""Typed loading for the bank/composition generation inventory.

A *bank* is one persisted, single-component waveform catalog (expensive:
population draw + ripple waveform generation). A *composition* names a bank
(or an MD bank plus a uniform bank) and cheap, in-memory mixture parameters
(``num_samples``, ``uniform_mixing_fraction``, ``mixture_seed``); it is never
written to disk. See :func:`astrogwb_paper.catalogs.compose_catalog`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb_paper.config.loading import deep_merge, load_inventory, load_mapping
from astrogwb_paper.paths import paper_project_root

CATALOGS_PATH = Path("inputs/catalogs.yaml")
INJECTION_CATALOG_NAME = "injection-bns-n32768-eps=0-df1"
_INVENTORY_SECTIONS = ("base", "banks", "catalogs")
_STRICT = ConfigDict(frozen=True, extra="forbid")


class PopulationRecipe(BaseModel):
    """Population graph config paths shared by every bank."""

    model_config = _STRICT

    base_config: Path
    md_redshift_config: Path
    uniform_redshift_config: Path


class WaveformRecipe(BaseModel):
    """Waveform settings shared by generated banks."""

    model_config = _STRICT

    approximant: str
    sampling_frequency: Annotated[float, Field(gt=0.0)]
    minimum_frequency: Annotated[float, Field(ge=0.0)]
    maximum_frequency: Annotated[float, Field(gt=0.0)]
    reference_frequency: Annotated[float, Field(gt=0.0)]
    frequency_resolution: Annotated[float, Field(gt=0.0)]
    chunk_size: Annotated[int, Field(gt=0)]


class BankRecipe(BaseModel):
    """Everything needed to generate one reusable single-component bank."""

    model_config = _STRICT

    name: str
    component: Literal["md", "uniform-redshift"]
    population: PopulationRecipe
    num_samples: Annotated[int, Field(gt=0)]
    seed: int
    waveform: WaveformRecipe


class CatalogComposition(BaseModel):
    """A cheap, in-memory mixture over up to two persisted banks."""

    model_config = _STRICT

    name: str
    md_bank: str
    uniform_bank: str | None = None
    num_samples: Annotated[int, Field(gt=0)]
    uniform_mixing_fraction: Annotated[
        float, Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ] = 0.0
    mixture_seed: int | None = None

    @model_validator(mode="after")
    def _validate_mixture_fields(self) -> CatalogComposition:
        mixed = self.uniform_mixing_fraction > 0.0
        if mixed and (self.uniform_bank is None or self.mixture_seed is None):
            raise ValueError(
                f"catalog {self.name!r}: uniform_mixing_fraction > 0 requires both "
                "uniform_bank and mixture_seed"
            )
        if not mixed and (
            self.uniform_bank is not None or self.mixture_seed is not None
        ):
            raise ValueError(
                f"catalog {self.name!r}: uniform_mixing_fraction == 0 forbids "
                "uniform_bank and mixture_seed"
            )
        return self


def inventory_path(root: Path | None = None) -> Path:
    """Return the committed catalog inventory path."""
    return (root or paper_project_root()) / CATALOGS_PATH


def _raw_inventory(path: Path) -> dict[str, Any]:
    return load_inventory(path, required=_INVENTORY_SECTIONS)


def load_banks(path: Path | None = None) -> dict[str, BankRecipe]:
    """Load and validate every committed bank recipe."""
    resolved = path or inventory_path()
    raw = _raw_inventory(resolved)
    base = raw["base"]
    banks_raw = raw["banks"]
    assert isinstance(base, Mapping)
    assert isinstance(banks_raw, Mapping)
    if not banks_raw:
        raise ValueError(f"{resolved} must define a non-empty banks mapping")

    banks: dict[str, BankRecipe] = {}
    for name, overlay in banks_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{resolved} has an invalid bank name")
        if overlay is not None and not isinstance(overlay, Mapping):
            raise TypeError(f"{resolved} bank {name!r} must be a mapping")
        banks[name] = BankRecipe.model_validate(
            {"name": name, **deep_merge(base, overlay or {})}
        )
    return banks


def bank_recipe(name: str, path: Path | None = None) -> BankRecipe:
    """Return a named bank recipe or raise a user-facing error."""
    return _resolve_bank(name, load_banks(path))


def _resolve_bank(name: str, banks: Mapping[str, BankRecipe]) -> BankRecipe:
    try:
        return banks[name]
    except KeyError:
        choices = ", ".join(banks)
        raise ValueError(f"unknown bank {name!r}; choose from {choices}") from None


def load_catalogs(path: Path | None = None) -> dict[str, CatalogComposition]:
    """Load and validate every committed catalog composition."""
    resolved = path or inventory_path()
    raw = _raw_inventory(resolved)
    catalogs_raw = raw["catalogs"]
    assert isinstance(catalogs_raw, Mapping)
    if not catalogs_raw:
        raise ValueError(f"{resolved} must define a non-empty catalogs mapping")
    banks = load_banks(resolved)

    catalogs: dict[str, CatalogComposition] = {}
    for name, overlay in catalogs_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{resolved} has an invalid catalog name")
        if not isinstance(overlay, Mapping):
            raise TypeError(f"{resolved} catalog {name!r} must be a mapping")
        composition = CatalogComposition.model_validate({"name": name, **overlay})
        _validate_composition_banks(composition, banks, path=resolved)
        catalogs[name] = composition
    return catalogs


def _validate_composition_banks(
    composition: CatalogComposition,
    banks: Mapping[str, BankRecipe],
    *,
    path: Path,
) -> None:
    """Reject a composition naming an unknown, oversized, or mismatched bank."""
    md_bank = _resolve_named_bank(
        composition.md_bank, banks, composition=composition, path=path
    )
    if composition.num_samples > md_bank.num_samples:
        raise ValueError(
            f"{path} catalog {composition.name!r} requests num_samples="
            f"{composition.num_samples} but bank {md_bank.name!r} holds only "
            f"{md_bank.num_samples}"
        )
    if composition.uniform_bank is None:
        return

    uniform_bank = _resolve_named_bank(
        composition.uniform_bank, banks, composition=composition, path=path
    )
    if md_bank.waveform != uniform_bank.waveform:
        raise ValueError(
            f"{path} catalog {composition.name!r} mixes banks with different "
            f"waveform settings: {md_bank.name!r} vs {uniform_bank.name!r}"
        )
    assert composition.mixture_seed is not None  # enforced by _validate_mixture_fields
    seeds = (md_bank.seed, uniform_bank.seed, composition.mixture_seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"{path} catalog {composition.name!r}: md_bank seed, uniform_bank seed, "
            f"and mixture_seed must all be distinct (got {seeds})"
        )


def _resolve_named_bank(
    name: str,
    banks: Mapping[str, BankRecipe],
    *,
    composition: CatalogComposition,
    path: Path,
) -> BankRecipe:
    try:
        return banks[name]
    except KeyError:
        choices = ", ".join(banks)
        raise ValueError(
            f"{path} catalog {composition.name!r} names unknown bank {name!r}; "
            f"choose from {choices}"
        ) from None


def catalog_recipe(name: str, path: Path | None = None) -> CatalogComposition:
    """Return a named catalog composition or raise a user-facing error."""
    catalogs = load_catalogs(path)
    try:
        return catalogs[name]
    except KeyError:
        choices = ", ".join(catalogs)
        raise ValueError(f"unknown catalog {name!r}; choose from {choices}") from None


def proposal_config(
    composition: CatalogComposition,
    *,
    banks: Mapping[str, BankRecipe] | None = None,
    root: Path | None = None,
) -> dict[str, float | int]:
    """Expand the generation-time redshift proposal represented by ``composition``."""
    project_root = root or paper_project_root()
    resolved_banks = banks if banks is not None else load_banks()
    md_bank = _resolve_bank(composition.md_bank, resolved_banks)
    md_sampler = _md_sampler_arguments(md_bank, root=project_root)

    minimum_redshift = float(md_sampler["z_min"])
    maximum_redshift = float(md_sampler["z_max"])
    h0 = float(md_sampler["hubble_constant"])
    omega_m = float(md_sampler["omega_m"])

    return {
        "uniform_mixing_fraction": composition.uniform_mixing_fraction,
        "minimum_redshift": minimum_redshift,
        "maximum_redshift": maximum_redshift,
        "n_grid": int(md_sampler.get("n_grid", 4096)),
        "H0": h0,
        "Omega_m": omega_m,
        "gamma": float(md_sampler["gamma"]),
        "kappa": float(md_sampler["kappa"]),
        "z_peak": float(md_sampler["z_peak"]),
    }


def generation_redshift_support(
    composition: CatalogComposition,
    *,
    banks: Mapping[str, BankRecipe] | None = None,
    root: Path | None = None,
) -> tuple[float, float]:
    """Return the redshift span catalog generation draws from for ``composition``.

    Requires the MD and (if present) uniform banks to agree on generation
    support: both redshift overlays are meant to span the same window (see
    ``inputs/populations/population.{md,uniform-redshift}.yaml``), so
    disagreement signals a misconfigured bank pairing.
    """
    project_root = root or paper_project_root()
    resolved_banks = banks if banks is not None else load_banks()
    md_bank = _resolve_bank(composition.md_bank, resolved_banks)
    md_support = _md_redshift_support(md_bank, root=project_root)
    if composition.uniform_bank is None:
        return md_support

    uniform_bank = _resolve_bank(composition.uniform_bank, resolved_banks)
    uniform_support = _uniform_redshift_support(uniform_bank, root=project_root)
    if uniform_support != md_support:
        raise ValueError(
            f"catalog {composition.name!r} mixes banks with disagreeing generation "
            f"redshift support: md {md_support} vs uniform-redshift {uniform_support}"
        )
    return md_support


def analysis_proposal_config(
    composition: CatalogComposition,
    *,
    minimum_redshift: float,
    maximum_redshift: float,
    banks: Mapping[str, BankRecipe] | None = None,
    root: Path | None = None,
) -> dict[str, float | int]:
    """Restrict the generation proposal to the analysis redshift window.

    Generation draws truncated to the window follow the same law as drawing
    directly from it, so only the support fields change.
    """
    return {
        **proposal_config(composition, banks=banks, root=root),
        "minimum_redshift": float(minimum_redshift),
        "maximum_redshift": float(maximum_redshift),
    }


def _md_sampler_arguments(bank: BankRecipe, *, root: Path) -> Mapping[str, Any]:
    md = load_mapping(root / bank.population.md_redshift_config)
    return _redshift_sampler_arguments(md, label="MD redshift sampler")


def _md_redshift_support(bank: BankRecipe, *, root: Path) -> tuple[float, float]:
    sampler = _md_sampler_arguments(bank, root=root)
    return float(sampler["z_min"]), float(sampler["z_max"])


def _uniform_redshift_support(bank: BankRecipe, *, root: Path) -> tuple[float, float]:
    uniform = load_mapping(root / bank.population.uniform_redshift_config)
    sampler = _redshift_sampler_arguments(uniform, label="uniform redshift sampler")
    return float(sampler["minimum"]), float(sampler["maximum"])


def _redshift_sampler_arguments(
    mapping: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    parameters = _mapping(mapping.get("parameters"), label=f"{label} parameters")
    redshift = _mapping(parameters.get("redshift"), label=f"{label} redshift parameter")
    sampler = _mapping(redshift.get("sampler"), label=label)
    return _mapping(sampler.get("arguments"), label=f"{label} arguments")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Narrow an arbitrary loaded YAML value to a mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value
