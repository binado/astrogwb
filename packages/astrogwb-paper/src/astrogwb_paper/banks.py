"""Bank provenance: what generated a bank, written once and read back forever.

A bank file records the population it was drawn from and -- crucially -- the
redshift *proposal density* those draws follow. That descriptor is extracted
from the population graph exactly once, at generation time
(:func:`extract_redshift_proposal`), and every later consumer reads it back
from the file (:func:`read_bank_provenance`) instead of re-parsing a config
that may have drifted since the bank was built.

netCDF attributes are flat scalars, so the descriptor travels as a single
JSON-encoded string under ``redshift_proposal``. Population name, seed, and
sample count travel as plain scalars alongside it.

Deliberately JAX-free -- xarray/h5netcdf and pydantic only. This is why it
lives here and not in :mod:`astrogwb_paper.catalogs`, which imports JAX at
module scope: the workflow and the config layer must be able to read
provenance without paying for a JAX import.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from astrogwb_paper.config.mcmc import ProposalConfig

_STRICT = ConfigDict(frozen=True, extra="forbid")

#: Bank attribute holding the JSON redshift-proposal descriptor.
PROPOSAL_ATTR = "redshift_proposal"
POPULATION_NAME_ATTR = "population_name"
POPULATION_SEED_ATTR = "population_seed"
POPULATION_SAMPLES_ATTR = "population_samples"

#: Constants a run's [fiducials] table must agree with for the recorded MD
#: proposal to be the density the importance weights divide by.
MD_FIDUCIAL_NAMES = ("H0", "Omega_m", "gamma", "kappa", "z_peak")

_DEFAULT_N_GRID = 4096


class MadauDickinsonProposal(BaseModel):
    """The Madau-Dickinson redshift density a bank's samples were drawn from."""

    model_config = _STRICT

    kind: Literal["madau_dickinson"] = "madau_dickinson"
    z_min: float
    z_max: float
    gamma: float
    kappa: float
    z_peak: float
    H0: float
    Omega_m: float
    n_grid: Annotated[int, Field(gt=1)] = _DEFAULT_N_GRID


class UniformRedshiftProposal(BaseModel):
    """The uniform redshift density a guard-component bank was drawn from."""

    model_config = _STRICT

    kind: Literal["uniform_redshift"] = "uniform_redshift"
    z_min: float
    z_max: float


RedshiftProposal = Annotated[
    MadauDickinsonProposal | UniformRedshiftProposal,
    Field(discriminator="kind"),
]
_PROPOSAL_ADAPTER: TypeAdapter[Any] = TypeAdapter(RedshiftProposal)


class BankProvenance(BaseModel):
    """What a bank file records about its own generation."""

    model_config = _STRICT

    population: str
    seed: int
    num_samples: Annotated[int, Field(gt=0)]
    redshift_proposal: RedshiftProposal

    @property
    def support(self) -> tuple[float, float]:
        """The redshift span this bank's samples were drawn over."""
        return self.redshift_proposal.z_min, self.redshift_proposal.z_max


# --------------------------------------------------------------------------- #
# Write side: population graph -> descriptor
# --------------------------------------------------------------------------- #
def extract_redshift_proposal(graph: Mapping[str, Any]) -> RedshiftProposal:
    """Read the redshift proposal density out of a population graph config.

    This is the only place the graph YAML is interpreted as a *density*, and it
    runs once per bank at generation time. An unrecognised sampler ``function``
    is rejected rather than defaulted: the descriptor is written once and
    trusted forever after, so a wrong guess here is a silently wrong likelihood
    in every analysis that follows.
    """
    sampler = _mapping(
        _redshift(graph).get("sampler"), label="parameters.redshift.sampler"
    )
    function = sampler.get("function")
    arguments = _mapping(
        sampler.get("arguments"), label="parameters.redshift.sampler.arguments"
    )
    match function:
        case "madau_dickinson_redshift":
            return MadauDickinsonProposal(
                z_min=float(arguments["z_min"]),
                z_max=float(arguments["z_max"]),
                gamma=float(arguments["gamma"]),
                kappa=float(arguments["kappa"]),
                z_peak=float(arguments["z_peak"]),
                H0=float(arguments["hubble_constant"]),
                Omega_m=float(arguments["omega_m"]),
                n_grid=int(arguments.get("n_grid", _DEFAULT_N_GRID)),
            )
        case "uniform":
            return UniformRedshiftProposal(
                z_min=float(arguments["minimum"]),
                z_max=float(arguments["maximum"]),
            )
        case _:
            raise ValueError(
                f"unsupported redshift sampler function {function!r}; the bank "
                "proposal descriptor can only be derived from "
                "'madau_dickinson_redshift' or 'uniform'"
            )


def provenance_attrs(provenance: BankProvenance) -> dict[str, str | int]:
    """Flatten provenance into the scalar attrs ``make_catalog`` will stamp on."""
    return {
        POPULATION_NAME_ATTR: provenance.population,
        POPULATION_SEED_ATTR: provenance.seed,
        POPULATION_SAMPLES_ATTR: provenance.num_samples,
        PROPOSAL_ATTR: provenance.redshift_proposal.model_dump_json(),
    }


# --------------------------------------------------------------------------- #
# Read side: bank file -> descriptor
# --------------------------------------------------------------------------- #
def read_bank_provenance(path: Path) -> BankProvenance:
    """Read one bank's recorded provenance from its HDF5 attributes.

    A bank written before provenance metadata existed carries no
    ``redshift_proposal`` attribute. That is an error, never a cue to fall back
    to parsing the population config: the point of the attribute is that the
    config may have drifted since the bank was built.
    """
    with xr.open_dataset(path, engine="h5netcdf") as bank:
        attrs = dict(bank.attrs)
    return provenance_from_attrs(attrs, label=str(path))


def provenance_from_attrs(attrs: Mapping[str, Any], *, label: str) -> BankProvenance:
    """Validate one bank's attribute mapping into a :class:`BankProvenance`."""
    missing = [
        name
        for name in (
            POPULATION_NAME_ATTR,
            POPULATION_SEED_ATTR,
            POPULATION_SAMPLES_ATTR,
            PROPOSAL_ATTR,
        )
        if name not in attrs
    ]
    if missing:
        raise ValueError(
            f"bank {label} was generated before proposal metadata "
            f"(missing {', '.join(missing)}); regenerate it"
        )
    return BankProvenance(
        population=str(attrs[POPULATION_NAME_ATTR]),
        seed=int(attrs[POPULATION_SEED_ATTR]),
        num_samples=int(attrs[POPULATION_SAMPLES_ATTR]),
        redshift_proposal=_PROPOSAL_ADAPTER.validate_json(str(attrs[PROPOSAL_ATTR])),
    )


def madau_dickinson_proposal(
    provenance: BankProvenance, *, label: str
) -> MadauDickinsonProposal:
    """Narrow a bank's recorded proposal to Madau-Dickinson, or fail clearly."""
    match provenance.redshift_proposal:
        case MadauDickinsonProposal() as proposal:
            return proposal
        case other:
            raise ValueError(
                f"bank {label} was drawn from a {other.kind!r} redshift density; "
                "the md_bank role requires a Madau-Dickinson bank"
            )


# --------------------------------------------------------------------------- #
# Derivation: descriptors + analysis window -> ProposalConfig
# --------------------------------------------------------------------------- #
def resolve_proposal(
    md: MadauDickinsonProposal,
    uniform: UniformRedshiftProposal | None,
    *,
    uniform_mixing_fraction: float,
    minimum_redshift: float,
    maximum_redshift: float,
) -> ProposalConfig:
    """Restrict a bank pair's generation density to the analysis window.

    Generation draws truncated to the window follow the same law as drawing
    directly from it, so only the support fields change. The window must lie
    inside what was actually generated, and a mixed proposal requires both
    components to span the same range.
    """
    if uniform_mixing_fraction > 0.0:
        if uniform is None:
            raise ValueError(
                "uniform_mixing_fraction > 0 requires a uniform-redshift bank"
            )
        if (uniform.z_min, uniform.z_max) != (md.z_min, md.z_max):
            raise ValueError(
                "mixed proposal banks disagree on generation redshift support: "
                f"md [{md.z_min:.4g}, {md.z_max:.4g}] vs uniform "
                f"[{uniform.z_min:.4g}, {uniform.z_max:.4g}]"
            )
    if not md.z_min <= minimum_redshift < maximum_redshift <= md.z_max:
        raise ValueError(
            f"analysis redshift support [{minimum_redshift:.4g}, "
            f"{maximum_redshift:.4g}] must lie within the bank generation "
            f"support [{md.z_min:.4g}, {md.z_max:.4g}]"
        )
    return ProposalConfig(
        uniform_mixing_fraction=uniform_mixing_fraction,
        minimum_redshift=float(minimum_redshift),
        maximum_redshift=float(maximum_redshift),
        n_grid=md.n_grid,
        H0=md.H0,
        Omega_m=md.Omega_m,
        gamma=md.gamma,
        kappa=md.kappa,
        z_peak=md.z_peak,
    )


def check_fiducials_match(
    provenance: BankProvenance,
    fiducials: Mapping[str, float],
    *,
    label: str,
) -> None:
    """Require a run's fiducials to equal what the proposal bank was drawn at.

    The importance weights divide the fiducial merger-rate density by the
    proposal density, so a mismatch here is not a bookkeeping slip: it silently
    reweights against the wrong denominator. Compared against what was actually
    generated, not against a re-parse of the population config.
    """
    proposal = madau_dickinson_proposal(provenance, label=label)
    mismatches: list[str] = []
    for name in MD_FIDUCIAL_NAMES:
        generated = float(getattr(proposal, name))
        if name not in fiducials:
            mismatches.append(f"{name} (missing from [fiducials])")
        elif float(fiducials[name]) != generated:
            mismatches.append(
                f"{name} (run {float(fiducials[name]):.6g} vs bank {generated:.6g})"
            )
    if mismatches:
        raise ValueError(
            f"run fiducials do not match proposal bank {label}: "
            + ", ".join(mismatches)
        )


def _redshift(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = _mapping(graph.get("parameters"), label="parameters")
    return _mapping(parameters.get("redshift"), label="parameters.redshift")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Narrow an arbitrary loaded YAML value to a mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value
