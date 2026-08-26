"""Bank configuration and provenance: what a bank is made of, recorded forever.

A *bank* is one persisted, single-component waveform catalog: expensive to
build (population draw + ripple waveform generation) and reused by every run
that names it. One TOML per bank under ``config/banks/``; the filename stem is
the bank name, and ``outputs/banks/<stem>.h5`` is what it produces. No
registry file translates between the two -- that convention is what replaced
the old ``inputs/catalogs.yaml`` ``banks:`` section.

Every bank file also records what generated it -- population name, seed,
sample count, and the redshift *proposal density* those draws follow -- as
a :class:`BankConfig`. That descriptor is extracted from the population graph
exactly once, at generation time (:func:`extract_redshift_proposal`), and
every later consumer reads it back from the file
(:func:`read_bank_provenance`) instead of re-parsing a config that may have
drifted since the bank was built. netCDF attributes are flat scalars, so the
proposal descriptor travels as a single JSON-encoded string under
``redshift_proposal``; population name, seed, and sample count travel as
plain scalars alongside it.

Mixture *compositions* are not banks: they are cheap, in-memory draws over one
or two banks, declared inline by each run. See
:class:`astrogwb_paper.catalogs.CatalogSource`.

Deliberately JAX-free -- pydantic at import time, xarray/h5netcdf lazily
inside the one function that reads a bank file. This is why it lives here and
not in :mod:`astrogwb_paper.catalogs`, which imports JAX at module scope: the
workflow and the config layer must be able to read configs and provenance
without paying for a JAX import.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from astrogwb_paper.config.mcmc import ProposalConfig
from astrogwb_paper.paths import paper_project_root
from astrogwb_paper.utils import load_mapping

_STRICT = ConfigDict(frozen=True, extra="forbid")

BANKS_DIR = Path("config/banks")
POPULATIONS_DIR = Path("config/populations")

#: Bank attribute holding the JSON redshift-proposal descriptor.
PROPOSAL_ATTR = "redshift_proposal"
POPULATION_NAME_ATTR = "population_name"
POPULATION_SEED_ATTR = "population_seed"
POPULATION_SAMPLES_ATTR = "population_samples"

#: Constants a run's [fiducials] table must agree with for the recorded MD
#: proposal to be the density the importance weights divide by.
MD_FIDUCIAL_NAMES = ("H0", "Omega_m", "gamma", "kappa", "z_peak")

_DEFAULT_N_GRID = 4096


# --------------------------------------------------------------------------- #
# Generation: the committed config/banks/<bank>.toml files
# --------------------------------------------------------------------------- #
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


class BankGenerationConfig(BaseModel):
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


def load_bank_config(path: Path) -> BankGenerationConfig:
    """Load and validate one bank config; the filename stem names the bank."""
    return BankGenerationConfig.model_validate(
        {"name": path.stem, **load_mapping(path)}
    )


def discover_banks(root: Path | None = None) -> dict[str, BankGenerationConfig]:
    """Load every committed bank config, keyed by name, in sorted order."""
    directory = banks_dir(root)
    paths = sorted(directory.glob("*.toml"))
    if not paths:
        raise ValueError(f"{directory} declares no bank configs")
    return {path.stem: load_bank_config(path) for path in paths}


# --------------------------------------------------------------------------- #
# Redshift proposal descriptors
# --------------------------------------------------------------------------- #
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

    @classmethod
    def from_gwmock_dict(cls, arguments: Mapping[str, Any]) -> Self:
        """Build from a gwmock ``madau_dickinson_redshift`` sampler arguments dict."""
        return cls(
            z_min=float(arguments["z_min"]),
            z_max=float(arguments["z_max"]),
            gamma=float(arguments["gamma"]),
            kappa=float(arguments["kappa"]),
            z_peak=float(arguments["z_peak"]),
            H0=float(arguments["hubble_constant"]),
            Omega_m=float(arguments["omega_m"]),
            n_grid=int(arguments.get("n_grid", _DEFAULT_N_GRID)),
        )


class UniformRedshiftProposal(BaseModel):
    """The uniform redshift density a guard-component bank was drawn from."""

    model_config = _STRICT

    kind: Literal["uniform_redshift"] = "uniform_redshift"
    z_min: float
    z_max: float

    @classmethod
    def from_gwmock_dict(cls, arguments: Mapping[str, Any]) -> Self:
        """Build from a gwmock ``uniform`` sampler arguments dict."""
        return cls(
            z_min=float(arguments["minimum"]),
            z_max=float(arguments["maximum"]),
        )


RedshiftProposal = Annotated[
    MadauDickinsonProposal | UniformRedshiftProposal,
    Field(discriminator="kind"),
]
_PROPOSAL_ADAPTER: TypeAdapter[Any] = TypeAdapter(RedshiftProposal)


# --------------------------------------------------------------------------- #
# Provenance: what a bank file records about its own generation
# --------------------------------------------------------------------------- #
class BankConfig(BaseModel):
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

    def to_dict(self) -> dict[str, str | int]:
        """Flatten into the scalar attrs ``make_catalog`` will stamp on."""
        return {
            POPULATION_NAME_ATTR: self.population,
            POPULATION_SEED_ATTR: self.seed,
            POPULATION_SAMPLES_ATTR: self.num_samples,
            PROPOSAL_ATTR: self.redshift_proposal.model_dump_json(),
        }

    @classmethod
    def from_dict(cls, attrs: Mapping[str, Any], *, label: str) -> Self:
        """Validate one bank's attribute mapping into a :class:`BankConfig`."""
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
        return cls(
            population=str(attrs[POPULATION_NAME_ATTR]),
            seed=int(attrs[POPULATION_SEED_ATTR]),
            num_samples=int(attrs[POPULATION_SAMPLES_ATTR]),
            redshift_proposal=_PROPOSAL_ADAPTER.validate_json(
                str(attrs[PROPOSAL_ATTR])
            ),
        )


def read_bank_provenance(path: Path) -> BankConfig:
    """Read one bank's recorded provenance from its HDF5 attributes.

    A bank written before provenance metadata existed carries no
    ``redshift_proposal`` attribute. That is an error, never a cue to fall back
    to parsing the population config: the point of the attribute is that the
    config may have drifted since the bank was built.
    """
    import xarray as xr

    with xr.open_dataset(path, engine="h5netcdf") as bank:
        attrs = dict(bank.attrs)
    return BankConfig.from_dict(attrs, label=str(path))


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
    parameters = _mapping(graph.get("parameters"), label="parameters")
    sampler = _mapping(
        _mapping(parameters.get("redshift"), label="parameters.redshift").get(
            "sampler"
        ),
        label="parameters.redshift.sampler",
    )
    function = sampler.get("function")
    arguments = _mapping(
        sampler.get("arguments"), label="parameters.redshift.sampler.arguments"
    )
    match function:
        case "madau_dickinson_redshift":
            return MadauDickinsonProposal.from_gwmock_dict(arguments)
        case "uniform":
            return UniformRedshiftProposal.from_gwmock_dict(arguments)
        case _:
            raise ValueError(
                f"unsupported redshift sampler function {function!r}; the bank "
                "proposal descriptor can only be derived from "
                "'madau_dickinson_redshift' or 'uniform'"
            )


def madau_dickinson_proposal(
    provenance: BankConfig, *, label: str
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


class _MDFiducials(BaseModel):
    """The MD constants a run's [fiducials] must share with its proposal bank."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    H0: float
    Omega_m: float
    gamma: float
    kappa: float
    z_peak: float


def check_fiducials_match(
    provenance: BankConfig,
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
    try:
        run = _MDFiducials.model_validate(dict(fiducials))
    except ValidationError as exc:
        missing = [
            str(error["loc"][0]) for error in exc.errors() if error["type"] == "missing"
        ]
        raise ValueError(
            f"run fiducials do not match proposal bank {label}: "
            + ", ".join(f"{name} (missing from [fiducials])" for name in missing)
        ) from None
    generated = _MDFiducials.model_validate(
        proposal.model_dump(include=set(MD_FIDUCIAL_NAMES))
    )
    mismatches = [
        f"{name} (run {getattr(run, name):.6g} vs bank {getattr(generated, name):.6g})"
        for name in MD_FIDUCIAL_NAMES
        if getattr(run, name) != getattr(generated, name)
    ]
    if mismatches:
        raise ValueError(
            f"run fiducials do not match proposal bank {label}: "
            + ", ".join(mismatches)
        )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Narrow an arbitrary loaded YAML value to a mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value
