"""Catalog configuration and provenance: what a catalog is made of, recorded forever.

A *catalog* is one persisted waveform draw: expensive to build (population
draw + ripple waveform generation) and reused by every run that names it.
Two layers under ``config/catalogs/`` -- a shared ``base/`` and one
``defs/<name>.toml`` per catalog, whose stem is the name and which produces
``outputs/catalogs/<stem>.h5``. No registry file translates between the two.

This replaced an earlier split between persisted single-component *banks* and
in-memory mixtures composed at run time. Storing only polarization power made
every catalog cheap enough to persist, so the composition step, the five-field
inline spec each run carried, and the ``--bank NAME=PATH`` plumbing all went
away: a run now names two catalogs, and each catalog is a file.

Every catalog file records what generated it -- seed, sample count, and the
redshift *proposal mixture* its draws follow -- as a :class:`CatalogProvenance`.
That descriptor is extracted from the population graphs exactly once, at
generation time (:func:`extract_mixture_proposal`), and every later consumer
reads it back from the file (:meth:`CatalogProvenance.from_file`) instead of
re-parsing a config that may have drifted since the catalog was built. netCDF
attributes are flat scalars, so the descriptor travels as a single
JSON-encoded string under ``redshift_proposal`` in
:class:`astrogwb.catalog.PopulationMetadata`.

Deliberately JAX-free *at import*: the ``Snakefile`` imports this module to
build the DAG, so the methods that reach ``astrogwb.catalog`` -- which pulls in
JAX -- import it in their own bodies rather than at module scope. The workflow
and config layers can therefore inspect provenance without initializing JAX or
loading polarization power.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from astrogwb.paper.config.mcmc import ProposalConfig, RunConfig
from astrogwb.paper.config.runs import (
    CATALOG_DEFS_DIR,
    catalog_config_paths,
    discover_catalog_names,
    merge_config_layers,
)

if TYPE_CHECKING:
    from astrogwb.catalog import PopulationMetadata

logger = logging.getLogger(__name__)

_STRICT = ConfigDict(frozen=True, extra="forbid")

#: Relative to the working directory; see astrogwb.paper.config.runs.
POPULATIONS_DIR = Path("config/populations")

#: Catalog attribute holding the JSON redshift-proposal descriptor.
PROPOSAL_ATTR = "redshift_proposal"
#: Constants a run's [fiducials] table must agree with for the recorded MD
#: proposal to be the density the importance weights divide by.
MD_FIDUCIAL_NAMES = ("H0", "Omega_m", "gamma", "kappa", "z_peak")

_DEFAULT_N_GRID = 4096


# --------------------------------------------------------------------------- #
# Generation: the committed config/catalogs/ files
# --------------------------------------------------------------------------- #
class WaveformConfig(BaseModel):
    """Waveform-generation settings for one catalog."""

    model_config = _STRICT

    approximant: str
    sampling_frequency: Annotated[float, Field(gt=0.0)]
    minimum_frequency: Annotated[float, Field(ge=0.0)]
    maximum_frequency: Annotated[float, Field(gt=0.0)]
    reference_frequency: Annotated[float, Field(gt=0.0)]
    frequency_resolution: Annotated[float, Field(gt=0.0)]
    chunk_size: Annotated[int, Field(gt=0)]


class ComponentSpec(BaseModel):
    """One population graph contributing to a catalog, with its own RNG stream.

    ``seed`` is per component rather than per catalog because a component's
    draw comes from its *construction* seed: ``MixtureSimulator`` passes each
    component a derived ``seed`` keyword that ``GraphSimulator`` discards.
    """

    model_config = _STRICT

    population: str
    seed: int
    weight: Annotated[float, Field(gt=0.0)] = 1.0

    def population_path(self, root: Path | None = None) -> Path:
        """Return the population graph this component is drawn from."""
        base = root if root is not None else Path()
        return base / POPULATIONS_DIR / f"{self.population}.yaml"


class CatalogDefinition(BaseModel):
    """Everything needed to generate one reusable catalog.

    ``seed`` and ``num_samples`` live here rather than in the population config
    on purpose: ``md-imrphenom-s41-n32768`` and ``md-imrphenom-s42-n16384`` are
    the *same* graph drawn at different seeds and sizes, so pushing either into
    the population config would mean near-identical graph files.
    """

    model_config = _STRICT

    name: str
    num_samples: Annotated[int, Field(gt=0)]
    components: Annotated[tuple[ComponentSpec, ...], Field(min_length=1)]
    waveform: WaveformConfig
    mixture_seed: int | None = None

    @model_validator(mode="after")
    def _validate_mixture_seed(self) -> CatalogDefinition:
        """A mixture seeds its assignment draw; a lone component has none to seed."""
        if len(self.components) > 1 and self.mixture_seed is None:
            raise ValueError("a multi-component catalog requires a mixture_seed")
        if len(self.components) == 1 and self.mixture_seed is not None:
            raise ValueError("a single-component catalog forbids mixture_seed")
        return self

    @property
    def is_mixture(self) -> bool:
        """Whether this catalog draws through a component-assignment step."""
        return len(self.components) > 1

    @property
    def draw_seed(self) -> int:
        """The seed the draw is actually made with.

        A mixture's own seed governs the component assignments; a lone
        component's governs its graph directly. The validator above ties
        ``mixture_seed`` to the component count, so exactly one applies.
        """
        if self.mixture_seed is not None:
            return self.mixture_seed
        return self.components[0].seed

    def population_paths(self, root: Path | None = None) -> tuple[Path, ...]:
        """Every population graph this catalog draws from, in component order."""
        return tuple(component.population_path(root) for component in self.components)


def load_catalog_layers(paths: Sequence[Path]) -> CatalogDefinition:
    """Merge and validate one catalog from its ordered layer files.

    The last layer is the catalog's own definition, and its filename stem is
    the catalog name -- the same convention that maps it to
    ``outputs/catalogs/<stem>.h5``. Taking layers on argv rather than a name
    keeps the generator symmetric with every other entrypoint: the workflow
    rule declares exactly these files as its ``input:``.
    """
    if not paths:
        raise ValueError("no catalog config layers given")
    merged = merge_config_layers(paths)
    return CatalogDefinition.model_validate({"name": Path(paths[-1]).stem, **merged})


def load_catalog_definition(name: str, root: Path | None = None) -> CatalogDefinition:
    """Merge and validate one catalog's layers, addressing it by name."""
    return load_catalog_layers(catalog_config_paths(name, root=root))


def discover_catalogs(root: Path | None = None) -> dict[str, CatalogDefinition]:
    """Load every committed catalog config, keyed by name, in sorted order."""
    return {
        name: load_catalog_definition(name, root)
        for name in discover_catalog_names(root)
    }


# --------------------------------------------------------------------------- #
# Redshift proposal descriptors
# --------------------------------------------------------------------------- #
class MadauDickinsonProposal(BaseModel):
    """The Madau-Dickinson redshift density a catalog's samples were drawn from."""

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
    """The uniform redshift density a guard component was drawn from."""

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


class ProposalComponent(BaseModel):
    """One weighted component of the density a catalog's samples follow."""

    model_config = _STRICT

    weight: Annotated[float, Field(gt=0.0, le=1.0)]
    density: RedshiftProposal


class MixtureProposal(BaseModel):
    """The complete redshift density a catalog's samples were drawn from.

    A single-component catalog is a one-entry mixture at weight 1, so the
    analysis reads one shape regardless of how the catalog was built. Weights
    are normalized at construction, mirroring ``MixtureSimulator``.
    """

    model_config = _STRICT

    components: Annotated[tuple[ProposalComponent, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _validate_support(self) -> MixtureProposal:
        supports = {(c.density.z_min, c.density.z_max) for c in self.components}
        if len(supports) > 1:
            described = ", ".join(
                f"{c.density.kind} [{c.density.z_min:.4g}, {c.density.z_max:.4g}]"
                for c in self.components
            )
            raise ValueError(
                f"mixture components disagree on generation redshift support: "
                f"{described}"
            )
        total = sum(c.weight for c in self.components)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"mixture weights must sum to 1, got {total:.6g}")
        return self

    @property
    def z_min(self) -> float:
        """The generation support's lower edge, shared by every component."""
        return self.components[0].density.z_min

    @property
    def z_max(self) -> float:
        """The generation support's upper edge, shared by every component."""
        return self.components[0].density.z_max

    def madau_dickinson(self, *, label: str) -> MadauDickinsonProposal:
        """The single Madau-Dickinson component, or fail clearly."""
        matches = [
            c.density
            for c in self.components
            if isinstance(c.density, MadauDickinsonProposal)
        ]
        if len(matches) != 1:
            kinds = ", ".join(c.density.kind for c in self.components)
            raise ValueError(
                f"catalog {label} must have exactly one madau_dickinson component "
                f"for the proposal role; found components [{kinds}]"
            )
        return matches[0]

    @property
    def uniform_mixing_fraction(self) -> float:
        """The total weight carried by uniform-redshift guard components."""
        return sum(
            c.weight
            for c in self.components
            if isinstance(c.density, UniformRedshiftProposal)
        )


_PROPOSAL_ADAPTER: TypeAdapter[MixtureProposal] = TypeAdapter(MixtureProposal)


# --------------------------------------------------------------------------- #
# Provenance: what a catalog file records about its own generation
# --------------------------------------------------------------------------- #
class CatalogProvenance(BaseModel):
    """What a catalog file records about its own generation."""

    model_config = _STRICT

    name: str
    seed: int
    num_samples: Annotated[int, Field(gt=0)]
    redshift_proposal: MixtureProposal

    def to_population_metadata(self) -> PopulationMetadata:
        """Convert catalog provenance into the core population metadata model."""
        from astrogwb.catalog import PopulationMetadata

        return PopulationMetadata(
            name=self.name,
            seed=self.seed,
            num_samples=self.num_samples,
            source_type="bns",
            provenance={PROPOSAL_ATTR: self.redshift_proposal.model_dump_json()},
        )

    @classmethod
    def from_population_metadata(
        cls, metadata: PopulationMetadata, *, label: str
    ) -> Self:
        """Convert decoded catalog metadata into validated provenance."""
        if PROPOSAL_ATTR not in metadata.provenance:
            raise ValueError(
                f"catalog {label} was generated before proposal metadata "
                f"(missing {PROPOSAL_ATTR}); regenerate it"
            )
        return cls(
            name=metadata.name,
            seed=metadata.seed,
            num_samples=metadata.num_samples,
            redshift_proposal=_PROPOSAL_ADAPTER.validate_json(
                str(metadata.provenance[PROPOSAL_ATTR])
            ),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Read one catalog's recorded provenance from its HDF5 attributes.

        A file written before mixture provenance existed carries no readable
        ``redshift_proposal`` attribute. That is an error, never a cue to fall
        back to parsing the population config: the point of the attribute is
        that the config may have drifted since the catalog was built.
        """
        from astrogwb.catalog.io import open_catalog, population_metadata_from_attrs

        path = Path(path)
        with open_catalog(path) as catalog:
            metadata = population_metadata_from_attrs(catalog.attrs, label=str(path))
        return cls.from_population_metadata(metadata, label=str(path))


# --------------------------------------------------------------------------- #
# Write side: population graphs -> descriptor
# --------------------------------------------------------------------------- #
def extract_redshift_proposal(graph: Mapping[str, Any]) -> RedshiftProposal:
    """Read the redshift proposal density out of a population graph config.

    This is the only place the graph YAML is interpreted as a *density*, and it
    runs once per catalog at generation time. An unrecognised sampler
    ``function`` is rejected rather than defaulted: the descriptor is written
    once and trusted forever after, so a wrong guess here is a silently wrong
    likelihood in every analysis that follows.
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
                f"unsupported redshift sampler function {function!r}; the catalog "
                "proposal descriptor can only be derived from "
                "'madau_dickinson_redshift' or 'uniform'"
            )


def extract_mixture_proposal(
    graphs: Sequence[tuple[Mapping[str, Any], float]],
) -> MixtureProposal:
    """Build a catalog's full density descriptor from its (graph, weight) pairs.

    Weights are normalized here so the descriptor records the actual mixing
    fractions, matching what ``MixtureSimulator`` draws from.
    """
    total = sum(weight for _, weight in graphs)
    if total <= 0.0:
        raise ValueError("catalog component weights must sum to a positive value")
    return MixtureProposal(
        components=tuple(
            ProposalComponent(
                weight=weight / total, density=extract_redshift_proposal(graph)
            )
            for graph, weight in graphs
        )
    )


# --------------------------------------------------------------------------- #
# Derivation: descriptor + analysis window -> ProposalConfig
# --------------------------------------------------------------------------- #
def resolve_proposal(
    mixture: MixtureProposal,
    *,
    minimum_redshift: float,
    maximum_redshift: float,
    label: str,
) -> ProposalConfig:
    """Restrict a catalog's generation density to the analysis window.

    Generation draws truncated to the window follow the same law as drawing
    directly from it, so only the support fields change. The window must lie
    inside what was actually generated.
    """
    md = mixture.madau_dickinson(label=label)
    if not mixture.z_min <= minimum_redshift < maximum_redshift <= mixture.z_max:
        raise ValueError(
            f"analysis redshift support [{minimum_redshift:.4g}, "
            f"{maximum_redshift:.4g}] must lie within the catalog generation "
            f"support [{mixture.z_min:.4g}, {mixture.z_max:.4g}]"
        )
    return ProposalConfig(
        uniform_mixing_fraction=mixture.uniform_mixing_fraction,
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
    """The MD constants a run's [fiducials] must share with its proposal catalog."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    H0: float
    Omega_m: float
    gamma: float
    kappa: float
    z_peak: float


def check_fiducials_match(
    provenance: CatalogProvenance,
    fiducials: Mapping[str, float],
    *,
    label: str,
) -> None:
    """Require a run's fiducials to equal what the proposal catalog was drawn at.

    The importance weights divide the fiducial merger-rate density by the
    proposal density, so a mismatch here is not a bookkeeping slip: it silently
    reweights against the wrong denominator. Compared against what was actually
    generated, not against a re-parse of the population config.
    """
    proposal = provenance.redshift_proposal.madau_dickinson(label=label)
    try:
        run = _MDFiducials.model_validate(dict(fiducials))
    except ValidationError as exc:
        missing = [
            str(error["loc"][0]) for error in exc.errors() if error["type"] == "missing"
        ]
        raise ValueError(
            f"run fiducials do not match proposal catalog {label}: "
            + ", ".join(f"{name} (missing from [fiducials])" for name in missing)
        ) from None
    generated = _MDFiducials.model_validate(
        proposal.model_dump(include=set(MD_FIDUCIAL_NAMES))
    )
    mismatches = [
        f"{name} (run {getattr(run, name):.6g} vs catalog {getattr(generated, name):.6g})"
        for name in MD_FIDUCIAL_NAMES
        if getattr(run, name) != getattr(generated, name)
    ]
    if mismatches:
        raise ValueError(
            f"run fiducials do not match proposal catalog {label}: "
            + ", ".join(mismatches)
        )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Narrow an arbitrary loaded YAML value to a mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


# --------------------------------------------------------------------------- #
# Run-config gate: does a run name catalogs that exist?
# --------------------------------------------------------------------------- #
def check_catalog_references(
    config: RunConfig,
    *,
    label: str,
    catalogs: Mapping[str, CatalogDefinition] | None = None,
) -> None:
    """Reject a run naming an unknown catalog.

    Runs against the committed catalog configs rather than the built files, so
    a typo fails without building anything expensive. Without it the typo would
    only surface as a Snakemake wildcard that matches no rule.

    Lives here rather than in :mod:`astrogwb.paper.config.runs` because it
    needs both a validated ``RunConfig`` and the catalog registry, and that
    module must stay importable by the ``Snakefile`` without pydantic.
    """
    known = catalogs if catalogs is not None else discover_catalogs()
    for role, name in (
        ("injection", config.catalog.injection),
        ("proposal", config.catalog.proposal),
    ):
        if name not in known:
            choices = ", ".join(known)
            raise ValueError(
                f"{label} catalog.{role} names unknown catalog {name!r}; "
                f"choose from {choices}"
            )


def validate_all_runs(root: Path | None = None) -> list[str]:
    """Merge, validate, and catalog-check every declared run; return their labels.

    The pre-flight gate ``astrogwb-assemble-config --all`` used to provide,
    kept because its real value was never the JSON it wrote: it fails on the
    first invalid run *before any catalog is built*, and a catalog is a GPU job.

    It lives here rather than in :mod:`astrogwb.paper.config.runs` for the same
    reason :func:`check_catalog_references` does -- that module must stay
    importable by the ``Snakefile`` without pydantic. Catalog configs are loaded
    once: this is one gate over all runs, not a per-run check.
    """
    from astrogwb.paper.config.mcmc import build_run_config
    from astrogwb.paper.config.runs import assemble_run, discover_runs

    catalogs = discover_catalogs(root)
    labels: list[str] = []
    for experiment, runs in discover_runs(root).items():
        for run in runs:
            label = f"{experiment}/{run}"
            config = build_run_config(assemble_run(experiment, run, root=root))
            check_catalog_references(config, label=label, catalogs=catalogs)
            logger.info("ok %s", label)
            labels.append(label)
    return labels


__all__ = [
    "CATALOG_DEFS_DIR",
    "CatalogDefinition",
    "CatalogProvenance",
    "ComponentSpec",
    "MadauDickinsonProposal",
    "MixtureProposal",
    "ProposalComponent",
    "UniformRedshiftProposal",
    "WaveformConfig",
    "check_catalog_references",
    "check_fiducials_match",
    "discover_catalogs",
    "extract_mixture_proposal",
    "extract_redshift_proposal",
    "load_catalog_definition",
    "load_catalog_layers",
    "resolve_proposal",
    "validate_all_runs",
]
