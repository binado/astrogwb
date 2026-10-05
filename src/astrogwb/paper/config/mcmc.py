"""Pydantic models and I/O for headless MCMC run configs.

Importing this module loads the detector registry, and with it JAX. ``numpyro``
is imported lazily inside :func:`materialize_prior` / :func:`prior_to_spec`.
The constructed distributions hold plain Python floats and no JAX operation is
evaluated, so validating a config leaves the XLA backend uninitialized. That
property is what :func:`astrogwb.paper.runtime.configure_runtime` relies on to
set host device count / platform after config validation; it is guarded by a
subprocess test in ``tests/test_prior_native_types.py`` (re-running
``set_host_device_count`` after a backend init is a silent no-op, hence the
subprocess).

The generic merge/load helpers (``deep_merge``, ``load_mapping``) live in
:mod:`astrogwb.paper.utils`, and the run-assembly merge semantics live in
:mod:`astrogwb.paper.config.runs`; the validated run models live here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    model_validator,
)

from astrogwb.metadata import PriorSpec
from astrogwb.paper.config.detectors import DetectorRegistry
from astrogwb.paper.config.runs import CATALOG_ROLES
from astrogwb.paper.utils import deep_merge
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.polarization_power import CatalogMetadata

_STRICT = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
# Restates astrogwb.populations.bns_madau_dickinson.AMPLITUDE_PARAMETERS rather
# than importing the population registry. tests/paper/test_config.py
# cross-checks the two lists.
AmplitudeParameter = Literal["H0", "local_merger_rate"]

#: Restates astrogwb.populations.DEFAULT_DENSITY_SITES, for the same reason and
#: under the same cross-check.
DEFAULT_DENSITY_SITES: tuple[str, ...] = (
    "redshift",
    "source_frame_mass_1",
    "source_frame_mass_2",
)

#: The construction kwargs a population must be given to be evaluated on a
#: redshift grid. Checked, through :func:`check_redshift_grid`, on the analysis
#: target and on each catalog request.
REDSHIFT_GRID_KWARGS: tuple[str, ...] = (
    "minimum_redshift",
    "maximum_redshift",
    "n_grid",
)


def check_redshift_grid(
    model_kwargs: Mapping[str, Any], *, label: str, required: bool = False
) -> None:
    """Reject a redshift window that is empty, or absent where it is needed.

    One check for the two places a population's construction kwargs are
    declared. ``required`` is what differs between them: an analysis target is
    *always* built on a grid -- the same grid its spectral integral runs on --
    while a catalog's population is only checked for a window when it declares
    one, since the shared layer, not this function, decides what a draw needs.
    """
    if required:
        missing = [name for name in REDSHIFT_GRID_KWARGS if name not in model_kwargs]
        if missing:
            raise ValueError(f"{label} must declare {', '.join(missing)}")
    window = ("minimum_redshift", "maximum_redshift")
    if all(name in model_kwargs for name in window) and not float(
        model_kwargs["minimum_redshift"]
    ) < float(model_kwargs["maximum_redshift"]):
        raise ValueError(
            f"{label}.minimum_redshift must be less than {label}.maximum_redshift"
        )


# Wire protocol for the priors tables:
#
#     {"dist": "<numpyro.distributions class name>", "kwargs": {...}}
#
# The spec itself is `astrogwb.metadata.PriorSpec`: a spectral-density draw
# records its sampled hyperparameters in the same format, so the record and its
# validation live in the core package and this module only adapts it to the
# run config's pydantic wire. Hand-rolled adapters rather than a PriorSpec
# field: the run config holds *live* distributions, which pydantic can never
# construct from raw config dicts natively (they are not models).


def materialize_prior(value: Any) -> Distribution:
    """Materialize a prior spec into a live ``numpyro`` distribution.

    Accepts a raw mapping (``{"dist": "Uniform", "kwargs": {"low": ...}}``) or
    an already-built distribution (passed through unchanged, so re-validation
    is idempotent).

    Construction only wraps Python floats and never evaluates a JAX op, which
    is what lets :func:`astrogwb.paper.runtime.configure_runtime` still set the
    host device count after a config has been validated. An unknown name is
    caught *after* numpyro is imported rather than before -- importing numpyro
    is not backend initialization, and nothing on the DAG-construction path
    calls this.
    """
    # Every malformed-spec branch below raises ValueError even where the fault
    # is a wrong *type*, which is what TRY004 objects to. It is deliberate:
    # this function runs as a pydantic BeforeValidator, and pydantic converts
    # only ValueError and AssertionError into a ValidationError. A TypeError
    # would escape as itself, past every `pytest.raises(ValidationError)` and
    # past the `except (ValueError, TypeError)` in scripts/validate_configs.py
    # that reports which run is broken.
    import numpyro.distributions as dist

    if isinstance(value, dist.Distribution):
        return value  # already materialized
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004
            f"cannot materialize a prior from {type(value).__name__!r}; expected a "
            "spec mapping or a numpyro Distribution"
        )
    try:
        spec = PriorSpec.model_validate(dict(value))
    except ValidationError as error:
        # Re-raised as a plain ValueError: a ValidationError raised inside a
        # validator is not re-wrapped with the outer field's location.
        raise ValueError(f"invalid prior spec {dict(value)!r}: {error}") from None
    return spec.build()


def prior_to_spec(prior: Distribution) -> dict[str, Any]:
    """Serialize a materialized prior back to its wire-format spec.

    Inverse of :func:`materialize_prior`; see
    :meth:`astrogwb.metadata.PriorSpec.from_distribution`.
    """
    return PriorSpec.from_distribution(prior).model_dump()


if TYPE_CHECKING:
    from numpyro.distributions import Distribution

    _PriorDists = Distribution
else:
    _PriorDists = Any

# Native pydantic wire for live prior distributions: validate from spec
# mappings/dists, serialize back to the spec dict so `model_dump(mode="json")`
# and `RunConfig.save` stay canonical. Numpyro types deliberately
# stay out of the runtime annotation (hence `Any`) so schema building never
# imports numpyro at module load.
PriorDistribution = Annotated[
    _PriorDists,
    BeforeValidator(materialize_prior),
    PlainSerializer(prior_to_spec),
]


class AnalysisConfig(BaseModel):
    model_config = _STRICT

    #: Resolved by `RunConfig._resolve_network` from the [networks] table that
    #: `config/detectors.toml` contributes to every run's merge. Required, and
    #: recorded by `RunConfig.save`: the chain's own config must say which
    #: detectors it was sampled with, not just which name they were reached by.
    detectors: tuple[str, ...]
    #: The name that resolved to `detectors`, kept alongside it so a saved
    #: config records the intent as well as the result. Optional at the model
    #: level so a saved config that already carries the resolved detector list
    #: re-validates. That every committed run names one is a repository test,
    #: not a model constraint.
    network: str | None = None
    observation_time: float = 1.0
    minimum_frequency: float
    maximum_frequency: float
    #: Unset (empty) -> every key in [priors] except a marginalized amplitude
    #: parameter; resolved by `RunConfig._resolve_sampled_params`, which needs
    #: the [priors] table and so cannot live here.
    sampled_params: tuple[str, ...] = ()
    #: The population the sampled hyperparameters describe: the target the
    #: importance weights are evaluated at. The same record a catalog carries,
    #: but evaluated rather than drawn from, so its seed is unused. Its
    #: ``model_kwargs`` carry the redshift window and grid, which is the one
    #: statement of them: the same three numbers build the target callables
    #: *and* define the grid its spectral integral runs on.
    #:
    #: The model name is validated against the registry by
    #: `astrogwb.paper.config.catalogs.check_population_model`, not here: this
    #: module must stay importable without JAX.
    population: PopulationMetadata
    #: The source-density factors importance weighting includes, for both
    #: sides of every weight. Declared by the analysis because no catalog
    #: depends on it -- the samples are the same whichever of their densities
    #: are counted. Ordered and load-bearing: the two mass sites are one
    #: conceptual ordered-pair contribution (the secondary's distribution is
    #: parameterized by the drawn primary), and dropping either gives silently
    #: wrong weights with no shape error anywhere.
    density_sites: tuple[str, ...] = DEFAULT_DENSITY_SITES
    #: The "observed" data: the catalog whose spectrum is the measurement.
    injection: CatalogMetadata
    #: The catalog the importance weights reweight.
    proposal: CatalogMetadata
    likelihood: Literal["default", "amplitude_marginalized"] = "default"
    amplitude_parameter: AmplitudeParameter | None = None
    amplitude_num_nodes: Annotated[int, Field(gt=1)] = 1024
    amplitude_prior_span_sigma: Annotated[float, Field(gt=0.0)] = 10.0

    @model_validator(mode="after")
    def _validate_target_grid(self) -> AnalysisConfig:
        check_redshift_grid(
            self.population.model_kwargs,
            label="analysis.population.model_kwargs",
            required=True,
        )
        return self

    @model_validator(mode="after")
    def _validate_amplitude_parameter(self) -> AnalysisConfig:
        marginalized = self.likelihood == "amplitude_marginalized"
        if marginalized and self.amplitude_parameter is None:
            raise ValueError(
                "analysis.amplitude_parameter is required when "
                "likelihood == 'amplitude_marginalized'"
            )
        if not marginalized and self.amplitude_parameter is not None:
            raise ValueError(
                "analysis.amplitude_parameter is only valid when "
                "likelihood == 'amplitude_marginalized'"
            )
        return self


class SamplerConfig(BaseModel):
    model_config = _STRICT

    #: The sampling RNG seed. It belongs to the sampler rather than to the run
    #: as a whole, so a run changes it by overriding `[sampler]` like any other
    #: sampler setting.
    seed: int = 42
    num_warmup: Annotated[int, Field(gt=0)]
    num_samples: Annotated[int, Field(gt=0)]
    num_chains: Annotated[int, Field(gt=0)] = 1
    target_accept: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.9
    dense_mass: bool = True
    max_tree_depth: Annotated[int, Field(gt=0, le=20)] = 10
    forward_mode_differentiation: bool = False
    progress_bar: bool = False
    jit_model_args: bool = True


class OutputConfig(BaseModel):
    model_config = _STRICT

    outdir: Path = Path("outputs/chains")
    label: str = ""


#: Top-level tables that exist only to be referenced: once the merge has
#: resolved every ``${...}``, what they said lives in the tables that referred
#: to them, so a run config drops them rather than validating or saving them.
AUTHORING_TABLES: tuple[str, ...] = ("catalog", "waveforms", "populations")


class RunConfig(BaseModel):
    model_config = _STRICT

    fiducials: dict[str, float]
    # Complete parameter name -> prior distribution table.
    # `analysis.sampled_params` selects the NUTS latents; every other
    # non-marginalized site is conditioned.
    priors: dict[str, PriorDistribution]
    analysis: AnalysisConfig
    sampler: SamplerConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    detector_registry: DetectorRegistry

    @model_validator(mode="before")
    @classmethod
    def _drop_authoring_tables(cls, data: Any) -> Any:
        """Drop the referenced-only tables; the saved config is resolved records."""
        if not isinstance(data, Mapping):
            return data
        return {
            key: value for key, value in data.items() if key not in AUTHORING_TABLES
        }

    @model_validator(mode="before")
    @classmethod
    def _resolve_network(cls, data: Any) -> Any:
        """Resolve overrides and membership, or restore a complete saved registry.

        Merged layers carry complete `[detectors]` and `[networks]` tables.
        Saved configs carry their resolved `detector_registry` instead, so
        they never reinterpret geometry against newer packaged defaults.
        """
        if not isinstance(data, Mapping):
            return data
        raw = dict(data)
        table = raw.pop("networks", None)
        overrides = raw.pop("detectors", None)
        saved_registry = raw.get("detector_registry")
        if saved_registry is not None:
            if table is not None or overrides is not None:
                raise ValueError(
                    "resolved detector_registry cannot accompany detector overrides"
                )
            registry = DetectorRegistry.model_validate(saved_registry)
        else:
            registry = DetectorRegistry.model_validate(
                {"detectors": overrides, "networks": table or {}}
            )
        raw["detector_registry"] = registry
        if saved_registry is not None:
            table = registry.networks
        analysis = raw.get("analysis")
        if not isinstance(analysis, Mapping):
            return raw  # let field validation report the real problem
        analysis = dict(analysis)
        name = analysis.get("network")
        if name is None:
            return raw
        declared = analysis.get("detectors")

        if not isinstance(table, Mapping) or name not in table:
            known = sorted(table) if isinstance(table, Mapping) else []
            raise ValueError(
                f"analysis.network {name!r} is not declared in [networks]; "
                f"known networks: {known}"
            )

        resolved = tuple(table[name])
        if declared is not None and tuple(declared) != resolved:
            raise ValueError(
                f"analysis declares detectors {tuple(declared)} but names "
                f"network {name!r}, which is {resolved}; drop the list and keep "
                "the name"
            )
        analysis["detectors"] = resolved
        raw["analysis"] = analysis
        return raw

    @model_validator(mode="after")
    def _resolve_sampled_params(self) -> RunConfig:
        if not self.fiducials:
            raise ValueError("config must define a non-empty [fiducials] table")
        if not self.priors:
            raise ValueError("config must define at least one [priors.<param>] table")

        self.detector_registry.validate_members(self.analysis.detectors)

        priors = dict(self.priors)
        amplitude_parameter = self.analysis.amplitude_parameter
        if amplitude_parameter is not None:
            if amplitude_parameter in self.analysis.sampled_params:
                raise ValueError(
                    f"analysis.amplitude_parameter {amplitude_parameter!r} "
                    "cannot also appear in sampled_params"
                )
            if amplitude_parameter not in self.fiducials:
                raise ValueError(
                    f"analysis.amplitude_parameter {amplitude_parameter!r} "
                    "missing from [fiducials]"
                )
            if amplitude_parameter not in priors:
                raise ValueError(
                    f"analysis.amplitude_parameter {amplitude_parameter!r} needs "
                    "a [priors.*] table"
                )

        missing_priors = [name for name in self.fiducials if name not in priors]
        if missing_priors:
            raise ValueError(f"fiducials without a [priors.*] table: {missing_priors}")
        missing_fiducials = [name for name in priors if name not in self.fiducials]
        if missing_fiducials:
            raise ValueError(f"priors missing from [fiducials]: {missing_fiducials}")

        sampled = self.analysis.sampled_params or tuple(
            p for p in priors if p != amplitude_parameter
        )
        missing_priors = [p for p in sampled if p not in priors]
        if missing_priors:
            raise ValueError(
                f"sampled_params without a [priors.*] table: {missing_priors}"
            )
        missing_fid = [p for p in sampled if p not in self.fiducials]
        if missing_fid:
            raise ValueError(f"sampled_params missing from [fiducials]: {missing_fid}")

        # The field lives on `analysis` but only `RunConfig` can default it:
        # the fallback is "every prior except a marginalized amplitude", and
        # [priors] is a sibling block. `object.__setattr__` writes straight into
        # the nested model's `__dict__`, past `frozen=True` -- the same escape
        # this validator has always used, one level down.
        object.__setattr__(self.analysis, "sampled_params", sampled)
        return self

    @property
    def fixed_params(self) -> dict[str, float]:
        """Every fiducial not sampled: values supplied by effect handlers.

        Includes the marginalized amplitude parameter when present: it has no
        NUTS latent, but its fiducial value is still what the model pins it
        to. This is a plain property and is not serialized.
        """
        return {
            k: v
            for k, v in self.fiducials.items()
            if k not in self.analysis.sampled_params
        }

    def catalog_request(self, role: str) -> CatalogMetadata:
        """The catalog one role of this run samples against, by role name.

        ``analysis.injection`` or ``analysis.proposal``: each is already a
        complete record, resolved by the merge that the ``Snakefile`` keys its
        catalog files with, so the file a run is handed and the request it
        checks that file against cannot be derived two ways.
        """
        if role not in CATALOG_ROLES:
            raise ValueError(
                f"unknown catalog role {role!r}; roles are {CATALOG_ROLES}"
            )
        return getattr(self.analysis, role)

    def save(self, path: Path) -> None:
        """Write the validated run config as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )


def build_run_config(
    raw: dict[str, Any],
    *,
    seed: int | None = None,
    outdir: Path | None = None,
    label: str | None = None,
    **overrides: Any,
) -> RunConfig:
    """Apply optional overrides and validate a raw config mapping into a RunConfig.

    Named ``seed`` / ``outdir`` / ``label`` remain for CLI compatibility. Extra
    ``**overrides`` are deep-merged into the raw mapping (nested dicts merge;
    other values replace) before validation. ``raw`` is already resolved, so an
    override reaches only the table it names: overriding ``fiducials`` does not
    re-draw the catalogs, whose records were resolved by the merge.
    """
    cli_overrides: dict[str, Any] = {}
    if seed is not None:
        cli_overrides["sampler"] = {"seed": seed}
    if outdir is not None or label is not None:
        output: dict[str, Any] = {}
        if outdir is not None:
            output["outdir"] = str(outdir)
        if label is not None:
            output["label"] = label
        cli_overrides["output"] = output

    merged = deep_merge(raw, deep_merge(overrides, cli_overrides))
    return RunConfig.model_validate(merged)
