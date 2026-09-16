"""Pydantic models and I/O for headless MCMC run configs.

Importing this module requires only stdlib and pydantic: ``numpyro`` is
imported lazily inside :func:`materialize_prior` / :func:`prior_to_spec`, so
importing this module stays cheap and -- because the constructed
distributions hold plain Python floats and no JAX op is ever evaluated --
validating a config does not initialize the XLA backend. That last property is what
:func:`astrogwb.paper.runtime.configure_runtime` relies on to set host device
count / platform after config validation; it is guarded by a subprocess test
in ``tests/test_prior_native_types.py`` (re-running ``set_host_device_count``
after a backend init is a silent no-op, hence the subprocess).

The generic merge/load helpers (``deep_merge``, ``load_mapping``) live in
:mod:`astrogwb.paper.utils`, and the run-assembly merge semantics live in
:mod:`astrogwb.paper.config.runs`; only the ``AnalysisGrid`` shared by every
experiment run lives here next to the models.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)

from astrogwb.paper.utils import deep_merge

_STRICT = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalysisGrid:
    """Frequency band and redshift grid shared by every experiment run."""

    observation_time: float
    minimum_frequency: float
    maximum_frequency: float
    minimum_redshift: float
    maximum_redshift: float
    n_grid: int


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
# Restates astrogwb.populations.bns_madau_dickinson.AMPLITUDE_PARAMETERS rather
# than importing it: this module must stay stdlib+pydantic only (see module
# docstring), so a @pytest.mark.integration paper test cross-checks the two
# lists instead.
AmplitudeParameter = Literal["H0", "local_merger_rate"]


# Wire protocol for the priors tables:
#
#     {"dist": "<numpyro.distributions class name>", "kwargs": {...}}
#
# The class is looked up on `numpyro.distributions` by name, so adding a
# distribution is zero-code -- it needs no registry entry here and no branch in
# `prior_to_spec`, which recovers the same keys from the class's own
# `arg_constraints`. Hand-rolled rather than a pydantic spec model: pydantic
# could never construct distributions from raw config dicts natively anyway
# (they are not models), so the validation lives next to the construction it
# guards.
#
# There is deliberately no positional `args` form. `prior_to_spec` can only
# ever emit kwargs, so a second spelling would make `RunConfig.save()` ->
# reload non-canonical.
_PRIOR_SPEC_KEYS = frozenset({"dist", "kwargs"})


def materialize_prior(value: Any) -> Distribution:
    """Materialize a prior spec into a live ``numpyro`` distribution.

    Accepts a raw mapping (``{"dist": "Uniform", "kwargs": {"low": ...}}``) or
    an already-built distribution (passed through unchanged, so re-validation
    is idempotent).

    Construction only wraps Python floats and never evaluates a JAX op, which
    is what lets :func:`astrogwb.paper.runtime.configure_runtime` still set the
    host device count after a config has been validated. Unlike the previous
    two-tag protocol, an unknown name is caught *after* numpyro is imported
    rather than before -- importing numpyro is not backend initialization, and
    nothing on the DAG-construction path calls this.
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

    extra = sorted({str(key) for key in value} - _PRIOR_SPEC_KEYS)
    if extra:
        raise ValueError(f"Extra inputs are not permitted for a prior spec: {extra}")
    name = value.get("dist")
    if not isinstance(name, str):
        raise ValueError("a prior spec must name a distribution in 'dist'")  # noqa: TRY004
    kwargs = value.get("kwargs")
    if not isinstance(kwargs, Mapping):
        raise ValueError(f"prior {name!r} must carry a 'kwargs' table")  # noqa: TRY004

    cls = getattr(dist, name, None)
    # `getattr` on a config-supplied string: the guard is what keeps it to
    # distribution classes rather than any attribute the module happens to
    # expose.
    if not (isinstance(cls, type) and issubclass(cls, dist.Distribution)):
        raise ValueError(f"{name!r} is not a numpyro distribution")  # noqa: TRY004

    expected = set(cls.arg_constraints)
    missing = sorted(expected - set(kwargs))
    if missing:
        raise ValueError(f"missing required key(s) {missing} for a {name} prior")
    unexpected = sorted(set(kwargs) - expected)
    if unexpected:
        raise ValueError(
            f"Extra inputs are not permitted for a {name} prior: {unexpected}"
        )
    return cls(**{key: float(val) for key, val in kwargs.items()})


def prior_to_spec(prior: Distribution) -> dict[str, Any]:
    """Serialize a materialized prior back to its wire-format spec.

    Inverse of :func:`materialize_prior`. The constructor keyword names are
    recovered from the class's own ``arg_constraints``, so this stays correct
    for any distribution without a branch per type. The spec-constructed
    distributions this module produces hold plain Python floats, so
    ``float(...)`` never touches JAX.
    """
    import numpyro.distributions as dist

    if not isinstance(prior, dist.Distribution):
        raise TypeError(f"cannot serialize {type(prior).__name__!r} as a prior spec")
    cls = type(prior)
    return {
        "dist": cls.__name__,
        "kwargs": {key: float(getattr(prior, key)) for key in cls.arg_constraints},
    }


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
    #: `config/networks.json` contributes to every run's merge. Required, and
    #: recorded by `RunConfig.save`: the chain's own config must say which
    #: detectors it was sampled with, not just which name they were reached by.
    detectors: tuple[str, ...]
    #: The name that resolved to `detectors`, kept alongside it so a saved
    #: config records the intent as well as the result. Optional at the model
    #: level so a saved config -- which carries `detectors` and no [networks]
    #: table -- re-validates. That every *committed* run names one is a repo
    #: test, not a model constraint.
    network: str | None = None
    observation_time: float = 1.0
    minimum_frequency: float
    maximum_frequency: float
    #: Unset (empty) -> every key in [priors] except a marginalized amplitude
    #: parameter; resolved by `RunConfig._resolve_sampled_params`, which needs
    #: the [priors] table and so cannot live here.
    sampled_params: tuple[str, ...] = ()
    # The registered population the sampled hyperparameters describe. The
    # default is the one every committed run uses; it reduces exactly to the
    # plain cosmological population at xi_0 = 1, which is how a run that does
    # not sample the propagation parameters gets the standard law without
    # naming a second population. An analysis target must declare a merger
    # rate, so a guard mixture cannot be named here. Validated against the
    # registry by `astrogwb.paper.config.catalogs.check_population_model`, not
    # here: this module must stay importable without JAX.
    population_model: str = "bns_md_modified_propagation"
    likelihood: Literal["default", "amplitude_marginalized"] = "default"
    amplitude_parameter: AmplitudeParameter | None = None
    amplitude_num_nodes: Annotated[int, Field(gt=1)] = 1024
    amplitude_prior_span_sigma: Annotated[float, Field(gt=0.0)] = 10.0

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


class CosmoConfig(BaseModel):
    model_config = _STRICT

    minimum_redshift: float
    maximum_redshift: float
    n_grid: int


class SamplerConfig(BaseModel):
    model_config = _STRICT

    #: The sampling RNG seed. It belongs to the sampler rather than to the run
    #: as a whole, which is what lets `config/sampler.json` be the single-key
    #: layer its stem names.
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


class CatalogConfig(BaseModel):
    """The two catalogs this run uses: the injection and the proposal.

    Each role names one persisted catalog under ``config/catalogs``, whose
    stem is both the config filename and the ``outputs/catalogs/<name>.h5`` it
    produces. The run records the *name* only: everything about how the catalog
    was drawn -- the population model, its construction settings, the
    hyperparameters, and the included density factors -- is recorded in the
    file itself and read back at run time. The two roles differ by filename and
    nothing else.
    """

    model_config = _STRICT

    injection: str
    proposal: str


class RunConfig(BaseModel):
    model_config = _STRICT

    fiducials: dict[str, float]
    # Complete parameter name -> prior distribution table.
    # `analysis.sampled_params` selects the NUTS latents; every other
    # non-marginalized site is conditioned.
    priors: dict[str, PriorDistribution]
    analysis: AnalysisConfig
    cosmology: CosmoConfig
    catalog: CatalogConfig
    sampler: SamplerConfig
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="before")
    @classmethod
    def _resolve_network(cls, data: Any) -> Any:
        """Turn ``analysis.network`` into ``analysis.detectors``, dropping the table.

        ``config/networks.json`` is a merge layer, so the lookup table arrives
        in the same mapping as the run that names one -- resolution is a pure
        function of the merged config and needs no file I/O, which is what
        keeps this module stdlib+pydantic.

        The table is *input to validation*, never a field: as a field it would
        either land in every ``save()`` next to every chain, or need
        ``exclude=True`` and break the save/reload round-trip. A validator
        rather than a step in :func:`build_run_config` because
        ``RunConfig.model_validate`` is public and callable directly, and
        because the resulting ``ValidationError`` is what
        ``scripts/validate_configs.py`` and the test suite already catch.

        A config that already carries ``detectors`` is the ``save()`` output
        being re-validated: ``save`` records the resolved list *and* the name it
        came from, so both are present, and no ``[networks]`` table is. That
        round-trip is why the two are cross-checked rather than rejected
        outright -- when a table *is* present, as it always is in the config
        tree, a hand-written list that disagrees with the named network is the
        real error worth catching.
        """
        if not isinstance(data, Mapping):
            return data
        raw = dict(data)
        table = raw.pop("networks", None)
        analysis = raw.get("analysis")
        if not isinstance(analysis, Mapping):
            return raw  # let field validation report the real problem
        analysis = dict(analysis)
        name = analysis.get("network")
        if name is None:
            return raw
        declared = analysis.get("detectors")

        if not isinstance(table, Mapping) or name not in table:
            if declared is not None:
                return raw  # a saved config, re-validating without the table
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

    @property
    def analysis_grid(self) -> AnalysisGrid:
        """The frequency band and redshift grid this run's inputs are built on.

        A plain property, deliberately not serialized: `save` writes
        only inputs needed to reconstruct the validated run configuration.
        """
        return AnalysisGrid(
            observation_time=self.analysis.observation_time,
            minimum_frequency=self.analysis.minimum_frequency,
            maximum_frequency=self.analysis.maximum_frequency,
            minimum_redshift=self.cosmology.minimum_redshift,
            maximum_redshift=self.cosmology.maximum_redshift,
            n_grid=self.cosmology.n_grid,
        )

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
    other values replace) before validation.
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
