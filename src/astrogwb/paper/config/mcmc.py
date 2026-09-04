"""Pydantic models and I/O for headless MCMC run configs.

Importing this module requires only stdlib and pydantic: ``numpyro`` is
imported lazily inside :func:`materialize_prior` / :func:`prior_to_spec`, so
parsing and validating a config stays cheap and -- because the constructed
distributions hold plain Python floats and no JAX op is ever evaluated --
does not initialize the XLA backend. That last property is what
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
    f_min: float
    f_max: float
    minimum_redshift: float
    maximum_redshift: float
    n_grid: int


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
# Restates astrogwb.importance.models.bns_madau_dickinson_modified_propagation
# .AMPLITUDE_PARAMETERS rather than importing it: this module must stay
# stdlib+pydantic only (see module docstring), so a
# @pytest.mark.integration paper test cross-checks the two lists instead.
AmplitudeParameter = Literal["H0", "local_merger_rate"]


# Wire protocol for the priors tables: tag name -> ordered parameter keys.
# Hand-rolled inside materialize_prior (no pydantic spec models): pydantic
# could never construct distributions from raw config dicts natively anyway
# (they are not models), so the mapping validation lives here, next to the
# construction it guards. Unsupported tags fail before numpyro is imported.
_PRIOR_PARAMS: dict[str, tuple[str, ...]] = {
    "uniform": ("low", "high"),
    "normal": ("loc", "scale"),
}


def materialize_prior(value: Any) -> Distribution:
    """Materialize a prior spec into a live ``numpyro`` distribution.

    Accepts a raw mapping (``{"type": "uniform", "low": ..., ...}``) or an
    already-built distribution (passed through unchanged, so re-validation is
    idempotent). Spec validation happens *before* numpyro is imported, so an
    unsupported ``type`` fails fast and cheap; construction itself only wraps
    Python floats and never evaluates a JAX op.
    """
    if isinstance(value, Mapping):
        kind = value.get("type")
        if kind not in _PRIOR_PARAMS:
            raise ValueError(
                f"prior type {kind!r} does not match any of the expected tags: "
                f"{sorted(_PRIOR_PARAMS)}"
            )
        params = _PRIOR_PARAMS[kind]
        missing = [p for p in params if p not in value]
        if missing:
            raise ValueError(f"missing required key(s) {missing} for a {kind} prior")
        extra = sorted({str(k) for k in value} - {"type", *params})
        if extra:
            raise ValueError(
                f"Extra inputs are not permitted for a {kind} prior: {extra}"
            )

        import numpyro.distributions as dist

        cls = dist.Uniform if kind == "uniform" else dist.Normal
        return cls(**{name: float(value[name]) for name in params})

    import numpyro.distributions as dist

    if isinstance(value, dist.Distribution):
        return value  # already materialized
    raise ValueError(
        f"cannot materialize a prior from {type(value).__name__!r}; expected a "
        "spec mapping or a numpyro Distribution"
    )


def prior_to_spec(prior: Distribution) -> dict[str, str | float]:
    """Serialize a materialized prior back to its wire-format spec.

    Inverse of :func:`materialize_prior` for the spec-constructed
    distributions this module produces: their parameters are plain Python
    floats, so ``float(...)`` never touches JAX.
    """
    import numpyro.distributions as dist

    match prior:
        case dist.Uniform():
            return {
                "type": "uniform",
                "low": float(prior.low),
                "high": float(prior.high),
            }
        case dist.Normal():
            return {
                "type": "normal",
                "loc": float(prior.loc),
                "scale": float(prior.scale),
            }
        case _:
            raise TypeError(
                f"cannot serialize {type(prior).__name__!r} as a prior spec"
            )


if TYPE_CHECKING:
    from numpyro.distributions import Distribution, Normal, Uniform

    _PriorDists = Uniform | Normal
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

    detectors: tuple[str, ...]
    f_min: float
    f_max: float
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


class ProposalConfig(BaseModel):
    """The fixed redshift density the importance weights divide by.

    Not a config *input*: it is derived at run time from the proposal
    catalog's recorded provenance plus the run's analysis window (see
    :func:`astrogwb.paper.config.catalogs.resolve_proposal`). Scripts and
    notebooks that reweight outside the sampler construct one directly.

    Note the name collision with ``RunConfig.catalog.proposal``, which is kept
    deliberately: that field names the *catalog* while this one is the
    *density* its samples follow.
    """

    model_config = _STRICT

    uniform_mixing_fraction: Annotated[
        float, Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ]
    minimum_redshift: float
    maximum_redshift: float
    n_grid: Annotated[int, Field(gt=1)]
    H0: float
    Omega_m: float
    gamma: float
    kappa: float
    z_peak: float

    @model_validator(mode="after")
    def _validate_support(self) -> ProposalConfig:
        if self.maximum_redshift <= self.minimum_redshift:
            raise ValueError(
                "proposal maximum_redshift must be greater than minimum_redshift"
            )
        return self


class CatalogConfig(BaseModel):
    """The two catalogs this run uses: the injection and the proposal.

    Each role names one persisted catalog under ``config/catalogs/defs``, whose
    stem is both the config filename and the ``outputs/catalogs/<name>.h5`` it
    produces. The run records the *name* only: everything about how the catalog
    was drawn -- components, seeds, mixing fractions, and the redshift density
    that follows from them -- is recorded in the file itself and read back at
    run time.
    """

    model_config = _STRICT

    injection: str
    proposal: str


class RunConfig(BaseModel):
    model_config = _STRICT

    seed: int = 42
    observation_time: float = 1.0
    fiducials: dict[str, float]
    # Complete parameter name -> prior distribution table. `sampled_params`
    # selects the NUTS latents; every other non-marginalized site is conditioned.
    priors: dict[str, PriorDistribution]
    # Unset (empty) -> default to every key in [priors] except a marginalized
    # amplitude parameter; resolved below.
    sampled_params: tuple[str, ...] = ()
    analysis: AnalysisConfig
    cosmology: CosmoConfig
    catalog: CatalogConfig
    sampler: SamplerConfig
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _resolve_sampled_params(self) -> RunConfig:
        if not self.fiducials:
            raise ValueError("config must define a non-empty [fiducials] table")
        if not self.priors:
            raise ValueError("config must define at least one [priors.<param>] table")

        priors = dict(self.priors)
        amplitude_parameter = self.analysis.amplitude_parameter
        if amplitude_parameter is not None:
            if amplitude_parameter in self.sampled_params:
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

        sampled = self.sampled_params or tuple(
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

        object.__setattr__(self, "sampled_params", sampled)
        return self

    @property
    def fixed_params(self) -> dict[str, float]:
        """Every fiducial not sampled: values supplied by effect handlers.

        Includes the marginalized amplitude parameter when present: it has no
        NUTS latent, but its fiducial value is still what the model pins it
        to. This is a plain property and is not serialized.
        """
        return {k: v for k, v in self.fiducials.items() if k not in self.sampled_params}

    @property
    def analysis_grid(self) -> AnalysisGrid:
        """The frequency band and redshift grid this run's inputs are built on.

        A plain property, deliberately not serialized: `save` writes
        only inputs needed to reconstruct the validated run configuration.
        """
        return AnalysisGrid(
            observation_time=self.observation_time,
            f_min=self.analysis.f_min,
            f_max=self.analysis.f_max,
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
        cli_overrides["seed"] = seed
    if outdir is not None or label is not None:
        output: dict[str, Any] = {}
        if outdir is not None:
            output["outdir"] = str(outdir)
        if label is not None:
            output["label"] = label
        cli_overrides["output"] = output

    merged = deep_merge(raw, deep_merge(overrides, cli_overrides))
    return RunConfig.model_validate(merged)
