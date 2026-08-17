"""Pydantic models and I/O for headless MCMC run configs.

Importing this module requires only stdlib and pydantic: ``numpyro`` is
imported lazily inside :func:`materialize_prior` / :func:`prior_to_spec`, so
parsing and validating a config stays cheap and -- because the constructed
distributions hold plain Python floats and no JAX op is ever evaluated --
does not initialize the XLA backend. That last property is what
:func:`astrogwb_paper.runtime.configure_runtime` relies on to set host device
count / platform after config validation; it is guarded by a subprocess test
in ``tests/test_prior_native_types.py`` (re-running ``set_host_device_count``
after a backend init is a silent no-op, hence the subprocess).
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
    model_validator,
)

from astrogwb_paper.config.hashing import canonical_sha256
from astrogwb_paper.config.loading import deep_merge

_STRICT = ConfigDict(frozen=True, extra="forbid")

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
# mappings/dists, serialize back to the spec dict so `model_dump(mode="json")`,
# `save_config`, and `config_sha256` stay canonical. Numpyro types deliberately
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

    z_min: float
    z_max: float
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

    outdir: Path = Path("chains")
    label: str = ""


class RunConfig(BaseModel):
    model_config = _STRICT

    seed: int = 42
    observation_time: float = 1.0
    fiducials: dict[str, float]
    priors: dict[str, PriorDistribution]  # parameter name -> prior distribution
    # Unset (empty) -> default to the keys present in [priors]; resolved below.
    sampled_params: tuple[str, ...] = ()
    analysis: AnalysisConfig
    cosmology: CosmoConfig
    sampler: SamplerConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    # Derived in the validator (every fiducial not sampled).
    constants: dict[str, float] = Field(default_factory=dict)
    # Derived in the validator: the amplitude parameter's prior, held out
    # of `priors` so `set(priors) == set(sampled_params)` keeps holding. Also
    # accepted as input so configs written by save_config reload unchanged.
    amplitude_prior: PriorDistribution | None = None

    @model_validator(mode="after")
    def _resolve_sampled_and_constants(self) -> RunConfig:
        if not self.fiducials:
            raise ValueError("config must define a non-empty [fiducials] table")
        if not self.priors and self.amplitude_prior is None:
            raise ValueError("config must define at least one [priors.<param>] table")

        priors = dict(self.priors)
        amplitude_prior: Distribution | None = None
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
            if amplitude_parameter in priors:
                amplitude_prior = priors.pop(amplitude_parameter)
            elif self.amplitude_prior is not None:
                # Reloaded save_config output: the pop above already happened
                # at generation time, so the prior arrives in `amplitude_prior`.
                amplitude_prior = self.amplitude_prior
            else:
                raise ValueError(
                    f"analysis.amplitude_parameter {amplitude_parameter!r} needs "
                    "a [priors.*] table (or an `amplitude_prior` entry in a "
                    "config previously written by save_config)"
                )

        sampled = self.sampled_params or tuple(priors)
        missing_priors = [p for p in sampled if p not in priors]
        if missing_priors:
            raise ValueError(
                f"sampled_params without a [priors.*] table: {missing_priors}"
            )
        missing_fid = [p for p in sampled if p not in self.fiducials]
        if missing_fid:
            raise ValueError(f"sampled_params missing from [fiducials]: {missing_fid}")

        aligned_priors = {name: priors[name] for name in sampled}
        constants = {k: v for k, v in self.fiducials.items() if k not in sampled}

        object.__setattr__(self, "sampled_params", sampled)
        object.__setattr__(self, "priors", aligned_priors)
        object.__setattr__(self, "constants", constants)
        object.__setattr__(self, "amplitude_prior", amplitude_prior)
        return self

    @property
    def posterior_params(self) -> tuple[str, ...]:
        """Parameters present in the saved posterior group.

        A superset of `sampled_params`, which means strictly "parameters NUTS
        has a latent for". Under an amplitude-marginalized likelihood the two
        sets differ: the amplitude parameter is integrated out of the potential
        and has no latent, so it must stay out of `sampled_params` (it drives
        `init_to_value` and the `set(priors) == set(sampled_params)`
        invariant), yet post-processing reconstructs it into the posterior via
        `amplitude_reconstruction_model`. Use this for anything describing the
        saved chain -- plot `var_names`, run records, summaries.
        """
        amplitude_parameter = self.analysis.amplitude_parameter
        if amplitude_parameter is None:
            return self.sampled_params
        return (*self.sampled_params, amplitude_parameter)

    @property
    def outdir(self) -> Path:
        return self.output.outdir

    @property
    def label(self) -> str:
        return self.output.label


def config_sha256(config: RunConfig) -> str:
    """Digest inference settings, excluding output routing."""
    return canonical_sha256(config.model_dump(mode="json", exclude={"output"}))


def save_config(config: RunConfig, path: Path) -> None:
    """Write a validated run config as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n",
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
