"""Operator CLI for catalog / MCMC / paper Snakemake workflows.

Scientific config and profiles live in the ``astrogwb-paper`` project.
Snakemake subcommands default to ``--dry-run``; pass ``--submit`` to run for real.

Usage::

    uv run astrogwb-workflow catalog outputs/catalogs/bns-n16384-df1.h5 --submit
    uv run astrogwb-workflow mcmc H0-all-detectors --profile local
    uv run astrogwb-workflow mcmc H0-all-detectors --profile slurm --submit
    uv run astrogwb-workflow paper --submit

MCMC runs take committed experiment names. Each experiment is one TOML with
explicit ``[runs.<id>]`` overlays; ``assemble_config`` merges the selected run
with the shared base as part of the same DAG that samples chains and builds
local figures.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from astrogwb_paper.config.experiments import experiment, load_experiments
from astrogwb_paper.paths import paper_project_root

PROFILE_CHOICES = ("local", "slurm", "slurm-cpu")
# Profiles that must force JAX onto CPU. Emitted as a CLI ``--config`` entry
# (not only via the profile YAML ``config:`` key) so a caller's catalog
# ``--config`` override cannot wipe ``jax_platforms`` — Snakemake replaces the
# entire profile ``config:`` list when any CLI ``--config`` is present, and
# repeated ``--config`` flags do not merge (last flag wins).
CPU_JAX_PROFILES = frozenset({"local", "slurm-cpu"})
DEFAULT_LOCAL_CORES = 8


@lru_cache(maxsize=1)
def _paper_root() -> Path:
    return paper_project_root()


@lru_cache(maxsize=1)
def _catalog_smk() -> Path:
    return _paper_root() / "workflow/catalog.smk"


@lru_cache(maxsize=1)
def _mcmc_smk() -> Path:
    return _paper_root() / "workflow/mcmc.smk"


def profile_dir(name: str) -> Path:
    return _paper_root() / "profiles" / name


def split_config_from_extra(extra: list[str]) -> tuple[list[str], list[str]]:
    """Split ``extra`` into ``--config``/``-C`` KEY=VALUE entries and the rest.

    Snakemake's ``--config`` uses ``nargs='*'`` without ``action='append'``, so
    each ``--config`` occurrence replaces the previous one. Collecting every
    KEY=VALUE into a single group lets callers merge safely.
    """
    config_entries: list[str] = []
    other: list[str] = []
    i = 0
    while i < len(extra):
        if extra[i] in ("--config", "-C"):
            i += 1
            while i < len(extra) and not extra[i].startswith("-"):
                config_entries.append(extra[i])
                i += 1
            continue
        other.append(extra[i])
        i += 1
    return config_entries, other


def coalesce_mcmc_extra(extra: list[str] | None, profile: str) -> list[str]:
    """Merge ``--config`` groups and inject ``jax_platforms=cpu`` for CPU profiles."""
    config_entries, other = split_config_from_extra(list(extra or []))
    if profile in CPU_JAX_PROFILES:
        keys = {entry.split("=", 1)[0] for entry in config_entries if "=" in entry}
        if "jax_platforms" not in keys:
            config_entries.insert(0, "jax_platforms=cpu")
    if not config_entries:
        return other
    return ["--config", *config_entries, *other]


def _add_run_mode(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only (default).",
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="Execute for real (omit snakemake --dry-run).",
    )


def _is_dry_run(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "submit", False))


def _passthrough(args: argparse.Namespace) -> list[str]:
    return list(getattr(args, "extra", None) or [])


def build_catalog_argv(
    target: str,
    *,
    dry_run: bool = True,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--package",
        "astrogwb-paper",
        "--group",
        "workflow",
        "snakemake",
        "--snakefile",
        str(_catalog_smk()),
        "--cores",
        "1",
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.append(target)
    if extra:
        cmd.extend(extra)
    return cmd


def build_mcmc_argv(
    experiments: list[str] | None,
    profile: str,
    *,
    dry_run: bool = True,
    cores: int | None = None,
    chains_only: bool = False,
    extra: list[str] | None = None,
) -> list[str]:
    """Build the Snakemake argv for one or more committed experiments."""
    if profile not in PROFILE_CHOICES:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILE_CHOICES}")

    resolved_cores = cores
    if resolved_cores is None and profile == "local":
        resolved_cores = DEFAULT_LOCAL_CORES

    cmd = [
        "uv",
        "run",
        "--package",
        "astrogwb-paper",
        "--group",
        "workflow",
        "snakemake",
        "--snakefile",
        str(_mcmc_smk()),
        "--profile",
        str(profile_dir(profile)),
    ]
    if resolved_cores is not None:
        cmd.extend(["--cores", str(resolved_cores)])
    if dry_run:
        cmd.append("--dry-run")
    selected = list(experiments or [])
    targets = [
        (experiment(name).chains_target if chains_only else experiment(name).target)
        for name in selected
    ]
    cmd.extend(targets or ["experiments"])
    cmd.extend(coalesce_mcmc_extra(extra, profile))
    return cmd


def build_paper_argv(
    target: str | None = None,
    *,
    dry_run: bool = True,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--package",
        "astrogwb-paper",
        "--group",
        "workflow",
        "snakemake",
        "--snakefile",
        str(_mcmc_smk()),
        "--cores",
        "1",
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.append(target if target else "standalone_figures")
    if extra:
        cmd.extend(extra)
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Operator shortcuts for catalog / MCMC / paper Snakemake workflows. "
            "Snakemake subcommands default to --dry-run; pass --submit to run."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser(
        "catalog",
        help="Build a catalog target (population first if stale).",
    )
    catalog.add_argument(
        "target", help="Catalog output path, e.g. outputs/catalogs/….h5"
    )
    _add_run_mode(catalog)

    mcmc = sub.add_parser(
        "mcmc",
        help="Run or dry-run committed experiments with a profile.",
    )
    mcmc.add_argument(
        "experiments",
        nargs="*",
        choices=tuple(load_experiments()),
        help=("Committed experiment names. Omit to run every experiment."),
    )
    mcmc.add_argument(
        "--chains-only",
        action="store_true",
        help="Build chains without the experiment's local figure rules.",
    )
    mcmc.add_argument(
        "--profile",
        required=True,
        choices=PROFILE_CHOICES,
        help="Committed Snakemake profile under profiles/.",
    )
    mcmc.add_argument(
        "--cores",
        type=int,
        default=None,
        help=(
            f"Snakemake --cores (default: {DEFAULT_LOCAL_CORES} for --profile local; "
            "omit for SLURM profiles unless set)."
        ),
    )
    _add_run_mode(mcmc)

    paper = sub.add_parser(
        "paper",
        help="Build standalone paper figures from the unified workflow.",
    )
    paper.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional figure path; default standalone_figures.",
    )
    _add_run_mode(paper)

    # parse_known_args so callers can forward snakemake / generator flags after --
    # without argparse.REMAINDER swallowing required options like --profile.
    args, extras = parser.parse_known_args(argv)
    if extras and extras[0] == "--":
        extras = extras[1:]
    if (
        args.command == "paper"
        and args.target is not None
        and args.target.startswith("-")
    ):
        # After ``--`` argparse stops option parsing, so a forwarded flag such
        # as ``--config KEY=VALUE`` lands in the optional positional target.
        # Reclassify it as an extra so it reaches Snakemake's CLI.
        extras.insert(0, args.target)
        args.target = None
    args.extra = extras
    return args


def _require_path(path: Path, *, kind: str) -> None:
    if not path.exists():
        raise SystemExit(f"error: {kind} not found: {path}")


def _validate(args: argparse.Namespace) -> None:
    if args.command == "catalog":
        _require_path(_catalog_smk(), kind="snakefile")
        return
    if args.command == "mcmc":
        _require_path(_mcmc_smk(), kind="snakefile")
        _require_path(profile_dir(args.profile), kind="profile")
        return
    if args.command == "paper":
        _require_path(_mcmc_smk(), kind="snakefile")
        return


def build_command(args: argparse.Namespace) -> list[str]:
    extra = _passthrough(args)
    if args.command == "catalog":
        return build_catalog_argv(args.target, dry_run=_is_dry_run(args), extra=extra)
    if args.command == "mcmc":
        return build_mcmc_argv(
            args.experiments,
            args.profile,
            dry_run=_is_dry_run(args),
            cores=args.cores,
            chains_only=args.chains_only,
            extra=extra,
        )
    if args.command == "paper":
        return build_paper_argv(args.target, dry_run=_is_dry_run(args), extra=extra)
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate(args)
    cmd = build_command(args)
    print("+", shlex.join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=_paper_root(), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
