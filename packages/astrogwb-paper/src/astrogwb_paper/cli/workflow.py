"""Operator CLI for catalog / MCMC / paper Snakemake workflows.

Scientific config and profiles live in the ``astrogwb-paper`` project.
Snakemake subcommands default to ``--dry-run``; pass ``--submit`` to run for real.

Usage::

    uv run astrogwb-workflow gen-configs
    uv run astrogwb-workflow catalog out/catalogs/bns-n16384-df1.h5 --submit
    uv run astrogwb-workflow mcmc batch.json --profile local
    uv run astrogwb-workflow mcmc batch.json --profile slurm --submit
    uv run astrogwb-workflow paper --submit
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from astrogwb_paper.paths import paper_project_root, workspace_root

WORKSPACE_ROOT = workspace_root()
PAPER_ROOT = paper_project_root()
CATALOG_SMK = PAPER_ROOT / "workflow/catalog.smk"
MCMC_SMK = PAPER_ROOT / "workflow/mcmc.smk"
PAPER_SMK = PAPER_ROOT / "workflow/paper.smk"

PROFILE_CHOICES = ("local", "slurm", "slurm-cpu")
# Profiles that must force JAX onto CPU. Emitted as a CLI ``--config`` entry
# (not only via the profile YAML ``config:`` key) so a caller's catalog
# ``--config`` override cannot wipe ``jax_platforms`` — Snakemake replaces the
# entire profile ``config:`` list when any CLI ``--config`` is present, and
# repeated ``--config`` flags do not merge (last flag wins).
CPU_JAX_PROFILES = frozenset({"local", "slurm-cpu"})
DEFAULT_LOCAL_CORES = 8


def profile_dir(name: str) -> Path:
    return PAPER_ROOT / "profiles" / name


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


def build_gen_configs_argv(extra: list[str] | None = None) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--package",
        "astrogwb-paper",
        "astrogwb-generate-mcmc-configs",
        "--write-manifests",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


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
        str(CATALOG_SMK),
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
    configfile: str | Path,
    profile: str,
    *,
    dry_run: bool = True,
    cores: int | None = None,
    extra: list[str] | None = None,
) -> list[str]:
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
        str(MCMC_SMK),
        "--profile",
        str(profile_dir(profile)),
        "--configfile",
        str(configfile),
    ]
    if resolved_cores is not None:
        cmd.extend(["--cores", str(resolved_cores)])
    if dry_run:
        cmd.append("--dry-run")
    cmd.append("mcmc")
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
        str(PAPER_SMK),
        "--cores",
        "1",
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.append(target if target else "paper_figures")
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

    sub.add_parser(
        "gen-configs",
        help="Generate sweep JSON configs and gitignored batch manifests.",
    )

    catalog = sub.add_parser(
        "catalog",
        help="Build a catalog target (population first if stale).",
    )
    catalog.add_argument("target", help="Catalog output path, e.g. out/catalogs/….h5")
    _add_run_mode(catalog)

    mcmc = sub.add_parser(
        "mcmc",
        help="Run or dry-run an MCMC batch with a committed profile.",
    )
    mcmc.add_argument("configfile", type=Path, help="Batch manifest JSON.")
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
        help="Build paper figures (default target: paper_figures).",
    )
    paper.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional figure path; default paper_figures.",
    )
    _add_run_mode(paper)

    # parse_known_args so callers can forward snakemake / generator flags after --
    # without argparse.REMAINDER swallowing required options like --profile.
    args, extras = parser.parse_known_args(argv)
    if extras and extras[0] == "--":
        extras = extras[1:]
    args.extra = extras
    return args


def _require_path(path: Path, *, kind: str) -> None:
    if not path.exists():
        raise SystemExit(f"error: {kind} not found: {path}")


def _validate(args: argparse.Namespace) -> None:
    if args.command == "gen-configs":
        return
    if args.command == "catalog":
        _require_path(CATALOG_SMK, kind="snakefile")
        return
    if args.command == "mcmc":
        _require_path(MCMC_SMK, kind="snakefile")
        config = args.configfile
        if not config.is_absolute():
            config = WORKSPACE_ROOT / config
        _require_path(config, kind="configfile")
        _require_path(profile_dir(args.profile), kind="profile")
        return
    if args.command == "paper":
        _require_path(PAPER_SMK, kind="snakefile")
        return


def build_command(args: argparse.Namespace) -> list[str]:
    extra = _passthrough(args)
    if args.command == "gen-configs":
        return build_gen_configs_argv(extra)
    if args.command == "catalog":
        return build_catalog_argv(args.target, dry_run=_is_dry_run(args), extra=extra)
    if args.command == "mcmc":
        return build_mcmc_argv(
            args.configfile,
            args.profile,
            dry_run=_is_dry_run(args),
            cores=args.cores,
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
    completed = subprocess.run(cmd, cwd=WORKSPACE_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
