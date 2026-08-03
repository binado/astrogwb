"""Unit tests for ``scripts/workflow.py`` argv construction."""

from __future__ import annotations

from scripts.workflow import (
    build_mcmc_argv,
    coalesce_mcmc_extra,
    split_config_from_extra,
)


def test_split_config_from_extra_collects_repeated_config_flags() -> None:
    entries, other = split_config_from_extra(
        [
            "--config",
            "jax_platforms=cpu",
            "--force",
            "-C",
            "catalog={'id':'x'}",
            "foo=1",
            "--cores",
            "4",
        ]
    )
    assert entries == [
        "jax_platforms=cpu",
        "catalog={'id':'x'}",
        "foo=1",
    ]
    assert other == ["--force", "--cores", "4"]


def test_coalesce_injects_jax_platforms_for_cpu_profiles() -> None:
    assert coalesce_mcmc_extra(
        ['--config', "catalog={'id':'bns-n8192-df1'}"],
        "slurm-cpu",
    ) == [
        "--config",
        "jax_platforms=cpu",
        "catalog={'id':'bns-n8192-df1'}",
    ]
    assert coalesce_mcmc_extra(None, "local") == [
        "--config",
        "jax_platforms=cpu",
    ]


def test_coalesce_preserves_explicit_jax_platforms() -> None:
    assert coalesce_mcmc_extra(
        ["--config", "jax_platforms=cuda", "catalog={'id':'x'}"],
        "local",
    ) == ["--config", "jax_platforms=cuda", "catalog={'id':'x'}"]


def test_coalesce_skips_injection_for_gpu_profile() -> None:
    assert coalesce_mcmc_extra(
        ['--config', "catalog={'id':'x'}"],
        "slurm",
    ) == ["--config", "catalog={'id':'x'}"]
    assert coalesce_mcmc_extra(None, "slurm") == []


def test_build_mcmc_argv_merges_catalog_override_with_cpu_backend() -> None:
    argv = build_mcmc_argv(
        "configs/mcmc.batch.example.json",
        "slurm-cpu",
        dry_run=False,
        extra=['--config', "catalog={'id':'bns-n8192-df1'}"],
    )
    config_idx = argv.index("--config")
    config_args = []
    for item in argv[config_idx + 1 :]:
        if item.startswith("-"):
            break
        config_args.append(item)
    assert config_args == [
        "jax_platforms=cpu",
        "catalog={'id':'bns-n8192-df1'}",
    ]
    assert argv.count("--config") == 1
