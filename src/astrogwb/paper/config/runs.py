"""Locate and merge the shared TOML config layers in ``config/``.

The shared layers, in merge order:

0. ``config/defaults.toml`` -- shared scientific defaults.
1. ``config/waveforms.toml`` -- named waveform settings.
2. ``config/populations.toml`` -- named populations.
3. ``config/detectors.toml`` -- shared networks and detector overrides.

Every layer is TOML, so each one can say in a comment why it sets what it
sets. :func:`merge_config_layers` folds them with ``knf``, the
engine behind the ``knf`` CLI, so the shell and Python spell one merge rule:
``knf src/astrogwb/detector/{geometry,sensitivity}.toml <layers>
--shallow 'priors.*' --interpolate --merge-key extends`` prints what the
layers resolve to. Both packaged tables use the same ``[detectors.<name>]``
layout as the shared registry file.

After the merge, every ``"${a.b}"`` string is replaced by the merged value at
``a.b``. That is how one table reuses another: ``[catalog]`` names its
waveform and population by reference. A reference resolves against the *final*
merge, and a reference to a table also merges as that table: a layer that sets
a key under one overrides that field and keeps the rest. A table that is a base
plus additions is written ``extends = "${a.b}"`` (:data:`MERGE_KEY`).

Entrypoints that take layers on argv (see :func:`add_config_arguments`) merge
them in process with :func:`load_merged_config`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import knf

from astrogwb.paper.utils import require_toml

logger = logging.getLogger(__name__)

#: Relative to the working directory, which for every script and notebook is
#: the repository root. Library code names no absolute path and does not go
#: looking for a checkout: the caller's cwd is the answer.
CONFIG_DIR = Path("config")

#: Shared scientific defaults, also consumed directly by notebooks.
DEFAULTS_PATH = CONFIG_DIR / "defaults.toml"
#: Named waveforms and populations that catalogs and targets refer to.
WAVEFORMS_PATH = CONFIG_DIR / "waveforms.toml"
POPULATIONS_PATH = CONFIG_DIR / "populations.toml"
DETECTORS_PATH = CONFIG_DIR / "detectors.toml"

#: Every shared layer, in merge order.
SHARED_LAYER_PATHS = (DEFAULTS_PATH, WAVEFORMS_PATH, POPULATIONS_PATH, DETECTORS_PATH)

#: Packaged geometry and sensitivity use the same registry tables as overrides.
DETECTOR_DEFAULT_PATHS = tuple(
    Path(__file__).parents[2] / "detector" / filename
    for filename in ("geometry.toml", "sensitivity.toml")
)

#: Presentation settings, read by `astrogwb.paper.plotting`. Deliberately *not*
#: a config layer: nothing a simulation depends on reads it.
PLOTTING_PATH = CONFIG_DIR / "plotting.toml"

#: Root of everything the scripts and notebooks write, so the output roots below
#: are composed rather than re-typed. Relative to the working directory.
BASE_OUT_DIR = Path("outputs")

CATALOGS_ROOT = BASE_OUT_DIR / "catalogs"
SPECTRA_ROOT = BASE_OUT_DIR / "spectra"
FIGURES_DIR = BASE_OUT_DIR / "figures"

#: The one exception to a deep merge, as a ``knf`` key-path glob: each
#: ``[priors.<param>]`` table replaces the inherited one wholesale, so a layer
#: that swaps a uniform prior for a normal one does not keep the uniform's
#: ``low`` / ``high`` beside ``loc`` / ``scale``.
PRIOR_SHALLOW = "priors.*"

#: The inheritance key: ``extends = "${a.b}"`` in a table starts it from the
#: table ``a.b`` and lets its own fields win. It is removed from the result.
MERGE_KEY = "extends"


def merge_config_layers(paths: Sequence[Path]) -> dict[str, Any]:
    """Fold config layer files into one raw mapping, in the order given.

    Packaged detector tables are the initial defaults. A deep merge, left to
    right -- arrays and scalars replace -- except at
    :data:`PRIOR_SHALLOW`, followed by resolving every ``${...}`` reference
    against the merged result and every :data:`MERGE_KEY` inheritance.

    Order is the caller's responsibility and it is not recoverable from the
    result, so :func:`load_merged_config` logs it.
    """
    if not paths:
        raise ValueError("no config layers given")
    for path in paths:
        require_toml(path)
    return knf.load(
        [*DETECTOR_DEFAULT_PATHS, *paths],
        interpolate=True,
        shallow=PRIOR_SHALLOW,
        merge_key=MERGE_KEY,
    )


def base_config_paths(root: Path | None = None) -> tuple[Path, ...]:
    """The shared scientific, variant and detector layers, in merge order.

    Named explicitly: config/plotting.toml is presentation and is not a
    layer. Paths are relative to the caller's working directory.
    """
    paths = tuple((root or Path()) / path for path in SHARED_LAYER_PATHS)
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing shared config layer: {path}")
    return paths


def load_base(root: Path | None = None) -> dict[str, Any]:
    """Merge every shared layer into one mapping."""
    return merge_config_layers(base_config_paths(root))


def add_config_arguments(
    parser: argparse.ArgumentParser, *, help: str | None = None
) -> None:
    """Add the repeated ``--config`` layer flag shared by every entrypoint.

    Mirrors :func:`astrogwb.paper.runtime.add_runtime_arguments`: one
    definition, so all entrypoints spell the flag the same way. ``help``
    replaces the default text for an entrypoint whose layers differ.
    """
    parser.add_argument(
        "--config",
        dest="config",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help=help
        or (
            "One config layer file, in merge order; repeat once per layer "
            "(the four shared config/*.toml layers, then any override)."
        ),
    )


def load_merged_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the ``--config`` layers an entrypoint was handed, order preserved.

    Merge order is the caller's to get right and a wrong-but-valid order fails
    silently, so the resolved order is logged before the merge.
    """
    paths: list[Path] = list(args.config)
    logger.info(
        "Config layers (merge order): %s", " -> ".join(str(path) for path in paths)
    )
    return merge_config_layers(paths)
