"""Locate and merge the shared TOML config layers in ``config/``.

The shared layers, in merge order:

0. ``config/defaults.toml`` -- shared scientific defaults.
1. ``config/waveforms.toml`` -- named waveform settings.
2. ``config/populations.toml`` -- named populations.
3. ``config/detectors.toml`` -- shared networks and detector overrides.

:func:`merge_config_layers` folds them with ``knf``; on the shell,
``knf src/astrogwb/detector/{geometry,sensitivity}.toml <layers>
--shallow 'priors.*' --interpolate --merge-key extends`` prints what the
layers resolve to.

After the merge, every ``"${a.b}"`` string is replaced by the merged value at
``a.b``. That is how one table reuses another: ``[catalog]`` names its
waveform and population by reference. A reference resolves against the *final*
merge, and a reference to a table also merges as that table: a layer that sets
a key under one overrides that field and keeps the rest. A table that is a base
plus additions is written ``extends = "${a.b}"`` (:data:`~astrogwb.config.MERGE_KEY`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from astrogwb.config import merge_layers
from astrogwb.detector.config import DETECTOR_DEFAULT_PATHS

#: Relative to the working directory, the repository root for every script and
#: notebook.
CONFIG_DIR = Path("config")

#: Shared scientific defaults, also consumed directly by notebooks.
DEFAULTS_PATH = CONFIG_DIR / "defaults.toml"
#: Named waveforms and populations that catalogs and targets refer to.
WAVEFORMS_PATH = CONFIG_DIR / "waveforms.toml"
POPULATIONS_PATH = CONFIG_DIR / "populations.toml"
DETECTORS_PATH = CONFIG_DIR / "detectors.toml"

#: Every shared layer, in merge order.
SHARED_LAYER_PATHS = (DEFAULTS_PATH, WAVEFORMS_PATH, POPULATIONS_PATH, DETECTORS_PATH)

#: Root of everything the scripts and notebooks write.
BASE_OUT_DIR = Path("outputs")

CATALOGS_ROOT = BASE_OUT_DIR / "catalogs"
#: Ensemble roots, ``<name>/<seed>-<index>.h5`` backgrounds, written by
#: ``Snakefile``.
ENSEMBLES_ROOT = BASE_OUT_DIR / "ensembles"
FIGURES_DIR = BASE_OUT_DIR / "figures"

#: The one exception to a deep merge, as a ``knf`` key-path glob: each
#: ``[priors.<param>]`` table replaces the inherited one wholesale, so a layer
#: that swaps a uniform prior for a normal one does not keep the uniform's
#: ``low`` / ``high`` beside ``loc`` / ``scale``.
PRIOR_SHALLOW = "priors.*"


def merge_config_layers(paths: Sequence[Path]) -> dict[str, Any]:
    """Fold config layer files into one raw mapping, in the order given.

    Packaged detector tables are the initial defaults. A deep merge, left to
    right -- arrays and scalars replace -- except at :data:`PRIOR_SHALLOW`,
    then every ``${...}`` reference and
    :data:`~astrogwb.config.MERGE_KEY` inheritance is resolved.

    Parameters
    ----------
    paths
        Configuration file paths, in merge order.
    """
    if not paths:
        raise ValueError("no config layers given")
    return merge_layers([*DETECTOR_DEFAULT_PATHS, *paths], shallow=PRIOR_SHALLOW)


def base_config_paths(root: Path | None = None) -> tuple[Path, ...]:
    """The shared scientific, variant and detector layers, in merge order.

    Paths are relative to ``root``, default the working directory.

    Parameters
    ----------
    root
        Root directory for configuration files.
    """
    paths = tuple((root or Path()) / path for path in SHARED_LAYER_PATHS)
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing shared config layer: {path}")
    return paths


def load_base(root: Path | None = None) -> dict[str, Any]:
    """Merge every shared layer into one mapping.

    Parameters
    ----------
    root
        Root directory for configuration files.
    """
    return merge_config_layers(base_config_paths(root))
