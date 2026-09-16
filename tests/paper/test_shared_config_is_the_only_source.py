"""Nothing outside ``config/`` may keep its own copy of the shared values.

These are text-level checks, and deliberately so. The drift this change closed
was not two values disagreeing at runtime -- it was three independent *literal
declarations* of the same thing, which a value-level test cannot see because a
freshly introduced copy agrees on the day it is written and only diverges
later. ``notebooks/`` is also outside ``[tool.ty.src].include``, so for the
notebook this is the only automated guard there is.

What drifted, before ``config/priors.json`` owned it:

- ``notebooks/mcmc.py`` sampled ``local_merger_rate`` under
  ``Uniform(7.6, 250.0)`` while naming 770.0 as its fiducial -- a prior that
  excluded its own fiducial.
- ``notebooks/mcmc.py`` and ``scripts/importance_weights_grid.py`` both gave
  ``Omega_m`` a broad uniform where the committed prior is a Planck-tight
  normal.
"""

from __future__ import annotations

import ast

import pytest
from repo import REPO_ROOT

#: `scripts/importance_weights_grid.py` is exempt for `GRID_SCAN_RANGES`
#: alone: that table is an explicit, documented *scan* range for a diagnostic
#: figure, which is allowed to be wider than the prior it diagnoses. It is an
#: exception list, not a second prior declaration, so it is named here rather
#: than left to be recognised by eye.
SCAN_RANGE_EXEMPTION = "GRID_SCAN_RANGES"

CONSUMERS = (
    "notebooks/catalog_convergence.py",
    "notebooks/mcmc.py",
    "scripts/importance_weights_grid.py",
    "scripts/mcmc_cosmological_parameters.py",
    "scripts/mcmc_modified_propagation.py",
    "scripts/fiducial_spectrum.py",
)

#: `notebooks/catalog_convergence.py` declares no prior at all -- it is a
#: convergence study, not an inference -- so the distribution rule below does
#: not apply to it. Its one `dist` call is the Gaussian *likelihood* of the
#: measured spectrum, which is a model, not a copied prior. It is still held to
#: the table rule above: it draws its catalog at `config/fiducials.json`.
PRIOR_FREE_CONSUMERS = ("notebooks/catalog_convergence.py",)


def _assignments(tree: ast.AST) -> list[tuple[list[str], ast.AST]]:
    """Every assignment as (target names, value), covering annotated ones.

    `GRID_SCAN_RANGES: dict[...] = {...}` is an `AnnAssign`, not an `Assign`,
    and missing that is how the exemption below silently stopped applying.
    """
    found: list[tuple[list[str], ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            found.append(
                ([t.id for t in node.targets if isinstance(t, ast.Name)], node)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.append(([node.target.id], node))
    return found


def _is_handwritten_table(value: ast.AST | None) -> bool:
    """A dict literal written out by hand, rather than one built from a source.

    `{**config.fiducials, "importance_relative_ess": ...}` is fine -- it
    extends what the run config already carries. A dict of nothing but literal
    keys is the thing that drifts.
    """
    if not isinstance(value, ast.Dict):
        return False
    if any(key is None for key in value.keys):  # a ** unpacking
        return False
    return any(isinstance(key, ast.Constant) for key in value.keys)


@pytest.mark.parametrize("relative", CONSUMERS)
def test_no_consumer_declares_its_own_fiducials_or_priors(relative: str) -> None:
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))

    for names, node in _assignments(tree):
        if SCAN_RANGE_EXEMPTION in names:
            continue
        retired = {"fiducials", "hyperprior_dists", "GRID_PRIORS", "VAR_LABELS"}
        clashing = retired.intersection(names)
        if clashing and _is_handwritten_table(getattr(node, "value", None)):
            pytest.fail(
                f"{relative} declares {sorted(clashing)} as a hand-written table; "
                "read it from astrogwb.paper.config or astrogwb.paper.plotting"
            )


@pytest.mark.parametrize(
    "relative", [name for name in CONSUMERS if name not in PRIOR_FREE_CONSUMERS]
)
def test_no_consumer_constructs_a_prior_distribution(relative: str) -> None:
    """No `dist.Uniform(...)` / `dist.Normal(...)` outside the scan-range table.

    Constructing one is how every drifted copy started. `dist.Distribution` in
    a type annotation is not a construction and is not flagged.
    """
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))

    exempt: set[int] = set()
    for names, node in _assignments(tree):
        if SCAN_RANGE_EXEMPTION in names:
            exempt.update(id(child) for child in ast.walk(node))

    for node in ast.walk(tree):
        if id(node) in exempt or not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "dist"
            and func.attr != "Distribution"
        ):
            pytest.fail(
                f"{relative} constructs dist.{func.attr}(...) directly; priors "
                "come from astrogwb.paper.config.priors()"
            )
