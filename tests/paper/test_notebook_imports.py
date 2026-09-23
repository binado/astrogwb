"""Notebooks import names that survive library refactors.

A rebase onto main can delete a helper a notebook still names --
``CatalogProvenance`` was the first of those. Parsing the committed
``.py`` sources and resolving each ``astrogwb`` import is cheaper than
executing the notebooks, and it fails for the same reason the notebook
fails: the name is not on the module.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from repo import REPO_ROOT

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.py"))


def _astrogwb_imports(path: Path) -> list[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("astrogwb"):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            names.append((node.module, alias.name))
    return names


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[path.name for path in NOTEBOOKS])
def test_notebook_astrogwb_imports_exist(notebook: Path) -> None:
    imports = _astrogwb_imports(notebook)
    if not imports:
        pytest.skip(f"{notebook.name} imports nothing from astrogwb")
    missing = [
        f"{module_name}.{attr}"
        for module_name, attr in imports
        if not hasattr(importlib.import_module(module_name), attr)
    ]
    assert not missing, f"{notebook.name} imports missing names: {missing}"
