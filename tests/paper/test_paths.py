from __future__ import annotations

import pytest

from astrogwb.paper.paths import paper_project_root, workspace_root


def test_workspace_root_from_repo_cwd() -> None:
    root = workspace_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src/astrogwb").is_dir()
    assert (root / "src/astrogwb/paper").is_dir()


def test_workspace_root_from_notebooks_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    notebooks = paper_project_root() / "notebooks"
    monkeypatch.chdir(notebooks)
    assert workspace_root() == notebooks.parents[2]


def test_workspace_root_from_explicit_start() -> None:
    notebooks = paper_project_root() / "notebooks"
    assert workspace_root(start=notebooks) == notebooks.parents[2]
