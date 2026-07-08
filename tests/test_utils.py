from __future__ import annotations

import pytest

from astrogwb.utils import repo_root


def test_repo_root_from_repo_cwd() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src/astrogwb").is_dir()


def test_repo_root_from_notebooks_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    notebooks = repo_root() / "notebooks"
    monkeypatch.chdir(notebooks)
    assert repo_root() == notebooks.parent


def test_repo_root_from_explicit_start() -> None:
    notebooks = repo_root() / "notebooks"
    assert repo_root(start=notebooks) == notebooks.parent
