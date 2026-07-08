from __future__ import annotations

import os
from pathlib import Path

from astrogwb.utils import repo_root


def test_repo_root_from_repo_cwd() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src/astrogwb").is_dir()


def test_repo_root_from_notebooks_cwd() -> None:
    notebooks = repo_root() / "notebooks"
    cwd = Path.cwd()
    os.chdir(notebooks)
    try:
        assert repo_root() == notebooks.parent
    finally:
        os.chdir(cwd)


def test_repo_root_from_explicit_start() -> None:
    notebooks = repo_root() / "notebooks"
    assert repo_root(start=notebooks) == notebooks.parent
