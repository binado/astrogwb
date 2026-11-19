import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

with (ROOT / "pyproject.toml").open("rb") as project_file:
    project_metadata = tomllib.load(project_file)["project"]

project = project_metadata["name"]
release = project_metadata["version"]
copyright = "2026, astrogwb contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
}
html_title = f"{project} {release}"

exclude_patterns = ["_build"]

napoleon_use_ivar = True
