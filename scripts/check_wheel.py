"""Assert the published wheel is the lean core, run against a no-extras install.

Three different claims, checked separately because they fail for different
reasons:

1. The core import surface works from the wheel alone -- including the detector
   TOMLs, which are package data and have been dropped by a packaging change
   before.
2. The distributions only the application needs are not installed.
3. The application library ships but does not *run*, and its checkout-only CLI
   package is absent.

On (3), two things are easy to get wrong and both make the check vacuous:

* ``importlib.util.find_spec('astrogwb.paper')`` returns a spec. The subpackage
  directory is inside the wheel; it is its *dependencies* that are absent.
* ``import astrogwb.paper`` succeeds. Its ``__init__`` deliberately imports
  nothing so callers can configure JAX first, and that is worth keeping. So the
  probe has to be a module that really needs an extras-only dependency.

``pydantic`` is deliberately absent from the "must not be installed" list: it
arrives transitively through ``gwmock-noise``, a genuine core dependency. What
matters for it is (3) and the requirement partition -- that *astrogwb* declares
it only under an extra -- not whether some other package pulled it in.

Together with ruff's ``TID251`` ban on ``astrogwb.paper`` inside core, this
replaces what the two-package workspace split used to enforce structurally.
"""

from __future__ import annotations

import importlib
import importlib.util
from importlib.metadata import requires

#: Installed by no core dependency, directly or transitively.
MUST_BE_ABSENT = ("gwmock_pop", "xarray", "h5netcdf", "arviz", "pandas")

#: An application module that imports an extras-only dependency at module
#: scope. `catalogs` imports xarray, which the `io` extra provides.
PAPER_PROBE = "astrogwb.paper.catalogs"

#: Distributions the application needs, mapped to the extra that must provide
#: them. None of these may be an unconditional requirement of astrogwb.
BEHIND_EXTRAS = {
    "gwmock-pop": "simulation",
    "xarray": "io",
    "h5netcdf": "io",
    "pydantic": "paper",
    "arviz": "paper",
}


def check_core_imports() -> None:
    from astrogwb.catalog import Catalog
    from astrogwb.detector import load_detector, load_sensitivity

    load_detector("H1")
    load_sensitivity("H1")
    assert callable(Catalog.from_generator)


def check_optional_distributions_are_absent() -> None:
    for name in MUST_BE_ABSENT:
        assert importlib.util.find_spec(name) is None, (
            f"{name} is installed in a no-extras environment: a core "
            "dependency started pulling it in"
        )


def check_paper_ships_but_does_not_run() -> None:
    assert importlib.util.find_spec("astrogwb.paper") is not None, (
        "astrogwb.paper is missing from the wheel entirely"
    )
    assert importlib.util.find_spec("astrogwb.paper.cli") is None, (
        "the checkout-only astrogwb.paper.cli package leaked into the wheel"
    )
    try:
        importlib.import_module(PAPER_PROBE)
    except ImportError:
        return
    raise AssertionError(
        f"{PAPER_PROBE} imported in a no-extras environment; the application's "
        "dependencies leaked into the core requirement set"
    )


def check_requirement_partition() -> None:
    requirements = [requirement.lower() for requirement in requires("astrogwb") or []]
    unconditional = [r for r in requirements if "extra ==" not in r]

    for name, extra in BEHIND_EXTRAS.items():
        assert not any(r.startswith(name) for r in unconditional), (
            f"{name} is an unconditional core requirement; it belongs to the "
            f"{extra!r} extra"
        )
        assert any(
            r.startswith(name) and f"extra == '{extra}'" in r for r in requirements
        ), f"{name} is not reachable through the {extra!r} extra"


def main() -> None:
    check_core_imports()
    check_optional_distributions_are_absent()
    check_paper_ships_but_does_not_run()
    check_requirement_partition()
    print("wheel smoke test passed: core imports, astrogwb.paper stays inert")


if __name__ == "__main__":
    main()
