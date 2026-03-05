from typing import Sequence

from astropy.cosmology import FlatLambdaCDM, Planck18

Cosmology = FlatLambdaCDM


def get_any_key(d: dict[str, float], keys: Sequence[str]) -> float:
    for key in keys:
        if key in d:
            return d.pop(key)

    raise KeyError(f"No key in {keys} found in {d}")


def get_cosmology(parameters: dict[str, float]) -> Cosmology:
    """
    Get astropy cosmology instance from a dictionary of parameters.
    Missing parameters are filled in with defaults from Planck18.
    """
    defaults = Planck18.parameters
    parameters.update(defaults)
    return Cosmology(**parameters)
