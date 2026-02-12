# /// script
# dependencies = [
#   "numpy",
#   "h5py",
# ]
# ///
import argparse
import csv
import logging
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray


def gpc_to_mpc(x: ArrayLike) -> NDArray:
    return 1e3 * np.asarray(x)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


KEY_DICT = {
    "Mc": "chirp_mass",
    "Phicoal": "phase",
    "dL": "luminosity_distance",
    "dec": "dec",
    "eta": "symmetric_mass_ratio",
    "iota": "theta_jn",  # inclination angle
    "m1_source": "mass_1_source",
    "m2_source": "mass_2_source",
    "phi": "phi",
    "psi": "psi",
    "ra": "ra",
    "tGPS": "t_gps",
    "tcoal": "geocent_time",
    "theta": "theta",
    "z": "redshift",
}

KEY_DICT_PRECESSING_SPIN = {
    "chi1": "a_1",
    "chi1x": "spin_1x",
    "chi1y": "spin_1y",
    "chi1z": "spin_1z",
    "chi2": "a_2",
    "chi2x": "spin_2x",
    "chi2y": "spin_2y",
    "chi2z": "spin_2z",
    "thetaJN": "theta_jn",
    "tilt1": "tilt_1",
    "tilt2": "tilt_2",
}

KEY_DICT_ALIGNED_SPIN = {
    "chi1z": "chi_1",
    "chi2z": "chi_2",
}

KEY_DICT_TIDAL = {
    "Lambda1": "lambda_1",
    "Lambda2": "lambda_2",
}

KEY_DICT_BBH = {**KEY_DICT, **KEY_DICT_PRECESSING_SPIN}

KEY_DICT_BNS = {
    **KEY_DICT,
    **KEY_DICT_ALIGNED_SPIN,
    **KEY_DICT_TIDAL,
}

DEFAULT_KEY_LIST_BNS = [
    "mass_1_source",
    "mass_2_source",
    "luminosity_distance",
    "redshift",
    "ra",
    "dec",
    "geocent_time",
    "theta_jn",
    "phase",
    "psi",
    "chi_1",
    "chi_2",
    "lambda_1",
    "lambda_2",
]

DEFAULT_KEY_LIST_BBH = [
    "redshift",
    "mass_1_source",
    "mass_2_source",
    "luminosity_distance",
    "ra",
    "dec",
    "geocent_time",
    "theta_jn",
    "phase",
    "psi",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
]

CALLBACKS = {"luminosity_distance": gpc_to_mpc}


def convert_file(
    input_filepath: Path,
    output_filepath: Path,
    source_type: Literal["BNS", "BBH"],
    *args: str,
) -> None:
    """Convert an injection file remapping dataset keys to the desired outputs."""
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    if source_type == "BNS":
        default_key_dict = KEY_DICT_BNS
    elif source_type == "BBH":
        default_key_dict = KEY_DICT_BBH
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    reversed_key_dict = {new: old for old, new in default_key_dict.items()}

    output_keys = list(args) if args else list(reversed_key_dict.keys())
    missing = [key for key in output_keys if key not in reversed_key_dict]
    if missing:
        raise KeyError(f"Unknown output key(s): {', '.join(missing)}")

    extracted = {}
    with h5py.File(input_path, "r") as data:
        for new_key in output_keys:
            old_key = reversed_key_dict[new_key]
            if old_key not in data:
                raise KeyError(
                    f"Dataset '{old_key}' not found in input file '{input_path}'."
                )
            ds = data[old_key]
            if not isinstance(ds, h5py.Dataset):
                raise TypeError(f"Dataset '{old_key}' is not a valid dataset.")

            logger.info("Extracting dataset %s -> %s", old_key, new_key)
            new_key_arr = ds[()]
            if new_key in CALLBACKS:
                new_key_arr = CALLBACKS[new_key](new_key_arr)
            extracted[new_key] = new_key_arr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    logger.info("Writing output to %s", output_path)
    if suffix in {".hdf5", ".h5"}:
        with h5py.File(output_path, "w") as output_file:
            for key in output_keys:
                output_file.create_dataset(key, data=extracted[key])
    elif suffix == ".csv":
        columns = []
        for key in output_keys:
            values = np.asarray(extracted[key])
            columns.append(values)

        rows = zip(*columns, strict=True) if columns else []
        with output_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=" ")
            writer.writerow(output_keys)
            writer.writerows(rows)
    else:
        raise ValueError(
            f"""Unsupported output file extension '{output_path.suffix}'.
            Use .hdf5, .h5, or .csv."""
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Convert COBA injection HDF5 file to bilby naming conventions")
    )
    parser.add_argument(
        "input_filepath", type=Path, help="Path to the source HDF5 file"
    )
    parser.add_argument(
        "output_filepath",
        type=Path,
        help="Path to the destination file (.hdf5/.h5 or .csv)",
    )
    parser.add_argument(
        "--source-type",
        choices=["BNS", "BBH"],
        default="BNS",
        help="Type of source (BNS or BBH)",
    )
    parser.add_argument(
        "keys",
        nargs="*",
        help="Optional list of output dataset keys to include (defaults to all)",
    )

    args = parser.parse_args()
    merged_keys = list(args.keys)
    if args.source_type == "BBH":
        default_key_list = DEFAULT_KEY_LIST_BBH
    elif args.source_type == "BNS":
        default_key_list = DEFAULT_KEY_LIST_BNS
    else:
        raise ValueError(f"Unsupported source type '{args.source_type}'")

    merged_keys += [key for key in default_key_list if key not in merged_keys]
    convert_file(
        args.input_filepath, args.output_filepath, args.source_type, *merged_keys
    )


if __name__ == "__main__":
    main()
