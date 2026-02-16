import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Annotated, Iterable, Iterator, Literal, TypedDict, cast

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters,
)
from bilby.gw.source import (
    gwsignal_binary_black_hole,
    lal_binary_black_hole,
    lal_binary_neutron_star,
)
from bilby.gw.waveform_generator import WaveformGenerator
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from utils import get_config_filepath, get_git_revision

from asgwb.io import load_injection_file

logger = logging.getLogger(__name__)


class Polarizations(TypedDict):
    plus: npt.NDArray[np.float64]
    cross: npt.NDArray[np.float64]


GWSIGNAL_WAVEFORM_APPROXIMANTS = ["SEOBNRv5HM", "SEOBNRv5PHM"]


class InjectionWaveformSettings(BaseSettings):
    """
    Given an input injection file (CSV or HDF5) containing binary parameters,
    generate the corresponding waveforms and save them to an output HDF5 file.

    Arguments may be provided from a config.toml file, environment variables,
    or command-line arguments.
    """

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )

    injection_file: Path
    output_file: Path
    waveform_approximant: str = "IMRPhenomPV2_NRTidalv2"
    reference_frequency: float = 50.0
    sampling_frequency: float = 2048.0
    duration: float = 8.0
    source_type: Literal["BBH", "BHNS", "BNS"] = "BNS"
    chunksize: int = 1000
    nworkers: int = 1
    offset: Annotated[int, Field(ge=0)] = 0
    batch: Annotated[int | None, Field(gt=0)] = None
    preserve_tempfiles: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        config_filepath = get_config_filepath(Path(__file__))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=config_filepath),
            file_secret_settings,
        )


def generate_injection_waveform(
    injection_parameters: dict[str, float],
    waveform_generator: WaveformGenerator,
) -> Polarizations:
    """Generate the plus and cross polarizations for a given set of injection parameters."""
    waveform = waveform_generator.frequency_domain_strain(injection_parameters)
    return cast(Polarizations, waveform)


def get_bilby_waveform_generator(
    waveform_approximant: str,
    reference_frequency: float,
    sampling_frequency: float,
    duration: float,
    source_type: Literal["BBH", "BHNS", "BNS"],
) -> WaveformGenerator:
    waveform_arguments = {
        "waveform_approximant": waveform_approximant,
        "reference_frequency": reference_frequency,
    }
    if source_type == "BBH":
        source_model = (
            gwsignal_binary_black_hole
            if waveform_approximant in GWSIGNAL_WAVEFORM_APPROXIMANTS
            else lal_binary_black_hole
        )
        parameter_conversion = convert_to_lal_binary_black_hole_parameters
    else:
        source_model = lal_binary_neutron_star
        parameter_conversion = convert_to_lal_binary_neutron_star_parameters
    return WaveformGenerator(
        parameters=None,
        frequency_domain_source_model=source_model,
        duration=duration,
        sampling_frequency=sampling_frequency,
        parameter_conversion=parameter_conversion,
        waveform_arguments=waveform_arguments,
    )


def generate_injection_waveforms_for_chunk(
    chunk: pd.DataFrame,
    tmpdir: Path,
    waveform_approximant: str,
    reference_frequency: float,
    sampling_frequency: float,
    duration: float,
    source_type: Literal["BBH", "BHNS", "BNS"],
    preserve_tempfiles: bool = False,
) -> None:
    """Generate waveforms for a chunk of injections, saving each as a .npz file.

    Each worker reconstructs its own WaveformGenerator to avoid pickling issues
    with LAL/bilby objects.
    """
    start = chunk.index[0]
    end = chunk.index[-1]
    logger.info("Processing injections %d-%d", start, end)
    waveform_generator = get_bilby_waveform_generator(
        waveform_approximant,
        reference_frequency,
        sampling_frequency,
        duration,
        source_type,
    )
    columns = chunk.columns.tolist()
    chunk_npz_files: list[Path] = []
    chunk_completed = False
    try:
        for row in chunk.itertuples():
            i = row.Index
            injection_parameters = dict(zip(columns, row[1:]))
            polarizations = generate_injection_waveform(
                injection_parameters, waveform_generator
            )
            npz_path = tmpdir / f"{i}.npz"
            chunk_npz_files.append(npz_path)
            np.savez(
                npz_path,
                plus=polarizations["plus"],
                cross=polarizations["cross"],
                **injection_parameters,
            )
        chunk_completed = True
    finally:
        if not chunk_completed and not preserve_tempfiles:
            for npz_path in chunk_npz_files:
                npz_path.unlink(missing_ok=True)


def consolidate_to_hdf5(
    tmpdir: Path,
    output_file: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Read all .npz files from tmpdir and write them into a single HDF5 file.

    Produces a columnar layout:
        polarizations/{plus,cross}  — shape (N, F), complex128, gzip
        parameters/{name}           — shape (N,), float64, gzip
    """
    npz_files = sorted(tmpdir.glob("*.npz"), key=lambda p: int(p.stem))
    if not npz_files:
        return

    n_injections = len(npz_files)

    # Peek at the first file to discover shapes and parameter keys.
    first = np.load(npz_files[0])
    freq_bins = first["plus"].shape[0]
    param_keys = [k for k in first.files if k not in ("plus", "cross")]
    first.close()

    with h5py.File(output_file, "w") as hf:
        pol_group = hf.create_group("polarizations")
        ds_plus = pol_group.create_dataset(
            "plus",
            shape=(n_injections, freq_bins),
            dtype=np.complex128,
            compression="gzip",
        )
        ds_cross = pol_group.create_dataset(
            "cross",
            shape=(n_injections, freq_bins),
            dtype=np.complex128,
            compression="gzip",
        )

        par_group = hf.create_group("parameters")
        ds_params = {
            key: par_group.create_dataset(
                key, shape=(n_injections,), dtype=np.float64, compression="gzip"
            )
            for key in param_keys
        }

        if metadata:
            for k, v in metadata.items():
                hf.attrs[k] = v

        for i, npz_path in enumerate(npz_files):
            data = np.load(npz_path)
            ds_plus[i] = data["plus"]
            ds_cross[i] = data["cross"]
            for key in param_keys:
                ds_params[key][i] = data[key].item()
            data.close()


def reindex_chunks(
    chunks: Iterable[pd.DataFrame], start_index: int
) -> Iterator[pd.DataFrame]:
    """Assign monotonically increasing global indices to each chunk."""
    next_index = start_index
    for chunk in chunks:
        stop_index = next_index + len(chunk)
        chunk.index = pd.RangeIndex(start=next_index, stop=stop_index)
        yield chunk
        next_index = stop_index


def generate_injection_waveforms(
    injection_file: Path,
    output_file: Path,
    waveform_approximant: str,
    reference_frequency: float,
    sampling_frequency: float,
    duration: float,
    source_type: Literal["BBH", "BHNS", "BNS"],
    chunksize: int = 1000,
    nworkers: int = 1,
    offset: int = 0,
    batch: int | None = None,
    preserve_tempfiles: bool = False,
    metadata: dict[str, str] | None = None,
) -> None:
    tmpdir = output_file.parent / f".tmp_{output_file.stem}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    read_kwargs: dict[str, int | range] = {}
    if offset > 0:
        # Keep the header row (line 0) and skip `offset` data rows after it.
        read_kwargs["skiprows"] = range(1, offset + 1)
    if batch is not None:
        read_kwargs["nrows"] = batch

    reader = load_injection_file(
        injection_file,
        iterator=True,
        chunksize=chunksize,
        **read_kwargs,
    )
    indexed_reader = reindex_chunks(reader, start_index=offset)
    process_chunk = partial(
        generate_injection_waveforms_for_chunk,
        tmpdir=tmpdir,
        waveform_approximant=waveform_approximant,
        reference_frequency=reference_frequency,
        sampling_frequency=sampling_frequency,
        duration=duration,
        source_type=source_type,
        preserve_tempfiles=preserve_tempfiles,
    )

    try:
        if nworkers > 1:
            with ProcessPoolExecutor(max_workers=nworkers) as executor:
                list(executor.map(process_chunk, indexed_reader))
        else:
            for chunk in indexed_reader:
                process_chunk(chunk)

        consolidate_to_hdf5(tmpdir, output_file, metadata=metadata)
    finally:
        if not preserve_tempfiles:
            shutil.rmtree(tmpdir, ignore_errors=True)


def run(settings: InjectionWaveformSettings) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    settings_as_dict = settings.model_dump()
    logger.info("Running with settings:")
    for k, v in settings_as_dict.items():
        logger.info("\t%s = %s", k, v)
    metadata: dict[str, str] = {"args": json.dumps(settings.model_dump(), default=str)}
    git_rev = get_git_revision()
    if git_rev is not None:
        metadata["git_revision"] = git_rev

    generate_injection_waveforms(
        injection_file=settings.injection_file,
        output_file=settings.output_file,
        waveform_approximant=settings.waveform_approximant,
        reference_frequency=settings.reference_frequency,
        sampling_frequency=settings.sampling_frequency,
        duration=settings.duration,
        source_type=settings.source_type,
        chunksize=settings.chunksize,
        nworkers=settings.nworkers,
        offset=settings.offset,
        batch=settings.batch,
        preserve_tempfiles=settings.preserve_tempfiles,
        metadata=metadata,
    )


def main() -> None:
    run(settings=InjectionWaveformSettings())


if __name__ == "__main__":
    main()
