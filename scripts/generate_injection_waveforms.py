import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Annotated, Iterable, Iterator

import h5py
import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
from utils import get_config_filepath, get_git_revision

from asgwb.io import load_injection_file
from asgwb.waveform import (
    FrequencyGrid,
    SourceType,
    WaveformGenerator,
    WaveformPolarizations,
)
from asgwb.waveform.io import dump_waveform_npz, load_waveform_npz

logger = logging.getLogger(__name__)


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
    minimum_frequency: float = 10.0
    maximum_frequency: float | None = None
    duration: float = 8.0
    source_type: SourceType = "BNS"
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
) -> WaveformPolarizations:
    """Generate the plus and cross polarizations for a given set of injection parameters."""
    return waveform_generator.frequency_domain_polarizations(injection_parameters)


def get_waveform_generator(
    waveform_approximant: str,
    reference_frequency: float,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None,
    duration: float,
    source_type: SourceType,
) -> WaveformGenerator:
    grid = FrequencyGrid(
        duration=duration,
        sampling_frequency=sampling_frequency,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    return WaveformGenerator(
        approximant=waveform_approximant,
        grid=grid,
        source_type=source_type,
    )


def generate_injection_waveforms_for_chunk(
    chunk: pd.DataFrame,
    tmpdir: Path,
    waveform_approximant: str,
    reference_frequency: float,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None,
    duration: float,
    source_type: SourceType,
    preserve_tempfiles: bool = False,
) -> None:
    """Generate waveforms for a chunk of injections, saving each as a .npz file.

    Each worker reconstructs its own WaveformGenerator to avoid pickling issues
    with LAL/bilby objects.
    """
    start = chunk.index[0]
    end = chunk.index[-1]
    logger.info("Processing injections %d-%d", start, end)
    waveform_generator = get_waveform_generator(
        waveform_approximant,
        reference_frequency,
        sampling_frequency,
        minimum_frequency,
        maximum_frequency,
        duration,
        source_type,
    )
    columns = chunk.columns.tolist()
    chunk_npz_files: list[Path] = []
    chunk_completed = False
    try:
        for row in chunk.itertuples():
            i = row.Index
            injection_parameters = {
                column: float(value)
                for column, value in zip(columns, row[1:], strict=True)
            }
            polarizations = generate_injection_waveform(
                injection_parameters, waveform_generator
            )
            npz_path = tmpdir / f"{i}.npz"
            chunk_npz_files.append(npz_path)
            dump_waveform_npz(
                path=npz_path,
                polarizations=polarizations,
                parameters=injection_parameters,
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

    first_pol, first_parameters = load_waveform_npz(npz_files[0])
    freq_bins = first_pol.hp.shape[0]
    param_keys = list(first_parameters.keys())

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
            polarizations, parameters = load_waveform_npz(npz_path, grid=first_pol.grid)
            if polarizations.hp.shape[0] != freq_bins:
                raise ValueError(
                    f"Frequency bins mismatch in {npz_path}: "
                    f"expected {freq_bins}, got {polarizations.hp.shape[0]}"
                )

            file_param_keys = list(parameters.keys())
            if file_param_keys != param_keys:
                raise ValueError(
                    f"Parameter keys mismatch in {npz_path}: "
                    f"expected {param_keys}, got {file_param_keys}"
                )

            ds_plus[i] = polarizations.hp
            ds_cross[i] = polarizations.hc
            for key in param_keys:
                ds_params[key][i] = parameters[key]


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
    source_type: SourceType,
    minimum_frequency: float = 10.0,
    maximum_frequency: float | None = None,
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
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
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
        minimum_frequency=settings.minimum_frequency,
        maximum_frequency=settings.maximum_frequency,
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
