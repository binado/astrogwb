"""Compute waveform power directly from injections without intermediate waveform I/O.

The script loads injections in chunks, has each worker generate waveforms for its
assigned chunk, accumulates ``sum_abs_sq[f] += |h_plus|^2 + |h_cross|^2``, and
writes the reduced spectrum to a compact HDF5 file.
"""

from __future__ import annotations

import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Iterable, Iterator, NamedTuple

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

try:
    from utils import get_config_filepath, get_git_revision
except (
    ModuleNotFoundError
):  # pragma: no cover - import path differs in test/module usage
    from scripts.utils import get_config_filepath, get_git_revision

from asgwb.io import load_injection_file
from asgwb.waveform import SourceType, WaveformGenerator
from asgwb.waveform.grid import FrequencyGrid

logger = logging.getLogger(__name__)

_WORKER_WAVEFORM_GENERATOR: WaveformGenerator | None = None


class SpectralDensityMetadata(BaseModel):
    duration: float
    sampling_frequency: float
    reference_frequency: float
    minimum_frequency: float
    maximum_frequency: float | None
    n_events_processed: int
    n_events_requested: int
    n_chunks: int
    args: str | None = None
    git_revision: str | None = None

    def to_frequency_grid(self) -> FrequencyGrid:
        return FrequencyGrid(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            reference_frequency=self.reference_frequency,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
        )


class ComputeSpectralDensitySettings(BaseSettings):
    """Settings for streaming waveform-power computation."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )

    output_file: Path

    injection_file: Path
    waveform_approximant: str = "IMRPhenomPV2_NRTidalv2"
    reference_frequency: float = 50.0
    sampling_frequency: float = 2048.0
    minimum_frequency: float = 10.0
    maximum_frequency: float | None = None
    duration: float = 8.0
    source_type: SourceType = "BNS"
    chunksize: Annotated[int, Field(gt=0)] = 1000
    nworkers: Annotated[int, Field(gt=0)] = 1
    offset: Annotated[int, Field(ge=0)] = 0
    batch: Annotated[int | None, Field(gt=0)] = None

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


class ChunkPartialSum(NamedTuple):
    partial_sum: npt.NDArray[np.float64]
    processed: int
    chunk_start: int
    chunk_end: int


class WaveformGeneratorConfig(NamedTuple):
    waveform_approximant: str
    reference_frequency: float
    sampling_frequency: float
    minimum_frequency: float
    maximum_frequency: float | None
    duration: float
    source_type: SourceType


def _init_worker(config: WaveformGeneratorConfig) -> None:
    global _WORKER_WAVEFORM_GENERATOR
    _WORKER_WAVEFORM_GENERATOR = WaveformGenerator.from_sampling(
        waveform_approximant=config.waveform_approximant,
        reference_frequency=config.reference_frequency,
        sampling_frequency=config.sampling_frequency,
        minimum_frequency=config.minimum_frequency,
        maximum_frequency=config.maximum_frequency,
        duration=config.duration,
        source_type=config.source_type,
    )


def _compute_partial_sum_for_chunk_with_generator(
    chunk: pd.DataFrame,
    waveform_generator: WaveformGenerator,
) -> ChunkPartialSum | None:
    """Compute a per-frequency partial sum for one injection chunk."""
    if chunk.empty:
        return None

    chunk_start = int(chunk.index[0])
    chunk_end = int(chunk.index[-1])
    logger.info("Processing injections %d-%d", chunk_start, chunk_end)

    columns = chunk.columns.tolist()
    partial_sum: npt.NDArray[np.float64] | None = None
    processed = 0
    for row in chunk.itertuples():
        injection_parameters = {
            column: float(value) for column, value in zip(columns, row[1:], strict=True)
        }
        polarizations = waveform_generator.frequency_domain_polarizations(
            injection_parameters
        )
        contribution = polarizations.squared_sum()
        if partial_sum is None:
            partial_sum = contribution.copy()
        else:
            if contribution.shape != partial_sum.shape:
                raise ValueError(
                    f"Frequency bins mismatch in chunk {chunk_start}-{chunk_end}: "
                    f"{contribution.shape} vs {partial_sum.shape}"
                )
            partial_sum += contribution
        processed += 1

    assert partial_sum is not None
    return ChunkPartialSum(
        partial_sum=partial_sum,
        processed=processed,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
    )


def _compute_partial_sum_for_chunk_in_worker(
    chunk: pd.DataFrame,
) -> ChunkPartialSum | None:
    if _WORKER_WAVEFORM_GENERATOR is None:
        raise RuntimeError("Waveform generator has not been initialized in worker")
    return _compute_partial_sum_for_chunk_with_generator(
        chunk=chunk,
        waveform_generator=_WORKER_WAVEFORM_GENERATOR,
    )


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


def write_power_output(
    output_file: Path,
    frequency_axis: npt.NDArray[np.float64],
    sum_abs_sq: npt.NDArray[np.float64],
    metadata: SpectralDensityMetadata,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as out:
        out.create_dataset(
            "frequency",
            data=frequency_axis,
            dtype=np.float64,
            compression="gzip",
        )
        out.create_dataset(
            "sum_abs_sq",
            data=sum_abs_sq,
            dtype=np.float64,
            compression="gzip",
        )
        out.attrs["definition"] = (
            "sum_abs_sq[f] = sum_events (|plus[event,f]|^2 + |cross[event,f]|^2)"
        )
        out.attrs["metadata_json"] = metadata.model_dump_json()
        out.attrs["n_events_processed"] = metadata.n_events_processed
        out.attrs["n_chunks"] = metadata.n_chunks


def _accumulate_chunk_result(
    sum_abs_sq: npt.NDArray[np.float64],
    chunk_result: ChunkPartialSum | None,
) -> tuple[int, int]:
    if chunk_result is None:
        return 0, 0

    if chunk_result.partial_sum.shape != sum_abs_sq.shape:
        raise ValueError(
            f"Frequency bins mismatch in chunk {chunk_result.chunk_start}-{chunk_result.chunk_end}: "
            f"{chunk_result.partial_sum.shape} vs {sum_abs_sq.shape}"
        )

    logger.info(
        "Processed %d events in chunk %d-%d",
        chunk_result.processed,
        chunk_result.chunk_start,
        chunk_result.chunk_end,
    )
    sum_abs_sq += chunk_result.partial_sum
    return chunk_result.processed, 1


def compute_spectral_density_with_injections(
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
    metadata: dict[str, str] | None = None,
) -> None:
    grid = FrequencyGrid(
        duration=duration,
        sampling_frequency=sampling_frequency,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    frequency_axis = grid.frequencies.astype(np.float64, copy=False)
    sum_abs_sq = np.zeros(frequency_axis.shape[0], dtype=np.float64)

    read_kwargs: dict[str, int | range] = {}
    if offset > 0:
        read_kwargs["skiprows"] = range(1, offset + 1)
    if batch is not None:
        read_kwargs["nrows"] = batch

    reader = load_injection_file(
        path=injection_file,
        iterator=True,
        chunksize=chunksize,
        **read_kwargs,
    )
    indexed_reader = reindex_chunks(reader, start_index=offset)

    generator_config = WaveformGeneratorConfig(
        waveform_approximant=waveform_approximant,
        reference_frequency=reference_frequency,
        sampling_frequency=sampling_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        duration=duration,
        source_type=source_type,
    )

    n_events_processed = 0
    n_chunks = 0
    if nworkers == 1:
        waveform_generator = WaveformGenerator.from_sampling(
            approximant=waveform_approximant,
            duration=duration,
            sampling_frequency=sampling_frequency,
            reference_frequency=reference_frequency,
            source_type=source_type,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        for chunk in indexed_reader:
            processed, chunk_count = _accumulate_chunk_result(
                sum_abs_sq=sum_abs_sq,
                chunk_result=_compute_partial_sum_for_chunk_with_generator(
                    chunk=chunk,
                    waveform_generator=waveform_generator,
                ),
            )
            n_events_processed += processed
            n_chunks += chunk_count
    else:
        with Pool(
            processes=nworkers,
            initializer=_init_worker,
            initargs=(generator_config,),
        ) as pool:
            for chunk_result in pool.imap_unordered(
                _compute_partial_sum_for_chunk_in_worker,
                indexed_reader,
                chunksize=1,
            ):
                processed, chunk_count = _accumulate_chunk_result(
                    sum_abs_sq=sum_abs_sq,
                    chunk_result=chunk_result,
                )
                n_events_processed += processed
                n_chunks += chunk_count

    n_events_requested = batch if batch is not None else -1
    run_metadata = SpectralDensityMetadata(
        duration=duration,
        sampling_frequency=sampling_frequency,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        n_events_processed=n_events_processed,
        n_events_requested=n_events_requested,
        n_chunks=n_chunks,
        args=metadata.get("args") if metadata else None,
        git_revision=metadata.get("git_revision") if metadata else None,
    )
    write_power_output(
        output_file=output_file,
        frequency_axis=frequency_axis,
        sum_abs_sq=sum_abs_sq,
        metadata=run_metadata,
    )
    logger.info("Wrote reduced spectrum to %s", output_file)


def run(settings: ComputeSpectralDensitySettings) -> None:
    logger.info("Running with settings:")
    for k, v in settings.model_dump().items():
        logger.info("\t%s = %s", k, v)

    metadata: dict[str, str] = {"args": json.dumps(settings.model_dump(), default=str)}
    git_rev = get_git_revision()
    if git_rev is not None:
        metadata["git_revision"] = git_rev

    compute_spectral_density_with_injections(
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
        metadata=metadata,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run(settings=ComputeSpectralDensitySettings())


if __name__ == "__main__":
    main()
