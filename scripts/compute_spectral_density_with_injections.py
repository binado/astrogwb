"""Compute waveform power directly from injections without intermediate waveform I/O.

This script supports two modes:

1) compute (default):
   - Load injections in chunks.
   - Each worker generates waveforms for its chunk and computes:
         sum_abs_sq[f] += |h_plus|^2 + |h_cross|^2
   - The parent process reduces all chunk partial sums and writes a compact HDF5 file.

2) merge:
   - Sum multiple partial output files (for example from SLURM array jobs).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from glob import glob
from pathlib import Path
from typing import Annotated, Iterable, Iterator, Literal

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field
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


class ComputeSpectralDensitySettings(BaseSettings):
    """Settings for streaming waveform-power computation and partial-file merging."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )

    mode: Literal["compute", "merge"] = "compute"
    output_file: Path

    # compute mode inputs
    injection_file: Path | None = None
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

    # merge mode inputs
    input_pattern: str | None = None
    no_sort: bool = False

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


def _natural_sort_key(path: str) -> list[int | str]:
    parts: list[int | str] = []
    for part in re.split(r"(\d+)", Path(path).stem):
        parts.append(int(part) if part.isdigit() else part)
    return parts


def _filter_macos_sidecars(paths: list[str]) -> list[Path]:
    filtered = [Path(p) for p in paths if not Path(p).name.startswith("._")]
    skipped = len(paths) - len(filtered)
    if skipped > 0:
        logger.warning("Skipping %d macOS sidecar files (._*)", skipped)
    return filtered


@dataclass(frozen=True, slots=True)
class ChunkPartialSum:
    partial_sum: npt.NDArray[np.float64]
    processed: int
    chunk_start: int
    chunk_end: int


def compute_partial_sum_for_chunk(
    chunk: pd.DataFrame,
    waveform_approximant: str,
    reference_frequency: float,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None,
    duration: float,
    source_type: SourceType,
) -> ChunkPartialSum | None:
    """Compute a per-frequency partial sum for one injection chunk."""
    if chunk.empty:
        return None

    chunk_start = int(chunk.index[0])
    chunk_end = int(chunk.index[-1])
    logger.info("Processing injections %d-%d", chunk_start, chunk_end)

    waveform_generator = WaveformGenerator.from_sampling(
        approximant=waveform_approximant,
        duration=duration,
        sampling_frequency=sampling_frequency,
        reference_frequency=reference_frequency,
        source_type=source_type,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

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


def expected_frequency_axis(
    duration: float,
    sampling_frequency: float,
    reference_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float | None,
) -> npt.NDArray[np.float64]:
    grid = FrequencyGrid(
        duration=duration,
        sampling_frequency=sampling_frequency,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    return grid.in_band_frequencies.astype(np.float64, copy=False)


def write_power_output(
    output_file: Path,
    sum_abs_sq: npt.NDArray[np.float64],
    frequency_axis: npt.NDArray[np.float64],
    n_events_processed: int,
    n_events_requested: int,
    n_chunks: int,
    metadata: dict[str, str] | None = None,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as out:
        out.create_dataset(
            "sum_abs_sq",
            data=sum_abs_sq,
            dtype=np.float64,
            compression="gzip",
        )
        out.create_dataset(
            "frequency",
            data=frequency_axis,
            dtype=np.float64,
            compression="gzip",
        )
        out["frequency"].attrs["units"] = "Hz"
        out.attrs["definition"] = (
            "sum_abs_sq[f] = sum_events (|plus[event,f]|^2 + |cross[event,f]|^2)"
        )
        out.attrs["n_events_processed"] = n_events_processed
        out.attrs["n_events_requested"] = n_events_requested
        out.attrs["n_chunks"] = n_chunks
        out.attrs["n_freq"] = int(frequency_axis.shape[0])
        if metadata:
            for k, v in metadata.items():
                out.attrs[k] = v


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
    frequency_axis = expected_frequency_axis(
        duration=duration,
        sampling_frequency=sampling_frequency,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
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

    process_chunk = partial(
        compute_partial_sum_for_chunk,
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
    if nworkers > 1:
        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            for chunk_result in executor.map(process_chunk, indexed_reader):
                if chunk_result is None:
                    continue
                if chunk_result.partial_sum.shape != sum_abs_sq.shape:
                    raise ValueError(
                        f"Frequency bins mismatch in chunk {chunk_result.chunk_start}-{chunk_result.chunk_end}: "
                        f"{chunk_result.partial_sum.shape} vs {sum_abs_sq.shape}"
                    )
                sum_abs_sq += chunk_result.partial_sum
                n_events_processed += chunk_result.processed
                n_chunks += 1
    else:
        for chunk in indexed_reader:
            chunk_result = process_chunk(chunk)
            if chunk_result is None:
                continue
            if chunk_result.partial_sum.shape != sum_abs_sq.shape:
                raise ValueError(
                    f"Frequency bins mismatch in chunk {chunk_result.chunk_start}-{chunk_result.chunk_end}: "
                    f"{chunk_result.partial_sum.shape} vs {sum_abs_sq.shape}"
                )
            sum_abs_sq += chunk_result.partial_sum
            n_events_processed += chunk_result.processed
            n_chunks += 1

    n_events_requested = batch if batch is not None else -1
    write_power_output(
        output_file=output_file,
        sum_abs_sq=sum_abs_sq,
        frequency_axis=frequency_axis,
        n_events_processed=n_events_processed,
        n_events_requested=n_events_requested,
        n_chunks=n_chunks,
        metadata=metadata,
    )
    logger.info("Wrote reduced spectrum to %s", output_file)


def _load_partial_file(
    path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    with h5py.File(path, "r") as hf:
        if "sum_abs_sq" not in hf or "frequency" not in hf:
            raise KeyError(
                f"Expected datasets '/sum_abs_sq' and '/frequency' in {path}"
            )
        sum_abs_sq = hf["sum_abs_sq"][:]
        frequency = hf["frequency"][:]
        n_events_processed = int(hf.attrs.get("n_events_processed", -1))
    if sum_abs_sq.ndim != 1 or frequency.ndim != 1:
        raise ValueError(f"Expected 1D datasets in {path}")
    if sum_abs_sq.shape != frequency.shape:
        raise ValueError(
            f"Shape mismatch in {path}: sum_abs_sq {sum_abs_sq.shape} vs frequency {frequency.shape}"
        )
    return (
        sum_abs_sq.astype(np.float64, copy=False),
        frequency.astype(np.float64, copy=False),
        n_events_processed,
    )


def merge_partial_sum_files(
    input_files: list[Path],
    output_file: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    if not input_files:
        logger.error("No input files provided")
        sys.exit(1)

    total_sum: npt.NDArray[np.float64] | None = None
    frequency_ref: npt.NDArray[np.float64] | None = None
    total_events_processed = 0

    for i, path in enumerate(input_files, start=1):
        part_sum, part_frequency, part_events = _load_partial_file(path)
        if total_sum is None:
            total_sum = part_sum.copy()
            frequency_ref = part_frequency
        else:
            assert frequency_ref is not None
            if part_sum.shape != total_sum.shape:
                raise ValueError(
                    f"Frequency bins mismatch in {path}: {part_sum.shape} vs {total_sum.shape}"
                )
            if not np.allclose(part_frequency, frequency_ref, rtol=0.0, atol=0.0):
                raise ValueError(f"Frequency axis mismatch in {path}")
            total_sum += part_sum
        if part_events >= 0:
            total_events_processed += part_events
        logger.info("Merged %d/%d: %s", i, len(input_files), path.name)

    assert total_sum is not None
    assert frequency_ref is not None
    write_power_output(
        output_file=output_file,
        sum_abs_sq=total_sum,
        frequency_axis=frequency_ref,
        n_events_processed=total_events_processed,
        n_events_requested=-1,
        n_chunks=len(input_files),
        metadata=metadata,
    )

    with h5py.File(output_file, "a") as out:
        out.attrs["n_input_files"] = len(input_files)
    logger.info("Wrote merged spectrum to %s", output_file)


def run(settings: ComputeSpectralDensitySettings) -> None:
    logger.info("Running with settings:")
    for k, v in settings.model_dump().items():
        logger.info("\t%s = %s", k, v)

    metadata: dict[str, str] = {"args": json.dumps(settings.model_dump(), default=str)}
    git_rev = get_git_revision()
    if git_rev is not None:
        metadata["git_revision"] = git_rev

    if settings.mode == "compute":
        if settings.injection_file is None:
            raise ValueError("injection_file is required when mode='compute'")
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
        return

    if settings.input_pattern is None:
        raise ValueError("input_pattern is required when mode='merge'")
    input_files = _filter_macos_sidecars(glob(settings.input_pattern))
    if not input_files:
        logger.error("No valid input files matched pattern: %s", settings.input_pattern)
        sys.exit(1)
    if not settings.no_sort:
        input_files.sort(key=lambda p: _natural_sort_key(str(p)))
    merge_partial_sum_files(
        input_files=input_files,
        output_file=settings.output_file,
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
