"""Read and write native frozen observation-equation file groups."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from lunarops.classes.time import Epoch, TimeScale
from lunarops.estimation.frozen_observation_equations import FrozenObservationEquations as _FrozenObservationEquations

from .archive import (
    atomic_text_writer,
    data_lines,
    decode_token,
    encode_token,
    format_float,
    open_text_reader,
    parse_float,
    parse_header,
    require_file_group_path,
    sha256_file,
)
from .matrix import read_matrix, write_matrix
from .parameters import read_parameter_names, write_parameter_names
from .structured_text import read_structured_text, write_structured_text


def _csr(design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_pointers = [0]
    columns: list[int] = []
    values: list[float] = []
    for row in np.asarray(design):
        nonzero = np.flatnonzero(row)
        columns.extend(int(value) for value in nonzero)
        values.extend(float(row[value]) for value in nonzero)
        row_pointers.append(len(columns))
    return (
        np.asarray(row_pointers, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(values, dtype=float),
    )


def _dense_from_csr(
    row_pointers: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    *,
    row_count: int,
    column_count: int,
) -> np.ndarray:
    pointers = np.asarray(row_pointers, dtype=np.int64).reshape(-1)
    indices = np.asarray(columns, dtype=np.int64).reshape(-1)
    data = np.asarray(values, dtype=float).reshape(-1)
    if pointers.size != row_count + 1 or pointers[0] != 0 or pointers[-1] != data.size:
        raise ValueError("Invalid CSR row pointers.")
    if indices.size != data.size or np.any(indices < 0) or np.any(indices >= column_count):
        raise ValueError("Invalid CSR column indices.")
    if np.any(np.diff(pointers) < 0):
        raise ValueError("CSR row pointers must be monotonic.")
    design = np.zeros((row_count, column_count), dtype=float)
    for row in range(row_count):
        start, stop = int(pointers[row]), int(pointers[row + 1])
        row_indices = indices[start:stop]
        if np.any(np.diff(row_indices) <= 0):
            raise ValueError("CSR columns must be strictly increasing within each row.")
        design[row, indices[start:stop]] = data[start:stop]
    return design


def _replace_directory(target: Path, temporary: Path) -> None:
    backup: Path | None = None
    try:
        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"Observation-equation target is not a directory: {target}")
            backup = target.parent / f".{target.name}.old.{os.getpid()}"
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


def write_observation_equations(
    equations: _FrozenObservationEquations,
    path: str | Path,
) -> Path:
    target = require_file_group_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        pointers, columns, values = _csr(equations.design)
        payloads = {
            "rowPointers.dat.gz": pointers,
            "columnIndices.dat.gz": columns,
            "designValues.dat.gz": values,
            "observationVector.dat.gz": equations.reduced_observations,
            "sigmas.dat.gz": equations.sigmas,
        }
        for name, payload in payloads.items():
            write_matrix(temporary / name, payload, kind="vector")
        write_parameter_names(
            temporary / "parameterNames.txt",
            equations.parameter_names,
            equations.parameter_units,
        )
        write_structured_text(
            temporary / "metadata.txt",
            "observationEquationMetadata",
            dict(equations.metadata),
        )
        with atomic_text_writer(temporary / "observations.txt", "observationEquationRows") as stream:
            stream.write(f"recordCount {len(equations.identities)}\n")
            stream.write("# identity source jd1_utc jd2_utc station reflector light_time_converged wavelength_nm\n")
            stream.write("data\n")
            for values_row in zip(
                equations.identities,
                equations.sources,
                equations.epochs,
                equations.station_keys,
                equations.reflector_keys,
                equations.light_time_converged,
                equations.wavelengths_nm,
            ):
                identity, source, epoch, station, reflector, light_time_converged, wavelength = values_row
                stream.write(
                    " ".join(
                        (
                            str(identity),
                            encode_token(source),
                            format_float(epoch.jd1),
                            format_float(epoch.jd2),
                            encode_token(station),
                            encode_token(reflector),
                            "1" if light_time_converged else "0",
                            "~" if wavelength is None else format_float(wavelength),
                        )
                    )
                    + "\n"
                )
        with atomic_text_writer(temporary / "info.txt", "observationEquationInfo") as stream:
            stream.write(f"recordCount {len(equations.identities)}\n")
            stream.write(f"parameterCount {len(equations.parameter_names)}\n")
            stream.write(f"nonzeroCount {len(values)}\n")
            stream.write("timeScale UTC\n")
            all_payloads = [
                *payloads,
                "parameterNames.txt",
                "observations.txt",
                "metadata.txt",
            ]
            stream.write(f"payloadCount {len(all_payloads)}\n")
            for name in all_payloads:
                stream.write(f"payload {name} {sha256_file(temporary / name)}\n")
        _replace_directory(target, temporary)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def read_observation_equations(path: str | Path) -> _FrozenObservationEquations:
    source = require_file_group_path(path)
    if not source.is_dir():
        raise FileNotFoundError(f"Observation-equation file group not found: {source}")
    with open_text_reader(source / "info.txt") as stream:
        parse_header(stream, "observationEquationInfo")
        lines = iter(data_lines(stream))

        def pair(key: str) -> str:
            try:
                parts = next(lines).split(maxsplit=1)
            except StopIteration as exc:
                raise ValueError(f"Truncated observation-equation info in {source}.") from exc
            if len(parts) != 2 or parts[0] != key:
                raise ValueError(f"Expected {key!r} in observation-equation info.")
            return parts[1]

        row_count = int(pair("recordCount"))
        parameter_count = int(pair("parameterCount"))
        nonzero_count = int(pair("nonzeroCount"))
        if pair("timeScale") != "UTC":
            raise ValueError("Observation-equation time scale must be UTC.")
        payload_count = int(pair("payloadCount"))
        expected_payloads = {
            "rowPointers.dat.gz",
            "columnIndices.dat.gz",
            "designValues.dat.gz",
            "observationVector.dat.gz",
            "sigmas.dat.gz",
            "parameterNames.txt",
            "observations.txt",
            "metadata.txt",
        }
        if min(row_count, parameter_count, nonzero_count) < 0:
            raise ValueError("Observation-equation counts must be non-negative.")
        if payload_count != len(expected_payloads):
            raise ValueError(f"Observation-equation group must declare {len(expected_payloads)} payloads.")
        payload_names: list[str] = []
        for _ in range(payload_count):
            try:
                parts = next(lines).split()
            except StopIteration as exc:
                raise ValueError(f"Truncated observation-equation payload list in {source}.") from exc
            if len(parts) != 3 or parts[0] != "payload":
                raise ValueError("Malformed observation-equation payload record.")
            if parts[1] not in expected_payloads or parts[1] in payload_names:
                raise ValueError(f"Unexpected observation-equation payload name {parts[1]!r}.")
            payload_names.append(parts[1])
            if sha256_file(source / parts[1]) != parts[2]:
                raise ValueError(f"Observation-equation checksum mismatch: {parts[1]}")
        if len(payload_names) != len(set(payload_names)) or set(payload_names) != expected_payloads:
            raise ValueError(f"Observation-equation group has unexpected payload names: {payload_names!r}.")
        try:
            extra = next(lines)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValueError(f"Unexpected observation-equation info row {extra!r}.")

    names, units = read_parameter_names(source / "parameterNames.txt")
    if len(names) != parameter_count:
        raise ValueError("Observation-equation parameter count mismatch.")
    pointers = read_matrix(source / "rowPointers.dat.gz", expected_kind="vector")
    columns = read_matrix(source / "columnIndices.dat.gz", expected_kind="vector")
    design_values = read_matrix(source / "designValues.dat.gz", expected_kind="vector")
    if len(design_values) != nonzero_count:
        raise ValueError("Observation-equation nonzero count mismatch.")
    design = _dense_from_csr(
        pointers,
        columns,
        design_values,
        row_count=row_count,
        column_count=parameter_count,
    )
    observations = read_matrix(source / "observationVector.dat.gz", expected_kind="vector")
    sigmas = read_matrix(source / "sigmas.dat.gz", expected_kind="vector")

    with open_text_reader(source / "observations.txt") as stream:
        parse_header(stream, "observationEquationRows")
        lines = iter(data_lines(stream))
        try:
            count_parts = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError("Truncated observation-equation row header.") from exc
        if len(count_parts) != 2 or count_parts[0] != "recordCount" or marker != "data":
            raise ValueError("Malformed observation-equation row header.")
        declared = int(count_parts[1])
        identities: list[int] = []
        sources: list[str] = []
        epochs: list[Epoch] = []
        stations: list[str] = []
        reflectors: list[str] = []
        light_time_converged: list[bool] = []
        wavelengths: list[float | None] = []
        for line in lines:
            fields = line.split()
            if len(fields) != 8:
                raise ValueError(f"Malformed observation-equation row {line!r}.")
            identities.append(int(fields[0]))
            sources.append(decode_token(fields[1]))
            epochs.append(
                Epoch(
                    parse_float(fields[2], field="jd1_utc"),
                    parse_float(fields[3], field="jd2_utc"),
                    TimeScale.UTC,
                )
            )
            stations.append(decode_token(fields[4]))
            reflectors.append(decode_token(fields[5]))
            if fields[6] not in {"0", "1"}:
                raise ValueError("Observation-equation light_time_converged flag must be 0 or 1.")
            light_time_converged.append(fields[6] == "1")
            wavelengths.append(None if fields[7] == "~" else parse_float(fields[7], field="wavelength_nm"))
    if declared != row_count or len(identities) != row_count:
        raise ValueError("Observation-equation row count mismatch.")
    metadata = read_structured_text(source / "metadata.txt", "observationEquationMetadata")
    return _FrozenObservationEquations(
        tuple(names),
        tuple(units),
        design,
        observations,
        sigmas,
        tuple(identities),
        tuple(sources),
        tuple(epochs),
        tuple(stations),
        tuple(reflectors),
        tuple(light_time_converged),
        tuple(wavelengths),
        metadata,
    )


__all__ = [
    "read_observation_equations",
    "write_observation_equations",
]
