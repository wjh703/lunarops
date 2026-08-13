"""Read and write covariance matrix directory artifacts."""

from __future__ import annotations

from pathlib import Path

from lunarops.estimation.parameter_products import CovarianceMatrix as _CovarianceMatrix

from .archive import atomic_directory_writer, atomic_text_writer, data_lines, decode_token, encode_token, open_text_reader, parse_header, require_file_group_path, sha256_file
from .matrix import read_matrix, write_matrix
from .parameter_names import read_parameter_names, write_parameter_names

FORMAT_VERSION = 1


def write_covariance(covariance: _CovarianceMatrix, path: str | Path) -> Path:
    target = require_file_group_path(path)
    with atomic_directory_writer(target) as directory:
        write_parameter_names(directory / "parameterNames.txt", covariance.parameter_names, covariance.units)
        write_matrix(directory / "covariance.dat.gz", covariance.matrix, kind="lowerSymmetric")
        with atomic_text_writer(directory / "info.txt", "covarianceInfo", version=FORMAT_VERSION) as stream:
            stream.write(f"covarianceKind {encode_token(covariance.kind)}\n")
            stream.write(f"parameterCount {len(covariance.parameter_names)}\n")
            stream.write(f"payload covariance.dat.gz {sha256_file(directory / 'covariance.dat.gz')}\n")
            stream.write(f"payload parameterNames.txt {sha256_file(directory / 'parameterNames.txt')}\n")
    return target


def read_covariance(path: str | Path) -> _CovarianceMatrix:
    source = require_file_group_path(path)
    with open_text_reader(source / "info.txt") as stream:
        parse_header(stream, "covarianceInfo", expected_version=FORMAT_VERSION)
        lines = list(data_lines(stream))
    if len(lines) != 4:
        raise ValueError(f"Malformed covariance info in {source}.")
    kind_parts = lines[0].split(maxsplit=1)
    count_parts = lines[1].split(maxsplit=1)
    if len(kind_parts) != 2 or kind_parts[0] != "covarianceKind":
        raise ValueError(f"Malformed covariance kind in {source}.")
    if len(count_parts) != 2 or count_parts[0] != "parameterCount":
        raise ValueError(f"Malformed covariance parameter count in {source}.")
    payloads: dict[str, str] = {}
    for line in lines[2:]:
        parts = line.split()
        if len(parts) != 3 or parts[0] != "payload" or parts[1] in payloads:
            raise ValueError(f"Malformed covariance payload record in {source}.")
        payloads[parts[1]] = parts[2]
    expected = {"covariance.dat.gz", "parameterNames.txt"}
    if set(payloads) != expected:
        raise ValueError(f"Unexpected covariance payloads in {source}.")
    for name, digest in payloads.items():
        if sha256_file(source / name) != digest:
            raise ValueError(f"Covariance payload checksum mismatch: {name}")
    names, units = read_parameter_names(source / "parameterNames.txt")
    if int(count_parts[1]) != len(names):
        raise ValueError("Covariance parameter count mismatch.")
    matrix = read_matrix(source / "covariance.dat.gz", expected_kind="lowerSymmetric")
    return _CovarianceMatrix(tuple(names), matrix, tuple(units), decode_token(kind_parts[1]))


__all__ = ["read_covariance", "write_covariance"]
