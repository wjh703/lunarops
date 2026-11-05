"""Read and write ``observationResult`` files of per-observation diagnostics.

The file is a records table (see :mod:`lunarops.fileio.result_table`) with artifact type
``observationResult`` and an empty header. The first field is ``source`` (text), naming the input source of
each row; the other fields are inferred from the rows. Field units come from the full diagnostic unit map of
:mod:`lunarops.classes.observation.measurement`, or from the name suffix for fields not listed there.
"""

from lunarops.classes.observation.measurement import FULL_DIAGNOSTIC_UNITS

from .result_table import infer_fields, read_records, write_records


def write_observation_results(results_by_source, path):
    """Write an ``observationResult`` file.

    Parameters
    ----------
    results_by_source : Mapping of str to Sequence of Mapping
        Diagnostic rows keyed by source name. Each row is written with a leading ``source`` field; fields
        missing from a row are written as ``~``.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    Path
        The written path.

    Raises
    ------
    ValueError
        If there are no rows, or a value cannot be written.
    TypeError
        If a field mixes value types.
    """
    rows = [{"source": source, **row} for source, source_rows in results_by_source.items() for row in source_rows]
    if not rows:
        raise ValueError("No observation results to write.")
    fields = infer_fields(rows)
    rows = [{name: row.get(name) for name, _ in fields} for row in rows]
    return write_records(path, "observationResult", {}, fields, rows, field_units=FULL_DIAGNOSTIC_UNITS)


def read_observation_results(path):
    """Read an ``observationResult`` file.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    list of dict
        Diagnostic rows in file order, each starting with ``source``.

    Raises
    ------
    ValueError
        If a unit differs from the expected unit, the first field is not a text ``source`` field, or the
        records table is invalid.
    """
    _, rows = read_records(path, "observationResult", field_units=FULL_DIAGNOSTIC_UNITS)
    if rows and (next(iter(rows[0])) != "source" or not isinstance(rows[0]["source"], str)):
        raise ValueError("Observation-result schema must begin with a text 'source' field.")
    return rows


__all__ = ["read_observation_results", "write_observation_results"]
