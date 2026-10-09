"""Observation-result schema over the shared scalar table archive."""

from .result_table import infer_fields, read_table, write_table

FORMAT_VERSION = 1


def write_observation_results(results_by_source, path):
    rows = [{"source": source, **row} for source, source_rows in results_by_source.items() for row in source_rows]
    if not rows:
        raise ValueError("No observation results to write.")
    fields = infer_fields(rows)
    rows = [{name: row.get(name) for name, _ in fields} for row in rows]
    return write_table(rows, path, artifact_type="observationResult", version=FORMAT_VERSION, fields=fields)


def read_observation_results(path):
    rows = read_table(path, artifact_type="observationResult", version=FORMAT_VERSION)
    if rows and (next(iter(rows[0])) != "source" or not isinstance(rows[0]["source"], str)):
        raise ValueError("Observation-result schema must begin with a text 'source' field.")
    return rows


__all__ = ["read_observation_results", "write_observation_results"]
