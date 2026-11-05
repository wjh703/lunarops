"""Loading of YAML run configs and expansion into an ordered list of program calls.

A run config has four top-level sections: ``variables`` (values referenced as ``{name}``), ``shared``
(run-level model and catalog configs), ``observationModel`` (default LLR observation models) and
``programs`` (ordered program entries). A program entry names its ``program`` and may carry a ``loop``,
which repeats it once per value of a loop variable, and a ``when`` condition, which skips it when false.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .expressions import (
    evaluate_resolved_condition,
    is_variable_name,
    resolve_variables,
    substitute_resolved,
)
from .overrides import parse_set_overrides
from .schema import ConfigSchema, field, mapping, sequence, string


def _validate_loop(config: dict[str, Any], path: str) -> dict[str, Any]:
    if not is_variable_name(config["variable"]):
        raise ValueError(f"{path}.variable must be an identifier.")
    return config


_LOOP_SCHEMA = ConfigSchema(
    fields=(
        string(
            "variable",
            required=True,
            non_empty=True,
            description="Name of the loop variable, usable as `{name}` in the program entry.",
        ),
        sequence(
            "values",
            required=True,
            min_items=1,
            description="Values taken by the loop variable; the program entry runs once per value.",
        ),
    ),
    description="Expand one program entry over a list of loop values.",
    validator=_validate_loop,
)
PROGRAM_CONTROL_SCHEMA = ConfigSchema(
    fields=(
        mapping("loop", nested=_LOOP_SCHEMA, description="Run this program entry once per loop value."),
        field("when", "any", description="Data-only condition evaluated after loop-variable substitution."),
    ),
    description="Execution controls shared by every program entry.",
)
RUN_CONFIG_SCHEMA = ConfigSchema(
    fields=(
        mapping("variables", default={}, description="Identifier-named values used by placeholders."),
        mapping("shared", default={}, description="Run-level resources shared by programs."),
        mapping("observationModel", default={}, description="Default LLR observation models."),
        sequence("programs", default=[], item_kind="mapping", description="Ordered program calls."),
    ),
    description="LunarOps YAML run configuration.",
)


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Run config with variables substituted and program entries expanded.

    Attributes
    ----------
    variables : dict
        Resolved variables, including ``--set`` overrides.
    shared : dict
        ``shared`` section with variables substituted.
    observation_model : dict
        ``observationModel`` section with variables substituted.
    calls : tuple of (str, dict)
        Program calls in run order, as (program name, program config) pairs.
    """

    variables: dict[str, Any]
    shared: dict[str, Any]
    observation_model: dict[str, Any]
    calls: tuple[tuple[str, dict[str, Any]], ...]


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a YAML run config file.

    Parameters
    ----------
    path : str or Path
        Path of a ``.yml`` or ``.yaml`` file, ``~`` expanded.

    Returns
    -------
    dict
        Parsed YAML document.

    Raises
    ------
    ValueError
        If the file suffix is not ``.yml`` or ``.yaml`` or the file is not valid YAML.
    """
    source = Path(path).expanduser()
    if source.suffix.lower() not in (".yml", ".yaml"):
        raise ValueError(f"LunarOps configuration files must use .yml or .yaml: {source}")
    import yaml

    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration {source}: {exc}") from exc
    return data


def _merge_overrides(variables: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return variables
    unknown = set(overrides) - set(variables)
    if unknown:
        raise ValueError(f"--set refers to undefined variable(s): {sorted(unknown)}")
    return {**variables, **overrides}


def _program_body(entry: Mapping[str, Any]) -> dict[str, Any]:
    controls = {"program", "loop", "when"}
    return {key: value for key, value in entry.items() if key not in controls}


def _expand_program(
    entry: Mapping[str, Any],
    index: int,
    variables: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    path = f"programs[{index}]"
    control_values = {"loop": entry["loop"]} if "loop" in entry else {}
    controls = PROGRAM_CONTROL_SCHEMA.resolve(
        substitute_resolved(control_values, variables),
        path=path,
    )
    loop = controls.get("loop")
    loop_variable = loop["variable"] if loop else None
    candidates = loop["values"] if loop else (None,)
    body = _program_body(entry)
    calls: list[tuple[str, dict[str, Any]]] = []
    for loop_value in candidates:
        local_variables = dict(variables)
        if loop_variable is not None:
            local_variables[loop_variable] = loop_value
        if "when" in entry and not evaluate_resolved_condition(
            substitute_resolved(entry["when"], local_variables), path=f"{path}.when"
        ):
            continue

        name = substitute_resolved(entry["program"], local_variables)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.program must resolve to a non-empty string.")
        resolved_body = substitute_resolved(body, local_variables)
        calls.append((name.strip(), resolved_body))
    return calls


def build_run_plan(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> RunPlan:
    """Resolve a run config into its program calls.

    Overrides replace variables before the variables are resolved. Each program entry is expanded over its
    ``loop`` values, entries whose ``when`` condition is false are dropped, and variables are substituted in
    the program name and config. Program configs are not validated against their schemas here.

    Parameters
    ----------
    config : Mapping
        Parsed run config.
    overrides : Mapping, optional
        Variable values from ``--set``, keyed by variable name.

    Returns
    -------
    RunPlan
        Resolved sections and program calls.

    Raises
    ------
    ValueError
        If an override names an undefined variable, a program entry has no ``program`` key or its name
        does not resolve to a non-empty string.
    """
    resolved = RUN_CONFIG_SCHEMA.resolve(config, path="configuration")
    variables = resolve_variables(_merge_overrides(resolved["variables"], overrides))
    shared = substitute_resolved(resolved["shared"], variables)
    observation_model = substitute_resolved(resolved["observationModel"], variables)
    calls: list[tuple[str, dict[str, Any]]] = []
    for index, entry in enumerate(resolved["programs"]):
        if "program" not in entry:
            raise ValueError(f"programs[{index}] requires a 'program' key.")
        calls.extend(_expand_program(entry, index, variables))
    return RunPlan(variables, shared, observation_model, tuple(calls))


__all__ = [
    "PROGRAM_CONTROL_SCHEMA",
    "RUN_CONFIG_SCHEMA",
    "RunPlan",
    "build_run_plan",
    "load_config_file",
    "parse_set_overrides",
    "resolve_variables",
]
