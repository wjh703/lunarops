"""Load a YAML scenario and compile it into an executable run plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .expressions import (
    CONDITION_OPERATORS,
    evaluate_condition,
    evaluate_resolved_condition,
    is_variable_name,
    resolve_variables,
    substitute,
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
        string("variable", required=True, non_empty=True, allow_none=False),
        sequence("values", required=True, min_items=1, allow_none=False),
    ),
    description="Expand one program entry over a list of loop values.",
    validator=_validate_loop,
)
_WHEN_FIELD = field(
    "when",
    "any",
    allow_none=False,
    description="Data-only condition evaluated after loop-variable substitution.",
)
_PROGRAM_CONTROL_SCHEMA = ConfigSchema(
    fields=(
        mapping("loop", nested=_LOOP_SCHEMA),
        _WHEN_FIELD,
    ),
    description="Execution controls shared by every program entry.",
)
_RUN_CONFIG_SCHEMA = ConfigSchema(
    fields=(
        mapping(
            "variables",
            default={},
            allow_none=False,
            allow_variable_reference=False,
            description="Identifier-named values used by placeholders.",
        ),
        mapping(
            "globals",
            default={},
            allow_none=False,
            allow_variable_reference=False,
            description="Shared class configurations and catalogs.",
        ),
        sequence(
            "programs",
            default=[],
            item_kind="mapping",
            allow_none=False,
            allow_variable_reference=False,
            description="Ordered program calls.",
        ),
    ),
    description="LunarOps YAML run configuration.",
)


def program_control_schema() -> ConfigSchema:
    return _PROGRAM_CONTROL_SCHEMA


def run_config_schema() -> ConfigSchema:
    return _RUN_CONFIG_SCHEMA


def condition_operators() -> tuple[str, ...]:
    return CONDITION_OPERATORS


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Fully expanded run configuration, ready for validation and execution."""

    variables: dict[str, Any]
    globals: dict[str, Any]
    calls: tuple[tuple[str, dict[str, Any]], ...]


def load_config_file(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.suffix.lower() not in (".yml", ".yaml"):
        raise ValueError(f"LunarOps configuration files must use .yml or .yaml: {source}")
    import yaml

    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration {source}: {exc}") from exc
    return _RUN_CONFIG_SCHEMA.resolve(data, path=f"configuration {source}")


def _config_sections(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    resolved = _RUN_CONFIG_SCHEMA.resolve(config, path="configuration")
    return resolved["variables"], resolved["globals"], resolved["programs"]


def _merge_overrides(variables: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return variables
    if not isinstance(overrides, Mapping):
        raise TypeError("Config overrides must be a mapping.")
    unknown = set(overrides) - set(variables)
    if unknown:
        raise ValueError(f"--set refers to undefined variable(s): {sorted(unknown)}")
    return {**variables, **overrides}


def _program_body(entry: Mapping[str, Any]) -> dict[str, Any]:
    controls = {"program", "loop", "when"}
    return {key: value for key, value in entry.items() if key not in controls}


def _validate_program_entry(entry: Any, index: int) -> Mapping[str, Any]:
    path = f"programs[{index}]"
    if not isinstance(entry, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    if any(not isinstance(key, str) for key in entry):
        raise TypeError(f"{path} keys must be strings.")
    if "program" not in entry:
        raise ValueError(f"{path} requires a 'program' key.")
    if "enabled" in entry:
        raise ValueError(f"{path}.enabled has been removed; omit the program entry instead.")
    return entry


def _expand_program(
    entry: Mapping[str, Any],
    index: int,
    variables: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    path = f"programs[{index}]"
    control_values = {"loop": entry["loop"]} if "loop" in entry else {}
    controls = _PROGRAM_CONTROL_SCHEMA.resolve(
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
        if "when" in entry:
            condition = _WHEN_FIELD.validate(
                substitute_resolved(entry["when"], local_variables),
                f"{path}.when",
            )
            if not evaluate_resolved_condition(condition, path=f"{path}.when"):
                continue

        name = substitute_resolved(entry["program"], local_variables)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.program must resolve to a non-empty string.")
        resolved_body = substitute_resolved(body, local_variables)
        if not isinstance(resolved_body, dict):
            raise TypeError(f"{path} body must resolve to a mapping.")
        calls.append((name.strip(), resolved_body))
    return calls


def build_run_plan(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> RunPlan:
    """Resolve globals once and expand each program's controls into calls."""
    raw_variables, raw_globals, programs = _config_sections(config)
    variables = resolve_variables(_merge_overrides(raw_variables, overrides))
    resolved_globals = substitute_resolved(raw_globals, variables)
    if not isinstance(resolved_globals, dict):
        raise TypeError("Resolved top-level 'globals' section must be a mapping.")

    calls: list[tuple[str, dict[str, Any]]] = []
    for index, raw_entry in enumerate(programs):
        entry = _validate_program_entry(raw_entry, index)
        calls.extend(_expand_program(entry, index, variables))
    return RunPlan(variables, resolved_globals, tuple(calls))


__all__ = [
    "RunPlan",
    "build_run_plan",
    "condition_operators",
    "evaluate_condition",
    "load_config_file",
    "parse_set_overrides",
    "program_control_schema",
    "resolve_variables",
    "run_config_schema",
    "substitute",
]
