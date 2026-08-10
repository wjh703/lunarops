"""Data-only expressions used by LunarOps YAML configuration.

This module has no dependency on schemas, registries, or runtime objects.  It
is the small expression language understood by a run configuration: variable
references and boolean conditions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


VARIABLE_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
VARIABLE_REFERENCE_PATTERN = rf"^\{{{VARIABLE_NAME_PATTERN}\}}$"
CONDITION_OPERATORS = ("equals", "notEquals", "in", "all", "any", "not")

_PLACEHOLDER = re.compile(rf"\{{({VARIABLE_NAME_PATTERN})\}}")
_VARIABLE_NAME = re.compile(VARIABLE_NAME_PATTERN)


def is_variable_name(value: Any) -> bool:
    return isinstance(value, str) and _VARIABLE_NAME.fullmatch(value) is not None


def variable_reference_json_schema() -> dict[str, str]:
    """Return the raw-YAML form of a full ``{variable}`` reference."""
    return {"type": "string", "pattern": VARIABLE_REFERENCE_PATTERN}


class _VariableResolver:
    def __init__(self, variables: Mapping[str, Any]) -> None:
        self.raw = dict(variables)
        invalid = [name for name in self.raw if not is_variable_name(name)]
        if invalid:
            raise ValueError(f"Config variable names must be identifiers: {invalid!r}")
        self.resolved: dict[str, Any] = {}
        self.active: list[str] = []

    def resolve_all(self) -> dict[str, Any]:
        for name in self.raw:
            self.resolve_name(name)
        return self.resolved

    def resolve_name(self, name: str) -> Any:
        if name in self.resolved:
            return self.resolved[name]
        if name not in self.raw:
            raise KeyError(f"Undefined config variable {{{name}}}")
        if name in self.active:
            start = self.active.index(name)
            cycle = [*self.active[start:], name]
            raise ValueError(f"Config variable cycle detected: {' -> '.join(cycle)}")

        self.active.append(name)
        try:
            result = self.resolve_value(self.raw[name])
            self.resolved[name] = result
            return result
        finally:
            self.active.pop()

    def resolve_value(self, value: Any) -> Any:
        if isinstance(value, str):
            match = _PLACEHOLDER.fullmatch(value)
            if match:
                return deepcopy(self.resolve_name(match.group(1)))
            return _PLACEHOLDER.sub(lambda item: str(self.resolve_name(item.group(1))), value)
        if isinstance(value, list):
            return [self.resolve_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.resolve_value(item) for item in value)
        if isinstance(value, Mapping):
            return {key: self.resolve_value(item) for key, item in value.items()}
        return deepcopy(value)


def resolve_variables(variables: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve variable-to-variable references with cycle detection."""
    if not isinstance(variables, Mapping):
        raise TypeError("Config variables must be a mapping.")
    return _VariableResolver(variables).resolve_all()


def substitute_resolved(value: Any, variables: Mapping[str, Any]) -> Any:
    """Substitute values using an already-resolved variable mapping."""
    if isinstance(value, str):
        match = _PLACEHOLDER.fullmatch(value)
        if match and match.group(1) in variables:
            return deepcopy(variables[match.group(1)])

        def replace(item: re.Match[str]) -> str:
            name = item.group(1)
            if name not in variables:
                raise KeyError(f"Undefined config variable {{{name}}} in {value!r}")
            return str(variables[name])

        return _PLACEHOLDER.sub(replace, value)
    if isinstance(value, list):
        return [substitute_resolved(item, variables) for item in value]
    if isinstance(value, tuple):
        return tuple(substitute_resolved(item, variables) for item in value)
    if isinstance(value, Mapping):
        return {key: substitute_resolved(item, variables) for key, item in value.items()}
    return deepcopy(value)


def substitute(value: Any, variables: Mapping[str, Any]) -> Any:
    """Resolve variables, then recursively substitute their placeholders."""
    return substitute_resolved(value, resolve_variables(variables))


def evaluate_resolved_condition(condition: Any, *, path: str = "when") -> bool:
    """Evaluate a condition after all variable references have been replaced."""
    if isinstance(condition, bool):
        return condition
    if not isinstance(condition, Mapping):
        raise TypeError(f"{path} must resolve to a boolean condition mapping or boolean.")
    if len(condition) != 1:
        raise ValueError(f"{path} must contain exactly one condition operator.")

    operator, operand = next(iter(condition.items()))
    if operator in {"equals", "notEquals"}:
        if isinstance(operand, (str, bytes)) or not isinstance(operand, Sequence) or len(operand) != 2:
            raise TypeError(f"{path}.{operator} must be a two-item list.")
        result = operand[0] == operand[1]
        return result if operator == "equals" else not result
    if operator == "in":
        if not isinstance(operand, Mapping) or set(operand) != {"value", "values"}:
            raise TypeError(f"{path}.in must contain value and values.")
        values = operand["values"]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{path}.in.values must be a list.")
        return operand["value"] in values
    if operator in {"all", "any"}:
        if isinstance(operand, (str, bytes)) or not isinstance(operand, Sequence):
            raise TypeError(f"{path}.{operator} must be a list of conditions.")
        results = (
            evaluate_resolved_condition(item, path=f"{path}.{operator}[{index}]")
            for index, item in enumerate(operand)
        )
        return all(results) if operator == "all" else any(results)
    if operator == "not":
        return not evaluate_resolved_condition(operand, path=f"{path}.not")
    raise ValueError(
        f"{path} has unknown condition operator {operator!r}; expected one of {list(CONDITION_OPERATORS)!r}."
    )


def evaluate_condition(condition: Any, variables: Mapping[str, Any]) -> bool:
    """Resolve variables and evaluate a data-only ``when`` predicate."""
    return evaluate_resolved_condition(substitute(condition, variables))


__all__ = [
    "CONDITION_OPERATORS",
    "VARIABLE_NAME_PATTERN",
    "VARIABLE_REFERENCE_PATTERN",
    "evaluate_condition",
    "evaluate_resolved_condition",
    "is_variable_name",
    "resolve_variables",
    "substitute",
    "substitute_resolved",
    "variable_reference_json_schema",
]
