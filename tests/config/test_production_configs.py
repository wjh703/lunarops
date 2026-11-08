"""Every production configuration under configs/ resolves to native YAML data with its recorded key set."""

import json

import pytest
import yaml

from lunarops.base.serialization import plain_data
from lunarops.config.context import RunContext
from lunarops.config.loader import build_run_plan, load_config_file
from lunarops.programs.registry import ensure_builtin_programs, resolve_program_config
from tests.conftest import CONFIGS_DIR, DATA_DIR, REPO_ROOT

CONFIGS = sorted((*CONFIGS_DIR.glob("*.yml"), *CONFIGS_DIR.glob("*.yaml")))
# Leaf paths of every resolved program configuration, generated at 269ea29 (see PROGRESS "F6 accepted").
SNAPSHOT = json.loads((DATA_DIR / "resolved_config_keys.json").read_text())


def test_the_production_configurations_are_collected():
    assert len(CONFIGS) == len(SNAPSHOT["configs"]) > 0
    assert SNAPSHOT["sourceCommit"] == "269ea29"


def _leaves(value, prefix=""):
    if isinstance(value, dict) and value:
        return [path for key, child in value.items() for path in _leaves(child, f"{prefix}.{key}" if prefix else key)]
    if isinstance(value, list) and value:
        return [path for index, child in enumerate(value) for path in _leaves(child, f"{prefix}[{index}]")]
    return [prefix]


def _native(value):
    assert type(value) in (str, int, float, bool, type(None), list, dict), type(value)
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _native(item)
    elif type(value) is list:
        for item in value:
            _native(item)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_configuration_resolves_to_native_data_with_the_recorded_key_set(path):
    ensure_builtin_programs()
    plan = build_run_plan(load_config_file(path))
    context = RunContext(model_configs=plan.models, working_dir=REPO_ROOT)
    context.resolve_defaults()
    resolved = [(name, resolve_program_config(name, body)) for name, body in plan.calls]

    for _, config in resolved:
        _native(config)
        yaml.safe_dump(plain_data(config))
    assert [{"program": name, "paths": sorted(_leaves(config))} for name, config in resolved] == (
        SNAPSHOT["configs"][path.name]
    )
