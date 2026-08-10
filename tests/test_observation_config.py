import pytest

import lunarops.cli as cli
from lunarops.classes.observation_factory import ensure_registered
from lunarops.config.registry import validate_global_class_configs
from lunarops.programs.registry import validate_program_config


def _register_observation_contracts():
    cli._import_programs()
    ensure_registered()


@pytest.mark.parametrize("key", ["uncertainty", "uncertaintyModel"])
def test_observation_program_schema_rejects_undeclared_options(key):
    _register_observation_contracts()
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_program_config("LlrResiduals", {key: "obsolete"})


def test_global_schema_rejects_undeclared_options():
    _register_observation_contracts()
    with pytest.raises(ValueError, match="unknown configuration key"):
        validate_global_class_configs({"uncertaintyModel": "obsolete"})
