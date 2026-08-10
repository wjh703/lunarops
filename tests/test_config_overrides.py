import pytest

from lunarops.config.loader import (
    build_run_plan,
    load_config_file,
    parse_set_overrides,
)


def test_parse_set_overrides_native_types():
    overrides = parse_set_overrides(
        [
            "flag=false",
            "count=5",
            "scale=0.25",
            "missing=null",
            "items=[1, 2, 3]",
            'mapping={"a": 1}',
            'text="001"',
            "raw=abc123",
        ]
    )

    assert overrides == {
        "flag": False,
        "count": 5,
        "scale": 0.25,
        "missing": None,
        "items": [1, 2, 3],
        "mapping": {"a": 1},
        "text": "001",
        "raw": "abc123",
    }


def test_full_placeholder_substitution_preserves_override_type():
    config = {
        "variables": {"enabled": True, "n": 1},
        "programs": [
            {"program": "Dummy", "flag": "{enabled}", "count": "{n}"},
        ],
    }
    overrides = parse_set_overrides(["enabled=false", "n=7"])
    plan = build_run_plan(config, overrides)
    _, program_config = plan.calls[0]

    assert program_config["flag"] is False
    assert program_config["count"] == 7


def test_run_configuration_is_yaml_only(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"programs": []}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.yml"):
        load_config_file(path)
