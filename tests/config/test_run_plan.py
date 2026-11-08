"""Run files: variables, loops, conditions and --set overrides."""

import pytest

from lunarops.config.loader import build_run_plan, load_config_file, parse_set_overrides, resolve_variables


def test_set_overrides_are_parsed_as_yaml_scalars():
    assert parse_set_overrides(
        [
            "flag=false",
            "count=5",
            "scale=0.25",
            "missing=null",
            "items=[1, 2, 3]",
            'mapping={"a": 1}',
            'text="001"',
            "raw=abc123",
            "leading=001",
            "scientific=1e3",
        ]
    ) == {
        "flag": False,
        "count": 5,
        "scale": 0.25,
        "missing": None,
        "items": [1, 2, 3],
        "mapping": {"a": 1},
        "text": "001",
        "raw": "abc123",
        "leading": 1,
        "scientific": "1e3",
    }


def test_whole_placeholders_keep_the_override_type():
    plan = build_run_plan(
        {
            "variables": {"enabled": True, "n": 1},
            "programs": [{"program": "Dummy", "flag": "{enabled}", "count": "{n}"}],
        },
        parse_set_overrides(["enabled=false", "n=7"]),
    )
    assert plan.calls[0][1] == {"flag": False, "count": 7}


def test_loops_and_conditions_expand_while_program_fields_are_left_to_the_program():
    plan = build_run_plan(
        {
            "variables": {"sites": ["A", "B"]},
            "shared": {"sites": "{sites}"},
            "observationModel": {"troposphere": {"type": "none"}},
            "programs": [
                {
                    "program": "P",
                    "loop": {"variable": "site", "values": "{sites}"},
                    "when": {"equals": ["{site}", "B"]},
                    "output": "{site}.txt",
                },
                {"program": "Q", "enabled": False},
            ],
        }
    )

    assert plan.shared == {"sites": ["A", "B"]}
    assert plan.observation_model == {"troposphere": {"type": "none"}}
    assert plan.calls == (("P", {"output": "B.txt"}), ("Q", {"enabled": False}))


def test_variable_cycles_and_non_yaml_run_files_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="cycle detected"):
        resolve_variables({"first": "{second}", "second": "{first}"})
    (tmp_path / "config.json").write_text('{"programs": []}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.yml"):
        load_config_file(tmp_path / "config.json")
