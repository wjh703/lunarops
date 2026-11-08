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
            "models": {"sites": "{sites}", "troposphere": {"type": "none"}},
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

    assert plan.models == {"sites": ["A", "B"], "troposphere": {"type": "none"}}
    assert plan.calls == (("P", {"output": "B.txt"}), ("Q", {"enabled": False}))


def test_variable_cycles_and_non_yaml_run_files_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="cycle detected"):
        resolve_variables({"first": "{second}", "second": "{first}"})
    (tmp_path / "config.json").write_text('{"programs": []}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.yml"):
        load_config_file(tmp_path / "config.json")


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_extends_merges_variables_and_models_by_key_and_replaces_programs(tmp_path):
    _write(
        tmp_path / "base" / "common.yml",
        """
variables: {a: 1, b: 2}
models:
  troposphere: {type: mendesPavlis, keep: 1}
  relativity: {type: iersShapiro}
programs: [{program: Base}]
""",
    )
    _write(tmp_path / "base" / "mixin.yml", "variables: {b: 3, c: 4}\n")
    leaf = _write(
        tmp_path / "leaf.yml",
        """
extends: [base/common.yml, base/mixin.yml]
variables: {c: 5}
models:
  troposphere: {type: none}
""",
    )

    assert load_config_file(leaf) == {
        "variables": {"a": 1, "b": 3, "c": 5},
        "models": {"troposphere": {"type": "none"}, "relativity": {"type": "iersShapiro"}},
        "programs": [{"program": "Base"}],
    }


def test_a_child_replaces_the_inherited_programs_and_parents_chain(tmp_path):
    _write(tmp_path / "grandparent.yml", "variables: {a: 1}\nprograms: [{program: Old}]\n")
    _write(tmp_path / "parent.yml", "extends: grandparent.yml\nvariables: {b: 2}\n")
    leaf = _write(tmp_path / "leaf.yml", "extends: parent.yml\nprograms: [{program: New}]\n")

    assert load_config_file(leaf) == {"variables": {"a": 1, "b": 2}, "programs": [{"program": "New"}]}


def test_set_overrides_reach_variables_defined_in_an_extended_file(tmp_path):
    _write(tmp_path / "base.yml", "variables: {name: a}\nprograms: [{program: P, file: '{name}.txt'}]\n")
    leaf = _write(tmp_path / "leaf.yml", "extends: base.yml\n")

    plan = build_run_plan(load_config_file(leaf), parse_set_overrides(["name=b"]))
    assert plan.calls == (("P", {"file": "b.txt"}),)


def test_extends_rejects_cycles_and_malformed_values(tmp_path):
    first = _write(tmp_path / "first.yml", "extends: second.yml\n")
    _write(tmp_path / "second.yml", "extends: first.yml\n")
    with pytest.raises(ValueError, match="extends cycle"):
        load_config_file(first)
    with pytest.raises(ValueError, match="extends must be"):
        load_config_file(_write(tmp_path / "bad.yml", "extends: 3\n"))
    with pytest.raises(ValueError, match="extends must be"):
        load_config_file(_write(tmp_path / "empty.yml", "extends: []\n"))
    with pytest.raises(ValueError, match="unknown configuration key"):
        build_run_plan({"extends": "unexpanded.yml"})
