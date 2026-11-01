from lunarops.base.parameter_name import ParameterName


def test_parameter_name_uses_non_builtin_field_names():
    name = ParameterName(
        object_name="apollo15",
        parameter_type="position.x",
        temporal="trend",
    )

    assert name.object_name == "apollo15"
    assert name.parameter_type == "position.x"
    assert str(name) == "apollo15:position.x:trend:"


def test_parameter_name_parse_round_trip_uses_renamed_fields():
    name = ParameterName.parse("GRASSE:rangeBias::")

    assert name.object_name == "GRASSE"
    assert name.parameter_type == "rangeBias"
    assert str(name) == "GRASSE:rangeBias::"
