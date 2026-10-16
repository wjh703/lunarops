from uuid import uuid4

from lunarops.config.registry import instantiate, register_factory, validate_class_config


def test_later_factory_declaration_replaces_earlier():
    category = f"test_registry_{uuid4().hex}"

    def original(config, context):
        return "original"

    def replacement(config, context):
        return "replacement"

    register_factory(category, "model", original)
    register_factory(category, "model", replacement)
    assert instantiate(category, validate_class_config(category, {"type": "model"})) == "replacement"
