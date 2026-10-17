import logging

from lunarops.config.context import close_resources


class Resource:
    def __init__(self):
        self.count = 0

    def close(self):
        self.count += 1


def test_close_resources_deduplicates_resources(caplog):
    resource = Resource()
    with caplog.at_level(logging.WARNING):
        close_resources([resource, resource], owner="test")
    assert resource.count == 1
