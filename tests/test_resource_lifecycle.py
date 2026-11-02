import pytest

from lunarops.config.context import ClassCache


def test_cache_exit_stack_closes_remaining_resources_after_failure():
    closed = []
    cache = ClassCache()
    cache.resources.callback(closed.append, "first")

    def fail():
        raise RuntimeError("close failure")

    cache.resources.callback(fail)
    cache.resources.callback(closed.append, "last")
    with pytest.raises(RuntimeError, match="close failure"):
        cache.close()
    assert closed == ["last", "first"]
