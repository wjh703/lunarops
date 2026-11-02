from functools import lru_cache


@lru_cache(maxsize=16)
def coefficient_file(factory, path):
    return factory(path)
