"""The installed package carries the Cython sources and no Fortran tree."""

import importlib.resources


def test_installed_iers_cython_sources_are_present():
    root = importlib.resources.files("lunarops")
    native = root.joinpath("_native")
    assert "Cython implementation" in native.joinpath("iers2010.pyx").read_text(encoding="utf-8")
    assert "Generated mechanically" in native.joinpath("iers2010_tables.pxi").read_text(encoding="utf-8")
    assert not root.joinpath("_external", "iers2010", "src").is_dir()
