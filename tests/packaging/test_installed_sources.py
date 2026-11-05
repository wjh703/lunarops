"""The installed package carries the Cython sources and no Fortran tree."""

import importlib.resources


def test_installed_iers_cython_sources_are_present():
    root = importlib.resources.files("lunarops")
    assert "Cython implementation" in root.joinpath("_iers2010_core.pyx").read_text(encoding="utf-8")
    assert "Generated mechanically" in root.joinpath("_iers2010_tables.pxi").read_text(encoding="utf-8")
    assert not root.joinpath("_external", "iers2010", "src").is_dir()
