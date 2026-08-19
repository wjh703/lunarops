from __future__ import annotations

from lunarops.config.context import RunContext
from lunarops.fileio.catalogs import read_reflector_catalog
from lunarops.programs.registry import ensure_builtin_programs, run_program


def test_reflector_catalog_create_from_inline_coordinates(tmp_path):
    ensure_builtin_programs()
    context = RunContext(working_dir=tmp_path)
    output = run_program(
        "ReflectorCatalogCreate",
        {
            "reflectorCoordinates": [{"key": "APOLLO11", "xyzM": [1.25, 2.5, 3.75]}],
            "outputFileReflectorCatalog": "reflectors.txt",
        },
        context,
    )

    record = read_reflector_catalog(output)["APOLLO11"]
    assert list(record.moon_fixed_xyz_m) == [1.25, 2.5, 3.75]
    assert record.name == "APOLLO11"
    assert record.aliases == ()
