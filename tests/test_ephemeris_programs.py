from __future__ import annotations

from lunarops.config.context import RunContext
from lunarops.fileio.catalogs import read_reflector_catalog
from lunarops.programs.registry import ensure_builtin_programs, run_program


def test_reflector_catalog_create_from_fit_csv(tmp_path):
    ensure_builtin_programs()
    source = tmp_path / "coordinates.csv"
    source.write_text(
        "reflector_key,final_x_m,final_y_m,final_z_m\n"
        "APOLLO11,1.25,2.5,3.75\n",
        encoding="ascii",
    )
    context = RunContext(working_dir=tmp_path)
    output = run_program(
        "ReflectorCatalogCreate",
        {
            "source": "csv",
            "inputFileCoordinates": "coordinates.csv",
            "outputFileReflectorCatalog": "reflectors.txt",
        },
        context,
    )

    record = read_reflector_catalog(output)["APOLLO11"]
    assert list(record.moon_fixed_xyz_m) == [1.25, 2.5, 3.75]
    assert record.name == "Apollo 11"
    assert "A11" in record.aliases
