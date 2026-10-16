from __future__ import annotations

import pytest

from lunarops.config.context import RunContext
from lunarops.fileio.mass_catalog import BodyMass, MassCatalog, read_mass_catalog
from lunarops.programs.mass_catalog_create import AU_M, DAY_S, earth_moon_gm_from_gmb_emrat
from lunarops.programs.registry import ensure_builtin_programs, run_program


def test_mass_catalog_create_normalizes_units_and_selects_groups(tmp_path):
    ensure_builtin_programs()
    output = run_program(
        "MassCatalogCreate",
        {
            "catalogName": "test system",
            "source": "unit test",
            "expectedBodyCount": 2,
            "expectedGroupCounts": {"major": 1, "sb441": 1},
            "outputFileMassCatalog": "masses.txt",
            "bodyMasses": [
                {
                    "bodyId": "EARTH",
                    "ephemerisTarget": 399,
                    "gm": 3.986e14,
                    "groups": ["major"],
                },
                {
                    "bodyId": "NAIF:2000001",
                    "name": "Ceres",
                    "ephemerisTarget": 2000001,
                    "gm": 1.0e-13,
                    "gmUnit": "au3/day2",
                    "groups": ["sb441", "asteroid"],
                    "source": "table",
                },
            ],
        },
        RunContext(working_dir=tmp_path),
    )
    catalog = read_mass_catalog(output)
    assert catalog.require("EARTH").gm_m3_s2 == 3.986e14
    assert catalog.require("NAIF:2000001").gm_m3_s2 == pytest.approx(1.0e-13 * AU_M**3 / DAY_S**2)
    assert [body.name for body in catalog.select(groups=["asteroid"])] == ["Ceres"]


def test_mass_catalog_rejects_duplicate_ids_and_inconsistent_naif_target():
    earth = BodyMass("EARTH", 1.0, ("major",), 399)
    with pytest.raises(ValueError, match="body IDs"):
        MassCatalog((earth, earth), name="duplicate")
    with pytest.raises(ValueError, match="canonical NAIF ID"):
        BodyMass("MOON", 1.0, ("major",), 399)


def test_mass_catalog_derives_earth_and_moon_from_de440_gmb_and_emrat(tmp_path):
    ensure_builtin_programs()
    output = run_program(
        "MassCatalogCreate",
        {
            "catalogName": "DE440 masses",
            "source": "DE440 integration constants",
            "earthMoonCanonical": {
                "gmbAu3Day2": 8.9970113929473466e-10,
                "emrat": 81.300568221497215,
            },
            "outputFileMassCatalog": "masses.txt",
            "bodyMasses": [
                {"bodyId": "EARTH", "ephemerisTarget": 399, "groups": ["major"]},
                {"bodyId": "MOON", "ephemerisTarget": 301, "groups": ["major"]},
            ],
        },
        RunContext(working_dir=tmp_path),
    )
    catalog = read_mass_catalog(output)
    earth, moon = earth_moon_gm_from_gmb_emrat(8.9970113929473466e-10, 81.300568221497215)
    assert catalog.require("EARTH").gm_m3_s2 == earth
    assert catalog.require("MOON").gm_m3_s2 == moon
    assert earth + moon == pytest.approx(8.9970113929473466e-10 * AU_M**3 / DAY_S**2, rel=2e-16)
