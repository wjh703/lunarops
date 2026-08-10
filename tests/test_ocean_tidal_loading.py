from __future__ import annotations

import numpy as np
import pytest

from lunarops.classes.time import Epoch, TimeScale
from lunarops.classes.displacement import (
    Iers2010OceanTidalLoading,
    OceanTidalLoadingCatalog,
    StationDisplacementInput,
)
from lunarops.classes.observation_factory import ensure_registered
from lunarops.config.context import RunContext


_AMPLITUDES = np.array(
    [
        [0.00352, 0.00123, 0.00080, 0.00032, 0.00187, 0.00112, 0.00063, 0.00003, 0.00082, 0.00044, 0.00037],
        [0.00144, 0.00035, 0.00035, 0.00008, 0.00053, 0.00049, 0.00018, 0.00009, 0.00012, 0.00005, 0.00006],
        [0.00086, 0.00023, 0.00023, 0.00006, 0.00029, 0.00028, 0.00010, 0.00007, 0.00004, 0.00002, 0.00001],
    ]
)
_PHASES = np.array(
    [
        [-64.7, -52.0, -96.2, -55.2, -58.8, -151.4, -65.6, -138.1, 8.4, 5.2, 2.1],
        [85.5, 114.5, 56.5, 113.6, 99.4, 19.1, 94.1, -10.4, -167.4, -170.0, -177.7],
        [109.5, 147.0, 92.7, 148.8, 50.5, -55.1, 36.4, -170.4, -15.0, 2.3, 5.2],
    ]
)

_FES2022B_APOLLO_AMPLITUDES = np.array(
    [
        [0.00139, 0.00161, 0.00017, 0.00046, 0.00559, 0.00377, 0.00176, 0.00072, 0.00016, 0.00008, 0.00007],
        [0.00071, 0.00020, 0.00016, 0.00005, 0.00227, 0.00150, 0.00071, 0.00029, 0.00007, 0.00003, 0.00002],
        [0.00162, 0.00080, 0.00038, 0.00023, 0.00075, 0.00046, 0.00023, 0.00010, 0.00002, 0.00001, 0.00001],
    ]
)
_FES2022B_APOLLO_PHASES = np.array(
    [
        [151.1, -133.1, -120.0, -130.4, 33.0, 20.0, 30.6, 13.2, -161.1, -168.9, -178.0],
        [161.6, 140.7, 126.9, 137.7, -140.7, -157.5, -143.6, -168.4, 128.9, 90.9, 18.6],
        [91.8, 107.3, 80.5, 105.3, 148.3, 147.2, 144.7, 152.8, 82.1, 53.9, 7.6],
    ]
)

# Generated independently with Orekit 13.1.7 OceanLoading, IERS 2010 UT1
# arguments (orekit-data 315cce51), and the APOLLO block from the FES2022b BLQ
# file with SHA-256 1eea03ce5fc69e8f2ae0f33014e075bd3da5af75f0d5b1c5e199ab36e9156b38.
_OREKIT_FES2022B_APOLLO_REFERENCE = (
    (
        "2016-01-01T00:00:00",
        (2.6671778134801040e-3, -1.2800550048212267e-3, -6.9261638871707410e-4),
        (6.9261638871707410e-4, 1.2800550048212267e-3, 2.6671778134801040e-3),
        (2.4398263247688800e-4, -1.6795203325826102e-3, 2.5202737338430340e-3),
    ),
    (
        "2016-01-01T06:00:00",
        (2.7841835861721200e-3, 9.3097545312787810e-4, -1.7057476787558112e-3),
        (1.7057476787558112e-3, -9.3097545312787810e-4, 2.7841835861721200e-3),
        (8.6556272848026390e-4, -3.2021222436673980e-3, 7.2469509388018720e-4),
    ),
    (
        "2016-01-01T12:00:00",
        (-3.0265039113960300e-3, -6.9101895034845280e-5, 1.5485572619211560e-3),
        (-1.5485572619211560e-3, 6.9101895034845280e-5, -3.0265039113960300e-3),
        (-7.8600039569691610e-4, 2.9063222618268803e-3, -1.5805141187016970e-3),
    ),
    (
        "2016-01-01T18:00:00",
        (-2.2980447528129066e-3, 5.5914279891444955e-5, 5.7767057562259300e-4),
        (-5.7767057562259300e-4, -5.5914279891444955e-5, -2.2980447528129066e-3),
        (-3.7311552964839520e-5, 1.9872596796984493e-3, -1.2912188815328786e-3),
    ),
)


def _row(values: np.ndarray) -> str:
    return " ".join(f"{value:.5f}" for value in values)


def _blq_text(
    *,
    station_name: str = "APOLLO",
    model: str = "FES2022b",
    amplitudes: np.ndarray = _AMPLITUDES,
    phases: np.ndarray = _PHASES,
) -> str:
    return "\n".join(
        [
            "$$ Ocean loading displacement",
            "$$ COLUMN ORDER: M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA",
            "$$ CMC: NO (corr.tide centre of mass)",
            f"$$ {model}: M2 S2 N2 K2 K1 O1",
            f"$$ {model}: P1 Q1 MF MM SSA",
            "$$ END HEADER",
            f"  {station_name}",
            f"$$ {station_name} RADI TANG lon/lat: 0.0000 0.0000 0.0000",
            *[_row(row) for row in amplitudes],
            *[_row(row) for row in phases],
            "$$ END TABLE",
            "",
        ]
    )


def _station_input(*, station_id: str | None = "APOLLO") -> StationDisplacementInput:
    return StationDisplacementInput(
        reference_position_itrf_m=np.asarray((6_378_137.0, 0.0, 0.0)),
        epoch_utc=Epoch.from_isot("2009-06-25T01:10:45", scale=TimeScale.UTC),
        station_id=station_id,
    )


def test_onsala_blq_catalog_parses_metadata_and_station_coefficients(tmp_path):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")

    catalog = OceanTidalLoadingCatalog(coefficient_file)
    coefficients = catalog.coefficients_for("7045")

    assert catalog.station_ids == ("APOLLO",)
    assert catalog.info.station_count == 1
    assert catalog.info.tidal_model == "FES2022b"
    assert catalog.info.center_of_mass_correction is False
    assert coefficients.station_id == "APOLLO"
    assert coefficients.source_station_name == "APOLLO"
    np.testing.assert_allclose(coefficients.amplitudes_m, _AMPLITUDES)
    np.testing.assert_allclose(coefficients.phases_deg, _PHASES)
    assert not coefficients.amplitudes_m.flags.writeable
    assert not coefficients.phases_deg.flags.writeable


def test_onsala_blq_catalog_rejects_malformed_and_duplicate_station_blocks(tmp_path):
    malformed = tmp_path / "malformed.blq"
    malformed.write_text(_blq_text().replace(_row(_AMPLITUDES[0]), "0.1 0.2"), encoding="utf-8")
    with pytest.raises(ValueError, match="11 values"):
        OceanTidalLoadingCatalog(malformed)

    invalid_value = tmp_path / "invalid-value.blq"
    invalid_value.write_text(
        _blq_text().replace("0.00123", "not-a-number", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid BLQ numeric value"):
        OceanTidalLoadingCatalog(invalid_value)

    incomplete = tmp_path / "incomplete.blq"
    incomplete.write_text(
        _blq_text().replace(f"{_row(_PHASES[-1])}\n$$ END TABLE", "$$ END TABLE"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected 6"):
        OceanTidalLoadingCatalog(incomplete)

    duplicate = tmp_path / "duplicate.blq"
    duplicate.write_text(
        _blq_text().replace(
            "$$ END TABLE",
            "  APOLLO\n"
            + "\n".join([*[_row(row) for row in _AMPLITUDES], *[_row(row) for row in _PHASES], "$$ END TABLE"]),
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate BLQ station"):
        OceanTidalLoadingCatalog(duplicate)

    bad_order = tmp_path / "bad-order.blq"
    bad_order.write_text(
        _blq_text().replace("M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA", "S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SSA", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="column order"):
        OceanTidalLoadingCatalog(bad_order)


def test_onsala_blq_catalog_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="BLQ file not found"):
        OceanTidalLoadingCatalog(tmp_path / "missing.blq")


def test_ocean_tidal_loading_passes_blq_unchanged_and_converts_usw_to_itrf(tmp_path, monkeypatch):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(coefficient_file))

    received = {}

    def hardisp(year, month, day, hour, minute, second, n, sample, amplitudes, phases):
        received["calendar"] = (year, month, day, hour, minute, second)
        received["n"] = n
        received["sample"] = sample
        received["amplitudes"] = np.array(amplitudes, copy=True)
        received["phases"] = np.array(phases, copy=True)
        return np.array([0.003]), np.array([0.002]), np.array([-0.001])

    import lunarops.classes.displacement.ocean_tidal_loading as ocean_tidal_loading

    monkeypatch.setattr(ocean_tidal_loading._iers2010, "hardisp", hardisp)
    result = model.evaluate(_station_input())

    assert received["calendar"] == (2009, 6, 25, 1, 10, 45)
    assert received["n"] == 1
    assert received["sample"] == 1.0
    np.testing.assert_allclose(received["amplitudes"], _AMPLITUDES)
    np.testing.assert_allclose(received["phases"], _PHASES)
    np.testing.assert_allclose(result.displacement_up_south_west_m, [0.003, 0.002, -0.001])
    np.testing.assert_allclose(result.displacement_enu_m, [0.001, -0.002, 0.003])
    np.testing.assert_allclose(result.displacement_itrf_m, [0.003, 0.001, -0.002])


@pytest.mark.parametrize(
    ("epoch_text", "expected_usw_m", "expected_enu_m", "expected_itrf_m"),
    _OREKIT_FES2022B_APOLLO_REFERENCE,
)
def test_ocean_tidal_loading_component_signs_match_independent_orekit_series(
    tmp_path,
    epoch_text,
    expected_usw_m,
    expected_enu_m,
    expected_itrf_m,
):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(
        _blq_text(
            amplitudes=_FES2022B_APOLLO_AMPLITUDES,
            phases=_FES2022B_APOLLO_PHASES,
        ),
        encoding="utf-8",
    )
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(coefficient_file))
    # WGS84 conversion of the BLQ header's lon/lat/height station position.
    data = StationDisplacementInput(
        reference_position_itrf_m=np.array(
            [
                -1_463_996.2265579700,
                -5_166_630.8462324440,
                3_435_016.8950683116,
            ]
        ),
        epoch_utc=Epoch.from_isot(epoch_text, scale=TimeScale.UTC),
        station_id="APOLLO",
    )

    result = model.evaluate(data)

    np.testing.assert_allclose(
        result.displacement_up_south_west_m,
        expected_usw_m,
        rtol=0.0,
        atol=5.0e-6,
    )
    np.testing.assert_allclose(result.displacement_enu_m, expected_enu_m, rtol=0.0, atol=5.0e-6)
    np.testing.assert_allclose(
        result.displacement_itrf_m,
        expected_itrf_m,
        rtol=0.0,
        atol=5.0e-6,
    )


def test_ocean_tidal_loading_requires_a_station_id_present_in_the_blq_file(tmp_path):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(coefficient_file))

    with pytest.raises(ValueError, match="station_id"):
        model.displacement_itrf_m(_station_input(station_id=None))
    with pytest.raises(KeyError, match="WETTZELL"):
        model.displacement_itrf_m(_station_input(station_id="WETTZELL"))


def test_ocean_tidal_loading_preserves_utc_leap_second_calendar_fields(tmp_path):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(coefficient_file))

    inputs = (
        (
            "2016-12-31T23:59:59",
            (2016, 12, 31, 23, 59, 59),
                (0.004732082132250071, -0.0005435922648757696, -0.0012600324116647243),
        ),
        ("2016-12-31T23:59:60", (2016, 12, 31, 23, 59, 60), None),
        (
            "2017-01-01T00:00:00",
            (2017, 1, 1, 0, 0, 0),
                (0.00473177433013916, -0.0005435359198600054, -0.0012598361354321241),
        ),
        (
            "2024-01-01T00:00:00",
            (2024, 1, 1, 0, 0, 0),
                (0.0042690373957157135, -0.00022759537387173623, -0.0016440704930573702),
        ),
    )
    for text, expected_calendar, expected_up_south_west_m in inputs:
        data = StationDisplacementInput(
            reference_position_itrf_m=np.array([6_378_137.0, 0.0, 0.0]),
            epoch_utc=Epoch.from_isot(text, scale=TimeScale.UTC),
            station_id="APOLLO",
        )
        assert model._utc_calendar_second(data.epoch_utc) == expected_calendar
        if expected_calendar[-1] == 60:
            with pytest.raises(ValueError, match="exact UTC leap-second label"):
                model.displacement_itrf_m(data)
        else:
            result = model.evaluate(data)
            assert expected_up_south_west_m is not None
            np.testing.assert_allclose(
                result.displacement_up_south_west_m,
                expected_up_south_west_m,
                rtol=0.0,
                atol=2.0e-9,
            )
            assert np.all(np.isfinite(result.displacement_itrf_m))


def test_ocean_tidal_loading_rejects_dates_outside_etutc_validity_range(
    tmp_path,
    monkeypatch,
):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(coefficient_file))
    data = StationDisplacementInput(
        reference_position_itrf_m=np.array([6_378_137.0, 0.0, 0.0]),
        epoch_utc=Epoch.from_isot("2027-07-01T00:00:00", scale=TimeScale.UTC),
        station_id="APOLLO",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("HARDISP must not receive an out-of-range date")

    import lunarops.classes.displacement.ocean_tidal_loading as ocean_tidal_loading

    monkeypatch.setattr(ocean_tidal_loading._iers2010, "hardisp", fail_if_called)
    with pytest.raises(ValueError, match="supports UTC epochs only"):
        model.displacement_itrf_m(data)


def test_ocean_tidal_loading_factory_checks_model_and_uses_explicit_file(tmp_path):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    ensure_registered()
    context = RunContext(working_dir=tmp_path)

    model = context.create_class(
        "stationDisplacement",
        {
            "type": "iers2010OceanTidalLoading",
            "coefficientFile": "fes2022b.txt",
            "model": "FES2022b",
        },
        cache=False,
    )
    assert isinstance(model, Iers2010OceanTidalLoading)

    with pytest.raises(ValueError, match="model mismatch"):
        context.create_class(
            "stationDisplacement",
            {
                "type": "iers2010OceanTidalLoading",
                "coefficientFile": "fes2022b.txt",
                "model": "FES2014b",
            },
            cache=False,
        )
