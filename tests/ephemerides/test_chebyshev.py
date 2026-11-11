"""ChebyshevEphemeris and the DAF reader on synthetic kernels, and on real kernels against a frozen reference."""

from __future__ import annotations

import erfa
import numpy as np
import pytest
from numpy.polynomial import chebyshev as cheb

from lunarops.classes.ephemerides.chebyshev import ChebyshevEphemeris, coverage_partition, split_epoch_seconds
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.daf import DafFile, chebyshev_layout
from tests.conftest import DATA_DIR, REPO_ROOT
from tests.support.daf_writer import SyntheticSegment, SyntheticType20Segment, write_daf

DAY = 86_400.0
J2000 = 2_451_545.0
INTERVAL = 8 * DAY
RECORDS = 10


def _random_segment(target, center, seed, *, count=7, records=RECORDS, init=0.0):
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(records, 3, count)) * np.logspace(5, -3, count)
    return SyntheticSegment(target, center, coefficients, init, INTERVAL)


def _constant_segment(target, center, value, start_day, end_day):
    records = round((end_day - start_day) * DAY / INTERVAL)
    coefficients = np.full((records, 3, 1), float(value))
    return SyntheticSegment(target, center, coefficients, start_day * DAY, INTERVAL)


def _expected(segment, t, order=2):
    """Reference state from numpy's Chebyshev routines, in m, m/s, m/s^2."""
    record = min(int(np.floor((t - segment.init_s) / segment.interval_s)), segment.coefficients.shape[0] - 1)
    radius = 0.5 * segment.interval_s
    mid = segment.init_s + (record + 0.5) * segment.interval_s
    tau = (t - mid) / radius
    coefficients = segment.coefficients[record]
    rows = [
        [cheb.chebval(tau, cheb.chebder(c, m)) / radius**m if m else cheb.chebval(tau, c) for c in coefficients]
        for m in range(order + 1)
    ]
    return 1000.0 * np.concatenate(rows)


def _directory(tmp_path, segments, *, extra=None):
    write_daf(tmp_path / "positions.bsp", segments)
    for name, more in (extra or {}).items():
        write_daf(tmp_path / name, more)
    write_daf(tmp_path / "attitude.bpc", [_random_segment(31008, 0, 99)], kind="pck")
    return tmp_path


def _open(directory):
    return ChebyshevEphemeris(directory, lunar_relativistic_scale_convention="alreadyScaled")


def _epoch(day):
    return Epoch(J2000, day, TimeScale.TDB)


@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_daf_reader_reads_summaries_and_record_directory(tmp_path, byte_order):
    segments = [_random_segment(301, 3, 1), _random_segment(3, 0, 2, count=4)]
    daf = DafFile(write_daf(tmp_path / "k.bsp", segments, byte_order=byte_order))

    assert daf.kind == "spk" and daf.byte_order == byte_order
    assert [(s.target, s.center, s.frame, s.data_type) for s in daf.segments] == [(301, 3, 1, 2), (3, 0, 1, 2)]
    layout = chebyshev_layout(daf, daf.segments[1])
    assert (layout.data_type, layout.record_count, layout.position_count, layout.velocity_count) == (2, RECORDS, 4, 0)
    assert layout.records.shape == (RECORDS, 14)
    assert layout.init_s == 0.0 and layout.interval_s == INTERVAL
    np.testing.assert_array_equal(layout.records[:, 2:].reshape(RECORDS, 3, 4), segments[1].coefficients)


def test_daf_reader_rejects_damaged_and_foreign_files(tmp_path):
    path = write_daf(tmp_path / "k.bsp", [_random_segment(10, 0, 1)])
    raw = bytearray(path.read_bytes())
    raw[702] = ord("\n")  # a text-mode transfer rewrites the line endings of the validation string
    (tmp_path / "damaged.bsp").write_bytes(bytes(raw))
    raw = bytearray(path.read_bytes())
    raw[0:8] = b"DAF/CK  "
    (tmp_path / "ck.bsp").write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="FTP validation"):
        DafFile(tmp_path / "damaged.bsp")
    with pytest.raises(ValueError, match="identification word"):
        DafFile(tmp_path / "ck.bsp")


def test_type2_states_match_numpy_chebyshev_derivatives_through_the_centre_chain(tmp_path):
    moon, emb, sun = _random_segment(301, 3, 1), _random_segment(3, 0, 2, count=11), _random_segment(10, 0, 3)
    ephemeris = _open(_directory(tmp_path, [moon, emb, sun]))
    days = np.random.default_rng(4).uniform(0.0, RECORDS * 8.0, 50)

    states = ephemeris.body_states_jd(("MOON", "SUN", "EARTH MOON BARYCENTER"), J2000, days, order=2)

    for index, day in enumerate(days):
        t = day * DAY
        expected_moon = _expected(moon, t) + _expected(emb, t)
        np.testing.assert_allclose(states[index, 0], expected_moon, rtol=1e-13, atol=1e-9)
        np.testing.assert_allclose(states[index, 1], _expected(sun, t), rtol=1e-13, atol=1e-9)
        np.testing.assert_allclose(states[index, 2], _expected(emb, t), rtol=1e-13, atol=1e-9)


def test_protocol_methods_return_si_states_and_accelerations(tmp_path):
    sun = _random_segment(10, 0, 3)
    ephemeris = _open(_directory(tmp_path, [sun]))
    epochs = [_epoch(1.25), _epoch(33.5)]

    state = ephemeris.body_state_bcrs("SUN", epochs[0])
    matrices = ephemeris.body_state_matrices_bcrs(("SUN",), epochs)
    accelerations = ephemeris.body_accelerations_bcrs("SUN", epochs)

    np.testing.assert_allclose(state.position_m, _expected(sun, 1.25 * DAY)[:3], rtol=1e-14)
    np.testing.assert_allclose(state.velocity_mps, _expected(sun, 1.25 * DAY)[3:6], rtol=1e-13)
    np.testing.assert_array_equal(ephemeris.body_position_bcrs("SUN", epochs[0]), state.position_m)
    np.testing.assert_array_equal(ephemeris.body_state_matrix_bcrs(("SUN",), epochs[1]), matrices[1])
    assert matrices.shape == (2, 1, 6) and accelerations.shape == (2, 3)
    assert accelerations.flags.c_contiguous  # consumed directly by the native RHS
    np.testing.assert_allclose(accelerations[1], _expected(sun, 33.5 * DAY)[6:9], rtol=1e-12)
    with pytest.raises(ValueError, match="TDB"):
        ephemeris.body_state_bcrs("SUN", Epoch(J2000, 0.0, TimeScale.TT))


def test_record_boundary_uses_the_later_record_and_the_segment_end_the_last(tmp_path):
    segment = SyntheticSegment(
        10, 0, np.arange(RECORDS, dtype=float)[:, None, None] * np.ones((1, 3, 1)), 0.0, INTERVAL
    )
    ephemeris = _open(_directory(tmp_path, [segment]))

    values = ephemeris.body_states_jd(("SUN",), J2000, [0.0, 8.0, 16.0 - 1e-9, 16.0, 80.0], order=0)[:, 0, 0]

    np.testing.assert_array_equal(values / 1000.0, [0.0, 1.0, 1.0, 2.0, 9.0])


def test_later_segments_and_later_kernels_take_precedence(tmp_path):
    base = _constant_segment(10, 0, 1.0, 0.0, 80.0)
    window = _constant_segment(10, 0, 2.0, 16.0, 40.0)
    earlier_contiguous = _constant_segment(4, 0, 3.0, 0.0, 40.0)
    later_contiguous = _constant_segment(4, 0, 4.0, 40.0, 80.0)
    override = _constant_segment(10, 0, 5.0, 48.0, 64.0)
    ephemeris = _open(
        _directory(tmp_path, [base, window, earlier_contiguous, later_contiguous], extra={"z_override.bsp": [override]})
    )
    days = [8.0, 16.0, 24.0, 40.0, 44.0, 48.0, 56.0, 64.0, 72.0]

    sun = ephemeris.body_states_jd(("SUN",), J2000, days, order=0)[:, 0, 0] / 1000.0
    mars = ephemeris.body_states_jd(("MARS BARYCENTER",), J2000, [39.0, 40.0, 41.0], order=0)[:, 0, 0] / 1000.0

    # The window wins inside and on both of its ends; the later kernel wins over both segments of the first.
    np.testing.assert_array_equal(sun, [1.0, 2.0, 2.0, 2.0, 1.0, 5.0, 5.0, 5.0, 1.0])
    np.testing.assert_array_equal(mars, [3.0, 4.0, 4.0])


def test_coverage_partition_merges_owners_and_marks_gaps():
    bounds, owners, priorities = coverage_partition([(0.0, 10.0, 0), (20.0, 30.0, 1), (5.0, 25.0, 2), (40.0, 40.0, 3)])

    np.testing.assert_array_equal(bounds, [0.0, 5.0, 25.0, 30.0])
    np.testing.assert_array_equal(owners, [0, 2, 1])
    np.testing.assert_array_equal(priorities, [0, 2, 1])
    _, owners, _ = coverage_partition([(0.0, 10.0, 0), (20.0, 30.0, 1)])
    np.testing.assert_array_equal(owners, [0, -1, 1])


def test_epochs_outside_coverage_or_in_a_gap_raise(tmp_path):
    ephemeris = _open(
        _directory(tmp_path, [_constant_segment(10, 0, 1.0, 0.0, 16.0), _constant_segment(10, 0, 2.0, 32.0, 48.0)])
    )

    with pytest.raises(ValueError, match=r"JD\(TDB\) 2451569\.000000000 is outside the SPK coverage of 'SUN'"):
        ephemeris.body_state_bcrs("SUN", _epoch(24.0))
    with pytest.raises(ValueError, match="outside the SPK coverage"):
        ephemeris.body_state_bcrs("SUN", _epoch(-1e-6))
    np.testing.assert_array_equal(ephemeris.body_states_jd(("SUN",), J2000, [16.0, 32.0], order=0)[:, 0, 0], [1e3, 2e3])


def test_results_do_not_depend_on_the_julian_date_split(tmp_path):
    ephemeris = _open(_directory(tmp_path, [_random_segment(10, 0, 5)]))
    # Dyadic epochs, so that every split below represents exactly the same epoch.
    days = np.random.default_rng(6).integers(0, 80 * 2**20, 200) / 2.0**20

    reference = ephemeris.body_states_jd(("SUN",), J2000, days, order=2)
    shifted = ephemeris.body_states_jd(("SUN",), J2000 + np.floor(days) - 0.5, days - np.floor(days) + 0.5, order=2)
    swapped = ephemeris.body_states_jd(("SUN",), days, J2000, order=2)
    a, f = split_epoch_seconds(J2000 + np.floor(days) - 0.5, days - np.floor(days) + 0.5)

    np.testing.assert_array_equal(shifted, reference)
    np.testing.assert_array_equal(swapped, reference)
    assert np.all(a % DAY == 0.0) and np.all(np.abs(f) < 2 * DAY)


def test_unknown_bodies_missing_records_and_unsupported_segments_raise_on_request(tmp_path):
    type21 = _random_segment(399, 3, 7)
    type21.data_type = 21
    ephemeris = _open(_directory(tmp_path, [_random_segment(4, 0, 1), _random_segment(3, 0, 2), type21]))

    with pytest.raises(KeyError, match=r"No SPK position record for 'MARS' \(NAIF 499\); try MARS BARYCENTER"):
        ephemeris.body_state_bcrs("MARS", _epoch(1.0))
    with pytest.raises(KeyError, match="Unknown ephemeris body name"):
        ephemeris.body_state_bcrs("KBO_RING_1", _epoch(1.0))
    with pytest.raises(ValueError, match="SPK segment of NAIF 399 relative to 3: SPK type 21 is not a supported"):
        ephemeris.body_state_bcrs("EARTH", _epoch(1.0))
    assert ephemeris.body_state_matrix_bcrs(("MARS BARYCENTER", "EARTH MOON BARYCENTER"), _epoch(1.0)).shape == (2, 6)


def test_directory_needs_position_and_orientation_kernels_and_close_releases_them(tmp_path):
    write_daf(tmp_path / "positions.bsp", [_random_segment(10, 0, 1)])
    with pytest.raises(ValueError, match=".bsp and .bpc"):
        _open(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        _open(tmp_path / "missing")
    write_daf(tmp_path / "attitude.bpc", [_random_segment(31008, 0, 2)], kind="pck")

    with _open(tmp_path) as ephemeris:
        assert ephemeris.body_state_bcrs("SUN", _epoch(1.0)).position_m.shape == (3,)
        assert ephemeris.source_path == tmp_path
        assert ephemeris.lunar_orientation.pa_to_lcrs_matrix(_epoch(1.0)).shape == (3, 3)
    with pytest.raises(ValueError, match="closed"):
        ephemeris.body_state_bcrs("SUN", _epoch(1.0))
    with pytest.raises(ValueError, match="closed"):
        ephemeris.lunar_orientation.pa_to_lcrs_matrix(_epoch(1.0))


def test_relative_states_cancel_the_shared_links_of_both_chains(tmp_path):
    moon, earth = _random_segment(301, 3, 21), _random_segment(399, 3, 22)
    emb = _random_segment(3, 0, 23, records=2)  # the shared link covers only the first 16 days
    ephemeris = _open(_directory(tmp_path, [moon, earth, emb, _random_segment(10, 0, 24)]))
    days = np.random.default_rng(25).uniform(0.0, 16.0, 20)

    relative = ephemeris.relative_states_jd("MOON", "EARTH", J2000, days, order=2)
    barycentric = ephemeris.body_states_jd(("MOON", "EARTH"), J2000, days, order=2)
    late = ephemeris.relative_states_jd("MOON", "EARTH", J2000, [40.0, 79.0], order=0)

    for row, day in zip(relative, days, strict=True):
        expected = _expected(moon, day * DAY) - _expected(earth, day * DAY)
        for block in (slice(0, 3), slice(3, 6), slice(6, 9)):  # relative to each vector's norm
            np.testing.assert_allclose(row[block], expected[block], atol=1e-14 * np.linalg.norm(expected[block]))
    np.testing.assert_allclose(relative, barycentric[:, 0] - barycentric[:, 1], rtol=1e-9, atol=1e-6)
    # Beyond the shared link's coverage the barycentric states fail but the relative state does not.
    np.testing.assert_allclose(late[1], (_expected(moon, 79 * DAY) - _expected(earth, 79 * DAY))[:3], rtol=1e-13)
    with pytest.raises(ValueError, match="outside the SPK coverage of 'MOON'"):
        ephemeris.body_states_jd(("MOON",), J2000, [40.0])
    sun_moon = ephemeris.relative_state_matrices("SUN", "MOON", [_epoch(day) for day in days[:3]])
    sun = ephemeris.body_states_jd(("SUN",), J2000, days[:3])[:, 0]
    np.testing.assert_allclose(sun_moon, sun - barycentric[:3, 0, :6], rtol=1e-12)
    np.testing.assert_array_equal(ephemeris.relative_states_jd("MOON", "MOON", J2000, days[:2]), np.zeros((2, 6)))


def test_type3_velocity_comes_from_its_own_expansion(tmp_path):
    rng = np.random.default_rng(12)
    coefficients = rng.normal(size=(RECORDS, 6, 6)) * np.logspace(4, -2, 6)
    segment = SyntheticSegment(10, 0, coefficients, 0.0, INTERVAL)
    ephemeris = _open(_directory(tmp_path, [segment]))
    days = rng.uniform(0.0, 80.0, 30)

    states = ephemeris.body_states_jd(("SUN",), J2000, days, order=2)[:, 0]

    radius = 0.5 * INTERVAL
    for row, day in zip(states, days, strict=True):
        record = min(int(day // 8.0), RECORDS - 1)
        tau = (day * DAY - (record + 0.5) * INTERVAL) / radius
        position, velocity = coefficients[record, :3], coefficients[record, 3:]
        expected = np.concatenate(
            [
                [cheb.chebval(tau, c) for c in position],
                [cheb.chebval(tau, c) for c in velocity],
                [cheb.chebval(tau, cheb.chebder(c)) / radius for c in velocity],
            ]
        )
        np.testing.assert_allclose(row, 1000.0 * expected, rtol=1e-13, atol=1e-9)


def _type20_segment(target, center, seed, *, scale=149_597_870.7, init_fraction=0.0):
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(5, 3, 6)) * np.logspace(-2, -8, 6)
    midpoints = rng.normal(size=(5, 3))
    return SyntheticType20Segment(target, center, coefficients, midpoints, scale, DAY, J2000, init_fraction, 4.0)


def test_type20_positions_integrate_the_velocity_expansion_from_the_midpoint(tmp_path):
    segment = _type20_segment(10, 0, 13)
    ephemeris = _open(_directory(tmp_path, [segment]))
    days = np.concatenate([np.random.default_rng(14).uniform(0.0, 20.0, 30), [2.0, 4.0, 20.0]])

    states = ephemeris.body_states_jd(("SUN",), J2000, days, order=2)[:, 0]

    radius = 2.0 * DAY
    rate = segment.scale / segment.time_scale  # velocity coefficients to km/s
    for row, day in zip(states, days, strict=True):
        record = min(int(day // 4.0), 4)
        tau = (day * DAY - (2 * record + 1) * radius) / radius
        a = segment.coefficients[record]
        expected = np.concatenate(
            [
                segment.midpoints[record] * segment.scale
                + [radius * rate * cheb.chebval(tau, cheb.chebint(c, lbnd=0.0)) for c in a],
                [rate * cheb.chebval(tau, c) for c in a],
                [rate * cheb.chebval(tau, cheb.chebder(c)) / radius for c in a],
            ]
        )
        np.testing.assert_allclose(row, 1000.0 * expected, rtol=1e-13, atol=1e-6)
    # At each midpoint the position is the stored midpoint value.
    midpoints = ephemeris.body_states_jd(("SUN",), J2000, 2.0 + 4.0 * np.arange(5), order=0)[:, 0]
    np.testing.assert_allclose(midpoints, 1000.0 * segment.scale * segment.midpoints, rtol=1e-15)


def test_type20_record_epochs_must_be_whole_seconds(tmp_path):
    ephemeris = _open(_directory(tmp_path, [_type20_segment(10, 0, 15, init_fraction=1e-7)]))

    with pytest.raises(ValueError, match=r"NAIF 10 relative to 0: Type 20 record epochs are not whole TDB seconds"):
        ephemeris.body_state_bcrs("SUN", _epoch(1.0))


@pytest.mark.parametrize("data_type", [2, 20])
def test_lunar_orientation_uses_the_pck_euler_angles(tmp_path, data_type):
    if data_type == 2:
        angles = _random_segment(31008, 0, 16)
        angles.coefficients *= 1e-5  # radians
    else:
        angles = _type20_segment(31008, 0, 16, scale=1.0)
    write_daf(tmp_path / "positions.bsp", [_random_segment(10, 0, 1)])
    write_daf(tmp_path / "attitude.bpc", [angles], kind="pck")
    ephemeris = _open(tmp_path)
    days = np.random.default_rng(17).uniform(0.0, 20.0, 20)

    euler = ephemeris.lunar_euler_jd(J2000, days, order=2)

    if data_type == 2:
        expected = np.asarray([_expected(angles, day * DAY) / 1000.0 for day in days])
    else:
        # The same records read as an SPK segment give the reference (km scaled to m by 1000).
        reference = tmp_path / "reference"
        reference.mkdir()
        _directory(reference, [_type20_segment(10, 0, 16, scale=1.0)])
        expected = _open(reference).body_states_jd(("SUN",), J2000, days, order=2)[:, 0] / 1000.0
    np.testing.assert_allclose(euler, expected, rtol=1e-13, atol=1e-20)
    phi, theta, psi = euler[3, :3]
    matrix = (erfa.rz(psi, np.eye(3)) @ erfa.rx(theta, np.eye(3)) @ erfa.rz(phi, np.eye(3))).T
    np.testing.assert_array_equal(ephemeris.lunar_orientation.pa_to_lcrs_matrix(_epoch(days[3])), matrix)
    np.testing.assert_array_equal(
        ephemeris.lunar_orientation.pa_to_lcrs_matrices([_epoch(day) for day in days])[3], matrix
    )
    with pytest.raises(ValueError, match="outside the PCK coverage of the lunar orientation"):
        ephemeris.lunar_orientation.pa_to_lcrs_matrix(_epoch(100.0))


def test_orientation_kernels_need_one_target_relative_to_icrf(tmp_path):
    write_daf(tmp_path / "positions.bsp", [_random_segment(10, 0, 1)])
    write_daf(tmp_path / "attitude.bpc", [_random_segment(31008, 0, 2), _random_segment(31009, 0, 3)], kind="pck")
    with pytest.raises(ValueError, match=r"Expected one lunar orientation target, found \[31008, 31009\]"):
        _open(tmp_path)
    off_frame = _random_segment(31008, 0, 2)
    off_frame.frame = 17
    write_daf(tmp_path / "attitude.bpc", [off_frame], kind="pck")
    with pytest.raises(ValueError, match="relative to ICRF frame 1, found \\[17\\]"):
        _open(tmp_path)


DATA_KERNELS = REPO_ROOT.parent / "data/kernels"
# Independent reference: CALCEPH 5.0.0 (calcephpy) states (order 2) and lunar Euler angles (order 2) at 40
# random epochs per kernel set, frozen on 2026-11-11 before the CALCEPH backend was removed.
REFERENCE = np.load(DATA_DIR / "ephemeris_reference.npz")


@pytest.mark.requires_data
@pytest.mark.parametrize("name", ["de440", "inpop21a", "epm2021"])
def test_real_kernels_agree_with_the_frozen_calceph_reference(name):
    directory = DATA_KERNELS / name
    if not directory.is_dir():
        pytest.skip(f"{directory} is not available")
    ephemeris = ChebyshevEphemeris(directory, lunar_relativistic_scale_convention="alreadyScaled")
    jd1, jd2 = REFERENCE[f"{name}_jd1"], REFERENCE[f"{name}_jd2"]
    bodies = tuple(str(body) for body in REFERENCE[f"{name}_bodies"])

    states = ephemeris.body_states_jd(bodies, jd1, jd2, order=2)
    euler = ephemeris.lunar_euler_jd(jd1, jd2, order=2)

    for index in range(len(bodies)):
        reference = REFERENCE[f"{name}_states"][:, index]
        ulp = np.spacing(np.linalg.norm(reference[:, :3], axis=1))
        assert np.max(np.linalg.norm(states[:, index, :3] - reference[:, :3], axis=1) / ulp) <= 4.0
        _assert_derivatives_close(states[:, index], reference)
    reference = REFERENCE[f"{name}_euler"]
    # Angles cross zero: compare in ulp of each angle's largest magnitude.
    assert np.max(np.abs(euler[:, :3] - reference[:, :3]) / np.spacing(np.abs(reference[:, :3]).max(axis=0))) <= 4.0
    _assert_derivatives_close(euler, reference)


def _assert_derivatives_close(ours, reference):
    for columns, limit in ((slice(3, 6), 1e-13), (slice(6, 9), 1e-11)):
        error = np.linalg.norm(ours[:, columns] - reference[:, columns], axis=1)
        assert np.max(error) <= limit * np.max(np.linalg.norm(reference[:, columns], axis=1))
