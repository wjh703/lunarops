"""LunarOrbitPropagation with the variational equations: the partials table, its content and its checks."""

import multiprocessing

import numpy as np
import pytest

from lunarops.classes.dynamics.assembly import build_lunar_propagation_problem
from lunarops.classes.dynamics.parameters import DYNAMICAL_PARAMETER_NAMES, INITIAL_VALUE_NAMES
from lunarops.fileio.lunar_partials import (
    STATE_NAMES,
    LunarDynamicsPartials,
    read_lunar_dynamics_partials,
    write_lunar_dynamics_partials,
)
from lunarops.fileio.numeric_table import read_numeric_table, write_numeric_table
from lunarops.fileio.structured_reports import read_lunar_orbit_metadata
from lunarops.programs import lunar_orbit
from lunarops.programs.registry import get_program
from tests.dynamics.test_parameters import _full_config
from tests.support.dynamics import AnalyticRunContext, orbit_program_config

# Characteristic sizes of the state components (position, velocity, Euler angle, angular velocity) that make
# the partials of different components comparable.
STATE_SCALES = np.r_[[3.84e8] * 3, [1.0e3] * 3, [1.0] * 3, [2.7e-6] * 6]


def _resolve(config):
    return get_program("LunarOrbitPropagation")[0].schema.resolve(config)


def _with_partials(config, parameters=None, step=120.0):
    return _resolve(
        {
            **config,
            "variationalEquations": {"parameters": parameters, "outputStepSeconds": step},
            "outputFilePartials": "partials.dat",
        }
    )


def _node_states(problem, offsets):
    trajectory = problem.integrator.integrate(
        problem.dynamics,
        problem.initial_epoch,
        problem.initial_vector,
        problem.duration_s,
        backward_duration_s=problem.backward_duration_s,
        seed_duration_s=problem.seed_duration_s,
        external_history_provider=problem.external_history,
    )
    return np.asarray([trajectory.vector(problem.initial_epoch.shifted(float(offset))) for offset in offsets])


def test_the_program_writes_the_partials_of_every_quantity(tmp_path):
    config, context = _full_config(tmp_path)
    paths = lunar_orbit.lunar_orbit_propagation(_with_partials(config), context)
    assert [path.name for path in paths] == ["orbit.dat", "accel.dat", "meta.txt", "rotation.dat", "partials.dat"]
    table = read_lunar_dynamics_partials(paths[-1])

    np.testing.assert_array_equal(table.offsets_s, np.arange(0.0, 1201.0, 120.0))
    assert table.states.shape == (11, 15) and table.partials.shape == (11, 15, 26)
    assert table.state_names == STATE_NAMES
    assert table.quantity_names == INITIAL_VALUE_NAMES + DYNAMICAL_PARAMETER_NAMES
    assert table.quantity_units[:6] == ("m",) * 3 + ("m/s",) * 3 and table.quantity_units[15] == "m3/s2"
    assert table.producer == "variational"
    assert (table.epoch_tdb_jd1, table.epoch_tdb_jd2) == (config["initialTdbJd1"], config["initialTdbJd2"])
    assert table.metadata["stepSeconds"] == 60 and table.metadata["outputStepSeconds"] == 120.0
    problem = build_lunar_propagation_problem(config, context)
    assert table.metadata["modelFingerprint"] == lunar_orbit.system_digest(
        problem.dynamics, problem.integrator.settings
    )
    assert set(table.metadata["parameterIncrements"]) == set(DYNAMICAL_PARAMETER_NAMES)
    # W(t0) = [I | 0]; every column responds by the end of the arc.
    np.testing.assert_array_equal(table.partials[0], np.eye(15, 26))
    assert np.all(np.abs(table.partials[-1]).max(axis=0) > 0)

    _, columns, _, metadata = read_numeric_table(paths[-1], "lunarDynamicsPartials")
    assert columns[:17] == ("offset_tdb_s", *STATE_NAMES, "d_r_x_d_r0_x")
    assert columns[16 + 26] == "d_r_y_d_r0_x" and columns[-1] == "d_wc_z_d_fc"
    assert metadata["quantityNames"] == list(table.quantity_names) and metadata["timeScale"] == "TDB"
    assert read_lunar_orbit_metadata(paths[2])["partialsFile"] == str(paths[-1])


def test_the_partials_leave_the_state_unchanged_bit_for_bit(tmp_path):
    config, context = _full_config(tmp_path)
    plain = {path.name: path.read_bytes() for path in lunar_orbit.lunar_orbit_propagation(config, context)}
    assert read_lunar_orbit_metadata(tmp_path / "meta.txt")["partialsFile"] is None
    paths = lunar_orbit.lunar_orbit_propagation(_with_partials(config), context)
    for path in paths[:-1]:
        if path.name != "meta.txt":
            assert path.read_bytes() == plain[path.name], path.name
    table = read_lunar_dynamics_partials(paths[-1])
    _, orbit_columns, orbit, _ = read_numeric_table(paths[0], "lunarOrbit")
    _, _, rotation, _ = read_numeric_table(paths[3], "lunarRotation")
    rows = np.searchsorted(orbit[:, 0], table.offsets_s)
    np.testing.assert_array_equal(orbit[rows, 0], table.offsets_s)
    first = orbit_columns.index("relative_x_m")
    np.testing.assert_array_equal(table.states[:, :6], orbit[rows, first : first + 6])
    np.testing.assert_array_equal(table.states[:, 6:], rotation[rows, 1:10])


@pytest.mark.parametrize(
    ("name", "increment", "tolerance"),
    [
        # Initial values: the orbit and the mantle spin.
        ("v0_y", 1.0e-2, 1.0e-10),
        ("wm0_z", 1.0e-11, 3.0e-10),
        # Dynamical parameters: the orbit and the lunar figure.
        ("gm_emb", 4.0e9, 3.0e-8),
        ("beta", 1.0e-6, 1.0e-5),
    ],
)
def test_partials_agree_with_global_central_differences(tmp_path, name, increment, tolerance):
    # Each column is compared with a central difference of two whole propagations with the quantity changed in
    # the parameters of the problem. The error is measured in units of the state scales, relative to the largest
    # scaled response of the column. Over the 20-minute arc the differences are limited by the rounding of the
    # state (they improve tenfold for a tenfold increment), not by the partials: about 8e-12 (v0_y), 3e-11
    # (wm0_z), 3e-9 (gm_emb) and 7e-7 (beta); the tolerances leave a factor of ten.
    config, context = _full_config(tmp_path)
    table = read_lunar_dynamics_partials(lunar_orbit.lunar_orbit_propagation(_with_partials(config), context)[-1])
    problem = build_lunar_propagation_problem(config, context)
    nominal = problem.parameters[name]
    plus, minus = (
        _node_states(
            build_lunar_propagation_problem(
                config, context, problem.parameters.with_value(name, nominal + sign * increment)
            ),
            table.offsets_s,
        )
        for sign in (1.0, -1.0)
    )
    difference = (plus - minus) / (2 * increment) / STATE_SCALES
    error = np.abs(table.column(name) / STATE_SCALES - difference).max() / np.abs(difference).max()
    assert error < tolerance


def test_an_orbit_without_rotation_has_six_state_components(tmp_path):
    config, context = orbit_program_config(tmp_path, eih=False)
    config = {**config, "earthTide": {"tauOrbDays": [0.002, -0.0005, -0.001]}}
    table = read_lunar_dynamics_partials(lunar_orbit.lunar_orbit_propagation(_with_partials(config), context)[-1])
    assert table.state_names == STATE_NAMES[:6] and table.state_units == ("m",) * 3 + ("m/s",) * 3
    assert table.quantity_names == (*INITIAL_VALUE_NAMES[:6], "gm_emb", "tau_r1", "tau_r2")
    assert table.partials.shape == (11, 6, 9)


def test_an_empty_parameter_list_gives_the_initial_state_partials_only(tmp_path):
    config, context = _full_config(tmp_path)
    paths = lunar_orbit.lunar_orbit_propagation(_with_partials(config, [], step=600.0), context)
    table = read_lunar_dynamics_partials(paths[-1])
    assert table.quantity_names == INITIAL_VALUE_NAMES and table.metadata["parameterIncrements"] == {}
    np.testing.assert_array_equal(table.offsets_s, [0.0, 600.0, 1200.0])


def test_a_parallel_backward_arc_writes_the_partials_of_a_sequential_one(tmp_path):
    config, _ = _full_config(tmp_path, backwardDurationSeconds=600.0)
    config = _with_partials(config, ["gm_emb", "tau_m", "c33", "fc"])
    files = {}
    for parallel in (False, True):
        with AnalyticRunContext(working_dir=tmp_path) as context:
            paths = lunar_orbit.lunar_orbit_propagation({**config, "parallelArcs": parallel}, context)
        files[parallel] = paths[-1].read_bytes()
        paths[-1].unlink()
    assert multiprocessing.active_children() == []
    assert files[True] == files[False]
    (tmp_path / "partials.dat").write_bytes(files[True])
    table = read_lunar_dynamics_partials(tmp_path / "partials.dat")
    np.testing.assert_array_equal(table.offsets_s, np.arange(-600.0, 1201.0, 120.0))
    np.testing.assert_array_equal(table.partials[5], np.eye(15, 19))
    assert np.all(np.abs(table.partials[0, :, 15:]).max(axis=0) > 0)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"variationalEquations": {"outputStepSeconds": 90.0}, "outputFilePartials": "partials.dat"},
            r"outputStepSeconds \(90 s\) must be a whole multiple of stepSeconds \(60 s\)",
        ),
        (
            {"variationalEquations": {"outputStepSeconds": 30.0}, "outputFilePartials": "partials.dat"},
            "must be a whole multiple of stepSeconds",
        ),
        ({"variationalEquations": {}}, "variationalEquations requires outputFilePartials"),
        ({"outputFilePartials": "partials.dat"}, "outputFilePartials is given, but variationalEquations is null"),
        (
            {"variationalEquations": {"parameters": ["k2m"]}, "outputFilePartials": "partials.dat"},
            "which are not dynamical parameters",
        ),
    ],
)
def test_invalid_variational_settings_are_rejected(tmp_path, updates, message):
    config, _ = orbit_program_config(tmp_path, eih=False)
    with pytest.raises(ValueError, match=message):
        _resolve({**config, **updates})


def test_a_parameter_the_model_does_not_have_is_rejected(tmp_path):
    config, context = orbit_program_config(tmp_path, eih=False)
    with pytest.raises(ValueError, match="'fc' is not a dynamical parameter of this model"):
        lunar_orbit.lunar_orbit_propagation(_with_partials(config, ["fc"]), context)


def test_the_partials_file_round_trips_and_checks_its_layout(tmp_path):
    rng = np.random.default_rng(1)
    table = LunarDynamicsPartials(
        2451545.0,
        0.25,
        [-60.0, 0.0, 60.0],
        rng.normal(size=(3, 12)),
        rng.normal(size=(3, 12, 13)),
        (*INITIAL_VALUE_NAMES[:12], "beta"),
        (*("m",) * 3, *("m/s",) * 3, *("rad",) * 3, *("rad/s",) * 3, "1"),
        "finiteDifference",
        {"incrementScale": 2.0},
    )
    back = read_lunar_dynamics_partials(write_lunar_dynamics_partials(tmp_path / "partials.dat", table))
    for name in ("offsets_s", "states", "partials"):
        np.testing.assert_array_equal(getattr(back, name), getattr(table, name))
    assert back.quantity_names == table.quantity_names and back.quantity_units == table.quantity_units
    assert (back.producer, dict(back.metadata)) == ("finiteDifference", {"incrementScale": 2.0})
    assert back.state_names == STATE_NAMES[:12]
    np.testing.assert_array_equal(back.column("beta"), table.partials[:, :, 12])

    def build(**changes):
        fields = {name: getattr(table, name) for name in ("offsets_s", "states", "partials", "quantity_names")}
        fields.update(changes)
        return LunarDynamicsPartials(
            2451545.0,
            0.0,
            fields["offsets_s"],
            fields["states"],
            fields["partials"],
            fields["quantity_names"],
            table.quantity_units,
            "variational",
            changes.get("metadata", {}),
        )

    with pytest.raises(ValueError, match="increase"):
        build(offsets_s=[0.0, 0.0, 60.0])
    with pytest.raises(ValueError, match="6, 12 or 15"):
        build(states=table.states[:, :7], partials=table.partials[:, :7])
    with pytest.raises(ValueError, match="unique"):
        build(quantity_names=(*INITIAL_VALUE_NAMES[:11], "beta", "beta"))
    with pytest.raises(ValueError, match="layout keys"):
        build(metadata={"producer": "other"})
    other = write_numeric_table(tmp_path / "other.dat", "lunarOrbit", ("offset_tdb_s",), [[0.0]])
    with pytest.raises(ValueError, match="lunarDynamicsPartials"):
        read_lunar_dynamics_partials(other)
