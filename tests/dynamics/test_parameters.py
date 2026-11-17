"""Initial values and dynamical parameters of the lunar dynamics: file, application and sibling models."""

import pickle

import numpy as np
import pytest

from lunarops.classes.dynamics.arc_process import system_digest
from lunarops.classes.dynamics.assembly import build_lunar_propagation_problem
from lunarops.classes.dynamics.parameters import (
    DYNAMICAL_PARAMETER_NAMES,
    ESTIMABLE_DYNAMICAL_PARAMETER_NAMES,
    INITIAL_VALUE_NAMES,
    OPTIONAL_DYNAMICAL_PARAMETER_NAMES,
    PARAMETER_DEFINITIONS,
    PARAMETER_NAMES,
    LunarDynamicsParameters,
    apply_to_config,
    batch_with_earth_moon_gm,
    build_sibling_dynamics,
    config_parameters,
    inertia_from_ratios,
    moment_ratios,
    parameter_definition,
    read_lunar_dynamics_parameters,
    write_lunar_dynamics_parameters,
)
from lunarops.programs import lunar_orbit  # noqa: F401 -- registers LunarOrbitPropagation
from lunarops.programs.registry import get_program
from tests.support.dynamics import EPOCH, orbit_program_config

DE440_MOMENTS = (0.3927622575169338, 0.3928517809190465, 0.3930102335781251)


def _full_config(tmp_path, **updates):
    """A resolved configuration in which all 26 quantities are defined: tide, deforming Moon with a core, degree-3 field."""
    config, context = orbit_program_config(tmp_path, eih=True)
    for name, radius, c20 in (("earth", 6378136.6, -4.8e-4), ("moon", 1738000.0, -9e-5)):
        (tmp_path / f"{name}.gfc").write_text(
            "begin_of_head\nproduct_type gravity_field\nmodelname parameter_test\n"
            f"gravity_constant 4.9e12\nradius {radius}\nmax_degree 3\n"
            "norm fully_normalized\ntide_system zero_tide\nend_of_head\n"
            f"gfc 0 0 1 0\ngfc 2 0 {c20} 0\ngfc 2 1 0 0\ngfc 2 2 3.4e-5 0\n"
            "gfc 3 0 -3.2e-6 0\ngfc 3 1 2.6e-5 5.9e-6\ngfc 3 2 1.4e-5 4.9e-6\ngfc 3 3 1.2e-5 -1.8e-6\n"
        )
    config.update(
        gravityFields={name: {"path": f"{name}.gfc"} for name in ("earth", "moon")},
        moonFigurePartners=["EARTH", "SUN"],
        # Delays inside the 600 s seed arc of the deformation.
        earthTide={"tauOrbDays": [0.002, -0.0005, -0.001]},
        outputFileRotation="rotation.dat",
        lunarRotation={
            "inertiaMoments": list(DE440_MOMENTS),
            "radiusM": 1738000.0,
            "initialEulerAnglesRad": [0.1, 0.4, 1.2],
            "initialAngularVelocityRadS": [1e-7, -2e-7, 2.7e-6],
            "initialCoreAngularVelocityRadS": [2e-7, 0.0, 2.6e-6],
            "sources": ["EARTH", "SUN"],
            "gravityTorque": "figure",
            "earthFigureTorque": True,
            "geodeticPrecession": True,
            "core": {"polarMomentFraction": 0.0004, "flattening": 0.0002, "viscosityPerS": 4.166805729506098e-14},
            "deformation": {
                "loveK2": 0.024,
                "delayDays": 0.002,
                "meanMotionRadS": 2.6616995e-06,
                "seedModel": "rigidBackward",
                "seedDurationSeconds": 600.0,
            },
        },
    )
    config.update(updates)
    return get_program("LunarOrbitPropagation")[0].schema.resolve(config), context


def _propagate(problem):
    trajectory = problem.integrator.integrate(
        problem.dynamics,
        problem.initial_epoch,
        problem.initial_vector,
        problem.duration_s,
        seed_duration_s=problem.seed_duration_s,
        external_history_provider=problem.external_history,
    )
    return trajectory.vector(problem.initial_epoch.shifted(problem.duration_s))


def _digest(problem):
    return system_digest(problem.dynamics, problem.integrator.settings)


def test_the_definitions_name_fifteen_initial_values_eleven_parameters_and_three_optional_delays():
    assert len(PARAMETER_NAMES) == len(set(PARAMETER_NAMES)) == 29
    assert len(INITIAL_VALUE_NAMES) == 15 and len(DYNAMICAL_PARAMETER_NAMES) == 11
    assert OPTIONAL_DYNAMICAL_PARAMETER_NAMES == ("tau_o0", "tau_o1", "tau_o2")
    assert ESTIMABLE_DYNAMICAL_PARAMETER_NAMES == DYNAMICAL_PARAMETER_NAMES + OPTIONAL_DYNAMICAL_PARAMETER_NAMES
    assert PARAMETER_NAMES == INITIAL_VALUE_NAMES + ESTIMABLE_DYNAMICAL_PARAMETER_NAMES
    assert all(definition.step > 0 and definition.unit for definition in PARAMETER_DEFINITIONS)
    with pytest.raises(KeyError, match="Unknown lunar dynamics parameter"):
        parameter_definition("k22")


def test_moment_ratios_invert_exactly_for_the_de440_moments():
    j2, beta, gamma = moment_ratios(DE440_MOMENTS)
    assert inertia_from_ratios(j2, beta, gamma) == DE440_MOMENTS
    # The DE440 header constants, to the rounding of the configured moments.
    assert j2 == pytest.approx(2.0321436013500001e-4, rel=1e-12)
    assert beta == pytest.approx(6.3122040737914229e-4, rel=1e-12)
    assert gamma == pytest.approx(2.2778898477433167e-4, rel=1e-12)
    with pytest.raises(ValueError, match="0 < gamma < beta < 1"):
        inertia_from_ratios(j2, gamma, beta)


def test_a_parameter_set_keeps_definition_order_and_is_immutable():
    parameters = LunarDynamicsParameters({"fc": 2e-4, "r0_x": 1.0}, 2451545.0)
    assert parameters.names == ("r0_x", "fc")
    changed = parameters.with_value("tau_m", 0.1)
    assert changed.names == ("r0_x", "tau_m", "fc") and "tau_m" not in parameters
    assert changed.vector(("fc", "tau_m")).tolist() == [2e-4, 0.1]
    assert pickle.loads(pickle.dumps(changed)) == changed
    with pytest.raises(TypeError):
        parameters.values["fc"] = 1.0
    with pytest.raises(KeyError):
        LunarDynamicsParameters({"k2m": 0.024}, 2451545.0)
    with pytest.raises(ValueError, match="finite"):
        LunarDynamicsParameters({"fc": float("nan")}, 2451545.0)


def test_the_file_reads_back_bit_for_bit_and_rejects_wrong_content(tmp_path):
    values = {name: (index + 1) * np.pi * 10.0 ** (index - 13) for index, name in enumerate(PARAMETER_NAMES)}
    parameters = LunarDynamicsParameters(values, 2433282.5, 0.25, "round trip")
    path = write_lunar_dynamics_parameters(parameters, tmp_path / "parameters.txt")
    assert read_lunar_dynamics_parameters(path) == parameters
    text = path.read_text()
    assert text.startswith("lunarops lunarDynamicsParameters\n")

    path.write_text(text.replace("unit: m\n", "unit: km\n", 1))
    with pytest.raises(ValueError, match="must be given in 'm'"):
        read_lunar_dynamics_parameters(path)
    path.write_text(text.replace(f"parameterCount: {len(PARAMETER_NAMES)}", "parameterCount: 25"))
    with pytest.raises(ValueError, match="parameterCount"):
        read_lunar_dynamics_parameters(path)
    path.write_text(text.replace("name: r0_y", "name: r0_x"))
    with pytest.raises(ValueError, match="twice"):
        read_lunar_dynamics_parameters(path)


def test_a_file_of_the_nominal_values_reproduces_the_model_bit_for_bit(tmp_path):
    config, context = _full_config(tmp_path)
    nominal = build_lunar_propagation_problem(config, context)
    assert nominal.parameters.names == PARAMETER_NAMES
    write_lunar_dynamics_parameters(nominal.parameters, tmp_path / "parameters.txt")

    same = build_lunar_propagation_problem({**config, "inputFileLunarDynamicsParameters": "parameters.txt"}, context)

    assert _digest(same) == _digest(nominal)
    assert same.parameters == nominal.parameters
    np.testing.assert_array_equal(same.initial_vector, nominal.initial_vector)
    np.testing.assert_array_equal(_propagate(same), _propagate(nominal))


def test_a_partial_file_changes_only_the_listed_quantities(tmp_path):
    config, context = _full_config(tmp_path)
    nominal = build_lunar_propagation_problem(config, context)
    listed = {"r0_y": nominal.parameters["r0_y"] + 3.0, "psi0": 1.25, "wc0_z": 2.5e-6, "tau_r2": 1e-3, "s32": 5e-6}
    write_lunar_dynamics_parameters(LunarDynamicsParameters(listed, EPOCH.jd1), tmp_path / "parameters.txt")

    changed = build_lunar_propagation_problem({**config, "inputFileLunarDynamicsParameters": "parameters.txt"}, context)

    for name in PARAMETER_NAMES:
        assert changed.parameters[name] == listed.get(name, nominal.parameters[name]), name
    expected = nominal.initial_vector.copy()
    expected[[1, 8, 14]] = listed["r0_y"], 1.25, 2.5e-6
    np.testing.assert_array_equal(changed.initial_vector, expected)
    tide = changed.dynamics.force_group.active_models[-1].tide_model
    assert tide.tide_parameters.tau_rot_days[2] == 1e-3
    assert tide.tide_parameters.tau_rot_days[:2] == tuple(config["earthTide"]["tauRotDays"][:2])
    field = changed.dynamics.rotation.deformation.field
    assert field.coefficients.coeffs[1, 3, 2] == 5e-6
    assert field.coefficients.source_metadata["parameterOverrides"] == {"s32": 5e-6}
    assert _digest(changed) != _digest(nominal)
    assert np.linalg.norm(_propagate(changed)[:3] - _propagate(nominal)[:3]) > 1.0


def test_explicit_parameters_take_precedence_over_the_file(tmp_path):
    config, context = _full_config(tmp_path)
    write_lunar_dynamics_parameters(LunarDynamicsParameters({"fc": 3e-4}, EPOCH.jd1), tmp_path / "parameters.txt")
    config = {**config, "inputFileLunarDynamicsParameters": "parameters.txt"}

    assert build_lunar_propagation_problem(config, context).parameters["fc"] == 3e-4
    explicit = LunarDynamicsParameters({"fc": 4e-4}, EPOCH.jd1)
    assert build_lunar_propagation_problem(config, context, explicit).parameters["fc"] == 4e-4


def test_gm_emb_scales_every_earth_and_moon_gm_and_keeps_their_ratio(tmp_path):
    config, context = _full_config(tmp_path)
    nominal = build_lunar_propagation_problem(config, context)
    scale = 1.0 + 1e-6
    changed = build_lunar_propagation_problem(
        config, context, LunarDynamicsParameters({"gm_emb": nominal.parameters["gm_emb"] * scale}, EPOCH.jd1)
    )

    def gms(problem):
        dynamics = problem.dynamics
        tide = dynamics.force_group.active_models[-1].tide_model
        rotation = dynamics.rotation
        fields = {
            type(model).__name__: model.field for model in dynamics.force_group.active_models if hasattr(model, "field")
        }
        earth = np.array(
            [
                dynamics.earth_body.gravitational_parameter_m3_s2,
                dynamics.gravitational_parameters_m3_s2[0],
                tide.earth_gravitational_parameter_m3_s2,
                rotation.deformation.earth_mu,
                fields["EarthFigureForce"].gm_m3_s2,
                dynamics.epoch_batch((EPOCH,)).gravitational_parameters_m3_s2[0, 0],
                problem.external_history._earth_gm,
            ]
        )
        moon = np.array(
            [
                dynamics.moon_body.gravitational_parameter_m3_s2,
                dynamics.gravitational_parameters_m3_s2[1],
                tide.moon_gravitational_parameter_m3_s2,
                tide.tide_raiser_gm["MOON"],
                rotation.deformation.moon_mu,
                fields["MoonFigureForce"].gm_m3_s2,
                rotation.deformation.field.gm_m3_s2,
                dynamics.epoch_batch((EPOCH,)).gravitational_parameters_m3_s2[0, 1],
                problem.external_history._moon_gm,
            ]
        )
        return earth, moon, dynamics.gravitational_parameters_m3_s2[2:]

    earth0, moon0, others0 = gms(nominal)
    earth1, moon1, others1 = gms(changed)
    assert len(set(earth0)) == 1 and len(set(moon0)) == 1
    np.testing.assert_allclose(earth1 / earth0, scale, rtol=4e-16)
    np.testing.assert_allclose(moon1 / moon0, scale, rtol=4e-16)
    assert len(set(earth1)) == 1 and len(set(moon1)) == 1
    assert earth1[0] / moon1[0] == pytest.approx(earth0[0] / moon0[0], rel=4e-16)
    np.testing.assert_array_equal(others1, others0)
    assert changed.parameters["gm_emb"] == pytest.approx(nominal.parameters["gm_emb"] * scale, rel=4e-16)
    with pytest.raises(ValueError, match="positive"):
        build_lunar_propagation_problem(config, context, LunarDynamicsParameters({"gm_emb": -1.0}, EPOCH.jd1))


@pytest.mark.parametrize("name", ["beta", "gamma"])
def test_a_moment_ratio_rebuilds_the_moments_and_keeps_kv_over_ct(tmp_path, name):
    config, context = _full_config(tmp_path)
    nominal = build_lunar_propagation_problem(config, context)
    value = nominal.parameters[name] * (1 + 1e-3)
    changed = build_lunar_propagation_problem(config, context, LunarDynamicsParameters({name: value}, EPOCH.jd1))

    rotation0, rotation1 = nominal.dynamics.rotation, changed.dynamics.rotation
    j2, beta, gamma = moment_ratios(np.diag(rotation1.total_inertia))
    expected = {"beta": nominal.parameters["beta"], "gamma": nominal.parameters["gamma"], name: value}
    assert j2 == pytest.approx(moment_ratios(DE440_MOMENTS)[0], rel=1e-12)
    assert beta == pytest.approx(expected["beta"], rel=1e-12) and gamma == pytest.approx(expected["gamma"], rel=1e-12)
    polar0, polar1 = rotation0.total_inertia[2, 2], rotation1.total_inertia[2, 2]
    assert polar1 != polar0
    # k_v / C_T, the core flattening and the core moment fraction are unchanged; the core moment follows C.
    assert rotation1.core.viscosity_per_s / polar1 == pytest.approx(rotation0.core.viscosity_per_s / polar0, rel=4e-16)
    assert changed.parameters["kv_ct"] == pytest.approx(nominal.parameters["kv_ct"], rel=4e-16)
    assert changed.parameters["fc"] == nominal.parameters["fc"]
    np.testing.assert_allclose(rotation1.core_inertia, rotation0.core_inertia * polar1 / polar0, rtol=4e-16)
    # The deformation and the torques share the rebuilt moments.
    np.testing.assert_array_equal(rotation1.deformation.undistorted_inertia, rotation1.total_inertia)
    assert all(
        np.array_equal(model.total_inertia, rotation1.total_inertia)
        for model in rotation1.torque_group.models
        if hasattr(model, "total_inertia")
    )


def test_core_and_delay_parameters_reach_their_models(tmp_path):
    config, context = _full_config(tmp_path)
    nominal = build_lunar_propagation_problem(config, context)
    values = {"kv_ct": 1.6e-8, "fc": 2.5e-4, "tau_m": 0.0025, "tau_r1": 0.008, "c32": 1.5e-5, "c33": 1.1e-5}
    changed = build_lunar_propagation_problem(config, context, LunarDynamicsParameters(values, EPOCH.jd1))

    rotation = changed.dynamics.rotation
    polar = rotation.total_inertia[2, 2]
    assert rotation.core.viscosity_per_s == 1.6e-8 * polar / 86400.0
    assert rotation.core.flattening == 2.5e-4
    np.testing.assert_array_equal(rotation.total_inertia, nominal.dynamics.rotation.total_inertia)
    assert rotation.deformation.parameters.delay_s == 0.0025 * 86400
    tide = changed.dynamics.force_group.active_models[-1].tide_model
    assert tide.tide_parameters.tau_rot_days[1] == 0.008
    coeffs = rotation.deformation.field.coefficients.coeffs
    assert (coeffs[0, 3, 2], coeffs[0, 3, 3]) == (1.5e-5, 1.1e-5)
    for name in PARAMETER_NAMES:
        assert changed.parameters[name] == pytest.approx(values.get(name, nominal.parameters[name]), rel=4e-16), name


def test_parameters_of_a_model_that_is_not_configured_are_rejected(tmp_path):
    config, context = orbit_program_config(tmp_path, eih=True)
    nominal = build_lunar_propagation_problem(config, context)
    assert nominal.parameters.names == (*INITIAL_VALUE_NAMES[:6], "gm_emb")
    for name, message in (("fc", "model is not configured"), ("c32", "no Moon gravity field")):
        with pytest.raises(ValueError, match=message):
            build_lunar_propagation_problem(config, context, LunarDynamicsParameters({name: 1e-4}, EPOCH.jd1))


def test_initial_values_of_another_epoch_are_rejected(tmp_path):
    config, context = _full_config(tmp_path)
    with pytest.raises(ValueError, match="initial values at"):
        build_lunar_propagation_problem(config, context, LunarDynamicsParameters({"r0_x": 3.8e8}, EPOCH.jd1 + 1))
    # Dynamical parameters do not refer to an epoch.
    build_lunar_propagation_problem(config, context, LunarDynamicsParameters({"fc": 2.1e-4}, EPOCH.jd1 + 1))


def test_applying_the_values_of_a_configuration_returns_it_unchanged(tmp_path):
    config, _ = _full_config(tmp_path)
    parameters = LunarDynamicsParameters(config_parameters(config), EPOCH.jd1)
    assert apply_to_config(config, parameters) is config
    assert apply_to_config(config, None) is config
    changed = apply_to_config(config, parameters.with_value("tau_m", 0.003))
    assert changed is not config and config["lunarRotation"]["deformation"]["delayDays"] == 0.002
    assert changed["lunarRotation"]["deformation"]["delayDays"] == 0.003


def test_a_sibling_differs_from_the_problem_in_the_changed_parameter_only(tmp_path):
    config, context = _full_config(tmp_path)
    problem = build_lunar_propagation_problem(config, context)
    digests = {_digest(problem)}
    for name in DYNAMICAL_PARAMETER_NAMES:
        step = parameter_definition(name).step
        sibling = build_sibling_dynamics(problem, context, {name: problem.parameters[name] + step})
        digests.add(system_digest(sibling, problem.integrator.settings))
    assert len(digests) == 1 + len(DYNAMICAL_PARAMETER_NAMES)
    same = build_sibling_dynamics(problem, context, {})
    assert system_digest(same, problem.integrator.settings) == _digest(problem)


def test_the_gm_batch_view_replaces_the_earth_and_moon_columns_only(tmp_path):
    config, context = _full_config(tmp_path)
    problem = build_lunar_propagation_problem(config, context)
    epochs = (EPOCH, EPOCH.shifted(60.0))
    batch = problem.dynamics.epoch_batch(epochs)
    scale = 1.0 + 1e-6
    sibling = build_sibling_dynamics(problem, context, {"gm_emb": problem.parameters["gm_emb"] * scale})
    earth, moon = sibling.gravitational_parameters_m3_s2[:2]

    view = batch_with_earth_moon_gm(batch, earth, moon)

    expected = sibling.epoch_batch(epochs)
    np.testing.assert_array_equal(view.gravitational_parameters_m3_s2, expected.gravitational_parameters_m3_s2)
    for field in ("positions_m", "source_accelerations_mps2", "earth_fixed2inertial_matrix"):
        assert getattr(view, field) is getattr(batch, field)
        np.testing.assert_array_equal(getattr(view, field), getattr(expected, field))
    assert batch.gravitational_parameters_m3_s2[0, 0] == problem.dynamics.gravitational_parameters_m3_s2[0]
    # The sibling evaluated on the view equals the sibling evaluated on its own batch.
    state = problem.initial_vector
    np.testing.assert_array_equal(
        sibling.derivatives(EPOCH, state, view, history=problem.external_history, rigid=True),
        sibling.derivatives(EPOCH, state, expected, history=problem.external_history, rigid=True),
    )
