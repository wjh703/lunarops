"""Lunar dynamics and Love-number partials, the parametrization blocks built on them, and their adjustment.

Partials are derivatives of the one-way computed range, so d(O - C)/dp = -partial.
"""

import numpy as np
import pytest

from lunarops.base.parameter_name import ParameterName, parameter_unit
from lunarops.classes.displacement.lunar_solid_tide import LunarSolidTide
from lunarops.classes.dynamics.parameters import LunarDynamicsParameters, read_lunar_dynamics_parameters
from lunarops.classes.dynamics.rotation import euler_matrix
from lunarops.classes.dynamics.trajectory import interpolate_regular
from lunarops.classes.ephemerides import FixedLunarOrientation, LunarRelativisticScale
from lunarops.classes.frames import ReferenceFrameSystem
from lunarops.classes.frames.coordinate_scaling import GM_EARTH, GM_MOON
from lunarops.classes.observation import NptDataset, ObservationOptions
from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.lunar_dynamics import (
    LunarDynamicsParametrization,
    euler_to_pa_rotation,
    interpolate_regular_rows,
)
from lunarops.classes.parametrization.lunar_love_number import LunarLoveNumberParametrization
from lunarops.classes.time import Epoch, TimeScale
from lunarops.estimation.lsq.normal_equations import NormalEquations
from lunarops.estimation.lsq.solve import solve_normal_equations
from lunarops.fileio.parameter_vectors import read_parameter_vector
from lunarops.fileio.processing_artifacts import read_processing_state
from lunarops.programs.llr_processing import llr_processing
from lunarops.programs.registry import resolve_program_config
from tests.support.lunar_partials import (
    INITIAL_STATE,
    NOMINAL,
    partial_values,
    state_values,
    synthetic_partials,
    write_recovery_case,
)
from tests.support.observation import (
    STATIC_BODIES,
    StaticKernelContext,
    ZeroEarthOrientation,
    npt_record,
    observation_processor,
    static_frames,
    station_facing_the_moon,
)
from tests.support.synthetic import LinearEphemeris
from tests.tolerances import REL, Tolerance, assert_close

OPTIONS = ObservationOptions(
    min_elevation_deg=-90.0,
    include_reflector_position_partials=True,
    include_lunar_orientation_partials=True,
    include_moon_position_partials=True,
    include_lunar_tide_partials=True,
)
EULER = np.array([0.3, 0.4, 2.2])
ROTATING_STATION = station_facing_the_moon(static_frames(), zenith_angle_deg=35.0)
# A station on the rotation axis does not move between the events, so the event epochs do not enter.
POLAR_STATION = np.array([0.0, 0.0, 6_356_752.0])
# The analytic partials ignore how the event epochs move with the parameter: |v|/c ~ 1.6e-6 for the rotating
# station (measured: 9e-7 for the Moon position, 7e-7 for the Euler angles, 1.3e-7 for h2).
MOON_POSITION_PARTIAL = Tolerance("MOON_POSITION_PARTIAL_V_OVER_C", rtol=5e-6, atol=2e-6)
EULER_PARTIAL = Tolerance("EULER_PARTIAL_V_OVER_C", rtol=5e-6, atol=2.0, unit="m/rad")
H2_PARTIAL = Tolerance("H2_PARTIAL_V_OVER_C", rtol=2e-6, unit="m")
# Without station motion the central difference over +-1 km agrees to 3.5e-10 (measured); the relativistic
# factor of the reflector partial, 1 - 9.9e-9 in this system, would be thirty times that.
MOON_POSITION_PARTIAL_AT_REST = Tolerance("MOON_POSITION_PARTIAL_AT_REST", atol=2e-9)
RECOVERY = Tolerance("LINEAR_RECOVERY", rtol=1e-6)


def _equations(*, shift_m=(0.0, 0.0, 0.0), euler=EULER, h2=None, station=ROTATING_STATION):
    """Evaluate one normal point with the Moon-minus-Earth vector shifted at a fixed Earth-Moon barycentre."""
    shift = np.asarray(shift_m, dtype=float)
    total = GM_EARTH + GM_MOON
    bodies = dict(STATIC_BODIES)
    bodies["EARTH"] = (np.asarray(bodies["EARTH"][0]) - GM_MOON / total * shift, (0, 0, 0))
    bodies["MOON"] = (np.asarray(bodies["MOON"][0]) + GM_EARTH / total * shift, (0, 0, 0))
    attitude = euler_matrix(euler)
    frames = ReferenceFrameSystem(
        LinearEphemeris(bodies),
        ZeroEarthOrientation(),
        FixedLunarOrientation(lambda epoch: attitude),
        LunarRelativisticScale.from_convention("alreadyScaled"),
    )
    tide = () if h2 is None else (LunarSolidTide(frames.relativistic_transform, h2=h2),)
    processor = observation_processor(frames=frames, station_itrf_m=station, reflector_displacement=tide)
    return processor.equations(NptDataset([npt_record()]), options=OPTIONS)


def _central(evaluate, step):
    plus, minus = evaluate(step), evaluate(-step)
    return -(plus.oc - minus.oc)[0] / (2.0 * step)


@pytest.mark.parametrize("axis", range(3))
def test_moon_position_partial_matches_moving_the_moon_about_a_fixed_barycentre(axis):
    nominal = _equations()
    numeric = _central(lambda h: _equations(shift_m=h * np.eye(3)[axis]), 100.0)
    assert_close(numeric, nominal.moon_position_bcrs[0, axis], MOON_POSITION_PARTIAL)


def test_moon_position_partial_has_no_lunar_coordinate_scale():
    nominal = _equations(station=POLAR_STATION)
    numeric = [
        _central(lambda h, axis=axis: _equations(shift_m=h * np.eye(3)[axis], station=POLAR_STATION), 1000.0)
        for axis in range(3)
    ]
    partial = nominal.moon_position_bcrs[0]
    # The reflector partial carries the LCRS-to-BCRS scale of the selenocentric vector; the Moon's centre does not.
    reflector_factor = np.linalg.norm(nominal.reflector_position_pa[0]) / np.linalg.norm(partial)

    assert_close(numeric, partial, MOON_POSITION_PARTIAL_AT_REST)
    assert abs(reflector_factor - 1.0) > 5e-9
    assert np.max(np.abs(numeric - reflector_factor * partial)) > 5e-9


@pytest.mark.parametrize("angle", range(3))
def test_orientation_partial_times_the_euler_map_matches_changing_an_euler_angle(angle):
    nominal = _equations()
    analytic = nominal.lunar_orientation_pa[0] @ euler_to_pa_rotation(EULER)
    numeric = _central(lambda h: _equations(euler=EULER + h * np.eye(3)[angle]), 1.0e-6)

    assert abs(analytic[angle]) > 1.0e5
    assert_close(numeric, analytic[angle], EULER_PARTIAL)


@pytest.mark.parametrize("angle", range(3))
def test_euler_map_is_the_small_rotation_of_the_principal_axes(angle):
    # R(e + de) = R(e) (I + [w]x) with w = E de; the central difference rounds at 1e-16 / h.
    h = 1.0e-6
    step = h * np.eye(3)[angle]
    rotation = euler_matrix(EULER).T @ (euler_matrix(EULER + step) - euler_matrix(EULER - step)) / (2 * h)
    w = np.array([rotation[2, 1], rotation[0, 2], rotation[1, 0]])

    assert_close(w, euler_to_pa_rotation(EULER)[:, angle], Tolerance("EULER_MAP_DIFFERENCE", atol=1e-9))
    assert_close(rotation + rotation.T, np.zeros((3, 3)), Tolerance("SKEW", atol=1e-9))


def test_h2_partial_matches_changing_the_love_number_and_is_zero_without_a_tide():
    nominal = _equations(h2=0.04)
    numeric = _central(lambda h: _equations(h2=0.04 + h), 0.01)

    assert nominal.lunar_h2.shape == (1, 1) and abs(nominal.lunar_h2[0, 0]) > 1.0
    assert_close(numeric, nominal.lunar_h2[0, 0], H2_PARTIAL)
    assert _equations().lunar_h2.tolist() == [[0.0]]


def test_h2_displacement_partial_is_the_displacement_of_unit_h2_without_l2():
    from lunarops.classes.displacement.base import ReflectorDisplacementInput

    transform = static_frames().relativistic_transform
    data = ReflectorDisplacementInput(np.array([1.0e6, 1.2e6, -0.7e6]), Epoch(2458849.5, 0.1, TimeScale.TDB))
    transform.begin_observation()
    unit = LunarSolidTide(transform, h2=1.0, l2=0.0).displacement_lcrs_m(data)

    assert_close(LunarSolidTide(transform).displacement_per_h2_lcrs_m(data), unit, REL)


def test_bounce_epoch_is_the_tdb_transmit_epoch_plus_the_uplink_light_time():
    batch = _equations()
    elapsed_days = (batch.bounce_tdb_jd1 - batch.epoch_jd1) + (batch.bounce_tdb_jd2 - batch.epoch_jd2)
    # TDB - UTC is 69.184 s on 2020-01-01 (to 2 ms); the uplink is about 381 000 km.
    assert elapsed_days[0] * 86400.0 - 69.184 == pytest.approx(1.27, abs=0.02)


def test_vectorised_interpolation_equals_the_trajectory_interpolation():
    nodes = -3.0e5 + 21600.0 * np.arange(40)
    values = np.stack([np.sin(nodes / 2.0e5), 8000.0 + nodes / 1.0e4, np.cos(nodes / 3.0e5) ** 2], axis=1)
    points = np.r_[nodes[0], nodes[3], nodes[-1], np.linspace(nodes[0] + 1.0, nodes[-1] - 1.0, 57)]
    expected = np.asarray(
        [interpolate_regular(nodes, values.T, float(point), 9, extrapolate=False) for point in points]
    )

    assert_close(
        interpolate_regular_rows(nodes, values, points, 9), expected, Tolerance("INTERPOLATION", rtol=1e-13, atol=1e-13)
    )
    assert_close(
        interpolate_regular_rows(nodes, values[:, :, None] * np.ones(4), points, 9)[:, :, 2],
        expected,
        Tolerance("INTERPOLATION", rtol=1e-13, atol=1e-13),
    )
    with pytest.raises(ValueError, match="outside"):
        interpolate_regular_rows(nodes, values, [nodes[-1] + 1.0], 9)
    with pytest.raises(ValueError, match="regular"):
        interpolate_regular_rows(np.r_[nodes, nodes[-1] + 1.0], np.r_[values, values[:1]], [0.0], 9)


EPOCH_JD = 2458849.5


def _batch(offsets_s, *, moon=None, orientation=None, h2=None):
    chunks = []
    for index, offset in enumerate(offsets_s):
        bounce = Epoch(EPOCH_JD, offset / 86400.0, TimeScale.TDB)
        transmit = Epoch(EPOCH_JD, offset / 86400.0, TimeScale.UTC)
        chunks.append(
            EquationBatch.single(
                oc=0.0,
                sigma=0.01,
                observation_id=index,
                station="A",
                reflector="R",
                epoch=transmit,
                receive_epoch=transmit,
                bounce_epoch_tdb=bounce,
                moon_position_bcrs=(0.6, -0.5, 0.3 + 0.01 * index) if moon is None else moon,
                lunar_orientation_pa=(1.0e6, -4.0e5, 7.0e5 + 1.0e4 * index) if orientation is None else orientation,
                lunar_h2=(-4.0 + 0.1 * index,) if h2 is None else h2,
            )
        )
    return EquationBatch.concatenate(chunks)


def _block_design(block, batch):
    out = np.zeros((len(batch), len(block.names)))
    block.design(batch, out)
    return out


def test_dynamics_design_is_the_chain_rule_at_the_interpolated_bounce_epoch():
    block = LunarDynamicsParametrization(partials=synthetic_partials(EPOCH_JD, start_s=-20 * 21600.0))
    offsets = np.array([-3.0e5, 1234.5, 21600.0 * 7, 2.9e6])
    batch = _batch(offsets)
    block.setup(batch)
    partials, euler = partial_values(offsets), state_values(offsets)[:, 6:9]
    expected = np.einsum("ni,niq->nq", batch.moon_position_bcrs, partials[:, 0:3]) + np.einsum(
        "ni,nij,njq->nq", batch.lunar_orientation_pa, euler_to_pa_rotation(euler), partials[:, 6:9]
    )

    # Ten-point interpolation of the analytic table on its 6 h grid (periods of 9 d and more).
    assert_close(_block_design(block, batch), expected, Tolerance("TABLE_INTERPOLATION", rtol=1e-9))
    assert [str(name) for name in block.names][:2] == ["moon:dynamics.r0_x::", "moon:dynamics.v0_y::"]
    assert [parameter_unit(name) for name in block.names] == ["m", "m/s", "rad", "rad/s", "m3/s2", "1"]
    assert parameter_unit(ParameterName("moon", "dynamics.tau_m")) == "d"
    assert parameter_unit(ParameterName("moon", "dynamics.kv_ct")) == "1/d"
    assert parameter_unit(ParameterName("moon", "loveNumber.h2")) == "1"


def test_dynamics_block_selects_quantities_and_works_without_rotation():
    table = synthetic_partials(EPOCH_JD)
    batch = _batch([1.0e5, 2.0e5])
    full = LunarDynamicsParametrization(partials=table)
    subset = LunarDynamicsParametrization(partials=table, quantities=["gm_emb", "r0_x"])
    orbit = LunarDynamicsParametrization(
        partials=synthetic_partials(EPOCH_JD, dimension=6, quantities=("r0_x", "v0_y", "gm_emb"))
    )
    for block in (full, subset, orbit):
        block.setup(batch)

    assert_close(_block_design(subset, batch), _block_design(full, batch)[:, [4, 0]], REL)
    assert_close(
        _block_design(orbit, batch),
        np.einsum("ni,niq->nq", batch.moon_position_bcrs, partial_values([1.0e5, 2.0e5])[:, 0:3][:, :, [0, 1, 4]]),
        Tolerance("TABLE_INTERPOLATION", rtol=1e-9),
    )
    with pytest.raises(ValueError, match="not in the partials table"):
        LunarDynamicsParametrization(partials=table, quantities=["fc"])
    with pytest.raises(ValueError, match="no rotation state"):
        LunarDynamicsParametrization(partials=synthetic_partials(EPOCH_JD, dimension=6, quantities=("psi0",)))


def test_dynamics_block_rejects_epochs_outside_the_table_and_missing_partials():
    block = LunarDynamicsParametrization(partials=synthetic_partials(EPOCH_JD, count=20))
    with pytest.raises(ValueError, match=r"1 normal point\(s\) have bounce epochs .* outside"):
        block.setup(_batch([1.0e5, 20 * 21600.0]))
    with pytest.raises(ValueError, match="Moon-position partials"):
        block.setup(_batch([1.0e5], moon=()))
    with pytest.raises(ValueError, match="lunar-orientation partials"):
        block.setup(_batch([1.0e5], orientation=()))
    with pytest.raises(ValueError, match="lunar h2 partial"):
        _block_design(LunarLoveNumberParametrization(), _batch([1.0e5], h2=()))


def test_dynamics_block_reports_nominal_values_and_updated_parameters():
    table = synthetic_partials(EPOCH_JD)
    nominal = LunarDynamicsParameters(NOMINAL, EPOCH_JD)
    block = LunarDynamicsParametrization(partials=table, parameters=nominal, quantities=["v0_y", "beta"])
    block.setup(_batch([1.0e5]))
    block.update(np.array([1.0e-6, -2.0e-9]))
    updated, state = block.updated_parameters(), block.state()

    assert updated.names == nominal.names and updated.is_at(EPOCH_JD)
    assert updated["v0_y"] == NOMINAL["v0_y"] + 1.0e-6 and updated["beta"] == NOMINAL["beta"] - 2.0e-9
    assert all(updated[name] == NOMINAL[name] for name in nominal.names if name not in ("v0_y", "beta"))
    assert state["quantities"]["beta"] == {
        "unit": "1",
        "nominal": NOMINAL["beta"],
        "correction": -2.0e-9,
        "updated": NOMINAL["beta"] - 2.0e-9,
    }
    # Without a parameter file the initial values are those of the table; dynamical parameters are unknown.
    bare = LunarDynamicsParametrization(partials=table)
    assert bare.nominal["psi0"] == INITIAL_STATE[8] and bare.nominal["gm_emb"] is None
    with pytest.raises(ValueError, match="nominal values of .*gm_emb.* are unknown"):
        bare.updated_parameters()
    assert LunarDynamicsParametrization(partials=table, quantities=["r0_x"]).updated_parameters()["r0_x"] == 3.6e8
    with pytest.raises(ValueError, match="epoch"):
        LunarDynamicsParametrization(partials=table, parameters=LunarDynamicsParameters(NOMINAL, EPOCH_JD + 1.0))
    with pytest.raises(ValueError, match="not of one integration"):
        LunarDynamicsParametrization(
            partials=table, parameters=LunarDynamicsParameters({**NOMINAL, "r0_x": 3.6e8 + 1.0}, EPOCH_JD)
        )


def test_update_norms_are_range_changes_in_metres():
    dynamics = LunarDynamicsParametrization(partials=synthetic_partials(EPOCH_JD))
    love = LunarLoveNumberParametrization(nominal_h2=0.04)
    batch = _batch(np.linspace(1.0e5, 3.0e6, 9))
    dynamics.setup(batch)
    design, column = _block_design(dynamics, batch), _block_design(love, batch)
    rms = np.sqrt(np.mean(design**2, axis=0))
    delta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0e-3]) / rms

    assert dynamics.max_update_norm(delta) == pytest.approx(3.0e-3)
    assert love.max_update_norm(np.array([0.002])) == pytest.approx(0.002 * np.sqrt(np.mean(column**2)))


def test_normal_equations_are_solved_in_scaled_form_whatever_the_units():
    rng = np.random.default_rng(3)
    design = rng.normal(size=(60, 5))
    units = np.array([1.0, 1.0e9, 1.0e-12, 1.0e4, 1.0e-7])  # parameter = unit * well-scaled parameter
    truth = np.array([0.3, -1.2, 0.8, 2.0, -0.5])
    names = [ParameterName("moon", f"dynamics.p{index}") for index in range(5)]

    def normals(scale):
        a = design / scale
        return NormalEquations.from_linearized_statistics(
            names,
            a.T @ a,
            a.T @ (design @ truth),
            float((design @ truth) @ (design @ truth)),
            obs_count=60,
            x0=np.zeros(5),
            meta={},
            parameter_units=["1"] * 5,
        )

    plain, mixed = normals(np.ones(5)), normals(units)
    solved, reference = solve_normal_equations(mixed), solve_normal_equations(plain)

    # The eigenvalues of the unscaled matrix span 42 decades: its rank test would call it singular.
    eigenvalues = np.linalg.eigvalsh(mixed.N)
    assert np.count_nonzero(eigenvalues > np.finfo(float).eps * 5 * eigenvalues[-1]) < 5
    assert solved.rank == 5 and mixed.condition == pytest.approx(plain.condition, rel=1e-9) and mixed.condition < 10
    assert_close(solved.values, units * truth, Tolerance("SCALED_SOLVE", rtol=1e-12))
    assert_close(solved.covariance, np.outer(units, units) * reference.covariance, Tolerance("SCALED_COV", rtol=1e-11))
    assert_close(mixed.cholesky @ mixed.cholesky.T, mixed.N, Tolerance("CHOLESKY", rtol=1e-12))


def test_workflow_requests_the_partials_the_blocks_need():
    from lunarops.config.context import RunContext
    from lunarops.config.schema import ConfigSchema
    from lunarops.llr_workflow import build_equation_source, build_parametrization
    from lunarops.programs.observation_schema import observation_fields

    calls = []

    class Processor:
        def equations(self, dataset, *, options):
            calls.append(options)
            return dataset

    def options(*types):
        config = dict(
            ConfigSchema(observation_fields()).resolve({}),
            showProgress=False,
            parametrization=[{"type": kind} for kind in types],
        )
        build_equation_source(config, RunContext(), {"a": EquationBatch.empty()}, Processor())(1)
        return calls[-1]

    dynamics, love, bias = options("lunarDynamics"), options("lunarLoveNumber"), options("stationRangeBias")
    assert dynamics.include_moon_position_partials and dynamics.include_lunar_orientation_partials
    assert not dynamics.include_lunar_tide_partials and not dynamics.include_reflector_position_partials
    assert love.include_lunar_tide_partials and not love.include_moon_position_partials
    assert not (bias.include_moon_position_partials or bias.include_lunar_tide_partials)
    with pytest.raises(ValueError, match="lunarSolidTide"):
        build_parametrization({"parametrization": [{"type": "lunarLoveNumber"}]}, RunContext(), None)


def _run(directory, config, outputs=None):
    if outputs is not None:
        # Keep the files of the module's reference run.
        steps = [*config["processingSteps"][:-1], {"type": "writeResults", "outputFileState": outputs}]
        config = {**config, "processingSteps": steps}
    resolved = resolve_program_config("LlrProcessing", {**config, "showProgress": False})
    with StaticKernelContext(working_dir=directory) as context:
        context.resolve_defaults()  # as the CLI does before running programs
        return llr_processing(resolved, context)


@pytest.fixture(scope="module")
def recovery(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lunar-dynamics")
    config, truth = write_recovery_case(directory)
    return directory, config, truth, _run(directory, config)


def test_adjustment_recovers_the_corrections_behind_the_observations(recovery):
    directory, _, truth, result = recovery
    vector = read_parameter_vector(directory / "solution.txt")
    eigenvalues = np.linalg.eigvalsh(result.normals.N)

    assert result.converged and len(result.adjustment_iterations) == 2
    assert [str(name) for name in vector.parameter_names] == list(truth)
    assert_close(vector.values, list(truth.values()), RECOVERY)
    assert list(vector.units) == ["m", "m/s", "rad", "rad/s", "m3/s2", "1", "1"]
    # Columns from 1e-4 m per m3/s2 to 1e12 m per rad/s: only the scaled matrix has a meaningful spectrum.
    assert eigenvalues[-1] / abs(eigenvalues[0]) > 1e25
    assert result.normals.condition < 100.0


def test_written_parameters_are_the_nominal_values_plus_the_corrections(recovery):
    directory, _, truth, result = recovery
    nominal = read_lunar_dynamics_parameters(directory / "nominal_parameters.txt")
    updated = read_lunar_dynamics_parameters(directory / "updated_parameters.txt")
    state = read_processing_state(directory / "state.txt")["parametrization"]
    corrections = result.state["lunarDynamics"]["values"]

    assert updated.names == nominal.names
    assert (updated.epoch_tdb_jd1, updated.epoch_tdb_jd2) == (nominal.epoch_tdb_jd1, nominal.epoch_tdb_jd2)
    for name in nominal.names:
        assert updated[name] == nominal[name] + corrections.get(name, 0.0), name
    assert state == result.state
    assert state["lunarDynamics"]["quantities"]["gm_emb"]["updated"] == updated["gm_emb"]
    love = state["lunarLoveNumber"]["h2"]
    assert love["nominal"] == 0.0423 and love["updated"] == love["nominal"] + love["correction"]
    assert love["correction"] == pytest.approx(truth["moon:loveNumber.h2::"], rel=1e-6)


def test_apriori_sigma_holds_a_dynamical_parameter_at_its_nominal_value(recovery):
    directory, config, truth, _ = recovery
    constrained = {
        **config,
        "constraints": [{"type": "aprioriSigma", "parameters": ["moon:dynamics.gm_emb::"], "sigma": 1.0e-6}],
    }
    result = _run(directory, constrained, "constrained_state.txt")

    # The formal sigma of gm_emb is 0.02 m3/s2 without the constraint.
    assert abs(result.state["lunarDynamics"]["values"]["gm_emb"]) < 1.0e-6 * abs(truth["moon:dynamics.gm_emb::"])


def test_a_saved_state_restores_the_corrections(recovery):
    directory, config, _, result = recovery
    continued = _run(directory, {**config, "inputFileProcessingState": "state.txt"}, "continued_state.txt")

    assert len(continued.adjustment_iterations) == 1
    assert_close(
        list(continued.state["lunarDynamics"]["values"].values()),
        list(result.state["lunarDynamics"]["values"].values()),
        RECOVERY,
    )
