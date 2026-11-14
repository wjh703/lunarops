"""Model fingerprint of a propagation problem: stable across builds and evaluation, sensitive to every input."""

from dataclasses import replace

import pytest

from lunarops.classes.dynamics import IntegratorSettings
from lunarops.classes.dynamics.assembly import build_lunar_propagation_problem
from lunarops.classes.dynamics.bodies import GravitatingBody
from lunarops.classes.dynamics.epoch_data import OrientationProvider
from lunarops.classes.dynamics.fingerprint import model_fingerprint
from lunarops.classes.dynamics.force_models import LunarForceGroup, NewtonianPointMassForce
from lunarops.classes.dynamics.lunar_dynamics import LunarDynamics
from lunarops.programs.registry import ensure_builtin_programs, get_program
from tests.support.dynamics import EARTH_MU, EPOCH, MOON_MU, FunctionEphemeris, History, build, orbit_program_config

SETTINGS = IntegratorSettings(step_s=2700.0, order=13)


def _program(tmp_path, change=None):
    """A program problem with tide, rotation, core, ring and solar GM drift; ``change`` edits the raw config."""
    ensure_builtin_programs()
    config, context = orbit_program_config(tmp_path, eih=True)
    config["earthTide"] = {}
    config["solarGmVariation"] = {"fractionalRatePerYear": -9e-14, "referenceJdTdb": 2451545.0}
    config["kuiperBeltRing"] = {"pointGmM3S2": 2.4768842184e11, "referenceJdTdb": 2451545.0, "pointCount": 4}
    config["lunarRotation"] = {
        "inertiaMoments": [0.393, 0.3932, 0.3935],
        "radiusM": 1738000.0,
        "initialEulerAnglesRad": [0.1, 0.4, 1.2],
        "initialAngularVelocityRadS": [1e-7, -2e-7, 2.7e-6],
        "initialCoreAngularVelocityRadS": [2e-7, 0.0, 2.6e-6],
        "sources": ["EARTH", "SUN"],
        "gravityTorque": "quadrupole",
        "earthFigureTorque": False,
        "geodeticPrecession": True,
        "core": {"polarMomentFraction": 0.0007, "flattening": 0.000246, "viscosityPerS": 7e-14},
    }
    config["outputFileRotation"] = "rotation.dat"
    if change is not None:
        change(config)
    config = get_program("LunarOrbitPropagation")[0].schema.resolve(config)
    return build_lunar_propagation_problem(config, context)


def _fingerprint(problem):
    return model_fingerprint(problem.dynamics, problem.integrator.settings)


def test_an_evaluated_system_keeps_the_fingerprint_of_a_fresh_build():
    system, state = build(fused=True)
    fresh = model_fingerprint(build(fused=True)[0], SETTINGS)
    history = History(state)
    for offset in (0.0, 3600.0, -7200.0):
        epoch = EPOCH.shifted(offset)
        system.terms(epoch, state, history=history)
        system.derivatives(epoch, state, history=history)
    assert model_fingerprint(system, SETTINGS) == fresh


def test_a_propagated_program_problem_keeps_the_fingerprint_of_a_fresh_build(tmp_path):
    problem = _program(tmp_path)
    fresh = _fingerprint(_program(tmp_path))
    problem.integrator.integrate(
        problem.dynamics,
        problem.initial_epoch,
        problem.initial_vector,
        problem.duration_s,
        external_history_provider=problem.external_history,
        seed_duration_s=problem.seed_duration_s,
    )
    assert _fingerprint(problem) == fresh


def _set(*path_and_value):
    *path, key, value = path_and_value

    def change(config):
        section = config
        for name in path:
            section = section[name]
        section[key] = value

    return change


@pytest.mark.parametrize(
    "change",
    [
        _set("earthTide", "k22", 0.284),
        _set("earthTide", "tauOrbDays", [0.077, -0.010, -0.2]),
        _set("bodyForceTerms", ["point_mass", "eih_1pn"]),
        _set("includeEih", False),
        _set("lunarRotation", "inertiaMoments", [0.393, 0.3932, 0.39351]),
        _set("lunarRotation", "core", "viscosityPerS", 8e-14),
        _set("lunarRotation", "geodeticPrecession", False),
        _set("kuiperBeltRing", "phaseDegrees", 1.0),
        _set("solarGmVariation", "fractionalRatePerYear", -8e-14),
        _set("stepSeconds", 30.0),
        _set("correctorIterations", 3),
    ],
    ids=[
        "tide-k22",
        "tide-delay",
        "force-selection",
        "eih",
        "inertia",
        "core-viscosity",
        "geodetic-precession",
        "ring-phase",
        "solar-gm-drift",
        "step",
        "corrector",
    ],
)
def test_any_changed_input_changes_the_fingerprint(tmp_path, change):
    assert _fingerprint(_program(tmp_path, change)) != _fingerprint(_program(tmp_path))


def test_a_changed_step_size_changes_the_fingerprint():
    system = build(fused=True)[0]
    assert model_fingerprint(system, replace(SETTINGS, step_s=1350.0)) != model_fingerprint(system, SETTINGS)


class _KernelEphemeris(FunctionEphemeris):
    """The constant analytic ephemeris, declared to come from the kernels of ``source``."""

    def __init__(self, source):
        super().__init__()
        self._source = source

    @property
    def source_path(self):
        return self._source


def _system(ephemeris, earth_axes=None):
    return LunarDynamics(
        GravitatingBody("EARTH", EARTH_MU),
        GravitatingBody("MOON", MOON_MU),
        perturbing_bodies=(),
        ephemeris=ephemeris,
        force_group=LunarForceGroup((NewtonianPointMassForce(),)),
        earth_fixed2inertial_matrix_provider=earth_axes,
    )


def _kernel_fingerprint(source):
    return model_fingerprint(_system(_KernelEphemeris(source)), SETTINGS)


def test_the_ephemeris_kernels_are_identified_by_their_contents(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        directory.mkdir()
        # Larger than the two 64 KiB samples, so that only the tail sample sees the last byte.
        (directory / "planets.bsp").write_bytes(bytes(range(256)) * 1024)
        (directory / "moon.bpc").write_bytes(b"orientation")
    reference = _kernel_fingerprint(first)

    assert _kernel_fingerprint(second) == reference
    planets = second / "planets.bsp"
    data = bytearray(planets.read_bytes())
    data[-1] ^= 1
    planets.write_bytes(bytes(data))
    assert _kernel_fingerprint(second) != reference
    (first / "moon.bpc").write_bytes(b"orientation 2")
    assert _kernel_fingerprint(first) != reference
    assert _kernel_fingerprint(None) != reference


def test_a_value_the_encoding_does_not_know_is_an_error():
    system = _system(FunctionEphemeris(), earth_axes=OrientationProvider(object()))
    with pytest.raises(TypeError, match="Cannot fingerprint a value of type builtins.object"):
        model_fingerprint(system, SETTINGS)
