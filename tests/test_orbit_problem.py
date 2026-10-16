from pathlib import Path

import numpy as np
import pytest

from lunarops.classes.ephemerides import BodyState, Ephemeris
from lunarops.classes.time import Epoch, TimeScale
from lunarops.config.context import RunContext
from lunarops.fileio.mass_catalog import BodyMass, MassCatalog, write_mass_catalog
from lunarops.fileio.numeric_table import read_numeric_table
from lunarops.programs import lunar_orbit
from lunarops.programs.registry import get_program

INITIAL_EPOCH = Epoch(2451545, 0, TimeScale.TDB)


class AnalyticEphemeris(Ephemeris):
    def body_state_matrix_bcrs(self, bodies, epoch):
        return np.asarray([np.r_[self.body_state_bcrs(body, epoch).position_m,
                                self.body_state_bcrs(body, epoch).velocity_mps] for body in bodies]).reshape(-1, 6)

    def body_state_matrices_bcrs(self, bodies, epochs):
        return np.asarray([self.body_state_matrix_bcrs(bodies, epoch) for epoch in epochs])

    def body_accelerations_bcrs(self, body, epochs):
        return np.asarray([self.body_acceleration_bcrs(body, epoch) for epoch in epochs])

    @property
    def source_path(self):
        return None

    def body_state_bcrs(self, name, epoch):
        if name == "EARTH":
            return BodyState(np.zeros(3), np.zeros(3))
        if name == "SUN":
            return BodyState(np.array([1.49e11, 2e10, 0]), np.zeros(3))
        angle = INITIAL_EPOCH.seconds_until(epoch) * 1000 / 3.844e8
        return BodyState(
            3.844e8 * np.array([np.cos(angle), np.sin(angle), 0]),
            1000 * np.array([-np.sin(angle), np.cos(angle), 0]),
        )

    def body_acceleration_bcrs(self, name, epoch):
        return np.zeros(3)


def prepare_run(root: Path, *, eih: bool, direction: int = 1):
    from lunarops.classes.observation_factory import ensure_registered

    ensure_registered()
    catalog = MassCatalog(
        (
            BodyMass("EARTH", 3.98600435507e14, ("major",), 399),
            BodyMass("MOON", 4.902800118e12, ("major",), 301),
            BodyMass("SUN", 1.32712440018e20, ("major",), 10),
        ),
        name="analytic test",
    )
    write_mass_catalog(catalog, root / "masses.txt")
    # Only the physical backend is injected; schema defaults and integration
    # are the production paths, including DOP853 startup and ABM nodes.
    config = get_program("LunarOrbitPropagation").spec.schema.resolve(
        {
            "inputFileMassCatalog": "masses.txt",
            "outputFileOrbit": "orbit.dat",
            "outputFileAccelerationDiagnostics": "accel.dat",
            "outputFileMetadata": "meta.txt",
            "ephemerides": {
                "type": "calceph",
                "directory": ".",
                "lunarRelativisticScaleConvention": "alreadyScaled",
            },
            "initialTdbJd1": INITIAL_EPOCH.jd1,
            "durationSeconds": direction * 1200,
            "stepSeconds": 60,
            "outputStepSeconds": 60,
            "startupStepSeconds": 15,
            "integratorOrder": 3,
            "historyInterpolationOrder": 3,
            "trajectoryInterpolationOrder": 3,
            "externalBodyIds": ["SUN"],
            "includeEih": eih,
            "accelerationDiagnosticsStepSeconds": 60,
        }
    )
    context = RunContext(working_dir=root)
    context.create_class = lambda *args, **kwargs: AnalyticEphemeris()
    return config, context


@pytest.mark.parametrize("eih", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
def test_orbit_problem_runs_both_directions_with_consistent_products(tmp_path, eih, direction):
    config, context = prepare_run(tmp_path, eih=eih, direction=direction)
    orbit, diagnostics, metadata_path = lunar_orbit.lunar_orbit_propagation(config, context)
    from lunarops.fileio.structured_reports import read_lunar_orbit_metadata

    metadata = read_lunar_orbit_metadata(metadata_path)
    assert metadata["timeScale"] == "TDB" and metadata["stepSeconds"] == 60
    assert metadata_path.read_text().splitlines()[0] == "lunarops lunarOrbitMetadata version=2"
    _, _, states, _ = read_numeric_table(orbit)
    _, _, accelerations, _ = read_numeric_table(diagnostics)
    np.testing.assert_array_equal(states[:, 0], direction * np.arange(0, 1201, 60))
    np.testing.assert_array_equal(accelerations[:, 0], states[:, 0])
    np.testing.assert_array_equal(states[0, -6:], np.zeros(6))
    np.testing.assert_array_equal(states[:, 19:25], states[:, 7:13])
    assert np.all(np.isfinite(states)) and np.all(np.isfinite(accelerations))
    assert np.linalg.norm(states[-1, -6:]) > 0


@pytest.mark.parametrize("core", [False, True])
@pytest.mark.parametrize("figure", [False, True])
def test_program_integrates_rotation_and_writes_normalized_torque_diagnostics(tmp_path, core, figure):
    config, context = prepare_run(tmp_path, eih=True)
    config["lunarRotation"] = {
        "inertiaMoments": [0.393, 0.3932, 0.3935],
        "radiusM": 1738000.0,
        "initialQuaternion": [1.0, 0.0, 0.0, 0.0],
        "initialAngularVelocityRadS": [1e-7, -2e-7, 2.7e-6],
        "initialCoreAngularVelocityRadS": [2e-7, 0.0, 2.6e-6] if core else None,
        "sources": ["EARTH", "SUN"],
        "gravityTorque": "quadrupole",
        "earthFigureTorque": False,
        "geodeticPrecession": True,
        "core": {
            "polarMomentFraction": 0.0007,
            "flattening": 0.000246,
            "viscosityPerS": 7e-14,
            "pressureSpin": "core",
        }
        if core
        else None,
    }
    config["outputFileRotation"] = "rotation.dat"
    if figure:
        for name, radius, c20 in (("earth", 6378136.6, -4.8e-4), ("moon", 1738000.0, -9e-5)):
            (tmp_path / f"{name}.gfc").write_text(
                "begin_of_head\nproduct_type gravity_field\nmodelname rotation_test\n"
                f"gravity_constant 4.9e12\nradius {radius}\nmax_degree 2\n"
                "norm fully_normalized\ntide_system zero_tide\nend_of_head\n"
                f"gfc 0 0 1 0\ngfc 2 0 {c20} 0\ngfc 2 1 0 0\ngfc 2 2 3.4e-5 0\n"
            )
        config["gravityFields"] = {name: {"path": f"{name}.gfc"} for name in ("earth", "moon")}
        config["figurePartners"] = {"earth": ["MOON"], "moon": ["EARTH", "SUN"]}
        config["lunarRotation"]["gravityTorque"] = "figure"
        config["lunarRotation"]["earthFigureTorque"] = True
    schema = get_program("LunarOrbitPropagation").spec.schema
    config = schema.resolve(config)
    problem = lunar_orbit.build_lunar_propagation_problem(config, context)
    assert problem.initial_vector.shape == ((16,) if core else (13,))
    assert not problem.initial_vector.flags.writeable
    orbit, _, _, rotation = lunar_orbit.lunar_orbit_propagation(config, context)
    _, _, positions, _ = read_numeric_table(orbit)
    _, columns, states, meta = read_numeric_table(rotation, "lunarRotation")
    np.testing.assert_array_equal(states[:, 0], positions[:, 0])
    assert np.all(np.isfinite(states))
    assert np.max(np.abs(states[:, columns.index("quaternion_norm_error")])) < 1e-12
    assert np.linalg.norm(states[-1, 1:5] - states[0, 1:5]) > 0
    assert meta["pressureSpin"] == ("core" if core else None)
    assert f"{'figure' if figure else 'quadrupole'}:SUN_x_per_s2" in columns
    assert ("core_omega_pa_x_rad_s" in columns) == core


def test_rotation_cannot_silently_reuse_prescribed_attitude_deformation(tmp_path):
    config, _context = prepare_run(tmp_path, eih=False)
    config["lunarRotation"] = {}
    config["lunarDegree2Gravity"] = {}
    with pytest.raises(ValueError, match="lunarRotation.deformation"):
        lunar_orbit._build_rotation(config, None, 4.9e12, 3.986e14)


@pytest.mark.parametrize("direction", [-1, 1])
def test_program_precomputes_ring_and_uses_epoch_gm(tmp_path, direction):
    config, context = prepare_run(tmp_path, eih=True, direction=direction)
    config["solarGmVariation"] = {"fractionalRatePerYear": -9e-14, "referenceJdTdb": 2451545.0}
    config["kuiperBeltRing"] = {"pointGmM3S2": 2.4768842184e11, "referenceJdTdb": 2451545.0}
    config = get_program("LunarOrbitPropagation").spec.schema.resolve(config)
    problem = lunar_orbit.build_lunar_propagation_problem(config, context)
    assert len(problem.dynamics.body_names) == 39
    assert problem.dynamics.history_body_names == ("EARTH", "SUN")
    final = INITIAL_EPOCH.shifted(direction * 1200)
    data = problem.dynamics.build_epoch_data(final)
    assert data.gravitational_parameters_m3_s2[2] == 1.32712440018e20 * (
        1 - 9e-14 * direction * 1200 / (365.25 * 86400)
    )
    orbit, _, _ = lunar_orbit.lunar_orbit_propagation(config, context)
    _, _, values, _ = read_numeric_table(orbit)
    assert np.isfinite(values).all()
