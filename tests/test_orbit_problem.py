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
    orbit, diagnostics, _ = lunar_orbit.lunar_orbit_propagation(config, context)
    _, _, states, _ = read_numeric_table(orbit)
    _, _, accelerations, _ = read_numeric_table(diagnostics)
    np.testing.assert_array_equal(states[:, 0], direction * np.arange(0, 1201, 60))
    np.testing.assert_array_equal(accelerations[:, 0], states[:, 0])
    np.testing.assert_array_equal(states[0, -6:], np.zeros(6))
    np.testing.assert_array_equal(states[:, 19:25], states[:, 7:13])
    assert np.all(np.isfinite(states)) and np.all(np.isfinite(accelerations))
    assert np.linalg.norm(states[-1, -6:]) > 0
