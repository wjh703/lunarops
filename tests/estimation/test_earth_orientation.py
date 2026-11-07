"""Earth-orientation corrections per station night, eliminated as random effects with tabulated prior sigmas."""

import numpy as np
import pytest

from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.classes.time import Epoch, TimeScale
from lunarops.estimation.adjustment_plan import (
    AccuracyScreeningSettings,
    EstimateControl,
    LlrAdjustmentSettings,
    ResidualScreening,
    RobustWeightSettings,
    VarianceComponentSettings,
)
from lunarops.estimation.adjustment_preprocessing import screen_observations
from lunarops.estimation.adjustment_solver import LlrAdjustmentSolver
from lunarops.estimation.earth_orientation_effects import EarthOrientationUncertainty, earth_orientation_partials
from tests.estimation.test_robust_vce import COMPONENT as _COMPONENT
from tests.estimation.test_robust_vce import START_JD, Linear

STATION = np.array([4_581_692.0, 556_196.0, 4_389_355.0])
SIGMAS = np.array([0.05, 1.0, 1.0])  # UT1 ms, x_p and y_p mas
COMPONENT = type(_COMPONENT)("A", type(_COMPONENT.interval)("STA_A", _COMPONENT.interval.start, None))


def _problem(nights=300, per_night=6, noise=0.02, seed=3):
    """One station; the Moon crosses the sky during each night, so the partials change within a night."""
    rng = np.random.default_rng(seed)
    n = nights * per_night
    night = np.repeat(np.arange(nights), per_night)
    hours = np.tile(np.linspace(-3.0, 3.0, per_night), nights)
    jd = START_JD + night + (hours + 12.0) / 24.0
    hour_angle = np.deg2rad(15.0 * hours) + np.arctan2(STATION[1], STATION[0])
    declination = np.deg2rad(rng.uniform(-25.0, 25.0, nights))[night]
    to_moon = np.stack([np.cos(declination) * np.cos(hour_angle), np.cos(declination) * np.sin(hour_angle),
                        np.sin(declination)], axis=1)
    station_partials = -to_moon
    eop = earth_orientation_partials(station_partials, np.repeat(STATION[None, :], n, axis=0))
    truth_eop = rng.normal(size=(nights, 3)) * SIGMAS
    rows = np.column_stack([np.ones(n), np.cos(2 * np.pi * night / 27.3), (night - nights / 2) / nights])
    truth = np.array([0.01, -0.02, 0.005])
    values = rows @ truth + np.einsum("ij,ij->i", eop, truth_eop[night]) + noise * rng.normal(size=n)
    equations = EquationBatch.concatenate([
        EquationBatch.single(oc=float(value), sigma=noise, observation_id=index, station="STA_A", reflector="REF",
                             epoch=Epoch(float(day), 0.0, TimeScale.UTC),
                             receive_epoch=Epoch(float(day), 0.0, TimeScale.UTC), wavelength_nm=532.0,
                             station_position_itrf=partial)
        for index, (value, day, partial) in enumerate(zip(values, jd, station_partials, strict=True))
    ])
    return equations, rows, truth, truth_eop


def _solve(equations, rows, *, eop, robust=False, sessions=False):
    settings = LlrAdjustmentSettings(
        residual=ResidualScreening(maximumAbsoluteM=None, maximumAbsoluteByStationM=None),
        variance_components=VarianceComponentSettings((COMPONENT,)),
        adjustment=EstimateControl(maxIterationCount=3, convergenceThreshold=1.0e-6, computeResiduals=True,
                                   estimateVarianceFactors=True, estimateRobustWeights=robust,
                                   sessionCorrelation=sessions, earthOrientationCorrection=eop),
        accuracy_screening=AccuracyScreeningSettings(minimumOneWayM=1.0e-3, minimumFractionOfGroupMedian=0.1),
        robust_weights=RobustWeightSettings(model="igg3", k0=1.5, k1=6.0),
    )
    block = Linear(rows)
    parametrization = ParametrizationList([block])
    domain = screen_observations(equations, parametrization, residual=settings.residual,
                                 reported_sigma=settings.accuracy_screening,
                                 variance_components=settings.variance_components)
    table = EarthOrientationUncertainty(np.array([0.0, 1.0e5]), np.vstack([SIGMAS, SIGMAS]))
    solver = LlrAdjustmentSolver(equation_source=lambda _: equations, parametrization=parametrization,
                                 settings=settings, observation_domain=domain,
                                 earth_orientation_uncertainty=table if eop else None,
                                 station_positions={"STA_A": STATION} if eop else None)
    return solver.run(), block


def test_earth_orientation_corrections_absorb_nightly_rotation_errors():
    equations, rows, truth, truth_eop = _problem()
    with_eop, block = _solve(equations, rows, eop=True)
    without, _ = _solve(equations, rows, eop=False)

    # The white sigma is recovered with the corrections and inflated by the rotation errors without them.
    assert with_eop.sigma_factors["A"] == pytest.approx(1.0, rel=0.06)
    assert without.sigma_factors["A"] > 1.3
    formal = with_eop.sigma0_post * np.sqrt(np.diag(with_eop.cofactor))
    assert np.all(np.abs(block.values - truth) < 3.0 * formal)
    records = with_eop.earth_orientation
    assert len(records) == 300 and records[0]["station"] == "STA_A" and records[0]["observationCount"] == 6
    # One station observes the range effect of the corrections, not UT1 and the polar motion separately.
    estimated = np.array([[record["ut1Ms"], record["xpMas"], record["ypMas"]] for record in records])
    partials = earth_orientation_partials(equations.station_position_itrf,
                                          np.repeat(STATION[None, :], len(equations), axis=0))
    night = np.repeat(np.arange(300), 6)
    effect, true_effect = (np.einsum("ij,ij->i", partials, values[night]) for values in (estimated, truth_eop))
    assert np.corrcoef(effect, true_effect)[0, 1] > 0.8
    assert records[0]["ut1MsPriorSigma"] == pytest.approx(SIGMAS[0])
    assert 0.0 < records[0]["ut1MsSigma"] < SIGMAS[0]


def test_earth_orientation_corrections_combine_with_session_offsets_and_robust_weights():
    equations, rows, truth, _ = _problem(seed=4)
    result, block = _solve(equations, rows, eop=True, robust=True, sessions=True)

    assert result.converged
    assert np.all(np.abs(block.values - truth) < 3.0 * result.sigma0_post * np.sqrt(np.diag(result.cofactor)))
    # No session errors were simulated: tau stays small next to the 20 mm white noise.
    assert result.variance_components[0]["sessionSigmaM"] < 0.008


def test_earth_orientation_correction_needs_its_uncertainties():
    equations, rows, _, _ = _problem(nights=10)
    settings = LlrAdjustmentSettings(
        residual=ResidualScreening(maximumAbsoluteM=None, maximumAbsoluteByStationM=None),
        variance_components=VarianceComponentSettings((COMPONENT,)),
        adjustment=EstimateControl(maxIterationCount=1, convergenceThreshold=0.0, computeResiduals=True,
                                   estimateVarianceFactors=True, estimateRobustWeights=False,
                                   earthOrientationCorrection=True),
        accuracy_screening=AccuracyScreeningSettings(minimumOneWayM=1.0e-3, minimumFractionOfGroupMedian=0.1),
        robust_weights=RobustWeightSettings(model="igg3", k0=1.5, k1=6.0),
    )
    with pytest.raises(ValueError, match="Earth-orientation uncertainties"):
        LlrAdjustmentSolver(equation_source=lambda _: equations, parametrization=ParametrizationList([Linear(rows)]),
                            settings=settings, observation_domain=None)


def test_c04_uncertainties_are_read_in_ms_and_mas(tmp_path):
    path = tmp_path / "c04.txt"
    path.write_text(
        "# header\n"
        "1990   1   1   0  47892.00    0.1 0.2 0.3 0 0 0 0 0.002    0.000700    0.000800  0.0000300 0 0 0 0 0\n"
        "1990   1   2   0  47893.00    0.1 0.2 0.3 0 0 0 0 0.002    0.000900    0.001000  0.0000500 0 0 0 0 0\n"
    )
    table = EarthOrientationUncertainty.from_c04(path)
    sigmas = table.at(np.array([47892.5 + 2_400_000.5]))[0]

    assert sigmas == pytest.approx([0.04, 0.8, 0.9])
    with pytest.raises(ValueError, match="outside"):
        table.at(np.array([2_400_000.5]))
