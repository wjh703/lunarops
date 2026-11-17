"""An analytic ``lunarDynamicsPartials`` table: smooth functions of time with realistic magnitudes."""

from __future__ import annotations

import numpy as np

from lunarops.classes.dynamics.parameters import INITIAL_VALUE_NAMES, parameter_definition
from lunarops.fileio.lunar_partials import LunarDynamicsPartials

QUANTITIES = ("r0_x", "v0_y", "psi0", "wm0_z", "gm_emb", "beta")
# Size of d(position, m)/d(quantity) and d(Euler angles, rad)/d(quantity): the columns of a range design
# built from them span about twenty orders of magnitude, as those of an integrated orbit and rotation do.
_POSITION_SCALE = {"r0_x": 3.0, "v0_y": 2.0e6, "psi0": 5.0e2, "wm0_z": 4.0e8, "gm_emb": 3.0e-7, "beta": 2.0e6}
_EULER_SCALE = {"r0_x": 2.0e-9, "v0_y": 1.0e-3, "psi0": 1.5, "wm0_z": 1.0e6, "gm_emb": 1.0e-15, "beta": 40.0}
INITIAL_STATE = np.array(
    [3.6e8, 1.2e8, 4.0e7, -300.0, 900.0, 80.0, 0.3, 0.4, 2.2, 1.0e-8, -2.0e-8, 2.66e-6, 0.0, 1.0e-7, 2.6e-6]
)
NOMINAL = {
    **dict(zip(INITIAL_VALUE_NAMES, INITIAL_STATE.tolist(), strict=True)),
    "gm_emb": 4.035032356e14,
    "beta": 6.31e-4,
}


def partial_values(offsets_s, quantities=QUANTITIES) -> np.ndarray:
    """Return the analytic partials at offsets from the initial epoch, shape ``(n, 15, q)``.

    Every state component and quantity has its own period and phase, so no two columns are proportional.
    """
    days = np.asarray(offsets_s, dtype=float)[:, None, None] / 86400.0
    rows = np.arange(15)[None, :, None]
    columns = np.asarray([QUANTITIES.index(name) for name in quantities])[None, None, :]
    period_days = 9.0 + 2.0 * rows + 5.0 * columns
    shape = np.sin(2.0 * np.pi * days / period_days + 0.7 * rows + 1.3 * columns) + 0.02 * days
    scale = np.ones((15, len(quantities)))
    scale[0:3] = [_POSITION_SCALE[name] for name in quantities]
    scale[6:9] = [_EULER_SCALE[name] for name in quantities]
    return shape * scale[None]


def state_values(offsets_s) -> np.ndarray:
    """Return the analytic state at offsets from the initial epoch, shape ``(n, 15)``; ``INITIAL_STATE`` at 0."""
    days = np.asarray(offsets_s, dtype=float)[:, None] / 86400.0
    rates = np.zeros(15)
    rates[6:9] = [1.0e-4, -2.0e-5, 0.23]  # rad/d; psi grows by a revolution per sidereal month
    return INITIAL_STATE[None] + rates[None] * days + 1.0e-3 * np.abs(INITIAL_STATE)[None] * np.sin(days / 4.0)


def synthetic_partials(
    epoch_tdb_jd1: float,
    epoch_tdb_jd2: float = 0.0,
    *,
    start_s: float = 0.0,
    count: int = 200,
    step_s: float = 21600.0,
    quantities=QUANTITIES,
    dimension: int = 15,
) -> LunarDynamicsPartials:
    """Return the analytic table on a regular grid of ``count`` epochs from ``start_s``."""
    offsets = start_s + step_s * np.arange(count)
    return LunarDynamicsPartials(
        epoch_tdb_jd1,
        epoch_tdb_jd2,
        offsets,
        state_values(offsets)[:, :dimension],
        partial_values(offsets, quantities)[:, :dimension],
        tuple(quantities),
        tuple(parameter_definition(name).unit for name in quantities),
        producer="analytic",
    )


RECOVERY_MODELS = {
    "ephemerides": {"type": "chebyshev", "directory": "kernels", "lunarRelativisticScaleConvention": "alreadyScaled"},
    "earthRotation": {"type": "file", "file": "eop.txt"},
    "troposphere": {"type": "none"},
    "relativity": {"type": "none"},
    "stationDisplacement": [{"type": "iers2010SolidEarthTide"}],
    "reflectorDisplacement": {"type": "lunarSolidTide"},
}
RECOVERY_OUTPUTS = ("solution.txt", "state.txt", "updated_parameters.txt", "report.txt")
_SIDEREAL_DAY_S = 86164.0905


def write_recovery_case(directory, *, count: int = 40):
    """Write an ``LlrProcessing`` case whose O-C is the design of known corrections; return config and truth.

    The static synthetic system is observed on ``count`` nights. The normal points are written with round-trip
    times such that the one-way O-C equals ``design @ truth`` for a ``lunarDynamics`` block on the analytic
    table (a two-sided grid around its initial epoch) and a ``lunarLoveNumber`` block.

    Returns
    -------
    config : dict
        Program config of ``LlrProcessing`` relative to ``directory``.
    truth : dict of str to float
        True corrections by parameter name.
    """
    from dataclasses import replace

    from lunarops.base.constants import C
    from lunarops.classes.dynamics.parameters import LunarDynamicsParameters, write_lunar_dynamics_parameters
    from lunarops.classes.observation import NptDataset, ReflectorRecord, StationRecord
    from lunarops.fileio.catalogs import write_reflector_catalog, write_station_catalog
    from lunarops.fileio.lunar_partials import write_lunar_dynamics_partials
    from lunarops.fileio.normal_points import write_normal_points
    from lunarops.programs.llr_processing import build_processing_problem
    from lunarops.programs.registry import ensure_builtin_programs, resolve_program_config
    from tests.support.observation import (
        TRANSMIT_UTC,
        StaticKernelContext,
        npt_record,
        static_frames,
        station_facing_the_moon,
    )

    ensure_builtin_programs()
    write_station_catalog(
        {"APOLLO": StationRecord(name="Apache Point", itrf_xyz_m=tuple(station_facing_the_moon(static_frames())))},
        directory / "stations.txt",
    )
    write_reflector_catalog(
        {"APOLLO15": ReflectorRecord(name="Apollo 15", moon_fixed_xyz_m=(1_737_400.0, 0.0, 0.0))},
        directory / "reflectors.txt",
    )
    # Inputs recorded by the processing provenance; the static system reads none of them.
    (directory / "kernels").mkdir()
    (directory / "kernels" / "static.bsp").write_bytes(b"static")
    (directory / "eop.txt").touch()
    # The station faces the Moon again after every sidereal day; the offsets within a night vary the geometry.
    records = [
        replace(
            npt_record(index),
            transmit_epoch=TRANSMIT_UTC.shifted(_SIDEREAL_DAY_S * (index % 30) + 2400.0 * np.sin(1.7 * index)),
        )
        for index in range(count)
    ]
    write_normal_points(NptDataset(records), directory / "normal.txt")
    epoch_jd = TRANSMIT_UTC.jd1 + TRANSMIT_UTC.jd2 + 11.0
    write_lunar_dynamics_partials(
        directory / "partials.dat", synthetic_partials(epoch_jd, start_s=-25.0 * 86400.0, count=200)
    )
    write_lunar_dynamics_parameters(LunarDynamicsParameters(NOMINAL, epoch_jd), directory / "nominal_parameters.txt")
    config = {
        **RECOVERY_MODELS,
        "inputFileStationCatalog": "stations.txt",
        "inputFileReflectorCatalog": "reflectors.txt",
        "inputFilesNormalPoints": ["normal.txt"],
        "rangeBias": {"type": "none"},
        "parametrization": [
            {
                "type": "lunarDynamics",
                "inputFilePartials": "partials.dat",
                "inputFileLunarDynamicsParameters": "nominal_parameters.txt",
                "quantities": list(QUANTITIES),
            },
            {"type": "lunarLoveNumber"},
        ],
        "varianceComponents": [{"id": "A", "station": "APOLLO", "start": "2020-01-01", "endExclusive": None}],
        "processingSteps": [
            {"type": "screenObservations", "residual": {"maximumAbsoluteM": None}},
            {
                "type": "estimate",
                "name": "dynamics",
                "estimateVarianceFactors": False,
                "estimateRobustWeights": False,
                "maxIterationCount": 3,
                "convergenceThreshold": 1.0e-4,
            },
            {
                "type": "writeResults",
                "outputFileSolution": "solution.txt",
                "outputFileReport": "report.txt",
                "outputFileState": "state.txt",
                "outputFileLunarDynamicsParameters": "updated_parameters.txt",
            },
        ],
    }
    resolved = resolve_program_config("LlrProcessing", {**config, "showProgress": False})
    with StaticKernelContext(working_dir=directory) as context:
        context.resolve_defaults()
        problem = build_processing_problem(resolved, context)
        batch = problem.equation_source(1)
        names = [str(name) for name in problem.parametrization.setup(batch)]
        design = problem.parametrization.design(batch)
    # Corrections that each change the range by about a metre.
    metres = np.array([0.6, -0.9, 1.3, -0.4, 0.8, -1.1, -0.043])
    corrections = metres / np.sqrt(np.mean(design**2, axis=0))
    row = {identity: index for index, identity in enumerate(batch.observation_id.tolist())}
    change_m = design @ corrections - batch.oc
    records = [
        replace(record, round_trip_time_s=record.round_trip_time_s + 2.0 * change_m[row[record.index]] / C)
        for record in records
    ]
    write_normal_points(NptDataset(records), directory / "normal.txt")
    return config, dict(zip(names, corrections.tolist(), strict=True))
