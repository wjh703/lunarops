"""Earth/Moon translational dynamics, force models, gravity, and integration."""

from lunarops.classes.ephemerides.body_ids import BodyId

from .bodies import GravitatingBody
from .context import AccelerationComponents, DynamicsEpochData, ForceEvaluationContext, StateHistoryProvider
from .earth_gravity_orientation import (
    DE440_EARTH_POLE_OFFSET_X_ARCSEC,
    DE440_EARTH_POLE_OFFSET_Y_ARCSEC,
    de430_earth_fixed2inertial_matrix,
    de440_earth_fixed2inertial_matrix,
    de440_earth_pole_vector_inertial,
)
from .force_models import (
    EarthTideForce,
    EihPointMassForce,
    FigureForce,
    ForceModel,
    LenseThirringForce,
    LunarDegree2GravityCorrectionForce,
    LunarForceGroup,
    NewtonianPointMassForce,
    SolarJ2Force,
    SolarRadiationPressureForce,
    TimeVaryingEarthFigureForce,
)
from .forces import (
    EarthTideModel,
    EarthTideParameters,
    PointMassGravityCache,
    PointMassGravityEvaluation,
    SolarSourceParameters,
    build_point_mass_gravity_cache,
    evaluate_eih_correction,
    evaluate_newtonian_point_mass_system,
)
from .gravity import (
    GravityCoefficients,
    GravityField,
    ensure_gravity_field,
    load_gravity_field,
    make_gravity_coefficients,
    make_j2_coefficients,
)
from .inertia import (
    LunarDegree2GravityEvaluation,
    LunarDegree2GravityModel,
    LunarDegree2GravityParameters,
    degree2_gravity_coefficients_from_inertia,
)
from .integrators import (
    AdamsBashforthMoultonIntegrator,
    IntegratorSettings,
)
from .lunar_dynamics import LunarDynamics
from .orientation import OrientationProvider, inertial2fixed_matrix_from_pole
from .state import BarycentricEarthMoonState, MoonRelativeState
from .trajectory import LunarTrajectory

__all__ = [
    "DE440_EARTH_POLE_OFFSET_X_ARCSEC",
    "DE440_EARTH_POLE_OFFSET_Y_ARCSEC",
    "AccelerationComponents",
    "AdamsBashforthMoultonIntegrator",
    "BarycentricEarthMoonState",
    "BodyId",
    "DynamicsEpochData",
    "EarthTideForce",
    "EarthTideModel",
    "EarthTideParameters",
    "EihPointMassForce",
    "FigureForce",
    "ForceEvaluationContext",
    "ForceModel",
    "GravitatingBody",
    "GravityCoefficients",
    "GravityField",
    "IntegratorSettings",
    "LenseThirringForce",
    "LunarDegree2GravityCorrectionForce",
    "LunarDegree2GravityEvaluation",
    "LunarDegree2GravityModel",
    "LunarDegree2GravityParameters",
    "LunarDynamics",
    "LunarForceGroup",
    "LunarTrajectory",
    "MoonRelativeState",
    "NewtonianPointMassForce",
    "OrientationProvider",
    "PointMassGravityCache",
    "PointMassGravityEvaluation",
    "SolarJ2Force",
    "SolarRadiationPressureForce",
    "SolarSourceParameters",
    "StateHistoryProvider",
    "TimeVaryingEarthFigureForce",
    "build_point_mass_gravity_cache",
    "de430_earth_fixed2inertial_matrix",
    "de440_earth_fixed2inertial_matrix",
    "de440_earth_pole_vector_inertial",
    "degree2_gravity_coefficients_from_inertia",
    "ensure_gravity_field",
    "evaluate_eih_correction",
    "evaluate_newtonian_point_mass_system",
    "inertial2fixed_matrix_from_pole",
    "load_gravity_field",
    "make_gravity_coefficients",
    "make_j2_coefficients",
]
