"""Lunar orbit and rotation dynamics.

The package propagates the Moon relative to the Earth in the BCRS, with the
Earth and all other bodies prescribed by an ephemeris, and optionally
integrates the lunar mantle and fluid-core rotation. Quantities are in SI
units, time is TDB, translational vectors use ICRF axes and rotational
vectors use Moon principal axes (PA).

Main entry points:

- :class:`LunarDynamics` - the right-hand side;
- :class:`AdamsBashforthMoultonIntegrator` - the fixed-step integrator;
- the force-model records of :mod:`.force_models` and the torque records of
  :mod:`.torques`;
- :class:`LunarRotationDynamics` and :class:`LunarDeformation` - rotation
  and delayed deformation;
- :func:`load_gravity_field` - ICGEM gravity-field loading.
"""

from .bodies import GravitatingBody
from .deformation import (
    LunarDeformation,
    LunarDeformationParameters,
    LunarDegree2GravityModel,
    LunarDegree2GravityParameters,
)
from .earth_figure_orientation import de440_earth_fixed2inertial_matrix, inertial2fixed_matrix_from_pole
from .epoch_data import OrientationProvider
from .force_models import (
    EarthFigureForce,
    EarthTideForce,
    EihPointMassForce,
    ForceModel,
    LenseThirringForce,
    LunarForceGroup,
    MoonFigureForce,
    NewtonianPointMassForce,
    SolarJ2Force,
    SolarRadiationPressureForce,
)
from .forces import EarthTideModel, EarthTideParameters, SolarSourceParameters
from .gravity import load_gravity_field, make_gravity_coefficients, make_j2_coefficients
from .integrators import AdamsBashforthMoultonIntegrator, IntegratorSettings
from .lunar_dynamics import LunarDynamics
from .rotation import LunarCoreParameters, LunarRotationDynamics, LunarRotationState
from .torques import EarthFigureTorque, LunarFigureTorque, LunarTorqueGroup, PointMassGravityTorque

__all__ = [
    "AdamsBashforthMoultonIntegrator",
    "EarthFigureForce",
    "EarthFigureTorque",
    "EarthTideForce",
    "EarthTideModel",
    "EarthTideParameters",
    "EihPointMassForce",
    "ForceModel",
    "GravitatingBody",
    "IntegratorSettings",
    "LenseThirringForce",
    "LunarCoreParameters",
    "LunarDeformation",
    "LunarDeformationParameters",
    "LunarDegree2GravityModel",
    "LunarDegree2GravityParameters",
    "LunarDynamics",
    "LunarFigureTorque",
    "LunarForceGroup",
    "LunarRotationDynamics",
    "LunarRotationState",
    "LunarTorqueGroup",
    "MoonFigureForce",
    "NewtonianPointMassForce",
    "OrientationProvider",
    "PointMassGravityTorque",
    "SolarJ2Force",
    "SolarRadiationPressureForce",
    "SolarSourceParameters",
    "de440_earth_fixed2inertial_matrix",
    "inertial2fixed_matrix_from_pole",
    "load_gravity_field",
    "make_gravity_coefficients",
    "make_j2_coefficients",
]
