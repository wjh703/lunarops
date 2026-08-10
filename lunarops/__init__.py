"""lunarops — GROOPS-inspired Lunar Laser Ranging processing system.

Layers (see docs/ARCHITECTURE.md):
  config/    class registry, config loader, run context
  base/      constants, array validation, parameter names
  fileio/    MINI/CRD/catalog/normal-equation/result IO
  classes/   time values and polymorphic model classes (ephemerides, frames,
             delays, displacements, parametrizations, observation equations)
  estimation/  generic adjustment and normal-equation accumulation
  programs/  one task per program, driven by a run config
  parallel/  MPI observation processing and worker caches
"""

__version__ = "0.1.0"
