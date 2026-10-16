"""IAU 2006/2000A CIP used as an alternative Earth gravity symmetry axis.

The CIP is expressed in GCRS axes aligned with BCRS/ICRF axes. It is neither
an observed gravity pole nor the instantaneous angular-velocity axis. No EOP,
polar motion or celestial-pole offsets are applied. ERFA includes frame bias.
"""

import erfa
import numpy as np

from lunarops.classes.time import TimeScaleConverter

from .orientation import inertial2fixed_matrix_from_pole

_TIME = TimeScaleConverter()


def iau2006_earth_pole_vector_inertial(epoch_tdb) -> np.ndarray:
    """Return the model CIP in GCRS axes at the TT corresponding to TDB."""
    tt = _TIME.tdb2tt(epoch_tdb)
    # pnm06a maps GCRS -> true equator/equinox of date; its row 2 is the CIP.
    pole = np.array(erfa.pnm06a(tt.jd1, tt.jd2)[2], copy=True)
    pole /= np.linalg.norm(pole)
    pole.setflags(write=False)
    return pole


def iau2006_earth_fixed2inertial_matrix(epoch_tdb) -> np.ndarray:
    """Gravity axes -> GCRS axes; use the same arbitrary longitude as DE440."""
    matrix = np.ascontiguousarray(inertial2fixed_matrix_from_pole(
        iau2006_earth_pole_vector_inertial(epoch_tdb)).T)
    matrix.setflags(write=False)
    return matrix
