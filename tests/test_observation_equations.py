import numpy as np

from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.time import Epoch, TimeScale

_UTC_EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def test_observation_equation_preserves_residual_epoch_and_partials():
    equation = EquationBatch.single(
        oc=0.25,
        sigma=0.01,
        reflector_position_pa=np.array([1.0, 2.0, 3.0]),
        observation_id=7,
        station="STA",
        reflector="REF",
        epoch=_UTC_EPOCH,
        wavelength_nm=532.0,
        receive_epoch=_UTC_EPOCH,
    )

    assert equation.oc[0] == 0.25
    assert equation.epoch(0) == _UTC_EPOCH
    assert np.allclose(equation.reflector_position_pa[0], [1.0, 2.0, 3.0])
