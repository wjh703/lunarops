"""Direct-rejection and IGGIII weight factors of standardized residuals."""

import numpy as np
import pytest

from lunarops.estimation.robust_weights import DIRECT_REJECTION_MODEL, IGG3_MODEL, robust_factors
from tests.tolerances import REL, assert_close


def test_direct_rejection_keeps_residuals_up_to_k0_and_rejects_beyond():
    standardized = np.array([0.0, 1.0, -2.999, 3.0, -3.0, 3.001, -40.0])
    np.testing.assert_array_equal(
        robust_factors(DIRECT_REJECTION_MODEL, 3.0, None, standardized), [1, 1, 1, 1, 1, 0, 0]
    )


def test_igg3_follows_the_three_zone_formula_and_is_even():
    k0, k1 = 1.5, 6.0
    v = np.array([0.0, 1.0, 1.5, 3.0, 4.5, 6.0, 7.0])
    middle = k0 / np.maximum(v, k0) * ((k1 - v) / (k1 - k0)) ** 2
    expected = np.where(v <= k0, 1.0, np.where(v <= k1, middle, 0.0))

    assert_close(robust_factors(IGG3_MODEL, k0, k1, v), expected, REL)
    assert_close(robust_factors(IGG3_MODEL, k0, k1, -v), expected, REL)
    # |v| = 3: k0/|v| ((k1 - |v|)/(k1 - k0))^2 = 0.5 * (3/4.5)^2.
    assert robust_factors(IGG3_MODEL, k0, k1, np.array([3.0]))[0] == pytest.approx(2.0 / 9.0)


def test_igg3_is_continuous_at_both_thresholds():
    k0, k1, eps = 1.5, 6.0, 1e-9
    just_above_k0, just_below_k1 = robust_factors(IGG3_MODEL, k0, k1, np.array([k0 + eps, k1 - eps]))
    assert just_above_k0 == pytest.approx(1.0, abs=1e-8)
    assert just_below_k1 == pytest.approx(0.0, abs=1e-8)


def test_unknown_robust_models_and_invalid_thresholds_are_rejected():
    from lunarops.estimation.adjustment_plan import RobustWeightSettings

    with pytest.raises(ValueError, match="Unknown robust weight model"):
        robust_factors("huber", 1.5, None, np.array([1.0]))
    with pytest.raises(ValueError, match="Unknown robust weight model"):
        RobustWeightSettings(model="huber", k0=1.5)
    with pytest.raises(ValueError, match="0 < k0 < k1"):
        RobustWeightSettings(model=IGG3_MODEL, k0=6.0, k1=1.5)
    with pytest.raises(ValueError, match="k0 > 0"):
        RobustWeightSettings(model=DIRECT_REJECTION_MODEL, k0=0.0, k1=None)
    assert RobustWeightSettings(model=DIRECT_REJECTION_MODEL, k0=3.0, k1=None).k1 is None
