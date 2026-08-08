import numpy as np
from core.conformal import calibrate_conformal_intervals


def test_conformal_calibration():
    residuals = {
        1: [1.0, 2.0, 1.5, 2.5, 1.2],
        2: [2.0, 3.0, 2.2, 2.8, 1.9],
    }
    point = np.array([10.0, 20.0])
    lower, upper, q_steps, is_approx = calibrate_conformal_intervals(residuals, point, alpha=0.05)

    assert len(lower) == 2
    assert len(upper) == 2
    assert lower[0] < point[0] < upper[0]
    assert lower[1] < point[1] < upper[1]
