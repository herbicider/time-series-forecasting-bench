import numpy as np
from typing import List, Tuple, Dict, Any


def calibrate_conformal_intervals(
    fold_residuals: Dict[int, List[float]],  # step h (1-indexed) -> list of absolute residuals across folds
    forecast_point: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[float], bool]:
    """
    Computes split conformal prediction intervals per forecast horizon step.

    Returns:
        lower: np.ndarray (calibrated lower bounds)
        upper: np.ndarray (calibrated upper bounds)
        q_steps: List[float] (quantiles per step)
        is_approximate: bool (True if k < 5 folds)
    """
    horizon = len(forecast_point)
    lower = np.zeros(horizon, dtype=float)
    upper = np.zeros(horizon, dtype=float)
    q_steps = []

    # Check number of folds available
    sample_residuals = fold_residuals.get(1, [])
    k = len(sample_residuals)
    is_approximate = k < 5

    for h in range(1, horizon + 1):
        res_list = fold_residuals.get(h, [])
        if len(res_list) == 0:
            # Fallback if no residuals
            q_h = 1.96 * np.std(forecast_point) if len(forecast_point) > 1 else 1.0
        else:
            abs_res = np.abs(res_list)
            num_k = len(abs_res)
            # Empirical quantile formula: ceil((k+1)(1-alpha)) / k
            p_val = min(1.0, np.ceil((num_k + 1) * (1.0 - alpha)) / num_k) if num_k > 0 else 0.95
            q_h = float(np.quantile(abs_res, p_val))

            # Finite sample correction when k < 5
            if is_approximate and num_k > 0:
                q_h *= (1.0 + 1.0 / num_k)

        q_steps.append(float(q_h))
        point_h = forecast_point[h - 1]
        lower[h - 1] = point_h - q_h
        upper[h - 1] = point_h + q_h

    return lower, upper, q_steps, is_approximate
