import numpy as np
from typing import Dict, Any


def calculate_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train_y: np.ndarray,
    seasonal_period: int = 1,
) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    train_y = np.asarray(train_y, dtype=float)

    mae_val = np.mean(np.abs(actual - forecast))

    m = max(1, seasonal_period)
    if len(train_y) > m:
        scale = np.mean(np.abs(train_y[m:] - train_y[:-m]))
    elif len(train_y) > 1:
        scale = np.mean(np.abs(np.diff(train_y)))
    else:
        scale = 1.0

    if scale < 1e-8:
        scale = 1e-8

    return float(mae_val / scale)


def calculate_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)))


def calculate_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def calculate_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    in_range = (actual >= lower) & (actual <= upper)
    return float(np.mean(in_range))


def calculate_pinball_loss(actual: np.ndarray, quantile_pred: np.ndarray, alpha: float) -> float:
    actual = np.asarray(actual, dtype=float)
    q = np.asarray(quantile_pred, dtype=float)
    err = actual - q
    loss = np.maximum(alpha * err, (alpha - 1.0) * err)
    return float(np.mean(loss))


def evaluate_forecast(
    actual: np.ndarray,
    forecast_point: np.ndarray,
    forecast_lower: np.ndarray,
    forecast_upper: np.ndarray,
    train_y: np.ndarray,
    seasonal_period: int = 1,
) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    point = np.asarray(forecast_point, dtype=float)
    lower = np.asarray(forecast_lower, dtype=float)
    upper = np.asarray(forecast_upper, dtype=float)

    mase = calculate_mase(actual, point, train_y, seasonal_period)
    mae = calculate_mae(actual, point)
    rmse = calculate_rmse(actual, point)
    cov = calculate_coverage(actual, lower, upper)

    pb_025 = calculate_pinball_loss(actual, lower, 0.025)
    pb_500 = calculate_pinball_loss(actual, point, 0.500)
    pb_975 = calculate_pinball_loss(actual, upper, 0.975)
    mean_pinball = (pb_025 + pb_500 + pb_975) / 3.0

    return {
        "mase": mase,
        "mae": mae,
        "rmse": rmse,
        "coverage": cov,
        "pinball_025": pb_025,
        "pinball_50": pb_500,
        "pinball_975": pb_975,
        "pinball": mean_pinball,
    }
