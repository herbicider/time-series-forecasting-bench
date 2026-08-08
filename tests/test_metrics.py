import numpy as np
import pytest
from core.metrics import (
    calculate_mase,
    calculate_mae,
    calculate_rmse,
    calculate_coverage,
    calculate_pinball_loss,
    evaluate_forecast,
)


def test_metrics_calculations():
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])
    train_y = np.array([5.0, 10.0, 15.0, 20.0])

    mase = calculate_mase(actual, pred, train_y, seasonal_period=1)
    mae = calculate_mae(actual, pred)
    rmse = calculate_rmse(actual, pred)

    assert mae == pytest.approx(2.333333, abs=1e-4)
    assert mase > 0
    assert rmse > mae

    lower = pred - 5.0
    upper = pred + 5.0
    cov = calculate_coverage(actual, lower, upper)
    assert cov == 1.0


def test_evaluate_forecast():
    actual = np.array([10.0, 20.0])
    point = np.array([10.0, 20.0])
    lower = np.array([5.0, 15.0])
    upper = np.array([15.0, 25.0])
    train_y = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = evaluate_forecast(actual, point, lower, upper, train_y, seasonal_period=1)
    assert metrics["mase"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["coverage"] == 1.0
