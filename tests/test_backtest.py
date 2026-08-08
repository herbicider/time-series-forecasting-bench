import numpy as np
from core.backtest import run_backtest, compute_fold_boundaries


def test_fold_boundary_computation():
    n = 36
    horizon = 5
    seasonal_period = 12
    k, boundaries = compute_fold_boundaries(n, horizon, seasonal_period)
    assert k >= 1
    assert len(boundaries) == k
    assert boundaries[-1].test_end == n


def test_run_backtest_basic():
    y = np.linspace(100, 200, 30)
    results, boundaries, warnings = run_backtest(y, horizon=5, seasonal_period=1)
    assert len(results) >= 2
    ok_res = [r for r in results if r.status == "ok"]
    assert len(ok_res) >= 2
    # Winner should have lowest mase
    assert ok_res[0].mean_mase <= ok_res[1].mean_mase
