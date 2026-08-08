import numpy as np
import pytest

from core.models.arima import ArimaForecaster
from core.models.baseline import DriftForecaster, SeasonalNaiveForecaster
from core.models.heuristics import (
    SeasonalWeightedAverageForecaster,
    SmoothedTrendForecaster,
)
from core.models.statistical import ETSForecaster, ThetaForecaster


def test_seasonal_naive_forecaster():
    y = np.array([10, 20, 30, 40, 15, 25, 35, 45], dtype=float)
    forecaster = SeasonalNaiveForecaster()
    fc = forecaster.fit_predict(y, horizon=4, seasonal_period=4)
    assert len(fc.point) == 4
    np.testing.assert_array_equal(fc.point, [15, 25, 35, 45])


def test_drift_forecaster():
    y = np.array([10, 20, 30, 40], dtype=float)
    forecaster = DriftForecaster()
    fc = forecaster.fit_predict(y, horizon=3, seasonal_period=1)
    assert len(fc.point) == 3
    # slope = (40-10)/3 = 10 -> [50, 60, 70]
    np.testing.assert_allclose(fc.point, [50, 60, 70])


@pytest.mark.parametrize(
    "forecaster",
    [
        ArimaForecaster(),
        ETSForecaster(),
        ThetaForecaster(),
        SmoothedTrendForecaster(),
        SeasonalWeightedAverageForecaster(),
    ],
    ids=lambda f: f.name,
)
def test_forecaster_returns_well_formed_band(forecaster):
    y = np.sin(np.linspace(0, 10, 30)) + 10.0
    with np.errstate(all="ignore"):
        fc = forecaster.fit_predict(y, horizon=5, seasonal_period=1)
    assert len(fc.point) == len(fc.lower) == len(fc.upper) == 5
    assert np.all(fc.lower <= fc.point + 1e-9)
    assert np.all(fc.point <= fc.upper + 1e-9)
    assert np.all(np.isfinite(fc.point))
