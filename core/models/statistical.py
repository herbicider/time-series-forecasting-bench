"""Classical statistical forecasters from statsforecast.

ETS and Theta are strong on the monthly and weekly business series this tool
targets — Theta won the M3 competition — and they ship in the dependency we
already have, so they cost nothing extra to include.
"""

import numpy as np

from core.models.base import Forecast


def _band(point: np.ndarray, y: np.ndarray, horizon: int):
    sigma = np.std(np.diff(y), ddof=1) if len(y) > 2 else float(np.std(y) or 1.0)
    scale = np.sqrt(np.arange(1, horizon + 1))
    return point - 1.96 * sigma * scale, point + 1.96 * sigma * scale


def _extract(pred: dict, key_mean: str, point: np.ndarray, y: np.ndarray, horizon: int):
    """Pull the 95% band out of a statsforecast prediction dict, tolerating
    both the `lo-95` and `lo_95` key spellings across versions."""
    lower = pred.get("lo-95", pred.get("lo_95"))
    upper = pred.get("hi-95", pred.get("hi_95"))
    if lower is None or upper is None:
        return _band(point, y, horizon)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


class _StatsForecastModel:
    """Shared plumbing: seasonal period guard, fit, predict, interval extraction."""

    name = "statistical"

    def _build(self, season_length: int):
        raise NotImplementedError

    def fit_predict(self, y: np.ndarray, horizon: int, seasonal_period: int) -> Forecast:
        y = np.asarray(y, dtype=float)
        m = max(1, seasonal_period)
        # Seasonal models need at least two full cycles to estimate a season.
        if len(y) < 2 * m:
            m = 1

        model = self._build(m)
        model.fit(y=y)
        pred = model.predict(h=horizon, level=[95])
        point = np.asarray(pred["mean"], dtype=float)
        lower, upper = _extract(pred, "mean", point, y, horizon)

        return Forecast(
            point=point,
            lower=lower,
            upper=upper,
            raw_quantiles={0.025: lower, 0.5: point, 0.975: upper},
        )


class ETSForecaster(_StatsForecastModel):
    name: str = "Exponential Smoothing (ETS)"

    def _build(self, season_length: int):
        from statsforecast.models import AutoETS

        return AutoETS(season_length=season_length)


class ThetaForecaster(_StatsForecastModel):
    name: str = "Theta"

    def _build(self, season_length: int):
        from statsforecast.models import AutoTheta

        return AutoTheta(season_length=season_length)
