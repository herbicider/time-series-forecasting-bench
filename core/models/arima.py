import numpy as np
from core.models.base import Forecast, Forecaster


class ArimaForecaster:
    name: str = "ARIMA"

    def fit_predict(
        self,
        y: np.ndarray,
        horizon: int,
        seasonal_period: int,
    ) -> Forecast:
        y = np.asarray(y, dtype=float)
        m = max(1, seasonal_period)
        if len(y) < 2 * m:
            # Fall back to m=1 if insufficient data for seasonal period
            m = 1

        try:
            from statsforecast.models import AutoARIMA
            model = AutoARIMA(season_length=m)
            model.fit(y=y)
            pred = model.predict(h=horizon, level=[95])
            point = np.asarray(pred["mean"], dtype=float)
            lower = np.asarray(pred.get("lo-95", pred.get("lo_95", point - 1.96 * np.std(y))), dtype=float)
            upper = np.asarray(pred.get("hi-95", pred.get("hi_95", point + 1.96 * np.std(y))), dtype=float)
        except Exception as e:
            # Fallback using statsmodels or simple exponential smoothing / autoregression if statsforecast fails
            point, lower, upper = self._fallback_arima(y, horizon, m)

        raw_quantiles = {
            0.025: lower,
            0.5: point,
            0.975: upper,
        }

        return Forecast(
            point=point,
            lower=lower,
            upper=upper,
            raw_quantiles=raw_quantiles,
        )

    def _fallback_arima(self, y: np.ndarray, horizon: int, m: int):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            # Simple ARIMA(1,1,1) or ARIMA(1,0,0)
            order = (1, 1, 1) if len(y) > 10 else (1, 0, 0)
            model = ARIMA(y, order=order)
            res = model.fit()
            fc = res.get_forecast(steps=horizon)
            point = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)
            lower = ci[:, 0]
            upper = ci[:, 1]
            return point, lower, upper
        except Exception:
            # Ultimate fallback to naive drift
            slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.0
            point = y[-1] + np.arange(1, horizon + 1) * slope
            sigma = np.std(np.diff(y)) if len(y) > 1 else 1.0
            lower = point - 1.96 * sigma
            upper = point + 1.96 * sigma
            return point, lower, upper
