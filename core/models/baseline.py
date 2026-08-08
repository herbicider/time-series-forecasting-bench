import numpy as np
from core.models.base import Forecast, Forecaster


class SeasonalNaiveForecaster:
    name: str = "Last-value baseline"

    def fit_predict(
        self,
        y: np.ndarray,
        horizon: int,
        seasonal_period: int,
    ) -> Forecast:
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            raise ValueError("Empty series provided to SeasonalNaiveForecaster")

        m = max(1, seasonal_period)
        point = np.zeros(horizon, dtype=float)

        for h in range(1, horizon + 1):
            if n >= m:
                # Seasonally lagged index
                idx = n - m + ((h - 1) % m)
                if idx < n:
                    point[h - 1] = y[idx]
                else:
                    point[h - 1] = y[-1]
            else:
                point[h - 1] = y[-1]

        # Calculate in-sample residuals for naive interval estimation
        if n > m:
            residuals = y[m:] - y[:-m]
            sigma = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0
        elif n > 1:
            sigma = np.std(np.diff(y), ddof=1)
        else:
            sigma = 0.0

        # Horizon-scaling factor for interval width
        step_scaling = np.sqrt(np.arange(1, horizon + 1))
        half_width = 1.96 * sigma * step_scaling

        lower = point - half_width
        upper = point + half_width

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


class DriftForecaster:
    name: str = "Trend baseline"

    def fit_predict(
        self,
        y: np.ndarray,
        horizon: int,
        seasonal_period: int,
    ) -> Forecast:
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            raise ValueError("Empty series provided to DriftForecaster")

        if n > 1:
            slope = (y[-1] - y[0]) / (n - 1)
        else:
            slope = 0.0

        steps = np.arange(1, horizon + 1, dtype=float)
        point = y[-1] + steps * slope

        # In-sample residuals
        fitted = y[0] + np.arange(n) * slope
        residuals = y - fitted
        sigma = np.std(residuals, ddof=1) if n > 1 else 0.0

        step_scaling = np.sqrt(steps * (1 + 1 / n))
        half_width = 1.96 * sigma * step_scaling

        lower = point - half_width
        upper = point + half_width

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
