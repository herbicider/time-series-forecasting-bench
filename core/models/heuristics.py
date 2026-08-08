"""Built-in forecasters that always work, with no download and no dependencies.

These two methods used to live inside timesfm.py and chronos.py as silent
`_fallback_forecast` methods, and their output was reported to the user under
Google's and Amazon's names. They are perfectly reasonable classical methods —
they just are not foundation models — so they now stand on their own and are
labelled for what they are.
"""

import numpy as np

from core.models.base import Forecast


def _interval_from_residuals(y: np.ndarray, point: np.ndarray, horizon: int):
    """Symmetric 95% band that widens with the square root of the horizon."""
    residuals = y[1:] - y[:-1] if len(y) > 1 else np.array([1.0])
    sigma = np.std(residuals, ddof=1) if len(residuals) > 1 else 1.0
    scale = np.sqrt(np.arange(1, horizon + 1))
    half_width = 1.96 * sigma * scale
    return point - half_width, point + half_width


class SmoothedTrendForecaster:
    """Holt-style exponential level + trend with a seasonal nudge."""

    name: str = "Smoothed Trend (built-in)"

    def fit_predict(self, y: np.ndarray, horizon: int, seasonal_period: int) -> Forecast:
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            raise ValueError("Empty series provided to SmoothedTrendForecaster")
        m = max(1, seasonal_period)

        alpha = 0.3
        level = y[0]
        trend = 0.0
        for i in range(1, n):
            prev_level = level
            level = alpha * y[i] + (1 - alpha) * (level + trend)
            trend = 0.2 * (level - prev_level) + 0.8 * trend

        point = np.zeros(horizon)
        for h in range(1, horizon + 1):
            s_idx = n - m + ((h - 1) % m) if n >= m else -1
            seasonal_comp = (y[s_idx] - y[-1]) if n >= m else 0.0
            point[h - 1] = level + h * trend + 0.5 * seasonal_comp

        lower, upper = _interval_from_residuals(y, point, horizon)
        return Forecast(
            point=point,
            lower=lower,
            upper=upper,
            raw_quantiles={0.025: lower, 0.5: point, 0.975: upper},
        )


class SeasonalWeightedAverageForecaster:
    """Exponentially weighted recent average, scaled by a seasonal factor."""

    name: str = "Seasonal Weighted Average (built-in)"

    def fit_predict(self, y: np.ndarray, horizon: int, seasonal_period: int) -> Forecast:
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            raise ValueError("Empty series provided to SeasonalWeightedAverageForecaster")
        m = max(1, seasonal_period)

        weights = np.exp(np.linspace(-1, 0, min(n, 12)))
        weights /= weights.sum()
        base_val = np.dot(y[-len(weights):], weights)

        slope = (y[-1] - y[max(0, n - 6)]) / min(n - 1, 6) if n > 1 else 0.0

        series_mean = np.mean(y)
        point = np.zeros(horizon)
        for h in range(1, horizon + 1):
            s_idx = n - m + ((h - 1) % m) if n >= m else -1
            seasonal_factor = (y[s_idx] / series_mean) if (n >= m and series_mean != 0) else 1.0
            point[h - 1] = (base_val + h * slope * 0.8) * (0.5 + 0.5 * seasonal_factor)

        lower, upper = _interval_from_residuals(y, point, horizon)
        return Forecast(
            point=point,
            lower=lower,
            upper=upper,
            raw_quantiles={0.025: lower, 0.5: point, 0.975: upper},
        )
