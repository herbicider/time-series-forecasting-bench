"""Google TimesFM 2.5 wrapper.

This class is only ever instantiated when the `timesfm` package is genuinely
importable (see core/models/manager.py). If loading or inference then fails,
it raises — the backtest records the model as "Did not run" with the reason.
It must never quietly substitute another method under Google's name, because
the ranking table, the verdict sentence and the exported PDF all present the
model name as fact.
"""

import logging

import numpy as np

from core.models.base import Forecast

logger = logging.getLogger(__name__)

REPO_ID = "google/timesfm-2.5-200m-pytorch"


class TimesFMForecaster:
    name: str = "Google TimesFM 2.5"

    def __init__(self, checkpoint: str = REPO_ID, device: str = "cpu"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def load_model(self):
        if self._model is not None:
            return
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch(huggingface_repo_id=self.checkpoint)
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=64,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = model
        logger.info("TimesFM model loaded successfully")

    def fit_predict(self, y: np.ndarray, horizon: int, seasonal_period: int) -> Forecast:
        y = np.asarray(y, dtype=float)
        self.load_model()

        # TimesFM's `freq` is a categorical granularity indicator, not a
        # seasonal period: 0 = high frequency (daily and finer), 1 = medium
        # (weekly/monthly), 2 = low (quarterly/yearly). Passing the seasonal
        # period here used to send out-of-range values like 12 or 52.
        if seasonal_period >= 12:
            freq_indicator = 0
        elif seasonal_period >= 4:
            freq_indicator = 1
        else:
            freq_indicator = 2

        point_forecast, quantile_forecast = self._model.forecast(
            inputs=[y],
            freq=[freq_indicator],
            forecast_horizon=horizon,
        )
        point = np.squeeze(np.asarray(point_forecast, dtype=float))

        if quantile_forecast is not None and len(quantile_forecast) > 0:
            q_arr = np.squeeze(np.asarray(quantile_forecast, dtype=float))
            if q_arr.ndim == 2:
                lower = q_arr[:, 0]
                upper = q_arr[:, -1]
            else:
                sigma = np.std(np.diff(y)) if len(y) > 1 else 1.0
                lower = point - 1.96 * sigma
                upper = point + 1.96 * sigma
        else:
            sigma = np.std(np.diff(y)) if len(y) > 1 else 1.0
            lower = point - 1.96 * sigma
            upper = point + 1.96 * sigma

        return Forecast(
            point=point,
            lower=lower,
            upper=upper,
            raw_quantiles={0.025: lower, 0.5: point, 0.975: upper},
        )
