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

        # TimesFM 2.5 signature: forecast(horizon: int, inputs: list[np.ndarray])
        # -> (point, quantiles). There is no `freq` argument; the 1.x API had
        # one and passing it raises TypeError.
        point_forecast, quantile_forecast = self._model.forecast(
            horizon=horizon,
            inputs=[y],
        )

        point = np.asarray(point_forecast, dtype=float)
        point = point.reshape(-1, point.shape[-1])[0][:horizon]

        lower, upper = self._interval(y, point, quantile_forecast, horizon)

        return Forecast(
            point=point,
            lower=lower,
            upper=upper,
            raw_quantiles={0.025: lower, 0.5: point, 0.975: upper},
        )

    @staticmethod
    def _interval(y, point, quantile_forecast, horizon):
        """Turn TimesFM's deciles into a 95% band.

        The model emits [mean, 0.1, 0.2, ... 0.9] along the last axis — deciles
        only, so there is no native 2.5%/97.5% pair to read off. Rather than
        pass off the 10th/90th as "95%", estimate the scale from the decile
        spread (q90 - q10 spans 2 x 1.2816 sigma under a normal) and widen to
        1.96 sigma. Every other forecaster in the app reports a 95% band, and
        the displayed interval is conformally recalibrated regardless.
        """
        if quantile_forecast is not None and len(quantile_forecast) > 0:
            q = np.asarray(quantile_forecast, dtype=float)
            if q.ndim >= 2 and q.shape[-1] >= 10:
                q = q.reshape(-1, q.shape[-2], q.shape[-1])[0][:horizon]
                q10, q90 = q[:, 1], q[:, 9]
                sigma = np.maximum((q90 - q10) / (2 * 1.2816), 1e-9)
                return point - 1.96 * sigma, point + 1.96 * sigma

        sigma = np.std(np.diff(y)) if len(y) > 1 else 1.0
        scale = np.sqrt(np.arange(1, horizon + 1))
        return point - 1.96 * sigma * scale, point + 1.96 * sigma * scale
