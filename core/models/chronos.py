"""Amazon Chronos-2 wrapper.

Only instantiated when the `chronos` package is genuinely importable (see
core/models/manager.py). Failures raise rather than falling back, so the model
name in the report always reflects what actually produced the numbers.
"""

import logging

import numpy as np

from core.models.base import Forecast

logger = logging.getLogger(__name__)

REPO_ID = "amazon/chronos-2"


class ChronosForecaster:
    name: str = "Amazon Chronos-2"

    def __init__(self, checkpoint: str = REPO_ID, device: str = "cpu"):
        self.checkpoint = checkpoint
        self.device = device
        self._pipeline = None

    def load_model(self):
        if self._pipeline is not None:
            return
        import torch
        from chronos import Chronos2Pipeline

        # bfloat16 matmul is emulated on most consumer CPUs and numpy cannot
        # represent it at all, so float32 is both faster and safer off-GPU.
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.checkpoint,
            device_map=self.device,
            torch_dtype=dtype,
        )
        logger.info("Chronos-2 pipeline loaded successfully")

    def fit_predict(self, y: np.ndarray, horizon: int, seasonal_period: int) -> Forecast:
        y = np.asarray(y, dtype=float)
        self.load_model()

        import torch

        # Chronos-2 signature: predict_quantiles(inputs, prediction_length,
        # quantile_levels) -> (quantiles, mean), where each is a *list* of
        # tensors, one per input series. The 1.x API took `context=` and
        # returned a single tensor; using it raises TypeError.
        # Chronos-2 is a multivariate model: it wants
        # (n_series, n_variates, history_length), and rejects a bare 1-D
        # series. One series, one variate.
        context = torch.tensor(y, dtype=torch.float32).reshape(1, 1, -1)

        quantiles, _mean = self._pipeline.predict_quantiles(
            inputs=context,
            prediction_length=horizon,
            quantile_levels=[0.025, 0.5, 0.975],
        )

        # quantiles[0] is (n_variates, prediction_length, n_quantiles); this is
        # a univariate series, so collapse the variate axis.
        q = quantiles[0]
        q_np = q.to(torch.float32).cpu().numpy()
        q_np = q_np.reshape(-1, q_np.shape[-2], q_np.shape[-1])[0][:horizon]

        lower = q_np[:, 0]
        point = q_np[:, 1]
        upper = q_np[:, 2]

        return Forecast(
            point=point,
            lower=lower,
            upper=upper,
            raw_quantiles={0.025: lower, 0.5: point, 0.975: upper},
        )
