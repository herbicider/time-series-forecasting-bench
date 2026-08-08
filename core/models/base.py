from dataclasses import dataclass
from typing import Protocol, Dict, Optional, Any
import numpy as np


@dataclass
class Forecast:
    point: np.ndarray        # shape (horizon,)
    lower: np.ndarray        # shape (horizon,) — 2.5th percentile
    upper: np.ndarray        # shape (horizon,) — 97.5th percentile
    raw_quantiles: Optional[Dict[float, np.ndarray]] = None   # model-native quantiles, if any

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "point": self.point.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }
        if self.raw_quantiles is not None:
            res["raw_quantiles"] = {
                str(k): v.tolist() for k, v in self.raw_quantiles.items()
            }
        return res


class Forecaster(Protocol):
    name: str

    def fit_predict(
        self,
        y: np.ndarray,
        horizon: int,
        seasonal_period: int,
    ) -> Forecast:
        ...
