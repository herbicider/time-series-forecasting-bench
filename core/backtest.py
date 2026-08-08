import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Tuple, Optional
import numpy as np

from core.models.base import Forecaster, Forecast
from core.models.manager import build_models
from core.metrics import evaluate_forecast, calculate_coverage
from core.conformal import calibrate_conformal_intervals

logger = logging.getLogger(__name__)


@dataclass
class FoldBoundary:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class ModelBacktestResult:
    model_name: str
    status: str  # "ok" or "error"
    error_reason: Optional[str] = None
    mean_mase: float = float("inf")
    mean_rmse: float = float("inf")
    mean_mae: float = float("inf")
    coverage: float = 0.0
    mean_pinball: float = 0.0
    final_forecast: Optional[Forecast] = None
    calibrated_lower: Optional[np.ndarray] = None
    calibrated_upper: Optional[np.ndarray] = None
    fold_evaluations: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "model": self.model_name,
            "status": self.status,
            "mase": round(self.mean_mase, 3) if self.mean_mase != float("inf") else None,
            "rmse": round(self.mean_rmse, 2) if self.mean_rmse != float("inf") else None,
            "mae": round(self.mean_mae, 2) if self.mean_mae != float("inf") else None,
            "coverage": round(self.coverage, 3),
            "pinball": round(self.mean_pinball, 3) if self.mean_pinball != 0.0 else None,
        }
        if self.error_reason:
            res["error_reason"] = self.error_reason
        return res


def compute_fold_boundaries(n: int, horizon: int, seasonal_period: int) -> Tuple[int, List[FoldBoundary]]:
    min_train = max(2 * seasonal_period, 16)
    if n <= min_train:
        min_train = max(10, n // 2)

    max_possible_folds = math.floor((n - min_train) / horizon) if horizon > 0 else 1
    k = max(1, min(max_possible_folds, 8))

    boundaries = []
    for i in range(1, k + 1):
        test_end = n - (k - i) * horizon
        test_start = test_end - horizon
        train_end = test_start
        train_start = 0

        boundaries.append(
            FoldBoundary(
                fold=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return k, boundaries


def _holdout_coverage(
    fold_points: List[np.ndarray],
    fold_actuals: List[np.ndarray],
    horizon: int,
    fallback_q_steps: List[float],
) -> Tuple[float, bool]:
    """Fraction of actuals inside a range calibrated on earlier folds only.

    Returns (coverage, is_optimistic). is_optimistic is True when there were
    too few folds to hold any out, in which case the number is measured on the
    calibration folds themselves and overstates reliability.
    """
    n_folds = len(fold_points)
    if n_folds == 0:
        return 0.0, True

    # Need at least one calibration fold and one scoring fold.
    if n_folds < 3:
        q_steps = fallback_q_steps
        score_idx = range(n_folds)
        optimistic = True
    else:
        n_cal = max(1, int(math.ceil(n_folds * 0.6)))
        cal_residuals: Dict[int, List[float]] = {h: [] for h in range(1, horizon + 1)}
        for i in range(n_cal):
            abs_err = np.abs(fold_actuals[i] - fold_points[i])
            for step_i, err_val in enumerate(abs_err):
                cal_residuals[step_i + 1].append(float(err_val))
        _, _, q_steps, _ = calibrate_conformal_intervals(
            fold_residuals=cal_residuals,
            forecast_point=np.zeros(horizon),
            alpha=0.05,
        )
        score_idx = range(n_cal, n_folds)
        optimistic = False

    coverages = []
    for i in score_idx:
        actual = fold_actuals[i]
        point = fold_points[i][: len(actual)]
        q_arr = np.array([q_steps[step_i] for step_i in range(len(actual))])
        coverages.append(calculate_coverage(actual, point - q_arr, point + q_arr))

    return (float(np.mean(coverages)) if coverages else 0.0), optimistic


def run_backtest(
    y: np.ndarray,
    horizon: int = 5,
    seasonal_period: int = 1,
    models: Optional[List[Forecaster]] = None,
    progress_cb: Optional[Callable[[str, float, str], None]] = None,
) -> Tuple[List[ModelBacktestResult], List[FoldBoundary], List[str]]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    warnings = []

    k, boundaries = compute_fold_boundaries(n, horizon, seasonal_period)
    if k < 2:
        warnings.append(f"Only {k} test window(s) available. Treat the ranking as a hint.")

    if models is None:
        models = build_models()

    def report(stage: str, pct: float, message: str) -> None:
        if progress_cb:
            progress_cb(stage, pct, message)

    total_models = max(1, len(models))
    results: List[ModelBacktestResult] = []

    for model_index, forecaster in enumerate(models):
        report(
            forecaster.name,
            model_index / total_models * 100.0,
            f"Testing {forecaster.name}…",
        )
        model_name = forecaster.name
        fold_evals = []
        fold_residuals: Dict[int, List[float]] = {h: [] for h in range(1, horizon + 1)}
        all_test_actuals = []
        all_calibrated_lowers = []
        all_calibrated_uppers = []

        model_failed = False
        fail_reason = None
        # Cached per-fold forecasts and actuals, reused for conformal coverage
        # below so the models are never run twice over the same folds.
        fold_points: List[np.ndarray] = []
        fold_actuals: List[np.ndarray] = []

        # 1. Backtest across folds
        for b in boundaries:
            train_y = y[b.train_start : b.train_end]
            test_y = y[b.test_start : b.test_end]
            act_len = len(test_y)

            try:
                # statsforecast's ARIMA solver emits divide/overflow warnings on
                # its way to a perfectly good fit; they are noise to the user.
                with np.errstate(all="ignore"):
                    fc = forecaster.fit_predict(
                        y=train_y,
                        horizon=horizon,
                        seasonal_period=seasonal_period,
                    )
                fc_point = fc.point[:act_len]
                fc_lower = fc.lower[:act_len]
                fc_upper = fc.upper[:act_len]

                eval_metrics = evaluate_forecast(
                    actual=test_y,
                    forecast_point=fc_point,
                    forecast_lower=fc_lower,
                    forecast_upper=fc_upper,
                    train_y=train_y,
                    seasonal_period=seasonal_period,
                )
                fold_evals.append(eval_metrics)
                fold_points.append(fc_point)
                fold_actuals.append(test_y)

                # Collect step-wise absolute residuals
                abs_err = np.abs(test_y - fc_point)
                for step_i, err_val in enumerate(abs_err):
                    step_h = step_i + 1
                    fold_residuals[step_h].append(float(err_val))

                all_test_actuals.extend(test_y.tolist())

            except Exception as e:
                logger.warning(f"Model {model_name} failed on fold {b.fold}: {e}")
                model_failed = True
                fail_reason = str(e)
                break

        if model_failed or not fold_evals:
            results.append(
                ModelBacktestResult(
                    model_name=model_name,
                    status="Did not run",
                    error_reason=fail_reason or "Unknown error during backtest",
                )
            )
            continue

        # 2. Fit final forecast on full dataset
        try:
            with np.errstate(all="ignore"):
                final_fc = forecaster.fit_predict(
                    y=y,
                    horizon=horizon,
                    seasonal_period=seasonal_period,
                )
        except Exception as e:
            results.append(
                ModelBacktestResult(
                    model_name=model_name,
                    status="Did not run",
                    error_reason=f"Failed final forecast fit: {e}",
                )
            )
            continue

        # 3. Conformal calibration on step-wise residuals
        cal_lower, cal_upper, q_steps, is_approx = calibrate_conformal_intervals(
            fold_residuals=fold_residuals,
            forecast_point=final_fc.point,
            alpha=0.05,
        )

        # 4. Observed coverage — calibrated and measured on DISJOINT folds.
        #
        # Measuring coverage on the same folds used to pick the conformal
        # quantile is circular: the quantile is chosen so those points fall
        # inside, so coverage came out ~100% for every model, every time. The
        # app's headline promise is that "95% means 95% empirically", so the
        # calibration folds and the folds we score on must not overlap.
        obs_coverage, coverage_is_optimistic = _holdout_coverage(
            fold_points=fold_points,
            fold_actuals=fold_actuals,
            horizon=horizon,
            fallback_q_steps=q_steps,
        )
        if coverage_is_optimistic and model_index == 0:
            warnings.append(
                "Too few test windows to check the range on unseen data, so the "
                "reliability percentage is optimistic."
            )

        # Mean metrics across folds
        mase_vals = [e["mase"] for e in fold_evals]
        rmse_vals = [e["rmse"] for e in fold_evals]
        mae_vals = [e["mae"] for e in fold_evals]
        pinball_vals = [e["pinball"] for e in fold_evals]

        results.append(
            ModelBacktestResult(
                model_name=model_name,
                status="ok",
                mean_mase=float(np.mean(mase_vals)),
                mean_rmse=float(np.mean(rmse_vals)),
                mean_mae=float(np.mean(mae_vals)),
                coverage=obs_coverage,
                mean_pinball=float(np.mean(pinball_vals)),
                final_forecast=final_fc,
                calibrated_lower=cal_lower,
                calibrated_upper=cal_upper,
                fold_evaluations=fold_evals,
            )
        )

    # Rank results: status=="ok" sorted by MASE ascending, then pinball ascending
    ok_results = [r for r in results if r.status == "ok"]
    failed_results = [r for r in results if r.status != "ok"]

    ok_results.sort(key=lambda r: (r.mean_mase, r.mean_pinball))
    sorted_results = ok_results + failed_results

    return sorted_results, boundaries, warnings
