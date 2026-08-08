"""Guards against the worst bug this codebase has had.

The app used to report hand-rolled heuristics under the names "TimesFM" and
"Chronos-2" because a blanket `except Exception` swallowed the ImportError for
packages that were never installed. Those names then appeared in the ranking
table, in the plain-English verdict, and in the exported PDF that a user might
show to their bank or their board.

The rule these tests enforce: a vendor's model name may appear in a report only
if that vendor's model actually produced the numbers.
"""

import numpy as np
import pytest

from core.backtest import run_backtest
from core.models import manager
from core.models.heuristics import (
    SeasonalWeightedAverageForecaster,
    SmoothedTrendForecaster,
)

VENDOR_CLAIMS = ["timesfm", "chronos", "google", "amazon"]


def _series(n=72):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    return 1000 + 12 * t + 180 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 60, n)


def test_builtin_heuristics_never_borrow_a_vendor_name():
    for forecaster in (SmoothedTrendForecaster(), SeasonalWeightedAverageForecaster()):
        lowered = forecaster.name.lower()
        assert not any(claim in lowered for claim in VENDOR_CLAIMS), (
            f"{forecaster.name!r} implies a vendor model it is not"
        )
        assert "built-in" in lowered


def test_no_vendor_names_in_report_when_packages_absent(monkeypatch):
    """With the AI packages unavailable, no report row may name a vendor."""
    monkeypatch.setattr(manager, "timesfm_ready", lambda: False)
    monkeypatch.setattr(manager, "chronos_ready", lambda: False)

    models = manager.build_models()
    names = [m.name.lower() for m in models]
    for name in names:
        assert not any(claim in name for claim in VENDOR_CLAIMS), (
            f"{name!r} names a vendor whose package is not installed"
        )

    results, _, _ = run_backtest(_series(), horizon=6, seasonal_period=12, models=models)
    for result in results:
        lowered = result.model_name.lower()
        assert not any(claim in lowered for claim in VENDOR_CLAIMS)


def test_foundation_models_raise_rather_than_downgrade():
    """A missing package must surface as an error, never as a quiet substitute."""
    if manager.package_installed("timesfm"):
        pytest.skip("timesfm is installed in this environment")

    from core.models.timesfm import TimesFMForecaster

    with pytest.raises(ImportError):
        TimesFMForecaster().fit_predict(np.linspace(10, 50, 25), horizon=5, seasonal_period=1)


def test_a_failing_model_is_reported_as_did_not_run():
    """The backtest must record failure, not hide it behind another model."""

    class ExplodingForecaster:
        name = "Deliberately Broken"

        def fit_predict(self, y, horizon, seasonal_period):
            raise RuntimeError("no weights on disk")

    results, _, _ = run_backtest(
        _series(), horizon=6, seasonal_period=12, models=[ExplodingForecaster()]
    )
    assert len(results) == 1
    assert results[0].status == "Did not run"
    assert "no weights on disk" in results[0].error_reason


def test_capability_report_is_honest_about_what_is_installed():
    report = manager.capability_report()
    for entry in report["models"]:
        if not entry["bundled"]:
            assert entry["state"] == "unavailable"
            assert entry["downloaded"] is False
