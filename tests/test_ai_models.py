"""Tests for the optional foundation models.

These skip entirely on the Standard edition (and on any Python < 3.10, where
the packages cannot be installed at all), so the normal suite is unaffected.

They exist because the TimesFM and Chronos wrappers had never been executed:
the packages were absent, and a blanket `except Exception` hid that fact while
reporting the fallback under a vendor name. Two levels of checking:

  1. API shape — cheap. Do the exact symbols our wrappers call still exist?
     Catches upstream renames without downloading anything.
  2. Real inference — expensive, opt in with FB_AI_WEIGHTS=1. Downloads ~1.3 GB
     and runs an actual forecast. This is the only thing that proves the
     wrappers work end to end.
"""

import os

import numpy as np
import pytest

from core.models import manager

pytestmark = pytest.mark.ai

RUN_WEIGHTS = os.environ.get("FB_AI_WEIGHTS") == "1"


def _series(n=96):
    rng = np.random.default_rng(7)
    t = np.arange(n)
    return 1000 + 9 * t + 160 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 40, n)


# ---------------------------------------------------------------------------
# 1. API shape
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not manager.package_installed("timesfm"), reason="timesfm not installed")
def test_timesfm_api_surface_is_what_we_call():
    import timesfm

    # core/models/timesfm.py instantiates exactly these.
    assert hasattr(timesfm, "TimesFM_2p5_200M_torch"), (
        "timesfm no longer exports TimesFM_2p5_200M_torch — core/models/timesfm.py "
        "needs updating"
    )
    assert hasattr(timesfm, "ForecastConfig")


@pytest.mark.skipif(not manager.package_installed("chronos"), reason="chronos not installed")
def test_chronos_api_surface_is_what_we_call():
    import chronos

    assert hasattr(chronos, "Chronos2Pipeline"), (
        "chronos no longer exports Chronos2Pipeline — this is the 2.2+ API that "
        "core/models/chronos.py depends on; 1.5.x shipped ChronosPipeline instead"
    )
    assert hasattr(chronos.Chronos2Pipeline, "from_pretrained")
    assert hasattr(chronos.Chronos2Pipeline, "predict_quantiles")


@pytest.mark.skipif(not manager.ai_edition(), reason="not an AI build")
def test_capability_report_marks_models_bundled():
    report = manager.capability_report()
    assert report["ai_edition"] is True
    for entry in report["models"]:
        assert entry["bundled"] is True
        assert entry["state"] in ("ready", "needs_download")


# ---------------------------------------------------------------------------
# 2. Real inference
# ---------------------------------------------------------------------------

def _check_forecast(fc, horizon):
    assert len(fc.point) == horizon
    assert len(fc.lower) == horizon == len(fc.upper)
    assert np.all(np.isfinite(fc.point)), "model returned NaN/inf"
    assert np.all(fc.lower <= fc.point + 1e-6)
    assert np.all(fc.point <= fc.upper + 1e-6)


@pytest.mark.skipif(not RUN_WEIGHTS, reason="set FB_AI_WEIGHTS=1 to download weights")
@pytest.mark.skipif(not manager.package_installed("timesfm"), reason="timesfm not installed")
def test_timesfm_really_forecasts():
    from core.models.timesfm import TimesFMForecaster

    manager.configure_model_cache()
    fc = TimesFMForecaster().fit_predict(_series(), horizon=6, seasonal_period=12)
    _check_forecast(fc, 6)


@pytest.mark.skipif(not RUN_WEIGHTS, reason="set FB_AI_WEIGHTS=1 to download weights")
@pytest.mark.skipif(not manager.package_installed("chronos"), reason="chronos not installed")
def test_chronos_really_forecasts():
    from core.models.chronos import ChronosForecaster

    manager.configure_model_cache()
    fc = ChronosForecaster().fit_predict(_series(), horizon=6, seasonal_period=12)
    _check_forecast(fc, 6)


@pytest.mark.skipif(not RUN_WEIGHTS, reason="set FB_AI_WEIGHTS=1 to download weights")
@pytest.mark.skipif(not manager.ai_edition(), reason="not an AI build")
def test_vendor_names_appear_only_after_real_weights_load():
    """The whole point: a vendor name in the report means that vendor ran."""
    from core.backtest import run_backtest

    manager.configure_model_cache()
    manager.download_weights()

    models = manager.build_models()
    names = [m.name for m in models]
    assert "Google TimesFM 2.5" in names
    assert "Amazon Chronos-2" in names

    results, _, _ = run_backtest(_series(), horizon=6, seasonal_period=12, models=models)
    for result in results:
        if result.model_name in ("Google TimesFM 2.5", "Amazon Chronos-2"):
            assert result.status == "ok", (
                f"{result.model_name} was listed but failed: {result.error_reason}"
            )
