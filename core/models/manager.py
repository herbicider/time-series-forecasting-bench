"""Single source of truth for which forecasters this build can actually run.

Two independent things decide whether a foundation model is usable:

  1. Is the Python package compiled into this build?  A frozen PyInstaller app
     cannot pip-install at runtime, so this is fixed when the app is built.
     The Standard edition ships without torch; the AI edition ships with it.
  2. Are the model weights downloaded?  ~1 GB, fetched once from HuggingFace
     into a per-user directory and reused forever after.

When a foundation model is unavailable we still forecast — the built-in
heuristics in heuristics.py are legitimate methods — but the report labels
them for what they are. A row must never claim to be Google TimesFM or Amazon
Chronos unless that model genuinely produced the numbers.
"""

import importlib.util
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.models.arima import ArimaForecaster
from core.models.baseline import DriftForecaster, SeasonalNaiveForecaster
from core.models.heuristics import (
    SeasonalWeightedAverageForecaster,
    SmoothedTrendForecaster,
)
from core.models.statistical import ETSForecaster, ThetaForecaster

logger = logging.getLogger(__name__)

APP_NAME = "ForecastingBench"

TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"
CHRONOS_REPO = "amazon/chronos-2"

# On-disk size, shown to the user before they commit to a download.
# Measured from the HuggingFace repo manifests, not estimated.
MODEL_DOWNLOAD_MB = {TIMESFM_REPO: 882, CHRONOS_REPO: 456}


# ---------------------------------------------------------------------------
# Where weights live
# ---------------------------------------------------------------------------

def user_data_dir() -> Path:
    """Per-user writable directory that survives moving the portable app."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def model_cache_dir() -> Path:
    return user_data_dir() / "models"


def configure_model_cache() -> Path:
    """Point HuggingFace at our own directory. Must run before torch imports.

    Without this, weights land in ~/.cache/huggingface, which is invisible to
    the user, not cleaned up when they delete the app, and not carried along
    when they move the portable folder to another machine.
    """
    cache = model_cache_dir()
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create model cache at %s: %s", cache, exc)
        return cache
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
    return cache


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def package_installed(module_name: str) -> bool:
    """True if the module is importable without actually importing it.

    find_spec is used rather than a real import because importing torch costs
    several seconds and we call this on every capability check.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def weights_present(repo_id: str) -> bool:
    """True if this repo has at least one materialised snapshot on disk."""
    hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE") or (model_cache_dir() / "hub"))
    snapshots = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(child.is_dir() and any(child.iterdir()) for child in snapshots.iterdir())


def timesfm_ready() -> bool:
    return package_installed("timesfm") and package_installed("torch")


def chronos_ready() -> bool:
    return package_installed("chronos") and package_installed("torch")


def ai_edition() -> bool:
    """True when this build ships the foundation-model packages."""
    return timesfm_ready() or chronos_ready()


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def build_models(use_ai: bool = True) -> List[Any]:
    """The forecasters to backtest, in display order.

    Each foundation model is included only if its package is installed AND its
    weights are already downloaded; otherwise the honestly-named built-in
    heuristic takes its slot so the comparison still has five to seven entries.
    """
    models: List[Any] = [
        SeasonalNaiveForecaster(),
        DriftForecaster(),
        ArimaForecaster(),
        ETSForecaster(),
        ThetaForecaster(),
    ]

    if use_ai and timesfm_ready() and weights_present(TIMESFM_REPO):
        from core.models.timesfm import TimesFMForecaster

        models.append(TimesFMForecaster())
    else:
        models.append(SmoothedTrendForecaster())

    if use_ai and chronos_ready() and weights_present(CHRONOS_REPO):
        from core.models.chronos import ChronosForecaster

        models.append(ChronosForecaster())
    else:
        models.append(SeasonalWeightedAverageForecaster())

    return models


def capability_report() -> Dict[str, Any]:
    """What the UI needs to explain the current state to a non-technical user."""
    entries = []
    for repo, label, installed in (
        (TIMESFM_REPO, "Google TimesFM 2.5", timesfm_ready()),
        (CHRONOS_REPO, "Amazon Chronos-2", chronos_ready()),
    ):
        downloaded = installed and weights_present(repo)
        entries.append(
            {
                "repo": repo,
                "label": label,
                "bundled": installed,
                "downloaded": downloaded,
                "approx_mb": MODEL_DOWNLOAD_MB.get(repo, 0),
                "state": "ready" if downloaded else ("needs_download" if installed else "unavailable"),
            }
        )

    pending_mb = sum(e["approx_mb"] for e in entries if e["state"] == "needs_download")
    return {
        "ai_edition": ai_edition(),
        "models": entries,
        "all_ready": all(e["state"] == "ready" for e in entries) if entries else False,
        "needs_download": any(e["state"] == "needs_download" for e in entries),
        "pending_mb": pending_mb,
        "cache_dir": str(model_cache_dir()),
    }


def download_weights(
    progress_cb: Optional[Callable[[str, float, str], None]] = None,
) -> Dict[str, Any]:
    """Fetch any missing foundation-model weights, reporting real progress.

    progress_cb(stage, pct, message) is called as each repo completes. pct is
    0-100 across the whole job, weighted by each repo's approximate size, so
    the bar advances proportionally rather than jumping.
    """
    def report(stage: str, pct: float, message: str) -> None:
        if progress_cb:
            progress_cb(stage, pct, message)

    targets = [
        (repo, label)
        for repo, label, installed in (
            (TIMESFM_REPO, "Google TimesFM 2.5", timesfm_ready()),
            (CHRONOS_REPO, "Amazon Chronos-2", chronos_ready()),
        )
        if installed and not weights_present(repo)
    ]

    if not targets:
        report("done", 100.0, "All models are already downloaded.")
        return {"downloaded": [], "failed": [], "ok": True}

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        message = f"Model downloading is unavailable in this build ({exc})."
        report("error", 0.0, message)
        return {"downloaded": [], "failed": [{"repo": r, "error": message} for r, _ in targets], "ok": False}

    total_mb = sum(MODEL_DOWNLOAD_MB.get(repo, 100) for repo, _ in targets) or 1
    done_mb = 0.0
    downloaded, failed = [], []

    for repo, label in targets:
        size_mb = MODEL_DOWNLOAD_MB.get(repo, 100)
        report(repo, done_mb / total_mb * 100.0, f"Downloading {label} (about {size_mb} MB)…")
        try:
            snapshot_download(repo_id=repo, cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"))
            downloaded.append(repo)
        except Exception as exc:  # network, disk, auth — all are user-visible
            logger.warning("Weight download failed for %s: %s", repo, exc)
            failed.append({"repo": repo, "label": label, "error": str(exc)})
        done_mb += size_mb
        report(repo, done_mb / total_mb * 100.0, f"Finished {label}.")

    ok = not failed
    report(
        "done" if ok else "error",
        100.0,
        "All models are ready." if ok else "Some models could not be downloaded.",
    )
    return {"downloaded": downloaded, "failed": failed, "ok": ok}
