"""Per-user file locations, with no heavy imports.

This lives apart from core.models.manager on purpose. The shell needs the log
directory *before* it imports anything that could fail — manager pulls in
statsforecast, and a crash-logger that depends on the thing most likely to
crash is not a crash-logger.
"""

import os
import sys
from pathlib import Path

APP_NAME = "ForecastingBench"


def user_data_dir() -> Path:
    """Per-user writable directory that survives moving the portable app."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def log_dir() -> Path:
    """Where startup logs go.

    FB_LOG_DIR overrides it, which is how CI collects the log after running the
    packaged executable.
    """
    override = os.environ.get("FB_LOG_DIR")
    if override:
        return Path(override)
    return user_data_dir() / "logs"
