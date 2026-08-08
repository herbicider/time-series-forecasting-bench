"""Desktop shell: starts the local service, then opens a native window on it.

Everything in here is arranged around one rule: **the app must never vanish
without saying why.** A packaged Windows build has no console, so `print` goes
nowhere and `sys.stderr` may be None. The previous version wrapped its startup
in two broad `except Exception` blocks whose only output was a print to that
missing stream, so any failure — a GUI backend that would not load, a port it
could not bind — looked identical to the user: the process appears in Task
Manager for a few seconds and disappears.

So every failure path now does two things instead:

  1. writes a full traceback to a log file under the per-user data directory
     (core.paths.log_dir), and
  2. shows a native message box naming the problem and the log file.

`--self-check` runs the same startup sequence headlessly and reports on it,
which is what CI runs against the packaged executable so a build that cannot
start can never be published again.
"""

import logging
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Deliberately the only import before logging is configured: core.paths pulls
# in nothing but the standard library, so it cannot be the thing that fails.
from core.paths import log_dir  # noqa: E402

logger = logging.getLogger("forecastingbench")

APP_TITLE = "Time Series Forecasting Bench"
STARTUP_TIMEOUT = 30.0

# How long the window may take to appear before we call it a failed start.
WINDOW_TIMEOUT = 60.0

_log_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Logging and failure reporting
# ---------------------------------------------------------------------------

def configure_logging(filename: str = "startup.log") -> Optional[Path]:
    """Send INFO and above to a log file, and to the console when there is one.

    Returns the log path, or None if no writable location could be found —
    logging must never be the reason the app fails to start.
    """
    global _log_path

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    path: Optional[Path] = None
    try:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        root.addHandler(handler)
    except Exception:  # noqa: BLE001 — an unwritable log directory is survivable
        path = None

    # In a windowed build sys.stderr is None, and StreamHandler(None) would
    # fall back to the real stderr; only add it when there is one.
    if sys.stderr is not None:
        root.addHandler(logging.StreamHandler(sys.stderr))

    _log_path = path
    return path


def dialogs_enabled() -> bool:
    """False when nobody is there to dismiss a modal dialog.

    MessageBoxW blocks until it is clicked. On a CI runner, or under pytest,
    that is not an error report — it is a hang that lasts until the job times
    out. The self-check and the test suite both turn dialogs off.
    """
    return os.environ.get("FB_NO_DIALOG") != "1"


def _message_box(title: str, text: str) -> None:
    """Show a native error dialog, falling back to whatever stream exists."""
    if sys.platform == "win32" and dialogs_enabled():
        try:
            import ctypes

            MB_OK, MB_ICONERROR, MB_SETFOREGROUND = 0x0, 0x10, 0x10000
            ctypes.windll.user32.MessageBoxW(
                None, text, title, MB_OK | MB_ICONERROR | MB_SETFOREGROUND
            )
            return
        except Exception:  # noqa: BLE001
            logger.exception("Could not show the error dialog")

    stream = sys.stderr or sys.stdout
    if stream is not None:
        print(f"{title}\n\n{text}", file=stream)


def report_fatal(summary: str, exc: Optional[BaseException] = None) -> None:
    """Log a startup failure in full, then tell the user in plain language."""
    if exc is not None:
        logger.error("%s", summary, exc_info=exc)
    else:
        logger.error("%s", summary)

    detail = f"{type(exc).__name__}: {exc}" if exc is not None else ""
    where = f"\n\nThe full technical details were written to:\n{_log_path}" if _log_path else ""

    _message_box(
        f"{APP_TITLE} could not start",
        f"{summary}\n\n{detail}"
        "\n\nThis is usually caused by security software blocking the app, or by "
        "running it from inside the ZIP file instead of extracting it first."
        f"{where}",
    )


def install_exception_hooks() -> None:
    """Make sure nothing dies quietly, on any thread."""

    def handle(exc_type, exc, tb) -> None:
        logger.error(
            "Unhandled exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb))
        )

    sys.excepthook = handle

    def handle_thread(args) -> None:
        handle(args.exc_type, args.exc_value, args.exc_traceback)

    threading.excepthook = handle_thread


def log_environment() -> None:
    logger.info("%s starting at %s", APP_TITLE, datetime.now(timezone.utc).isoformat())
    logger.info("python=%s platform=%s", sys.version.split()[0], sys.platform)
    logger.info("frozen=%s executable=%s", getattr(sys, "frozen", False), sys.executable)
    logger.info("bundle=%s cwd=%s", getattr(sys, "_MEIPASS", "(not frozen)"), os.getcwd())


# ---------------------------------------------------------------------------
# Service plumbing
# ---------------------------------------------------------------------------

class Api:
    """Bridge exposed to the page as window.pywebview.api."""

    def open_url(self, url: str) -> bool:
        """Open a link in the user's real browser.

        Without this, clicking the LinkedIn link navigates the app window to
        linkedin.com with no back button — which strands the user inside a
        desktop app and breaks the one conversion this tool exists for.
        """
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return False
        try:
            webbrowser.open(url, new=2)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open %s: %s", url, exc)
            return False


def bind_free_port(host: str = "127.0.0.1") -> "tuple[socket.socket, int]":
    """Reserve a port and hold the socket, so nothing can take it in between.

    Returning only the number left a window where another process could claim
    it before uvicorn bound.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    return sock, sock.getsockname()[1]


def run_server(sock: socket.socket, fastapi_app) -> None:
    import uvicorn

    config = uvicorn.Config(app=fastapi_app, log_level="warning")
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[sock])
    except Exception:  # noqa: BLE001
        logger.exception("The local service stopped unexpectedly")


def wait_for_service(url: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Poll the health endpoint that already exists instead of guessing."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    last_error: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            time.sleep(0.1)
    logger.error("The service did not answer within %.0fs (last error: %s)", timeout, last_error)
    return False


def start_service() -> str:
    """Start the local API on a free port and return its base URL.

    The service module is imported here, on the calling thread, so an import
    error is raised where it can be reported rather than disappearing into a
    worker thread and surfacing 30 seconds later as an unexplained timeout.
    """
    from service.main import app as fastapi_app

    host = "127.0.0.1"
    sock, port = bind_free_port(host)
    threading.Thread(
        target=run_server, args=(sock, fastapi_app), daemon=True, name="service"
    ).start()
    base_url = f"http://{host}:{port}"
    logger.info("Local service starting on %s", base_url)
    return base_url


def _error_page(message: str) -> str:
    return f"""
    <html><body style="font-family:-apple-system,Segoe UI,sans-serif;
                       background:#F4F5F3;color:#151C24;padding:44px;line-height:1.6">
      <h2 style="margin:0 0 12px">Couldn't start</h2>
      <p style="color:#59636E;white-space:pre-wrap">{message}</p>
    </body></html>
    """


SPLASH_HTML = """
<html><body style="margin:0;height:100vh;display:flex;align-items:center;
                   justify-content:center;background:#F4F5F3;
                   font-family:-apple-system,Segoe UI,sans-serif">
  <div style="text-align:center">
    <div style="font-family:Georgia,serif;font-size:26px;color:#151C24">
      Time Series Forecasting Bench</div>
    <div style="margin-top:8px;color:#59636E;font-size:14px">Starting up…</div>
    <div style="margin:22px auto 0;width:180px;height:4px;background:#E2E6E9;
                border-radius:999px;overflow:hidden">
      <div style="width:40%;height:100%;background:#7C2D3A;border-radius:999px;
                  animation:slide 1.1s ease-in-out infinite"></div>
    </div>
  </div>
  <style>@keyframes slide{0%{margin-left:-40%}100%{margin-left:100%}}</style>
</body></html>
"""


# ---------------------------------------------------------------------------
# Normal startup
# ---------------------------------------------------------------------------

def run_app() -> int:
    from core.models import manager

    # Weights must be redirected before anything can import torch.
    manager.configure_model_cache()

    import webview

    base_url = start_service()

    # A splash while the server boots, so the window is never blank.
    window = webview.create_window(
        title=APP_TITLE,
        html=SPLASH_HTML,
        width=1180,
        height=820,
        min_size=(900, 640),
        js_api=Api(),
    )

    shown = threading.Event()
    window.events.shown += lambda *args, **kwargs: shown.set()

    def on_ready() -> None:
        try:
            if wait_for_service(base_url):
                logger.info("Service is up; loading the app")
                window.load_url(base_url)
            else:
                window.load_html(
                    _error_page(
                        "The local engine did not respond in time.\n\n"
                        "Security software blocking local connections is the usual cause."
                    )
                )
        except Exception:  # noqa: BLE001 — a dead callback thread must still be visible
            logger.exception("Failed to hand the window over to the local service")

    logger.info("Opening the application window")
    webview.start(on_ready, private_mode=False)

    # pywebview also *returns* from start() — without raising — when a window
    # fails to initialise. Left unchecked that is exactly the silent exit this
    # module exists to prevent.
    if not shown.is_set():
        report_fatal("The application window never opened.")
        return 1

    logger.info("Window closed; shutting down")
    return 0


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _sample_series(months: int = 36) -> str:
    """Three years of monthly numbers with a trend and a seasonal wobble."""
    rows = ["date,value"]
    for i in range(months):
        year, month = 2021 + i // 12, i % 12 + 1
        rows.append(f"{year}-{month:02d}-01,{1000 + i * 18 + (i % 12) * 45}")
    return "\n".join(rows)


class SelfCheck:
    """Runs the startup path headlessly and records what worked.

    Everything the packaged app does before a user sees a window is exercised
    here — the frozen imports, the bundled UI and sample data, the service, a
    real forecast, and loading the native GUI backend — so a build that cannot
    start fails CI instead of a download.
    """

    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, fn) -> None:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 — every failure is a result, not a crash
            self.failures += 1
            logger.error("FAIL  %s", name, exc_info=exc)
        else:
            logger.info("ok    %s%s", name, f" — {detail}" if detail else "")

    # -- individual checks --------------------------------------------------

    def _bundled_files(self) -> str:
        from service.main import ROOT_DIR as SERVICE_ROOT, SAMPLES_DIR, UI_DIR

        required = [
            UI_DIR / "index.html",
            UI_DIR / "app.js",
            UI_DIR / "style.css",
            UI_DIR / "vendor" / "echarts.min.js",
            SAMPLES_DIR / "monthly_revenue.csv",
        ]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing bundled files: {', '.join(missing)}")
        return f"resource root {SERVICE_ROOT}"

    def _capabilities(self) -> str:
        from core.models import manager

        manager.configure_model_cache()
        report = manager.capability_report()
        return (
            f"{'AI' if report['ai_edition'] else 'Standard'} edition, "
            f"cache {report['cache_dir']}"
        )

    def _service(self) -> str:
        import json
        import urllib.request

        base_url = start_service()
        if not wait_for_service(base_url):
            raise TimeoutError(f"{base_url}/api/health never answered")

        request = urllib.request.Request(
            f"{base_url}/api/forecast/sync",
            data=json.dumps({"data": _sample_series(), "horizon": 3}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as resp:
            report = json.loads(resp.read())

        ranking = report.get("ranking") or []
        succeeded = [r for r in ranking if r.get("status") == "ok"]
        if not succeeded:
            raise ValueError(f"no model produced a forecast (ranking: {ranking})")
        return f"{base_url}, {len(succeeded)}/{len(ranking)} models ran, winner {report['winner']}"

    def _gui_backend(self) -> str:
        import webview
        from webview.guilib import initialize

        guilib = initialize()
        version = getattr(webview, "__version__", None) or "unknown"
        return f"pywebview {version}, renderer {getattr(guilib, 'renderer', '?')}"

    def _window(self) -> str:
        """Open the real window, wait for it to show, then close it."""
        import webview

        window = webview.create_window(
            title=f"{APP_TITLE} (self-check)", html=SPLASH_HTML, width=640, height=480
        )
        shown = threading.Event()
        window.events.shown += lambda *args, **kwargs: shown.set()

        def close_when_shown() -> None:
            if shown.wait(WINDOW_TIMEOUT):
                time.sleep(1.0)  # let the renderer paint at least one frame
            try:
                window.destroy()
            except Exception:  # noqa: BLE001
                logger.exception("Could not close the self-check window")

        threading.Thread(target=close_when_shown, daemon=True, name="closer").start()

        # A window that never opens would otherwise block CI until the job
        # timeout, with no log to show for it.
        watchdog = threading.Timer(
            WINDOW_TIMEOUT + 30.0, lambda: _abort("the window never opened")
        )
        watchdog.daemon = True
        watchdog.start()
        try:
            webview.start(private_mode=False)
        finally:
            watchdog.cancel()

        if not shown.is_set():
            raise RuntimeError("webview.start() returned but the window never appeared")
        return "opened and closed a real window"

    # -- driver -------------------------------------------------------------

    def run(self, with_window: bool) -> int:
        log_environment()
        self.check("bundled UI and sample files", self._bundled_files)
        self.check("model capability report", self._capabilities)
        self.check("local service and a real forecast", self._service)
        self.check("native GUI backend", self._gui_backend)
        if with_window:
            self.check("application window", self._window)

        if self.failures:
            logger.error("SELF-CHECK FAILED (%d failing checks)", self.failures)
            return 1
        logger.info("SELF-CHECK PASSED")
        return 0


def _abort(reason: str) -> None:
    logger.error("Self-check watchdog fired: %s", reason)
    logging.shutdown()
    os._exit(2)


# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    self_check = "--self-check" in argv or os.environ.get("FB_SELF_CHECK") == "1"
    if self_check:
        # Nothing is watching a self-check, so a modal dialog would hang it.
        os.environ["FB_NO_DIALOG"] = "1"

    configure_logging("selfcheck.log" if self_check else "startup.log")
    install_exception_hooks()

    if self_check:
        try:
            return SelfCheck().run(with_window="--window" in argv)
        except Exception as exc:  # noqa: BLE001
            logger.error("The self-check itself crashed", exc_info=exc)
            return 3

    log_environment()
    try:
        return run_app()
    except Exception as exc:  # noqa: BLE001 — the last line before a silent exit
        report_fatal("Something went wrong while starting the app.", exc)
        return 1
    finally:
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
