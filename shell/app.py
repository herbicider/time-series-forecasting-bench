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

# Shown when the window opens but its web view never renders anything.
WEBVIEW2_FAILURE = (
    "The window opened, but the web view inside it could not start, so the "
    "app cannot show anything.\n\n"
    "This means the Microsoft Edge WebView2 Runtime is missing or needs "
    "repairing. It is a free Microsoft component, and installing it fixes "
    "this:\n\n"
    "https://developer.microsoft.com/microsoft-edge/webview2/"
)

# Shown when the bundle is still blocked and we could not unblock it ourselves.
BLOCKED_FILES_FAILURE = (
    "Windows is blocking the files this app needs, because they arrived in a "
    "download.\n\n"
    "To fix it: right-click the ZIP you downloaded, choose Properties, tick "
    "Unblock at the bottom, click OK — then extract it again and start the "
    "app.\n\n"
    "The app tries to clear this itself and could not, which usually means the "
    "folder it is in is read-only. Moving the folder to your Desktop or "
    "Documents also works."
)

_log_path: Optional[Path] = None

# Held open for the process lifetime: faulthandler writes to this descriptor
# from a signal handler, so it must outlive every normal code path.
_crash_file = None

# What configure_logging installed, so a repeat call replaces those handlers
# instead of stacking a second set on top of them.
_installed_handlers: list = []


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

    # A repeat call must replace what the last one installed rather than stack
    # on top of it: two file handlers write every line twice, and on Windows
    # the first one's open file cannot be renamed aside to make room for this
    # run's.
    for installed in _installed_handlers:
        root.removeHandler(installed)
        installed.close()
    _installed_handlers.clear()

    path: Optional[Path] = None
    try:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        _keep_the_previous_log(path)
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        root.addHandler(handler)
        _installed_handlers.append(handler)
    except Exception:  # noqa: BLE001 — an unwritable log directory is survivable
        path = None

    # In a windowed build sys.stderr is None, and StreamHandler(None) would
    # fall back to the real stderr; only add it when there is one.
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        root.addHandler(console)
        _installed_handlers.append(console)

    _log_path = path
    _enable_crash_handler(path)
    return path


def _keep_the_previous_log(path: Path) -> None:
    """Move the last run's log aside before this one truncates it.

    START-HERE tells a user whose app closed on its own to give it another go
    and then send the log. Opening the file in "w" mode means that second
    launch overwrites the report of the first — the failure they are actually
    writing in about. One generation back is kept as <name>.prev.
    """
    try:
        if path.exists():
            path.replace(path.with_suffix(path.suffix + ".prev"))
    except OSError:  # a locked or unwritable old log is not worth failing over
        logger.debug("Could not keep the previous log", exc_info=True)


def _enable_crash_handler(log_path: Optional[Path]) -> None:
    """Catch the failures Python never gets to report.

    A segfault in a native extension — numpy, pyarrow, torch, the WebView2
    bindings — kills the process outright. There is no exception, so no log
    line and no dialog: precisely the "it vanished" symptom, and the one thing
    a pure-Python handler cannot see. faulthandler writes the C-level stack.
    """
    global _crash_file

    if log_path is None:
        return
    try:
        import faulthandler

        crash_path = log_path.with_name("crash.log")
        _keep_the_previous_log(crash_path)
        _crash_file = open(crash_path, "w", encoding="utf-8")
        faulthandler.enable(file=_crash_file, all_threads=True)
    except Exception:  # noqa: BLE001 — diagnostics must never break startup
        logger.debug("Could not enable the crash handler", exc_info=True)


# Only these get executed as code. Data files carry the same mark, but nothing
# loads them as code, and skipping them keeps this to a fraction of a second
# across a bundle of seven thousand files.
EXECUTABLE_SUFFIXES = frozenset({".dll", ".exe", ".pyd"})


def clear_zone_markers(directory: Optional[Path] = None) -> int:
    """Take the "came from the internet" mark off our own files.

    Windows stamps every file extracted from a downloaded ZIP with a
    Zone.Identifier stream, and the .NET Framework host that pywebview's
    WinForms backend loads through refuses to execute an assembly carrying it.
    Python.Runtime.Loader.Initialize then fails to resolve, `import clr` fails,
    and pywebview finds no GUI backend at all: the app starts, the service
    runs, forecasts work — and there is no way to show any of it.

    That is not a defect in this code, and a build that was never downloaded
    cannot reproduce it, which is how CI passed a release that failed on every
    machine that installed it the normal way. Clearing the mark is exactly what
    the Unblock checkbox in a file's Properties dialog does, and these are our
    own files, already on this disk, about to be executed anyway.
    """
    if sys.platform != "win32":
        return 0

    if directory is None:
        bundle = getattr(sys, "_MEIPASS", None)
        if bundle is None:
            return 0  # running from source: nothing here was ever downloaded
        directory = Path(bundle)

    cleared = 0
    started = time.monotonic()
    try:
        for path in directory.rglob("*"):
            if path.suffix.lower() not in EXECUTABLE_SUFFIXES:
                continue
            marker = f"{path}:Zone.Identifier"
            try:
                if os.path.exists(marker):
                    os.remove(marker)
                    cleared += 1
            except OSError:
                # A read-only folder is not fixable from here. Say nothing now;
                # if it actually breaks the GUI, the failure dialog explains it.
                logger.debug("Could not unblock %s", path, exc_info=True)
    except OSError:
        logger.debug("Could not scan the bundle for download markers", exc_info=True)

    if cleared:
        logger.info(
            "Unblocked %d downloaded files in %.1fs", cleared, time.monotonic() - started
        )
    return cleared


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


def _looks_like_a_blocked_bundle(exc: BaseException) -> bool:
    """Is this the CLR refusing to run an assembly marked as downloaded?

    pythonnet reports it as a plain RuntimeError naming the entry point it
    could not resolve, with nothing in it about zones or downloads, so the
    message has to be recognised to be explained.
    """
    text = f"{type(exc).__name__}: {exc}"
    return "Python.Runtime" in text or "Failed to resolve" in text


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

    # log_config=None is not a preference. uvicorn's default config calls
    # logging.config.dictConfig, which calls _clearExistingHandlers() and
    # *closes the stream of every handler already installed* — including the
    # log file this app writes its startup report to. The handler stays
    # attached, so nothing raises: every line logged after the service starts
    # is written to a closed file and silently dropped, because a windowed
    # build has no stderr for logging to report the error on. The log would
    # end mid-startup and look like a crash.
    config = uvicorn.Config(app=fastapi_app, log_level="warning", log_config=None)
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[sock])
    except BaseException:  # noqa: BLE001 — uvicorn calls sys.exit() on a failed start
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

    # `shown` fires for the native form, not for the browser control inside
    # it. When WebView2 fails to initialise, pywebview logs the error and
    # returns, leaving a window that is up but permanently blank — so a
    # separate signal is needed for "the page actually rendered". This one is
    # never cleared; window.load_url() resets pywebview's own loaded event
    # each time it navigates.
    ever_loaded = threading.Event()
    window.events.loaded += lambda *args, **kwargs: ever_loaded.set()

    def watch_for_a_dead_view() -> None:
        if not shown.wait(WINDOW_TIMEOUT):
            return  # a window that never opens is reported by run_app itself
        if ever_loaded.wait(WINDOW_TIMEOUT):
            return
        report_fatal(WEBVIEW2_FAILURE)

    threading.Thread(target=watch_for_a_dead_view, daemon=True, name="viewwatch").start()

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
        # BaseException, not Exception: uvicorn and pywebview both call
        # sys.exit() on a failed start, and a SystemExit that escapes here
        # would end the run with no record of which check raised it.
        try:
            detail = fn()
        except BaseException as exc:  # noqa: BLE001 — every failure is a result
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
        """Open the real window, wait for it to render a page, then close it.

        Waiting on `shown` is not enough, and reported a pass on a runner
        where WebView2 had failed with E_ABORT: the native form appears
        whether or not the browser control inside it ever came up. `loaded`
        is the first signal that something was actually rendered.
        """
        import webview

        window = webview.create_window(
            title=f"{APP_TITLE} (self-check)", html=SPLASH_HTML, width=640, height=480
        )
        shown = threading.Event()
        window.events.shown += lambda *args, **kwargs: shown.set()
        loaded = threading.Event()
        window.events.loaded += lambda *args, **kwargs: loaded.set()

        def close_when_shown() -> None:
            if shown.wait(WINDOW_TIMEOUT):
                loaded.wait(WINDOW_TIMEOUT)
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
        if not loaded.is_set():
            raise RuntimeError(
                "the window opened but never rendered a page — the web view failed "
                "to start (see the WebView2 error above)"
            )
        return "opened a real window and rendered a page in it"

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

    # Before anything imports the GUI backend: `import clr` is what fails when
    # the bundle is still marked as downloaded, and it happens deep inside
    # webview.start().
    clear_zone_markers()

    if self_check:
        try:
            return SelfCheck().run(with_window="--window" in argv)
        except Exception as exc:  # noqa: BLE001
            logger.error("The self-check itself crashed", exc_info=exc)
            return 3

    log_environment()
    try:
        return run_app()
    except SystemExit as exc:
        # SystemExit is not an Exception, so a narrower handler here lets it
        # straight through — and Python's response to an escaping SystemExit is
        # to end the process with no traceback, no log line and no dialog.
        # That is precisely the "it vanished" symptom this module exists to
        # prevent, and a library calling sys.exit() during startup (uvicorn
        # does it on a failed bind) is enough to trigger it. A zero code is
        # somebody quitting deliberately, not a crash.
        code = exc.code
        if code is None or code == 0:
            logger.info("Startup ended before the window opened, with no error")
            return 0
        report_fatal(f"The app stopped while starting up (exit code {code}).", exc)
        return code if isinstance(code, int) else 1
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
        return 130
    except Exception as exc:  # noqa: BLE001 — the last line before a silent exit
        if _looks_like_a_blocked_bundle(exc):
            report_fatal(BLOCKED_FILES_FAILURE, exc)
        else:
            report_fatal("Something went wrong while starting the app.", exc)
        return 1
    finally:
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
