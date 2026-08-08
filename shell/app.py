"""Desktop shell: starts the local service, then opens a native window on it.

The previous version slept 0.5 s and hoped, had no error handling, and left
external links to navigate the app window itself. All three produced the same
user experience: a blank or stuck window with nothing to act on.
"""

import logging
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Redirect the HuggingFace cache before anything can import torch.
from core.models import manager  # noqa: E402

manager.configure_model_cache()

import uvicorn  # noqa: E402
import webview  # noqa: E402

from service.main import app as fastapi_app  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

APP_TITLE = "Time Series Forecasting Bench"
STARTUP_TIMEOUT = 30.0


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


def run_server(sock: socket.socket) -> None:
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
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    return False


def show_startup_failure() -> None:
    message = (
        f"{APP_TITLE} could not start its local engine.\n\n"
        "This is usually caused by security software blocking local connections. "
        "Try restarting the app, or allow it through your firewall."
    )
    try:
        webview.create_window(APP_TITLE, html=_error_page(message), width=620, height=380)
        webview.start()
    except Exception:  # noqa: BLE001
        print(message, file=sys.stderr)


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


def main() -> None:
    host = "127.0.0.1"
    try:
        sock, port = bind_free_port(host)
    except OSError as exc:
        logger.error("Could not reserve a local port: %s", exc)
        show_startup_failure()
        return

    base_url = f"http://{host}:{port}"
    threading.Thread(target=run_server, args=(sock,), daemon=True).start()

    # A splash while the server boots, so the window is never blank.
    window = webview.create_window(
        title=APP_TITLE,
        html=SPLASH_HTML,
        width=1180,
        height=820,
        min_size=(900, 640),
        js_api=Api(),
    )

    def on_ready() -> None:
        if wait_for_service(base_url):
            window.load_url(base_url)
        else:
            window.load_html(
                _error_page(
                    "The local engine did not respond in time.\n\n"
                    "Security software blocking local connections is the usual cause."
                )
            )

    try:
        webview.start(on_ready, private_mode=False)
    except Exception:  # noqa: BLE001
        logger.exception("The application window could not be opened")
        show_startup_failure()


if __name__ == "__main__":
    main()
