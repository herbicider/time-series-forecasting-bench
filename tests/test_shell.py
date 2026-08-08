"""Tests for the desktop shell's startup path.

The shell is the one part of the app with no test coverage and the only part
that runs in the packaged build, which is how a release that could not open a
window shipped as working. These tests cover the parts that do not need a
display: the failure reporting that must never be silent, and the headless
self-check that CI runs against the packaged executable.
"""

import logging
import socket
import sys

import pytest

from shell import app as shell_app


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path, monkeypatch):
    """Point logs at a temp directory and undo handler changes afterwards."""
    monkeypatch.setenv("FB_LOG_DIR", str(tmp_path / "logs"))
    root = logging.getLogger()
    original = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in original:
            handler.close()
            root.removeHandler(handler)


def test_log_dir_follows_the_environment(tmp_path, monkeypatch):
    from core.paths import log_dir

    monkeypatch.setenv("FB_LOG_DIR", str(tmp_path / "elsewhere"))
    assert log_dir() == tmp_path / "elsewhere"


def test_configure_logging_writes_a_file(tmp_path):
    path = shell_app.configure_logging("test.log")

    assert path is not None and path.exists()
    logging.getLogger("forecastingbench").error("something broke")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "something broke" in path.read_text(encoding="utf-8")


def test_configure_logging_survives_an_unwritable_directory(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("FB_LOG_DIR", str(blocker / "logs"))

    assert shell_app.configure_logging("test.log") is None


def test_report_fatal_never_raises_without_a_console(monkeypatch, capsys):
    """A windowed build has sys.stderr set to None; reporting must still work."""
    monkeypatch.setattr(sys, "stderr", None)
    shell_app.configure_logging("test.log")

    shell_app.report_fatal("The window never opened.", RuntimeError("no display"))

    assert "The window never opened." in capsys.readouterr().out


def test_report_fatal_names_the_log_file(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stderr", None)
    path = shell_app.configure_logging("test.log")

    shell_app.report_fatal("Boom.", RuntimeError("why"))

    out = capsys.readouterr().out
    assert "RuntimeError: why" in out
    assert str(path) in out
    assert "why" in path.read_text(encoding="utf-8")


def test_bind_free_port_holds_the_socket():
    sock, port = shell_app.bind_free_port()
    try:
        assert 1024 < port < 65536
        assert sock.getsockname()[1] == port

        taken = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OSError):
            taken.bind(("127.0.0.1", port))
        taken.close()
    finally:
        sock.close()


def test_open_url_only_follows_web_links(monkeypatch):
    opened = []
    monkeypatch.setattr(shell_app.webbrowser, "open", lambda url, new=0: opened.append(url))
    api = shell_app.Api()

    assert api.open_url("https://example.com") is True
    assert api.open_url("file:///etc/passwd") is False
    assert api.open_url("javascript:alert(1)") is False
    assert api.open_url(None) is False
    assert opened == ["https://example.com"]


def test_sample_series_is_long_enough_to_forecast():
    rows = shell_app._sample_series().splitlines()

    assert rows[0] == "date,value"
    assert len(rows) - 1 >= 20  # ingest rejects anything shorter
    assert len(set(rows[1:])) == len(rows) - 1


def test_self_check_passes_its_headless_checks():
    """The checks CI relies on, minus the ones that need a display."""
    check = shell_app.SelfCheck()

    check.check("bundled files", check._bundled_files)
    check.check("capabilities", check._capabilities)
    check.check("service", check._service)

    assert check.failures == 0


def test_self_check_counts_failures_instead_of_raising():
    check = shell_app.SelfCheck()

    def explode():
        raise RuntimeError("nope")

    check.check("deliberate failure", explode)

    assert check.failures == 1
