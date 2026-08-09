"""Tests for the desktop shell's startup path.

The shell is the one part of the app with no test coverage and the only part
that runs in the packaged build, which is how a release that could not open a
window shipped as working. These tests cover the parts that do not need a
display: the failure reporting that must never be silent, and the headless
self-check that CI runs against the packaged executable.
"""

import logging
import sys

import pytest

from shell import app as shell_app


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path, monkeypatch):
    """Point logs at a temp directory and undo handler changes afterwards.

    FB_NO_DIALOG matters as much as the log path: on Windows a failure report
    would otherwise open a modal message box that no test run can dismiss.
    """
    monkeypatch.setenv("FB_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("FB_NO_DIALOG", "1")
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


def test_dialogs_are_suppressed_for_unattended_runs(monkeypatch, capsys):
    """A modal dialog with nobody to click it is a hang, not a report."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stderr", None)
    shell_app.configure_logging("test.log")

    assert shell_app.dialogs_enabled() is False  # set by the autouse fixture
    shell_app.report_fatal("No window.", RuntimeError("headless"))

    assert "No window." in capsys.readouterr().out


def test_self_check_mode_disables_dialogs(monkeypatch):
    monkeypatch.delenv("FB_NO_DIALOG", raising=False)
    monkeypatch.setattr(shell_app.SelfCheck, "run", lambda self, with_window: 0)

    assert shell_app.main(["--self-check"]) == 0
    assert shell_app.dialogs_enabled() is False


def test_report_fatal_names_the_log_file(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stderr", None)
    path = shell_app.configure_logging("test.log")

    shell_app.report_fatal("Boom.", RuntimeError("why"))

    out = capsys.readouterr().out
    assert "RuntimeError: why" in out
    assert str(path) in out
    assert "why" in path.read_text(encoding="utf-8")


def test_a_library_calling_sys_exit_is_still_reported(monkeypatch, capsys):
    """SystemExit is not an Exception, and Python exits on it in silence.

    uvicorn calls sys.exit() on a failed start. Caught only as `Exception`, it
    would sail past the handler and end the process with nothing written down
    anywhere — the original bug, wearing a different exception type.
    """
    from core.paths import log_dir

    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(shell_app, "run_app", lambda: sys.exit(4))

    assert shell_app.main([]) == 4
    assert "The app stopped while starting up" in capsys.readouterr().out
    # main() opens its own startup.log, and closes it on the way out.
    assert "exit code 4" in (log_dir() / "startup.log").read_text(encoding="utf-8")


def test_a_deliberate_clean_exit_is_not_called_a_crash(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(shell_app, "run_app", lambda: sys.exit(0))

    assert shell_app.main([]) == 0
    assert "could not start" not in capsys.readouterr().out


def test_the_previous_log_survives_the_next_launch():
    """A user who closes and reopens the app must not erase the evidence."""
    shell_app.configure_logging("test.log")
    logging.getLogger("forecastingbench").error("the failure being reported")
    for handler in logging.getLogger().handlers:
        handler.flush()

    path = shell_app.configure_logging("test.log")

    assert path is not None
    kept = path.with_suffix(path.suffix + ".prev")
    assert "the failure being reported" in kept.read_text(encoding="utf-8")
    assert "the failure being reported" not in path.read_text(encoding="utf-8")


def test_configure_logging_does_not_stack_handlers():
    """Twice through must not mean every line written twice."""
    root = logging.getLogger()
    before = len(root.handlers)

    shell_app.configure_logging("test.log")
    after_one = len(root.handlers)
    shell_app.configure_logging("test.log")

    assert after_one > before
    assert len(root.handlers) == after_one


def test_bind_free_port_holds_the_socket():
    """The point of returning the socket is that the port stays reserved."""
    sock, port = shell_app.bind_free_port()
    try:
        assert 1024 < port < 65536
        assert sock.getsockname() == ("127.0.0.1", port)
        assert sock.fileno() != -1  # still open, so nothing else can take it

        second_sock, second_port = shell_app.bind_free_port()
        second_sock.close()
        assert second_port != port
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


def test_starting_the_service_keeps_our_log_file_open():
    """uvicorn's default log config closes every handler it finds.

    logging.config.dictConfig calls _clearExistingHandlers(), which closes the
    stream of the file handler this app reports startup failures through. The
    handler stays attached, so nothing raises — the log simply stops, mid
    startup, exactly where the reader most needs it.
    """
    path = shell_app.configure_logging("test.log")
    log = logging.getLogger("forecastingbench")

    base_url = shell_app.start_service()
    assert shell_app.wait_for_service(base_url, timeout=30)

    log.error("logged after the service came up")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "logged after the service came up" in path.read_text(encoding="utf-8")


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
