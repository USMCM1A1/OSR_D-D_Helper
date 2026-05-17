"""Tests for main.BrowserOpener.

The opener prefers pywebview (each window in its own subprocess to
side-step the macOS main-thread conflict with pygame) and falls back
to webbrowser.open when pywebview isn't available. Tests mock both
backends so they run offline and don't pop real windows.
"""

from __future__ import annotations

import subprocess
import sys

import main as main_mod


class _FakeOpener(main_mod.BrowserOpener):
    """Pin _pywebview_ok explicitly so tests cover both branches
    regardless of whether the test env actually has pywebview."""
    def __init__(self, pywebview_ok: bool):
        super().__init__(min_interval_seconds=60.0)
        self._pywebview_ok = pywebview_ok


def test_uses_webbrowser_when_pywebview_unavailable(monkeypatch):
    opens = []
    monkeypatch.setattr(main_mod.webbrowser, "open",
                        lambda url, **kw: opens.append(url))
    opener = _FakeOpener(pywebview_ok=False)
    opener.open("http://127.0.0.1:8765/", label="Editor")
    assert opens == ["http://127.0.0.1:8765/"]


class _FakePopen:
    """Stand-in for subprocess.Popen with controllable poll() result.
    Returns None from poll() while 'alive', a status code when 'dead'."""
    _next_pid = 9000

    def __init__(self):
        _FakePopen._next_pid += 1
        self.pid = _FakePopen._next_pid
        self._returncode = None

    def poll(self):
        return self._returncode

    def exit(self, code: int = 0) -> None:
        self._returncode = code


def test_uses_subprocess_when_pywebview_available(monkeypatch):
    calls = []
    procs = []
    def fake_popen(cmd, **kw):
        p = _FakePopen()
        calls.append((cmd, kw))
        procs.append(p)
        return p
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    opener = _FakeOpener(pywebview_ok=True)
    opener.open("http://127.0.0.1:8765/player",
                label="OSR Dungeon — Player View",
                width=1280, height=720)
    assert len(calls) == 1
    cmd, kw = calls[0]
    # Verify the command line points at the per-window script with
    # the right URL/title/dimensions. The label is passed straight
    # through as --title — the caller owns the full string.
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("osr_webview_window.py")
    assert "--url" in cmd and "http://127.0.0.1:8765/player" in cmd
    assert "--title" in cmd
    assert "OSR Dungeon — Player View" in cmd
    assert "--width" in cmd and "1280" in cmd
    assert "--height" in cmd and "720" in cmd
    # Detached so closing the window or pygame doesn't cascade.
    assert kw.get("start_new_session") is True


def test_pywebview_reuse_focuses_existing_window(monkeypatch):
    # Same URL twice → only one Popen, plus an osascript focus call.
    popen_calls = []
    focus_calls = []
    def fake_popen(cmd, **kw):
        popen_calls.append(cmd)
        return _FakePopen()
    def fake_run(cmd, **kw):
        focus_calls.append(cmd)
        class _R: returncode = 0
        return _R()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "platform", "darwin")
    opener = _FakeOpener(pywebview_ok=True)
    opener.open("http://x/", label="Editor")
    opener.open("http://x/", label="Editor")  # alive → focus, no new Popen
    assert len(popen_calls) == 1
    assert len(focus_calls) == 1
    assert focus_calls[0][0] == "osascript"


def test_pywebview_respawn_after_window_closed(monkeypatch):
    # When the user closes the window, the subprocess exits. The next
    # press of the hotkey must spawn a fresh window, not just try to
    # focus a dead pid.
    procs = []
    focus_calls = []
    def fake_popen(cmd, **kw):
        p = _FakePopen()
        procs.append(p)
        return p
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: focus_calls.append(a))
    opener = _FakeOpener(pywebview_ok=True)
    opener.open("http://x/", label="Editor")
    procs[0].exit(0)  # user closed the window
    opener.open("http://x/", label="Editor")
    assert len(procs) == 2  # second Popen because first was dead
    assert focus_calls == []  # no focus call needed for a fresh window


def test_cooldown_suppresses_rapid_reopen_in_webbrowser_fallback(monkeypatch, capsys):
    # The cooldown only applies to the webbrowser fallback — the
    # pywebview path uses the live-subprocess check instead.
    opens = []
    monkeypatch.setattr(main_mod.webbrowser, "open",
                        lambda url, **kw: opens.append(url))
    opener = _FakeOpener(pywebview_ok=False)
    opener.open("http://x/", label="Editor")
    opener.open("http://x/", label="Editor")
    assert opens == ["http://x/"]  # second call suppressed
    err = capsys.readouterr().err
    assert "suppressed re-open" in err


def test_subprocess_oserror_falls_back_to_webbrowser(monkeypatch):
    def fail_popen(*args, **kwargs):
        raise OSError("no such file")
    opens = []
    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    monkeypatch.setattr(main_mod.webbrowser, "open",
                        lambda url, **kw: opens.append(url))
    opener = _FakeOpener(pywebview_ok=True)
    opener.open("http://127.0.0.1:8765/", label="Editor")
    assert opens == ["http://127.0.0.1:8765/"]
