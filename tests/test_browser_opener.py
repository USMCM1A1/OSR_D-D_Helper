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


def test_uses_subprocess_when_pywebview_available(monkeypatch):
    calls = []
    def fake_popen(cmd, **kw):
        calls.append((cmd, kw))
        class _P: pass
        return _P()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    opener = _FakeOpener(pywebview_ok=True)
    opener.open("http://127.0.0.1:8765/player", label="Player View",
                width=1280, height=720)
    assert len(calls) == 1
    cmd, kw = calls[0]
    # Verify the command line points at the per-window script with
    # the right URL/title/dimensions.
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("osr_webview_window.py")
    assert "--url" in cmd and "http://127.0.0.1:8765/player" in cmd
    assert "--title" in cmd
    assert "OSR Dungeon — Player View" in cmd
    assert "--width" in cmd and "1280" in cmd
    assert "--height" in cmd and "720" in cmd
    # Detached so closing the window or pygame doesn't cascade.
    assert kw.get("start_new_session") is True


def test_cooldown_suppresses_rapid_reopen(monkeypatch, capsys):
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
