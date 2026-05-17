"""Tests for main._ensure_editor_port_free.

Covers the four real cases:
  - port already free → True without touching the system
  - port held by a previous instance of this same app → kill + free
  - port held by something else → False with a helpful stderr message
  - lsof unavailable → falls through to the generic error path
"""

from __future__ import annotations

import errno
import os
import socket
import subprocess
import time

import pytest

import main as main_mod


def _free_port() -> int:
    """Bind a probe socket to port 0 and return the kernel-assigned
    port number. The socket is closed immediately, so the port is
    momentarily 'free' (within the test's timing window)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------


def test_free_port_returns_true_without_subprocess(monkeypatch):
    """Common case: port is bindable, no lsof/ps calls happen."""
    calls = []

    def boom(*a, **kw):
        calls.append(a)
        raise AssertionError("subprocess should not run when port is free")

    monkeypatch.setattr(subprocess, "run", boom)
    port = _free_port()
    assert main_mod._ensure_editor_port_free(port) is True
    assert calls == []


def test_port_held_by_our_own_instance_is_killed(monkeypatch):
    """When the bound process is a Python running main.py from this
    project, the helper SIGTERMs it and reports True after the port
    frees up."""
    fake_pid = 99999
    project_root = str(main_mod.PROJECT_ROOT.resolve())

    # First _try_bind call returns False (busy); after our fake "kill"
    # subsequent calls return True.
    bind_attempts = {"n": 0}
    real_socket = socket.socket

    class _FakeSocket:
        def __init__(self, *a, **kw):
            self._real = real_socket(*a, **kw)

        def setsockopt(self, *a, **kw):
            self._real.setsockopt(*a, **kw)

        def bind(self, addr):
            bind_attempts["n"] += 1
            if bind_attempts["n"] == 1:
                err = OSError("Address already in use")
                err.errno = errno.EADDRINUSE
                raise err
            # subsequent attempts succeed
            return self._real.bind(("127.0.0.1", 0))

        def close(self):
            self._real.close()

    monkeypatch.setattr(socket, "socket", _FakeSocket)

    def fake_run(args, **kw):
        if args[0] == "lsof":
            return subprocess.CompletedProcess(args, 0, stdout=f"{fake_pid}\n", stderr="")
        if args[:2] == ["ps", "-p"]:
            # Cmdline that looks like our own app instance.
            cmd = f"/opt/anaconda3/bin/python {project_root}/main.py dungeons/x"
            return subprocess.CompletedProcess(args, 0, stdout=cmd + "\n", stderr="")
        raise AssertionError(f"unexpected subprocess call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    killed = []
    monkeypatch.setattr(os, "kill",
                        lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    assert main_mod._ensure_editor_port_free(8765) is True
    assert killed and killed[0][0] == fake_pid


def test_port_held_by_foreign_process_returns_false(monkeypatch, capsys):
    """When the port is held by something other than this app (e.g.
    a different web server), we don't touch it and report a clear error."""
    err = OSError("Address already in use")
    err.errno = errno.EADDRINUSE
    real_socket = socket.socket

    class _AlwaysBusy:
        def __init__(self, *a, **kw):
            self._real = real_socket(*a, **kw)

        def setsockopt(self, *a, **kw):
            self._real.setsockopt(*a, **kw)

        def bind(self, _addr):
            raise err

        def close(self):
            self._real.close()

    monkeypatch.setattr(socket, "socket", _AlwaysBusy)

    def fake_run(args, **kw):
        if args[0] == "lsof":
            return subprocess.CompletedProcess(args, 0, stdout="42\n", stderr="")
        if args[:2] == ["ps", "-p"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="/usr/sbin/nginx -g daemon off;\n", stderr=""
            )
        raise AssertionError(f"unexpected: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    killed = []
    monkeypatch.setattr(os, "kill", lambda *a: killed.append(a))

    assert main_mod._ensure_editor_port_free(8765) is False
    assert killed == []
    captured = capsys.readouterr()
    assert "in use by something other than this app" in captured.err
    assert "lsof -ti :8765 | xargs kill" in captured.err


def test_lsof_unavailable_falls_through_to_generic_error(monkeypatch, capsys):
    """If lsof isn't installed, we can't identify the holder — print
    the manual fix and return False."""
    err = OSError("Address already in use")
    err.errno = errno.EADDRINUSE
    real_socket = socket.socket

    class _AlwaysBusy:
        def __init__(self, *a, **kw):
            self._real = real_socket(*a, **kw)

        def setsockopt(self, *a, **kw):
            self._real.setsockopt(*a, **kw)

        def bind(self, _addr):
            raise err

        def close(self):
            self._real.close()

    monkeypatch.setattr(socket, "socket", _AlwaysBusy)

    def fake_run(args, **kw):
        raise FileNotFoundError("lsof not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert main_mod._ensure_editor_port_free(8765) is False
    captured = capsys.readouterr()
    assert "Port 8765 is in use" in captured.err
    assert "lsof -ti :8765 | xargs kill" in captured.err
