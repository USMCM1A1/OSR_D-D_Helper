"""Pygame entry point — DM-side dungeon tracker.

A *dungeon* is a self-contained folder under `dungeons/`:

    dungeons/<name>/
      ├── dungeon.json      # the level + room data, edited via the browser tab
      ├── level1.png        # one PNG per level
      ├── level2.png
      └── session.db        # in-progress fog state, turn count, supplies

To open a dungeon (resume if a session.db exists, else create one):

    python main.py dungeons/ancient-temple-of-torrel

Other commands:

    python main.py --list
    python main.py --reset dungeons/<name>
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

from session import (
    DUNGEON_DB_NAME,
    DUNGEON_JSON_NAME,
    Session,
    SchemaVersionMismatch,
)


PROJECT_ROOT = Path(__file__).parent
DEFAULT_DUNGEONS_DIR = PROJECT_ROOT / "dungeons"
RENDER_DIR = PROJECT_ROOT / "render_output"
DM_HTML = RENDER_DIR / "dm.html"
PLAYER_HTML = RENDER_DIR / "player.html"
DEFAULT_EDITOR_PORT = 8765

# Suppress repeat-opens of the same URL within this many seconds. See
# BrowserOpener — defensive against duplicate tab spawns from any source.
MIN_TAB_REOPEN_SECONDS = 60.0


def _ensure_editor_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Make sure `port` is bindable. If it's held by a stale instance
    of this same app (e.g. the previous run didn't shut down cleanly
    when the terminal closed), kill it and proceed. If something else
    owns the port, print a clear message and return False so `main`
    can exit non-zero.

    Returns True when the port is free or was successfully freed."""
    import errno
    import os
    import signal as _signal
    import socket
    import subprocess

    def _try_bind() -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            return False
        finally:
            s.close()

    if _try_bind():
        return True

    # Identify what's holding the port. lsof is preinstalled on macOS;
    # if it's missing we fall back to the generic error path.
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    except (FileNotFoundError, subprocess.SubprocessError):
        pids = []

    project_root = str(PROJECT_ROOT.resolve())
    own_pids: list[int] = []
    foreign_descriptions: list[str] = []
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            ps = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, timeout=5, check=False,
            )
            cmdline = ps.stdout.strip()
        except subprocess.SubprocessError:
            cmdline = ""
        # A stale instance of this app: a Python process running
        # main.py from this project directory. We won't kill anything
        # else.
        if "main.py" in cmdline and project_root in cmdline:
            own_pids.append(pid)
        else:
            foreign_descriptions.append(
                f"  pid={pid}: {cmdline[:90] or '(unknown command)'}"
            )

    for pid in own_pids:
        print(
            f"[port {port}] previous instance still running (pid={pid}); "
            f"sending SIGTERM",
            file=sys.stderr,
        )
        try:
            os.kill(pid, _signal.SIGTERM)
        except (OSError, ProcessLookupError):
            continue
        # Wait up to ~3 s for the OS to release the socket. Re-probe
        # rather than sleep blindly so the common case is fast.
        for _ in range(15):
            time.sleep(0.2)
            if _try_bind():
                return True
        # SIGTERM didn't work — escalate to SIGKILL and try once more.
        try:
            os.kill(pid, _signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        for _ in range(10):
            time.sleep(0.2)
            if _try_bind():
                return True

    if foreign_descriptions:
        print(
            f"\nPort {port} is in use by something other than this app:",
            file=sys.stderr,
        )
        for desc in foreign_descriptions:
            print(desc, file=sys.stderr)
    else:
        print(f"\nPort {port} is in use.", file=sys.stderr)
    print(
        f"\nTo free it manually:\n"
        f"  lsof -ti :{port} | xargs kill\n"
        f"...then re-run.",
        file=sys.stderr,
    )
    return False


class BrowserOpener:
    """webbrowser.open with per-URL rate limiting and stderr tracing."""

    def __init__(self, *, min_interval_seconds: float = MIN_TAB_REOPEN_SECONDS):
        self._monotonic = time.monotonic
        self._min_interval = min_interval_seconds
        self._last_open_times: dict[str, float] = {}

    def open(self, url: str, *, label: str = "tab") -> None:
        now = self._monotonic()
        last = self._last_open_times.get(url)
        if last is not None and now - last < self._min_interval:
            elapsed = now - last
            print(f"[browser] suppressed re-open of {label} ({url}) — "
                  f"opened {elapsed:.0f}s ago (cooldown {self._min_interval:.0f}s)",
                  file=sys.stderr)
            return
        self._last_open_times[url] = now
        print(f"[browser] opening {label}: {url}", file=sys.stderr)
        webbrowser.open(url, new=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSR Dungeon System — DM editor.",
        epilog=(
            "Examples:\n"
            "  python main.py dungeons/ancient-temple-of-torrel\n"
            "  python main.py --list\n"
            "  python main.py --reset dungeons/ancient-temple-of-torrel"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "dungeon", nargs="?", type=Path,
        help="Path to a dungeon folder containing dungeon.json + PNGs. "
             "If omitted, --list or --reset must be passed.",
    )
    p.add_argument("--list", action="store_true",
                   help=f"List dungeons under {DEFAULT_DUNGEONS_DIR.name}/ "
                        "and exit.")
    p.add_argument("--reset", type=Path, metavar="DUNGEON_FOLDER",
                   help="Delete the session.db inside the given dungeon "
                        "folder (annotations and level metadata kept). "
                        "Useful to start a fresh playthrough.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed RNG when creating a new session.")
    p.add_argument("--no-open", action="store_true",
                   help="Don't open the browser tabs at startup.")
    p.add_argument("--play", action="store_true",
                   help="Play mode: skip the localhost editor server and "
                        "the browser tabs. Annotation mode (A) still works "
                        "in-pygame for mid-session room sketches; the "
                        "browser room editor and the dungeon assistant "
                        "are both unavailable in play mode.")
    p.add_argument("--dungeons-dir", type=Path, default=DEFAULT_DUNGEONS_DIR,
                   help="Override the dungeons/ root directory used by --list.")
    return p.parse_args(argv)


def _list_dungeons(dungeons_dir: Path) -> int:
    infos = Session.list_dungeons(dungeons_dir)
    if not infos:
        print(f"No dungeons found under {dungeons_dir}.")
        return 0
    print(f"Dungeons under {dungeons_dir}:")
    for i in infos:
        if i.has_session:
            extra = (f"L{i.current_level} of {i.n_levels}, turn {i.current_turn}, "
                     f"saved {i.last_saved_at}")
        else:
            extra = f"{i.n_levels} levels, no progress yet"
        print(f"  {i.folder.name:35}  {i.name!r}  ({extra})")
    return 0


def _reset_dungeon(folder: Path) -> int:
    if not (folder / DUNGEON_JSON_NAME).exists():
        print(f"error: {folder} does not contain {DUNGEON_JSON_NAME}", file=sys.stderr)
        return 2
    removed = Session.reset_dungeon(folder)
    if removed:
        print(f"Reset {folder}: deleted session.db. Annotations preserved.")
    else:
        print(f"{folder} had no session.db — nothing to reset.")
    return 0


def _ensure_render_output_exists() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)


def _print_startup_summary(session, folder: Path) -> list[str]:
    """Print a multi-line summary of what was opened. Returns a list of
    user-visible warning strings (empty when everything looks normal),
    so the renderer can surface them as an in-window overlay too."""
    d = session.dungeon
    total_rooms = sum(len(lv.rooms) for lv in d.levels)
    per_level = " + ".join(str(len(lv.rooms)) for lv in d.levels)
    is_fresh = session.tracker.turn == 0
    db_path = folder / DUNGEON_DB_NAME
    json_path = folder / DUNGEON_JSON_NAME

    warnings: list[str] = []
    # If the dungeon.json was edited externally after the last session
    # save, the player view may not reflect those edits until they hit
    # something that triggers a reload. Worth a heads-up.
    if (not is_fresh and db_path.exists() and json_path.exists()
            and json_path.stat().st_mtime > db_path.stat().st_mtime + 1):
        warnings.append(
            "dungeon.json was modified after the last session save — "
            "your in-progress run may be missing recent edits."
        )

    print(f"Opened: {d.name}")
    print(f"  Levels: {len(d.levels)} (current: {d.current_level})")
    print(f"  Rooms:  {total_rooms} total ({per_level})")
    if is_fresh:
        print("  Session: fresh, turn 0")
    else:
        print(f"  Session: resuming, turn {session.tracker.turn} "
              f"(level {d.current_level} of {len(d.levels)})")
    for w in warnings:
        print(f"  ⚠ {w}")
    return warnings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        return _list_dungeons(args.dungeons_dir)
    if args.reset is not None:
        return _reset_dungeon(args.reset)
    if args.dungeon is None:
        print("error: must pass a dungeon folder path "
              "(or --list / --reset). See --help.", file=sys.stderr)
        return 2

    folder = args.dungeon.resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"error: {folder} is not a directory", file=sys.stderr)
        return 2
    json_path = folder / DUNGEON_JSON_NAME
    if not json_path.exists():
        print(f"error: {folder} has no {DUNGEON_JSON_NAME}", file=sys.stderr)
        return 2

    _ensure_render_output_exists()

    # Defer pygame import so --list / --reset work without SDL.
    from renderer import run
    import editor_server

    opener = BrowserOpener()
    first_iteration = True

    # Reload loop: each iteration opens a session + editor server and calls
    # run(). When the user clicks "Open Different Dungeon…" or confirms a
    # full reset, run() returns a ReloadRequest; we tear the iteration's
    # resources down, optionally wipe the dungeon JSON, and re-enter run()
    # against the new folder. The pygame display window survives across
    # iterations because run() does NOT call pygame.quit() on the reload
    # path — re-initialising SDL post-teardown is unreliable on macOS.
    while True:
        if first_iteration:
            first_iteration = False
        else:
            print(f"Reloading into {folder}…")
        try:
            session = Session.open_dungeon(folder, rng_seed=args.seed)
        except SchemaVersionMismatch as e:
            # User-facing error — no traceback. The exception message
            # already tells them exactly what to run to recover.
            print(f"error: {e}", file=sys.stderr)
            return 2

        warnings = _print_startup_summary(session, folder)

        # In --play mode the localhost editor server and the auto-opened
        # browser tabs are skipped: the DM is running the session, not
        # editing dungeon content. Annotation mode (A) inside pygame is
        # still available for mid-session sketches.
        server = None
        editor_url = None
        if not args.play:
            if not _ensure_editor_port_free(DEFAULT_EDITOR_PORT):
                return 2
            server, _thread = editor_server.start_editor_server(
                json_path, port=DEFAULT_EDITOR_PORT,
                dungeons_dir=args.dungeons_dir,
            )
            bound_port = server.server_address[1]
            editor_url = f"http://127.0.0.1:{bound_port}/"
            print(f"Room editor: {editor_url}")
        else:
            print("Play mode: editor server disabled, browser tabs skipped.")

        def open_all(_url=editor_url) -> None:
            if args.play:
                return
            opener.open(DM_HTML.resolve().as_uri(), label="DM map")
            opener.open(PLAYER_HTML.resolve().as_uri(), label="Player map")
            if _url is not None:
                opener.open(_url, label="editor")

        def open_player_tab() -> None:
            if args.play:
                return
            opener.open(PLAYER_HTML.resolve().as_uri(), label="Player map")

        def open_editor_tab(_url=editor_url) -> None:
            if args.play or _url is None:
                return
            opener.open(_url, label="editor")

        if not args.no_open and not args.play:
            open_all()

        try:
            request = run(
                session,
                dungeon_path=json_path,
                dungeons_dir=args.dungeons_dir,
                on_open_browser=open_all,
                on_open_editor=open_editor_tab,
                on_open_player=open_player_tab,
                startup_warnings=warnings,
            )
        finally:
            if server is not None:
                server.shutdown()
            # Explicit close: the renderer's run-loop also calls
            # session.close() on normal exit, but that path can be
            # missed if run() raises. Calling close() unconditionally
            # here flushes any pending state to SQLite and releases
            # the connection before we open the next dungeon.
            try:
                session.close()
            except Exception:
                # Already-closed sessions raise; that's fine — the
                # renderer just got there first.
                pass

        if request is None:
            return 0

        # Handle the reload request: optionally wipe the target folder,
        # then loop with the new folder so the next iteration opens it.
        folder = request.folder.resolve()
        json_path = folder / DUNGEON_JSON_NAME
        if request.do_full_reset:
            backup = Session.full_reset(folder)
            print(f"Full reset of {folder.name}: backup written to "
                  f"{backup.name}, dungeon.json wiped, session.db deleted.")


if __name__ == "__main__":
    sys.exit(main())
