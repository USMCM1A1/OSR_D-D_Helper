"""Pygame entry point — DM-side dungeon tracker.

The pygame window is the *editor* (renderer.py): drag rooms, click to
cycle state, scroll to zoom. Whenever you change something, the SVG
renderer (svg_render.py) rewrites `render_output/dm_map.svg` and
`render_output/player_map.svg`. Open the matching HTML pages in a
browser tab for the polished pen-and-ink view; both auto-refresh every
2 seconds. Press `V` from the editor to launch them.

Usage:
    python main.py --new data/example_dungeon.json
    python main.py --resume 3
    python main.py --list
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from dungeon import load as load_dungeon
from session import Session


PROJECT_ROOT = Path(__file__).parent
DEFAULT_DB = PROJECT_ROOT / "saves" / "session.db"
RENDER_DIR = PROJECT_ROOT / "render_output"
DM_HTML = RENDER_DIR / "dm.html"
PLAYER_HTML = RENDER_DIR / "player.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OSR Dungeon System — DM editor.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--new", type=Path, metavar="DUNGEON_JSON",
                   help="Start a new session from this dungeon JSON file.")
    g.add_argument("--resume", type=int, metavar="SESSION_ID",
                   help="Resume an existing session by id.")
    g.add_argument("--list", action="store_true",
                   help="List sessions in the db and exit.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB,
                   help=f"SQLite session db path (default: {DEFAULT_DB}).")
    p.add_argument("--name", type=str, default=None,
                   help="Optional name for a new session.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed RNG when creating a new session.")
    p.add_argument("--no-open", action="store_true",
                   help="Don't open the browser tabs at startup.")
    return p.parse_args(argv)


def _list_sessions(db_path: Path) -> int:
    infos = Session.list_sessions(db_path)
    if not infos:
        print(f"No sessions in {db_path}.")
        return 0
    print(f"Sessions in {db_path}:")
    for i in infos:
        print(f"  [{i.id:>3}] {i.name!r}  turn={i.current_turn}  "
              f"started={i.started_at}  saved={i.last_saved_at}")
    return 0


def _ensure_render_output_exists() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)


def _open_browser_tabs() -> None:
    """Open both DM and Player HTML pages in the default browser."""
    for path in (DM_HTML, PLAYER_HTML):
        webbrowser.open(path.resolve().as_uri(), new=2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        return _list_sessions(args.db)

    if args.new is not None:
        dungeon = load_dungeon(args.new)
        session = Session.create(
            args.db, dungeon, args.new,
            name=args.name, rng_seed=args.seed,
        )
        dungeon_path = Path(args.new).resolve()
        print(f"Created session {session.session_id} ({session.dungeon.name}).")
    else:
        session = Session.resume(args.db, args.resume)
        # Resume reads dungeon_file from the sessions table.
        cur = session.conn.execute(
            "SELECT dungeon_file FROM sessions WHERE id = ?",
            (session.session_id,),
        )
        row = cur.fetchone()
        dungeon_path = Path(row[0]).resolve() if row else None
        print(f"Resumed session {session.session_id} at turn {session.tracker.turn}.")

    _ensure_render_output_exists()

    # Defer pygame import so --list works without SDL.
    from renderer import run

    if not args.no_open:
        _open_browser_tabs()

    # The renderer snapshots PNGs to render_output/ on every state change;
    # the HTML pages auto-refresh from those files via a 2 s JS poll.
    run(session, dungeon_path=dungeon_path, on_open_browser=_open_browser_tabs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
