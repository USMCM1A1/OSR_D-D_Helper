"""SQLite-backed session persistence with multi-level dungeons.

A single .db file holds many sessions; rows are scoped by session_id.
For multi-level dungeons, per-level state (room reveals, alignment
positions, party position, active effects) is further scoped by
level_number so each level remembers its own fog-of-war and layout.

Public API:
    Session.create(db_path, dungeon, dungeon_file, *, name=None, rng_seed=None)
    Session.resume(db_path, session_id)
    Session.list_sessions(db_path) -> list[SessionInfo]
    Session.save() / .close()
    Session.export_journal() -> str
    Session.switch_level(direction) / .set_current_level(n)
    Session.update_room_state / .update_room_position (level-scoped)
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config
import journal as journal_mod
from dungeon import Dungeon, DungeonValidationError, load as load_dungeon
from journal import Journal, JournalEntry, format_entry
from tracker import LightSource, Tracker


SCHEMA_VERSION = 3


# Pool-tracked supplies (per-session counts, not per-character). Defaults
# tuned for a 4-PC delve: torches at 1/turn × 6 turns = enough for ~6 turn-
# zones lit; 1 lantern + 3 oil flasks ≈ ~108 turns of light reserve;
# rations at 1/PC/day × 4 PCs × 6 days; water at 1 gallon/day × 1.5 days
# × 4 PCs (carry weight is the practical cap on water).
DEFAULT_SUPPLY_KINDS = ("torches", "hooded_lantern", "oil_flask",
                        "ration", "water_gallon")
DEFAULT_SUPPLY_COUNTS: dict[str, int] = {
    "torches": 4,
    "hooded_lantern": 1,
    "oil_flask": 3,
    "ration": 24,
    "water_gallon": 6,
}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    dungeon_file  TEXT NOT NULL,
    party_level   INTEGER NOT NULL,
    started_at    TEXT NOT NULL,
    last_saved_at TEXT NOT NULL,
    current_turn  INTEGER NOT NULL DEFAULT 0,
    current_level INTEGER NOT NULL DEFAULT 1,
    rng_state     TEXT,
    notes         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS room_state (
    session_id   INTEGER NOT NULL,
    level_number INTEGER NOT NULL,
    room_id      TEXT NOT NULL,
    state        TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    x            REAL,
    y            REAL,
    PRIMARY KEY (session_id, level_number, room_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS resources (
    session_id      INTEGER NOT NULL,
    slot            INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    label           TEXT NOT NULL,
    turns_remaining INTEGER NOT NULL,
    PRIMARY KEY (session_id, slot),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS characters (
    session_id INTEGER NOT NULL,
    name       TEXT NOT NULL,
    darkvision INTEGER NOT NULL,
    exhaustion INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, name),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS active_effects (
    session_id      INTEGER NOT NULL,
    level_number    INTEGER NOT NULL,
    slot            INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    label           TEXT NOT NULL,
    turns_remaining INTEGER NOT NULL,
    PRIMARY KEY (session_id, level_number, slot),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS turn_log (
    session_id INTEGER NOT NULL,
    sequence   INTEGER NOT NULL,
    turn       INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    message    TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS party_position (
    session_id   INTEGER NOT NULL,
    level_number INTEGER NOT NULL,
    room_id      TEXT NOT NULL,
    PRIMARY KEY (session_id, level_number),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS supplies (
    session_id INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    count      INTEGER NOT NULL,
    PRIMARY KEY (session_id, kind),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
"""


class SessionNotFound(LookupError):
    """No session with the requested id exists in the database."""


@dataclass(frozen=True)
class SessionInfo:
    id: int
    name: str
    dungeon_file: str
    party_level: int
    started_at: str
    last_saved_at: str
    current_turn: int
    current_level: int


@dataclass(frozen=True)
class DungeonInfo:
    """Summary of a dungeon folder shown by --list and the in-app picker."""
    folder: Path
    name: str
    n_levels: int
    has_session: bool
    current_turn: int   # 0 if no session
    current_level: int  # 1 if no session
    last_saved_at: str  # "" if no session


# Filenames inside a dungeon folder.
DUNGEON_JSON_NAME = "dungeon.json"
DUNGEON_DB_NAME = "session.db"


# --- Connection / schema -----------------------------------------------------


def _open_db(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
    if cur.fetchone() is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- RNG state serialization -------------------------------------------------


def _serialize_rng(rng: random.Random) -> str:
    state = rng.getstate()
    return json.dumps(state, default=list)


def _deserialize_rng(blob: str) -> random.Random:
    raw = json.loads(blob)
    state = (raw[0], tuple(raw[1]), raw[2])
    rng = random.Random()
    rng.setstate(state)
    return rng


# --- Helpers -----------------------------------------------------------------


def _default_room_for_level(level) -> str:
    """Pick the default starting room when entering a level for the first time."""
    if level.rooms:
        return level.rooms[0].id
    raise ValueError(f"Level {level.level_number} has no rooms")


def _entry_room_for_descend(level) -> str:
    """When descending INTO `level`, prefer a stairs_up room (foot of the
    stair just descended); fall back to the first room."""
    sup = level.stairs_up_room_id()
    return sup if sup is not None else _default_room_for_level(level)


def _entry_room_for_ascend(level) -> str:
    """When ascending INTO `level`, prefer a stairs_down room."""
    sdown = level.stairs_down_room_id()
    return sdown if sdown is not None else _default_room_for_level(level)


# --- Session class -----------------------------------------------------------


class Session:
    """An open session — a Tracker bound to a SQLite row."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        session_id: int,
        dungeon: Dungeon,
        tracker: Tracker,
    ) -> None:
        self.conn = conn
        self.session_id = session_id
        self.dungeon = dungeon
        self.tracker = tracker
        self._journal_offset = 0

    # --- Constructors ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        db_path: str | Path,
        dungeon: Dungeon,
        dungeon_file: str | Path,
        *,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> "Session":
        conn = _open_db(db_path)
        session_name = name if name is not None else dungeon.name
        now = _now_iso()
        cur = conn.execute(
            "INSERT INTO sessions "
            "(name, dungeon_file, party_level, started_at, last_saved_at, "
            "current_turn, current_level) "
            "VALUES (?, ?, ?, ?, ?, 0, ?)",
            (session_name, str(dungeon_file), dungeon.party_level, now, now,
             dungeon.current_level),
        )
        sid = cur.lastrowid
        assert sid is not None
        conn.commit()

        rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        tracker = Tracker(dungeon, rng=rng)

        # Seed room_state for every annotated room. Levels without rooms
        # (yet to be annotated) get *no* party_position row — the runtime
        # will insert one when the DM annotates the first room.
        room_rows = []
        party_rows = []
        for level in dungeon.levels:
            for r in level.rooms:
                room_rows.append((sid, level.level_number, r.id, r.state, r.notes))
            if level.rooms:
                party_rows.append(
                    (sid, level.level_number, level.rooms[0].id)
                )
        if room_rows:
            conn.executemany(
                "INSERT INTO room_state (session_id, level_number, room_id, state, notes) "
                "VALUES (?, ?, ?, ?, ?)",
                room_rows,
            )
        if party_rows:
            conn.executemany(
                "INSERT INTO party_position (session_id, level_number, room_id) "
                "VALUES (?, ?, ?)",
                party_rows,
            )
        # Seed pool-tracked supplies with default starter counts.
        conn.executemany(
            "INSERT INTO supplies (session_id, kind, count) VALUES (?, ?, ?)",
            [(sid, kind, DEFAULT_SUPPLY_COUNTS[kind])
             for kind in DEFAULT_SUPPLY_KINDS],
        )
        conn.commit()

        session = cls(conn, sid, dungeon, tracker)
        session.save()
        return session

    @classmethod
    def resume(cls, db_path: str | Path, session_id: int) -> "Session":
        conn = _open_db(db_path)
        cur = conn.execute(
            "SELECT name, dungeon_file, current_turn, current_level, rng_state "
            "FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            raise SessionNotFound(f"No session with id {session_id}")
        _, dungeon_file, current_turn, current_level, rng_state = row

        dungeon = load_dungeon(dungeon_file)
        dungeon.current_level = int(current_level)
        rng = _deserialize_rng(rng_state) if rng_state else random.Random()
        tracker = Tracker(dungeon, rng=rng)
        tracker.turn = current_turn

        # Journal
        cur = conn.execute(
            "SELECT turn, kind, message FROM turn_log "
            "WHERE session_id = ? ORDER BY sequence",
            (session_id,),
        )
        for turn, kind, message in cur.fetchall():
            tracker.journal.entries.append(JournalEntry(turn=turn, kind=kind, message=message))

        # Light sources
        cur = conn.execute(
            "SELECT kind, label, turns_remaining FROM resources "
            "WHERE session_id = ? ORDER BY slot",
            (session_id,),
        )
        for kind, label, remaining in cur.fetchall():
            tracker.light_sources.append(LightSource(kind=kind, label=label, turns_remaining=remaining))

        # Character exhaustion
        cur = conn.execute(
            "SELECT name, exhaustion FROM characters WHERE session_id = ?",
            (session_id,),
        )
        exhaustion_by_name = dict(cur.fetchall())
        for c in dungeon.party.characters:
            if c.name in exhaustion_by_name:
                c.exhaustion = int(exhaustion_by_name[c.name])

        # Per-level room states.
        cur = conn.execute(
            "SELECT level_number, room_id, state FROM room_state "
            "WHERE session_id = ?",
            (session_id,),
        )
        for level_number, room_id, state in cur.fetchall():
            level = dungeon.levels_by_number.get(int(level_number))
            if level is None:
                continue
            room = level.rooms_by_id.get(room_id)
            if room is not None:
                room.state = state

        session = cls(conn, session_id, dungeon, tracker)
        session._journal_offset = len(tracker.journal)
        return session

    # --- Dungeon-folder API (preferred) ------------------------------------

    @classmethod
    def open_dungeon(
        cls,
        folder: str | Path,
        *,
        rng_seed: int | None = None,
    ) -> "Session":
        """Open the dungeon at `folder`. If `<folder>/session.db` already
        has a session row, resume the most recent one; else create a fresh
        session. The folder must contain `dungeon.json`."""
        folder = Path(folder)
        json_path = folder / DUNGEON_JSON_NAME
        db_path = folder / DUNGEON_DB_NAME
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing {DUNGEON_JSON_NAME} in {folder}"
            )
        # If a session.db already exists with at least one row, resume.
        # Order by id DESC as a tiebreaker — last_saved_at is second-resolution,
        # so two sessions saved in the same second tie on timestamp.
        if db_path.exists():
            conn = _open_db(db_path)
            try:
                row = conn.execute(
                    "SELECT id FROM sessions "
                    "ORDER BY last_saved_at DESC, id DESC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            if row is not None:
                return cls.resume(db_path, row[0])
        # Otherwise create a new session in this folder.
        dungeon = load_dungeon(json_path)
        return cls.create(db_path, dungeon, json_path, rng_seed=rng_seed)

    @staticmethod
    def reset_dungeon(folder: str | Path) -> bool:
        """Delete `<folder>/session.db` so the next open starts fresh.
        Annotations + level metadata in dungeon.json are untouched.
        Returns True if a DB existed and was removed."""
        db = Path(folder) / DUNGEON_DB_NAME
        if not db.exists():
            return False
        db.unlink()
        return True

    @staticmethod
    def full_reset(folder: str | Path) -> Path:
        """Wipe a dungeon down to its level skeleton:
          - back up the current dungeon.json to a timestamped .bak
          - rewrite dungeon.json with rooms[]=[] and corridors[]=[] for
            every level (level metadata + WM tables preserved)
          - delete session.db if it exists

        The dungeon name, party config, level images, and WM tables are
        preserved so the dungeon can be re-annotated from scratch. Returns
        the path of the backup file written. Raises FileNotFoundError if
        the folder has no dungeon.json."""
        folder = Path(folder)
        json_path = folder / DUNGEON_JSON_NAME
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing {DUNGEON_JSON_NAME} in {folder}"
            )
        # Back up the current JSON before we touch it.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = folder / f"{DUNGEON_JSON_NAME}.{ts}.bak"
        backup.write_bytes(json_path.read_bytes())
        # Rewrite the JSON with empty rooms+corridors per level. We work
        # on the raw dict (not the validated Dungeon) so we don't have to
        # round-trip through the loader for fields the loader would reject.
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        for lv in raw.get("levels", []):
            lv["rooms"] = []
            lv["corridors"] = []
        json_path.write_text(json.dumps(raw, indent=2) + "\n",
                             encoding="utf-8")
        # Drop the session DB.
        db = folder / DUNGEON_DB_NAME
        if db.exists():
            db.unlink()
        return backup

    @staticmethod
    def list_dungeons(root: str | Path) -> list[DungeonInfo]:
        """Walk `root` (typically PROJECT_ROOT/dungeons/) and return one
        DungeonInfo per folder containing a valid dungeon.json."""
        root = Path(root)
        out: list[DungeonInfo] = []
        if not root.exists() or not root.is_dir():
            return out
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            json_path = child / DUNGEON_JSON_NAME
            if not json_path.exists():
                continue
            try:
                d = load_dungeon(json_path)
            except (DungeonValidationError, FileNotFoundError):
                continue
            db_path = child / DUNGEON_DB_NAME
            current_turn = 0
            current_level = d.current_level
            last_saved = ""
            has_session = False
            if db_path.exists():
                conn = _open_db(db_path)
                try:
                    row = conn.execute(
                        "SELECT current_turn, current_level, last_saved_at "
                        "FROM sessions "
                        "ORDER BY last_saved_at DESC, id DESC LIMIT 1"
                    ).fetchone()
                finally:
                    conn.close()
                if row is not None:
                    has_session = True
                    current_turn, current_level, last_saved = row
            out.append(DungeonInfo(
                folder=child,
                name=d.name,
                n_levels=len(d.levels),
                has_session=has_session,
                current_turn=int(current_turn),
                current_level=int(current_level),
                last_saved_at=last_saved or "",
            ))
        return out

    # --- Legacy session-DB introspection -----------------------------------

    @staticmethod
    def list_sessions(db_path: str | Path) -> list[SessionInfo]:
        if not Path(db_path).exists():
            return []
        conn = _open_db(db_path)
        try:
            cur = conn.execute(
                "SELECT id, name, dungeon_file, party_level, started_at, "
                "last_saved_at, current_turn, current_level FROM sessions ORDER BY id"
            )
            return [SessionInfo(*row) for row in cur.fetchall()]
        finally:
            conn.close()

    # --- Tracker delegations (auto-save) -------------------------------------

    def advance_turn(self) -> list[JournalEntry]:
        new = self.tracker.advance_turn()
        self.save()
        return new

    def short_rest(self) -> list[JournalEntry]:
        new = self.tracker.short_rest()
        self.save()
        return new

    def long_rest(self) -> list[JournalEntry]:
        new = self.tracker.long_rest()
        self.save()
        return new

    def add_light_source(self, kind: str, label: str | None = None) -> LightSource:
        ls = self.tracker.add_light_source(kind, label)
        self.save()
        return ls

    # --- Save / export -------------------------------------------------------

    def save(self) -> None:
        with self.conn:
            self._flush_journal()
            self._flush_resources()
            self._flush_characters()
            self._flush_meta()

    def _flush_journal(self) -> None:
        new = self.tracker.journal.entries[self._journal_offset:]
        if not new:
            return
        rows = [
            (self.session_id, self._journal_offset + i, e.turn, e.kind, e.message)
            for i, e in enumerate(new)
        ]
        self.conn.executemany(
            "INSERT INTO turn_log (session_id, sequence, turn, kind, message) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._journal_offset = len(self.tracker.journal)

    def _flush_resources(self) -> None:
        self.conn.execute("DELETE FROM resources WHERE session_id = ?", (self.session_id,))
        rows = [
            (self.session_id, slot, ls.kind, ls.label, ls.turns_remaining)
            for slot, ls in enumerate(self.tracker.light_sources)
        ]
        if rows:
            self.conn.executemany(
                "INSERT INTO resources (session_id, slot, kind, label, turns_remaining) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def _flush_characters(self) -> None:
        self.conn.execute("DELETE FROM characters WHERE session_id = ?", (self.session_id,))
        rows = [
            (self.session_id, c.name, int(c.darkvision), int(c.exhaustion))
            for c in self.dungeon.party.characters
        ]
        self.conn.executemany(
            "INSERT INTO characters (session_id, name, darkvision, exhaustion) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )

    def _flush_meta(self) -> None:
        self.conn.execute(
            "UPDATE sessions SET current_turn = ?, current_level = ?, "
            "last_saved_at = ?, rng_state = ? WHERE id = ?",
            (self.tracker.turn, self.dungeon.current_level,
             _now_iso(), _serialize_rng(self.tracker.rng), self.session_id),
        )

    def export_journal(self) -> str:
        if not self.tracker.journal.entries:
            return ""
        return "\n".join(format_entry(e) for e in self.tracker.journal) + "\n"

    # --- Per-room mutations (level-scoped) -----------------------------------

    @property
    def current_level(self) -> int:
        return self.dungeon.current_level

    def _level_or_current(self, level_number: int | None) -> int:
        return self.current_level if level_number is None else int(level_number)

    def update_room_state(self, room_id: str, state: str, *,
                          level_number: int | None = None) -> None:
        if state not in config.ROOM_STATES:
            raise ValueError(f"unknown room state: {state!r}")
        ln = self._level_or_current(level_number)
        level = self.dungeon.levels_by_number.get(ln)
        if level is None:
            raise KeyError(f"unknown level_number: {ln!r}")
        room = level.rooms_by_id.get(room_id)
        if room is None:
            raise KeyError(f"unknown room id {room_id!r} on level {ln}")
        room.state = state
        with self.conn:
            self.conn.execute(
                "UPDATE room_state SET state = ? "
                "WHERE session_id = ? AND level_number = ? AND room_id = ?",
                (state, self.session_id, ln, room_id),
            )

    def restore_fog_of_war(self) -> int:
        """Reset every room on every level to 'unexplored'. Turn count,
        supplies, journal, party position, character exhaustion, and
        active effects are all preserved — this is a pure visual reset
        of the fog mask. Returns the number of rooms updated."""
        # In-memory: walk every level's rooms.
        n = 0
        for level in self.dungeon.levels:
            for room in level.rooms:
                if room.state != "unexplored":
                    n += 1
                room.state = "unexplored"
        # Persist: one UPDATE for the whole session.
        with self.conn:
            self.conn.execute(
                "UPDATE room_state SET state = 'unexplored' "
                "WHERE session_id = ?",
                (self.session_id,),
            )
        return n

    def update_room_position(self, room_id: str, x: float, y: float, *,
                             level_number: int | None = None) -> None:
        ln = self._level_or_current(level_number)
        level = self.dungeon.levels_by_number.get(ln)
        if level is None:
            raise KeyError(f"unknown level_number: {ln!r}")
        if room_id not in level.rooms_by_id:
            raise KeyError(f"unknown room id {room_id!r} on level {ln}")
        with self.conn:
            self.conn.execute(
                "UPDATE room_state SET x = ?, y = ? "
                "WHERE session_id = ? AND level_number = ? AND room_id = ?",
                (float(x), float(y), self.session_id, ln, room_id),
            )

    def get_room_positions(self, level_number: int | None = None) -> dict[str, tuple[float, float] | None]:
        ln = self._level_or_current(level_number)
        cur = self.conn.execute(
            "SELECT room_id, x, y FROM room_state "
            "WHERE session_id = ? AND level_number = ?",
            (self.session_id, ln),
        )
        out: dict[str, tuple[float, float] | None] = {}
        for room_id, x, y in cur.fetchall():
            out[room_id] = (float(x), float(y)) if x is not None and y is not None else None
        return out

    def update_party_position(self, room_id: str, *,
                              level_number: int | None = None) -> None:
        ln = self._level_or_current(level_number)
        level = self.dungeon.levels_by_number.get(ln)
        if level is None:
            raise KeyError(f"unknown level_number: {ln!r}")
        if room_id not in level.rooms_by_id:
            raise KeyError(f"unknown room id {room_id!r} on level {ln}")
        with self.conn:
            self.conn.execute(
                "UPDATE party_position SET room_id = ? "
                "WHERE session_id = ? AND level_number = ?",
                (room_id, self.session_id, ln),
            )

    def get_party_position(self, level_number: int | None = None) -> str:
        ln = self._level_or_current(level_number)
        cur = self.conn.execute(
            "SELECT room_id FROM party_position "
            "WHERE session_id = ? AND level_number = ?",
            (self.session_id, ln),
        )
        row = cur.fetchone()
        if row is None:
            raise LookupError(f"no party_position for session {self.session_id} level {ln}")
        return row[0]

    # --- Level switching -----------------------------------------------------

    def switch_level(self, direction: int) -> int:
        """Move +1 (descend) or -1 (ascend) levels. Returns the new
        level_number. Raises ValueError if at a boundary."""
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 (descend) or -1 (ascend)")
        new_number = self.current_level + direction
        if new_number not in self.dungeon.levels_by_number:
            raise ValueError(
                f"no level {new_number} (current {self.current_level}, "
                f"have {sorted(self.dungeon.levels_by_number)})"
            )
        return self.set_current_level(new_number)

    def set_current_level(self, level_number: int) -> int:
        """Switch to `level_number`. Logs the transition, sets the party
        position to the relevant stair room, flushes state to disk."""
        if level_number == self.current_level:
            return self.current_level
        if level_number not in self.dungeon.levels_by_number:
            raise ValueError(f"no level {level_number}")

        going_deeper = level_number > self.current_level
        new_level = self.dungeon.levels_by_number[level_number]

        self.save()
        self.dungeon.current_level = level_number
        # Pick the entry room only if the destination has any annotated rooms.
        if new_level.rooms:
            entry = (_entry_room_for_descend(new_level) if going_deeper
                     else _entry_room_for_ascend(new_level))
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO party_position "
                    "(session_id, level_number, room_id) VALUES (?, ?, ?)",
                    (self.session_id, level_number, entry),
                )
        verb = "descends" if going_deeper else "ascends"
        msg = f"Party {verb} to {new_level.display_name}"
        self.tracker.journal.record(
            self.tracker.turn, journal_mod.KIND_LEVEL_TRANSITION, msg,
        )
        self.save()
        return level_number

    def can_ascend(self) -> bool:
        return self.current_level > self.dungeon.shallowest_level_number

    def can_descend(self) -> bool:
        return self.current_level < self.dungeon.deepest_level_number

    # --- Pool-tracked supplies ----------------------------------------------

    def get_supplies(self) -> dict[str, int]:
        """Current count of every pool-tracked supply for this session."""
        cur = self.conn.execute(
            "SELECT kind, count FROM supplies WHERE session_id = ?",
            (self.session_id,),
        )
        return {kind: int(count) for kind, count in cur.fetchall()}

    def set_supply_count(self, kind: str, count: int) -> None:
        """Replace the running count for `kind`. Use for one-off corrections
        (typo, found-loot adjustments). Routine consumption goes through
        consume_supply so it ends up in the journal."""
        if count < 0:
            raise ValueError(f"supply count cannot be negative: {count}")
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO supplies (session_id, kind, count) "
                "VALUES (?, ?, ?)",
                (self.session_id, kind, int(count)),
            )

    def consume_supply(self, kind: str, n: int = 1) -> int:
        """Decrement `kind` by `n`, clamping at 0. Returns the new count.
        Logs a journal entry so the DM can scroll back through usage."""
        if n < 0:
            raise ValueError(f"consume_supply n must be ≥ 0, got {n}")
        current = self.get_supplies().get(kind, 0)
        new_count = max(0, current - n)
        self.set_supply_count(kind, new_count)
        if n > 0:
            self.tracker.journal.record(
                self.tracker.turn, journal_mod.KIND_NOTE,
                f"Consumed {n}× {kind} (now {new_count})",
            )
            # Persist the journal entry.
            self.save()
        return new_count

    # --- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        try:
            self.save()
        finally:
            self.conn.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
