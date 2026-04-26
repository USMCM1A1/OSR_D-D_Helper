"""Phase 3 test gate — SQLite persistence: create, save, resume, export."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import dungeon
from session import (
    SCHEMA_SQL,
    Session,
    SessionInfo,
    SessionNotFound,
    _serialize_rng,
    _deserialize_rng,
)


EXAMPLE_PATH = Path(__file__).parent.parent / "data" / "example_dungeon.json"


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "session.db"


@pytest.fixture
def example() -> dungeon.Dungeon:
    return dungeon.load(EXAMPLE_PATH)


@pytest.fixture
def fresh(db_path: Path, example: dungeon.Dungeon) -> Session:
    return Session.create(db_path, example, EXAMPLE_PATH, rng_seed=42)


# --- Schema / construction --------------------------------------------------


class TestSchema:
    def test_create_initializes_all_tables(self, db_path: Path, example: dungeon.Dungeon) -> None:
        Session.create(db_path, example, EXAMPLE_PATH).close()
        conn = sqlite3.connect(db_path)
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert {
            "schema_version", "sessions", "room_state", "resources",
            "characters", "active_effects", "turn_log", "party_position",
        } <= names

    def test_create_seeds_room_state_and_party_position(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        sid = s.session_id
        s.close()
        # Multi-level dungeons seed room_state for *every* level and a
        # party_position row per level (room defaults to rooms[0] of each).
        conn = sqlite3.connect(db_path)
        rs_count = conn.execute(
            "SELECT COUNT(*) FROM room_state WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        pp_count = conn.execute(
            "SELECT COUNT(*) FROM party_position WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        l1_pos = conn.execute(
            "SELECT room_id FROM party_position "
            "WHERE session_id = ? AND level_number = 1", (sid,)
        ).fetchone()
        conn.close()
        total_rooms_across_levels = sum(len(lv.rooms) for lv in example.levels)
        assert rs_count == total_rooms_across_levels
        assert pp_count == len(example.levels)
        assert l1_pos[0] == example.get_level(1).rooms[0].id


# --- RNG state round-trip ----------------------------------------------------


class TestRNGSerialization:
    def test_round_trip_preserves_sequence(self) -> None:
        import random
        rng = random.Random(123)
        # Burn some state.
        for _ in range(7):
            rng.random()
        blob = _serialize_rng(rng)
        restored = _deserialize_rng(blob)
        assert [rng.randint(1, 1_000_000) for _ in range(20)] == \
               [restored.randint(1, 1_000_000) for _ in range(20)]


# --- Save / resume round-trip -----------------------------------------------


class TestSaveResume:
    def test_round_trip_preserves_turn_and_journal(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s1 = Session.create(db_path, example, EXAMPLE_PATH, rng_seed=42)
        s1.add_light_source("torch")
        s1.add_light_source("torch")
        for _ in range(8):
            s1.advance_turn()
        before_turn = s1.tracker.turn
        before_journal = list(s1.tracker.journal.entries)
        before_lights = [(ls.kind, ls.label, ls.turns_remaining)
                         for ls in s1.tracker.light_sources]
        sid = s1.session_id
        s1.close()

        s2 = Session.resume(db_path, sid)
        assert s2.tracker.turn == before_turn
        assert list(s2.tracker.journal.entries) == before_journal
        assert [(ls.kind, ls.label, ls.turns_remaining)
                for ls in s2.tracker.light_sources] == before_lights
        s2.close()

    def test_resume_continues_rng_deterministically(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        # Run A: do everything in one session.
        s_a = Session.create(db_path / "a.db", example, EXAMPLE_PATH, rng_seed=42)
        s_a.add_light_source("torch")
        for _ in range(20):
            s_a.advance_turn()
        journal_a = list(s_a.tracker.journal.entries)
        s_a.close()

        # Run B: kill at turn 10, resume, advance another 10.
        s_b1 = Session.create(db_path / "b.db", example, EXAMPLE_PATH, rng_seed=42)
        s_b1.add_light_source("torch")
        for _ in range(10):
            s_b1.advance_turn()
        sid = s_b1.session_id
        s_b1.close()

        s_b2 = Session.resume(db_path / "b.db", sid)
        for _ in range(10):
            s_b2.advance_turn()
        journal_b = list(s_b2.tracker.journal.entries)
        s_b2.close()

        assert journal_a == journal_b

    def test_kill_and_resume_then_advance(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        """Phase 3 acceptance gate: simulate a process kill, then resume
        and advance one more turn — state intact, journal grows by one turn."""
        s1 = Session.create(db_path, example, EXAMPLE_PATH, rng_seed=7)
        for _ in range(5):
            s1.advance_turn()
        prev_len = len(s1.tracker.journal)
        sid = s1.session_id
        # Simulate a hard kill: just drop the connection without close().
        s1.conn.close()

        s2 = Session.resume(db_path, sid)
        assert s2.tracker.turn == 5
        assert len(s2.tracker.journal) == prev_len
        s2.advance_turn()
        assert s2.tracker.turn == 6
        # Two new entries: turn_advance + wm_check.
        assert len(s2.tracker.journal) == prev_len + 2
        s2.close()

    def test_resume_unknown_id_raises(self, db_path: Path, example: dungeon.Dungeon) -> None:
        Session.create(db_path, example, EXAMPLE_PATH).close()
        with pytest.raises(SessionNotFound):
            Session.resume(db_path, 999)


# --- Character exhaustion persistence ---------------------------------------


class TestCharacterPersistence:
    def test_exhaustion_round_trips(self, db_path: Path, example: dungeon.Dungeon) -> None:
        s1 = Session.create(db_path, example, EXAMPLE_PATH)
        s1.dungeon.party.characters[0].exhaustion = 4
        s1.save()
        sid = s1.session_id
        s1.close()
        s2 = Session.resume(db_path, sid)
        assert s2.dungeon.party.characters[0].exhaustion == 4
        s2.close()


# --- Journal export ----------------------------------------------------------


class TestJournalExport:
    def test_export_format_matches_in_memory_format(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        from journal import format_entry
        s = Session.create(db_path, example, EXAMPLE_PATH, rng_seed=42)
        for _ in range(5):
            s.advance_turn()
        expected = "\n".join(format_entry(e) for e in s.tracker.journal) + "\n"
        assert s.export_journal() == expected
        s.close()

    def test_export_round_trips_through_resume(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s1 = Session.create(db_path, example, EXAMPLE_PATH, rng_seed=42)
        for _ in range(5):
            s1.advance_turn()
        export_before = s1.export_journal()
        sid = s1.session_id
        s1.close()
        s2 = Session.resume(db_path, sid)
        assert s2.export_journal() == export_before
        s2.close()

    def test_empty_export_is_empty_string(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        assert s.export_journal() == ""
        s.close()


# --- list_sessions ----------------------------------------------------------


class TestListSessions:
    def test_lists_all_sessions(self, db_path: Path, example: dungeon.Dungeon) -> None:
        Session.create(db_path, example, EXAMPLE_PATH, name="Run 1").close()
        Session.create(db_path, example, EXAMPLE_PATH, name="Run 2").close()
        infos = Session.list_sessions(db_path)
        assert [i.name for i in infos] == ["Run 1", "Run 2"]
        assert all(isinstance(i, SessionInfo) for i in infos)

    def test_returns_empty_when_db_missing(self, tmp_path: Path) -> None:
        assert Session.list_sessions(tmp_path / "nope.db") == []


# --- Context manager --------------------------------------------------------


class TestContextManager:
    def test_with_block_saves_on_exit(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        with Session.create(db_path, example, EXAMPLE_PATH, rng_seed=42) as s:
            s.advance_turn()
            sid = s.session_id
        # After the with block: connection closed, but data persisted.
        with Session.resume(db_path, sid) as s2:
            assert s2.tracker.turn == 1


class TestLevelSwitching:
    def test_switch_descends_and_logs_transition(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH, rng_seed=42)
        assert s.current_level == 1
        assert s.can_descend() and not s.can_ascend()
        s.switch_level(+1)
        assert s.current_level == 2
        # Party landed on the stairs_up room (R10) because Level 2 has it.
        assert s.get_party_position() == "R10"
        # Journal recorded the transition.
        from journal import KIND_LEVEL_TRANSITION
        transitions = s.tracker.journal.of_kind(KIND_LEVEL_TRANSITION)
        assert len(transitions) == 1
        assert "descends" in transitions[0].message
        s.close()

    def test_switch_ascends_back_preserves_per_level_state(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        # Reveal R02 on Level 1.
        s.update_room_state("R02", "known")
        s.switch_level(+1)  # descend to L2
        # Reveal R11 on Level 2.
        s.update_room_state("R11", "known")
        s.switch_level(-1)  # ascend back to L1
        # R02 still revealed on L1.
        assert s.dungeon.get_level(1).rooms_by_id["R02"].state == "known"
        # And R11 was preserved on L2.
        assert s.dungeon.get_level(2).rooms_by_id["R11"].state == "known"
        s.close()

    def test_switch_at_boundary_raises(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        with pytest.raises(ValueError, match="no level"):
            s.switch_level(-1)  # already at level 1
        s.switch_level(+1)
        with pytest.raises(ValueError, match="no level"):
            s.switch_level(+1)  # already at deepest
        s.close()

    def test_resume_restores_current_level(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s1 = Session.create(db_path, example, EXAMPLE_PATH)
        s1.switch_level(+1)
        sid = s1.session_id
        s1.close()
        s2 = Session.resume(db_path, sid)
        assert s2.current_level == 2
        assert s2.dungeon.current_level == 2
        s2.close()
