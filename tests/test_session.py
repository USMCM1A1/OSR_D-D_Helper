"""Phase 3 test gate — SQLite persistence: create, save, resume, export."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import dungeon
from session import (
    SCHEMA_SQL,
    SCHEMA_VERSION,
    Session,
    SessionInfo,
    SessionNotFound,
    SchemaVersionMismatch,
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


class TestSchemaVersionGuard:
    """Phase 3: opening a session.db with an unfamiliar schema_version
    must raise a clear error pointing at the recovery command, instead
    of silently lying about state."""

    def test_mismatch_raises(self, tmp_path: Path) -> None:
        # Need a real dungeon folder layout for open_dungeon to find
        # dungeon.json (the schema check fires inside _open_db, but
        # open_dungeon validates the JSON presence before that).
        import shutil
        folder = tmp_path / "test-dungeon"
        folder.mkdir()
        shutil.copy(EXAMPLE_PATH, folder / "dungeon.json")
        # Open + close to materialise session.db at the right schema.
        Session.open_dungeon(folder).close()
        # Tamper with the schema_version row to simulate a future build's db.
        db = folder / "session.db"
        conn = sqlite3.connect(db)
        conn.execute("UPDATE schema_version SET version = ?",
                     (SCHEMA_VERSION + 999,))
        conn.commit()
        conn.close()
        with pytest.raises(SchemaVersionMismatch, match="--reset"):
            Session.open_dungeon(folder)

    def test_same_version_loads(self, db_path: Path, example: dungeon.Dungeon) -> None:
        # Sanity check: matching version still opens cleanly.
        Session.create(db_path, example, EXAMPLE_PATH).close()
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        conn.close()
        assert row[0] == SCHEMA_VERSION


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


class TestSupplies:
    """Pool-tracked supplies (torches, lanterns, oil, rations, water).
    Seeded at session creation, mutable via consume_supply/set_supply_count,
    journal entries on consumption, persist across resume."""

    def test_create_seeds_default_supplies(self, db_path: Path,
                                            example: dungeon.Dungeon) -> None:
        from session import DEFAULT_SUPPLY_KINDS, DEFAULT_SUPPLY_COUNTS
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            supplies = s.get_supplies()
            for kind in DEFAULT_SUPPLY_KINDS:
                assert supplies[kind] == DEFAULT_SUPPLY_COUNTS[kind]
        finally:
            s.close()

    def test_consume_supply_decrements(self, db_path: Path,
                                        example: dungeon.Dungeon) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            before = s.get_supplies()["ration"]
            new = s.consume_supply("ration", 4)
            assert new == before - 4
            assert s.get_supplies()["ration"] == before - 4
        finally:
            s.close()

    def test_consume_clamps_at_zero(self, db_path: Path,
                                    example: dungeon.Dungeon) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            s.set_supply_count("torches", 2)
            new = s.consume_supply("torches", 5)
            assert new == 0
        finally:
            s.close()

    def test_consume_writes_journal_entry(self, db_path: Path,
                                           example: dungeon.Dungeon) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            s.consume_supply("torches", 1)
            from journal import KIND_NOTE
            notes = s.tracker.journal.of_kind(KIND_NOTE)
            assert any("torches" in e.message for e in notes)
        finally:
            s.close()

    def test_consume_zero_is_noop(self, db_path: Path,
                                   example: dungeon.Dungeon) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            before_journal = len(s.tracker.journal)
            s.consume_supply("ration", 0)
            assert len(s.tracker.journal) == before_journal
        finally:
            s.close()

    def test_supplies_persist_across_resume(self, db_path: Path,
                                             example: dungeon.Dungeon) -> None:
        s1 = Session.create(db_path, example, EXAMPLE_PATH)
        s1.consume_supply("oil_flask", 1)
        s1.consume_supply("ration", 4)
        sid = s1.session_id
        before = s1.get_supplies()
        s1.close()
        s2 = Session.resume(db_path, sid)
        try:
            after = s2.get_supplies()
            assert after == before
        finally:
            s2.close()

    def test_negative_count_raises(self, db_path: Path,
                                    example: dungeon.Dungeon) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            with pytest.raises(ValueError):
                s.set_supply_count("torches", -1)
        finally:
            s.close()


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


# --- Dungeon-folder API (open_dungeon / reset_dungeon / list_dungeons) ------


import shutil  # noqa: E402

from session import (  # noqa: E402
    DUNGEON_DB_NAME,
    DUNGEON_JSON_NAME,
    DungeonInfo,
)


@pytest.fixture
def dungeon_folder(tmp_path: Path) -> Path:
    """A self-contained dungeon folder with dungeon.json copied from the
    example fixture. No PNGs — tests don't load images."""
    folder = tmp_path / "test-dungeon"
    folder.mkdir()
    shutil.copy(EXAMPLE_PATH, folder / DUNGEON_JSON_NAME)
    return folder


class TestOpenDungeon:
    def test_creates_session_when_none_exists(self, dungeon_folder: Path) -> None:
        s = Session.open_dungeon(dungeon_folder, rng_seed=7)
        try:
            assert s.tracker.turn == 0
            assert (dungeon_folder / DUNGEON_DB_NAME).exists()
        finally:
            s.close()

    def test_resumes_existing_session(self, dungeon_folder: Path) -> None:
        s1 = Session.open_dungeon(dungeon_folder, rng_seed=7)
        s1.advance_turn()
        s1.advance_turn()
        s1.close()

        s2 = Session.open_dungeon(dungeon_folder)
        try:
            assert s2.tracker.turn == 2  # picked up where we left off
        finally:
            s2.close()

    def test_resumes_most_recent_when_multiple_rows(
        self, dungeon_folder: Path
    ) -> None:
        # First session.
        s1 = Session.open_dungeon(dungeon_folder, rng_seed=1)
        s1.advance_turn()
        s1.close()
        # Force a second session in the same DB by calling Session.create
        # directly (multiple sessions in one DB are legal but rare).
        d = dungeon.load(dungeon_folder / DUNGEON_JSON_NAME)
        db_path = dungeon_folder / DUNGEON_DB_NAME
        s2 = Session.create(db_path, d, dungeon_folder / DUNGEON_JSON_NAME,
                            rng_seed=2)
        for _ in range(5):
            s2.advance_turn()
        s2.close()
        # open_dungeon should pick up the most-recent (5-turn) session,
        # not the older one.
        s3 = Session.open_dungeon(dungeon_folder)
        try:
            assert s3.tracker.turn == 5
        finally:
            s3.close()

    def test_raises_when_dungeon_json_missing(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            Session.open_dungeon(empty)


class TestResetDungeon:
    def test_returns_false_when_no_db(self, dungeon_folder: Path) -> None:
        assert Session.reset_dungeon(dungeon_folder) is False

    def test_unlinks_existing_db(self, dungeon_folder: Path) -> None:
        Session.open_dungeon(dungeon_folder).close()
        assert (dungeon_folder / DUNGEON_DB_NAME).exists()
        assert Session.reset_dungeon(dungeon_folder) is True
        assert not (dungeon_folder / DUNGEON_DB_NAME).exists()
        # JSON is preserved.
        assert (dungeon_folder / DUNGEON_JSON_NAME).exists()

    def test_reset_then_open_starts_fresh(self, dungeon_folder: Path) -> None:
        s1 = Session.open_dungeon(dungeon_folder)
        for _ in range(3):
            s1.advance_turn()
        s1.close()
        Session.reset_dungeon(dungeon_folder)
        s2 = Session.open_dungeon(dungeon_folder)
        try:
            assert s2.tracker.turn == 0
        finally:
            s2.close()


class TestRestoreFogOfWar:
    def test_resets_all_rooms_to_unexplored(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            # Reveal a few rooms across both levels.
            l1 = s.dungeon.get_level(1)
            l2 = s.dungeon.get_level(2)
            s.update_room_state(l1.rooms[0].id, "known", level_number=1)
            s.update_room_state(l2.rooms[0].id, "cleared", level_number=2)
            assert l1.rooms[0].state == "known"
            assert l2.rooms[0].state == "cleared"

            n = s.restore_fog_of_war()
            assert n >= 2  # at least the two we just changed
            for level in s.dungeon.levels:
                for room in level.rooms:
                    assert room.state == "unexplored"
        finally:
            s.close()

    def test_persists_across_resume(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        s.update_room_state(s.dungeon.get_level(1).rooms[0].id, "known")
        s.restore_fog_of_war()
        sid = s.session_id
        s.close()
        s2 = Session.resume(db_path, sid)
        try:
            for level in s2.dungeon.levels:
                for room in level.rooms:
                    assert room.state == "unexplored"
        finally:
            s2.close()

    def test_does_not_touch_turn_supplies_journal(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH, rng_seed=1)
        try:
            s.advance_turn()
            s.advance_turn()
            turn_before = s.tracker.turn
            supplies_before = dict(s.get_supplies())
            journal_before_len = len(s.tracker.journal.entries)
            s.update_room_state(s.dungeon.get_level(1).rooms[0].id, "known")

            s.restore_fog_of_war()

            assert s.tracker.turn == turn_before
            assert dict(s.get_supplies()) == supplies_before
            # Allowed: a single new "Fog restored" journal entry would be
            # added by MapView.action_restore_fog, but Session.restore_fog_of_war
            # itself does not write to the journal.
            assert len(s.tracker.journal.entries) == journal_before_len
        finally:
            s.close()


class TestResetProgress:
    """Phase: Reset Dungeon Progress wipes runtime state to a fresh-
    session baseline (turn=0, fog, supplies, exhaustion, journal,
    party position, current_level) while preserving annotations and
    dungeon metadata. Lighter than full_reset (which also wipes the
    JSON's rooms/corridors)."""

    def test_resets_turn_and_journal(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH, rng_seed=1)
        try:
            s.advance_turn()
            s.advance_turn()
            s.advance_turn()
            assert s.tracker.turn == 3
            assert len(s.tracker.journal.entries) > 0
            s.reset_progress()
            assert s.tracker.turn == 0
            assert s.tracker.journal.entries == []
            assert s.tracker.light_sources == []
            assert s.tracker.noisy is False
            assert s.tracker.last_wm is None
        finally:
            s.close()

    def test_refogs_and_refills_supplies(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        from session import DEFAULT_SUPPLY_KINDS, DEFAULT_SUPPLY_COUNTS
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            # Reveal something and burn some supplies.
            l1 = s.dungeon.get_level(1)
            s.update_room_state(l1.rooms[0].id, "cleared")
            s.consume_supply("torches", 2)
            s.consume_supply("oil_flask", 1)
            assert s.get_supplies()["torches"] < DEFAULT_SUPPLY_COUNTS["torches"]

            s.reset_progress()

            # Fog restored.
            for level in s.dungeon.levels:
                for room in level.rooms:
                    assert room.state == "unexplored"
            # Supplies back to defaults for every kind.
            after = s.get_supplies()
            for kind in DEFAULT_SUPPLY_KINDS:
                assert after[kind] == DEFAULT_SUPPLY_COUNTS[kind]
        finally:
            s.close()

    def test_returns_party_to_shallowest_level(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        s = Session.create(db_path, example, EXAMPLE_PATH)
        try:
            # Ascend/descend to anywhere other than the start.
            shallowest = s.dungeon.shallowest_level_number
            if len(s.dungeon.levels) > 1:
                s.switch_level(1)  # descend
                assert s.dungeon.current_level != shallowest
            s.reset_progress()
            assert s.dungeon.current_level == shallowest
        finally:
            s.close()

    def test_persists_across_resume(
        self, db_path: Path, example: dungeon.Dungeon
    ) -> None:
        # After reset_progress, closing and resuming should still see a
        # fresh baseline (no leftover turn / journal / fog from before).
        s = Session.create(db_path, example, EXAMPLE_PATH, rng_seed=2)
        try:
            s.advance_turn()
            s.update_room_state(s.dungeon.get_level(1).rooms[0].id, "known")
            s.reset_progress()
            sid = s.session_id
        finally:
            s.close()

        s2 = Session.resume(db_path, sid)
        try:
            assert s2.tracker.turn == 0
            assert s2.tracker.journal.entries == []
            for level in s2.dungeon.levels:
                for room in level.rooms:
                    assert room.state == "unexplored"
        finally:
            s2.close()


class TestFullReset:
    def test_raises_when_dungeon_json_missing(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            Session.full_reset(empty)

    def test_writes_timestamped_backup(self, dungeon_folder: Path) -> None:
        original = (dungeon_folder / DUNGEON_JSON_NAME).read_text()
        backup = Session.full_reset(dungeon_folder)
        assert backup.exists()
        assert backup.parent == dungeon_folder
        assert backup.name.startswith(DUNGEON_JSON_NAME + ".")
        assert backup.name.endswith(".bak")
        # Backup matches the pre-reset content byte-for-byte.
        assert backup.read_text() == original

    def test_clears_rooms_and_corridors_per_level(
        self, dungeon_folder: Path
    ) -> None:
        # Sanity check the fixture has rooms before we wipe.
        before = dungeon.load(dungeon_folder / DUNGEON_JSON_NAME)
        assert any(lv.rooms for lv in before.levels)
        Session.full_reset(dungeon_folder)
        after = dungeon.load(dungeon_folder / DUNGEON_JSON_NAME)
        for lv in after.levels:
            assert lv.rooms == ()
            assert lv.corridors == ()

    def test_preserves_level_metadata_and_wm_table(
        self, dungeon_folder: Path
    ) -> None:
        before = dungeon.load(dungeon_folder / DUNGEON_JSON_NAME)
        Session.full_reset(dungeon_folder)
        after = dungeon.load(dungeon_folder / DUNGEON_JSON_NAME)
        assert after.name == before.name
        assert len(after.levels) == len(before.levels)
        for b, a in zip(before.levels, after.levels):
            assert a.level_number == b.level_number
            assert a.display_name == b.display_name
            assert a.map_image == b.map_image
            assert a.wm_check_method == b.wm_check_method
            assert a.wm_check_threshold == b.wm_check_threshold
            assert a.wandering_monster_table == b.wandering_monster_table

    def test_deletes_session_db(self, dungeon_folder: Path) -> None:
        Session.open_dungeon(dungeon_folder).close()
        assert (dungeon_folder / DUNGEON_DB_NAME).exists()
        Session.full_reset(dungeon_folder)
        assert not (dungeon_folder / DUNGEON_DB_NAME).exists()

    def test_no_session_db_is_fine(self, dungeon_folder: Path) -> None:
        # full_reset must not error if there's nothing to delete.
        assert not (dungeon_folder / DUNGEON_DB_NAME).exists()
        Session.full_reset(dungeon_folder)  # no raise
        assert not (dungeon_folder / DUNGEON_DB_NAME).exists()

    def test_open_after_full_reset_starts_empty(
        self, dungeon_folder: Path
    ) -> None:
        Session.full_reset(dungeon_folder)
        s = Session.open_dungeon(dungeon_folder)
        try:
            assert s.tracker.turn == 0
            for lv in s.dungeon.levels:
                assert lv.rooms == ()
        finally:
            s.close()


class TestListDungeons:
    def test_empty_when_root_missing(self, tmp_path: Path) -> None:
        assert Session.list_dungeons(tmp_path / "nope") == []

    def test_skips_non_directories_and_invalid_folders(
        self, tmp_path: Path, dungeon_folder: Path
    ) -> None:
        # tmp_path/test-dungeon is a valid dungeon (from fixture). Add noise.
        (tmp_path / "loose-file.txt").write_text("ignored")
        (tmp_path / "empty-folder").mkdir()  # no dungeon.json
        bad = tmp_path / "broken"
        bad.mkdir()
        (bad / DUNGEON_JSON_NAME).write_text("{not json")
        infos = Session.list_dungeons(tmp_path)
        assert len(infos) == 1
        assert infos[0].folder == dungeon_folder
        assert isinstance(infos[0], DungeonInfo)

    def test_no_session_branch(self, dungeon_folder: Path, tmp_path: Path) -> None:
        infos = Session.list_dungeons(tmp_path)
        info = infos[0]
        assert info.has_session is False
        assert info.current_turn == 0
        assert info.last_saved_at == ""
        assert info.n_levels >= 1
        assert info.name  # non-empty

    def test_with_session_reports_progress(
        self, dungeon_folder: Path, tmp_path: Path
    ) -> None:
        s = Session.open_dungeon(dungeon_folder, rng_seed=1)
        s.advance_turn()
        s.advance_turn()
        s.close()
        infos = Session.list_dungeons(tmp_path)
        info = infos[0]
        assert info.has_session is True
        assert info.current_turn == 2
        assert info.last_saved_at  # ISO timestamp string
