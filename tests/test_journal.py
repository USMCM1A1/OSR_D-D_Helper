"""Tests for the Journal event log and entry formatter."""

from __future__ import annotations

import journal as jmod
from journal import Journal, JournalEntry, format_entry


class TestJournalRecord:
    def test_record_appends_entry(self) -> None:
        j = Journal()
        e = j.record(turn=1, kind=jmod.KIND_NOTE, message="hello")
        assert isinstance(e, JournalEntry)
        assert len(j) == 1
        assert j.entries[0] is e

    def test_iteration_preserves_order(self) -> None:
        j = Journal()
        for n in range(5):
            j.record(turn=n, kind=jmod.KIND_NOTE, message=f"#{n}")
        assert [e.turn for e in j] == [0, 1, 2, 3, 4]

    def test_since_slices_correctly(self) -> None:
        j = Journal()
        j.record(1, jmod.KIND_NOTE, "a")
        snapshot = len(j)
        j.record(2, jmod.KIND_NOTE, "b")
        j.record(3, jmod.KIND_NOTE, "c")
        new = j.since(snapshot)
        assert [e.message for e in new] == ["b", "c"]

    def test_of_kind_filters(self) -> None:
        j = Journal()
        j.record(1, jmod.KIND_NOTE, "n1")
        j.record(2, jmod.KIND_WM_CHECK, "wm1")
        j.record(3, jmod.KIND_NOTE, "n2")
        notes = j.of_kind(jmod.KIND_NOTE)
        assert [e.message for e in notes] == ["n1", "n2"]


class TestFormatEntry:
    def test_format_includes_turn_and_message(self) -> None:
        e = JournalEntry(turn=7, kind=jmod.KIND_WM_CHECK,
                         message="WM Check: rolled 19 — ENCOUNTER → Ghoul")
        s = format_entry(e)
        assert "Turn   7" in s
        assert "WM Check: rolled 19" in s

    def test_format_elapsed_time(self) -> None:
        # Turn 7 × 10 min = 70 min = 1h 10m.
        e = JournalEntry(turn=7, kind=jmod.KIND_TURN_ADVANCE, message="Elapsed: 1h10m")
        assert "(1h10m)" in format_entry(e)

    def test_format_zero_padded_minutes(self) -> None:
        # Turn 6 × 10 min = 60 min = 1h 00m.
        e = JournalEntry(turn=6, kind=jmod.KIND_TURN_ADVANCE, message="Elapsed: 1h00m")
        assert "(1h00m)" in format_entry(e)
