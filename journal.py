"""In-memory session journal — append-only log of turn events.

Phase 2 deliverable. SQLite persistence comes in Phase 3, which will
serialize Journal entries to the `turn_log` table. Until then, entries
live only in process memory.

Format goal (per CLAUDE.md "Session Journal"):
    Turn  7 (1h10m) — WM Check: rolled 19 — ENCOUNTER → Ghoul
    Turn 14 (2h20m) — Torch #1: 2 turns remaining ⚠
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import config


# Recognized entry kinds. Extend as new event types are introduced; the
# renderer and journal export rely on these to choose styling/icons.
KIND_TURN_ADVANCE = "turn_advance"
KIND_WM_CHECK = "wm_check"
KIND_LIGHT_WARNING = "light_warning"
KIND_LIGHT_OUT = "light_out"
KIND_SHORT_REST_START = "short_rest_start"
KIND_SHORT_REST_END = "short_rest_end"
KIND_LONG_REST_START = "long_rest_start"
KIND_LONG_REST_END = "long_rest_end"
KIND_LEVEL_TRANSITION = "level_transition"
KIND_NOTE = "note"  # manual DM note


@dataclass(frozen=True)
class JournalEntry:
    turn: int
    kind: str
    message: str


class Journal:
    """Append-only event log. Stable iteration order = insertion order."""

    def __init__(self) -> None:
        self.entries: list[JournalEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[JournalEntry]:
        return iter(self.entries)

    def record(self, turn: int, kind: str, message: str) -> JournalEntry:
        entry = JournalEntry(turn=turn, kind=kind, message=message)
        self.entries.append(entry)
        return entry

    def since(self, index: int) -> list[JournalEntry]:
        """Return entries appended at or after `index` (a snapshot offset)."""
        return self.entries[index:]

    def of_kind(self, kind: str) -> list[JournalEntry]:
        return [e for e in self.entries if e.kind == kind]

    def format_lines(self) -> list[str]:
        return [format_entry(e) for e in self.entries]


def format_entry(entry: JournalEntry) -> str:
    """Render a single journal entry in CLAUDE.md style."""
    elapsed_min = entry.turn * config.TURN_MINUTES
    h, m = divmod(elapsed_min, 60)
    return f"Turn {entry.turn:>3} ({h}h{m:02d}m) — {entry.message}"
