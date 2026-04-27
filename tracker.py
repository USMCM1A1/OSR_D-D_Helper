"""Turn engine: turn counter, light source timers, wandering monster rolls.

The Tracker is the headless core of the app. It owns no UI, does not
render, and is fully deterministic given a seeded RNG — Phase 2 tests
exercise it via a golden journal file.

Public API:
    Tracker(dungeon, *, rng=None, journal=None)
    Tracker.advance_turn() -> list[JournalEntry]
    Tracker.short_rest()   -> list[JournalEntry]   # advances 6 turns
    Tracker.long_rest()    -> list[JournalEntry]   # advances 48 turns
    Tracker.add_light_source(kind, label=None) -> LightSource
    Tracker.roll_wm() -> WMResult
    wm_triggered(method, roll, threshold) -> bool
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import config
import journal as journal_mod
from dungeon import Dungeon
from journal import Journal, JournalEntry


@dataclass
class LightSource:
    """An active light source consuming turns. Mutable: tick() decrements."""

    kind: str            # key into config.LIGHT_DURATIONS_TURNS
    label: str           # display name e.g. "Torch #1"
    turns_remaining: int

    def tick(self) -> bool:
        """Decrement remaining turns; return True if extinguished this tick."""
        self.turns_remaining -= 1
        return self.turns_remaining <= 0


@dataclass(frozen=True)
class WMResult:
    method: str          # "d20" or "d6"
    roll: int
    triggered: bool
    encounter: str | None = None  # populated when triggered


def wm_triggered(method: str, roll: int, threshold: int) -> bool:
    """True when a wandering monster check should resolve to an encounter.

    DMG p. 82 wording is asymmetric: d20 triggers on *high* rolls (≥ 18),
    d6 triggers on *low* rolls (a 1). We honor both directions explicitly.
    """
    if method == config.WM_METHOD_D20:
        return roll >= threshold
    if method == config.WM_METHOD_D6:
        return roll <= threshold
    raise ValueError(f"unknown WM method: {method!r}")


class Tracker:
    """Headless turn engine. One instance per active session."""

    def __init__(
        self,
        dungeon: Dungeon,
        *,
        rng: random.Random | None = None,
        journal: Journal | None = None,
    ) -> None:
        self.dungeon = dungeon
        self.rng = rng if rng is not None else random.Random()
        self.journal = journal if journal is not None else Journal()
        self.turn = 0
        self.light_sources: list[LightSource] = []
        # Phase-8d Noisy flag — set after combat / failed stealth / etc.
        # The threshold inflation behaviour is deferred (planned in CLAUDE.md
        # Phase 8d); for now this is a journaled flag the renderer can show.
        self.noisy: bool = False
        # Most recent WM roll result — cached for the status-strip UI; the
        # journal remains the source of truth across save/resume.
        self.last_wm: WMResult | None = None

    # --- Properties ----------------------------------------------------------

    @property
    def elapsed_minutes(self) -> int:
        return self.turn * config.TURN_MINUTES

    @property
    def elapsed_hm(self) -> tuple[int, int]:
        return divmod(self.elapsed_minutes, 60)

    # --- Light sources -------------------------------------------------------

    def add_light_source(self, kind: str, label: str | None = None) -> LightSource:
        if kind not in config.LIGHT_DURATIONS_TURNS:
            raise ValueError(
                f"unknown light source kind: {kind!r}; "
                f"known: {sorted(config.LIGHT_DURATIONS_TURNS)}"
            )
        if label is None:
            existing_of_kind = sum(1 for ls in self.light_sources if ls.kind == kind)
            label = f"{kind.replace('_', ' ').title()} #{existing_of_kind + 1}"
        ls = LightSource(
            kind=kind,
            label=label,
            turns_remaining=config.LIGHT_DURATIONS_TURNS[kind],
        )
        self.light_sources.append(ls)
        return ls

    # --- Turn advance --------------------------------------------------------

    def advance_turn(self) -> list[JournalEntry]:
        """Advance one turn: increment counter, tick lights, maybe roll WM.

        WM rolls only fire on turns that are a multiple of the current
        level's wm_check_every_n_turns (default 1 = every turn).
        """
        start = len(self.journal)
        self.turn += 1
        h, m = self.elapsed_hm
        self.journal.record(
            self.turn, journal_mod.KIND_TURN_ADVANCE,
            f"Elapsed: {h}h{m:02d}m",
        )
        self._tick_light_sources()
        every_n = max(1, self.dungeon.current.wm_check_every_n_turns)
        if self.turn % every_n == 0:
            self.roll_wm()
        return self.journal.since(start)

    # -- Noisy flag ----------------------------------------------------------

    def set_noisy(self, noisy: bool) -> None:
        """Toggle the Noisy state and journal the change. The threshold
        inflation behaviour is deferred (CLAUDE.md Phase 8d) — at the moment
        this is purely a log + UI signal."""
        if self.noisy == noisy:
            return
        self.noisy = noisy
        msg = "Party becomes Noisy." if noisy else "Party Noisy flag cleared."
        self.journal.record(self.turn, journal_mod.KIND_NOTE, msg)

    def _tick_light_sources(self) -> None:
        expired: list[LightSource] = []
        for ls in self.light_sources:
            extinguished = ls.tick()
            if extinguished:
                self.journal.record(
                    self.turn, journal_mod.KIND_LIGHT_OUT,
                    f"{ls.label}: EXTINGUISHED",
                )
                expired.append(ls)
            elif ls.turns_remaining <= config.LIGHT_LOW_WARNING_TURNS:
                self.journal.record(
                    self.turn, journal_mod.KIND_LIGHT_WARNING,
                    f"{ls.label}: {ls.turns_remaining} turns remaining ⚠",
                )
        for ls in expired:
            self.light_sources.remove(ls)

    # --- Wandering monster ---------------------------------------------------

    def roll_wm(self) -> WMResult:
        # Read the WM rules from the *current level* dynamically — switching
        # levels picks up the new method / threshold / table on the next roll.
        level = self.dungeon.current
        method = level.wm_check_method
        threshold = level.wm_check_threshold
        die_max = 20 if method == config.WM_METHOD_D20 else 6
        roll = self.rng.randint(1, die_max)
        triggered = wm_triggered(method, roll, threshold)
        encounter: str | None = None
        if triggered:
            encounter = self._roll_wm_table()
            self.journal.record(
                self.turn, journal_mod.KIND_WM_CHECK,
                f"WM Check: rolled {roll} — ENCOUNTER → {encounter}",
            )
        else:
            self.journal.record(
                self.turn, journal_mod.KIND_WM_CHECK,
                f"WM Check: rolled {roll} — No encounter",
            )
        result = WMResult(method=method, roll=roll, triggered=triggered, encounter=encounter)
        self.last_wm = result
        return result

    def _roll_wm_table(self) -> str:
        table = self.dungeon.current.wandering_monster_table
        n = max(e.roll for e in table)
        roll = self.rng.randint(1, n)
        for e in table:
            if e.roll == roll:
                return e.encounter
        # Sparse table — fall back to the closest entry by roll value.
        return min(table, key=lambda e: abs(e.roll - roll)).encounter

    # --- Rests ---------------------------------------------------------------

    def short_rest(self) -> list[JournalEntry]:
        return self._rest(
            config.SHORT_REST_TURNS,
            journal_mod.KIND_SHORT_REST_START,
            journal_mod.KIND_SHORT_REST_END,
            label="Short Rest",
        )

    def long_rest(self) -> list[JournalEntry]:
        return self._rest(
            config.LONG_REST_TURNS,
            journal_mod.KIND_LONG_REST_START,
            journal_mod.KIND_LONG_REST_END,
            label="Long Rest",
        )

    def _rest(self, turns: int, kind_start: str, kind_end: str, *, label: str) -> list[JournalEntry]:
        start = len(self.journal)
        self.journal.record(self.turn, kind_start, f"{label} begins.")
        for _ in range(turns):
            self.advance_turn()
        self.journal.record(self.turn, kind_end, f"{label} ends.")
        return self.journal.since(start)
