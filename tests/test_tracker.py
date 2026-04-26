"""Phase 2 test gate — turn engine behavior, WM logic, rest helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

import config
import dungeon
import journal as jmod
from tracker import LightSource, Tracker, WMResult, wm_triggered


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def base_dict() -> dict:
    """Reusable multi-level dungeon dict that tests parametrize."""
    return {
        "dungeon_name": "T",
        "party_level": 3,
        "current_level": 1,
        "party": {"size": 1, "characters": [{"name": "Solo"}]},
        "levels": [
            {
                "level_number": 1,
                "display_name": "Level 1 — T",
                "map_image": "target_dungeon_maps/t1.png",
                "map_image_scale": 1.0,
                "wm_check_method": "d20",
                "wm_check_threshold": 18,
                "wm_check_frequency": "every_turn",
                "wandering_monster_table": [
                    {"roll": 1, "encounter": "Skeletons"},
                    {"roll": 2, "encounter": "Zombies"},
                    {"roll": 3, "encounter": "Spider"},
                    {"roll": 4, "encounter": "Cube"},
                    {"roll": 5, "encounter": "Shadows"},
                    {"roll": 6, "encounter": "Ghoul"},
                ],
                "rooms": [
                    {"id": "A", "name": "A", "state": "unexplored", "tags": ["empty"]},
                    {"id": "B", "name": "B", "state": "unexplored", "tags": ["empty"]},
                ],
                "corridors": [{"from": "A", "to": "B", "distance_ft": 10, "tags": []}],
            },
        ],
    }


@pytest.fixture
def d20_dungeon(base_dict: dict) -> dungeon.Dungeon:
    return dungeon._from_dict(base_dict, source="<test>")


@pytest.fixture
def d6_dungeon(base_dict: dict) -> dungeon.Dungeon:
    base_dict["levels"][0]["wm_check_method"] = "d6"
    base_dict["levels"][0]["wm_check_threshold"] = 1
    return dungeon._from_dict(base_dict, source="<test>")


def _tracker(d: dungeon.Dungeon, seed: int = 42) -> Tracker:
    return Tracker(d, rng=random.Random(seed))


# --- WM threshold logic ------------------------------------------------------


class TestWMTriggered:
    @pytest.mark.parametrize("roll,expected", [
        (1, False), (10, False), (17, False),
        (18, True), (19, True), (20, True),
    ])
    def test_d20_threshold_18(self, roll: int, expected: bool) -> None:
        assert wm_triggered("d20", roll, 18) is expected

    @pytest.mark.parametrize("roll,expected", [
        (1, True), (2, False), (3, False), (6, False),
    ])
    def test_d6_threshold_1(self, roll: int, expected: bool) -> None:
        assert wm_triggered("d6", roll, 1) is expected

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown WM method"):
            wm_triggered("d100", 50, 90)


# --- Turn advance + light timers --------------------------------------------


class TestAdvanceTurn:
    def test_first_advance_lands_on_turn_1(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        assert t.turn == 0
        t.advance_turn()
        assert t.turn == 1

    def test_elapsed_minutes_tracks_turn(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        for _ in range(7):
            t.advance_turn()
        assert t.elapsed_minutes == 70
        assert t.elapsed_hm == (1, 10)

    def test_advance_returns_only_this_turns_entries(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        first = t.advance_turn()
        second = t.advance_turn()
        # Each turn produces a turn_advance + a wm_check (no lights here).
        assert {e.kind for e in first} == {jmod.KIND_TURN_ADVANCE, jmod.KIND_WM_CHECK}
        assert all(e.turn == 1 for e in first)
        assert all(e.turn == 2 for e in second)


class TestLightSources:
    def test_torch_expires_after_six_turns(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        torch = t.add_light_source("torch")
        assert torch.turns_remaining == 6
        for _ in range(6):
            t.advance_turn()
        # After 6 ticks: removed from active list.
        assert t.light_sources == []
        # Last entry of kind=light_out should reference the torch.
        outs = t.journal.of_kind(jmod.KIND_LIGHT_OUT)
        assert len(outs) == 1
        assert "Torch #1" in outs[0].message

    def test_torch_warning_emitted_at_two_and_one_remaining(self, d20_dungeon: dungeon.Dungeon) -> None:
        # Torch starts at 6. Each advance_turn ticks it down: turn 1 → 5,
        # turn 2 → 4, ... turn 4 → 2, turn 5 → 1, turn 6 → 0 (extinguished).
        # Warning fires when remaining ≤ 2 *and* not extinguished: turns 4 & 5.
        t = _tracker(d20_dungeon)
        t.add_light_source("torch")
        for _ in range(6):
            t.advance_turn()
        warnings = t.journal.of_kind(jmod.KIND_LIGHT_WARNING)
        assert [w.turn for w in warnings] == [4, 5]

    def test_lantern_lasts_36_turns(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        ls = t.add_light_source("hooded_lantern")
        assert ls.turns_remaining == 36

    def test_unknown_light_kind_raises(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        with pytest.raises(ValueError, match="unknown light source kind"):
            t.add_light_source("magic_glowstick")

    def test_labels_are_numbered(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        a = t.add_light_source("torch")
        b = t.add_light_source("torch")
        assert a.label == "Torch #1"
        assert b.label == "Torch #2"


# --- WM rolls ----------------------------------------------------------------


class TestWMRoll:
    def test_seeded_d20_sequence_is_deterministic(self, d20_dungeon: dungeon.Dungeon) -> None:
        t1 = _tracker(d20_dungeon, seed=42)
        t2 = _tracker(d20_dungeon, seed=42)
        rolls1 = [t1.roll_wm().roll for _ in range(10)]
        rolls2 = [t2.roll_wm().roll for _ in range(10)]
        assert rolls1 == rolls2

    def test_d20_roll_in_range(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        for _ in range(50):
            r = t.roll_wm()
            assert 1 <= r.roll <= 20
            assert r.method == "d20"

    def test_d6_roll_in_range(self, d6_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d6_dungeon)
        for _ in range(50):
            r = t.roll_wm()
            assert 1 <= r.roll <= 6
            assert r.method == "d6"

    def test_triggered_yields_encounter(self, d20_dungeon: dungeon.Dungeon) -> None:
        # Force-trigger by lowering the *current level's* threshold to 1.
        d20_dungeon.current.wm_check_threshold = 1
        t = _tracker(d20_dungeon, seed=1)
        r = t.roll_wm()
        assert r.triggered is True
        assert r.encounter is not None
        assert r.encounter in {e.encounter for e in d20_dungeon.current.wandering_monster_table}


# --- Rest helpers ------------------------------------------------------------


class TestShortRest:
    def test_short_rest_advances_six_turns(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        t.short_rest()
        assert t.turn == config.SHORT_REST_TURNS == 6

    def test_short_rest_rolls_six_wm_checks(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        t.short_rest()
        wm = t.journal.of_kind(jmod.KIND_WM_CHECK)
        assert len(wm) == 6

    def test_short_rest_emits_start_and_end(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        t.short_rest()
        kinds = [e.kind for e in t.journal]
        assert kinds[0] == jmod.KIND_SHORT_REST_START
        assert kinds[-1] == jmod.KIND_SHORT_REST_END


class TestLongRest:
    def test_long_rest_advances_48_turns(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        t.long_rest()
        assert t.turn == config.LONG_REST_TURNS == 48

    def test_long_rest_rolls_48_wm_checks(self, d20_dungeon: dungeon.Dungeon) -> None:
        t = _tracker(d20_dungeon)
        t.long_rest()
        wm = t.journal.of_kind(jmod.KIND_WM_CHECK)
        assert len(wm) == 48


# --- Golden 50-turn run ------------------------------------------------------


GOLDEN_PATH = Path(__file__).parent / "golden" / "turn_engine_50turns.txt"
EXAMPLE_PATH = Path(__file__).parent.parent / "data" / "example_dungeon.json"


def _run_50turn_simulation() -> str:
    """Reproducible 50-turn run with seed=42, 2 torches, against the example."""
    d = dungeon.load(EXAMPLE_PATH)
    t = Tracker(d, rng=random.Random(42))
    t.add_light_source("torch")
    t.add_light_source("torch")
    for _ in range(50):
        t.advance_turn()
    from journal import format_entry
    return "\n".join(format_entry(e) for e in t.journal) + "\n"


class TestGolden:
    def test_50_turn_run_matches_golden(self) -> None:
        actual = _run_50turn_simulation()
        if os.environ.get("REGENERATE_GOLDENS") == "1":
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(actual)
            pytest.skip("Regenerated golden; rerun without REGENERATE_GOLDENS to verify.")
        assert GOLDEN_PATH.exists(), (
            f"Golden file missing at {GOLDEN_PATH}. "
            f"Run with REGENERATE_GOLDENS=1 to create it."
        )
        expected = GOLDEN_PATH.read_text()
        assert actual == expected, (
            "50-turn journal output diverged from golden. "
            "If this is intentional, rerun with REGENERATE_GOLDENS=1."
        )
