"""Tests for level_scaler.

Covers each class type: full caster, half caster, warlock,
extra-attack classes, rogue sneak attack, and the no-op case.
"""

from __future__ import annotations

import pytest

import level_scaler


def _base(**overrides) -> dict:
    """Minimal L1 character; tests override fields they care about."""
    return {
        "name": "Test",
        "class": "fighter",
        "level": 1,
        "ac": 14,
        "hp_max": 10,
        "spells": {"slots": {}, "memorized": []},
        "features": {},
        **overrides,
    }


class TestNoOp:
    def test_same_level_returns_copy_unchanged(self):
        c = _base(level=3, hp_max=24)
        out = level_scaler.scale_character(c, 3)
        assert out == c
        # Defensive copy — caller mutations don't leak.
        out["hp_max"] = 99
        assert c["hp_max"] == 24

    def test_zero_level_treated_as_one(self):
        c = _base()
        out = level_scaler.scale_character(c, 0)
        assert out["level"] == 1


class TestHP:
    def test_linear_scale_doubles_at_2x_level(self):
        c = _base(hp_max=10, level=1)
        out = level_scaler.scale_character(c, 2)
        assert out["hp_max"] == 20

    def test_scale_rounded_not_truncated(self):
        c = _base(hp_max=10, level=3)
        # 10 * 5/3 = 16.66 → 17
        out = level_scaler.scale_character(c, 5)
        assert out["hp_max"] == 17

    def test_hp_never_below_one(self):
        c = _base(hp_max=1, level=20)
        out = level_scaler.scale_character(c, 1)
        assert out["hp_max"] >= 1


class TestRogue:
    def test_sneak_attack_dice_progression(self):
        for lvl, expected in [(1, 1), (2, 1), (3, 2), (5, 3), (7, 4), (10, 5)]:
            c = _base(**{"class": "rogue", "features": {"sneak_attack_dice": 1}})
            out = level_scaler.scale_character(c, lvl)
            assert out["features"]["sneak_attack_dice"] == expected, \
                f"L{lvl}: expected {expected}, got {out['features']['sneak_attack_dice']}"


class TestExtraAttack:
    @pytest.mark.parametrize("cls", ["fighter", "paladin", "ranger", "barbarian"])
    def test_extra_attack_off_below_5(self, cls):
        c = _base(**{"class": cls})
        out = level_scaler.scale_character(c, 4)
        assert out["features"].get("extra_attack") is False

    @pytest.mark.parametrize("cls", ["fighter", "paladin", "ranger", "barbarian"])
    def test_extra_attack_on_at_5(self, cls):
        c = _base(**{"class": cls})
        out = level_scaler.scale_character(c, 5)
        assert out["features"].get("extra_attack") is True

    def test_non_warrior_class_has_no_extra_attack_flag(self):
        c = _base(**{"class": "wizard"})
        out = level_scaler.scale_character(c, 5)
        assert "extra_attack" not in out["features"]


class TestFullCasterSlots:
    @pytest.mark.parametrize("cls", ["cleric", "druid", "wizard", "sorcerer", "bard"])
    def test_l3_has_l1_and_l2_slots(self, cls):
        c = _base(**{"class": cls, "spells": {"slots": {1: 2}, "memorized": []}})
        out = level_scaler.scale_character(c, 3)
        assert out["spells"]["slots"] == {1: 4, 2: 2}

    def test_l5_caster_gets_l3_slot(self):
        c = _base(**{"class": "wizard"})
        out = level_scaler.scale_character(c, 5)
        assert out["spells"]["slots"][3] == 2


class TestHalfCasterSlots:
    def test_paladin_l1_has_no_slots(self):
        c = _base(**{"class": "paladin"})
        out = level_scaler.scale_character(c, 1)
        assert out["spells"]["slots"] == {}

    def test_paladin_l2_gets_l1_slots(self):
        c = _base(**{"class": "paladin"})
        out = level_scaler.scale_character(c, 2)
        assert out["spells"]["slots"] == {1: 2}

    def test_paladin_l5_gets_l2_slots(self):
        c = _base(**{"class": "paladin"})
        out = level_scaler.scale_character(c, 5)
        assert out["spells"]["slots"][2] == 2


class TestWarlock:
    def test_warlock_l3_gets_l2_slots(self):
        c = _base(**{"class": "warlock"})
        out = level_scaler.scale_character(c, 3)
        # Warlock skips L1 slots entirely at L3 (pact magic upgrades).
        assert 1 not in out["spells"]["slots"]
        assert out["spells"]["slots"][2] == 2


class TestNoCaster:
    def test_fighter_slots_left_alone(self):
        c = _base(**{"class": "fighter", "spells": {"slots": {}, "memorized": []}})
        out = level_scaler.scale_character(c, 5)
        assert out["spells"]["slots"] == {}


class TestScaleParty:
    def test_scales_all(self):
        party = [
            _base(name="A", level=1, hp_max=10),
            _base(name="B", **{"class": "rogue", "level": 1, "hp_max": 8}),
        ]
        out = level_scaler.scale_party(party, 3)
        assert all(c["level"] == 3 for c in out)
        # Linear scaling: A 10→30, B 8→24.
        assert out[0]["hp_max"] == 30
        assert out[1]["hp_max"] == 24
        # Rogue's sneak attack scaled.
        assert out[1]["features"]["sneak_attack_dice"] == 2


class TestSimulatorIntegration:
    """An L3-scaled L1 party should be visibly tougher in combat."""

    def test_scaled_party_wins_more(self):
        # 2 Ghouls vs a tiny L1 party should be brutal; same fight at
        # scaled L4 should be much more winnable.
        import encounter_simulator
        import srd_lookup
        import statblock_parser

        party_l1 = [
            {
                "name": "Cleric", "class": "cleric", "level": 1,
                "ac": 16, "hp_max": 10, "init_bonus": 0,
                "attacks": [{"name": "Mace", "to_hit": 4, "damage": "1d6+2",
                             "damage_type": "bludgeoning", "range": "melee"}],
                "features": {},
                "spells": {"slots": {1: 2}, "memorized": [
                    {"name": "Cure Wounds", "level": 1, "type": "heal",
                     "amount": "1d8+3"},
                ]},
            },
            {
                "name": "Rogue", "class": "rogue", "level": 1,
                "ac": 14, "hp_max": 11, "init_bonus": 3,
                "attacks": [{"name": "Shortsword", "to_hit": 5,
                             "damage": "1d6+3", "damage_type": "piercing",
                             "range": "melee"}],
                "features": {"sneak_attack_dice": 1},
                "spells": {"slots": {}, "memorized": []},
            },
        ]
        party_l4 = level_scaler.scale_party(party_l1, 4)

        ghoul = statblock_parser.parse(srd_lookup.find("Ghoul"))

        r1 = encounter_simulator.monte_carlo(
            party_l1, [ghoul, ghoul], trials=40, base_seed=0,
        )
        r4 = encounter_simulator.monte_carlo(
            party_l4, [ghoul, ghoul], trials=40, base_seed=0,
        )
        # The L4 party should have a noticeably better win rate.
        assert r4.party_win_pct > r1.party_win_pct
