"""Tests for tactics — pure decision logic, no RNG required."""

from __future__ import annotations

import random

import pytest

from encounter_simulator import Action, Combatant, CombatState
import tactics


def _pc(class_: str, *, name="P", hp=20, hp_max=20, ac=15,
        attacks=None, spells=None, features=None,
        position="melee", saves=None) -> Combatant:
    return Combatant(
        id=f"pc-{name}",
        side="pc",
        name=name,
        ac=ac,
        hp=hp,
        hp_max=hp_max,
        init_bonus=1,
        speed=30,
        position=position,
        attacks=attacks or [{
            "name": "Mace", "to_hit": 5, "damage_dice": "1d6+3",
            "damage_type": "bludgeoning", "range": "melee", "save": None,
        }],
        spells=spells or {"slots": {}, "memorized": []},
        features=features or {},
        class_=class_,
        saves=saves or {},
    )


def _monster(*, name="Goblin", hp=7, ac=15, ctype="humanoid") -> Combatant:
    return Combatant(
        id=f"mn-{name}",
        side="monster",
        name=name,
        ac=ac,
        hp=hp,
        hp_max=hp,
        init_bonus=0,
        speed=30,
        position="melee",
        attacks=[{
            "name": "Scimitar", "to_hit": 4, "damage_dice": "1d6+2",
            "damage_type": "slashing", "range": "melee", "save": None,
        }],
        creature_type=ctype,
    )


def _state(combatants):
    return CombatState(combatants=combatants, rng=random.Random(0))


# ---------- Cleric -----------------------------------------------------


class TestCleric:
    def test_heals_low_ally(self):
        cleric = _pc(
            "cleric", name="Mira",
            spells={
                "slots": {1: 4},
                "memorized": [
                    {"name": "Cure Wounds", "level": 1, "type": "heal",
                     "amount": "1d8+3"},
                ],
            },
        )
        wounded = _pc("fighter", name="Thorin", hp=3, hp_max=20)
        monster = _monster()
        state = _state([cleric, wounded, monster])
        action = tactics.pick_action(cleric, state)
        assert action.kind == "heal"
        assert action.target_id == wounded.id
        assert action.spell_name == "Cure Wounds"

    def test_attacks_when_no_one_low(self):
        cleric = _pc(
            "cleric", name="Mira",
            spells={
                "slots": {1: 4},
                "memorized": [
                    {"name": "Cure Wounds", "level": 1, "type": "heal",
                     "amount": "1d8+3"},
                    {"name": "Sacred Flame", "level": 0, "type": "save_attack",
                     "save": "dex", "dc": 13, "damage": "1d8"},
                ],
            },
        )
        ally = _pc("fighter", name="Thorin", hp=20, hp_max=20)
        monster = _monster()
        state = _state([cleric, ally, monster])
        action = tactics.pick_action(cleric, state)
        # Sacred Flame preferred over weapon attack.
        assert action.kind == "spell_save"
        assert action.spell_name == "Sacred Flame"
        assert action.target_id == monster.id

    def test_falls_back_to_weapon_when_no_spell(self):
        cleric = _pc("cleric", name="Mira")
        monster = _monster()
        state = _state([cleric, monster])
        action = tactics.pick_action(cleric, state)
        assert action.kind == "attack"
        assert action.target_id == monster.id

    def test_does_not_heal_if_no_slot(self):
        cleric = _pc(
            "cleric", name="Mira",
            spells={
                "slots": {1: 0},
                "memorized": [
                    {"name": "Cure Wounds", "level": 1, "type": "heal",
                     "amount": "1d8+3"},
                ],
            },
        )
        wounded = _pc("fighter", name="Thorin", hp=3, hp_max=20)
        monster = _monster()
        state = _state([cleric, wounded, monster])
        action = tactics.pick_action(cleric, state)
        # No slot → falls back to attacking.
        assert action.kind == "attack"


# ---------- Fighter ----------------------------------------------------


class TestFighter:
    def test_attacks_normally_when_full_hp(self):
        fighter = _pc(
            "fighter", name="Thorin", hp=20, hp_max=20,
            features={"second_wind": True},
        )
        monster = _monster()
        state = _state([fighter, monster])
        action = tactics.pick_action(fighter, state)
        assert action.kind == "attack"
        assert action.target_id == monster.id

    def test_uses_second_wind_when_bloodied(self):
        fighter = _pc(
            "fighter", name="Thorin", hp=8, hp_max=20,
            features={"second_wind": True},
        )
        monster = _monster()
        state = _state([fighter, monster])
        action = tactics.pick_action(fighter, state)
        assert action.kind == "second_wind"

    def test_does_not_repeat_second_wind(self):
        fighter = _pc(
            "fighter", name="Thorin", hp=8, hp_max=20,
            features={"second_wind": True},
        )
        fighter.used_second_wind = True
        monster = _monster()
        state = _state([fighter, monster])
        action = tactics.pick_action(fighter, state)
        assert action.kind == "attack"


# ---------- Rogue ------------------------------------------------------


class TestRogue:
    def test_sneak_attack_when_ally_engaged(self):
        rogue = _pc(
            "rogue", name="Sera",
            features={"sneak_attack_dice": 2},
        )
        flanker = _pc("fighter", name="Thorin", position="melee")
        monster = _monster()
        state = _state([rogue, flanker, monster])
        action = tactics.pick_action(rogue, state)
        assert action.kind == "attack"
        assert action.spell_name == "__sneak_attack__"

    def test_no_sneak_attack_when_alone(self):
        rogue = _pc(
            "rogue", name="Sera",
            features={"sneak_attack_dice": 2},
        )
        monster = _monster()
        state = _state([rogue, monster])
        action = tactics.pick_action(rogue, state)
        assert action.kind == "attack"
        # No flanker → no sneak attack rider.
        assert action.spell_name != "__sneak_attack__"


# ---------- Wizard / sorcerer -----------------------------------------


class TestCaster:
    def test_aoe_when_two_plus_targets(self):
        caster = _pc(
            "wizard", name="Aldric", position="ranged",
            spells={
                "slots": {1: 2},
                "memorized": [
                    {"name": "Burning Hands", "level": 1, "type": "save_attack",
                     "save": "dex", "dc": 13, "damage": "3d6",
                     "half_on_save": True, "aoe": True},
                    {"name": "Fire Bolt", "level": 0, "type": "save_attack",
                     "save": "dex", "dc": 13, "damage": "1d10"},
                ],
            },
        )
        m1, m2 = _monster(name="G1"), _monster(name="G2")
        state = _state([caster, m1, m2])
        action = tactics.pick_action(caster, state)
        assert action.kind == "spell_save"
        assert action.spell_name == "Burning Hands"
        assert set(action.aoe_target_ids) == {m1.id, m2.id}

    def test_single_cantrip_when_one_target(self):
        caster = _pc(
            "wizard", name="Aldric", position="ranged",
            spells={
                "slots": {1: 0},
                "memorized": [
                    {"name": "Fire Bolt", "level": 0, "type": "save_attack",
                     "save": "dex", "dc": 13, "damage": "1d10"},
                ],
            },
        )
        monster = _monster()
        state = _state([caster, monster])
        action = tactics.pick_action(caster, state)
        assert action.kind == "spell_save"
        assert action.spell_name == "Fire Bolt"


# ---------- Paladin ----------------------------------------------------


class TestPaladin:
    def test_smite_vs_undead(self):
        paladin = _pc(
            "paladin", name="Aldric",
            features={"divine_smite_slots_usable": True},
            spells={"slots": {1: 2}, "memorized": []},
        )
        undead = _monster(name="Skeleton", ctype="undead")
        state = _state([paladin, undead])
        action = tactics.pick_action(paladin, state)
        assert action.kind == "attack"
        assert action.spell_name == "__divine_smite__"
        assert action.spell_level == 1

    def test_no_smite_vs_humanoid(self):
        paladin = _pc(
            "paladin", name="Aldric",
            features={"divine_smite_slots_usable": True},
            spells={"slots": {1: 2}, "memorized": []},
        )
        goblin = _monster(name="Goblin", ctype="humanoid")
        state = _state([paladin, goblin])
        action = tactics.pick_action(paladin, state)
        assert action.kind == "attack"
        assert action.spell_name != "__divine_smite__"


# ---------- Default & monster -----------------------------------------


class TestDefaults:
    def test_unknown_class_attacks(self):
        unknown = _pc("druid", name="Mira")
        monster = _monster()
        state = _state([unknown, monster])
        action = tactics.pick_action(unknown, state)
        assert action.kind == "attack"

    def test_no_targets_passes(self):
        lone = _pc("fighter", name="Thorin")
        state = _state([lone])
        action = tactics.pick_action(lone, state)
        assert action.kind == "pass"

    def test_monster_attacks_pc(self):
        monster = _monster()
        pc = _pc("fighter", name="Thorin")
        state = _state([monster, pc])
        action = tactics.pick_action(monster, state)
        assert action.kind == "attack"
        assert action.target_id == pc.id
