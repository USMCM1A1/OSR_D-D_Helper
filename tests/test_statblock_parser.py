"""Tests for statblock_parser.

The SRD index is a real dependency (loaded from srd-resources/) — these
tests assert the parser pulls correct numbers off three well-known
creatures whose statblocks are stable: Ghoul (paralysis save rider),
Goblin (melee + ranged), Skeleton (multiple weapons, no rider).
"""

from __future__ import annotations

import random

import pytest

import srd_lookup
import statblock_parser as sp


# ---------- Dice helpers -------------------------------------------------


class TestRollDice:
    def test_static_int(self):
        assert sp.roll_dice("5", random.Random(0)) == 5

    def test_zero(self):
        assert sp.roll_dice("0", random.Random(0)) == 0

    def test_empty(self):
        assert sp.roll_dice("", random.Random(0)) == 0

    def test_simple_d6_in_range(self):
        rng = random.Random(0)
        for _ in range(20):
            v = sp.roll_dice("1d6", rng)
            assert 1 <= v <= 6

    def test_2d6_plus_2_in_range(self):
        rng = random.Random(0)
        for _ in range(20):
            v = sp.roll_dice("2d6+2", rng)
            assert 4 <= v <= 14

    def test_with_spaces(self):
        rng = random.Random(0)
        for _ in range(20):
            v = sp.roll_dice("2d6 + 2", rng)
            assert 4 <= v <= 14

    def test_minus_modifier(self):
        rng = random.Random(0)
        for _ in range(20):
            v = sp.roll_dice("1d4-1", rng)
            assert 0 <= v <= 3

    def test_negative_clamps_to_zero(self):
        # 1d4-10 always lands negative pre-clamp.
        for _ in range(10):
            assert sp.roll_dice("1d4-10", random.Random(_)) == 0

    def test_garbage_returns_zero(self):
        assert sp.roll_dice("not a roll", random.Random(0)) == 0

    def test_deterministic_with_same_seed(self):
        a = [sp.roll_dice("3d8+4", random.Random(42)) for _ in range(5)]
        b = [sp.roll_dice("3d8+4", random.Random(42)) for _ in range(5)]
        assert a == b


class TestRollCount:
    def test_plain_integer_prefix(self):
        rng = random.Random(0)
        # "3 Skeletons" returns 3.
        assert sp.roll_count("3 Skeletons", rng) == 3

    def test_dice_prefix(self):
        rng = random.Random(0)
        for _ in range(20):
            n = sp.roll_count("2d6 Skeletons", rng)
            assert 2 <= n <= 12

    def test_parenthesised_dice(self):
        rng = random.Random(0)
        for _ in range(20):
            n = sp.roll_count("Shadows (1d3)", rng)
            assert 1 <= n <= 3

    def test_no_count_defaults_to_one(self):
        assert sp.roll_count("Ghoul", random.Random(0)) == 1

    def test_empty_string(self):
        assert sp.roll_count("", random.Random(0)) == 1

    def test_at_least_one(self):
        # Even a 0d-style or negative shouldn't ever drop below 1.
        assert sp.roll_count("0 Goblins", random.Random(0)) >= 1


# ---------- ParsedMonster from SRD ---------------------------------------


@pytest.fixture
def ghoul() -> sp.ParsedMonster:
    sb = srd_lookup.find("Ghoul")
    assert sb is not None
    return sp.parse(sb)


@pytest.fixture
def goblin() -> sp.ParsedMonster:
    sb = srd_lookup.find("Goblin")
    assert sb is not None
    return sp.parse(sb)


@pytest.fixture
def skeleton() -> sp.ParsedMonster:
    sb = srd_lookup.find("Skeleton")
    assert sb is not None
    return sp.parse(sb)


class TestGhoul:
    def test_top_level_fields(self, ghoul):
        assert ghoul.name == "Ghoul"
        assert ghoul.ac == 12
        assert ghoul.hp_avg == 22
        assert ghoul.hp_dice == "5d8"
        assert ghoul.speed == 30
        assert ghoul.cr == "1"
        assert ghoul.creature_type == "undead"

    def test_two_attacks(self, ghoul):
        names = sorted(a.name for a in ghoul.attacks)
        assert names == ["Bite", "Claws"]

    def test_bite(self, ghoul):
        bite = next(a for a in ghoul.attacks if a.name == "Bite")
        assert bite.to_hit == 2
        assert bite.damage_dice == "2d6+2"
        assert bite.damage_type == "piercing"
        assert bite.range == "melee"
        assert bite.save is None

    def test_claws_paralysis_save(self, ghoul):
        claws = next(a for a in ghoul.attacks if a.name == "Claws")
        assert claws.to_hit == 4
        assert claws.damage_dice == "2d4+2"
        assert claws.damage_type == "slashing"
        assert claws.save is not None
        assert claws.save["ability"] == "con"
        assert claws.save["dc"] == 10
        # Paralysis isn't half-on-save.
        assert claws.save["half_on_save"] is False
        assert "saving throw" in claws.rider_text.lower()


class TestGoblin:
    def test_ac_15(self, goblin):
        assert goblin.ac == 15

    def test_hp_dice(self, goblin):
        assert goblin.hp_avg == 7
        assert goblin.hp_dice == "2d6"

    def test_humanoid_type(self, goblin):
        assert goblin.creature_type == "humanoid"

    def test_cr_quarter(self, goblin):
        assert goblin.cr == "1/4"

    def test_melee_and_ranged(self, goblin):
        ranges = sorted(a.range for a in goblin.attacks)
        assert ranges == ["melee", "ranged"]

    def test_scimitar(self, goblin):
        sci = next(a for a in goblin.attacks if a.name == "Scimitar")
        assert sci.to_hit == 4
        assert sci.damage_dice == "1d6+2"
        assert sci.damage_type == "slashing"

    def test_shortbow(self, goblin):
        bow = next(a for a in goblin.attacks if a.name == "Shortbow")
        assert bow.to_hit == 4
        assert bow.range == "ranged"


class TestSkeleton:
    def test_hp_with_modifier(self, skeleton):
        assert skeleton.hp_avg == 13
        assert skeleton.hp_dice == "2d8+4"

    def test_undead_type(self, skeleton):
        assert skeleton.creature_type == "undead"

    def test_two_attacks_no_rider(self, skeleton):
        assert len(skeleton.attacks) == 2
        for atk in skeleton.attacks:
            assert atk.save is None


class TestScoutMiscFormat:
    """Scout lives in Miscellaneous-creatures.txt and uses `### Actions`
    rather than the `#### Actions` of the main Monsters.txt format. This
    regression test catches header-depth assumptions in the parser."""

    @pytest.fixture
    def scout(self) -> sp.ParsedMonster:
        sb = srd_lookup.find("Scout")
        assert sb is not None
        return sp.parse(sb)

    def test_attacks_parsed(self, scout):
        names = sorted(a.name for a in scout.attacks)
        assert names == ["Longbow", "Shortsword"]

    def test_multiattack_captured(self, scout):
        assert "two melee" in scout.multiattack.lower()


class TestMultiattackParsing:
    """The simulator was previously running every attack in a monster's
    attacks list each turn — Lizardfolk swinging four times instead of
    two, etc. These tests pin the count and the index selection."""

    def test_count_from_english_word(self):
        assert sp.parse_multiattack_count(
            "The giant makes two morningstar attacks."
        ) == 2
        assert sp.parse_multiattack_count(
            "The lizardfolk makes two melee attacks, each one with a "
            "different weapon."
        ) == 2
        assert sp.parse_multiattack_count("makes three attacks") == 3
        assert sp.parse_multiattack_count("makes four attacks") == 4

    def test_count_from_digit(self):
        assert sp.parse_multiattack_count("makes 3 attacks") == 3

    def test_default_one_on_garbage(self):
        assert sp.parse_multiattack_count("") == 1
        assert sp.parse_multiattack_count("no number here") == 1

    def test_cloud_giant_indices_are_morningstar_twice(self):
        sb = srd_lookup.find("Cloud Giant")
        m = sp.parse(sb)
        # attacks are [Morningstar, Rock]
        indices = sp.select_multiattack_indices(m.attacks, m.multiattack)
        # Two morningstar attacks → (0, 0).
        assert indices == (0, 0)
        assert m.attacks[0].name == "Morningstar"

    def test_lizardfolk_indices_are_two_different_melee(self):
        sb = srd_lookup.find("Lizardfolk")
        m = sp.parse(sb)
        # attacks are [Bite, Heavy Club, Javelin, Spiked Shield] —
        # all melee. Multiattack says "two melee attacks, each one
        # with a different weapon", so the first two indices.
        indices = sp.select_multiattack_indices(m.attacks, m.multiattack)
        assert indices == (0, 1)
        assert len(indices) == 2

    def test_hill_giant_indices_are_greatsword_twice(self):
        sb = srd_lookup.find("Hill Giant")
        m = sp.parse(sb)
        indices = sp.select_multiattack_indices(m.attacks, m.multiattack)
        assert len(indices) == 2
        # Both indices point to the named weapon.
        for i in indices:
            assert m.attacks[i].name.lower() == "greatclub" or \
                   m.attacks[i].name.lower() == "greatsword" or \
                   "greatclub" in m.attacks[i].name.lower()

    def test_no_multiattack_returns_empty(self):
        sb = srd_lookup.find("Ghoul")
        m = sp.parse(sb)
        # Ghoul has Bite + Claws but no Multiattack action.
        assert m.multiattack == ""
        indices = sp.select_multiattack_indices(m.attacks, m.multiattack)
        assert indices == ()


class TestRobustness:
    def test_unknown_body_returns_defaults(self):
        sb = srd_lookup.StatBlock(name="Mystery", body="### Mystery\n\nno body here")
        m = sp.parse(sb)
        assert m.name == "Mystery"
        assert m.ac == 10
        assert m.hp_avg == 1
        assert m.attacks == ()

    def test_partial_body_does_not_crash(self):
        sb = srd_lookup.StatBlock(
            name="Half",
            body=(
                "### Half\n\n_Medium humanoid, neutral_\n\n"
                "- **Armor Class** 14\n"
                "- **Hit Points** 8 (2d6+1)\n"
                "- **Speed** 30 ft.\n"
            ),
        )
        m = sp.parse(sb)
        assert m.ac == 14
        assert m.hp_avg == 8
        assert m.attacks == ()
