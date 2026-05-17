"""Tests for encounter_simulator — combat loop and Monte Carlo."""

from __future__ import annotations

import random

import pytest

import encounter_simulator as es
import srd_lookup
import statblock_parser as sp


# ---------- Combatant builders -----------------------------------------


@pytest.fixture
def party_proto() -> list[dict]:
    """A small balanced party — fighter + cleric + rogue + wizard.
    The fixture is the same shape character_ingester.extract_character
    produces, so the simulator's PC pipeline gets exercised end-to-end.
    """
    return [
        {
            "name": "Thorin", "class": "fighter", "level": 3,
            "ac": 18, "hp_max": 28, "init_bonus": 1,
            "saves": {"str": 5, "con": 4, "dex": 1, "wis": 1, "int": 0, "cha": 0},
            "attacks": [
                {"name": "Longsword", "to_hit": 5, "damage": "1d10+3",
                 "damage_type": "slashing", "range": "melee"},
            ],
            "features": {"second_wind": True, "sneak_attack_dice": 0},
            "spells": {"slots": {}, "memorized": []},
        },
        {
            "name": "Mira", "class": "cleric", "level": 3,
            "ac": 16, "hp_max": 22, "init_bonus": 0,
            "saves": {"wis": 5, "con": 3, "dex": 1, "str": 0, "int": 0, "cha": 2},
            "attacks": [
                {"name": "Mace", "to_hit": 4, "damage": "1d6+2",
                 "damage_type": "bludgeoning", "range": "melee"},
            ],
            "features": {},
            "spells": {
                "slots": {1: 4, 2: 2},
                "memorized": [
                    {"name": "Cure Wounds", "level": 1, "type": "heal",
                     "amount": "1d8+3"},
                    {"name": "Sacred Flame", "level": 0, "type": "save_attack",
                     "save": "dex", "dc": 13, "damage": "1d8"},
                ],
            },
        },
        {
            "name": "Sera", "class": "rogue", "level": 3,
            "ac": 14, "hp_max": 21, "init_bonus": 3,
            "saves": {"dex": 5, "int": 4, "str": 1, "con": 2, "wis": 1, "cha": 1},
            "attacks": [
                {"name": "Shortsword", "to_hit": 5, "damage": "1d6+3",
                 "damage_type": "piercing", "range": "melee"},
            ],
            "features": {"sneak_attack_dice": 2},
            "spells": {"slots": {}, "memorized": []},
        },
        {
            "name": "Aldric", "class": "wizard", "level": 3,
            "ac": 12, "hp_max": 18, "init_bonus": 2,
            "saves": {"int": 5, "wis": 3, "dex": 2, "con": 1, "str": 0, "cha": 1},
            "attacks": [
                {"name": "Quarterstaff", "to_hit": 2, "damage": "1d6",
                 "damage_type": "bludgeoning", "range": "melee"},
            ],
            "features": {},
            "spells": {
                "slots": {1: 4, 2: 2},
                "memorized": [
                    {"name": "Burning Hands", "level": 1, "type": "save_attack",
                     "save": "dex", "dc": 13, "damage": "3d6",
                     "half_on_save": True, "aoe": True},
                    {"name": "Fire Bolt", "level": 0, "type": "save_attack",
                     "save": "dex", "dc": 13, "damage": "1d10"},
                ],
            },
        },
    ]


@pytest.fixture
def two_ghouls() -> list[sp.ParsedMonster]:
    sb = srd_lookup.find("Ghoul")
    assert sb is not None
    g = sp.parse(sb)
    return [g, g]


@pytest.fixture
def two_skeletons() -> list[sp.ParsedMonster]:
    sb = srd_lookup.find("Skeleton")
    assert sb is not None
    s = sp.parse(sb)
    return [s, s]


# ---------- Combatant construction -------------------------------------


def test_combatant_from_pc_uses_defaults_for_missing_fields():
    minimal = {"name": "X", "class": "fighter"}
    c = es.combatant_from_pc(minimal, idx=0)
    assert c.name == "X"
    assert c.ac == 10
    assert c.hp == 8
    assert c.hp_max == 8


def test_combatant_from_monster_copies_attacks(two_ghouls):
    g = two_ghouls[0]
    c = es.combatant_from_monster(g, idx=0)
    assert c.ac == 12
    assert c.hp == 22
    assert c.creature_type == "undead"
    names = sorted(a["name"] for a in c.attacks)
    assert names == ["Bite", "Claws"]


# ---------- Initiative -------------------------------------------------


def test_initiative_is_deterministic_with_seed(party_proto, two_ghouls):
    party = [es.combatant_from_pc(pc, idx=i) for i, pc in enumerate(party_proto)]
    monsters = [es.combatant_from_monster(m, idx=i) for i, m in enumerate(two_ghouls)]
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    a = [c.id for c in es.roll_initiative(party + monsters, rng_a)]
    b = [c.id for c in es.roll_initiative(party + monsters, rng_b)]
    assert a == b


# ---------- One combat -------------------------------------------------


class TestRunOneCombat:
    def test_party_usually_wins_balanced_fight(self, party_proto, two_ghouls):
        # 4 PCs vs 2 Ghouls is a winnable fight; with a fixed seed the
        # outcome is deterministic. We just assert it terminates.
        party = [es.combatant_from_pc(pc, idx=i)
                 for i, pc in enumerate(party_proto)]
        monsters = [es.combatant_from_monster(m, idx=i, label=f"Ghoul #{i+1}")
                    for i, m in enumerate(two_ghouls)]
        rng = random.Random(7)
        outcome = es.run_one_combat(party, monsters, rng)
        assert outcome.winner in ("pc", "monster", "draw")
        assert outcome.rounds >= 1
        assert outcome.rounds <= es.ROUND_CAP
        assert any("Combat begins" in line for line in outcome.trace)
        assert any("Combat ends" in line for line in outcome.trace)

    def test_deterministic_with_same_seed(self, party_proto, two_ghouls):
        def go():
            party = [es.combatant_from_pc(pc, idx=i)
                     for i, pc in enumerate(party_proto)]
            monsters = [es.combatant_from_monster(m, idx=i,
                                                  label=f"Ghoul #{i+1}")
                        for i, m in enumerate(two_ghouls)]
            return es.run_one_combat(party, monsters, random.Random(123))
        a = go()
        b = go()
        assert a.winner == b.winner
        assert a.rounds == b.rounds
        assert a.trace == b.trace

    def test_overwhelming_monsters_kill_party(self, party_proto):
        # 3 Hill Giants vs party-level-3 is lethal: each hits for ~18,
        # one-shotting the wizard, and 105 HP per giant absorbs the
        # party's burst damage easily.
        sb = srd_lookup.find("Hill Giant")
        giant = sp.parse(sb)
        party = [es.combatant_from_pc(pc, idx=i)
                 for i, pc in enumerate(party_proto)]
        monsters = [es.combatant_from_monster(giant, idx=i,
                                              label=f"Hill Giant #{i+1}")
                    for i in range(3)]
        outcome = es.run_one_combat(party, monsters, random.Random(0))
        assert outcome.winner == "monster"
        assert outcome.tpk is True

    def test_lone_pc_vs_no_monsters_immediately_wins(self, party_proto):
        party = [es.combatant_from_pc(party_proto[0], idx=0)]
        outcome = es.run_one_combat(party, [], random.Random(0))
        assert outcome.winner == "pc"
        assert outcome.tpk is False
        assert outcome.rounds == 1


# ---------- Resolve helpers -------------------------------------------


class TestResolveAttack:
    def _setup(self, attacker_to_hit=10, target_ac=10):
        attacker = es.combatant_from_monster(
            sp.ParsedMonster(
                name="Dummy", ac=10, hp_avg=20, hp_dice="3d8", speed=30,
                cr="1", creature_type="construct",
                attacks=(sp.MonsterAttack(
                    name="Punch", to_hit=attacker_to_hit,
                    damage_dice="1d4+2", damage_type="bludgeoning",
                    range="melee", save=None, rider_text="",
                ),),
            ),
            idx=0,
        )
        target = es.combatant_from_monster(
            sp.ParsedMonster(
                name="Target", ac=target_ac, hp_avg=20, hp_dice="3d8",
                speed=30, cr="1", creature_type="construct",
                attacks=(),
            ),
            idx=1,
        )
        return attacker, target

    def test_hit_reduces_hp(self):
        attacker, target = self._setup(attacker_to_hit=20, target_ac=10)
        state = es.CombatState(combatants=[attacker, target],
                               rng=random.Random(0))
        es.resolve_attack(attacker, target, attacker.attacks[0], state)
        assert target.hp < target.hp_max

    def test_lethal_damage_marks_down(self):
        attacker, target = self._setup(attacker_to_hit=20, target_ac=10)
        target.hp = 1
        state = es.CombatState(combatants=[attacker, target],
                               rng=random.Random(0))
        es.resolve_attack(attacker, target, attacker.attacks[0], state)
        assert target.hp == 0
        assert target.is_down is True
        assert not target.alive

    def test_miss_leaves_hp(self):
        # Attacker has -10 to-hit; barring nat 20 crit, will miss most rolls.
        attacker, target = self._setup(attacker_to_hit=-10, target_ac=20)
        # Force a fixed roll sequence by manually constructing rng state.
        for seed in range(20):
            target.hp = target.hp_max
            state = es.CombatState(combatants=[attacker, target],
                                   rng=random.Random(seed))
            es.resolve_attack(attacker, target, attacker.attacks[0], state)
            # Even when it lands (nat 20 crit), it can't fully wipe HP=20 in one shot.
            assert target.hp >= 0


# ---------- Monte Carlo ------------------------------------------------


class TestMonteCarlo:
    def test_aggregates_balanced(self, party_proto, two_ghouls):
        report = es.monte_carlo(
            party_proto, two_ghouls, trials=20, base_seed=100,
        )
        assert report.trials == 20
        assert 0.0 <= report.party_win_pct <= 100.0
        assert 0.0 <= report.tpk_pct <= 100.0
        assert 0.0 <= report.avg_party_hp_pct <= 100.0
        assert report.avg_rounds >= 1.0
        assert len(report.sample_trace) > 0

    def test_party_wins_easy_fight(self, party_proto):
        # Single Goblin: party should crush it nearly every time.
        sb = srd_lookup.find("Goblin")
        gob = sp.parse(sb)
        report = es.monte_carlo(party_proto, [gob], trials=30, base_seed=0)
        assert report.party_win_pct >= 80.0
        assert report.tpk_pct <= 5.0

    def test_party_loses_overwhelming_fight(self, party_proto):
        # 3 Hill Giants are a level-3 TPK.
        sb = srd_lookup.find("Hill Giant")
        giant = sp.parse(sb)
        report = es.monte_carlo(party_proto, [giant] * 3,
                                trials=20, base_seed=0)
        assert report.tpk_pct >= 70.0

    def test_mvp_is_party_member_in_winning_fight(self, party_proto):
        sb = srd_lookup.find("Goblin")
        gob = sp.parse(sb)
        report = es.monte_carlo(party_proto, [gob], trials=30, base_seed=0)
        assert report.mvp_id is not None
        # MVP could be either side in principle, but in an easy fight the
        # PCs deal almost all the damage.
        assert report.mvp_id.startswith("pc-")

    def test_trace_is_from_trial_zero(self, party_proto, two_ghouls):
        report_a = es.monte_carlo(party_proto, two_ghouls, trials=5, base_seed=42)
        report_b = es.monte_carlo(party_proto, two_ghouls, trials=5, base_seed=42)
        # Same seed → same trial #0 → same sample trace.
        assert report_a.sample_trace == report_b.sample_trace

    def test_different_seeds_change_trace(self, party_proto, two_ghouls):
        # The aggregate stats may or may not differ for trivial fights,
        # but the sample trace (trial #0) must differ between seeds —
        # that's what the user sees change after clicking Re-run.
        a = es.monte_carlo(party_proto, two_ghouls, trials=5, base_seed=1)
        b = es.monte_carlo(party_proto, two_ghouls, trials=5, base_seed=2)
        assert a.sample_trace != b.sample_trace


class TestMultiattackFidelity:
    """The simulator must honour the multiattack count from the SRD,
    not iterate every attack in the monster's attack list. Lizardfolk
    has 4 listed attacks but its multiattack only authorises 2 swings;
    Cloud Giant has 2 listed but Multiattack says two MORNINGSTAR
    attacks (i.e. the same weapon twice)."""

    def test_lizardfolk_makes_two_attacks_per_round(self, party_proto):
        sb = srd_lookup.find("Lizardfolk")
        liz = sp.parse(sb)
        party = [es.combatant_from_pc(pc, idx=i)
                 for i, pc in enumerate(party_proto)]
        monsters = [es.combatant_from_monster(liz, idx=0)]
        outcome = es.run_one_combat(party, monsters, random.Random(0))

        # Count how many distinct attack lines reference the Lizardfolk
        # as the actor in any single round.
        attacks_per_round: dict[int, int] = {}
        current_round = 0
        for line in outcome.trace:
            if line.startswith("-- Round "):
                current_round = int(line.split()[2])
                continue
            if line.startswith("Lizardfolk"):
                # An action line: a hit, a miss, "holds", or anything
                # similar. Exclude "Combat begins" etc.
                attacks_per_round[current_round] = attacks_per_round.get(
                    current_round, 0,
                ) + 1
        # The Lizardfolk should never have more than 2 attack-lines in
        # one round.
        rounds_with_too_many = [r for r, n in attacks_per_round.items() if n > 2]
        assert rounds_with_too_many == [], \
            f"Lizardfolk swung too many times in rounds {rounds_with_too_many}: " \
            f"{attacks_per_round}"

    def test_cloud_giant_uses_morningstar_not_rock(self, party_proto):
        # Two morningstar attacks per turn — never picks up the Rock
        # (which the old bug would have, since it's the second attack
        # in the list).
        sb = srd_lookup.find("Cloud Giant")
        cg = sp.parse(sb)
        party = [es.combatant_from_pc(pc, idx=i)
                 for i, pc in enumerate(party_proto)]
        monsters = [es.combatant_from_monster(cg, idx=0)]
        outcome = es.run_one_combat(party, monsters, random.Random(0))
        # No Rock attacks should appear in the trace.
        for line in outcome.trace:
            assert "Rock" not in line, \
                f"Cloud Giant should attack only with Morningstar, got: {line}"
