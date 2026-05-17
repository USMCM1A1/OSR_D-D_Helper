"""Approximate 5e level-scaling for the encounter simulator.

The simulator reads character stats from JSON files extracted once
from PDFs. Re-extracting every time the DM wants to test the same
party at a different level would be tedious, so this module scales a
character dict to a target level for one simulation run.

Scaling is approximate, not RAW:
  - HP scales linearly by `target / original`. (RAW would be
    `hit_die_avg + con_mod` per added level; for "is this survivable"
    testing the linear approximation is within ~10%.)
  - Proficiency bonus is left at the JSON's stored value, since the
    encounter simulator reads attack `to_hit` and spell `dc` fields
    directly rather than recomputing them. Proficiency only changes
    at L5/L9/L13/L17 anyway.
  - Rogue Sneak Attack dice scale: `ceil(level / 2)`.
  - Fighter / Paladin / Ranger / Barbarian gain Extra Attack at L≥5
    (the simulator's `_take_turn` re-runs the attack action when the
    flag is set).
  - Full-caster classes (cleric, druid, wizard, sorcerer, bard) get
    their spell slots replaced from FULL_CASTER_SLOTS. Half-casters
    (paladin, ranger) use HALF_CASTER_SLOTS. Warlocks use WARLOCK_SLOTS
    (pact magic). Other classes are left alone.
  - Memorised spells are NOT auto-added. A wizard scaled L1 → L5
    will have L1-L3 slots but only L1 spells in their JSON, so they
    cast cantrips/L1 spells out of the higher slots. This is
    deliberate: we'd otherwise have to invent spell choices for the
    DM.

The DM sees a "scaled to party level N" note on the results page so
they know what the simulator did.
"""

from __future__ import annotations

import copy


# Spell-slot tables — keyed by character level, mapping slot level → count.
# Sources: PHB class progression tables. We carry levels 1–20 even though
# the simulator's tactics module won't dispatch beyond ~L10 meaningfully.

FULL_CASTER_SLOTS: dict[int, dict[int, int]] = {
    1:  {1: 2},
    2:  {1: 3},
    3:  {1: 4, 2: 2},
    4:  {1: 4, 2: 3},
    5:  {1: 4, 2: 3, 3: 2},
    6:  {1: 4, 2: 3, 3: 3},
    7:  {1: 4, 2: 3, 3: 3, 4: 1},
    8:  {1: 4, 2: 3, 3: 3, 4: 2},
    9:  {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    10: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
    11: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1},
    12: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1},
    13: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1},
    14: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1},
    15: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1},
    16: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1},
    17: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1},
    18: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 1, 7: 1, 8: 1, 9: 1},
    19: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 2, 7: 1, 8: 1, 9: 1},
    20: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 2, 7: 2, 8: 1, 9: 1},
}

HALF_CASTER_SLOTS: dict[int, dict[int, int]] = {
    1:  {},
    2:  {1: 2},
    3:  {1: 3},
    4:  {1: 3},
    5:  {1: 4, 2: 2},
    6:  {1: 4, 2: 2},
    7:  {1: 4, 2: 3},
    8:  {1: 4, 2: 3},
    9:  {1: 4, 2: 3, 3: 2},
    10: {1: 4, 2: 3, 3: 2},
    11: {1: 4, 2: 3, 3: 3},
    12: {1: 4, 2: 3, 3: 3},
    13: {1: 4, 2: 3, 3: 3, 4: 1},
    14: {1: 4, 2: 3, 3: 3, 4: 1},
    15: {1: 4, 2: 3, 3: 3, 4: 2},
    16: {1: 4, 2: 3, 3: 3, 4: 2},
    17: {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    18: {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    19: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
    20: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
}

# Warlock pact magic — fewer slots, but they recharge on short rest.
# The simulator doesn't model short rests during a single combat, so
# warlocks effectively get only these N slots per fight.
WARLOCK_SLOTS: dict[int, dict[int, int]] = {
    1:  {1: 1},
    2:  {1: 2},
    3:  {2: 2},
    4:  {2: 2},
    5:  {3: 2},
    6:  {3: 2},
    7:  {4: 2},
    8:  {4: 2},
    9:  {5: 2},
    10: {5: 2},
    11: {5: 3},
    12: {5: 3},
    13: {5: 3},
    14: {5: 3},
    15: {5: 3},
    16: {5: 3},
    17: {5: 4},
    18: {5: 4},
    19: {5: 4},
    20: {5: 4},
}

FULL_CASTER_CLASSES = frozenset({"cleric", "druid", "wizard", "sorcerer", "bard"})
HALF_CASTER_CLASSES = frozenset({"paladin", "ranger"})
EXTRA_ATTACK_CLASSES = frozenset({"fighter", "paladin", "ranger", "barbarian"})


def scale_character(character: dict, target_level: int) -> dict:
    """Return a deep copy of `character` scaled to `target_level`.

    Pass `target_level` as int ≥ 1. If it equals the character's
    stored level, the copy is returned unchanged. If the character
    has no `level` field, we assume 1.
    """
    out = copy.deepcopy(character)
    orig_level = int(out.get("level") or 1)
    target_level = max(1, int(target_level))

    if target_level == orig_level:
        return out

    # HP — linear scale, never below 1.
    orig_hp = int(out.get("hp_max") or 8)
    out["hp_max"] = max(1, round(orig_hp * target_level / orig_level))
    out["level"] = target_level

    cls = str(out.get("class") or "").lower()
    features = out.setdefault("features", {})
    spells = out.setdefault("spells", {})

    if cls == "rogue":
        features["sneak_attack_dice"] = (target_level + 1) // 2

    if cls in EXTRA_ATTACK_CLASSES:
        features["extra_attack"] = target_level >= 5

    if cls in FULL_CASTER_CLASSES:
        new_slots = FULL_CASTER_SLOTS.get(target_level, {})
        spells["slots"] = {int(k): v for k, v in new_slots.items()}
    elif cls in HALF_CASTER_CLASSES:
        new_slots = HALF_CASTER_SLOTS.get(target_level, {})
        spells["slots"] = {int(k): v for k, v in new_slots.items()}
    elif cls == "warlock":
        new_slots = WARLOCK_SLOTS.get(target_level, {})
        spells["slots"] = {int(k): v for k, v in new_slots.items()}

    return out


def scale_party(characters: list[dict], target_level: int) -> list[dict]:
    """Scale every character in `characters` to `target_level`."""
    return [scale_character(c, target_level) for c in characters]
