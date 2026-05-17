"""Tactics — pick an Action for a combatant on their turn.

Pure functions. No RNG, no I/O. The simulator owns randomness; tactics
only inspects state and returns a deterministic decision. Test by
constructing a synthetic CombatState and asserting the chosen Action.

Design choices:

  - Class dispatch for PCs (`actor.class_`). Unknown classes fall back
    to a generic "attack with the highest-bonus weapon" routine.
  - Monsters use a single generic playbook — pick a primary target and
    attack. Multiattack is handled by the simulator (it iterates the
    monster's full attack list against the chosen primary).
  - Spell selection is heuristic, not optimal: a cleric will choose
    *Cure Wounds* over *Healing Word* if both are memorised because
    Cure heals more, even though Healing Word is bonus action and the
    simulator doesn't model action economy. We deliberately accept
    that abstraction loss — the goal is "rational, not optimal".
  - Sneak attack is signalled by setting `Action.spell_name` to the
    sentinel `"__sneak_attack__"`; the simulator reads
    `actor.features.sneak_attack_dice` and adds the bonus damage.
    Same trick for Divine Smite (`"__divine_smite__"`).
"""

from __future__ import annotations

from encounter_simulator import Action, Combatant, CombatState


# How low an ally must drop before a cleric prioritises healing over damage.
HEAL_TRIGGER_HP_FRACTION = 0.40

# Fighter Second Wind threshold.
SECOND_WIND_TRIGGER = 0.50

# Max creatures an AoE spell can catch. 5e cones/spheres rarely get more
# than ~3 medium creatures absent ideal packing — capping here keeps the
# simulator honest without modelling positions on a grid.
AOE_TARGET_CAP = 3


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def pick_action(actor: Combatant, state: CombatState) -> Action:
    """Top-level dispatcher used by the simulator."""
    if not actor.alive:
        return Action(kind="pass")
    if actor.side == "monster":
        return pick_monster_action(actor, state)
    return pick_pc_action(actor, state)


# ---------------------------------------------------------------------------
# PC tactics by class
# ---------------------------------------------------------------------------


def pick_pc_action(actor: Combatant, state: CombatState) -> Action:
    cls = actor.class_.lower()
    if cls == "cleric":
        return _cleric_action(actor, state)
    if cls == "fighter":
        return _fighter_action(actor, state)
    if cls == "rogue":
        return _rogue_action(actor, state)
    if cls in ("wizard", "sorcerer"):
        return _caster_action(actor, state)
    if cls == "paladin":
        return _paladin_action(actor, state)
    return _default_attack(actor, state)


def _cleric_action(actor: Combatant, state: CombatState) -> Action:
    # Heal if any ally (including self) is below the trigger fraction.
    ally_pool = state.allies_of(actor) + [actor]
    needy = [a for a in ally_pool if a.hp < a.hp_max * HEAL_TRIGGER_HP_FRACTION]
    if needy:
        # Lowest fraction first — biggest bang per slot.
        needy.sort(key=lambda a: a.hp / max(1, a.hp_max))
        target = needy[0]
        heal = _best_heal(actor)
        if heal is not None:
            return Action(
                kind="heal", target_id=target.id,
                spell_name=heal["name"],
                spell_level=int(heal.get("level") or 1),
            )
    # Otherwise: best save-attack spell (Sacred Flame), then weapon.
    save_spell = _best_save_attack(actor)
    hostiles = state.hostiles_of(actor)
    if save_spell and hostiles:
        target = hostiles[0]
        return Action(
            kind="spell_save", target_id=target.id,
            spell_name=save_spell["name"],
            spell_level=int(save_spell.get("level") or 0),
        )
    return _default_attack(actor, state)


def _fighter_action(actor: Combatant, state: CombatState) -> Action:
    # Self-preservation: spend Second Wind once when bloodied.
    if (
        not actor.used_second_wind
        and actor.features.get("second_wind")
        and actor.hp < actor.hp_max * SECOND_WIND_TRIGGER
    ):
        return Action(kind="second_wind")
    return _default_attack(actor, state)


def _rogue_action(actor: Combatant, state: CombatState) -> Action:
    hostiles = state.hostiles_of(actor)
    if not hostiles:
        return Action(kind="pass")
    # Sneak attack when any other ally is engaged with a hostile (i.e.
    # the hostile has any ally besides the rogue in melee with it). With
    # the abstract positioning model, "engaged" reduces to: any ally in
    # melee + the hostile is reachable. We use: at least one ally
    # (other than the rogue) is alive AND the rogue has any attack —
    # equivalent to having a flanker.
    other_melee_ally = any(
        a.position == "melee" for a in state.allies_of(actor)
    )
    sneak_dice = int(actor.features.get("sneak_attack_dice") or 0)
    target = hostiles[0]
    idx = _best_attack_index(actor, target)
    if idx is None:
        return Action(kind="pass")
    if sneak_dice > 0 and other_melee_ally:
        return Action(
            kind="attack", target_id=target.id, attack_index=idx,
            spell_name="__sneak_attack__",
        )
    return Action(kind="attack", target_id=target.id, attack_index=idx)


def _caster_action(actor: Combatant, state: CombatState) -> Action:
    hostiles = state.hostiles_of(actor)
    if not hostiles:
        return Action(kind="pass")
    # AoE save spell if 2+ hostiles and a slot is available.
    aoe = _best_aoe(actor)
    if aoe and len(hostiles) >= 2 and _has_slot(actor, int(aoe.get("level") or 1)):
        return Action(
            kind="spell_save",
            spell_name=aoe["name"],
            spell_level=int(aoe.get("level") or 1),
            aoe_target_ids=tuple(h.id for h in hostiles[:AOE_TARGET_CAP]),
        )
    # Single-target damage spell.
    single = _best_single_target_spell(actor)
    if single and _has_slot(actor, int(single.get("level") or 0)):
        target = hostiles[0]
        return Action(
            kind="spell_save",
            target_id=target.id,
            spell_name=single["name"],
            spell_level=int(single.get("level") or 0),
        )
    return _default_attack(actor, state)


def _paladin_action(actor: Combatant, state: CombatState) -> Action:
    hostiles = state.hostiles_of(actor)
    if not hostiles:
        return Action(kind="pass")
    target = hostiles[0]
    idx = _best_attack_index(actor, target)
    if idx is None:
        return Action(kind="pass")
    smite_slot = _highest_available_slot(actor)
    if (
        actor.features.get("divine_smite_slots_usable")
        and smite_slot is not None
        and target.creature_type in ("undead", "fiend")
    ):
        return Action(
            kind="attack", target_id=target.id, attack_index=idx,
            spell_name="__divine_smite__", spell_level=smite_slot,
        )
    return Action(kind="attack", target_id=target.id, attack_index=idx)


def _default_attack(actor: Combatant, state: CombatState) -> Action:
    hostiles = state.hostiles_of(actor)
    if not hostiles or not actor.attacks:
        return Action(kind="pass")
    target = hostiles[0]
    idx = _best_attack_index(actor, target)
    if idx is None:
        return Action(kind="pass")
    return Action(kind="attack", target_id=target.id, attack_index=idx)


# ---------------------------------------------------------------------------
# Monster tactics (single playbook, multiattack handled by simulator)
# ---------------------------------------------------------------------------


def pick_monster_action(actor: Combatant, state: CombatState) -> Action:
    hostiles = state.hostiles_of(actor)
    if not hostiles or not actor.attacks:
        return Action(kind="pass")
    # Prefer melee targets first (rough analogue of "closest target").
    melee_pcs = [h for h in hostiles if h.position == "melee"]
    target = melee_pcs[0] if melee_pcs else hostiles[0]
    idx = _best_attack_index(actor, target)
    if idx is None:
        return Action(kind="pass")
    return Action(kind="attack", target_id=target.id, attack_index=idx)


# ---------------------------------------------------------------------------
# Helpers — spell + attack pickers
# ---------------------------------------------------------------------------


def _best_heal(actor: Combatant) -> dict | None:
    """Pick the highest-tier heal spell with a slot available."""
    heals = [s for s in actor.spells.get("memorized", [])
             if str(s.get("type", "")).lower() == "heal"]
    heals = [s for s in heals if _has_slot(actor, int(s.get("level") or 1))]
    if not heals:
        return None
    heals.sort(key=lambda s: int(s.get("level") or 0), reverse=True)
    return heals[0]


def _best_aoe(actor: Combatant) -> dict | None:
    spells = [s for s in actor.spells.get("memorized", [])
              if str(s.get("type", "")).lower() == "save_attack"
              and bool(s.get("aoe"))]
    if not spells:
        return None
    spells.sort(key=lambda s: int(s.get("level") or 0), reverse=True)
    return spells[0]


def _best_single_target_spell(actor: Combatant) -> dict | None:
    """Highest-damage save_attack or attack spell. Cantrips (level 0)
    are eligible; the simulator treats level 0 as no slot consumption."""
    spells = [s for s in actor.spells.get("memorized", [])
              if str(s.get("type", "")).lower() in ("save_attack", "attack")
              and not bool(s.get("aoe"))]
    if not spells:
        return None
    spells.sort(key=lambda s: int(s.get("level") or 0), reverse=True)
    return spells[0]


def _best_save_attack(actor: Combatant) -> dict | None:
    """Best damage cantrip/spell that uses a save (Sacred Flame for clerics)."""
    spells = [s for s in actor.spells.get("memorized", [])
              if str(s.get("type", "")).lower() == "save_attack"]
    if not spells:
        return None
    # Prefer cantrips (level 0) so we don't drain slots; fall back to leveled.
    spells.sort(key=lambda s: (int(s.get("level") or 0) > 0,
                                -int(s.get("level") or 0)))
    return spells[0]


def _best_attack_index(actor: Combatant, target: Combatant) -> int | None:
    """Return the index of the highest-to-hit attack that can reach the
    target. Falls back to any attack if none clearly preferred."""
    if not actor.attacks:
        return None
    # Filter: ranged target → both ranges work; melee target → melee
    # attack preferred but ranged still allowed.
    indices = list(range(len(actor.attacks)))
    indices.sort(key=lambda i: (-actor.attacks[i]["to_hit"],
                                 actor.attacks[i]["range"] != "melee"))
    return indices[0]


def _has_slot(actor: Combatant, level: int) -> bool:
    if level <= 0:
        return True
    slots = actor.spells.get("slots") or {}
    return int(slots.get(level, 0)) > 0


def _highest_available_slot(actor: Combatant) -> int | None:
    slots = actor.spells.get("slots") or {}
    available = [int(lvl) for lvl, n in slots.items() if int(n) > 0]
    if not available:
        return None
    return max(available)
